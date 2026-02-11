from copy import deepcopy
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch import autocast
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from common import layers, math, init
from common.nvtx_utils import maybe_range
from tensordict import TensorDict
from tensordict.nn import TensorDictParams

from common.logger import get_logger
log = get_logger(__name__)


class WorldModel(nn.Module):
	"""
	TD-MPC2 implicit world model architecture.
	Can be used for both single-task and multi-task experiments.
	
	This implementation uses State-Value functions V(s) instead of 
	State-Action Value functions Q(s,a). The value heads take only
	the latent state z as input, not the action.
	"""

	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		
		# Validate critic_type - only V is supported
		critic_type = getattr(cfg, 'critic_type', 'V')
		if critic_type != 'V':
			raise ValueError(
				f"critic_type='{critic_type}' is not supported. "
				"This codebase only implements state-value functions (V). "
				"Set critic_type='V' in your config."
			)
		
		if self.cfg.dtype == 'float16':
			self.autocast_dtype = torch.float16
		elif self.cfg.dtype == 'bfloat16':
			self.autocast_dtype = torch.bfloat16	
		else:
			self.autocast_dtype = None
   
		if cfg.multitask:
			self._task_emb = nn.Embedding(len(cfg.tasks), cfg.task_dim, max_norm=1)
			self.register_buffer("_action_masks", torch.zeros(len(cfg.tasks), cfg.action_dim))
			for i in range(len(cfg.tasks)):
				self._action_masks[i, :cfg.action_dims[i]] = 1.
		self._encoder = layers.enc(cfg)

		# -----------------------------------------------------------------
		# Model groups: resolve num_groups and validate divisibility
		# -----------------------------------------------------------------
		h_total = int(cfg.planner_num_dynamics_heads)
		num_reward_heads = int(cfg.num_reward_heads)
		num_value_heads = int(cfg.num_q)
		G = int(cfg.num_groups)
		if G == -1:
			G = h_total  # fully grouped: one group per dynamics head
		assert G >= 1, f"num_groups must be >= 1 or -1, got {G}"
		assert h_total % G == 0, (
			f"planner_num_dynamics_heads={h_total} must be divisible by num_groups={G}")
		assert num_reward_heads % G == 0, (
			f"num_reward_heads={num_reward_heads} must be divisible by num_groups={G}")
		assert num_value_heads % G == 0, (
			f"num_q={num_value_heads} must be divisible by num_groups={G}")
		# Store resolved group dimensions on cfg for downstream access
		cfg.num_groups = G
		cfg.heads_per_group = h_total // G      # H_g
		cfg.reward_heads_per_group = num_reward_heads // G  # R_g
		cfg.value_heads_per_group = num_value_heads // G    # Ve_g
		self.G = G
		self.H_g = cfg.heads_per_group
		self.R_g = cfg.reward_heads_per_group
		self.Ve_g = cfg.value_heads_per_group

		# -----------------------------------------------------------------
		# Number of policies: P policies serve G groups (P <= G, G % P == 0)
		# -----------------------------------------------------------------
		num_policies_raw = cfg.get('num_policies', None)
		if num_policies_raw is None or num_policies_raw == 'None':
			P = G  # Default: one policy per group
		else:
			P = int(num_policies_raw)
		assert P >= 1, f"num_policies must be >= 1, got {P}"
		assert G % P == 0, (
			f"num_groups={G} must be divisible by num_policies={P}")
		cfg.num_policies = P
		cfg.groups_per_policy = G // P
		self.P = P
		self.groups_per_policy = cfg.groups_per_policy

		log.info(
			'Model groups: G=%d, H_g=%d (H=%d), R_g=%d (R=%d), Ve_g=%d (Ve=%d), P=%d (groups_per_policy=%d)',
			G, self.H_g, h_total, self.R_g, num_reward_heads, self.Ve_g, num_value_heads, P, self.groups_per_policy,
		)

		# Multi-head dynamics ensemble: vectorized via Ensemble (vmap + functional_call).
		# NOTE: We apply weight_init to each head before wrapping in Ensemble because
		# Ensemble stores params in TensorDictParams which self.apply() does not traverse.
		# Unified prior config for ensemble diversity
		prior_hidden_div = int(cfg.prior_hidden_div)
		prior_scale = float(cfg.prior_scale)
		# Dynamics head config: layer count and dropout
		dynamics_num_layers = int(getattr(cfg, 'dynamics_num_layers', 2))
		dynamics_dropout = float(getattr(cfg, 'dynamics_dropout', 0.0))
		dyn_heads = [
			layers.DynamicsHeadWithPrior(
				in_dim=cfg.latent_dim + cfg.action_dim + cfg.task_dim,
				mlp_dims=dynamics_num_layers * [cfg.mlp_dim],
				out_dim=cfg.latent_dim,
				cfg=cfg,
				prior_hidden_div=prior_hidden_div,
				prior_scale=prior_scale,
				dropout=dynamics_dropout,
			)
			for _ in range(h_total)
		]
		for dyn_head in dyn_heads:
			dyn_head.apply(init.weight_init)
		self._dynamics_heads = layers.Ensemble(dyn_heads)
		# Keep a reference for __repr__ (legacy code paths expecting _dynamics)
		self._dynamics = self._dynamics_heads
		
		# Multi-head reward ensemble for pessimistic reward estimation
		# Uses Ensemble (vmap) for efficient vectorized forward pass.
		# NOTE: We apply weight_init to each MLP before wrapping in Ensemble because
		# Ensemble stores params in TensorDictParams which apply() does not traverse.
		num_reward_heads = int(getattr(cfg, 'num_reward_heads', 1))
		reward_hidden_dim = cfg.mlp_dim // cfg.reward_dim_div
		num_reward_layers = getattr(cfg, 'num_reward_layers', 2)  # Default 2 for backward compat
		reward_dropout = cfg.dropout if getattr(cfg, 'reward_dropout_enabled', False) else 0.0
		prior_logit_scale = getattr(cfg, 'prior_logit_scale', 5.0)  # Default 5.0 for backward compat
		reward_mlps = [
			layers.MLPWithPrior(
				in_dim=cfg.latent_dim + cfg.action_dim + cfg.task_dim,
				hidden_dims=num_reward_layers * [reward_hidden_dim],
				out_dim=max(cfg.num_bins, 1),
				prior_hidden_div=prior_hidden_div,
				prior_scale=prior_scale,
				prior_logit_scale=prior_logit_scale,
				dropout=reward_dropout,
				distributional=True,  # Prior outputs scalar → two-hot for numerical stability
				cfg=cfg,
			)
			for _ in range(num_reward_heads)
		]
		for mlp_with_prior in reward_mlps:
			mlp_with_prior.main_mlp.apply(init.weight_init)
		self._Rs = layers.Ensemble(reward_mlps)
		
		# Termination head: same sizing as reward head (hidden dim, num layers)
		self._termination = layers.mlp(
			cfg.latent_dim + cfg.task_dim,
			num_reward_layers * [reward_hidden_dim],
			1
		) if cfg.episodic else None
		# -----------------------------------------------------------------
		# Per-policy policy networks: Ensemble(P) so each policy serves G/P groups.
		# When P=G this is one policy per group (current behavior).
		# When P<G, multiple groups share each policy network. The policy
		# is trained to maximize the average value across all groups it serves.
		# -----------------------------------------------------------------
		pi_in_dim = cfg.latent_dim + cfg.task_dim
		pi_out_dim = 2 * cfg.action_dim
		pi_mlps = [layers.mlp(pi_in_dim, 2 * [cfg.mlp_dim], pi_out_dim) for _ in range(P)]
		for pi_mlp in pi_mlps:
			pi_mlp.apply(init.weight_init)
		self._pi_groups = layers.Ensemble(pi_mlps)
		# Backward-compat alias used in optimizer param groups and __repr__
		self._pi = self._pi_groups

		# Optimistic per-policy policy for dual-policy architecture
		if cfg.dual_policy_enabled:
			pi_opti_mlps = [layers.mlp(pi_in_dim, 2 * [cfg.mlp_dim], pi_out_dim) for _ in range(P)]
			for pi_mlp in pi_opti_mlps:
				pi_mlp.apply(init.weight_init)
			self._pi_optimistic_groups = layers.Ensemble(pi_opti_mlps)
		else:
			self._pi_optimistic_groups = None
		# Backward-compat alias
		self._pi_optimistic = self._pi_optimistic_groups
		# V-function ensemble: input is latent only (no action)
		# NOTE: cfg.num_q is kept as the config name for backward compatibility in experiment tracking
		# Apply weight_init to each MLP before wrapping in Ensemble (same reason as reward heads).
		v_input_dim = cfg.latent_dim + cfg.task_dim
		v_mlp_dim = cfg.mlp_dim // cfg.value_dim_div  # Allow smaller V networks via value_dim_div
		num_value_layers = cfg.num_value_layers  # Number of hidden layers (default 2)
		v_mlps = [
			layers.MLPWithPrior(
				in_dim=v_input_dim,
				hidden_dims=num_value_layers * [v_mlp_dim],
				out_dim=max(cfg.num_bins, 1),
				prior_hidden_div=prior_hidden_div,
				prior_scale=prior_scale,
				prior_logit_scale=prior_logit_scale,
				dropout=cfg.dropout,
				distributional=True,  # Prior outputs scalar → two-hot for numerical stability
				cfg=cfg,
			)
			for _ in range(cfg.num_q)
		]
		for mlp_with_prior in v_mlps:
			mlp_with_prior.main_mlp.apply(init.weight_init)
		self._Vs = layers.Ensemble(v_mlps)

		# ------------------------------------------------------------------
		# Auxiliary multi-gamma state-value heads (training-only)
		# Simplified (no ensemble dimension) per user request.
		# We construct either:
		#   joint: a single MLP outputting (G_aux * K) logits reshaped to (G_aux,K)
		#   separate: ModuleList of G_aux MLP heads each outputting K logits
		# These heads are used only for auxiliary supervision and never for
		# planning or policy bootstrapping. Therefore, we do not maintain
		# target networks or ensembles (min/avg collapse to identity).
		# ------------------------------------------------------------------
		self._num_aux_gamma = 0
		self._aux_joint_Vs = None      # single MLP head for joint aux values
		self._aux_separate_Vs = None   # ModuleList[MLP] for separate aux values
		self._target_aux_joint_Vs = None
		self._target_aux_separate_Vs = None
		self._detach_aux_joint_Vs = None
		self._detach_aux_separate_Vs = None	
  
		# Only create auxiliary heads if gammas are specified AND loss weight > 0
		# When loss_weight=0, skip creating heads entirely to save memory/compute
		if getattr(cfg, 'multi_gamma_gammas', None) and cfg.multi_gamma_loss_weight != 0:
			gammas = cfg.multi_gamma_gammas
			self._num_aux_gamma = len(gammas)
			# Auxiliary V heads: input is latent only (no action)
			# Use same v_mlp_dim and num_value_layers as primary V heads for consistency
			aux_in_dim = cfg.latent_dim + (cfg.task_dim if cfg.multitask else 0)
			aux_v_mlp_dim = v_mlp_dim  # Already computed above as cfg.mlp_dim // cfg.value_dim_div
			if cfg.multi_gamma_head == 'joint':
				self._aux_joint_Vs = layers.mlp(aux_in_dim, num_value_layers*[aux_v_mlp_dim*cfg.joint_aux_dim_mult], max(cfg.num_bins * self._num_aux_gamma, 1), dropout=cfg.dropout)
			elif cfg.multi_gamma_head == 'separate':
				self._aux_separate_Vs = torch.nn.ModuleList([
					layers.mlp(aux_in_dim, num_value_layers*[aux_v_mlp_dim], max(cfg.num_bins, 1), dropout=cfg.dropout)
					for _ in range(self._num_aux_gamma)
				])
			else:
				raise ValueError(f"Unsupported multi_gamma_head: {cfg.multi_gamma_head}")

		self.apply(init.weight_init)
		# Zero-init main network output layers for reward and value heads.
		# When prior_scale=0: output = main(x) = 0 initially → V≈0, R≈0
		# When prior_scale>0: output = main(x) + prior(x) = 0 + prior(x) → diversity from prior
		# The main network can still learn to compensate since its output layer is trainable.
		# NOTE: MLPWithPrior nests the main MLP under "main_mlp" in TensorDictParams
		reward_output_layer_key = str(num_reward_layers)
		v_output_layer_key = str(num_value_layers)
		init.zero_([
			self._Rs.params["main_mlp", reward_output_layer_key, "weight"],
			self._Vs.params["main_mlp", v_output_layer_key, "weight"],
		])

		self.register_buffer("log_std_min", torch.tensor(cfg.log_std_min))
		self.register_buffer("log_std_dif", torch.tensor(cfg.log_std_max) - self.log_std_min)
		self.init()

	def _autocast_context(self):
		dtype = self.autocast_dtype
		if dtype is None or dtype == torch.float32:
			return nullcontext()
		device = next(self.parameters()).device
		if device.type != 'cuda':
			return nullcontext()
		return autocast(device_type=device.type, dtype=dtype)

	def init(self):
		# Create params
		self._detach_Vs_params = TensorDictParams(self._Vs.params.data, no_convert=True)
		self._target_Vs_params = TensorDictParams(self._Vs.params.data.clone(), no_convert=True)
  
		with self._detach_Vs_params.data.to("meta").to_module(self._Vs.module):
			self._detach_Vs = deepcopy(self._Vs)
			self._target_Vs = deepcopy(self._Vs)

		# Assign params to modules
		# We do this strange assignment to avoid having duplicated tensors in the state-dict -- working on a better API for this
		delattr(self._detach_Vs, "params")
		self._detach_Vs.__dict__["params"] = self._detach_Vs_params
		delattr(self._target_Vs, "params")
		self._target_Vs.__dict__["params"] = self._target_Vs_params

		# Auxiliary heads (non-ensemble): vectorized param buffers + target/detach clones
		if self._aux_joint_Vs is not None:
			# Initialize parameter vectors as buffers once
			if not hasattr(self, 'aux_joint_target_vec'):
				vec = parameters_to_vector(self._aux_joint_Vs.parameters()).detach().clone()
				self.register_buffer('aux_joint_target_vec', vec)
				self.register_buffer('aux_joint_detach_vec', vec.clone())
			# Create frozen clones and load vectors
			self._target_aux_joint_Vs = deepcopy(self._aux_joint_Vs)
			self._detach_aux_joint_Vs = deepcopy(self._aux_joint_Vs)
			for p in self._target_aux_joint_Vs.parameters():
				p.requires_grad_(False)
			for p in self._detach_aux_joint_Vs.parameters():
				p.requires_grad_(False)
			vector_to_parameters(self.aux_joint_target_vec, self._target_aux_joint_Vs.parameters())
			vector_to_parameters(self.aux_joint_detach_vec, self._detach_aux_joint_Vs.parameters())
		elif self._aux_separate_Vs is not None:
			# Build concatenated vectors and size map once
			if not hasattr(self, 'aux_separate_sizes'):
				self.aux_separate_sizes = [sum(p.numel() for p in h.parameters()) for h in self._aux_separate_Vs]
				vecs = [parameters_to_vector(h.parameters()).detach().clone() for h in self._aux_separate_Vs]
				full = torch.cat(vecs, dim=0)
				self.register_buffer('aux_separate_target_vec', full)
				self.register_buffer('aux_separate_detach_vec', full.clone())
			# Create frozen clones and load
			self._target_aux_separate_Vs = nn.ModuleList([deepcopy(h) for h in self._aux_separate_Vs])
			self._detach_aux_separate_Vs = nn.ModuleList([deepcopy(h) for h in self._aux_separate_Vs])
			for head in list(self._target_aux_separate_Vs) + list(self._detach_aux_separate_Vs):
				for p in head.parameters():
					p.requires_grad_(False)
			offset = 0
			for h, sz in zip(self._target_aux_separate_Vs, self.aux_separate_sizes):
				vector_to_parameters(self.aux_separate_target_vec[offset:offset+sz], h.parameters())
				offset += sz
			offset = 0
			for h, sz in zip(self._detach_aux_separate_Vs, self.aux_separate_sizes):
				vector_to_parameters(self.aux_separate_detach_vec[offset:offset+sz], h.parameters())
				offset += sz

		# Target policy for trust region regularization OR for TD target imagination
		# Create if trust region is enabled (coef > 0) or EMA policy is used for TD targets
		# Now targets are Ensemble(P) clones to match _pi_groups / _pi_optimistic_groups.
		need_target_pi = (
			getattr(self.cfg, 'policy_trust_region_coef', 0.0) > 0 or
			getattr(self.cfg, 'td_target_use_ema_policy', False)
		)
		if need_target_pi:
			# Create frozen target clone of per-group policy ensemble (only once)
			if not hasattr(self, '_target_pi_groups') or self._target_pi_groups is None:
				self._target_pi_groups = deepcopy(self._pi_groups)
				for p in self._target_pi_groups.parameters():
					p.requires_grad_(False)
			# Keep backward-compat alias
			self._target_pi = self._target_pi_groups
			# Optimistic policy target (if dual policy enabled)
			if self.cfg.dual_policy_enabled and self._pi_optimistic_groups is not None:
				if not hasattr(self, '_target_pi_optimistic_groups') or self._target_pi_optimistic_groups is None:
					self._target_pi_optimistic_groups = deepcopy(self._pi_optimistic_groups)
					for p in self._target_pi_optimistic_groups.parameters():
						p.requires_grad_(False)
				self._target_pi_optimistic = self._target_pi_optimistic_groups
			else:
				self._target_pi_optimistic_groups = None
				self._target_pi_optimistic = None
		else:
			self._target_pi_groups = None
			self._target_pi = None
			self._target_pi_optimistic_groups = None
			self._target_pi_optimistic = None


	def __repr__(self):
		repr = 'TD-MPC2 World Model\n'
		modules = ['Encoder', 'Dynamics', 'Reward', 'Termination', 'Policy prior', 'V-functions']
		# Optionally append auxiliary V ensembles to representation
		aux_modules = []
		if self._num_aux_gamma > 0:
			modules.append('Aux V-functions')
			aux_modules = [self._aux_joint_Vs if self._aux_joint_Vs is not None else self._aux_separate_Vs]
		for i, m in enumerate([self._encoder, self._dynamics, self._Rs, self._termination, self._pi, self._Vs] + aux_modules):
			if m == self._termination and not self.cfg.episodic:
				continue
			repr += f"{modules[i]}: {m}\n"
		repr += "Learnable parameters: {:,}".format(self.total_params)
		return repr

	@property
	def total_params(self):
		return sum(p.numel() for p in self.parameters() if p.requires_grad)

	def to(self, *args, **kwargs):
		super().to(*args, **kwargs)
		self.init()
		return self

	def train(self, mode=True):
		"""
		Overriding `train` method to keep target V-networks in eval mode.
		"""
		super().train(mode)
		self._target_Vs.train(False)
		if self._target_aux_joint_Vs is not None:
			self._target_aux_joint_Vs.train(False)
		if getattr(self, '_detach_aux_joint_Vs', None) is not None:
			self._detach_aux_joint_Vs.train(False)
		if getattr(self, '_target_aux_separate_Vs', None) is not None:
			for h in self._target_aux_separate_Vs:
				h.train(False)
		if getattr(self, '_detach_aux_separate_Vs', None) is not None:
			for h in self._detach_aux_separate_Vs:
				h.train(False)
		return self

	def soft_update_target_V(self):
		"""
		Soft-update target V-networks using Polyak averaging.
		"""
		with maybe_range('WM/soft_update_target_V', self.cfg):
			self._target_Vs_params.lerp_(self._detach_Vs_params, self.cfg.tau)
			# Vectorized EMA for aux heads; emit debug checks if enabled
			if getattr(self, '_aux_joint_Vs', None) is not None:
				with torch.no_grad():
					online = parameters_to_vector(self._aux_joint_Vs.parameters()).detach()
					prev = self.aux_joint_target_vec.detach().clone()
					self.aux_joint_target_vec.lerp_(online, self.cfg.tau)
					vector_to_parameters(self.aux_joint_target_vec, self._target_aux_joint_Vs.parameters())
					# Detach mirrors online snapshot
					self.aux_joint_detach_vec.copy_(online)
					vector_to_parameters(self.aux_joint_detach_vec, self._detach_aux_joint_Vs.parameters())
					
	
			elif getattr(self, '_aux_separate_Vs', None) is not None:
				with torch.no_grad():
					vecs = [parameters_to_vector(h.parameters()).detach() for h in self._aux_separate_Vs]
					online = torch.cat(vecs, dim=0)
					prev = self.aux_separate_target_vec.detach().clone()
					self.aux_separate_target_vec.lerp_(online, self.cfg.tau)
					# Assign to target heads
					offset = 0
					for h, sz in zip(self._target_aux_separate_Vs, self.aux_separate_sizes):
						vector_to_parameters(self.aux_separate_target_vec[offset:offset+sz], h.parameters())
						offset += sz
					# Detach mirrors online
					self.aux_separate_detach_vec.copy_(online)
					offset = 0
					for h, sz in zip(self._detach_aux_separate_Vs, self.aux_separate_sizes):
						vector_to_parameters(self.aux_separate_detach_vec[offset:offset+sz], h.parameters())
						offset += sz

	def _soft_update_module(self, target_module, source_module, tau):
		with torch.no_grad():
			target_params = list(target_module.parameters())
			source_params = list(source_module.parameters())
			if len(target_params) != len(source_params):
				raise RuntimeError('Source/target parameter count mismatch during EMA update')
			for target, source in zip(target_params, source_params):
				target.data.lerp_(source.data, tau)
			online_vec = parameters_to_vector(source_params).detach()
			target_vec = parameters_to_vector(target_params).detach()
			return torch.max(torch.abs(online_vec - target_vec))

	def soft_update_target_pi(self):
		"""Soft-update target policy networks for trust region regularization."""
		if self._target_pi is None:
			return
		# Update primary policy target (policy_ema_tau resolved at config parse time)
		self._soft_update_module(self._target_pi, self._pi, self.cfg.policy_ema_tau)
		# Update optimistic policy target (if dual policy)
		if self._target_pi_optimistic is not None:
			self._soft_update_module(self._target_pi_optimistic, self._pi_optimistic, self.cfg.policy_ema_tau)
				
	def task_emb(self, x, task):
		"""
		Continuous task embedding for multi-task experiments.
		Retrieves the task embedding for a given task ID `task`
		and concatenates it to the input `x`.
		"""
		if isinstance(task, int):
			task = torch.tensor([task], device=x.device)
		emb = self._task_emb(task.long())
		if x.ndim == 3:
			emb = emb.unsqueeze(0).repeat(x.shape[0], 1, 1)
		elif emb.shape[0] == 1:
			emb = emb.repeat(x.shape[0], 1)
		return torch.cat([x, emb], dim=-1)

	def encode(self, obs, task):
		"""
		Encodes an observation into its latent representation.
		This implementation assumes a single state-based observation.
		"""
		with maybe_range('WM/encode', self.cfg):
			encoder = self._encoder
			if self.cfg.multitask:
				obs = self.task_emb(obs, task)
			if self.cfg.obs == 'rgb' and obs.ndim == 5:
				with self._autocast_context():
					encoded = [encoder[self.cfg.obs](o) for o in obs]
				return torch.stack([e.float() for e in encoded])
			with self._autocast_context():
				out = encoder[self.cfg.obs](obs)
			return out.float()

	def next(self, z, a, task, split_data=False):
		"""Predict next latent(s) using all dynamics heads in one vectorized call.

		Args:
			z: Current latent(s).
				Broadcast (split_data=False): Tensor[B, L] — same data to all heads.
				Split    (split_data=True):  Tensor[H, B, L] — per-head data.
			a: Current action(s).  Shape mirrors z (with A instead of L).
			task: Optional task id for multitask (passed to task_emb when enabled).
			split_data: If True, z and a have a leading H dim sliced per-head.

		Returns:
			Tensor[H, B, L]: Next latent(s) with leading head dimension.
		"""
		with maybe_range('WM/dynamics', self.cfg):
			if self.cfg.multitask:
				z = self.task_emb(z, task)
			za = torch.cat([z, a], dim=-1)
			with self._autocast_context():
				out = self._dynamics_heads(za, split_data=split_data)  # float32[H, B, L]
			return out.float()

	def reward(self, z, a, task, head_mode='single', split_data=False):
		"""
		Predicts instantaneous (single-step) reward logits.

		Args:
			z (Tensor[..., L]): Latent states. When split_data=True, leading dim
				must equal num_reward_heads (R) so each head gets its own slice.
			a (Tensor[..., A]): Actions. Shape must match z.
			task: Task index for multitask setup.
			head_mode (str): 'single' returns logits from head 0 [1, ..., K],
				'all' returns logits from all heads [R, ..., K].
			split_data (bool): If True, z[i] and a[i] are fed only to head i
				via Ensemble split_data. Requires z.shape[0] == R.

		Returns:
			Tensor[R, ..., K] where R=1 if head_mode='single', R=num_heads if 'all'.
		"""
		with maybe_range('WM/reward', self.cfg):
			if self.cfg.multitask:
				z = self.task_emb(z, task)
			za = torch.cat([z, a], dim=-1)
			with self._autocast_context():
				# Ensemble forward returns [num_heads, ..., K] via vmap
				out = self._Rs(za, split_data=split_data)
			out = out.float()
			if head_mode == 'single':
				return out[:1]  # Keep leading dim, take first head only
			elif head_mode == 'all':
				return out
			else:
				raise ValueError(f"Unsupported head_mode for reward: {head_mode}. Use 'single' or 'all'.")
	
	def termination(self, z, task, unnormalized=False):
		"""
		Predicts termination signal.
		"""
		assert task is None
		if self.cfg.multitask:
			z = self.task_emb(z, task)
		with self._autocast_context():
			logits = self._termination(z)
		logits = logits.float()
		if unnormalized:
			return logits
		return torch.sigmoid(logits)
		

	def pi(self, z, task, search_noise=False, optimistic=False, target=False):
		"""Sample actions from per-policy policy ensemble.

		P policy networks serve G groups. When P < G, multiple groups share
		a policy network. Input z has shape [G, batch, L]; internally this is
		reshaped to [P, batch*groups_per_policy, L] for the Ensemble(P) call,
		then the output is reshaped back to [G, batch, A].

		All post-MLP operations (chunk, squash, entropy) are element-wise and
		work transparently on the [G, batch, A] output layout.

		Args:
			z (Tensor[G, batch, L]): Latent states with leading group dim.
			task: Task identifier for multitask setup.
			search_noise: Unused (legacy parameter).
			optimistic: If True and dual_policy_enabled, use optimistic policy.
			target: If True, use EMA target policy.

		Returns:
			Tuple[Tensor[G, batch, A], TensorDict[G, batch, ...]]: Actions and info.
		"""
		with maybe_range('WM/pi', self.cfg):
			# Select Ensemble(P) module based on optimistic and target flags
			if target:
				if optimistic and self.cfg.dual_policy_enabled:
					module = self._target_pi_optimistic_groups
				else:
					module = self._target_pi_groups
				# Fall back to online if target not available
				if module is None:
					module = self._pi_optimistic_groups if (optimistic and self.cfg.dual_policy_enabled) else self._pi_groups
			else:
				if optimistic and self.cfg.dual_policy_enabled:
					module = self._pi_optimistic_groups
				else:
					module = self._pi_groups

			G = self.G
			P = self.P
			groups_per_policy = self.groups_per_policy
			assert z.shape[0] == G, (
				f"pi() expects z with leading dim G={G}, got {z.shape[0]}")
			batch_per_group = z.shape[1]  # Assuming z is 3D: [G, batch, L]

			if self.cfg.multitask:
				z = self.task_emb(z, task)

			# Reshape [G, batch, L] → [P, batch*groups_per_policy, L] for Ensemble(P)
			if P < G:
				z_for_ensemble = z.view(P, groups_per_policy, batch_per_group, -1)  # [P, g_p, batch, L+task]
				z_for_ensemble = z_for_ensemble.view(P, groups_per_policy * batch_per_group, -1)  # [P, batch_per_policy, L+task]
			else:
				z_for_ensemble = z  # P == G, no reshape needed

			with self._autocast_context():
				# Ensemble(P) with split_data=True: each policy p gets z_for_ensemble[p]
				raw = module(z_for_ensemble, split_data=True)  # float32[P, batch_per_policy, 2*A]

			# Reshape back to [G, batch_per_group, 2*A]
			if P < G:
				raw = raw.view(P, groups_per_policy, batch_per_group, -1)  # [P, g_p, batch, 2*A]
				raw = raw.view(G, batch_per_group, -1)  # [G, batch, 2*A]

			mean, log_std = raw.float().chunk(2, dim=-1)
			log_std = math.log_std(log_std, self.log_std_min, self.log_std_dif)
			eps = torch.randn_like(mean, device=mean.device)

			if self.cfg.multitask:
				mean = mean * self._action_masks[task]
				log_std = log_std * self._action_masks[task]
				eps = eps * self._action_masks[task]
				action_dims = self._action_masks.sum(-1)[task].unsqueeze(-1)
			else:
				action_dims = None

			# Compute Gaussian log prob before squashing
			log_prob_presquash = math.gaussian_logprob(eps, log_std)  # float32[..., 1]
			action_dim = eps.shape[-1] if action_dims is None else action_dims
			
			# --- Squash actions & compute two entropy variants ---
			presquash_mean = mean
			if self.cfg.bmpc_policy_parameterization:
				# BMPC-style: squash mean only, add noise with clamp.
				# No Jacobian correction since the *sample* is not squashed.
				mean = torch.tanh(presquash_mean)
				action = (mean + eps * log_std.exp()).clamp(-1, 1)
				true_log_prob = log_prob_presquash
				maximized_log_prob = log_prob_presquash
			else:
				presquash_action = presquash_mean + eps * log_std.exp()

				# True entropy: squash with full Jacobian (scale=1.0)
				mean, action, true_log_prob = math.squash(
					presquash_mean, presquash_action, log_prob_presquash, jacobian_scale=1.0
				)

				# Maximized entropy: squash with configured Jacobian scale.
				# This is what the policy loss actually maximizes.
				jacobian_scale = float(self.cfg.jacobian_correction_scale)
				_, _, maximized_log_prob = math.squash(
					presquash_mean, presquash_action, log_prob_presquash, jacobian_scale=jacobian_scale
				)

			# Entropy = -log_prob; scale by action_dim^power
			entropy_multiplier = torch.tensor(
				action_dim, dtype=true_log_prob.dtype, device=true_log_prob.device
			).pow(self.cfg.entropy_action_dim_power)

			true_entropy = -true_log_prob                                    # float32[..., 1]
			true_scaled_entropy = true_entropy * entropy_multiplier          # float32[..., 1]
			maximized_entropy = -maximized_log_prob                          # float32[..., 1]
			maximized_scaled_entropy = maximized_entropy * entropy_multiplier  # float32[..., 1]

			info = TensorDict({
				"presquash_mean": presquash_mean,
				"mean": mean,
				"log_std": log_std,
				"action_prob": 1.,
				"entropy": maximized_entropy,                # used in policy loss
				"scaled_entropy": maximized_scaled_entropy,  # used in policy loss
				"true_entropy": true_entropy,                # for logging only
				"true_scaled_entropy": true_scaled_entropy,  # for logging only
				"entropy_multiplier": entropy_multiplier,
			}, device=z.device, non_blocking=True)
			return action, info

	def V_aux(self, z, task, return_type='all', target=False, detach=False):
		"""
		Compute auxiliary state-value predictions for multiple discount factors.
		
		Args:
			z (Tensor[T, B, L]): Latent state embeddings.
			task: Task identifier for multitask setup.
			return_type (str): 'all' returns logits, 'min'/'avg' return scalar values.
			target (bool): Use target network.
			detach (bool): Use detached (non-gradient) network.
			
		Returns:
			Tensor[G_aux, T, B, K] if return_type='all' (logits).
			Tensor[G_aux, T, B, 1] otherwise (values).
		"""
		with maybe_range('WM/V_aux', self.cfg):
			if self._num_aux_gamma == 0:
				return None
			assert return_type in {'all', 'min', 'avg', 'mean'}

			if self.cfg.multitask:
				z = self.task_emb(z, task)
			# V-function: input is state only (no action concatenation)
			if self._aux_joint_Vs is not None:
				with maybe_range('WM/V_aux/joint', self.cfg):
					with self._autocast_context():
						if target:
							raw = self._target_aux_joint_Vs(z)
						elif detach:
							raw = self._detach_aux_joint_Vs(z)
						else:
							raw = self._aux_joint_Vs(z)  # (T,B,G_aux*K)
					out = raw.float()
					# Reshape joint logits: (T,B,G*K) -> (G,T,B,K)
					T, B = out.shape[0], out.shape[1]
					G = self._num_aux_gamma
					K = self.cfg.num_bins
					if out.shape[-1] != G * K:
						raise RuntimeError(f"V_aux joint head expected last dim {G*K}, got {out.shape[-1]}")
					out = out.view(T, B, G, K).permute(2, 0, 1, 3)  # (G,T,B,K)
			elif self._aux_separate_Vs is not None:
				with maybe_range('WM/V_aux/separate', self.cfg):
					with self._autocast_context():
						if target:
							raw_list = [head(z) for head in self._target_aux_separate_Vs]
						elif detach:
							raw_list = [head(z) for head in self._detach_aux_separate_Vs]
						else:
							raw_list = [head(z) for head in self._aux_separate_Vs]
					outs = [rl.float() for rl in raw_list]
					out = torch.stack(outs, dim=0)  # (G_aux,T,B,K)
	
			if return_type == 'all':
				return out
			vals = math.two_hot_inv(out, self.cfg)  # (G_aux,T,B,1) 
			return vals  # 'min'/'avg' identical with single head

	def V(self, z, task, return_type='min', target=False, detach=False, split_data=False):
		"""
		Compute state-value predictions.
		
		Args:
			z (Tensor[..., L]): Latent state embeddings. When split_data=True,
				leading dim must equal num_q (Ve) so each head gets its own slice.
			task: Task identifier for multitask setup.
			return_type (str): 'all' returns logits, 'all_values' returns scalar values per head,
			                   'min'/'max'/'avg'/'mean' return scalar values reduced over heads.
			target (bool): Use target network.
			detach (bool): Use detached (non-gradient) network.
			split_data (bool): If True, z[i] is fed only to head i via Ensemble split_data.
			
		Returns:
			Tensor[num_q, ..., K] if return_type='all' (logits).
			Tensor[num_q, ..., 1] if return_type='all_values' (scalar values per head).
			Tensor[..., 1] otherwise (values after two-hot inverse, reduced over all num_q heads).
		"""
		assert return_type in {'min', 'avg', 'mean', 'max', 'all', 'all_values'}

		with maybe_range('WM/V', self.cfg):
			if self.cfg.multitask:
				z = self.task_emb(z, task)

			# V-function: input is state only (no action concatenation)
			if target:
				vnet = self._target_Vs
			elif detach:
				vnet = self._detach_Vs
			else:
				vnet = self._Vs
			with self._autocast_context():
				out = vnet(z, split_data=split_data)
			out = out.float()  # float32[num_q, ..., K]

			if return_type == 'all':
				return out

			# Use ALL num_q ensemble members for reduction (no subsampling)
			V = math.two_hot_inv(out, self.cfg)  # float32[num_q, ..., 1]
			
			if return_type == 'all_values':
				return V  # Return scalar values for all heads without reduction
			if return_type == "min":
				return torch.amin(V, dim=0)
			if return_type == "max":
				return torch.amax(V, dim=0)
			# 'avg' or 'mean' - compute average over all heads
			return V.mean(0)

	def rollout_latents(self, z0, actions=None, use_policy=False, horizon=None, num_rollouts=None, task=None, policy_action_noise_std: float = 0.0, use_optimistic_policy: bool = False, use_target_policy: bool = False):
		"""Roll out latent trajectories vectorized over all H dynamics heads.

		All H dynamics heads are always used (no head_mode selection).
		At each timestep each head advances its own latent independently via
		a single batched ``next(..., split_data=True)`` call.

		When ``use_policy=True``, per-group policies sample actions from the
		first dynamics head of each group.  Within a group all H_g heads
		share the same action; different groups may produce different actions.

		Args:
			z0 (Tensor[L] or Tensor[B,L]): Initial latent(s).
			actions (Tensor[B,N,T,A], optional): Action sequences when ``use_policy=False``.
			use_policy (bool): If True, ignore ``actions`` and sample from per-group
				policies using each group's first-head latents.
			horizon (int, optional): Number of steps T when ``use_policy=True``.
			num_rollouts (int, optional): Number of sequences N when ``use_policy=True``.
			task: Optional task id for multitask.
			policy_action_noise_std (float): Std of Gaussian noise added to policy
				actions when ``use_policy=True``.
			use_optimistic_policy (bool): If True and dual_policy_enabled, sample from
				the optimistic policy.
			use_target_policy (bool): If True, use EMA target policy.

		Returns:
			When use_policy=True:
				Tuple[Tensor[H,B,N,T+1,L], Tensor[H,B,N,T,A]]:
					Latent trajectories and per-head actions (within-group heads
					share actions; different groups may differ).
			When use_policy=False:
				Tuple[Tensor[H,B,N,T+1,L], Tensor[B,N,T,A]]:
					Latent trajectories and the input actions (unchanged).
		"""
		if use_policy:
			if actions is not None:
				raise ValueError('Provide either actions or use_policy=True, not both.')
			if horizon is None or num_rollouts is None:
				raise ValueError('horizon and num_rollouts required when use_policy=True.')
			T = int(horizon)  # steps
			N = int(num_rollouts)  # sequences
		else:
			if actions is None:
				raise ValueError('actions must be provided when use_policy=False.')
			if actions.ndim != 4:
				raise ValueError(f'actions must be [B,N,T,A], got shape {tuple(actions.shape)}')
			B, N, T, A = actions.shape  # actions: float32[B,N,T,A]
			if horizon is not None and horizon != T:
				raise ValueError('Provided horizon does not match actions.shape[2].')

		# Normalize z0 to [B,L]
		if z0.ndim == 1:
			z0 = z0.unsqueeze(0)  # [1,L] -> [B=1,L]
		B = z0.shape[0]
		L = z0.shape[-1]

		H = len(self._dynamics_heads)
		G = self.G
		H_g = self.H_g

		# Determine action dim
		if not use_policy:
			A = actions.shape[-1]
		else:
			A = self.cfg.action_dim

		# t=0 state for all heads: broadcast z0 over N, then expand over H
		z0_bn = z0.unsqueeze(1).expand(B, N, L)         # float32[B, N, L]
		z0_hbn = z0_bn.unsqueeze(0).expand(H, B, N, L)  # float32[H, B, N, L]
		latents_steps = [z0_hbn]
		actions_steps = []  # each entry shape depends on use_policy

		with maybe_range('WM/rollout_latents', self.cfg):
			for t in range(T):
				with maybe_range('WM/rollout_latents/step_loop', self.cfg):
					# --- Action selection ---
					if use_policy:
						with maybe_range('WM/rollout_latents/policy_action', self.cfg):
							# Per-group policy: use first head of each group's latents
							# latents_steps[t]: float32[H, B, N, L]
							z_grouped = latents_steps[t].view(G, H_g, B, N, L)  # float32[G, H_g, B, N, L]
							z_for_pi = z_grouped[:, 0].reshape(G, B * N, L)     # float32[G, B*N, L]
							a_group, _ = self.pi(z_for_pi, task, optimistic=use_optimistic_policy, target=use_target_policy)
							# a_group: float32[G, B*N, A]
							a_group = a_group.detach()
							if policy_action_noise_std > 0.0:
								noise = torch.randn_like(a_group) * float(policy_action_noise_std)
								a_group = (a_group + noise).clamp(-1.0, 1.0)
							# Expand to per-head: [G, B*N, A] → [H, B*N, A]
							a_all = a_group.repeat_interleave(H_g, dim=0)  # float32[H, B*N, A]
							a_t_hbn = a_all.view(H, B, N, A)        # float32[H, B, N, A]
							actions_steps.append(a_t_hbn)
					else:
						a_t = actions[:, :, t, :]  # float32[B, N, A]
						actions_steps.append(a_t)

					# --- Advance all H dynamics heads in one batched call ---
					with maybe_range('WM/rollout_latents/dynamics_head', self.cfg):
						z_all = latents_steps[t].reshape(H, B * N, L)    # float32[H, B*N, L]
						if use_policy:
							# a_all already [H, B*N, A] from above
							pass
						else:
							a_all = a_t.contiguous().reshape(B * N, A)       # float32[B*N, A]
							a_all = a_all.unsqueeze(0).expand(H, -1, -1)    # float32[H, B*N, A]
						# next() with split_data=True: each head gets its own z,a slice
						next_all = self.next(z_all, a_all, task, split_data=True)  # float32[H, B*N, L]
						next_hbn = next_all.reshape(H, B, N, L)          # float32[H, B, N, L]
					latents_steps.append(next_hbn)

		# Stack over time to produce final outputs
		with maybe_range('WM/rollout_latents/finalize', self.cfg):
			latents = torch.stack(latents_steps, dim=3).contiguous()      # float32[H, B, N, T+1, L]
			if use_policy:
				# Per-head actions (within-group heads share actions)
				actions_out = torch.stack(actions_steps, dim=3).contiguous()  # float32[H, B, N, T, A]
			else:
				actions_out = torch.stack(actions_steps, dim=2).contiguous()  # float32[B, N, T, A]

		assert latents.is_contiguous(), "latents expected contiguous"
		assert actions_out.is_contiguous(), "actions_out expected contiguous"
		return latents, actions_out

	def generate_grouped_policy_seeds(
		self,
		z0: torch.Tensor,
		total_seeds: int,
		horizon: int,
		task=None,
		policy_action_noise_std: float = 0.0,
		use_optimistic_policy: bool = False,
	) -> torch.Tensor:
		"""Generate policy-seeded action sequences from per-group policies.

		Each of the G groups produces ``total_seeds // G`` action sequences
		of length ``horizon`` by auto-regressively sampling from its own
		policy.  The sequences are concatenated so the caller can feed them
		to ``rollout_latents(actions=..., use_policy=False)`` for full H-head
		evaluation.

		The auto-regressive rollout uses the *first* dynamics head of each
		group (head index ``g * H_g``) to advance latents across the horizon.

		Args:
			z0 (Tensor[B, L]): Initial encoded latent(s).
			total_seeds (int): Total number of policy-seeded sequences S.
				Must be divisible by G.
			horizon (int): Number of time steps T per sequence.
			task: Optional task indices for multitask.
			policy_action_noise_std (float): Std of Gaussian noise on actions.
			use_optimistic_policy (bool): Use optimistic per-group policy.

		Returns:
			Tensor[B, S, T, A]: Concatenated action sequences ready for
				``rollout_latents(actions=..., use_policy=False)``.
		"""
		G = self.G
		H_g = self.H_g
		assert total_seeds % G == 0, (
			f"total_seeds={total_seeds} must be divisible by num_groups={G}")
		S_g = total_seeds // G  # seeds per group
		if z0.ndim == 1:
			z0 = z0.unsqueeze(0)
		B = z0.shape[0]
		L = z0.shape[-1]
		A = self.cfg.action_dim

		# Each group uses its first dynamics head to advance latents.
		# Dynamics head indices for group-first-head: [0, H_g, 2*H_g, ...]
		# We collect per-group latents of shape [G, B*S_g, L] at each step
		# and sample actions from the per-group Ensemble(G) policy.

		# Initial latents: broadcast z0 to [G, B*S_g, L]
		z_cur = z0.unsqueeze(1).expand(B, S_g, L).reshape(B * S_g, L)  # float32[B*S_g, L]
		z_cur = z_cur.unsqueeze(0).expand(G, -1, -1).contiguous()      # float32[G, B*S_g, L]

		action_steps = []  # each entry: float32[G, B*S_g, A]

		with maybe_range('WM/generate_grouped_policy_seeds', self.cfg):
			for t in range(horizon):
				# Sample actions from per-group policies
				a_group, _ = self.pi(z_cur, task, optimistic=use_optimistic_policy)
				# a_group: float32[G, B*S_g, A]
				a_group = a_group.detach()
				if policy_action_noise_std > 0.0:
					noise = torch.randn_like(a_group) * float(policy_action_noise_std)
					a_group = (a_group + noise).clamp(-1.0, 1.0)
				action_steps.append(a_group)

				# Advance latents using each group's first dynamics head.
				# We need split_data=True with the G-member dynamics subset.
				# Instead, use the full H-head dynamics with split_data=True
				# and extract only the first head per group from the output.
				# Expand z_cur and a_group to [H, B*S_g, ?] with repeat_interleave
				z_for_dyn = z_cur.repeat_interleave(H_g, dim=0)  # float32[H, B*S_g, L]
				a_for_dyn = a_group.repeat_interleave(H_g, dim=0)  # float32[H, B*S_g, A]
				next_z_all = self.next(z_for_dyn, a_for_dyn, task, split_data=True)  # float32[H, B*S_g, L]
				# Take first head per group: view [G, H_g, B*S_g, L] → [:, 0]
				z_cur = next_z_all.view(G, H_g, B * S_g, L)[:, 0].contiguous()  # float32[G, B*S_g, L]

		# Stack: [T, G, B*S_g, A] → rearrange to [B, S, T, A] where S = G*S_g
		actions_stacked = torch.stack(action_steps, dim=0)  # float32[T, G, B*S_g, A]
		# Reshape to [T, G, B, S_g, A]
		actions_stacked = actions_stacked.view(horizon, G, B, S_g, A)
		# Permute to [B, G, S_g, T, A] → [B, G*S_g, T, A] = [B, S, T, A]
		actions_out = actions_stacked.permute(2, 1, 3, 0, 4).contiguous()  # float32[B, G, S_g, T, A]
		actions_out = actions_out.reshape(B, total_seeds, horizon, A)       # float32[B, S, T, A]

		return actions_out
