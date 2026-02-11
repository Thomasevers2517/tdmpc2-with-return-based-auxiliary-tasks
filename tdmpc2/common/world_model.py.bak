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
		# Multi-head dynamics ensemble: vectorized via Ensemble (vmap + functional_call).
		# NOTE: We apply weight_init to each head before wrapping in Ensemble because
		# Ensemble stores params in TensorDictParams which self.apply() does not traverse.
		h_total = int(getattr(cfg, 'planner_num_dynamics_heads', 1))
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
		self._pi = layers.mlp(cfg.latent_dim + cfg.task_dim, 2*[cfg.mlp_dim], 2*cfg.action_dim)
		# Optimistic policy for dual-policy architecture (same architecture as _pi)
		self._pi_optimistic = layers.mlp(cfg.latent_dim + cfg.task_dim, 2*[cfg.mlp_dim], 2*cfg.action_dim) if cfg.dual_policy_enabled else None
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
		need_target_pi = (
			getattr(self.cfg, 'policy_trust_region_coef', 0.0) > 0 or
			getattr(self.cfg, 'td_target_use_ema_policy', False)
		)
		if need_target_pi:
			# Create frozen target clone (only once)
			if not hasattr(self, '_target_pi') or self._target_pi is None:
				self._target_pi = deepcopy(self._pi)
				for p in self._target_pi.parameters():
					p.requires_grad_(False)
			# Optimistic policy target (if dual policy enabled)
			if self.cfg.dual_policy_enabled and self._pi_optimistic is not None:
				if not hasattr(self, '_target_pi_optimistic') or self._target_pi_optimistic is None:
					self._target_pi_optimistic = deepcopy(self._pi_optimistic)
					for p in self._target_pi_optimistic.parameters():
						p.requires_grad_(False)
			else:
				self._target_pi_optimistic = None
		else:
			self._target_pi = None
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

	def reward(self, z, a, task, head_mode='single'):
		"""
		Predicts instantaneous (single-step) reward logits.

		Args:
			z (Tensor[..., L]): Latent states.
			a (Tensor[..., A]): Actions.
			task: Task index for multitask setup.
			head_mode (str): 'single' returns logits from head 0 [1, ..., K],
				'all' returns logits from all heads [R, ..., K].

		Returns:
			Tensor[R, ..., K] where R=1 if head_mode='single', R=num_heads if 'all'.
		"""
		with maybe_range('WM/reward', self.cfg):
			if self.cfg.multitask:
				z = self.task_emb(z, task)
			za = torch.cat([z, a], dim=-1)
			with self._autocast_context():
				# Ensemble forward returns [num_heads, ..., K] via vmap
				out = self._Rs(za)
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
		"""
		Samples an action from the policy prior.
		The policy prior is a Gaussian distribution with
		mean and (log) std predicted by a neural network.
		
		Args:
			z: Latent state tensor.
			task: Task identifier for multitask setup.
			search_noise: Unused (legacy parameter).
			optimistic: If True and dual_policy_enabled, use optimistic policy.
			target: If True, use EMA target policy (for trust region regularization).
		"""
		with maybe_range('WM/pi', self.cfg):
			# Select module based on optimistic and target flags
			if target:
				if optimistic and self.cfg.dual_policy_enabled:
					module = self._target_pi_optimistic
				else:
					module = self._target_pi
				# Fall back to online if target not available
				if module is None:
					module = self._pi_optimistic if (optimistic and self.cfg.dual_policy_enabled) else self._pi
			else:
				if optimistic and self.cfg.dual_policy_enabled:
					module = self._pi_optimistic
				else:
					module = self._pi
			if self.cfg.multitask:
				z = self.task_emb(z, task)
			with self._autocast_context():
				raw = module(z)
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

	def V(self, z, task, return_type='min', target=False, detach=False):
		"""
		Compute state-value predictions.
		
		Args:
			z (Tensor[..., L]): Latent state embeddings.
			task: Task identifier for multitask setup.
			return_type (str): 'all' returns logits, 'all_values' returns scalar values per head,
			                   'min'/'max'/'avg'/'mean' return scalar values reduced over heads.
			target (bool): Use target network.
			detach (bool): Use detached (non-gradient) network.
			
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
				out = vnet(z)
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

		Args:
			z0 (Tensor[L] or Tensor[B,L]): Initial latent(s).
			actions (Tensor[B,N,T,A], optional): Action sequences when ``use_policy=False``.
			use_policy (bool): If True, ignore ``actions`` and sample from the policy
				using head-0 latents.
			horizon (int, optional): Number of steps T when ``use_policy=True``.
			num_rollouts (int, optional): Number of sequences N when ``use_policy=True``.
			task: Optional task id for multitask.
			policy_action_noise_std (float): Std of Gaussian noise added to policy
				actions when ``use_policy=True``.
			use_optimistic_policy (bool): If True and dual_policy_enabled, sample from
				the optimistic policy.
			use_target_policy (bool): If True, use EMA target policy.

		Returns:
			Tuple[Tensor[H,B,N,T+1,L], Tensor[B,N,T,A]]:
				Latent trajectories (all H heads) and actions used.
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

		# Determine action dim
		if not use_policy:
			A = actions.shape[-1]
		else:
			A = self.cfg.action_dim

		# t=0 state for all heads: broadcast z0 over N, then expand over H
		z0_bn = z0.unsqueeze(1).expand(B, N, L)         # float32[B, N, L]
		z0_hbn = z0_bn.unsqueeze(0).expand(H, B, N, L)  # float32[H, B, N, L]
		latents_steps = [z0_hbn]
		actions_steps = []  # each entry: float32[B, N, A]

		with maybe_range('WM/rollout_latents', self.cfg):
			for t in range(T):
				with maybe_range('WM/rollout_latents/step_loop', self.cfg):
					# --- Action selection ---
					if use_policy:
						with maybe_range('WM/rollout_latents/policy_action', self.cfg):
							# Use head-0 latents for policy sampling
							z_for_pi = latents_steps[t][0].reshape(B * N, L)  # float32[B*N, L]
							a_flat, _ = self.pi(z_for_pi, task, optimistic=use_optimistic_policy, target=use_target_policy)
							a_t = a_flat.reshape(B, N, A).detach()  # float32[B, N, A]
							if policy_action_noise_std > 0.0:
								noise = torch.randn_like(a_t) * float(policy_action_noise_std)
								a_t = (a_t + noise).clamp(-1.0, 1.0)
					else:
						a_t = actions[:, :, t, :]  # float32[B, N, A]
					actions_steps.append(a_t)

					# --- Advance all H dynamics heads in one batched call ---
					with maybe_range('WM/rollout_latents/dynamics_head', self.cfg):
						z_all = latents_steps[t].reshape(H, B * N, L)    # float32[H, B*N, L]
						a_all = a_t.contiguous().reshape(B * N, A)       # float32[B*N, A]
						a_all = a_all.unsqueeze(0).expand(H, -1, -1)    # float32[H, B*N, A]
						# next() with split_data=True: each head gets its own z,a slice
						next_all = self.next(z_all, a_all, task, split_data=True)  # float32[H, B*N, L]
						next_hbn = next_all.reshape(H, B, N, L)          # float32[H, B, N, L]
					latents_steps.append(next_hbn)

		# Stack over time to produce final outputs
		with maybe_range('WM/rollout_latents/finalize', self.cfg):
			latents = torch.stack(latents_steps, dim=3).contiguous()      # float32[H, B, N, T+1, L]
			actions_out = torch.stack(actions_steps, dim=2).contiguous()   # float32[B, N, T, A]

		assert latents.is_contiguous(), "latents expected contiguous"
		assert actions_out.is_contiguous(), "actions_out expected contiguous"
		return latents, actions_out
