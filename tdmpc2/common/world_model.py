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
	"""

	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
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
		# Multi-head dynamics: independent heads; first head exposed for legacy uses/repr
		h_total = int(getattr(cfg, 'planner_num_dynamics_heads', 1))
		self._dynamics_heads = nn.ModuleList([
			layers.mlp(
				cfg.latent_dim + cfg.action_dim + cfg.task_dim,
				2 * [cfg.mlp_dim],
				cfg.latent_dim,
				act=layers.SimNorm(cfg),
			)
			for _ in range(h_total)
		])
		# Keep a reference for __repr__ and any legacy code paths expecting `_dynamics`
		self._dynamics = self._dynamics_heads[0]
		
		self._reward = layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[ cfg.mlp_dim // cfg.reward_dim_div ], max(cfg.num_bins, 1))
		self._termination = layers.mlp(cfg.latent_dim + cfg.task_dim, 2*[cfg.mlp_dim], 1) if cfg.episodic else None
		self._pi = layers.mlp(cfg.latent_dim + cfg.task_dim, 2*[cfg.mlp_dim], 2*cfg.action_dim)
		self._Qs = layers.Ensemble([layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], max(cfg.num_bins, 1), dropout=cfg.dropout) for _ in range(cfg.num_q)])
		self._target_encoder = None
		self._target_pi = None
		self.encoder_target_max_delta = 0.0
		self.policy_target_max_delta = 0.0
		if cfg.encoder_ema_enabled:
			self._target_encoder = deepcopy(self._encoder)
			for param in self._target_encoder.parameters():
				param.requires_grad_(False)
			self._target_encoder.eval()
		if cfg.policy_ema_enabled:
			self._target_pi = deepcopy(self._pi)
			for param in self._target_pi.parameters():
				param.requires_grad_(False)
			self._target_pi.eval()

		# ------------------------------------------------------------------
		# Auxiliary multi-gamma action-value heads (training-only)
		# Simplified (no ensemble dimension) per user request.
		# We construct either:
		#   joint: a single MLP outputting (G_aux * K) logits reshaped to (G_aux,K)
		#   separate: ModuleList of G_aux MLP heads each outputting K logits
		# These heads are used only for auxiliary supervision and never for
		# planning or policy bootstrapping. Therefore, we do not maintain
		# target networks or ensembles (min/avg collapse to identity).
		# ------------------------------------------------------------------
		self._num_aux_gamma = 0
		self._aux_joint_Qs = None      # now a single MLP head (not an Ensemble)
		self._aux_separate_Qs = None   # ModuleList[MLP]
		self._target_aux_joint_Qs = None
		self._target_aux_separate_Qs = None
		self._detach_aux_joint_Qs = None
		self._detach_aux_separate_Qs = None	
  
		if getattr(cfg, 'multi_gamma_gammas', None):
			gammas = cfg.multi_gamma_gammas
			self._num_aux_gamma = len(gammas)
			in_dim = cfg.latent_dim + cfg.action_dim + (cfg.task_dim if cfg.multitask else 0)
			if cfg.multi_gamma_head == 'joint':
				self._aux_joint_Qs = layers.mlp(in_dim, 2*[cfg.mlp_dim*cfg.joint_aux_dim_mult], max(cfg.num_bins * self._num_aux_gamma, 1), dropout=cfg.dropout)
			elif cfg.multi_gamma_head == 'separate':
				self._aux_separate_Qs = torch.nn.ModuleList([
					layers.mlp(in_dim, 2*[cfg.mlp_dim], max(cfg.num_bins, 1), dropout=cfg.dropout)
					for _ in range(self._num_aux_gamma)
				])
			else:
				raise ValueError(f"Unsupported multi_gamma_head: {cfg.multi_gamma_head}")

		self.apply(init.weight_init)
		init.zero_([self._reward[-1].weight, self._Qs.params["2", "weight"]])
		self.reset_policy_encoder_targets()

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
		self._detach_Qs_params = TensorDictParams(self._Qs.params.data, no_convert=True)
		self._target_Qs_params = TensorDictParams(self._Qs.params.data.clone(), no_convert=True)
  
		with self._detach_Qs_params.data.to("meta").to_module(self._Qs.module):
			self._detach_Qs = deepcopy(self._Qs)
			self._target_Qs = deepcopy(self._Qs)

		# Assign params to modules
		# We do this strange assignment to avoid having duplicated tensors in the state-dict -- working on a better API for this
		delattr(self._detach_Qs, "params")
		self._detach_Qs.__dict__["params"] = self._detach_Qs_params
		delattr(self._target_Qs, "params")
		self._target_Qs.__dict__["params"] = self._target_Qs_params

		# Auxiliary heads (non-ensemble): vectorized param buffers + target/detach clones
		if self._aux_joint_Qs is not None:
			# Initialize parameter vectors as buffers once
			if not hasattr(self, 'aux_joint_target_vec'):
				vec = parameters_to_vector(self._aux_joint_Qs.parameters()).detach().clone()
				self.register_buffer('aux_joint_target_vec', vec)
				self.register_buffer('aux_joint_detach_vec', vec.clone())
			# Create frozen clones and load vectors
			self._target_aux_joint_Qs = deepcopy(self._aux_joint_Qs)
			self._detach_aux_joint_Qs = deepcopy(self._aux_joint_Qs)
			for p in self._target_aux_joint_Qs.parameters():
				p.requires_grad_(False)
			for p in self._detach_aux_joint_Qs.parameters():
				p.requires_grad_(False)
			vector_to_parameters(self.aux_joint_target_vec, self._target_aux_joint_Qs.parameters())
			vector_to_parameters(self.aux_joint_detach_vec, self._detach_aux_joint_Qs.parameters())
		elif self._aux_separate_Qs is not None:
			# Build concatenated vectors and size map once
			if not hasattr(self, 'aux_separate_sizes'):
				self.aux_separate_sizes = [sum(p.numel() for p in h.parameters()) for h in self._aux_separate_Qs]
				vecs = [parameters_to_vector(h.parameters()).detach().clone() for h in self._aux_separate_Qs]
				full = torch.cat(vecs, dim=0)
				self.register_buffer('aux_separate_target_vec', full)
				self.register_buffer('aux_separate_detach_vec', full.clone())
			# Create frozen clones and load
			self._target_aux_separate_Qs = nn.ModuleList([deepcopy(h) for h in self._aux_separate_Qs])
			self._detach_aux_separate_Qs = nn.ModuleList([deepcopy(h) for h in self._aux_separate_Qs])
			for head in list(self._target_aux_separate_Qs) + list(self._detach_aux_separate_Qs):
				for p in head.parameters():
					p.requires_grad_(False)
			offset = 0
			for h, sz in zip(self._target_aux_separate_Qs, self.aux_separate_sizes):
				vector_to_parameters(self.aux_separate_target_vec[offset:offset+sz], h.parameters())
				offset += sz
			offset = 0
			for h, sz in zip(self._detach_aux_separate_Qs, self.aux_separate_sizes):
				vector_to_parameters(self.aux_separate_detach_vec[offset:offset+sz], h.parameters())
				offset += sz
   

	def __repr__(self):
		repr = 'TD-MPC2 World Model\n'
		modules = ['Encoder', 'Dynamics', 'Reward', 'Termination', 'Policy prior', 'Q-functions']
		# Optionally append auxiliary Q ensembles to representation
		aux_modules = []
		if self._num_aux_gamma > 0:
			modules.append('Aux Q-functions')
			aux_modules = [self._aux_joint_Qs if self._aux_joint_Qs is not None else self._aux_separate_Qs]
		for i, m in enumerate([self._encoder, self._dynamics, self._reward, self._termination, self._pi, self._Qs] + aux_modules):
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
		Overriding `train` method to keep target Q-networks in eval mode.
		"""
		super().train(mode)
		self._target_Qs.train(False)
		if self._target_aux_joint_Qs is not None:
			self._target_aux_joint_Qs.train(False)
		if getattr(self, '_detach_aux_joint_Qs', None) is not None:
			self._detach_aux_joint_Qs.train(False)
		if getattr(self, '_target_aux_separate_Qs', None) is not None:
			for h in self._target_aux_separate_Qs:
				h.train(False)
		if getattr(self, '_detach_aux_separate_Qs', None) is not None:
			for h in self._detach_aux_separate_Qs:
				h.train(False)
		return self

	def soft_update_target_Q(self):
		"""
		Soft-update target Q-networks using Polyak averaging.
		"""
		with maybe_range('WM/soft_update_target_Q', self.cfg):
			self._target_Qs_params.lerp_(self._detach_Qs_params, self.cfg.tau)
			# Vectorized EMA for aux heads; emit debug checks if enabled
			if getattr(self, '_aux_joint_Qs', None) is not None:
				with torch.no_grad():
					online = parameters_to_vector(self._aux_joint_Qs.parameters()).detach()
					prev = self.aux_joint_target_vec.detach().clone()
					self.aux_joint_target_vec.lerp_(online, self.cfg.aux_value_tau)
					vector_to_parameters(self.aux_joint_target_vec, self._target_aux_joint_Qs.parameters())
					# Detach mirrors online snapshot
					self.aux_joint_detach_vec.copy_(online)
					vector_to_parameters(self.aux_joint_detach_vec, self._detach_aux_joint_Qs.parameters())
					
	
			elif getattr(self, '_aux_separate_Qs', None) is not None:
				with torch.no_grad():
					vecs = [parameters_to_vector(h.parameters()).detach() for h in self._aux_separate_Qs]
					online = torch.cat(vecs, dim=0)
					prev = self.aux_separate_target_vec.detach().clone()
					self.aux_separate_target_vec.lerp_(online, self.cfg.aux_value_tau)
					# Assign to target heads
					offset = 0
					for h, sz in zip(self._target_aux_separate_Qs, self.aux_separate_sizes):
						vector_to_parameters(self.aux_separate_target_vec[offset:offset+sz], h.parameters())
						offset += sz
					# Detach mirrors online
					self.aux_separate_detach_vec.copy_(online)
					offset = 0
					for h, sz in zip(self._detach_aux_separate_Qs, self.aux_separate_sizes):
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

	def soft_update_policy_encoder_targets(self):
		updates = {}
		if self._target_encoder is not None:
			updates['encoder'] = self._soft_update_module(self._target_encoder, self._encoder, float(self.cfg.encoder_ema_tau))
			self.encoder_target_max_delta = updates['encoder']
		if self._target_pi is not None:
			updates['policy'] = self._soft_update_module(self._target_pi, self._pi, float(self.cfg.policy_ema_tau))
			self.policy_target_max_delta = updates['policy']
		return updates

	def reset_policy_encoder_targets(self):
		if self._target_encoder is not None:
			self._target_encoder.load_state_dict(self._encoder.state_dict())
			self.encoder_target_max_delta = 0.0
		if self._target_pi is not None:
			self._target_pi.load_state_dict(self._pi.state_dict())
			self.policy_target_max_delta = 0.0
				
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

	def encode(self, obs, task, use_ema=False):
		"""
		Encodes an observation into its latent representation.
		This implementation assumes a single state-based observation.
		"""
		with maybe_range('WM/encode', self.cfg):
			if use_ema:
				if self._target_encoder is None:
					raise RuntimeError('Encoder target requested but not initialized')
				encoder = self._target_encoder
			else:
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

	def next(self, z, a, task, head_mode=None):
		"""Predict next latent(s) for one step.

		Args:
			z (Tensor[B, L]): Current latent.
			a (Tensor[B, A]): Current action.
			task: Optional task id for multitask (passed to task_emb when enabled).
			head_mode (str|None): If None, returns Tensor[B,L] using head 0 (legacy path).
				If 'single', returns Tensor[1,B,L] using head 0.
				If 'random', returns Tensor[1,B,L] using a random head per call.
				If 'all', returns Tensor[H,B,L] stacked over all dynamics heads.

		Returns:
			Tensor[..., B, L]: Next latent(s) with optional head dimension leading.
		"""
		with maybe_range('WM/dynamics', self.cfg):
			if self.cfg.multitask:
				z = self.task_emb(z, task)
			za = torch.cat([z, a], dim=-1)
			H = len(self._dynamics_heads)
			if head_mode is None:
				# Legacy behavior: single head (0), no head dimension in output
				with self._autocast_context():
					out = self._dynamics_heads[0](za)
				return out.float()
			if head_mode == 'single':
				with self._autocast_context():
					out = self._dynamics_heads[0](za)
				return out.float().unsqueeze(0)
			elif head_mode == 'random':
				idx = int(torch.randint(low=0, high=H, size=(1,), device=za.device))
				with self._autocast_context():
					out = self._dynamics_heads[idx](za)
				return out.float().unsqueeze(0)
			elif head_mode == 'all':
				outs = []
				with self._autocast_context():
					for head in self._dynamics_heads:
						outs.append(head(za)) 
				return torch.stack([o.float() for o in outs], dim=0)  # [H,B,L]
			else:
				raise ValueError(f"Unsupported head_mode: {head_mode}")

	def reward(self, z, a, task):
		"""
		Predicts instantaneous (single-step) reward.
		"""
		with maybe_range('WM/reward', self.cfg):
			if self.cfg.multitask:
				z = self.task_emb(z, task)
			za = torch.cat([z, a], dim=-1)
			with self._autocast_context():
				out = self._reward(za)
			return out.float()
	
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
		

	def pi(self, z, task, use_ema=False, search_noise=False):
		"""
		Samples an action from the policy prior.
		The policy prior is a Gaussian distribution with
		mean and (log) std predicted by a neural network.
		"""
		with maybe_range('WM/pi', self.cfg):
			if use_ema:
				if self._target_pi is None:
					raise RuntimeError('Policy target requested but not initialized')
				module = self._target_pi
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

			log_prob = math.gaussian_logprob(eps, log_std)
			size = eps.shape[-1] if action_dims is None else action_dims
			scaled_log_prob = log_prob * size
			action = mean + eps * log_std.exp()
			presquash_mean = mean
			mean, action, log_prob = math.squash(mean, action, log_prob)
			entropy_scale = scaled_log_prob / (log_prob + 1e-8)
			info = TensorDict({
				"presquash_mean": presquash_mean,
				"mean": mean,
				"log_std": log_std,
				"action_prob": 1.,
				"entropy": -log_prob,
				"scaled_entropy": -log_prob * entropy_scale,
			}, device=z.device, non_blocking=True)
			return action, info

	def Q_aux(self, z, a, task, return_type='all', target=False, detach=False):

		with maybe_range('WM/Q_aux', self.cfg):
			if self._num_aux_gamma == 0:
				return None
			assert return_type in {'all','min','avg'}

			if self.cfg.multitask:
				z = self.task_emb(z, task)
			za = torch.cat([z, a], dim=-1)
			if self._aux_joint_Qs is not None:
				with maybe_range('WM/Q_aux/joint', self.cfg):
					with self._autocast_context():
						if target:
							raw = self._target_aux_joint_Qs(za)
						elif detach:
							raw = self._detach_aux_joint_Qs(za)
						else:
							raw = self._aux_joint_Qs(za)  # (T,B,G_aux*K)
					out = raw.float()
					# Reshape joint logits: (T,B,G*K) -> (G,T,B,K)
					T, B = out.shape[0], out.shape[1]
					G = self._num_aux_gamma
					K = self.cfg.num_bins
					if out.shape[-1] != G * K:
						raise RuntimeError(f"Q_aux joint head expected last dim {G*K}, got {out.shape[-1]}")
					out = out.view(T, B, G, K).permute(2, 0, 1, 3)  # (G,T,B,K)
			elif self._aux_separate_Qs is not None:
				with maybe_range('WM/Q_aux/separate', self.cfg):
					with self._autocast_context():
						if target:
							raw_list = [head(za) for head in self._target_aux_separate_Qs]
						elif detach:
							raw_list = [head(za) for head in self._detach_aux_separate_Qs]
						else:
							raw_list = [head(za) for head in self._aux_separate_Qs]
					outs = [rl.float() for rl in raw_list]
					out = torch.stack(outs, dim=0)  # (G_aux,T,B,K)
	
			if return_type == 'all':
				return out
			vals = math.two_hot_inv(out, self.cfg)  # (G_aux,T,B,1) 
			return vals  # 'min'/'avg' identical with single head

	def Q(self, z, a, task, return_type='min', target=False, detach=False):
		
		assert return_type in {'min', 'avg',  'max', 'all'}

		with maybe_range('WM/Q', self.cfg):
			if self.cfg.multitask:
				z = self.task_emb(z, task)

			za = torch.cat([z, a], dim=-1)
			if target:
				qnet = self._target_Qs
			elif detach:
				qnet = self._detach_Qs
			else:
				qnet = self._Qs
			with self._autocast_context():
				out = qnet(za)
			out = out.float()

			if return_type == 'all':
				return out

			qidx = torch.randperm(self.cfg.num_q, device=out.device)[:2]
			Q = math.two_hot_inv(out[qidx], self.cfg)
			if return_type == "min":
				return Q.min(0).values
			if return_type == "max":
				return Q.max(0).values
			return Q.sum(0) / 2

	def rollout_latents(self, z0, actions=None, use_policy=False, horizon=None, num_rollouts=None, head_mode='single', task=None, policy_action_noise_std: float = 0.0):
		"""Roll out latent trajectories vectorized over heads (H), batch (B), and sequences (N).

		Args:
			z0 (Tensor[L] or Tensor[B,L]): Initial latent(s).
			actions (Tensor[B,N,T,A], optional): Action sequences when `use_policy=False`.
			use_policy (bool): If True, ignore `actions` and sample actions from policy using head-0 latents.
			horizon (int, optional): Number of steps `T` when `use_policy=True`.
			num_rollouts (int, optional): Number of sequences `N` when `use_policy=True`.
			head_mode (str): 'single' | 'all' | 'random'.
			task: Optional task id for multitask.
			policy_action_noise_std (float): Std of Gaussian noise added to policy actions
				when `use_policy=True`. Ignored when `use_policy=False`.

		Returns:
			Tuple[Tensor[H_sel,B,N,T+1,L], Tensor[B,N,T,A]]: Latent trajectories and actions used.
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

		device = next(self.parameters()).device
		# Normalize z0 to [B,L]
		if z0.ndim == 1:
			z0 = z0.unsqueeze(0)  # [1,L] -> [B=1,L]
		B = z0.shape[0]
		L = z0.shape[-1]

		H_total = int(getattr(self.cfg, 'planner_num_dynamics_heads', 1))
		if head_mode == 'single':
			H_sel = 1
			head_indices = [0]
		elif head_mode == 'all':
			H_sel = H_total
			head_indices = list(range(H_total))
		elif head_mode == 'random':
			H_sel = 1
			idx = int(torch.randint(low=0, high=H_total, size=(1,), device=z0.device))
			head_indices = [idx]
		else:
			raise ValueError(f'Unsupported head_mode: {head_mode}')

		# Determine action dim
		if not use_policy:
			A = actions.shape[-1]
		else:
			A = self.cfg.action_dim

		# Build per-step lists to avoid in-place writes on graph tensors
		# t=0 state for all heads: broadcast z0 over N, then tile over heads
		z0_bn = z0.unsqueeze(1).expand(B, N, L)  # float32[B,N,L]
		z0_hbn = torch.stack([z0_bn for _ in range(H_sel)], dim=0)  # float32[H_sel,B,N,L]
		latents_steps = [z0_hbn]
		actions_steps = []  # each entry: float32[B,N,A]

		with maybe_range('WM/rollout_latents', self.cfg):
			# Time loop over T
			for t in range(T):
				with maybe_range('WM/rollout_latents/step_loop', self.cfg):
					# Select actions at time t
					if use_policy:
						with maybe_range('WM/rollout_latents/policy_action', self.cfg):
							# Use head-0 latents for policy sampling
							z_for_pi = latents_steps[t][0].view(B * N, L)  # float32[B*N,L]
							a_flat, _ = self.pi(z_for_pi, task, use_ema=getattr(self.cfg, 'policy_ema_enabled', False))
							a_t = a_flat.view(B, N, A).detach()  # float32[B,N,A]
							if policy_action_noise_std > 0.0:
								noise = torch.randn_like(a_t) * float(policy_action_noise_std)
								a_t = (a_t + noise).clamp(-1.0, 1.0)
					else:
						a_t = actions[:, :, t, :]  # float32[B,N,A]
					actions_steps.append(a_t)

					# Advance each selected head independently
					next_heads = []  # will hold float32[B,N,L] per head
					for hi, h in enumerate(head_indices):
						with maybe_range('WM/rollout_latents/dynamics_head', self.cfg):
							z_curr = latents_steps[t][hi].view(B * N, L)  # float32[B*N,L]
							a_curr = a_t.contiguous().view(B * N, A)  # float32[B*N,A]
							if self.cfg.multitask:
								z_cat = self.task_emb(z_curr, task)  # float32[B*N, L+T]
							else:
								z_cat = z_curr
							za = torch.cat([z_cat, a_curr], dim=-1)  # float32[B*N, L(+T)+A]
							with self._autocast_context():
								next_flat = self._dynamics_heads[h](za)  # float32[B*N,L]
							next_bn = next_flat.float().view(B, N, L)  # float32[B,N,L]
							next_heads.append(next_bn)

					# Aggregate heads for next timestep
					next_hbn = torch.stack(next_heads, dim=0)  # float32[H_sel,B,N,L]
					latents_steps.append(next_hbn)

		# Stack over time to produce final outputs
		with maybe_range('WM/rollout_latents/finalize', self.cfg):
			latents = torch.stack(latents_steps, dim=3).contiguous()  # float32[H_sel,B,N,T+1,L]
			actions_out = torch.stack(actions_steps, dim=2).contiguous()  # float32[B,N,T,A]

		# Cheap sanity: ensure contiguity for downstream views
		assert latents.is_contiguous(), "latents expected contiguous"
		assert actions_out.is_contiguous(), "actions_out expected contiguous"
		return latents, actions_out
