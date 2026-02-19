from copy import deepcopy
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch import autocast

from common import layers, math, init
from common.nvtx_utils import maybe_range
from tensordict import TensorDict
from tensordict.nn import TensorDictParams


class WorldModel(nn.Module):
	"""TD-MPC2 implicit world model architecture (single-task, V-function only).

	Modules: encoder, dynamics (multi-head ensemble), reward (multi-head ensemble),
	termination (optional), policy, V-function (ensemble).
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

		self._encoder = layers.enc(cfg)

		# Multi-head dynamics ensemble (vmap + functional_call)
		h_total = int(cfg.planner_num_dynamics_heads)
		prior_hidden_div = int(cfg.prior_hidden_div)
		dynamics_prior_scale = float(cfg.dynamics_prior_scale)
		value_prior_scale = float(cfg.value_prior_scale)
		dynamics_num_layers = int(cfg.dynamics_num_layers)
		dynamics_dropout = float(cfg.dynamics_dropout)
		dyn_heads = [
			layers.DynamicsHeadWithPrior(
				in_dim=cfg.latent_dim + cfg.action_dim,
				mlp_dims=dynamics_num_layers * [cfg.mlp_dim],
				out_dim=cfg.latent_dim,
				cfg=cfg,
				prior_hidden_div=prior_hidden_div,
				prior_scale=dynamics_prior_scale,
				dropout=dynamics_dropout,
			)
			for _ in range(h_total)
		]
		for dyn_head in dyn_heads:
			dyn_head.apply(init.weight_init)
		self._dynamics_heads = layers.Ensemble(dyn_heads)
		self._dynamics = self._dynamics_heads  # legacy alias

		# Multi-head reward ensemble
		num_reward_heads = int(cfg.num_reward_heads)
		reward_hidden_dim = cfg.mlp_dim // cfg.reward_dim_div
		num_reward_layers = cfg.num_reward_layers
		reward_dropout = cfg.dropout if cfg.reward_dropout_enabled else 0.0
		reward_mlps = [
			layers.MLPWithPrior(
				in_dim=cfg.latent_dim + cfg.action_dim,
				hidden_dims=num_reward_layers * [reward_hidden_dim],
				out_dim=max(cfg.num_bins, 1),
				prior_hidden_div=prior_hidden_div,
				prior_scale=value_prior_scale,
				dropout=reward_dropout,
				distributional=True,
				cfg=cfg,
			)
			for _ in range(num_reward_heads)
		]
		for mlp_with_prior in reward_mlps:
			mlp_with_prior.main_mlp.apply(init.weight_init)
		self._Rs = layers.Ensemble(reward_mlps)

		# Termination head
		self._termination = layers.mlp(
			cfg.latent_dim, num_reward_layers * [reward_hidden_dim], 1
		) if cfg.episodic else None

		# Policy heads
		self._pi = layers.mlp(cfg.latent_dim, 2 * [cfg.mlp_dim], 2 * cfg.action_dim)
		self._pi_optimistic = layers.mlp(
			cfg.latent_dim, 2 * [cfg.mlp_dim], 2 * cfg.action_dim
		) if cfg.dual_policy_enabled else None

		# V-function ensemble (input is latent only)
		v_mlp_dim = cfg.mlp_dim // cfg.value_dim_div
		num_value_layers = cfg.num_value_layers
		v_mlps = [
			layers.MLPWithPrior(
				in_dim=cfg.latent_dim,
				hidden_dims=num_value_layers * [v_mlp_dim],
				out_dim=max(cfg.num_bins, 1),
				prior_hidden_div=prior_hidden_div,
				prior_scale=value_prior_scale,
				dropout=cfg.dropout,
				distributional=True,
				cfg=cfg,
			)
			for _ in range(cfg.num_q)
		]
		for mlp_with_prior in v_mlps:
			mlp_with_prior.main_mlp.apply(init.weight_init)
		self._Vs = layers.Ensemble(v_mlps)

		self.apply(init.weight_init)
		# Zero-init main network output layers for reward and value heads
		reward_output_layer_key = str(num_reward_layers)
		v_output_layer_key = str(num_value_layers)
		init.zero_([
			self._Rs.params["main_mlp", reward_output_layer_key, "weight"],
			self._Vs.params["main_mlp", v_output_layer_key, "weight"],
		])

		self.register_buffer("log_std_min", torch.tensor(cfg.log_std_min))
		self.register_buffer("log_std_dif", torch.tensor(cfg.log_std_max) - self.log_std_min)
		self._init_target_networks()

	def _autocast_context(self):
		dtype = self.autocast_dtype
		if dtype is None or dtype == torch.float32:
			return nullcontext()
		device = next(self.parameters()).device
		if device.type != 'cuda':
			return nullcontext()
		return autocast(device_type=device.type, dtype=dtype)

	def _init_target_networks(self):
		"""Create target and detach V-networks, and optionally target policy."""
		# Target/detach V-networks
		self._detach_Vs_params = TensorDictParams(self._Vs.params.data, no_convert=True)
		self._target_Vs_params = TensorDictParams(self._Vs.params.data.clone(), no_convert=True)

		with self._detach_Vs_params.data.to("meta").to_module(self._Vs.module):
			self._detach_Vs = deepcopy(self._Vs)
			self._target_Vs = deepcopy(self._Vs)

		delattr(self._detach_Vs, "params")
		self._detach_Vs.__dict__["params"] = self._detach_Vs_params
		delattr(self._target_Vs, "params")
		self._target_Vs.__dict__["params"] = self._target_Vs_params

		# Target policy for TD target imagination (if EMA policy used)
		need_target_pi = self.cfg.td_target_use_ema_policy
		if need_target_pi:
			if not hasattr(self, '_target_pi') or self._target_pi is None:
				self._target_pi = deepcopy(self._pi)
				for p in self._target_pi.parameters():
					p.requires_grad_(False)
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
		repr_str = 'TD-MPC2 World Model\n'
		modules = ['Encoder', 'Dynamics', 'Reward', 'Termination', 'Policy prior', 'V-functions']
		for i, m in enumerate([self._encoder, self._dynamics, self._Rs, self._termination, self._pi, self._Vs]):
			if m == self._termination and not self.cfg.episodic:
				continue
			repr_str += f"{modules[i]}: {m}\n"
		repr_str += "Learnable parameters: {:,}".format(self.total_params)
		return repr_str

	@property
	def total_params(self):
		return sum(p.numel() for p in self.parameters() if p.requires_grad)

	def to(self, *args, **kwargs):
		super().to(*args, **kwargs)
		self._init_target_networks()
		return self

	def train(self, mode=True):
		"""Override train to keep target V-networks in eval mode."""
		super().train(mode)
		self._target_Vs.train(False)
		return self

	def soft_update_target_V(self):
		"""Soft-update target V-networks using Polyak averaging."""
		with maybe_range('WM/soft_update_target_V', self.cfg):
			self._target_Vs_params.lerp_(self._detach_Vs_params, self.cfg.tau)

	def _soft_update_module(self, target_module, source_module, tau):
		"""Generic EMA update from source → target."""
		with torch.no_grad():
			for target, source in zip(target_module.parameters(), source_module.parameters()):
				target.data.lerp_(source.data, tau)

	def soft_update_target_pi(self):
		"""Soft-update target policy networks."""
		if self._target_pi is None:
			return
		self._soft_update_module(self._target_pi, self._pi, self.cfg.policy_ema_tau)
		if self._target_pi_optimistic is not None:
			self._soft_update_module(self._target_pi_optimistic, self._pi_optimistic, self.cfg.policy_ema_tau)

	def encode(self, obs):
		"""Encode observation to latent representation.

		Args:
			obs (Tensor[..., *obs_shape]): Observation.

		Returns:
			Tensor[..., L]: Latent embedding.
		"""
		with maybe_range('WM/encode', self.cfg):
			encoder = self._encoder
			if self.cfg.obs == 'rgb' and obs.ndim == 5:
				with self._autocast_context():
					encoded = [encoder[self.cfg.obs](o) for o in obs]
				return torch.stack([e.float() for e in encoded])
			with self._autocast_context():
				out = encoder[self.cfg.obs](obs)
			return out.float()

	def next(self, z, a, split_data=False):
		"""Predict next latent(s) using all dynamics heads in one vectorized call.

		Args:
			z: Broadcast (split_data=False): Tensor[B, L]. Split (split_data=True): Tensor[H, B, L].
			a: Shape mirrors z (with A instead of L).
			split_data: If True, z and a have a leading H dim sliced per-head.

		Returns:
			Tensor[H, B, L]: Next latent(s) with leading head dimension.
		"""
		with maybe_range('WM/dynamics', self.cfg):
			za = torch.cat([z, a], dim=-1)
			with self._autocast_context():
				out = self._dynamics_heads(za, split_data=split_data)  # float32[H, B, L]
			return out.float()

	def reward(self, z, a, head_mode='single'):
		"""Predict instantaneous reward logits.

		Args:
			z (Tensor[..., L]): Latent states.
			a (Tensor[..., A]): Actions.
			head_mode (str): 'single' → head 0 only, 'all' → all R heads.

		Returns:
			Tensor[R, ..., K]: Reward logits.
		"""
		with maybe_range('WM/reward', self.cfg):
			za = torch.cat([z, a], dim=-1)
			with self._autocast_context():
				out = self._Rs(za)
			out = out.float()
			if head_mode == 'single':
				return out[:1]
			elif head_mode == 'all':
				return out
			else:
				raise ValueError(f"Unsupported head_mode: {head_mode}")

	def termination(self, z, unnormalized=False):
		"""Predict termination signal.

		Args:
			z (Tensor[..., L]): Latent states.
			unnormalized (bool): If True, return raw logits.

		Returns:
			Tensor[..., 1]: Termination probability or logits.
		"""
		with self._autocast_context():
			logits = self._termination(z)
		logits = logits.float()
		if unnormalized:
			return logits
		return torch.sigmoid(logits)

	def pi(self, z, optimistic=False, target=False):
		"""Sample action from the policy prior (Gaussian with squash).

		Args:
			z (Tensor[..., L]): Latent state.
			optimistic (bool): Use optimistic policy (dual_policy mode).
			target (bool): Use EMA target policy.

		Returns:
			Tuple[Tensor[..., A], TensorDict]: Sampled action and info dict.
		"""
		with maybe_range('WM/pi', self.cfg):
			# Select correct policy module
			if target:
				if optimistic and self.cfg.dual_policy_enabled:
					module = self._target_pi_optimistic
				else:
					module = self._target_pi
				if module is None:
					module = self._pi_optimistic if (optimistic and self.cfg.dual_policy_enabled) else self._pi
			else:
				if optimistic and self.cfg.dual_policy_enabled:
					module = self._pi_optimistic
				else:
					module = self._pi

			with self._autocast_context():
				raw = module(z)
			mean, log_std = raw.float().chunk(2, dim=-1)
			log_std = math.log_std(log_std, self.log_std_min, self.log_std_dif)
			eps = torch.randn_like(mean, device=mean.device)

			log_prob_presquash = math.gaussian_logprob(eps, log_std)  # float32[..., 1]
			action_dim = eps.shape[-1]
			presquash_mean = mean

			if self.cfg.bmpc_policy_parameterization:
				# BMPC-style: squash mean only, add noise with clamp
				mean = torch.tanh(presquash_mean)
				action = (mean + eps * log_std.exp()).clamp(-1, 1)
				true_log_prob = log_prob_presquash
				maximized_log_prob = log_prob_presquash
			else:
				presquash_action = presquash_mean + eps * log_std.exp()
				# True entropy: full Jacobian (scale=1.0)
				mean, action, true_log_prob = math.squash(
					presquash_mean, presquash_action, log_prob_presquash, jacobian_scale=1.0
				)
				# Maximized entropy: configured Jacobian scale
				jacobian_scale = float(self.cfg.jacobian_correction_scale)
				_, _, maximized_log_prob = math.squash(
					presquash_mean, presquash_action, log_prob_presquash, jacobian_scale=jacobian_scale
				)

			entropy_multiplier = torch.tensor(
				action_dim, dtype=true_log_prob.dtype, device=true_log_prob.device
			).pow(self.cfg.entropy_action_dim_power)

			true_entropy = -true_log_prob
			true_scaled_entropy = true_entropy * entropy_multiplier
			maximized_entropy = -maximized_log_prob
			maximized_scaled_entropy = maximized_entropy * entropy_multiplier

			info = TensorDict({
				"presquash_mean": presquash_mean,
				"mean": mean,
				"log_std": log_std,
				"action_prob": 1.,
				"entropy": maximized_entropy,
				"scaled_entropy": maximized_scaled_entropy,
				"true_entropy": true_entropy,
				"true_scaled_entropy": true_scaled_entropy,
				"entropy_multiplier": entropy_multiplier,
			}, device=z.device, non_blocking=True)
			return action, info

	def V(self, z, return_type='min', target=False, detach=False, split_data=False):
		"""Compute state-value predictions.

		Args:
			z (Tensor[..., L]): Latent state embeddings.
				Broadcast (split_data=False): all Ve heads see same z.
				Split (split_data=True): z[Ve, *, L], head i sees z[i].
			return_type (str): 'all' → logits, 'all_values' → values per head,
				'min'/'max'/'avg'/'mean' → reduced values.
			target (bool): Use target network.
			detach (bool): Use detached network.
			split_data (bool): If True, z has leading Ve dim sliced per head.

		Returns:
			Tensor: Shape depends on return_type.
		"""
		assert return_type in {'min', 'avg', 'mean', 'max', 'all', 'all_values'}
		with maybe_range('WM/V', self.cfg):
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
			V = math.two_hot_inv(out, self.cfg)  # float32[num_q, ..., 1]
			if return_type == 'all_values':
				return V
			if return_type == "min":
				return torch.amin(V, dim=0)
			if return_type == "max":
				return torch.amax(V, dim=0)
			return V.mean(0)

	def model_diagnostics(
		self,
		z_true: torch.Tensor,
		z_rollout: torch.Tensor,
	) -> TensorDict:
		"""Run diagnostic measurements on the model (called when log_detailed).

		Compares V-function outputs on encoder-derived latents (z_true) vs
		dynamics-derived latents (z_rollout), and reports distributional entropy
		of the V-function softmax logits.

		Args:
			z_true (Tensor[T+1, B, L]): Encoder latents (detached).
			z_rollout (Tensor[T+1, H, B, L]): Dynamics-rollout latents (detached).

		Returns:
			TensorDict with diagnostic statistics.
		"""
		info = TensorDict({}, device=z_true.device)
		info.update(
			self.rollout_value_diagnostics(z_true, z_rollout),
			non_blocking=True,
		)
		return info

	@torch.no_grad()
	def rollout_value_diagnostics(
		self,
		z_true: torch.Tensor,
		z_rollout: torch.Tensor,
	) -> TensorDict:
		"""Compute per-timestep V(z_true) vs V(z_rollout) diagnostics.

		Logs:
		  - |E_h[V(z_h)] - V(z_true)| per timestep (dynamics value drift)
		  - std of V across dynamics heads per timestep (head disagreement)
		  - Distributional entropy of V at t=0 for encoder and dynamics latents

		Args:
			z_true (Tensor[T+1, B, L]): Encoder latents.
			z_rollout (Tensor[T+1, H, B, L]): Dynamics-rollout latents.

		Returns:
			TensorDict with per-timestep and aggregate diagnostics.
		"""
		T_plus_1, B, L = z_true.shape
		_, H, _, _ = z_rollout.shape
		info = TensorDict({}, device=z_true.device)

		# --- V values on true latents ---
		v_true_logits = self.V(
			z_true.reshape(T_plus_1 * B, L), return_type='all',
		)  # float32[Ve, T+1*B, K]
		Ve, _, K = v_true_logits.shape
		v_true_logits = v_true_logits.view(Ve, T_plus_1, B, K)  # float32[Ve, T+1, B, K]

		v_true_vals = math.two_hot_inv(
			v_true_logits.reshape(Ve * T_plus_1 * B, K), self.cfg,
		).view(Ve, T_plus_1, B, 1)  # float32[Ve, T+1, B, 1]
		v_true_mean = v_true_vals.mean(dim=0).squeeze(-1)  # float32[T+1, B]

		# --- V values on rollout latents ---
		z_roll_flat = z_rollout.reshape(T_plus_1 * H * B, L)  # float32[T+1*H*B, L]
		v_roll_logits = self.V(
			z_roll_flat, return_type='all',
		)  # float32[Ve, T+1*H*B, K]
		v_roll_logits = v_roll_logits.view(Ve, T_plus_1, H, B, K)  # float32[Ve, T+1, H, B, K]

		v_roll_vals = math.two_hot_inv(
			v_roll_logits.reshape(Ve * T_plus_1 * H * B, K), self.cfg,
		).view(Ve, T_plus_1, H, B, 1)  # float32[Ve, T+1, H, B, 1]
		v_roll_mean = v_roll_vals.mean(dim=(0, 2)).squeeze(-1)  # float32[T+1, B]

		# Per-head values for std computation: float32[T+1, H, B]
		v_per_head = v_roll_vals.mean(dim=0).squeeze(-1)  # float32[T+1, H, B]

		# --- V(mean_z): value of head-averaged latent ---
		z_roll_avg = z_rollout.mean(dim=1)  # float32[T+1, B, L]
		v_avg_z_logits = self.V(
			z_roll_avg.reshape(T_plus_1 * B, L), return_type='all',
		)  # float32[Ve, T+1*B, K]
		v_avg_z_vals = math.two_hot_inv(
			v_avg_z_logits.reshape(Ve * T_plus_1 * B, K), self.cfg,
		).view(Ve, T_plus_1, B, 1)  # float32[Ve, T+1, B, 1]
		v_avg_latent = v_avg_z_vals.mean(dim=0).squeeze(-1)  # float32[T+1, B]

		# --- Per-timestep logging ---
		for t in range(T_plus_1):
			diff_t = v_roll_mean[t] - v_true_mean[t]  # float32[B]
			diff_avg_t = v_avg_latent[t] - v_true_mean[t]  # float32[B]
			info.update({
				# |E_h[V(z_h)] - V(z_true)|  (dynamics value drift)
				f'v_diag/mean_V_heads_minus_V_encoder/step{t}': diff_t.abs().mean(),
				# |V(E_h[z_h]) - V(z_true)|  (avg-latent value drift)
				f'v_diag/V_avg_latent_minus_V_encoder/step{t}': diff_avg_t.abs().mean(),
				# Std of V across dynamics heads (head disagreement)
				# Guard: unbiased=False when H=1 to avoid NaN from division by zero
				f'v_diag/std_V_across_dyn_heads/step{t}': v_per_head[t].std(dim=0, unbiased=(H > 1)).mean(),
			}, non_blocking=True)

		# --- Bin entropy at t=0 only ---
		v_true_probs_0 = torch.softmax(v_true_logits[:, 0], dim=-1)  # float32[Ve, B, K]
		v_true_entropy_0 = -(v_true_probs_0 * torch.log(v_true_probs_0 + 1e-8)).sum(dim=-1)  # float32[Ve, B]

		v_roll_probs_0 = torch.softmax(v_roll_logits[:, 0], dim=-1)  # float32[Ve, H, B, K]
		v_roll_entropy_0 = -(v_roll_probs_0 * torch.log(v_roll_probs_0 + 1e-8)).sum(dim=-1)  # float32[Ve, H, B]

		info.update({
			'v_diag/bin_entropy_V_encoder_z/step0': v_true_entropy_0.mean(),
			'v_diag/bin_entropy_V_dyn_z/step0': v_roll_entropy_0.mean(),
		}, non_blocking=True)

		# --- Aggregate metrics (averaged over all timesteps and batch) ---
		all_diff = v_roll_mean - v_true_mean  # float32[T+1, B]
		all_diff_avg = v_avg_latent - v_true_mean  # float32[T+1, B]
		info.update({
			'v_diag/mean_V_heads_minus_V_encoder': all_diff.abs().mean(),
			'v_diag/V_avg_latent_minus_V_encoder': all_diff_avg.abs().mean(),
			# Guard: unbiased=False when H=1 to avoid NaN from division by zero
			'v_diag/std_V_across_dyn_heads': v_per_head.std(dim=1, unbiased=(H > 1)).mean(),
		}, non_blocking=True)

		return info

	def rollout_latents(self, z0, actions=None, use_policy=False, horizon=None,
						num_rollouts=None, policy_action_noise_std=0.0,
						use_optimistic_policy=False, use_target_policy=False):
		"""Roll out latent trajectories vectorized over all H dynamics heads.

		Args:
			z0 (Tensor[B, L]): Initial latent(s).
			actions (Tensor[B, N, T, A], optional): Action sequences when use_policy=False.
			use_policy (bool): If True, sample from policy instead of using actions.
			horizon (int, optional): Steps T when use_policy=True.
			num_rollouts (int, optional): Sequences N when use_policy=True.
			policy_action_noise_std (float): Noise std for policy actions.
			use_optimistic_policy (bool): Use optimistic policy if dual_policy enabled.
			use_target_policy (bool): Use EMA target policy.

		Returns:
			Tuple[Tensor[H, B, N, T+1, L], Tensor[B, N, T, A]]:
				Latent trajectories and actions used.
		"""
		if use_policy:
			assert actions is None, 'Provide either actions or use_policy=True, not both.'
			assert horizon is not None and num_rollouts is not None
			T = int(horizon)
			N = int(num_rollouts)
		else:
			assert actions is not None and actions.ndim == 4
			B, N, T, A = actions.shape

		if z0.ndim == 1:
			z0 = z0.unsqueeze(0)
		B = z0.shape[0]
		L = z0.shape[-1]
		H = len(self._dynamics_heads)
		A = actions.shape[-1] if not use_policy else self.cfg.action_dim

		# t=0 state for all heads
		z0_bn = z0.unsqueeze(1).expand(B, N, L)
		z0_hbn = z0_bn.unsqueeze(0).expand(H, B, N, L)
		latents_steps = [z0_hbn]
		actions_steps = []

		with maybe_range('WM/rollout_latents', self.cfg):
			for t in range(T):
				with maybe_range('WM/rollout_latents/step_loop', self.cfg):
					if use_policy:
						with maybe_range('WM/rollout_latents/policy_action', self.cfg):
							z_for_pi = latents_steps[t][0].reshape(B * N, L)
							a_flat, _ = self.pi(z_for_pi, optimistic=use_optimistic_policy, target=use_target_policy)
							a_t = a_flat.reshape(B, N, A).detach()
							if policy_action_noise_std > 0.0:
								noise = torch.randn_like(a_t) * float(policy_action_noise_std)
								a_t = (a_t + noise).clamp(-1.0, 1.0)
					else:
						a_t = actions[:, :, t, :]
					actions_steps.append(a_t)

					with maybe_range('WM/rollout_latents/dynamics_head', self.cfg):
						z_all = latents_steps[t].reshape(H, B * N, L)
						a_all = a_t.contiguous().reshape(B * N, A)
						a_all = a_all.unsqueeze(0).expand(H, -1, -1)
						next_all = self.next(z_all, a_all, split_data=True)
						next_hbn = next_all.reshape(H, B, N, L)
					latents_steps.append(next_hbn)

		with maybe_range('WM/rollout_latents/finalize', self.cfg):
			latents = torch.stack(latents_steps, dim=3).contiguous()
			actions_out = torch.stack(actions_steps, dim=2).contiguous()

		return latents, actions_out
