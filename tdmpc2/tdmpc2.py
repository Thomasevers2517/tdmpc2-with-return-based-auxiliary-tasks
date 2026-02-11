import torch
import torch.nn.functional as F

from common import math
from common.planner.planner import Planner
from common.nvtx_utils import maybe_range
from common.scale import RunningScale
from common.world_model import WorldModel
from common.layers import api_model_conversion
from tensordict import TensorDict
from common.logger import get_logger

log = get_logger(__name__)

# -----------------------------------------------------------------------------
# File: tdmpc2.py
# Purpose: Main TD-MPC2 agent class (model-free + model-based planning hybrid).
#
# Conventions & Notation:
#   T  = horizon (cfg.horizon)
#   B  = batch size (cfg.batch_size)
#   A  = action_dim (cfg.action_dim)
#   L  = latent_dim (cfg.latent_dim)
#   K  = num_bins (distributional support for reward & V-value regression)
#   Ve = num_q (ensemble size for V-functions)
#   H  = planner_num_dynamics_heads
#   R  = num_reward_heads
#   S  = replay buffer horizon + 1 (starting states from different replay steps)
#   N  = num_rollouts
# -----------------------------------------------------------------------------


class TDMPC2(torch.nn.Module):
	"""TD-MPC2 agent. Implements training + inference (single-task only)."""

	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		self.device = torch.device('cuda:0')
		self.model = WorldModel(cfg).to(self.device)
		if self.cfg.dtype == 'float16':
			self.autocast_dtype = torch.float16
		elif self.cfg.dtype == 'bfloat16':
			self.autocast_dtype = torch.bfloat16
		else:
			self.autocast_dtype = None

		# ------------------------------------------------------------------
		# Optimizer parameter groups
		# ------------------------------------------------------------------
		# Ensemble-aware LR scaling:
		# When using ensembles with mean-reduced losses, each head sees 1/N of
		# the gradient it would see with a single head. Scale LRs by ensemble size.
		# ------------------------------------------------------------------
		num_dynamics_heads = int(self.cfg.planner_num_dynamics_heads)
		num_reward_heads = int(self.cfg.num_reward_heads)
		num_q = int(self.cfg.num_q)
		ensemble_lr_scaling = self.cfg.ensemble_lr_scaling

		lr_encoder = self.cfg.lr * self.cfg.enc_lr_scale
		if ensemble_lr_scaling:
			lr_dynamics = self.cfg.lr * num_dynamics_heads
			lr_reward = self.cfg.lr * num_reward_heads
			lr_value = self.cfg.lr * num_q / 5
		else:
			lr_dynamics = self.cfg.lr
			lr_reward = self.cfg.lr
			lr_value = self.cfg.lr

		param_groups = [
			{'params': self.model._encoder.parameters(), 'lr': lr_encoder},
			{'params': self.model._dynamics_heads.parameters(), 'lr': lr_dynamics},
			{'params': self.model._Rs.parameters(), 'lr': lr_reward},
			{'params': self.model._termination.parameters() if self.cfg.episodic else [], 'lr': self.cfg.lr},
			{'params': self.model._Vs.parameters(), 'lr': lr_value},
		]

		log.info('Effective learning rates (ensemble_lr_scaling=%s):', ensemble_lr_scaling)
		log.info('  encoder:    %.6f (base_lr * enc_lr_scale = %.4f * %.2f)', lr_encoder, self.cfg.lr, self.cfg.enc_lr_scale)
		if ensemble_lr_scaling:
			log.info('  dynamics:   %.6f (base_lr * %d dynamics heads)', lr_dynamics, num_dynamics_heads)
			log.info('  reward:     %.6f (base_lr * %d reward heads)', lr_reward, num_reward_heads)
			log.info('  value:      %.6f (base_lr * %d / 5 value heads)', lr_value, num_q)
		else:
			log.info('  dynamics:   %.6f (base_lr, no scaling)', lr_dynamics)
			log.info('  reward:     %.6f (base_lr, no scaling)', lr_reward)
			log.info('  value:      %.6f (base_lr, no scaling)', lr_value)

		# Select optimizer class
		optim_type = self.cfg.optimizer_type.lower()
		weight_decay = float(self.cfg.weight_decay)
		if optim_type == 'adamw':
			OptimClass = torch.optim.AdamW
			optim_kwargs = {'weight_decay': weight_decay}
			log.info('Using AdamW optimizer (weight_decay=%.2e)', weight_decay)
		else:
			OptimClass = torch.optim.Adam
			optim_kwargs = {}
			log.info('Using Adam optimizer')

		self.optim = OptimClass(param_groups, lr=self.cfg.lr, capturable=True, **optim_kwargs)
		lr_pi = self.cfg.lr * self.cfg.pi_lr_scale
		log.info('  policy:     %.6f (base_lr * pi_lr_scale = %.4f * %.2f)', lr_pi, self.cfg.lr, self.cfg.pi_lr_scale)

		if self.cfg.dual_policy_enabled:
			pi_params = list(self.model._pi.parameters()) + list(self.model._pi_optimistic.parameters())
			log.info('  dual_policy: enabled (pessimistic + optimistic)')
		else:
			pi_params = self.model._pi.parameters()
		self.pi_optim = OptimClass(pi_params, lr=lr_pi, eps=1e-5, capturable=True, **optim_kwargs)

		# Store initial encoder LR for step-change schedule
		self._enc_lr_initial = lr_encoder
		self._enc_lr_stepped = False

		self.model.eval()
		self.q_scale = RunningScale(cfg, min_scale=1.0)

		# Heuristic for large action spaces
		extra_iter_thresh = int(cfg.extra_iter_action_dim_threshold)
		high_dim_adjustment = 2 * int(extra_iter_thresh > 0 and cfg.action_dim >= extra_iter_thresh)
		self.cfg.iterations += high_dim_adjustment

		self._step = 0
		self.log_detailed = None

		# Frozen random encoder for KNN entropy estimation (state obs only)
		if self.cfg.obs == 'state':
			obs_dim = list(self.cfg.obs_shape.values())[0][0]
			self._knn_encoder = torch.nn.Sequential(
				torch.nn.Linear(obs_dim, 64),
				torch.nn.ReLU(),
				torch.nn.Linear(64, int(self.cfg.knn_entropy_dim)),
			).to(self.device)
			for p in self._knn_encoder.parameters():
				p.requires_grad_(False)
			self._knn_encoder.eval()
		else:
			self._knn_encoder = None

		self.register_buffer(
			"dynamic_entropy_coeff",
			torch.tensor(self.cfg.start_entropy_coeff, device=self.device, dtype=torch.float32),
		)

		# Discount: scalar float tensor
		# NOTE: Keeping as tensor is required for learning (see discount_tensor_note.md).
		self.discount = torch.tensor(self._get_discount(cfg.episode_length), device=self.device)
		self._all_gammas = [float(self.discount)]
		log.info('Episode length: %s', cfg.episode_length)
		log.info('Discount factor: %s', str(self.discount))

		# Modular planner
		self.planner = Planner(cfg=self.cfg, world_model=self.model, scale=None)

		if cfg.compile:
			log.info('Compiling update function with torch.compile...')
			self._compute_loss_components_eager = self._compute_loss_components
			self._compute_loss_components = torch.compile(self._compute_loss_components, mode=self.cfg.compile_type, fullgraph=False)
			self.calc_pi_losses_eager = self.calc_pi_losses
			self.calc_pi_losses = torch.compile(self.calc_pi_losses, mode=self.cfg.compile_type, fullgraph=False)

			@torch.compile(mode=self.cfg.compile_type, fullgraph=False)
			def optim_step():
				self.optim.step()
				return

			@torch.compile(mode=self.cfg.compile_type, fullgraph=False)
			def pi_optim_step():
				self.pi_optim.step()
				return

			self.optim_step = optim_step
			self.pi_optim_step = pi_optim_step
			self.act = torch.compile(self.act, mode=self.cfg.compile_type, dynamic=False)
			self.planner.plan = torch.compile(self.planner.plan, mode=self.cfg.compile_type, fullgraph=False, dynamic=False)
		else:
			self._compute_loss_components_eager = self._compute_loss_components
			self.calc_pi_losses_eager = self.calc_pi_losses
			self.optim_step = self.optim.step
			self.pi_optim_step = self.pi_optim.step

	def reset_planner_state(self):
		"""Reset planner warm-start state at episode boundaries."""
		self.planner.reset_warm_start()

	def _get_discount(self, episode_length):
		"""Returns discount factor for a given episode length.

		Args:
			episode_length (int): Length of the episode.

		Returns:
			float: Discount factor for the task.
		"""
		frac = episode_length / self.cfg.discount_denom
		return min(max((frac - 1) / frac, self.cfg.discount_min), self.cfg.discount_max)

	def save(self, fp):
		"""Save state dict of the agent to filepath."""
		torch.save({"model": self.model.state_dict()}, fp)

	def load(self, fp):
		"""Load a saved state dict from filepath (or dictionary) into current agent."""
		if isinstance(fp, dict):
			state_dict = fp
		else:
			state_dict = torch.load(fp, map_location=torch.get_default_device(), weights_only=False)
		state_dict = state_dict["model"] if "model" in state_dict else state_dict
		state_dict = api_model_conversion(self.model.state_dict(), state_dict)
		self.model.load_state_dict(state_dict)

	@torch.no_grad()
	def act(self, obs, eval_mode: bool = False, mpc: bool = True, eval_head_reduce: str = 'default', log_detailed: bool = False):
		"""Select an action.

		Args:
			obs (Tensor[1, *obs_shape]): Observation (batched with leading dim 1).
			eval_mode (bool): Evaluation flag.
			mpc (bool): Whether to use planning.
			eval_head_reduce (str): 'default' or 'mean' for eval head reduction.
			log_detailed (bool): If True, planner returns PlannerAdvancedInfo.

		Returns:
			Tensor[A]: Action.
			Dict: Planning or policy info.
		"""
		self.model.eval()
		with maybe_range('Agent/act', self.cfg):
			if mpc:
				z0 = self.model.encode(obs)  # float32[1, L]
				value_std_coef_override = 0.0 if eval_head_reduce == 'mean' else None

				chosen_action, planner_info, mean, std = self.planner.plan(
					z0, eval_mode=eval_mode, log_detailed=log_detailed,
					train_noise_multiplier=(0.0 if eval_mode else float(self.cfg.train_act_std_coeff)),
					value_std_coef_override=value_std_coef_override,
					use_warm_start=True,
					update_warm_start=True,
				)
				if planner_info is not None:
					planner_info.action_mean = mean.squeeze(0)  # float32[T, A]
					planner_info.action_std = std.squeeze(0)    # float32[T, A]
				return chosen_action.squeeze(0), planner_info

			# Policy-prior action (non-MPC path)
			z = self.model.encode(obs)
			action_pi, info_pi = self.model.pi(z)
			if eval_mode:
				action_pi = info_pi['mean']
			return action_pi[0], None

	# ------------------------------ Gradient logging helpers ------------------------------
	def _grad_param_groups(self):
		"""Return mapping of component group name -> list of parameters."""
		groups = {}
		enc_params = []
		for enc in self.model._encoder.values():
			enc_params.extend(list(enc.parameters()))
		if len(enc_params) > 0:
			groups["encoder"] = enc_params
		groups["dynamics"] = list(self.model._dynamics_heads.parameters())
		groups["reward"] = list(self.model._Rs.parameters())
		if self.cfg.episodic:
			groups["termination"] = list(self.model._termination.parameters())
		groups["Vs"] = list(self.model._Vs.parameters())
		groups["policy"] = list(self.model._pi.parameters())
		return groups

	@staticmethod
	def _grad_norm(params):
		"""Compute true L2 norm across all gradients."""
		accum = None
		device = None
		for p in params:
			device = p.device
			g = p.grad
			if g is None:
				continue
			val = g.detach().float()
			ss = (val * val).sum()
			accum = ss if accum is None else (accum + ss)
		if accum is None:
			return torch.tensor(0.0, device=device or torch.device('cuda:0'))
		return torch.sqrt(accum)

	def calc_pi_losses(self, z, optimistic=False):
		"""Compute SVG policy loss using V(s) with per-head dynamics.

		Maximizes: r(z, a) + γ * V(z') + entropy_bonus
		where z' is reached via dynamics from policy action.

		Args:
			z (Tensor[T+1, H, B, L]): Per-head latent states from dynamics rollout.
			optimistic (bool): If True, use optimistic policy with +1.0 std_coef.

		Returns:
			Tuple[Tensor, TensorDict]: Policy loss and info dict.
		"""
		assert z.ndim == 4, f"Expected z: [T+1, H, B, L], got {z.shape}"
		T_plus_1, H, B, L = z.shape
		T = T_plus_1 - 1
		N = int(self.cfg.num_rollouts) if self.cfg.pi_multi_rollout else 1
		B_eff = H * B  # effective batch with all heads flattened

		z = z[:-1].contiguous()  # float32[T, H, B, L]

		if optimistic:
			value_std_coef = self.cfg.optimistic_policy_value_std_coef
			entropy_coeff = self.dynamic_entropy_coeff * self.cfg.optimistic_entropy_mult
		else:
			value_std_coef = self.cfg.policy_value_std_coef
			entropy_coeff = self.dynamic_entropy_coeff

		with maybe_range('Agent/update_pi', self.cfg):
			z_for_pi = z.reshape(T, B_eff, L)  # float32[T, H*B, L]
			z_expanded = z_for_pi.unsqueeze(2).expand(T, B_eff, N, L).reshape(T, B_eff * N, L)

			action, info = self.model.pi(z_expanded, optimistic=optimistic)  # float32[T, H*B*N, A]
			A = action.shape[-1]
			gamma = self.discount
			gamma_scalar = float(self.discount)

			# Reward r(z, a) from ALL R reward heads
			z_rew_flat = z_expanded.view(T * B_eff * N, L)
			a_rew_flat = action.view(T * B_eff * N, A)
			reward_logits_all = self.model.reward(z_rew_flat, a_rew_flat, head_mode='all')  # float32[R, T*H*B*N, K]
			R = reward_logits_all.shape[0]
			reward_all = math.two_hot_inv(reward_logits_all, self.cfg)  # float32[R, T*H*B*N, 1]

			# Dynamics with split_data=True
			z_for_dyn = z_expanded.view(T, H, B * N, L).permute(1, 0, 2, 3).reshape(H, T * B * N, L)
			a_for_dyn = action.view(T, H, B * N, A).permute(1, 0, 2, 3).reshape(H, T * B * N, A)
			next_z_all = self.model.next(z_for_dyn, a_for_dyn, split_data=True)  # float32[H, T*B*N, L]

			# V(z') for all Ve value heads
			next_z_flat = next_z_all.reshape(H * T * B * N, L)
			v_next_flat = self.model.V(next_z_flat, return_type='all_values', detach=True)  # float32[Ve, H*T*B*N, 1]
			Ve = v_next_flat.shape[0]
			v_next_per_h = v_next_flat.view(Ve, H, T * B * N, 1)
			v_next_reord = v_next_per_h.view(Ve, H, T, B * N, 1).permute(0, 2, 1, 3, 4).reshape(Ve, T * B_eff * N, 1)

			# Q estimate per sample
			reward_mean = reward_all.mean(dim=0)                         # float32[T*H*B*N, 1]
			reward_std = reward_all.std(dim=0, unbiased=(R > 1))         # float32[T*H*B*N, 1]
			v_mean = v_next_reord.mean(dim=0)                            # float32[T*H*B*N, 1]
			v_std = v_next_reord.std(dim=0, unbiased=(Ve > 1))           # float32[T*H*B*N, 1]

			q_mean = reward_mean + gamma * v_mean
			q_std = reward_std + gamma_scalar * v_std
			q_estimate_flat = q_mean + value_std_coef * q_std
			q_estimate = q_estimate_flat.view(T, B_eff * N, 1)

			# Scale and loss
			if not optimistic:
				self.q_scale.update(q_estimate[0])
			q_scaled = self.q_scale(q_estimate)

			rho_pows = torch.pow(self.cfg.rho, torch.arange(T, device=self.device))
			if self.cfg.normalize_rho_weights:
				rho_pows = rho_pows / rho_pows.sum()

			entropy_term = info["scaled_entropy"]  # float32[T, H*B*N, 1]
			objective = q_scaled + entropy_coeff * entropy_term
			pi_loss = -(objective.mean(dim=(1, 2)) * rho_pows).mean()

			# Logging (average over H and N)
			info_entropy = info["entropy"].view(T, B_eff, N, 1).mean(dim=2)
			info_scaled_entropy = info["scaled_entropy"].view(T, B_eff, N, 1).mean(dim=2)
			info_true_entropy = info["true_entropy"].view(T, B_eff, N, 1).mean(dim=2)
			info_true_scaled_entropy = info["true_scaled_entropy"].view(T, B_eff, N, 1).mean(dim=2)
			info_mean = info["mean"].view(T, B_eff, N, -1).mean(dim=2)
			info_log_std = info["log_std"].view(T, B_eff, N, -1).mean(dim=2)
			info_presquash_mean = info["presquash_mean"].view(T, B_eff, N, -1).mean(dim=2)
			q_estimate_avg = q_estimate.view(T, B_eff, N, 1).mean(dim=2)

			info = TensorDict({
				"pi_loss": pi_loss,
				"pi_loss_weighted": pi_loss * self.cfg.policy_coef,
				"pi_entropy": info_entropy,
				"pi_scaled_entropy": info_scaled_entropy,
				"pi_true_entropy": info_true_entropy,
				"pi_true_scaled_entropy": info_true_scaled_entropy,
				"pi_entropy_multiplier": info["entropy_multiplier"],
				"pi_q_scale": self.q_scale.value,
				"pi_std": info_log_std.mean(),
				"pi_mean": info_mean.mean(),
				"pi_abs_mean": info_mean.abs().mean(),
				"pi_presquash_mean": info_presquash_mean.mean(),
				"pi_presquash_abs_mean": info_presquash_mean.abs().mean(),
				"pi_presquash_abs_std": info_presquash_mean.abs().std(),
				"pi_presquash_abs_min": info_presquash_mean.abs().min(),
				"pi_presquash_abs_median": info_presquash_mean.abs().median(),
				"pi_frac_sat_095": (info_mean.abs() > 0.95).float().mean(),
				"entropy_coeff": self.dynamic_entropy_coeff,
				"entropy_coeff_effective": entropy_coeff,
				"pi_q_estimate_mean": q_estimate_avg.mean(),
				"pi_reward_mean": reward_all.mean(),
				"pi_v_next_mean": v_next_flat.mean(),
				"pi_q_std": q_std.mean(),
				"pi_num_rollouts": float(N),
			}, device=self.device)

			return pi_loss, info

	def update_pi(self, zs):
		"""Update policy using SVG (backprop through world model).

		Args:
			zs (Tensor[T+1, H, B, L]): Per-head latent states from dynamics rollout.

		Returns:
			Tuple[Tensor, TensorDict]: Total policy loss and info dict.
		"""
		assert zs.ndim == 4, f"Expected zs: [T+1, H, B, L], got {zs.shape}"
		log_grads = self.cfg.log_gradients_per_loss and self.log_detailed

		# Pessimistic policy (SVG)
		if not log_grads or not self.cfg.compile:
			pi_loss, info = self.calc_pi_losses(zs, optimistic=False)
		else:
			pi_loss, info = self.calc_pi_losses_eager(zs, optimistic=False)

		# Optimistic policy (if dual policy enabled)
		if self.cfg.dual_policy_enabled:
			opti_pi_loss, opti_info = self.calc_pi_losses(zs, optimistic=True)
			for k, v in opti_info.items():
				info[f'opti_{k}'] = v
			pi_loss = pi_loss + opti_pi_loss

		return pi_loss, info

	@torch.no_grad()
	def _td_target(self, next_z, reward, terminated):
		"""Compute TD-target: r + γ * (1 - terminated) * V(next_z).

		Handles per-head dynamics and reward/value ensemble uncertainty.

		Args:
			next_z (Tensor[T, H, B, L]): Latent states at following time steps.
			reward (Tensor[T, R, H, B, 1]): Rewards with R reward heads, H dynamics heads.
			terminated (Tensor[T, H, B, 1]): Termination signals.

		Returns:
			Tuple[Tensor[Ve, T, B, 1], Tensor[T, B, 1], Tensor[T, B, 1]]:
				(TD-targets per Ve head, td_mean_log, td_std_log).
		"""
		T, H, B, L = next_z.shape
		R = reward.shape[1]

		next_z_flat = next_z.view(T, H * B, L)  # float32[T, H*B, L]
		v_logits_flat = self.model.V(next_z_flat, return_type='all', target=True)  # float32[Ve, T, H*B, K]
		Ve = v_logits_flat.shape[0]
		v_values_flat = math.two_hot_inv(v_logits_flat, self.cfg)  # float32[Ve, T, H*B, 1]
		v_values = v_values_flat.view(Ve, T, H, B, 1)

		discount = self.discount
		discount_scalar = float(self.discount)
		std_coef = float(self.cfg.td_target_std_coef)

		if self.cfg.local_td_bootstrap:
			# LOCAL: Each Ve head bootstraps itself
			r_mean_per_h = reward.mean(dim=1)  # float32[T, H, B, 1]
			r_std_per_h = reward.std(dim=1, unbiased=(R > 1))
			terminated_exp = terminated.unsqueeze(0)  # float32[1, T, H, B, 1]
			r_mean_exp = r_mean_per_h.unsqueeze(0)

			td_mean_per_ve_h = r_mean_exp + discount * (1 - terminated_exp) * v_values  # float32[Ve, T, H, B, 1]
			td_std_exp = r_std_per_h.unsqueeze(0)  # float32[1, T, H, B, 1]
			td_per_ve_h = td_mean_per_ve_h + std_coef * td_std_exp

			dyn_reduction = self.cfg.td_target_dynamics_reduction
			if dyn_reduction == "from_std_coef":
				dyn_reduction = "max" if std_coef > 0 else ("min" if std_coef < 0 else "mean")

			if dyn_reduction == "max":
				td_targets, _ = td_per_ve_h.max(dim=2)
			elif dyn_reduction == "min":
				td_targets, _ = td_per_ve_h.min(dim=2)
			else:
				td_targets = td_per_ve_h.mean(dim=2)

			td_mean_log = td_mean_per_ve_h.mean(dim=(0, 2))  # float32[T, B, 1]
			td_std_log = r_std_per_h.mean(dim=1)

		else:
			# GLOBAL: All Ve heads get same target
			r_mean_per_h = reward.mean(dim=1)
			r_std_per_h = reward.std(dim=1, unbiased=(R > 1))
			v_mean_per_h = v_values.mean(dim=0)
			v_std_per_h = v_values.std(dim=0, unbiased=(Ve > 1))

			td_mean_per_h = r_mean_per_h + discount * (1 - terminated) * v_mean_per_h
			td_std_per_h = r_std_per_h + discount_scalar * v_std_per_h
			td_per_h = td_mean_per_h + std_coef * td_std_per_h

			dyn_reduction = self.cfg.td_target_dynamics_reduction
			if dyn_reduction == "from_std_coef":
				dyn_reduction = "max" if std_coef > 0 else ("min" if std_coef < 0 else "mean")

			if dyn_reduction == "max":
				td_reduced, _ = td_per_h.max(dim=1)
			elif dyn_reduction == "min":
				td_reduced, _ = td_per_h.min(dim=1)
			else:
				td_reduced = td_per_h.mean(dim=1)

			td_targets = td_reduced.unsqueeze(0).expand(Ve, T, B, 1)
			td_mean_log = td_mean_per_h.mean(dim=1)
			td_std_log = td_std_per_h.mean(dim=1)

		return td_targets, td_mean_log, td_std_log

	def world_model_losses(self, z_true, z_target, action, reward, terminated):
		"""Compute world-model losses (consistency, reward, termination).

		Args:
			z_true (Tensor[T+1, B, L]): Encoded latents with gradients.
			z_target (Tensor[T+1, B, L] | None): Stable encoder targets (eval mode).
			action (Tensor[T, B, A]): Replay actions.
			reward (Tensor[T, B, 1]): Replay rewards.
			terminated (Tensor[T, B, 1]): Termination flags.

		Returns:
			Tuple[Tensor, TensorDict, Tensor]:
				wm_loss, info, z_rollout float32[T+1, H, B, L].
		"""
		T, B, A = action.shape
		device = z_true.device
		dtype = z_true.dtype

		rho_pows = torch.pow(self.cfg.rho, torch.arange(T, device=device, dtype=dtype))
		if self.cfg.normalize_rho_weights:
			rho_pows = rho_pows / rho_pows.sum()

		latent_batch_variance = z_true[0].var(dim=0).mean()
		z_consistency_target = z_target if z_target is not None else z_true

		# 1. Dynamics rollout (all H heads, vectorised)
		with maybe_range('Agent/world_model_rollout', self.cfg):
			actions_in = action.permute(1, 0, 2).unsqueeze(1)  # float32[B, 1, T, A]
			lat_all, _ = self.model.rollout_latents(
				z_true[0], actions=actions_in, use_policy=False,
			)  # float32[H, B, 1, T+1, L]

		z_rollout = lat_all[:, :, 0]  # float32[H, B, T+1, L]
		H = z_rollout.shape[0]
		L = z_rollout.shape[-1]

		z_curr = z_rollout[:, :, :-1, :].permute(0, 2, 1, 3)  # float32[H, T, B, L]
		z_next = z_rollout[:, :, 1:, :].permute(0, 2, 1, 3)   # float32[H, T, B, L]

		# 2. Consistency loss
		with maybe_range('WM/consistency', self.cfg):
			target_TBL = z_consistency_target[1:].unsqueeze(0)
			delta = z_next - target_TBL.detach()
			delta_enc = z_next.detach() - target_TBL
			consistency_losses = delta.pow(2).mean(dim=(0, 2, 3))          # float32[T]
			encoder_consistency_losses = delta_enc.pow(2).mean(dim=(0, 2, 3))

		consistency_loss = (rho_pows * consistency_losses).mean()
		encoder_consistency_loss = (rho_pows * encoder_consistency_losses).mean()

		# 3. Reward loss
		with maybe_range('WM/reward_term', self.cfg):
			z_flat = z_curr.permute(1, 0, 2, 3).reshape(T, H * B, L)
			a_flat = action.unsqueeze(1).expand(T, H, B, A).reshape(T, H * B, A)

			reward_logits_all = self.model.reward(z_flat, a_flat, head_mode='all')  # float32[R, T, H*B, K]
			R = reward_logits_all.shape[0]
			K = self.cfg.num_bins

			reward_target_exp = reward.unsqueeze(0).unsqueeze(2).expand(R, T, H, B, 1)
			rew_ce = math.soft_ce(
				reward_logits_all.reshape(R * T * H * B, K),
				reward_target_exp.reshape(R * T * H * B, 1),
				self.cfg,
			).view(R, T, H, B)

			rew_ce_per_t = rew_ce.mean(dim=(0, 2, 3))
			reward_loss = (rho_pows * rew_ce_per_t).mean()

			reward_pred_all = math.two_hot_inv(reward_logits_all, self.cfg).view(R, T, H, B, 1)
			reward_pred = reward_pred_all.mean(dim=(0, 2))
			reward_error = (reward_pred.detach() - reward)

		# 4. Termination loss
		if self.cfg.episodic:
			z_next_flat = z_next.reshape(H * T * B, L)
			term_logits_flat = self.model.termination(z_next_flat, unnormalized=True)
			term_logits = term_logits_flat.view(H, T, B, 1)
			terminated_exp = terminated.unsqueeze(0).expand(H, T, B, 1)
			termination_loss = F.binary_cross_entropy_with_logits(term_logits, terminated_exp)
		else:
			term_logits = torch.zeros(H, T, B, 1, device=device, dtype=dtype)
			termination_loss = torch.zeros((), device=device, dtype=dtype)

		# 5. Weighted total
		warmup_ratio = self.cfg.encoder_consistency_warmup_ratio
		warmup_steps = int(warmup_ratio * self.cfg.steps)
		enc_warmup = 0.0 if self._step < warmup_steps else 1.0

		wm_total = (
			self.cfg.consistency_coef * consistency_loss
			+ self.cfg.encoder_consistency_coef * enc_warmup * encoder_consistency_loss
			+ self.cfg.reward_coef * reward_loss
			+ self.cfg.termination_coef * termination_loss
		)

		# 6. Logging
		info = TensorDict({
			'consistency_losses': consistency_losses,
			'consistency_loss': consistency_loss,
			'consistency_loss_weighted': consistency_losses * self.cfg.consistency_coef * H,
			'encoder_consistency_loss': encoder_consistency_loss,
			'encoder_consistency_loss_weighted': encoder_consistency_losses * self.cfg.encoder_consistency_coef * H,
			'latent_batch_variance': latent_batch_variance,
			'reward_loss': reward_loss,
			'reward_loss_weighted': reward_loss * self.cfg.reward_coef * R,
			'termination_loss': termination_loss,
			'termination_loss_weighted': termination_loss * self.cfg.termination_coef,
			'world_model_loss': wm_total,
		}, device=device, non_blocking=True)

		for t in range(T):
			info.update({
				f'consistency_loss/step{t}': consistency_losses[t],
				f'encoder_consistency_loss/step{t}': encoder_consistency_losses[t],
				f'reward_loss/step{t}': rew_ce_per_t[t],
			}, non_blocking=True)

		if self.log_detailed:
			reward_pred_all_for_log = reward_pred_all.mean(dim=2)
			for i in range(T):
				rp_step = reward_pred_all_for_log[:, i, :, 0]
				info.update({
					f'reward_error_abs_mean/step{i}': reward_error[i].abs().mean(),
					f'reward_error_std/step{i}': reward_error[i].std(),
					f'reward_error_max/step{i}': reward_error[i].abs().max(),
					f'reward_pred_mean/step{i}': rp_step.mean(),
					f'reward_pred_head_std/step{i}': rp_step.std(dim=0).mean(),
				}, non_blocking=True)

		if self.cfg.episodic and self.log_detailed:
			last_logits = term_logits[:, -1].mean(dim=0)
			info.update(math.termination_statistics(torch.sigmoid(last_logits), terminated[-1]), non_blocking=True)

		z_rollout = z_rollout.permute(2, 0, 1, 3).contiguous()  # float32[T+1, H, B, L]
		return wm_total, info, z_rollout

	def imagined_rollout(self, start_z, rollout_len=None, use_target_policy=False):
		"""Roll out imagined trajectories from latent start states.

		Args:
			start_z (Tensor[S, B_orig, L]): Starting latents.
			rollout_len (int): Number of imagination steps (must be 1 when H > 1).
			use_target_policy (bool): If True, use EMA target policy.

		Returns:
			Dict with z_seq, actions, rewards, terminated, termination_logits.
		"""
		S, B_orig, L = start_z.shape
		A = self.cfg.action_dim
		n_rollouts = int(self.cfg.num_rollouts)
		H = int(self.cfg.planner_num_dynamics_heads)

		if H > 1:
			assert rollout_len == 1, (
				f"Multi-head imagination (H={H}) requires rollout_len=1. Got {rollout_len}."
			)

		B_total = S * B_orig
		start_flat = start_z.view(B_total, L)

		with maybe_range('Agent/imagined_rollout', self.cfg):
			latents, actions = self.model.rollout_latents(
				start_flat,
				use_policy=True,
				horizon=rollout_len,
				num_rollouts=n_rollouts,
				use_target_policy=use_target_policy,
			)
		assert latents.shape[0] == H, f"Expected {H} heads, got {latents.shape[0]}"

		B = B_total * n_rollouts

		with maybe_range('Imagined/permute_view', self.cfg):
			lat_perm = latents.permute(3, 0, 1, 2, 4).contiguous()  # float32[T+1, H, B_total, N, L]
			z_seq = lat_perm.view(rollout_len + 1, H, B, L)

		with maybe_range('Imagined/act_seq', self.cfg):
			actions_perm = actions.permute(2, 0, 1, 3).contiguous()
			actions_flat = actions_perm.view(rollout_len, B, A)
			actions_seq = actions_flat.unsqueeze(1)  # float32[T, 1, B, A]

		with maybe_range('Imagined/rewards_term', self.cfg):
			actions_expanded = actions_seq.expand(rollout_len, H, B, A)
			z_for_reward = z_seq[:-1].view(rollout_len, H * B, L)
			actions_for_reward = actions_expanded.reshape(rollout_len, H * B, A)

			reward_logits_all = self.model.reward(z_for_reward, actions_for_reward, head_mode='all')
			R = reward_logits_all.shape[0]
			rewards_flat = math.two_hot_inv(reward_logits_all, self.cfg)
			rewards = rewards_flat.permute(1, 0, 2, 3).view(rollout_len, R, H, B, 1)

			if self.cfg.episodic:
				term_logits_flat = self.model.termination(z_for_reward, unnormalized=True)
				term_logits = term_logits_flat.view(rollout_len, H, B, 1)
				terminated = (torch.sigmoid(term_logits) > 0.5).float()
			else:
				term_logits = torch.zeros(rollout_len, H, B, 1, device=z_seq.device, dtype=z_seq.dtype)
				terminated = torch.zeros(rollout_len, H, B, 1, device=z_seq.device, dtype=z_seq.dtype)

		with maybe_range('Imagined/final_pack', self.cfg):
			z_seq_out = torch.cat([z_seq[:1], z_seq[1:].detach()], dim=0).clone()

		return {
			'z_seq': z_seq_out,
			'actions': actions_seq.detach(),
			'rewards': rewards.detach(),
			'terminated': terminated.detach(),
			'termination_logits': term_logits.detach(),
		}

	def calculate_value_loss(self, z_true, z_rollout, update_world_model):
		"""Compute primary critic loss with proper rho weighting.

		critic_source selects latents for V predictions; critic_target_source selects
		the starting latents for the imagined TD-target rollout.

		Args:
			z_true (Tensor[S, B, L]): True encoded states from encoder.
			z_rollout (Tensor[S, H, B, L]): Dynamics rollout from all H heads.
			update_world_model (bool): Whether WM is being updated.

		Returns:
			Tuple[Tensor, TensorDict, dict]: (loss, info, imagined_data).
		"""
		S, B, L = z_true.shape
		device = z_true.device
		dtype = z_true.dtype
		K = self.cfg.num_bins
		N = int(self.cfg.num_rollouts)
		R = int(self.cfg.num_reward_heads)
		T_imag = int(self.cfg.imagination_horizon)

		rho_pows = torch.pow(self.cfg.rho, torch.arange(S, device=device, dtype=dtype))
		if self.cfg.normalize_rho_weights:
			rho_pows = rho_pows / rho_pows.sum()

		# Select latents for V predictions based on critic_source
		if self.cfg.critic_source == 'replay_rollout':
			z_for_v = self._apply_head_strategy(z_rollout)  # float32[S, B_v, L]
		else:  # replay_true
			z_for_v = z_true  # float32[S, B, L]
		B_v = z_for_v.shape[1]

		# Select starting latents for TD-target imagination
		if self.cfg.critic_target_source == 'replay_true':
			start_z = z_true.detach() if not update_world_model else z_true
		else:  # replay_rollout
			start_z = self._apply_head_strategy(z_rollout).detach()

		# Imagination rollout for TD targets
		imagined = self.imagined_rollout(
			start_z,
			rollout_len=T_imag,
			use_target_policy=self.cfg.td_target_use_ema_policy,
		)

		H = imagined['z_seq'].shape[1]
		BN = B * N
		BN_v = B_v * N

		z_seq = imagined['z_seq'].view(T_imag + 1, H, S, B, N, L)
		rewards = imagined['rewards'].view(T_imag, R, H, S, B, N, 1)
		terminated = imagined['terminated'].view(T_imag, H, S, B, N, 1)
		rewards = rewards.detach()
		terminated = terminated.detach()

		# V predictions on critic_source states
		z_for_v_expanded = z_for_v.unsqueeze(0).unsqueeze(3).expand(T_imag, S, B_v, N, L)
		z_for_v_merged = z_for_v_expanded.reshape(T_imag, S, BN_v, L)
		z_for_v_flat = z_for_v_merged.reshape(T_imag * S * BN_v, L)

		vs_flat = self.model.V(z_for_v_flat, return_type='all')  # float32[Ve, T_imag*S*BN_v, K]
		Ve = vs_flat.shape[0]
		vs = vs_flat.view(Ve, T_imag, S, BN_v, K)

		# TD targets from imagined next states
		with maybe_range('Value/td_target', self.cfg):
			with torch.no_grad():
				next_z = z_seq[1:]
				next_z_flat = next_z.view(T_imag, H, S * BN, L)
				rewards_flat = rewards.view(T_imag, R, H, S * BN, 1)
				terminated_flat = terminated.view(T_imag, H, S * BN, 1)

				td_targets_flat, td_mean_flat, td_std_flat = self._td_target(next_z_flat, rewards_flat, terminated_flat)
				td_targets = td_targets_flat.view(Ve, T_imag, S, BN, 1)
				td_mean = td_mean_flat.view(T_imag, S, BN, 1)
				td_std = td_std_flat.view(T_imag, S, BN, 1)

				# When B_v > B (concat strategy), tile TD targets
				if BN_v != BN:
					H_mult = BN_v // BN
					td_targets = td_targets.unsqueeze(3).expand(Ve, T_imag, S, H_mult, BN, 1).reshape(Ve, T_imag, S, BN_v, 1)
					td_mean = td_mean.unsqueeze(2).expand(T_imag, S, H_mult, BN, 1).reshape(T_imag, S, BN_v, 1)
					td_std = td_std.unsqueeze(2).expand(T_imag, S, H_mult, BN, 1).reshape(T_imag, S, BN_v, 1)

		# Cross-entropy loss
		with maybe_range('Value/ce', self.cfg):
			vs_flat_ce = vs.contiguous().view(Ve * T_imag * S * BN_v, K)
			td_flat_ce = td_targets.contiguous().view(Ve * T_imag * S * BN_v, 1)
			val_ce_flat = math.soft_ce(vs_flat_ce, td_flat_ce, self.cfg)
			val_ce = val_ce_flat.view(Ve, T_imag, S, BN_v)

		val_ce_per_s = val_ce.mean(dim=(0, 1, 3))
		weighted = val_ce_per_s * rho_pows
		loss = weighted.mean()

		# Logging
		info = TensorDict({'value_loss': loss}, device=device, non_blocking=True)
		for s in range(S):
			info.update({f'value_loss/replay_step{s}': val_ce_per_s[s]}, non_blocking=True)

		value_pred = math.two_hot_inv(vs, self.cfg)
		info.update({'td_target_mean': td_mean.mean(), 'td_target_std': td_std.mean()}, non_blocking=True)

		if self.log_detailed:
			info.update({
				'td_target_reduced_mean': td_targets.mean(),
				'td_target_reduced_std': td_targets.std(),
				'td_target_reduced_min': td_targets.min(),
				'td_target_reduced_max': td_targets.max(),
				'value_pred_mean': value_pred.mean(),
				'value_pred_std': value_pred.std(),
				'value_pred_min': value_pred.min(),
				'value_pred_max': value_pred.max(),
			}, non_blocking=True)
			td_targets_with_n = td_targets.view(Ve, T_imag, S, B_v, N, 1)
			info.update({'td_target_std_across_rollouts': td_targets_with_n.std(dim=4).mean()}, non_blocking=True)

		value_error = value_pred - td_targets
		for s in range(S):
			info.update({
				f'value_error_abs_mean/replay_step{s}': value_error[:, :, s].abs().mean(),
				f'value_error_std/replay_step{s}': value_error[:, :, s].std(),
				f'value_error_max/replay_step{s}': value_error[:, :, s].abs().max(),
			}, non_blocking=True)

		imagined_data = {
			'z_seq': z_seq,
			'rewards': rewards,
			'terminated': terminated,
		}
		return loss, info, imagined_data

	def _apply_head_strategy(self, z_hb):
		"""Collapse H dim from [S, H, B, L] → [S, B_v, L] per rollout_head_strategy."""
		strategy = self.cfg.rollout_head_strategy
		S_in, H_in, B_in, L_in = z_hb.shape
		if strategy == 'single':
			return z_hb[:, 0]
		elif strategy == 'concat':
			return z_hb.reshape(S_in, H_in * B_in, L_in)
		elif strategy == 'split':
			chunk = B_in // H_in
			assert chunk * H_in == B_in, f"B={B_in} must be divisible by H={H_in} for split strategy"
			slices = [z_hb[:, h, h * chunk:(h + 1) * chunk] for h in range(H_in)]
			return torch.cat(slices, dim=1)
		else:
			raise ValueError(f"Unknown rollout_head_strategy='{strategy}'")

	def _compute_loss_components(self, obs, action, reward, terminated, update_value, log_grads, update_world_model=True):
		"""Compute world model and value losses.

		Args:
			obs: float32[T+1, B, ...].
			action: float32[T, B, A].
			reward: float32[T, B, 1].
			terminated: float32[T, B, 1].
			update_value: If False, skip value losses.
			log_grads: Whether to log gradient statistics.
			update_world_model: If False, skip WM losses.
		"""
		device = self.device

		def encode_obs(obs_seq, grad_enabled, eval_mode=False):
			"""Encode observations with optional eval mode for stable targets."""
			steps, batch = obs_seq.shape[0], obs_seq.shape[1]
			flat_obs = obs_seq.view(steps * batch, *obs_seq.shape[2:])
			with maybe_range('_compute/encode_obs', self.cfg):
				with torch.set_grad_enabled(grad_enabled and torch.is_grad_enabled()):
					if eval_mode:
						was_training = self.model._encoder.training
						self.model._encoder.eval()
						latents_flat = self.model.encode(flat_obs)
						if was_training:
							self.model._encoder.train()
					else:
						latents_flat = self.model.encode(flat_obs)
			return latents_flat.view(steps, batch, *latents_flat.shape[1:])

		z_true = encode_obs(obs, grad_enabled=True, eval_mode=False)
		z_target = encode_obs(obs, grad_enabled=False, eval_mode=True) if self.cfg.encoder_dropout > 0 else None

		if update_world_model:
			wm_loss, wm_info, z_rollout = self.world_model_losses(z_true, z_target, action, reward, terminated)
		else:
			wm_loss = torch.zeros((), device=device)
			wm_info = TensorDict({}, device=device)
			z_rollout = None

		if update_value:
			assert z_rollout is not None, "Value loss requires z_rollout (update_world_model must be True when update_value is True)"
			value_loss, value_info, imagined_data = self.calculate_value_loss(z_true, z_rollout, update_world_model)
		else:
			value_loss = torch.zeros((), device=device)
			value_info = TensorDict({}, device=device)

		info = TensorDict({}, device=device)
		info.update(wm_info, non_blocking=True)
		info.update(value_info, non_blocking=True)

		critic_weighted = self.cfg.value_coef * value_loss * self.cfg.imagine_value_loss_coef_mult
		total_loss = wm_loss + critic_weighted

		info.update({
			'total_loss': total_loss,
			'wm_loss': wm_loss,
			'value_loss_weighted': critic_weighted,
		}, non_blocking=True)

		return {
			'wm_loss': wm_loss,
			'value_loss': value_loss,
			'info': info,
			'z_true': z_true,
			'z_rollout': z_rollout,
		}

	def _update(self, obs, action, reward, terminated, update_value=True, update_pi=True, update_world_model=True):
		"""Single gradient update step over world model, critic, and policy.

		Args:
			update_value: If True, compute and apply value losses.
			update_pi: If True, update policy.
			update_world_model: If True, compute WM losses.
		"""
		log_grads = self.cfg.log_gradients_per_loss and self.log_detailed

		with maybe_range('Agent/update', self.cfg):
			self.model.train(True)
			if log_grads:
				components = self._compute_loss_components_eager(obs, action, reward, terminated, update_value, log_grads, update_world_model)
			else:
				components = self._compute_loss_components(obs, action, reward, terminated, update_value, log_grads, update_world_model)

			info = components['info']
			z_true = components['z_true']
			z_rollout = components['z_rollout']

			total_loss = info['total_loss']
			self.optim.zero_grad(set_to_none=True)

			if log_grads and update_world_model:
				info = self.probe_wm_gradients(info)

			total_loss.backward()
			grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
			if log_grads:
				self.optim.step()
			else:
				self.optim_step()

			self.optim.zero_grad(set_to_none=True)

			# Policy update
			pi_grad_norm = torch.zeros((), device=self.device)
			pi_info = TensorDict({}, device=self.device)
			if update_pi:
				if z_rollout is not None:
					z_for_pi = z_rollout.detach()  # float32[T+1, H, B, L]
				else:
					z_for_pi = z_true.detach().unsqueeze(1)  # float32[T+1, 1, B, L]

				pi_loss, pi_info = self.update_pi(z_for_pi)
				pi_total = pi_loss * self.cfg.policy_coef
				pi_total.backward()
				if log_grads:
					info = self.probe_pi_gradients(info)

				if self.cfg.dual_policy_enabled:
					pi_params = list(self.model._pi.parameters()) + list(self.model._pi_optimistic.parameters())
				else:
					pi_params = self.model._pi.parameters()
				pi_grad_norm = torch.nn.utils.clip_grad_norm_(pi_params, self.cfg.grad_clip_norm)
				if self.cfg.compile and log_grads:
					self.pi_optim.step()
				else:
					self.pi_optim_step()
				self.pi_optim.zero_grad(set_to_none=True)

			# Soft updates
			if update_value:
				self.model.soft_update_target_V()
			if update_pi:
				self.model.soft_update_target_pi()

			if self._step % self.cfg.log_freq == 0 or self.log_detailed:
				info = self.update_end(info.detach(), grad_norm.detach(), pi_grad_norm.detach(), total_loss.detach(), pi_info.detach())
			else:
				info = TensorDict({}, device=self.device).detach()
		return info.to('cpu')

	@torch.compile(mode='reduce-overhead')
	def update_end(self, info, grad_norm, pi_grad_norm, total_loss, pi_info):
		"""Function called at the end of each update iteration."""
		info.update({
			'grad_norm': grad_norm,
			'pi_grad_norm': pi_grad_norm,
			'total_loss': total_loss.detach()
		}, non_blocking=True)
		info.update(pi_info, non_blocking=True)
		self.model.eval()
		return info.detach().mean()

	def update(self, buffer, step=0, update_value=True, update_pi=True, update_world_model=True):
		"""Main update function. Corresponds to one iteration of model learning.

		Args:
			buffer: Replay buffer.
			step: Current training step.
			update_value: If True, compute and apply value losses.
			update_pi: If True, update policy.
			update_world_model: If True, compute WM losses.

		Returns:
			dict: Dictionary of training statistics.
		"""
		with maybe_range('update/sample_buffer', self.cfg):
			obs, action, reward, terminated, indices = buffer.sample()

		self._step = step
		self.log_detailed = (self._step % self.cfg.log_detail_freq == 0) and update_value

		# Encoder LR step-change
		enc_lr_cutoff = int((1 - self.cfg.enc_lr_step_ratio) * self.cfg.steps)
		if not self._enc_lr_stepped and self._step >= enc_lr_cutoff:
			new_enc_lr = self._enc_lr_initial * self.cfg.enc_lr_step_scale
			self.optim.param_groups[0]['lr'] = new_enc_lr
			self._enc_lr_stepped = True
			log.info('Step %d: encoder LR stepped from %.6f to %.6f',
					 self._step, self._enc_lr_initial, new_enc_lr)

		self.dynamic_entropy_coeff.fill_(self.get_entropy_coeff(self._step))

		torch.compiler.cudagraph_mark_step_begin()

		info = self._update(obs, action, reward, terminated,
							update_value=update_value, update_pi=update_pi,
							update_world_model=update_world_model)

		# Log current encoder LR for W&B tracking
		info['encoder_lr'] = self.optim.param_groups[0]['lr']

		# KNN entropy logging (sparse, only when log_detailed)
		if self.log_detailed and self._knn_encoder is not None:
			with torch.no_grad():
				obs_flat = obs[0]
				encoded = self._knn_encoder(obs_flat)
				batch_knn_entropy = math.compute_knn_entropy(encoded, k=int(self.cfg.knn_entropy_k))
				info['batch_knn_entropy'] = batch_knn_entropy.item()

		return info

	@torch._dynamo.disable()
	def probe_wm_gradients(self, info):
		"""Probe gradient norms from each WM loss component to each parameter group."""
		groups = self._grad_param_groups()

		loss_parts = {
			'consistency': self.cfg.consistency_coef * info['consistency_loss'],
			'encoder_consistency': self.cfg.encoder_consistency_coef * info['encoder_consistency_loss'],
			'reward': self.cfg.reward_coef * info['reward_loss'],
			'value': self.cfg.value_coef * info['value_loss'],
		}
		if self.cfg.episodic:
			loss_parts['termination'] = self.cfg.termination_coef * info['termination_loss']

		self.optim.zero_grad(set_to_none=True)

		flat_params = []
		index = []
		for gname, params in groups.items():
			if gname == 'policy':
				continue
			for p in params:
				if p.requires_grad:
					flat_params.append(p)
					index.append((gname, p))

		for lname, lval in loss_parts.items():
			if (not torch.is_tensor(lval)) or (not lval.requires_grad):
				continue
			grads = torch.autograd.grad(
				lval, flat_params, retain_graph=True, create_graph=False, allow_unused=True,
			)
			per_group_ss = {}
			for (gname, p), g in zip(index, grads):
				if g is None:
					continue
				ss = (g.detach().float() ** 2).sum()
				per_group_ss[gname] = per_group_ss.get(gname, 0.0) + ss.item()
			for gname, ss in per_group_ss.items():
				info.update({f'grad_norm/{lname}/{gname}': (ss ** 0.5)}, non_blocking=True)
		return info

	@torch._dynamo.disable()
	def probe_pi_gradients(self, info):
		"""Probe gradient norms from policy loss to all parameter groups."""
		groups = self._grad_param_groups()
		per_group_ss = {}
		for gname, params in groups.items():
			for p in params:
				if p.requires_grad and p.grad is not None:
					ss = (p.grad.detach().float() ** 2).sum()
					per_group_ss[gname] = per_group_ss.get(gname, 0.0) + ss.item()
		for gname, ss in per_group_ss.items():
			info.update({f'grad_norm/pi_loss/{gname}': (ss ** 0.5)}, non_blocking=True)
		return info

	def validate(self, buffer, num_batches=1):
		"""Perform validation on a separate dataset.

		Args:
			buffer: Replay buffer.
			num_batches (int): Number of batches.

		Returns:
			dict: Dictionary of validation statistics.
		"""
		self.model.eval()
		with torch.no_grad():
			infos = []
			for _ in range(num_batches):
				obs, action, reward, terminated, indices = buffer.sample()
				with maybe_range('Agent/validate', self.cfg):
					self.log_detailed = True
					components = self._compute_loss_components(obs, action, reward, terminated, update_value=True, log_grads=False)
					val_info = components['info']

					z_for_pi = components['z_rollout'].detach() if components['z_rollout'] is not None else components['z_true'].detach().unsqueeze(1)
					pi_loss, pi_info = self.update_pi(z_for_pi)
					val_info.update(pi_info, non_blocking=True)
					self.log_detailed = False
				infos.append(val_info)

			if not infos:
				return TensorDict({}, device=self.device)
			avg_info = TensorDict({}, device=self.device)
			for key in infos[0].keys():
				avg_info[key] = torch.stack([info[key] for info in infos], dim=0).mean(dim=0)
		return avg_info.detach()

	def get_entropy_coeff(self, step):
		"""Get the current entropy coefficient based on the training step."""
		if self.cfg.end_entropy_coeff is None:
			return float(self.cfg.start_entropy_coeff)
		start_dynamic = int(self.cfg.start_dynamic_entropy_ratio * self.cfg.steps)
		if self.cfg.end_dynamic_entropy_ratio == -1:
			end_dynamic = start_dynamic
		else:
			end_dynamic = int(self.cfg.end_dynamic_entropy_ratio * self.cfg.steps)

		if step < start_dynamic:
			return float(self.cfg.start_entropy_coeff)
		elif step > end_dynamic:
			return float(self.cfg.end_entropy_coeff)
		else:
			lin_step = (step - start_dynamic)
			duration_dynamic = end_dynamic - start_dynamic + 1e-6
			if self.cfg.dynamic_entropy_schedule == 'linear':
				coeff = self.cfg.start_entropy_coeff + (self.cfg.end_entropy_coeff - self.cfg.start_entropy_coeff) * (lin_step / duration_dynamic)
			elif self.cfg.dynamic_entropy_schedule == 'exponential':
				ratio = lin_step / duration_dynamic
				coeff = self.cfg.start_entropy_coeff * ((self.cfg.end_entropy_coeff / self.cfg.start_entropy_coeff) ** ratio)
		return float(coeff)
