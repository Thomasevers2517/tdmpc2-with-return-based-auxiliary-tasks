import torch
import torch.nn.functional as F


from common import math
from common.planner.planner import Planner  # New modular planner
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
# Conventions & Notation (single-task unless stated multi-task):
#   T  = horizon (cfg.horizon)
#   B  = batch size (cfg.batch_size)
#   A  = action_dim (cfg.action_dim)
#   L  = latent_dim (cfg.latent_dim)
#   K  = num_bins (distributional support for reward & Q-value regression)
#   Qe = num_q (ensemble size for Q-functions)
#   G_aux = number of auxiliary discounts (len(cfg.multi_gamma_gammas))
#
# Shapes Summary (core tensors used during update):
#   obs              : (T+1, B, *obs_shape)            raw observations subsequence
#   action           : (T,   B, A)                     actions aligned with first T obs
#   reward           : (T,   B, 1)                     scalar rewards (pre two-hot projection)
#   terminated       : (T,   B, 1)                     binary termination flags (0/1)
#   next_z           : (T,   B, L)                     encoded latents for time steps 1..T
#   zs               : (T+1, B, L)                     latent rollout (predicted) with zs[0] = encode(obs[0])
#   _zs              : (T,   B, L)                     alias zs[:-1] for action-aligned states
#   qs (primary)     : (Qe,  T, B, K)                  distributional Q logits per ensemble head
#   td_targets       : (T,   B, K)                     primary distributional TD target (two-hot supervision)
#   reward_preds     : (T,   B, K)                     reward prediction logits
#   termination_pred : (T,   B, 1) (episodic only)     termination logits
#   q_aux_logits     : (T, B, G_aux, K)                auxiliary multi-gamma Q logits (if enabled; no ensemble)
#   aux_td_targets   : (G_aux, T, B, 1)                scalar TD targets per auxiliary gamma
#
# Loss Terms:
#   consistency_loss : latent prediction consistency (averaged over T)
#   reward_loss      : distributional CE over reward bins
#   value_loss       : distributional CE over value bins (primary)
#   aux_value_losses : vector (G_aux,) distributional CE vs scalar targets (aux)
#   termination_loss : BCE (episodic only)
#   total_loss       : weighted sum including optional auxiliary mean loss
#
# Planner (_plan): Uses MPPI/CEM style sampling with mixture of policy prior
# trajectories & Gaussian noise. Q-values used for value estimation via model.
#
# Policy Update: Optimizes expected Q (scaled) + entropy bonus along latent
# trajectory produced by world model (not environment).
#
# Multi-Gamma Extension: Adds auxiliary action-value ensembles predicting
# discounted returns for alternative γ's; training-only supervision improves
# representation without affecting planner or policy targets directly.
# -----------------------------------------------------------------------------


class TDMPC2(torch.nn.Module):
	"""
	TD-MPC2 agent. Implements training + inference.
	Can be used for both single-task and multi-task experiments,
	and supports both state and pixel observations.
	"""

	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		self.device = torch.device('cuda:0')
		self.model = WorldModel(cfg).to(self.device)  # World model modules (encoder, dynamics, reward, termination, policy prior, Q ensembles, aux Q ensembles)
		if self.cfg.dtype == 'float16':
			self.autocast_dtype = torch.float16
		elif self.cfg.dtype == 'bfloat16':
			self.autocast_dtype = torch.bfloat16
		else:	
			self.autocast_dtype = None

  		# ------------------------------------------------------------------
		# Optimizer parameter groups
		# Base groups mirror original implementation; we now optionally append
		# auxiliary Q ensemble parameters (joint or separate) so they are
		# trained with identical learning rate / schedule semantics.
		# NOTE: No auxiliary target networks exist yet; if added later they'd
		# require their own soft-update call.
		# ------------------------------------------------------------------
		param_groups = [
			{'params': self.model._encoder.parameters(), 'lr': self.cfg.lr*self.cfg.enc_lr_scale},
			{'params': self.model._dynamics_heads.parameters()},
			{'params': self.model._reward.parameters()},
			{'params': self.model._termination.parameters() if self.cfg.episodic else []},
			{'params': self.model._Qs.parameters()},
			{'params': self.model._task_emb.parameters() if self.cfg.multitask else []}
		]
		if getattr(self.cfg, 'multi_gamma_gammas', None) and len(self.cfg.multi_gamma_gammas) > 0:
			# Append auxiliary head params (single heads, not ensembles)
			if self.model._aux_joint_Qs is not None:
				param_groups.append({'params': self.model._aux_joint_Qs.parameters()})
			elif self.model._aux_separate_Qs is not None:
				for head in self.model._aux_separate_Qs:
					param_groups.append({'params': head.parameters()})
		self.optim = torch.optim.Adam(param_groups, lr=self.cfg.lr, capturable=True)
		self.pi_optim = torch.optim.Adam(self.model._pi.parameters(), lr=self.cfg.lr, eps=1e-5, capturable=True)
		self.model.eval()
		self.scale = RunningScale(cfg)
		self.cfg.iterations += 2*int(cfg.action_dim >= 20) # Heuristic for large action spaces
		# Logging/instrumentation step counter (used for per-loss gradient logging gating)
		self._step = 0  # incremented at end of _update
		self.log_detailed = None  # whether to log detailed gradients (set via external signal)
		self.register_buffer(
			"dynamic_entropy_coeff",
			torch.tensor(self.cfg.start_entropy_coeff, device=self.device, dtype=torch.float32),
		)
		self.register_buffer(
			"detach_encoder_flag",
			torch.tensor(False, device=self.device, dtype=torch.bool),
		)
		# Discount(s): multi-task -> vector (num_tasks,), single-task -> scalar float
		self.discount = torch.tensor(
			[self._get_discount(ep_len) for ep_len in cfg.episode_lengths], device=self.device
		) if self.cfg.multitask else torch.tensor(self._get_discount(cfg.episode_length), device=self.device)
		# Compose full gamma list internally: primary discount first + auxiliary gammas from config.
		# New semantics: cfg.multi_gamma_gammas contains ONLY auxiliary discounts.
		if getattr(self.cfg, 'multi_gamma_gammas', None):
			self._all_gammas = [float(self.discount)] + list(self.cfg.multi_gamma_gammas)
		else:
			self._all_gammas = [float(self.discount)]
		log.info('Episode length: %s', cfg.episode_length)
		log.info('Discount factor: %s', str(self.discount))

		if self.cfg.policy_ema_enabled:
			log.info('Policy EMA enabled (tau=%s)', self.cfg.policy_ema_tau)
		else:
			log.info('Policy EMA disabled')

		if self.cfg.encoder_ema_enabled:
			log.info('Encoder EMA enabled (tau=%s)', self.cfg.encoder_ema_tau)
		else:
			log.info('Encoder EMA disabled')
		# Modular planner (replaces legacy _plan / _prev_mean logic)
		self.planner = Planner(cfg=self.cfg, world_model=self.model, scale=self.scale)
		if cfg.compile:
			log.info('Compiling update function with torch.compile...')
			# Keep eager references
			self._compute_loss_components_eager = self._compute_loss_components
			# Relax fullgraph to reduce guard creation / trace size
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


			self.act = torch.compile(self.act, mode=self.cfg.compile_type, dynamic=True)
		else:
			self._compute_loss_components_eager = self._compute_loss_components
			self.calc_pi_losses_eager = self.calc_pi_losses
			self.optim_step = self.optim.step
			self.pi_optim_step = self.pi_optim.step

   

	def reset_planner_state(self):
		"""Reset planner warm-start state at episode boundaries."""
		self.planner.reset_warm_start()

	# Legacy `plan` property removed; external callers should use `act(mpc=True)`.

	def _get_discount(self, episode_length):
		"""
		Returns discount factor for a given episode length.
		Simple heuristic that scales discount linearly with episode length.
		Default values should work well for most tasks, but can be changed as needed.

		Args:
			episode_length (int): Length of the episode. Assumes episodes are of fixed length.

		Returns:
			float: Discount factor for the task.
		"""
		frac = episode_length/self.cfg.discount_denom
		return min(max((frac-1)/(frac), self.cfg.discount_min), self.cfg.discount_max)

	def save(self, fp):
		"""
		Save state dict of the agent to filepath.

		Args:
			fp (str): Filepath to save state dict to.
		"""
		torch.save({"model": self.model.state_dict()}, fp)

	def load(self, fp):
		"""
		Load a saved state dict from filepath (or dictionary) into current agent.

		Args:
			fp (str or dict): Filepath or state dict to load.
		"""
		if isinstance(fp, dict):
			state_dict = fp
		else:
			state_dict = torch.load(fp, map_location=torch.get_default_device(), weights_only=False)
		state_dict = state_dict["model"] if "model" in state_dict else state_dict
		state_dict = api_model_conversion(self.model.state_dict(), state_dict)
		self.model.load_state_dict(state_dict)
		self.model.reset_policy_encoder_targets()
		return

	@torch.no_grad()
	def act(self, obs, eval_mode: bool = False, task=None, mpc: bool = True):
		"""Select an action.

		If `mpc=True`, uses modular `Planner` over latent space; else falls back to single policy prior.

		Args:
			obs (Tensor): Observation (already batched with leading dim 1).
			eval_mode (bool): Evaluation flag (planner switches to value-only scoring / argmax selection).
			task: Optional task index (unsupported for planner; passed to policy when mpc=False).
			mpc (bool): Whether to use planning.

		Returns:
			Tensor: Action.
			Dict: Planning or policy info (backward-compatible keys 'mean','std').
		"""
		self.model.eval()
		with maybe_range('Agent/act', self.cfg):
			if task is not None:
				# Preserve prior interface; planner asserts multitask unsupported internally.
				task_tensor = torch.tensor([task], device=self.device)
			else:
				task_tensor = None
			if mpc:
				# Encode observation -> latent start (shape [1,L])
				z0 = self.model.encode(obs, task_tensor)
				chosen_action, planner_info, mean, std = self.planner.plan(
					z0.squeeze(0), task=None, eval_mode=eval_mode, step=self._step,
					train_noise_multiplier=(0.0 if eval_mode else float(self.cfg.train_act_std_coeff))
				)

				# Planner already applies any training noise and clamps
				return chosen_action, planner_info
			# Policy-prior action (non-MPC path)
			z = self.model.encode(obs, task_tensor)
			action_pi, info_pi = self.model.pi(z, task_tensor, use_ema=self.cfg.policy_ema_enabled)
			if eval_mode:
				action_pi = info_pi['mean']
			return action_pi[0], None

	# Legacy helper methods `_estimate_value`, `_plan`, `update_planner_mean` removed.

	# ------------------------------ Gradient logging helpers ------------------------------
	def _grad_param_groups(self):
		"""Return mapping of component group name -> list of parameters.

		Groups: encoder, dynamics, reward, termination (if episodic), Qs, aux_Qs (combined),
		task_emb (if multitask), policy.
		"""
		groups = {}
		# encoder: merge all encoders
		enc_params = []
		for enc in self.model._encoder.values():
			enc_params.extend(list(enc.parameters()))
		if len(enc_params) > 0:
			groups["encoder"] = enc_params
		# dynamics (all heads)
		groups["dynamics"] = list(self.model._dynamics_heads.parameters())
		# reward
		groups["reward"] = list(self.model._reward.parameters())
		# termination (optional)
		if self.cfg.episodic:
			groups["termination"] = list(self.model._termination.parameters())
		# primary Q ensemble
		groups["Qs"] = list(self.model._Qs.parameters())
		# auxiliary Q heads (combined)
		aux_params = []
		if getattr(self.model, "_aux_joint_Qs", None) is not None:
			aux_params.extend(list(self.model._aux_joint_Qs.parameters()))
		elif getattr(self.model, "_aux_separate_Qs", None) is not None:
			for head in self.model._aux_separate_Qs:
				aux_params.extend(list(head.parameters()))
		if len(aux_params) > 0:
			groups["aux_Qs"] = aux_params
		# task embedding (optional)
		if self.cfg.multitask:
			groups["task_emb"] = list(self.model._task_emb.parameters())
		# policy
		groups["policy"] = list(self.model._pi.parameters())
		return groups

	@staticmethod
	def _grad_norm(params):
		"""Compute true L2 norm across all gradients: sqrt(sum_i ||g_i||^2))."""
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

	def calc_hinge_loss(self, presquash_mean: torch.Tensor, rho_pows: torch.Tensor) -> torch.Tensor:
		"""Hinge^p penalty on pre-squash mean μ, aggregated with same time-weighting as policy loss.

		Args:
			presquash_mean: Tensor of shape (T,B,A) or (T,2*B,A) when pred_from=="both".
			rho_pows: Tensor of shape (T,) containing rho^t weights.

		Config (must exist):
			- pi_hinge_power (int)
			- pi_hinge_tau (float)
			- pi_hinge_lambda (float) [not used here, applied at caller]
		"""
		p = int(self.cfg.hinge_power)
		tau = float(self.cfg.hinge_tau)
		if self.cfg.pred_from == "both":
			T = presquash_mean.shape[0]
			B2 = presquash_mean.shape[1]
			assert B2 % 2 == 0, "Expected second dim to be 2*B for pred_from=='both'"
			B = B2 // 2
			mu_both = presquash_mean.view(T, 2, B, -1)
			hinge_tb = F.relu(mu_both.abs() - tau).pow(p).mean(dim=(2, 3))  # (T,2)
			hinge_true = hinge_tb[:, 0].mean() * rho_pows.mean()
			hinge_roll = (hinge_tb[:, 1] * rho_pows).mean()
			return 0.5 * (hinge_true + hinge_roll)
		elif self.cfg.pred_from == "rollout":
			hinge_t = F.relu(presquash_mean.abs() - tau).pow(p).mean(dim=(1, 2))  # (T,)
			return (hinge_t * rho_pows).mean()
		else:  # true_state
			hinge_t = F.relu(presquash_mean.abs() - tau).pow(p).mean(dim=(1, 2))
			return hinge_t.mean() * rho_pows.mean()

	def calc_pi_losses(self, z, task):
		#Z is shape (T,B,L) or (T,2,B,L)
		if self.cfg.pred_from == "both":
			assert z.dim() == 4 and z.shape[1] == 2, "For 'both' pred_from, z must have shape (T,2,B,L)"
			T, B, L = z.shape[0], z.shape[2], z.shape[3]
			z = z.view(T, 2*B, L)  # z: float32[T,2*B,L]
		with maybe_range('Agent/update_pi', self.cfg):
			action, info = self.model.pi(z, task)
			qs = self.model.Q(z, action, task, return_type='avg', detach=True)
			self.scale.update(qs[0])
			qs = self.scale(qs)
			rho_pows = torch.pow(self.cfg.rho,
				torch.arange(z.shape[0], device=self.device)
			)
			# Loss is a weighted sum of Q-values
			if self.cfg.actor_source == "ac":
				if self.cfg.ac_source == "imagine" or self.cfg.ac_source == "replay_rollout":
					pi_loss = (-(self.dynamic_entropy_coeff * info["scaled_entropy"] + qs).mean(dim=(1,2)) * rho_pows).mean()
				else:
					raise NotImplementedError(f"ac_source {self.cfg.ac_source} not implemented for TD-MPC2")
			elif self.cfg.actor_source == "imagine" or self.cfg.actor_source == "replay_rollout":
				pi_loss = (-(self.dynamic_entropy_coeff * info["scaled_entropy"] + qs).mean(dim=(1,2)) * rho_pows).mean()
			elif self.cfg.actor_source == "replay_true":
				pi_loss = (-(self.dynamic_entropy_coeff * info["scaled_entropy"] + qs).mean(dim=(1,2)) * rho_pows.mean()).mean()
    

			# Add hinge^p penalty on pre-squash mean μ
			lam = float(self.cfg.hinge_coef)
			hinge_loss = self.calc_hinge_loss(info["presquash_mean"], rho_pows)
			pi_loss = pi_loss + lam * hinge_loss

			info = TensorDict({
			"pi_loss": pi_loss,
			"pi_loss_weighted": pi_loss * self.cfg.policy_coef,
			"pi_entropy": info["entropy"],
			"pi_scaled_entropy": info["scaled_entropy"],
			"pi_scale": self.scale.value,
			"pi_std": info["log_std"].mean(),
			"pi_mean": info["mean"].mean(),
			"pi_abs_mean": info["mean"].abs().mean(),
			"pi_presquash_mean": info["presquash_mean"].mean(),
			"pi_presquash_abs_mean": info["presquash_mean"].abs().mean(),

			"pi_presquash_abs_std": info["presquash_mean"].abs().std(),
			"pi_presquash_abs_min": info["presquash_mean"].abs().min(),
			"pi_presquash_abs_median": info["presquash_mean"].abs().median(),
			"pi_frac_sat_095": (info["mean"].abs() > 0.95).float().mean(),
			"entropy_coeff": self.dynamic_entropy_coeff
				}, device=self.device)

			return pi_loss, info

	def update_pi(self, zs, task):
		"""
		Update policy using a sequence of latent states.

		Args:
			zs (torch.Tensor): Sequence of latent states.
			task (torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			float: Loss of the policy update.
		"""
		log_grads = self.cfg.log_gradients_per_loss and self.log_detailed
		pi_loss, info = self.calc_pi_losses(zs, task) if (not log_grads or not self.cfg.compile) else self.calc_pi_losses_eager(zs, task)
		# if log_grads:
		# 	# self.probe_pi_gradients(pi_loss, info)

		return pi_loss, info

	@torch.no_grad()
	def _td_target(self, next_z, reward, terminated, task):
		"""
		Compute the TD-target from a reward and the observation at the following time step.

		Args:
			next_z (torch.Tensor): Latent state at the following time step.
			reward (torch.Tensor): Reward at the current time step.
			terminated (torch.Tensor): Termination signal at the current time step.
			task (torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			torch.Tensor: TD-target.
		"""
		# Monte Carlo bootstrap over policy actions.
		# Shapes:
		#   next_z: (T, B, L)
		#   N = self.cfg.n_mc_samples_target
		# We create a contiguous repeated latent tensor along the time axis:
		#   expanded_seq = next_z.repeat(N, 1, 1) -> (N*T, B, L)
		# Make N explicit (N, T, B, L), then flatten only the leading (N*T*B) dims
		# into a single batch to call policy/critic once:
		#   flat_expanded = expanded_seq.view(N, T, B, L).reshape(N*T*B, L)
		T, B, L = next_z.shape
		N = int(self.cfg.n_mc_samples_target)
		expanded_seq = next_z.contiguous().repeat(N, 1, 1)  # (N*T, B, L) contiguous (no stride-0 dims)
		expanded_seq = expanded_seq.view(N, T, B, L)       # (N, T, B, L) — explicit MC dim
		flat_expanded = expanded_seq.reshape(N * T * B, L)  # (N*T*B, L) — single batch for model calls
		action, info = self.model.pi(flat_expanded, task, use_ema=self.cfg.policy_ema_enabled)
		# Evaluate critic on all (N*T*B) latent-action pairs, returns logits over K bins per pair.
		q_values = self.model.Q(flat_expanded, action, task, return_type='min', target=True)  # (N*T*B, 1)
		# Reshape back to (N, T, B, K), average across MC sample
		q_value = q_values.view(N, T, B, -1).mean(dim=0)  #  (T, B, K)
		if self.cfg.sac_style_td:
			scaled_entropy = info["scaled_entropy"].view(N, T, B).mean(dim=0)  # (T, B)
			scale = self.scale.value
			q_value = q_value - self.dynamic_entropy_coeff * scaled_entropy.unsqueeze(-1) * scale
  
		discount = self.discount[task].unsqueeze(-1) if self.cfg.multitask else self.discount
		# Primary TD target distribution (still stored as distributional bins via CE loss)
		return reward + discount * (1 - terminated) * q_value  # 
  

	@torch.no_grad()
	def _td_target_aux(self, next_z, reward, terminated, task):

		G_aux = len(self._all_gammas) - 1
		if G_aux <= 0:
			return None

		N = int(self.cfg.n_mc_samples_target)
		# MC-average over N policy samples for auxiliary discounts.
		# Shapes mirror _td_target:
		#   next_z: (T, B, L)
		#   expanded_seq = next_z.repeat(N, 1, 1) -> (N*T, B, L)
		# IMPORTANT: Q_aux expects inputs with explicit (T, B, ...) leading dims (see world_model.Q_aux).
		# Therefore, we keep the (N*T, B, L) shape (fold N into T) and DO NOT flatten B for Q_aux.
		T, B, L = next_z.shape
		expanded_seq = next_z.contiguous().repeat(N, 1, 1)  # (N*T, B, L)
		# Sample actions on the same (N*T, B, L) grid; policy supports arbitrary leading dims.
		action, _ = self.model.pi(expanded_seq, task, use_ema=self.cfg.policy_ema_enabled)
		# Evaluate auxiliary critics; with return_type!='all' this returns values with shape (G_aux, N*T, B, 1).
		q_values = self.model.Q_aux(expanded_seq, action, task, return_type='min', target=True)  # (G_aux, N*T, B, 1)
		# Reshape (fold N out of T) and average across MC samples -> (G_aux, T, B, 1)
		q_values = q_values.view(G_aux, N, T, B, -1).mean(dim=1)  #  (G_aux, T, B, 1)
		td_targets_aux = []
		discount = self.discount[task].unsqueeze(-1) if self.cfg.multitask else self.discount
		for g in range(G_aux):
			gamma_aux = self._all_gammas[g + 1]
			td_target_aux_g = reward + gamma_aux * discount * (1 - terminated) * q_values[g]  # (T, B, 1)
			td_targets_aux.append(td_target_aux_g)
		return torch.stack(td_targets_aux, dim=0)  # (G_aux, T, B, 1)


	def world_model_losses(self, z_true, z_target, action, reward, terminated, task=None):
		"""Compute world-model losses (consistency, reward, termination)."""
		T, B, _ = action.shape
		device = z_true.device
		dtype = z_true.dtype

		rho_pows = torch.pow(self.cfg.rho, torch.arange(T, device=device, dtype=dtype))  # float32[T]

		consistency_losses = torch.zeros(T, device=device, dtype=dtype)
		encoder_consistency_losses = torch.zeros(T, device=device, dtype=dtype)

		with maybe_range('Agent/world_model_rollout', self.cfg):
			# Use vectorized multi-head rollout over provided actions
			actions_in = action.permute(1, 0, 2).unsqueeze(1)  # [B,1,T,A]
			lat_all, _ = self.model.rollout_latents(
				z_true[0], actions=actions_in, use_policy=False, head_mode='all', task=task
			)  # lat_all: [H,B,1,T+1,L]
			# Consistency over heads: average MSE across heads and batch per time step
			# Align dims to [H,T,B,L] for both predicted and true latents
			with maybe_range('WM/consistency', self.cfg):
				pred_TBL = lat_all[:, :, 0, 1:, :].permute(0, 2, 1, 3)  # float32[H,T,B,L]
				true_TBL = z_true[1:].unsqueeze(0)  # [1,T,B,L]
				delta = pred_TBL - true_TBL.detach()
				delta_enc = pred_TBL.detach() - true_TBL
				consistency_losses = (delta.pow(2).mean(dim=(0, 2, 3)))  # float32[T]
				encoder_consistency_losses = (delta_enc.pow(2).mean(dim=(0, 2, 3)))  # [T]
			# For downstream consumers expecting a single rollout tensor, expose head-0 rollout
			z_rollout = lat_all[0, :, 0].permute(1, 0, 2)  # float32[T+1,B,L]

		consistency_loss = (rho_pows * consistency_losses).mean()
		encoder_consistency_loss = (rho_pows * encoder_consistency_losses).mean()

		branches = []
		if self.cfg.pred_from == 'rollout':
			branches.append({
				'latents': z_rollout[:-1],
				'next_latents': z_rollout[1:],
				'actions': action,
				'reward_target': reward,
				'terminated': terminated,
				'weight_mode': 'rollout'
			})
		elif self.cfg.pred_from == 'both':
			if self.cfg.split_batch:
				if B % 2 != 0:
					raise ValueError('Batch size must be even when split_batch is True and pred_from=="both".')
				half = B // 2
				branches.append({
					'latents': z_true[:-1, :half],
					'next_latents': z_true[1:, :half],
					'actions': action[:, :half],
					'reward_target': reward[:, :half],
					'terminated': terminated[:, :half],
					'weight_mode': 'true'
				})
				branches.append({
					'latents': z_rollout[:-1, half:],
					'next_latents': z_rollout[1:, half:],
					'actions': action[:, half:],
					'reward_target': reward[:, half:],
					'terminated': terminated[:, half:],
					'weight_mode': 'rollout'
				})
			else:
				branches.append({
					'latents': z_true[:-1],
					'next_latents': z_true[1:],
					'actions': action,
					'reward_target': reward,
					'terminated': terminated,
					'weight_mode': 'true'
				})
				branches.append({
					'latents': z_rollout[:-1],
					'next_latents': z_rollout[1:],
					'actions': action,
					'reward_target': reward,
					'terminated': terminated,
					'weight_mode': 'rollout'
				})
		else:
			raise ValueError(f"Unsupported pred_from='{self.cfg.pred_from}' for world_model_losses")

		branch_reward_losses = []
		branch_rew_ce = []
		branch_term_losses = []
		termination_logits_cache = []
		branch_reward_error = []
  
		for branch in branches:
			latents = branch['latents']
			actions_branch = branch['actions']
			reward_target = branch['reward_target']
			terminated_target = branch['terminated']
			next_latents = branch['next_latents']

			# Reward/termination losses; if rollout branch, average over dynamics heads
			if branch['weight_mode'] == 'rollout':
				# Prepare per-head latents: [H,T,B,L] for t..t+T-1 and next latents [H,T,B,L]
				lat_all = lat_all  # [H,B,1,T+1,L]
				lat_TBL = lat_all[:, :, 0, :-1, :].permute(0, 2, 1, 3)  # [H,T,B,L]
				next_TBL = lat_all[:, :, 0, 1:, :].permute(0, 2, 1, 3)  # [H,T,B,L]
				head_rew_losses = []
				head_rew_ce = []
				head_term_losses = []
				head_reward_pred = []
				with maybe_range('WM/reward_term', self.cfg):
					for h in range(lat_TBL.shape[0]):
						reward_logits_h = self.model.reward(lat_TBL[h], actions_branch, task)  # float32[T,B,K]
						rew_ce_h = math.soft_ce(
							reward_logits_h.view(-1, self.cfg.num_bins),  # float32[T*B,K]
							reward_target.view(-1, 1),  # float32[T*B,1]
							self.cfg,
						).view(latents.shape[0], latents.shape[1], 1).mean(dim=1).squeeze(-1)
					head_rew_ce.append(rew_ce_h)
					reward_loss_h = (rho_pows * rew_ce_h).mean()
					head_rew_losses.append(reward_loss_h)
					# Expected reward prediction for error logging
					head_reward_pred.append(math.two_hot_inv(reward_logits_h, self.cfg))  # (T,B,1)
					if self.cfg.episodic:
						term_logits_h = self.model.termination(next_TBL[h], task, unnormalized=True)
						term_loss_h = F.binary_cross_entropy_with_logits(term_logits_h, terminated_target)
					else:
						term_logits_h = torch.zeros_like(reward_target)
						term_loss_h = torch.zeros((), device=device, dtype=dtype)
					head_term_losses.append(term_loss_h)
				reward_loss_branch = torch.stack(head_rew_losses).mean()
				term_loss_branch = torch.stack(head_term_losses).mean()
				# For per-step logging, average CE across heads
				rew_ce = torch.stack(head_rew_ce).mean(dim=0)
				# Average reward prediction across heads for error logging
				reward_pred = torch.stack(head_reward_pred).mean(dim=0)  # (T,B,1)
				# Average term logits across heads if episodic (for stats only)
				term_logits = torch.stack([self.model.termination(next_TBL[h], task, unnormalized=True)
											  if self.cfg.episodic else torch.zeros_like(reward_target)
											  for h in range(lat_TBL.shape[0])]).mean(dim=0)
			else:
				# True-state branch uses single (true) latents
				reward_logits = self.model.reward(latents, actions_branch, task)  # float32[T,B,K]
				with maybe_range('WM/reward_term', self.cfg):
					rew_ce = math.soft_ce(
						reward_logits.view(-1, self.cfg.num_bins),  # float32[T*B,K]
						reward_target.view(-1, 1),  # float32[T*B,1]
						self.cfg,
					).view(latents.shape[0], latents.shape[1], 1).mean(dim=1).squeeze(-1)
				reward_loss_branch = rew_ce.mean() * rho_pows.mean()
				# Expected reward prediction for error logging
				reward_pred = math.two_hot_inv(reward_logits, self.cfg)
				if self.cfg.episodic:
					term_logits = self.model.termination(next_latents, task, unnormalized=True)
					term_loss_branch = F.binary_cross_entropy_with_logits(term_logits, terminated_target)
				else:
					term_logits = torch.zeros_like(reward_target)
					term_loss_branch = torch.zeros((), device=device, dtype=dtype)

			branch_reward_losses.append(reward_loss_branch)
			branch_rew_ce.append(rew_ce)
			branch_term_losses.append(term_loss_branch)
			termination_logits_cache.append(term_logits)
			branch_reward_error.append(reward_pred.detach() - reward_target)

		reward_loss = torch.stack(branch_reward_losses).mean()
		termination_loss = torch.stack(branch_term_losses).mean()
		rew_ce_mean = torch.stack(branch_rew_ce).mean(dim=0)

		wm_total = (
			self.cfg.consistency_coef * consistency_loss
			+ self.cfg.encoder_consistency_coef * encoder_consistency_loss
			+ self.cfg.reward_coef * reward_loss
			+ self.cfg.termination_coef * termination_loss
		)

		info = TensorDict({
			'consistency_losses': consistency_losses,
			'consistency_loss': consistency_loss,
			'consistency_loss_weighted': consistency_losses * self.cfg.consistency_coef,
			'encoder_consistency_loss': encoder_consistency_loss,
			'encoder_consistency_loss_weighted': encoder_consistency_losses * self.cfg.encoder_consistency_coef,
			'reward_loss': reward_loss,
			'reward_loss_weighted': reward_loss * self.cfg.reward_coef,
			'termination_loss': termination_loss,
			'termination_loss_weighted': termination_loss * self.cfg.termination_coef,
			'world_model_loss': wm_total,
		}, device=device, non_blocking=True)

		for t in range(T):
			info.update({
				f'consistency_loss/step{t}': consistency_losses[t],
				f'encoder_consistency_loss/step{t}': encoder_consistency_losses[t],
				f'reward_loss/step{t}': rew_ce_mean[t],
			}, non_blocking=True)
   
		if self.log_detailed:
			for idx, branch in enumerate(branches):
				weight_mode = branch['weight_mode']
				info.update({
					f'reward_loss_{weight_mode}': branch_reward_losses[idx],
					f'reward_loss_{weight_mode}_weighted': branch_reward_losses[idx] * self.cfg.reward_coef,
					f'termination_loss_{weight_mode}': branch_term_losses[idx],
					f'termination_loss_{weight_mode}_weighted': branch_term_losses[idx] * self.cfg.termination_coef,
				}, non_blocking=True)
				reward_error = branch_reward_error[idx]
				for i in range(T):
					info.update({
						f"reward_error_abs_mean/step{i}": reward_error[i].abs().mean(),
						f"reward_error_std/step{i}": reward_error[i].std(),
						f"reward_error_max/step{i}": reward_error[i].abs().max()
					}, non_blocking=True)

		if self.cfg.episodic and self.log_detailed:
			last_logits = torch.stack([logits[-1] for logits in termination_logits_cache]).mean(dim=0)
			info.update(math.termination_statistics(torch.sigmoid(last_logits), terminated[-1]), non_blocking=True)

		return wm_total, info, z_rollout

	# rollout_dynamics removed; world_model.rollout_latents handles vectorized rollouts

	def imagined_rollout(self, start_z, task=None, rollout_len=None):
		"""Roll out imagined trajectories from latent start states using world_model.rollout_latents."""

		S, B, L = start_z.shape  # start_z: float32[S,B,L]
		A = self.cfg.action_dim
		n_rollouts = int(self.cfg.num_rollouts)
		device = start_z.device
		dtype = start_z.dtype

		start_flat = start_z.view(S * B, L)  # float32[B_total,L]
		with maybe_range('Agent/imagined_rollout', self.cfg):
			latents, actions = self.model.rollout_latents(
				start_flat,
				use_policy=True,
				horizon=rollout_len,
				num_rollouts=n_rollouts,
				head_mode='single',
				task=task,
			)
		# latents: float32[1, B_total, N, T+1, L]; actions: float32[B_total, N, T, A]
		with maybe_range('Imagined/permute_view', self.cfg):
			lat_seq = latents[0].permute(2, 0, 1, 3).contiguous()  # float32[T+1, B_total, N, L]
			z_seq = lat_seq.view(rollout_len + 1, S * B * n_rollouts, L)  # float32[T+1, S*B*N, L]
		with maybe_range('Imagined/act_seq', self.cfg):
			actions_seq = actions.permute(2, 0, 1, 3).contiguous().view(rollout_len, S * B * n_rollouts, A)  # float32[T, S*B*N, A]

		# Compute rewards and termination logits along imagined trajectories
		reward_logits = self.model.reward(z_seq[:-1], actions_seq, task)
		rewards = math.two_hot_inv(reward_logits, self.cfg)
		if self.cfg.episodic:
			term_logits = self.model.termination(z_seq[:-1], task, unnormalized=True)
			terminated = (torch.sigmoid(term_logits) > 0.5).float()
		else:
			term_logits = torch.zeros(rollout_len, S * B * n_rollouts, 1, device=device, dtype=dtype)
			terminated = torch.zeros_like(rewards)
		# Avoid in-place detach on a view; build a fresh contiguous tensor

		with maybe_range('Imagined/final_pack', self.cfg):
			z_seq_out = torch.cat([z_seq[:1], z_seq[1:].detach()], dim=0).clone()  # float32[T+1, S*B*N, L]

		return {
			'z_seq': z_seq_out,
			'actions': actions_seq.detach(),
			'rewards': rewards.detach(),
			'terminated': terminated.detach(),
			'termination_logits': term_logits.detach(),
		}



	def calculate_value_loss(self, z_seq, actions, rewards, terminated, full_detach, task=None, z_td=None):
		"""Compute primary critic loss on arbitrary latent sequences."""
		if z_td is None:
			assert self.cfg.ac_source != "replay_rollout", "Need to supply z_td for ac_source=replay_rollout in calculate_value_loss"
		T, B, _ = actions.shape  # actions: float32[T,B,A]
		K  = self.cfg.num_bins
		device = z_seq.device

		rho_pows = torch.pow(self.cfg.rho, torch.arange(T, device=device, dtype=z_seq.dtype))  # float32[T]

		z_seq = z_seq.detach() if full_detach else z_seq  # z_seq: float32[T+1,B,L]
		actions = actions.detach()  # float32[T,B,A]
		rewards = rewards.detach()  # float32[T,B,1]
		terminated = terminated.detach()  # float32[T,B,1]

		qs = self.model.Q(z_seq[:-1], actions, task, return_type='all')  # float32[Qe,T,B,K]
		with maybe_range('Value/td_target', self.cfg):
			with torch.no_grad():
				if z_td is not None:
					td_targets = self._td_target(z_td, rewards, terminated, task)  # float32[T,B,K]
				else:
					td_targets = self._td_target(z_seq[1:], rewards, terminated, task)  # float32[T,B,K]

		Qe = qs.shape[0]
		with maybe_range('Value/ce', self.cfg):
			qs_flat = qs.view(Qe * T * B, K)  # float32[Qe*T*B,K]
			td_expanded = td_targets.unsqueeze(0).expand(Qe, -1, -1, -1)  # float32[Qe,T,B,K]
			td_flat = td_expanded.contiguous().view(Qe * T * B, 1)  # float32[Qe*T*B,1]
			val_ce = math.soft_ce(
				qs_flat,
				td_flat,
				self.cfg,
			).view(Qe, T, B, 1).mean(dim=(0, 2)).squeeze(-1)  # float32[T]

		weighted = val_ce * rho_pows
		loss = weighted.mean()

		info = TensorDict({
			'value_loss': loss
		}, device=device, non_blocking=True)

		for t in range(T):
			info.update({f'value_loss/step{t}': val_ce[t]}, non_blocking=True)
		value_pred = math.two_hot_inv(qs, self.cfg)  # float32[Qe,T,B,1]
		if self.log_detailed:
			info.update({
				'td_target_mean': td_targets.mean(),
				'td_target_std': td_targets.std(),
				'td_target_min': td_targets.min(),
				'td_target_max': td_targets.max(),
				'value_pred_mean': value_pred.mean(),
				'value_pred_std': value_pred.std(),
				'value_pred_min': value_pred.min(),
				'value_pred_max': value_pred.max(),
			}, non_blocking=True)
   
		td_full = td_targets.unsqueeze(0).expand(Qe, -1, -1, -1)  # float32[Qe,T,B,K]
		value_error = (value_pred - td_full.contiguous().view(Qe, T, B, 1))  # float32[Qe,T,B,1]
		for i in range(T):
			info.update({f"value_error_abs_mean/step{i}": value_error[:,i].abs().mean(),
						f"value_error_std/step{i}": value_error[:,i].std(),
						f"value_error_max/step{i}": value_error[:,i].abs().max()}, non_blocking=True)


		return loss, info

	def calculate_aux_value_loss(self, z_seq, actions, rewards, terminated, full_detach, task=None, z_td=None):
		"""Compute auxiliary multi-gamma critic losses."""
		if z_td is None:
			assert self.cfg.aux_value_source != "replay_rollout", "Need to supply z_td for ac_source=replay_rollout in calculate_value_loss"
		if self.model._num_aux_gamma == 0:
			return torch.zeros((), device=z_seq.device), TensorDict({}, device=z_seq.device)

		T, B, _ = actions.shape  # actions: float32[T,B,A]
		device = z_seq.device

		rho_pows = torch.pow(self.cfg.rho, torch.arange(T, device=device, dtype=z_seq.dtype))  # float32[T]

		z_seq = z_seq.detach() if full_detach else z_seq  # z_seq: float32[T+1,B,L]
		actions = actions.detach()  # float32[T,B,A]
		rewards = rewards.detach()  # float32[T,B,1]
		terminated = terminated.detach()  # float32[T,B,1]

		q_aux_logits = self.model.Q_aux(z_seq[:-1], actions, task, return_type='all')  # float32[G_aux,T,B,K] or None
		if q_aux_logits is None:
			return torch.zeros((), device=device), TensorDict({}, device=device)

		with maybe_range('Aux/td_target', self.cfg):
			with torch.no_grad():
				if z_td is not None:
					aux_td_targets = self._td_target_aux(z_td, rewards, terminated, task)
				else:
					aux_td_targets = self._td_target_aux(z_seq[1:], rewards, terminated, task) # bootstrap using rolledout

		G_aux = q_aux_logits.shape[0]
		with maybe_range('Aux/ce', self.cfg):
			qaux_flat = q_aux_logits.contiguous().view(G_aux * T * B, self.cfg.num_bins)  # float32[G_aux*T*B,K]
			aux_targets_flat = aux_td_targets.view(G_aux * T * B, 1)  # float32[G_aux*T*B,1]
			aux_ce = math.soft_ce(
				qaux_flat,
				aux_targets_flat,
				self.cfg,
			).view(G_aux, T, B, 1).mean(dim=2).squeeze(-1)

		weighted = aux_ce * rho_pows.unsqueeze(0)
		losses = weighted.mean(dim=1)  # (G_aux,)
		loss_mean = losses.mean()

		info = TensorDict({
			'aux_value_loss_mean': loss_mean
		}, device=device, non_blocking=True)

		for g, gamma in enumerate(self.cfg.multi_gamma_gammas):
			info.update({
				f'aux_value_loss/gamma{gamma:.4f}': losses[g],
				f'aux_value_loss_weighted/gamma{gamma:.4f}': losses[g] * self.cfg.multi_gamma_loss_weight,
			}, non_blocking=True)

		if self.log_detailed:
			for g, gamma in enumerate(self.cfg.multi_gamma_gammas):
				info.update({
					f'aux_td_target_mean/gamma{gamma:.4f}': aux_td_targets[g].mean(),
					f'aux_td_target_std/gamma{gamma:.4f}': aux_td_targets[g].std(),
				}, non_blocking=True)

		return loss_mean, info

	def _compute_loss_components(self, obs, action, reward, terminated, task, ac_only, log_grads, detach_encoder_active):
		device = self.device

		wm_fn = self.world_model_losses
		value_fn = self.calculate_value_loss
		aux_fn = self.calculate_aux_value_loss

		def encode_obs(obs_seq, use_ema, grad_enabled):
			if detach_encoder_active:
				grad_enabled = False
			steps, batch = obs_seq.shape[0], obs_seq.shape[1]
			flat_obs = obs_seq.view(steps * batch, *obs_seq.shape[2:])  # float32[steps*batch,...]
			if self.cfg.multitask:
				if task is None:
					raise RuntimeError('Multitask encoding requires task indices')
				base_task = task.reshape(-1)
				if base_task.numel() != batch:
					raise ValueError(f'Task batch mismatch: expected {batch}, got {base_task.numel()}')
				task_flat = base_task.repeat(steps).to(flat_obs.device).long()  # int64[steps*batch]
			else:
				task_flat = task
			with maybe_range('_compute/encode_obs', self.cfg):
				with torch.set_grad_enabled(grad_enabled and torch.is_grad_enabled()):
					latents_flat = self.model.encode(flat_obs, task_flat, use_ema=use_ema)  # float32[steps*batch,L]
			if detach_encoder_active:
				latents_flat = latents_flat.detach()
			return latents_flat.view(steps, batch, *latents_flat.shape[1:])  # float32[steps,batch,L]

		if not ac_only:
			z_true = encode_obs(obs, use_ema=False, grad_enabled=True)
			z_target = encode_obs(obs, use_ema=True, grad_enabled=False) if self.cfg.encoder_ema_enabled else None
			wm_loss, wm_info, z_rollout = wm_fn(z_true, z_target, action, reward, terminated, task)
		else:
			z_true = encode_obs(obs, use_ema=False, grad_enabled=True)
			z_rollout = self.rollout_dynamics(z_start=z_true[0], action=action, task=task).detach()
			wm_loss = torch.zeros((), device=device)
			wm_info = TensorDict({}, device=device)

		source_cache = {}

		def fetch_source(name):
			if name in source_cache:
				return source_cache[name]
			if name == 'replay_true':
				src = {
					'z_seq': z_true,
					'actions': action,
					'rewards': reward,
					'terminated': terminated,
					'full_detach': False,
					'z_td': None,
				}
			elif name == 'replay_rollout':
				src = {
					'z_seq': z_rollout,
					'actions': action,
					'rewards': reward,
					'terminated': terminated,
					'full_detach': False,
					'z_td': z_true[1:],
				}
			elif name == 'imagine':
				start_z = z_true.detach() if ac_only else z_true
				imagined = self.imagined_rollout(start_z, task=task, rollout_len=self.cfg.imagination_horizon)
				src = {
					'z_seq': imagined['z_seq'], # rollout_length+1, S*B*n_rollouts, L
					'actions': imagined['actions'],
					'rewards': imagined['rewards'],
					'terminated': imagined['terminated'],
					'full_detach': ac_only or self.cfg.detach_imagine_value,
					'z_td': None,
				}
			else:
				raise ValueError(f'Unsupported value source "{name}"')
			source_cache[name] = src
			return src

		value_inputs = fetch_source(self.cfg.ac_source)
		

		value_loss, value_info = value_fn(
			value_inputs['z_seq'],
			value_inputs['actions'],
			value_inputs['rewards'],
			value_inputs['terminated'],
			value_inputs['full_detach'],
			z_td=value_inputs['z_td'],
			task=task,
		)

		aux_loss = torch.zeros((), device=device)
		aux_info = TensorDict({}, device=device)
		if self.cfg.multi_gamma_loss_weight != 0 and self.model._num_aux_gamma > 0:
			aux_inputs = fetch_source(self.cfg.aux_value_source)

			aux_loss, aux_info = aux_fn(
				aux_inputs['z_seq'],
				aux_inputs['actions'],
				aux_inputs['rewards'],
				aux_inputs['terminated'],
				aux_inputs['full_detach'],
				z_td=aux_inputs['z_td'],
				task=task,
			)

		info = TensorDict({}, device=device)
		info.update(wm_info, non_blocking=True)
		info.update(value_info, non_blocking=True)
		info.update(aux_info, non_blocking=True)
  
		critic_weighted = self.cfg.value_coef * value_loss
		critic_weighted =  critic_weighted * self.cfg.imagine_value_loss_coef_mult if self.cfg.ac_source == 'imagine' else critic_weighted
		aux_weighted =  self.cfg.multi_gamma_loss_weight * aux_loss
		aux_weighted = aux_weighted * self.cfg.imagine_value_loss_coef_mult if self.cfg.aux_value_source == 'imagine' else aux_weighted

		total_loss = wm_loss + critic_weighted + aux_weighted
		info.update({
			'total_loss': total_loss,
			'wm_loss': wm_loss,
			'value_loss_weighted': critic_weighted,
			'aux_loss_mean_weighted': aux_weighted
		}, non_blocking=True)

		return {
			'wm_loss': wm_loss,
			'value_loss': value_loss,
			'aux_loss': aux_loss,
			'info': info,
			'z_true': z_true,
			'z_rollout': z_rollout,
			'value_inputs': value_inputs,

		}

	def _update(self, obs, action, reward, terminated, ac_only, task=None):
		"""Single gradient update step over world model, critic, and policy."""
		log_grads = self.cfg.log_gradients_per_loss and self.log_detailed

		with maybe_range('Agent/update', self.cfg):
			self.model.train(True)
			if log_grads:
				components = self._compute_loss_components_eager(obs, action, reward, terminated, task, ac_only, log_grads, self.detach_encoder_flag.item())
			else:
				components = self._compute_loss_components(obs, action, reward, terminated, task, ac_only, log_grads, self.detach_encoder_flag.item())
    
			wm_loss = components['wm_loss']
			value_loss = components['value_loss']
			aux_loss = components['aux_loss']
			info = components['info']
			z_true = components['z_true']
			z_rollout = components['z_rollout']
			value_inputs = components['value_inputs']
      
			total_loss = info['total_loss']		
   
			if log_grads:
				info = self.probe_wm_gradients(info)

			self.optim.zero_grad(set_to_none=True)
			total_loss.backward()
			grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
			if log_grads:
				self.optim.step()
			else:
				self.optim_step()

			self.optim.zero_grad(set_to_none=True)

			if self.cfg.actor_source == 'ac':
				z_for_pi = value_inputs['z_seq'].detach()
			elif self.cfg.actor_source == 'replay_rollout':
				z_for_pi = z_rollout.detach()
			elif self.cfg.actor_source == 'replay_true':
				z_for_pi = z_true.detach()
    
			pi_loss, pi_info = self.update_pi(z_for_pi, task)
			pi_total = pi_loss * self.cfg.policy_coef
			pi_total.backward()
			pi_grad_norm = torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm)
			if self.cfg.compile and log_grads:
				self.pi_optim.step()
			else:
				self.pi_optim_step()
			self.pi_optim.zero_grad(set_to_none=True)

			self.model.soft_update_target_Q()
			self.model.soft_update_policy_encoder_targets()
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

		if self.cfg.encoder_ema_enabled:
			info.update({
				'encoder_ema_max_delta': torch.tensor(self.model.encoder_target_max_delta, device=self.device)
			}, non_blocking=True)
		if self.cfg.policy_ema_enabled:
			info.update({
				'policy_ema_max_delta': torch.tensor(self.model.policy_target_max_delta, device=self.device)
			}, non_blocking=True)

		self.model.eval()
		#TODO this mean adds a lot of computation time; consider removing or replacing with a more efficient logging
		return info.detach().mean()
  
	def update(self, buffer, step=0, ac_only=False):
		"""
		Main update function. Corresponds to one iteration of model learning.

		Args:
			buffer (common.buffer.Buffer): Replay buffer.

		Returns:
			dict: Dictionary of training statistics.
		"""
		with maybe_range('update/sample_buffer', self.cfg):
			obs, action, reward, terminated, task = buffer.sample()

		self._step = step
		if (((self._step // self.cfg.ac_utd_multiplier) * self.cfg.ac_utd_multiplier) % self.cfg.log_detail_freq == 0) and not ac_only:
			self.log_detailed = True
		else:
			self.log_detailed = False
   
		detach_cutoff = (1 - self.cfg.detach_encoder_ratio) * self.cfg.steps
		if self._step >= detach_cutoff:
			self.detach_encoder_flag.fill_(True)
		else:
			self.detach_encoder_flag.fill_(False)
		self.dynamic_entropy_coeff.fill_(self.get_entropy_coeff(self._step))
		kwargs = {}
		if task is not None:
			kwargs["task"] = task
		torch.compiler.cudagraph_mark_step_begin()
		log_grads = self.cfg.log_gradients_per_loss and self.log_detailed
  
		return self._update(obs, action, reward, terminated, ac_only=ac_only, **kwargs)

		# if log_grads:
			
		# 	return self._update_eager(obs, action, reward, terminated, ac_only=ac_only, **kwargs)
		# else:
		# 	return self._update(obs, action, reward, terminated, ac_only=ac_only, **kwargs)


	@torch._dynamo.disable()
	def probe_wm_gradients(self, info):
		groups = self._grad_param_groups()

		# Build loss parts as you already do
		loss_parts = {
			'consistency': self.cfg.consistency_coef * info['consistency_loss'],
			'encoder_consistency': self.cfg.encoder_consistency_coef * info['encoder_consistency_loss'],
			'reward': self.cfg.reward_coef * info['reward_loss'],
			'value': self.cfg.value_coef * info['value_loss'],
		}
		if self.cfg.episodic:
			loss_parts['termination'] = self.cfg.termination_coef * info['termination_loss']
		if 'aux_value_loss_mean' in info and self.cfg.multi_gamma_loss_weight != 0:
			loss_parts['aux_value_mean'] = self.cfg.multi_gamma_loss_weight * info['aux_value_loss_mean']


		# Drop existing grad buffers so cudagraph outputs are not mutated in-place
		self.optim.zero_grad(set_to_none=True)

		# Flatten all params once for grad calls and remember mapping
		flat_params = []
		index = []  # (group_name, param_obj) pairs
		for gname, params in groups.items():
			if gname == 'policy':  # skip policy here
				continue
			for p in params:
				flat_params.append(p)
				index.append((gname, p))

		for lname, lval in loss_parts.items():
			if (not torch.is_tensor(lval)) or (not lval.requires_grad):
				continue

			grads = torch.autograd.grad(
				lval,
				flat_params,
				retain_graph=True,   # we’ll still do total_loss.backward() later
				create_graph=False,
				allow_unused=True,
			)

			# accumulate L2 per component
			per_group_ss = {}
			for (gname, p), g in zip(index, grads):
				if g is None:
					continue
				ss = (g.detach().float() ** 2).sum()
				per_group_ss[gname] = per_group_ss.get(gname, 0.0) + ss.item()

			for gname, ss in per_group_ss.items():
				info.update({f'grad_norm/{lname}/{gname}': (ss ** 0.5)}, non_blocking=True)
		return info

	def validate(self, buffer, num_batches=1):
		"""
		Perform validation on a separate dataset.

		Args:
			buffer (common.buffer.Buffer): Replay buffer.
			num_batches (int): Number of batches to use for validation.

		Returns:
			dict: Dictionary of validation statistics.
		"""
		self.model.eval()

		with torch.no_grad():
			infos = []
			for _ in range(num_batches):
				obs, action, reward, terminated, task = buffer.sample()
				with maybe_range('Agent/validate', self.cfg):
					self.log_detailed = True
					components = self._compute_loss_components(obs, action, reward, terminated, task, ac_only=False, log_grads=False, detach_encoder_active=False)
					val_info = components['info']

					
					if self.cfg.actor_source == 'ac':
						z_for_pi = components['value_inputs']['z_seq'].detach()
					elif self.cfg.actor_source == 'replay_rollout':
						z_for_pi = components['z_rollout'].detach()
					elif self.cfg.actor_source == 'replay_true':
						z_for_pi = components['z_true'].detach()
      
					pi_loss, pi_info = self.update_pi(z_for_pi, task)
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
		"""
		Get the current entropy coefficient based on the training step.

		Args:
			step (int): Current training step.

		Returns:
			float: Current entropy coefficient.
		"""
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
				coeff = self.cfg.start_entropy_coeff + (self.cfg.end_entropy_coeff - self.cfg.start_entropy_coeff) * (lin_step / (duration_dynamic))
			elif self.cfg.dynamic_entropy_schedule == 'exponential':
				ratio = lin_step / duration_dynamic
				coeff = self.cfg.start_entropy_coeff * ( (self.cfg.end_entropy_coeff / self.cfg.start_entropy_coeff) ** ratio )
		return float(coeff)
