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
#   K  = num_bins (distributional support for reward & V-value regression)
#   Ve = num_q (ensemble size for V-functions, kept as num_q for config compatibility)
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
#   vs (primary)     : (Ve,  T, B, K)                  distributional V logits per ensemble head
#   td_targets       : (T,   B, K)                     primary distributional TD target (two-hot supervision)
#   reward_preds     : (T,   B, K)                     reward prediction logits
#   termination_pred : (T,   B, 1) (episodic only)     termination logits
#   v_aux_logits     : (T, B, G_aux, K)                auxiliary multi-gamma V logits (if enabled; no ensemble)
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
# trajectories & Gaussian noise. V-values used for value estimation via model.
#
# Policy Update: With V-function, we optimize expected V (scaled) + entropy bonus
# by backpropagating through frozen dynamics to find actions that lead to high-value states.
#
# Multi-Gamma Extension: Adds auxiliary state-value heads predicting
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
		self.model = WorldModel(cfg).to(self.device)  # World model modules (encoder, dynamics, reward, termination, policy prior, V ensembles, aux V heads)
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
			{'params': self.model._reward_heads.parameters()},
			{'params': self.model._termination.parameters() if self.cfg.episodic else []},
			{'params': self.model._Vs.parameters()},
			{'params': self.model._task_emb.parameters() if self.cfg.multitask else []}
		]
		if getattr(self.cfg, 'multi_gamma_gammas', None) and len(self.cfg.multi_gamma_gammas) > 0:
			# Append auxiliary head params (single heads, not ensembles)
			if self.model._aux_joint_Vs is not None:
				param_groups.append({'params': self.model._aux_joint_Vs.parameters()})
			elif self.model._aux_separate_Vs is not None:
				for head in self.model._aux_separate_Vs:
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

		Groups: encoder, dynamics, reward, termination (if episodic), Vs, aux_Vs (combined),
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
		groups["reward"] = list(self.model._reward_heads.parameters())
		# termination (optional)
		if self.cfg.episodic:
			groups["termination"] = list(self.model._termination.parameters())
		# primary V ensemble
		groups["Vs"] = list(self.model._Vs.parameters())
		# auxiliary V heads (combined)
		aux_params = []
		if getattr(self.model, "_aux_joint_Vs", None) is not None:
			aux_params.extend(list(self.model._aux_joint_Vs.parameters()))
		elif getattr(self.model, "_aux_separate_Vs", None) is not None:
			for head in self.model._aux_separate_Vs:
				aux_params.extend(list(head.parameters()))
		if len(aux_params) > 0:
			groups["aux_Vs"] = aux_params
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
			presquash_mean: Tensor of shape (T, B, A).
			rho_pows: Tensor of shape (T,) containing rho^t weights.

		Config (must exist):
			- pi_hinge_power (int)
			- pi_hinge_tau (float)
			- pi_hinge_lambda (float) [not used here, applied at caller]
		"""
		p = int(self.cfg.hinge_power)
		tau = float(self.cfg.hinge_tau)
		if self.cfg.pred_from == "rollout":
			hinge_t = F.relu(presquash_mean.abs() - tau).pow(p).mean(dim=(1, 2))  # (T,)
			return (hinge_t * rho_pows).mean()
		else:  # true_state
			hinge_t = F.relu(presquash_mean.abs() - tau).pow(p).mean(dim=(1, 2))
			return hinge_t.mean() * rho_pows.mean()

	def calc_pi_losses(self, z, task):
		"""
		Compute policy loss using state-value function V(s).
		
		For V-learning, we maximize: r(z, a) + γ * V(z') + entropy_bonus
		where z' is reached by taking action a from policy pi(z) and rolling
		through dynamics. The dynamics, reward, and value function are frozen
		during policy optimization (SAC-style backprop through frozen model).
		
		Uses policy_head_reduce to aggregate over both reward heads (R) and
		dynamics heads (H) when computing the Q-estimate.
		
		Args:
			z (Tensor[T, B, L]): Current latent states.
			task: Task identifier for multitask setup.
			
		Returns:
			Tuple[Tensor, TensorDict]: Policy loss and info dict.
		"""
		T, B, L = z.shape
		
		# Ensure contiguity for torch.compile compatibility
		z = z.contiguous()  # Required: z may be non-contiguous after detach/indexing ops
		
		# Get reduction mode for policy loss
		policy_reduce = self.cfg.policy_head_reduce  # 'mean', 'min', or 'max'
			
		with maybe_range('Agent/update_pi', self.cfg):
			# Sample action from policy at current state (policy has gradients)
			action, info = self.model.pi(z, task)  # action: float32[T, B, A]
			
			# Flatten for model calls
			z_flat = z.view(T * B, L)              # float32[T*B, L]
			action_flat = action.view(T * B, -1)  # float32[T*B, A]
			
			# Get discount factor
			if self.cfg.multitask:
				task_flat = task.repeat(T) if task is not None else None
				gamma = self.discount[task_flat].unsqueeze(-1)  # float32[T*B, 1]
			else:
				task_flat = None
				gamma = self.discount  # scalar
			
			# Predict reward r(z, a) from all reward heads
			# reward() returns distributional logits [R, T*B, K], convert to scalar
			# NOTE: Gradients flow through reward/dynamics/V to the action, but only
			# policy params are updated (pi_optim only contains policy parameters).
			reward_logits_all = self.model.reward(z_flat, action_flat, task_flat, head_mode='all')  # float32[R, T*B, K]
			R = reward_logits_all.shape[0]  # number of reward heads
			reward_all = math.two_hot_inv(reward_logits_all, self.cfg)  # float32[R, T*B, 1]
			
			# Reduce over reward heads using policy_head_reduce
			if policy_reduce == 'mean':
				reward_flat = reward_all.mean(dim=0)  # float32[T*B, 1]
			elif policy_reduce == 'min':
				reward_flat = reward_all.min(dim=0).values  # float32[T*B, 1]
			elif policy_reduce == 'max':
				reward_flat = reward_all.max(dim=0).values  # float32[T*B, 1]
			else:
				raise ValueError(f"Invalid policy_head_reduce '{policy_reduce}'. Expected 'mean', 'min', or 'max'.")
			
			# Roll through ALL dynamics heads to get next states z', then reduce V
			# This adds pessimism/optimism based on policy_head_reduce to prevent policy from exploiting errors
			next_z_all = self.model.next(z_flat, action_flat, task_flat, head_mode='all')  # float32[H, T*B, L]
			H = next_z_all.shape[0]  # number of dynamics heads
			
			# Evaluate V(z') for each dynamics head using detached network
			# Use return_type='min' to also take minimum over V ensemble heads
			# Reshape to evaluate all heads at once: [H*T*B, L]
			next_z_all_flat = next_z_all.view(H * T * B, L)  # float32[H*T*B, L]
			task_flat_expanded = task_flat.repeat(H) if task_flat is not None else None
			v_next_all_flat = self.model.V(next_z_all_flat, task_flat_expanded, return_type='min', detach=True)  # float32[H*T*B, 1]
			v_next_all = v_next_all_flat.view(H, T * B, 1)  # float32[H, T*B, 1]
			
			# Reduce over dynamics heads using policy_head_reduce
			if policy_reduce == 'mean':
				v_next_flat = v_next_all.mean(dim=0)  # float32[T*B, 1]
			elif policy_reduce == 'min':
				v_next_flat = v_next_all.min(dim=0).values  # float32[T*B, 1]
			elif policy_reduce == 'max':
				v_next_flat = v_next_all.max(dim=0).values  # float32[T*B, 1]
			else:
				raise ValueError(f"Invalid policy_head_reduce '{policy_reduce}'. Expected 'mean', 'min', or 'max'.")
			
			# Compute Q-like estimate: r(z, a) + γ * V(z')
			# This is what the action "earns" - immediate reward plus discounted future value
			q_estimate_flat = reward_flat + gamma * v_next_flat  # float32[T*B, 1]
			q_estimate = q_estimate_flat.view(T, B, 1)           # float32[T, B, 1]
			
			# Update scale with the Q-estimate (first timestep batch for stability)
			self.scale.update(q_estimate[0])
			q_scaled = self.scale(q_estimate)  # float32[T, B, 1]
			
			# Temporal weighting
			rho_pows = torch.pow(self.cfg.rho, torch.arange(T, device=self.device))  # float32[T]
			
			# Entropy from policy (already scaled by action_dim in model.pi)
			scaled_entropy = info["scaled_entropy"]  # float32[T, B, 1]
			
			# Policy loss: maximize (q_scaled + entropy_coeff * entropy)
			# Apply rho weighting across time
			objective = q_scaled + self.dynamic_entropy_coeff * scaled_entropy  # float32[T, B, 1]
			pi_loss = -(objective.mean(dim=(1, 2)) * rho_pows).mean()

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
				"entropy_coeff": self.dynamic_entropy_coeff,
				"pi_q_estimate_mean": q_estimate.mean(),
				"pi_reward_mean": reward_flat.mean(),
				"pi_v_next_mean": v_next_flat.mean(),
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
		return pi_loss, info

	@torch.no_grad()
	def _td_target(self, next_z, reward, terminated, task):
		"""
		Compute the TD-target from a reward and the observation at the following time step.
		
		With V-function, the TD target is: r + γ * (1 - terminated) * V(next_z)
		For multi-head inputs, computes per-head TD targets then takes min over heads
		for pessimistic value estimation. First reduces over reward heads (R), then
		over dynamics heads (H).

		Args:
			next_z (Tensor[T, H, B, L]): Latent states at following time steps with H dynamics heads.
			reward (Tensor[T, R, H, B, 1]): Rewards with R reward heads, H dynamics heads.
			terminated (Tensor[T, H, B, 1]): Termination signals (no R dimension).
			task (torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			Tuple[Tensor[T, B, 1], Tensor[T, B, 1]]: (Pessimistic TD-target, std across all heads).
		"""
		T, H, B, L = next_z.shape  # next_z: float32[T, H, B, L]
		R = reward.shape[1]  # reward: float32[T, R, H, B, 1]
		
		# Merge H and B for V evaluation: [T, H, B, L] -> [T, H*B, L]
		next_z_flat = next_z.view(T, H * B, L)  # float32[T, H*B, L]
		
		# Evaluate V on next states using target network
		v_values_flat = self.model.V(next_z_flat, task, return_type='min', target=True)  # float32[T, H*B, 1]
		
		# Reshape back to [T, H, B, 1]
		v_values = v_values_flat.view(T, H, B, 1)  # float32[T, H, B, 1]
		
		discount = self.discount[task].unsqueeze(-1) if self.cfg.multitask else self.discount
		
		# Pessimistically reduce over reward heads first: min over R (dim=1)
		reward_pessimistic = reward.min(dim=1).values  # float32[T, H, B, 1]
		
		# Per-head TD targets: r + γ * (1 - done) * V(s')
		td_per_head = reward_pessimistic + discount * (1 - terminated) * v_values  # float32[T, H, B, 1]
		
		# Compute std across dynamics heads before final reduction
		td_std_across_heads = td_per_head.std(dim=1, unbiased=False).squeeze(-1)  # float32[T, B]
		td_std_across_heads = td_std_across_heads.unsqueeze(-1)  # float32[T, B, 1]
		
		# Pessimistic aggregation: min over dynamics heads
		td_targets = td_per_head.min(dim=1).values  # float32[T, B, 1]
		
		return td_targets, td_std_across_heads 
  

	@torch.no_grad()
	def _td_target_aux(self, next_z, reward, terminated, task):
		"""
		Compute auxiliary multi-gamma TD targets with multi-head pessimism.
		First reduces over reward heads (R), then over dynamics heads (H).

		Args:
			next_z (Tensor[T, H, B, L]): Latent states at following time steps with H dynamics heads.
			reward (Tensor[T, R, H, B, 1]): Rewards with R reward heads, H dynamics heads.
			terminated (Tensor[T, H, B, 1]): Termination signals (no R dimension).
			task (torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			Tensor[G_aux, T, B, 1]: Pessimistic TD targets per auxiliary gamma.
		"""
		G_aux = len(self._all_gammas) - 1
		if G_aux <= 0:
			return None

		T, H, B, L = next_z.shape  # next_z: float32[T, H, B, L]
		R = reward.shape[1]  # reward: float32[T, R, H, B, 1]
		
		# Merge H and B for V_aux evaluation: [T, H, B, L] -> [T, H*B, L]
		next_z_flat = next_z.view(T, H * B, L)  # float32[T, H*B, L]
		
		# Evaluate auxiliary V on next states using target network
		# V_aux returns (G_aux, T, H*B, 1) for scalar values
		v_values_flat = self.model.V_aux(next_z_flat, task, return_type='min', target=True)  # float32[G_aux, T, H*B, 1]
		
		# Reshape back to [G_aux, T, H, B, 1]
		v_values = v_values_flat.view(G_aux, T, H, B, 1)  # float32[G_aux, T, H, B, 1]
		
		discount = self.discount[task].unsqueeze(-1) if self.cfg.multitask else self.discount
		
		# Pessimistically reduce over reward heads first: min over R (dim=1)
		reward_pessimistic = reward.min(dim=1).values  # float32[T, H, B, 1]
		
		# Expand reward/terminated for broadcasting: [T, H, B, 1] -> [1, T, H, B, 1]
		reward_exp = reward_pessimistic.unsqueeze(0)  # float32[1, T, H, B, 1]
		terminated_exp = terminated.unsqueeze(0)  # float32[1, T, H, B, 1]
		
		# Compute per-head TD targets for all auxiliary gammas at once
		# gamma_aux: scalar per g, broadcast over [G_aux, T, H, B, 1]
		gammas_aux = torch.tensor(self._all_gammas[1:], device=next_z.device, dtype=next_z.dtype)  # float32[G_aux]
		gammas_aux = gammas_aux.view(G_aux, 1, 1, 1, 1)  # float32[G_aux, 1, 1, 1, 1]
		
		td_per_head = reward_exp + gammas_aux * discount * (1 - terminated_exp) * v_values  # float32[G_aux, T, H, B, 1]
		
		# Pessimistic aggregation: min over dynamics heads (dim=2)
		td_targets_aux = td_per_head.min(dim=2).values  # float32[G_aux, T, B, 1]
		
		return td_targets_aux


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
		elif self.cfg.pred_from == 'true_state':
			branches.append({
				'latents': z_true[:-1],
				'next_latents': z_true[1:],
				'actions': action,
				'reward_target': reward,
				'terminated': terminated,
				'weight_mode': 'true'
			})
		else:
			raise ValueError(f"Unsupported pred_from='{self.cfg.pred_from}'. Must be 'rollout' or 'true_state'.")

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
				H_dyn = lat_TBL.shape[0]  # number of dynamics heads
				L_lat = lat_TBL.shape[3]  # latent dimension
				
				with maybe_range('WM/reward_term', self.cfg):
					# Train all reward heads in parallel using Ensemble
					# Flatten dynamics heads into batch: [H,T,B,L] -> [T, H*B, L]
					lat_flat = lat_TBL.permute(1, 0, 2, 3).reshape(T, H_dyn * B, L_lat)  # float32[T, H*B, L]
					actions_expanded = actions_branch.unsqueeze(1).expand(T, H_dyn, B, -1)  # [T, H, B, A]
					actions_flat = actions_expanded.reshape(T, H_dyn * B, -1)  # float32[T, H*B, A]
					
					# Get all reward head logits: [R, T, H*B, K]
					reward_logits_all = self.model.reward(lat_flat, actions_flat, task, head_mode='all')
					R = reward_logits_all.shape[0]  # number of reward heads
					
					# Compute soft cross-entropy for all R heads: [R, T, H*B, K] -> [R, T, H, B]
					reward_target_exp = reward_target.unsqueeze(0).unsqueeze(2).expand(R, T, H_dyn, B, 1)
					reward_target_flat = reward_target_exp.reshape(R * T * H_dyn * B, 1)  # [R*T*H*B, 1]
					logits_flat = reward_logits_all.reshape(R * T * H_dyn * B, self.cfg.num_bins)  # [R*T*H*B, K]
					
					rew_ce_flat = math.soft_ce(logits_flat, reward_target_flat, self.cfg)  # [R*T*H*B]
					rew_ce_all = rew_ce_flat.view(R, T, H_dyn, B)  # [R, T, H, B]
					
					# Average over reward heads, dynamics heads, and batch; keep T for rho weighting
					rew_ce = rew_ce_all.mean(dim=(0, 2, 3))  # float32[T]
					reward_loss_branch = (rho_pows * rew_ce).mean()
					
					# Expected reward prediction for error logging: average over R and H
					reward_pred_all = math.two_hot_inv(reward_logits_all, self.cfg)  # [R, T, H*B, 1]
					reward_pred_all = reward_pred_all.view(R, T, H_dyn, B, 1)  # [R, T, H, B, 1]
					reward_pred = reward_pred_all.mean(dim=(0, 2))  # [T, B, 1]
					
					# Termination loss over dynamics heads
					head_term_losses = []
					for h in range(H_dyn):
						if self.cfg.episodic:
							term_logits_h = self.model.termination(next_TBL[h], task, unnormalized=True)
							term_loss_h = F.binary_cross_entropy_with_logits(term_logits_h, terminated_target)
						else:
							term_logits_h = torch.zeros_like(reward_target)
							term_loss_h = torch.zeros((), device=device, dtype=dtype)
						head_term_losses.append(term_loss_h)
					term_loss_branch = torch.stack(head_term_losses).mean()
					
					# Average term logits across heads if episodic (for stats only)
					term_logits = torch.stack([self.model.termination(next_TBL[h], task, unnormalized=True)
												  if self.cfg.episodic else torch.zeros_like(reward_target)
												  for h in range(H_dyn)]).mean(dim=0)
			else:
				# True-state branch uses single (true) latents, train all reward heads
				# Get all reward head logits: [R, T, B, K]
				reward_logits_all = self.model.reward(latents, actions_branch, task, head_mode='all')
				R = reward_logits_all.shape[0]
				
				with maybe_range('WM/reward_term', self.cfg):
					# Compute soft cross-entropy for all R heads
					reward_target_exp = reward_target.unsqueeze(0).expand(R, T, B, 1)  # [R, T, B, 1]
					reward_target_flat = reward_target_exp.reshape(R * T * B, 1)  # [R*T*B, 1]
					logits_flat = reward_logits_all.reshape(R * T * B, self.cfg.num_bins)  # [R*T*B, K]
					
					rew_ce_flat = math.soft_ce(logits_flat, reward_target_flat, self.cfg)  # [R*T*B]
					rew_ce_all = rew_ce_flat.view(R, T, B)  # [R, T, B]
					
					# Average over reward heads and batch; keep T for rho weighting
					rew_ce = rew_ce_all.mean(dim=(0, 2))  # float32[T]
					reward_loss_branch = (rho_pows * rew_ce).mean()
					
					# Expected reward prediction for error logging: average over R
					reward_pred_all = math.two_hot_inv(reward_logits_all, self.cfg)  # [R, T, B, 1]
					reward_pred = reward_pred_all.mean(dim=0)  # [T, B, 1]
					
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

		# Scale losses by number of ensemble heads to restore per-head gradient magnitude.
		# Without this, mean() over H heads dilutes gradients by 1/H per head.
		H = int(getattr(self.cfg, 'planner_num_dynamics_heads', 1))  # dynamics heads
		R = int(getattr(self.cfg, 'num_reward_heads', 1))  # reward heads

		wm_total = (
			self.cfg.consistency_coef * consistency_loss * H
			+ self.cfg.encoder_consistency_coef * encoder_consistency_loss * H
			+ self.cfg.reward_coef * reward_loss * R
			+ self.cfg.termination_coef * termination_loss
		)

		info = TensorDict({
			'consistency_losses': consistency_losses,
			'consistency_loss': consistency_loss,
			'consistency_loss_weighted': consistency_losses * self.cfg.consistency_coef * H,
			'encoder_consistency_loss': encoder_consistency_loss,
			'encoder_consistency_loss_weighted': encoder_consistency_losses * self.cfg.encoder_consistency_coef * H,
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
				f'reward_loss/step{t}': rew_ce_mean[t],
			}, non_blocking=True)
   
		if self.log_detailed:
			for idx, branch in enumerate(branches):
				weight_mode = branch['weight_mode']
				info.update({
					f'reward_loss_{weight_mode}': branch_reward_losses[idx],
					f'reward_loss_{weight_mode}_weighted': branch_reward_losses[idx] * self.cfg.reward_coef * R,
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
		"""Roll out imagined trajectories from latent start states using world_model.rollout_latents.

		Uses all dynamics heads (head_mode='all') for multi-head pessimism.
		When rollout_len=1, all heads share the same action since the policy samples
		from the initial state before any dynamics step.

		Args:
			start_z (Tensor[S, B_orig, L]): Starting latents where S = pi_rollout_horizon + 1.
			task: Optional task index for multitask.
			rollout_len (int): Number of imagination steps (must be 1 when H > 1).

		Returns:
			Dict with tensors:
			- z_seq: float32[T_imag+1, H, B_expanded, L] - latent trajectory
			- actions: float32[T_imag, 1, B_expanded, A] - actions (shared across H)
			- rewards: float32[T_imag, R, H, B_expanded, 1] - rewards per reward/dynamics head
			- terminated: float32[T_imag, H, B_expanded, 1] - termination signals

			Where:
			- T_imag = rollout_len (imagination horizon, e.g., 1)
			- H = planner_num_dynamics_heads
			- R = num_reward_heads
			- B_expanded = S * B_orig * num_rollouts

			Note: Due to num_rollouts, z_seq[0] contains each starting state duplicated
			num_rollouts times. When using for policy training (actor_source='ac'),
			consider skipping t=0 to avoid training on duplicated initial states.
		"""
		S, B_orig, L = start_z.shape  # start_z: float32[S, B_orig, L]
		A = self.cfg.action_dim
		n_rollouts = int(self.cfg.num_rollouts)
		H = int(getattr(self.cfg, 'planner_num_dynamics_heads', 1))
		device = start_z.device
		dtype = start_z.dtype

		# Multi-head imagination requires rollout_len=1 so all heads share the same action
		# (policy samples at z[0] before dynamics step)
		if H > 1:
			assert rollout_len == 1, (
				f"Multi-head imagination (H={H}) requires rollout_len=1 so all heads share "
				f"the same action (sampled before dynamics). Got rollout_len={rollout_len}."
			)

		B_total = S * B_orig  # flattened starting batch
		start_flat = start_z.view(B_total, L)  # float32[B_total, L]

		with maybe_range('Agent/imagined_rollout', self.cfg):
			latents, actions = self.model.rollout_latents(
				start_flat,
				use_policy=True,
				horizon=rollout_len,
				num_rollouts=n_rollouts,
				head_mode='all',
				task=task,
			)
		# latents: float32[H, B_total, N, T+1, L]; actions: float32[B_total, N, T, A]

		with maybe_range('Imagined/permute_view', self.cfg):
			# Reshape to [T+1, H, B, L] where B = B_total * n_rollouts
			# latents: [H, B_total, N, T+1, L] -> [T+1, H, B_total*N, L]
			B = B_total * n_rollouts  # final batch dimension
			# permute: [H, B_total, N, T+1, L] -> [T+1, H, B_total, N, L]
			lat_perm = latents.permute(3, 0, 1, 2, 4).contiguous()  # float32[T+1, H, B_total, N, L]
			z_seq = lat_perm.view(rollout_len + 1, H, B, L)  # float32[T+1, H, B, L]

		with maybe_range('Imagined/act_seq', self.cfg):
			# actions: [B_total, N, T, A] -> [T, B_total*N, A] -> [T, 1, B, A]
			actions_perm = actions.permute(2, 0, 1, 3).contiguous()  # float32[T, B_total, N, A]
			actions_flat = actions_perm.view(rollout_len, B, A)  # float32[T, B, A]
			# Add H=1 dim since actions are shared across heads
			actions_seq = actions_flat.unsqueeze(1)  # float32[T, 1, B, A]

		# Compute rewards and termination logits along imagined trajectories per head
		# Need to process each head's latents through reward/termination predictors
		with maybe_range('Imagined/rewards_term', self.cfg):
			# z_seq[:-1]: [T, H, B, L], actions for reward: need [T, H, B, A]
			# Expand actions to match heads: [T, 1, B, A] -> [T, H, B, A]
			actions_expanded = actions_seq.expand(rollout_len, H, B, A)  # float32[T, H, B, A]

			# Flatten H*B for reward/termination calls
			z_for_reward = z_seq[:-1].view(rollout_len, H * B, L)  # float32[T, H*B, L]
			actions_for_reward = actions_expanded.reshape(rollout_len, H * B, A)  # float32[T, H*B, A]

			# Get reward logits from all reward heads: [R, T, H*B, K]
			reward_logits_all = self.model.reward(z_for_reward, actions_for_reward, task, head_mode='all')
			R = reward_logits_all.shape[0]  # number of reward heads
			# Convert to scalar rewards: [R, T, H*B, 1]
			rewards_flat = math.two_hot_inv(reward_logits_all, self.cfg)  # float32[R, T, H*B, 1]
			# Reshape to [T, R, H, B, 1]
			rewards = rewards_flat.permute(1, 0, 2, 3).view(rollout_len, R, H, B, 1)  # float32[T, R, H, B, 1]

			if self.cfg.episodic:
				term_logits_flat = self.model.termination(z_for_reward, task, unnormalized=True)
				term_logits = term_logits_flat.view(rollout_len, H, B, 1)  # float32[T, H, B, 1]
				terminated = (torch.sigmoid(term_logits) > 0.5).float()
			else:
				term_logits = torch.zeros(rollout_len, H, B, 1, device=device, dtype=dtype)
				terminated = torch.zeros(rollout_len, H, B, 1, device=device, dtype=dtype)

		# Avoid in-place detach on a view; build a fresh contiguous tensor
		with maybe_range('Imagined/final_pack', self.cfg):
			z_seq_out = torch.cat([z_seq[:1], z_seq[1:].detach()], dim=0).clone()  # float32[T+1, H, B, L]

		return {
			'z_seq': z_seq_out,  # float32[T+1, H, B, L]
			'actions': actions_seq.detach(),  # float32[T, 1, B, A] (shared across heads)
			'rewards': rewards.detach(),  # float32[T, R, H, B, 1]
			'terminated': terminated.detach(),  # float32[T, H, B, 1]
			'termination_logits': term_logits.detach(),  # float32[T, H, B, 1]
		}



	def calculate_value_loss(self, z_seq, actions, rewards, terminated, full_detach, task=None, z_td=None):
		"""Compute primary critic loss on arbitrary latent sequences with multi-head support.
		
		With V-function, we predict V(z) for each state in the sequence and
		train against TD targets r + γ * V(next_z). For multi-head inputs,
		TD targets are computed per-head then min'd for pessimism.

		Args:
			z_seq (Tensor[T+1, H, B, L]): Latent sequences with H heads.
			actions (Tensor[T, H_a, B, A]): Actions (H_a=1 when shared across heads).
			rewards (Tensor[T, H, B, 1]): Rewards per head.
			terminated (Tensor[T, H, B, 1]): Termination signals per head.
			full_detach (bool): Whether to detach z_seq from graph.
			task: Task index for multitask.
			z_td (Tensor[T, H, B, L], optional): Override next_z for TD target.

		Returns:
			Tuple[Tensor, TensorDict]: Scalar loss and info dict.
		"""
		if z_td is None:
			assert self.cfg.ac_source != "replay_rollout", "Need to supply z_td for ac_source=replay_rollout in calculate_value_loss"
		
		# z_seq: [T+1, H, B, L]; actions: [T, H_a, B, A] where H_a may be 1
		T_plus_1, H, B, L = z_seq.shape
		T = T_plus_1 - 1
		K = self.cfg.num_bins
		device = z_seq.device

		rho_pows = torch.pow(self.cfg.rho, torch.arange(T, device=device, dtype=z_seq.dtype))  # float32[T]

		z_seq = z_seq.detach() if full_detach else z_seq  # z_seq: float32[T+1, H, B, L]
		rewards = rewards.detach()  # float32[T, R, H, B, 1]
		terminated = terminated.detach()  # float32[T, H, B, 1]

		# V-function: input is state only, merge H into B
		# z_seq[:-1]: [T, H, B, L] -> [T, H*B, L]
		z_for_v = z_seq[:-1].view(T, H * B, L)  # float32[T, H*B, L]
		vs_flat = self.model.V(z_for_v, task, return_type='all')  # float32[Ve, T, H*B, K]
		Ve = vs_flat.shape[0]
		# Reshape back: [Ve, T, H*B, K] -> [Ve, T, H, B, K]
		vs = vs_flat.view(Ve, T, H, B, K)  # float32[Ve, T, H, B, K]

		with maybe_range('Value/td_target', self.cfg):
			with torch.no_grad():
				if z_td is not None:
					td_targets, td_std_across_heads = self._td_target(z_td, rewards, terminated, task)
				else:
					td_targets, td_std_across_heads = self._td_target(z_seq[1:], rewards, terminated, task)
				# td_targets: float32[T, B, 1], td_std_across_heads: float32[T, B, 1]

		with maybe_range('Value/ce', self.cfg):
			# TD targets are [T, B, 1] after min over heads
			# Expand to match vs: [Ve, T, H, B, 1] - same target for all heads
			td_expanded = td_targets.unsqueeze(0).unsqueeze(2).expand(Ve, T, H, B, 1)  # float32[Ve, T, H, B, 1]
			
			# Flatten for soft_ce: [Ve*T*H*B, K] and [Ve*T*H*B, 1]
			vs_flat_ce = vs.contiguous().view(Ve * T * H * B, K)
			td_flat = td_expanded.contiguous().view(Ve * T * H * B, 1)
			
			val_ce_flat = math.soft_ce(vs_flat_ce, td_flat, self.cfg)  # float32[Ve*T*H*B]
			val_ce = val_ce_flat.view(Ve, T, H, B, 1).mean(dim=(0, 2, 3)).squeeze(-1)  # float32[T]

		weighted = val_ce * rho_pows
		loss = weighted.mean()

		info = TensorDict({
			'value_loss': loss
		}, device=device, non_blocking=True)

		for t in range(T):
			info.update({f'value_loss/step{t}': val_ce[t]}, non_blocking=True)
		
		value_pred = math.two_hot_inv(vs, self.cfg)  # float32[Ve, T, H, B, 1]
		if self.log_detailed:
			info.update({
				'td_target_mean': td_targets.mean(),
				'td_target_std': td_targets.std(),
				'td_target_min': td_targets.min(),
				'td_target_max': td_targets.max(),
				'td_target_std_across_heads': td_std_across_heads.mean(),
				'value_pred_mean': value_pred.mean(),
				'value_pred_std': value_pred.std(),
				'value_pred_min': value_pred.min(),
				'value_pred_max': value_pred.max(),
			}, non_blocking=True)
   
		# Value error: expand td_targets to [Ve, T, H, B, 1] for comparison
		td_full = td_targets.unsqueeze(0).unsqueeze(2).expand(Ve, T, H, B, 1)  # float32[Ve, T, H, B, 1]
		value_error = value_pred - td_full  # float32[Ve, T, H, B, 1]
		# Average over heads for per-step logging
		value_error_avg = value_error.mean(dim=2)  # float32[Ve, T, B, 1]
		for i in range(T):
			info.update({f"value_error_abs_mean/step{i}": value_error_avg[:, i].abs().mean(),
						f"value_error_std/step{i}": value_error_avg[:, i].std(),
						f"value_error_max/step{i}": value_error_avg[:, i].abs().max()}, non_blocking=True)


		return loss, info

	def calculate_aux_value_loss(self, z_seq, actions, rewards, terminated, full_detach, task=None, z_td=None):
		"""Compute auxiliary multi-gamma critic losses with multi-head support.
		
		With V-function, we predict V_aux(z) for each state and auxiliary gamma.
		For multi-head inputs, TD targets are computed per-head then min'd for pessimism.

		Args:
			z_seq (Tensor[T+1, H, B, L]): Latent sequences with H dynamics heads.
			actions (Tensor[T, H_a, B, A]): Actions (H_a=1 when shared across heads).
			rewards (Tensor[T, R, H, B, 1]): Rewards with R reward heads, H dynamics heads.
			terminated (Tensor[T, H, B, 1]): Termination signals (no R dimension).
			full_detach (bool): Whether to detach z_seq from graph.
			task: Task index for multitask.
			z_td (Tensor[T, H, B, L], optional): Override next_z for TD target.

		Returns:
			Tuple[Tensor, TensorDict]: Scalar loss and info dict.
		"""
		if z_td is None:
			assert self.cfg.aux_value_source != "replay_rollout", "Need to supply z_td for ac_source=replay_rollout in calculate_value_loss"
		if self.model._num_aux_gamma == 0:
			return torch.zeros((), device=z_seq.device), TensorDict({}, device=z_seq.device)

		# z_seq: [T+1, H, B, L]
		T_plus_1, H, B, L = z_seq.shape
		T = T_plus_1 - 1
		K = self.cfg.num_bins
		device = z_seq.device

		rho_pows = torch.pow(self.cfg.rho, torch.arange(T, device=device, dtype=z_seq.dtype))  # float32[T]

		z_seq = z_seq.detach() if full_detach else z_seq  # z_seq: float32[T+1, H, B, L]
		rewards = rewards.detach()  # float32[T, R, H, B, 1]
		terminated = terminated.detach()  # float32[T, H, B, 1]

		# V_aux: input is state only, merge H into B
		# z_seq[:-1]: [T, H, B, L] -> [T, H*B, L]
		z_for_v = z_seq[:-1].view(T, H * B, L)  # float32[T, H*B, L]
		v_aux_logits_flat = self.model.V_aux(z_for_v, task, return_type='all')  # float32[G_aux, T, H*B, K] or None
		if v_aux_logits_flat is None:
			return torch.zeros((), device=device), TensorDict({}, device=device)

		G_aux = v_aux_logits_flat.shape[0]
		# Reshape back: [G_aux, T, H*B, K] -> [G_aux, T, H, B, K]
		v_aux_logits = v_aux_logits_flat.view(G_aux, T, H, B, K)  # float32[G_aux, T, H, B, K]

		with maybe_range('Aux/td_target', self.cfg):
			with torch.no_grad():
				if z_td is not None:
					aux_td_targets = self._td_target_aux(z_td, rewards, terminated, task)  # float32[G_aux, T, B, 1]
				else:
					aux_td_targets = self._td_target_aux(z_seq[1:], rewards, terminated, task)  # float32[G_aux, T, B, 1]

		with maybe_range('Aux/ce', self.cfg):
			# TD targets are [G_aux, T, B, 1] after min over heads
			# Expand to match v_aux_logits: [G_aux, T, H, B, 1] - same target for all heads
			td_expanded = aux_td_targets.unsqueeze(2).expand(G_aux, T, H, B, 1)  # float32[G_aux, T, H, B, 1]
			
			# Flatten for soft_ce: [G_aux*T*H*B, K] and [G_aux*T*H*B, 1]
			vaux_flat = v_aux_logits.contiguous().view(G_aux * T * H * B, K)
			aux_targets_flat = td_expanded.contiguous().view(G_aux * T * H * B, 1)
			
			aux_ce_flat = math.soft_ce(vaux_flat, aux_targets_flat, self.cfg)  # float32[G_aux*T*H*B]
			aux_ce = aux_ce_flat.view(G_aux, T, H, B, 1).mean(dim=(2, 3)).squeeze(-1)  # float32[G_aux, T]

		weighted = aux_ce * rho_pows.unsqueeze(0)  # float32[G_aux, T]
		losses = weighted.mean(dim=1)  # float32[G_aux]
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
			"""Fetch source data for value loss computation with unified [T, H, B, ...] format.
			
			Non-imagined sources (replay_true, replay_rollout) have H=1 and R=1 added via unsqueeze.
			Rewards use format [T, R, H, B, 1] where R=num_reward_heads, H=num_dynamics_heads.
			The 'imagine' source uses native multi-head output from imagined_rollout.
			"""
			if name in source_cache:
				return source_cache[name]
			if name == 'replay_true':
				# Add H=1 dimension: [T, B, ...] -> [T, 1, B, ...]
				# Add R=1 dimension for rewards: [T, B, 1] -> [T, 1, 1, B, 1]
				src = {
					'z_seq': z_true.unsqueeze(1),  # float32[T+1, 1, B, L]
					'actions': action.unsqueeze(1),  # float32[T, 1, B, A]
					'rewards': reward.unsqueeze(1).unsqueeze(1),  # float32[T, 1, 1, B, 1] (R=1, H=1)
					'terminated': terminated.unsqueeze(1),  # float32[T, 1, B, 1]
					'full_detach': False,
					'z_td': None,
				}
			elif name == 'replay_rollout':
				# Add H=1 dimension: [T, B, ...] -> [T, 1, B, ...]
				# Add R=1 dimension for rewards: [T, B, 1] -> [T, 1, 1, B, 1]
				src = {
					'z_seq': z_rollout.unsqueeze(1),  # float32[T+1, 1, B, L]
					'actions': action.unsqueeze(1),  # float32[T, 1, B, A]
					'rewards': reward.unsqueeze(1).unsqueeze(1),  # float32[T, 1, 1, B, 1] (R=1, H=1)
					'terminated': terminated.unsqueeze(1),  # float32[T, 1, B, 1]
					'full_detach': False,
					'z_td': z_true[1:].unsqueeze(1),  # float32[T, 1, B, L]
				}
			elif name == 'imagine':
				start_z = z_true.detach() if ac_only else z_true
				imagined = self.imagined_rollout(start_z, task=task, rollout_len=self.cfg.imagination_horizon)
				# imagined_rollout already returns [T, R, H, B, 1] format for rewards
				src = {
					'z_seq': imagined['z_seq'],  # float32[T+1, H, B, L]
					'actions': imagined['actions'],  # float32[T, 1, B, A] (shared across heads)
					'rewards': imagined['rewards'],  # float32[T, R, H, B, 1]
					'terminated': imagined['terminated'],  # float32[T, H, B, 1]
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
				# value_inputs['z_seq'] is [T_imag+1, H, B_expanded, L] where:
				#   T_imag = imagination_horizon (e.g., 1)
				#   B_expanded = S * batch_size * num_rollouts (e.g., 2 * 256 * 4 = 2048)
				# 
				# Problem: Due to num_rollouts, the initial states (t=0) are duplicated
				# num_rollouts times. Training on these duplicates is wasteful.
				# Solution: Skip t=0 and train only on rolled-out states (t=1+).
				# Trade-off: Policy never sees true encoded states, only 1-step imagined ones.
				z_for_pi = value_inputs['z_seq'][1:, 0].detach()  # float32[T_imag, B_expanded, L]
			elif self.cfg.actor_source == 'replay_rollout':
				# z_rollout is [T+1, B, L] (no H dimension from world_model_losses)
				z_for_pi = z_rollout.detach()
			elif self.cfg.actor_source == 'replay_true':
				# z_true is [T+1, B, L] (no H dimension from encoder)
				z_for_pi = z_true.detach()
    
			pi_loss, pi_info = self.update_pi(z_for_pi, task)
			pi_total = pi_loss * self.cfg.policy_coef
			if log_grads:
				info = self.probe_pi_gradients(pi_total, info)
			pi_total.backward()
			pi_grad_norm = torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm)
			if self.cfg.compile and log_grads:
				self.pi_optim.step()
			else:
				self.pi_optim_step()
			self.pi_optim.zero_grad(set_to_none=True)

			self.model.soft_update_target_V()
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

	@torch._dynamo.disable()
	def probe_pi_gradients(self, pi_total, info):
		"""Probe gradient norms from policy loss to all parameter groups.

		This reveals SVG-style gradient flow: policy loss backprops through
		reward/dynamics models even though only policy params are updated.

		Args:
			pi_total (Tensor): Scaled policy loss (pi_loss * policy_coef).
			info (TensorDict): Info dict to update with gradient norms.

		Returns:
			TensorDict: Updated info dict with grad_norm/pi_loss/{group} entries.
		"""
		groups = self._grad_param_groups()

		# Flatten all params (including policy) and remember mapping
		flat_params = []
		index = []  # (group_name, param_obj) pairs
		for gname, params in groups.items():
			for p in params:
				flat_params.append(p)
				index.append((gname, p))

		if (not torch.is_tensor(pi_total)) or (not pi_total.requires_grad):
			return info

		grads = torch.autograd.grad(
			pi_total,
			flat_params,
			retain_graph=True,  # we'll still do pi_total.backward() after
			create_graph=False,
			allow_unused=True,
		)

		# Accumulate L2 norm per component group
		per_group_ss = {}
		for (gname, p), g in zip(index, grads):
			if g is None:
				continue
			ss = (g.detach().float() ** 2).sum()
			per_group_ss[gname] = per_group_ss.get(gname, 0.0) + ss.item()

		for gname, ss in per_group_ss.items():
			info.update({f'grad_norm/pi_loss/{gname}': (ss ** 0.5)}, non_blocking=True)

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
						# Same logic as training: skip t=0 to avoid duplicated initial states,
						# and index H=0 to remove dynamics head dimension.
						# z_seq shape: [T_imag+1, H, B_expanded, L] -> [T_imag, B_expanded, L]
						z_for_pi = components['value_inputs']['z_seq'][1:, 0].detach()
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
