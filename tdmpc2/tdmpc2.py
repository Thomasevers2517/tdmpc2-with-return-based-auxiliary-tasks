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
		# ------------------------------------------------------------------
		# Ensemble-aware LR scaling:
		# When using ensembles with mean-reduced losses, each head sees 1/N of
		# the gradient it would see with a single head. To maintain the same
		# per-head learning dynamics as original TDMPC2 (which had 1 dynamics,
		# 1 reward, 5 value heads), we scale LRs by ensemble size.
		#
		# Formula per component:
		#   - Dynamics: lr * num_dynamics_heads (baseline had 1)
		#   - Reward:   lr * num_reward_heads   (baseline had 1)
		#   - Value:    lr * num_q / 5          (baseline had 5)
		#   - Aux value: lr * num_aux_heads / 5 (to match original V head gradient)
		# ------------------------------------------------------------------
		num_dynamics_heads = int(getattr(self.cfg, 'planner_num_dynamics_heads', 1))
		num_reward_heads = int(getattr(self.cfg, 'num_reward_heads', 1))
		num_q = int(getattr(self.cfg, 'num_q', 5))
		num_aux_heads = len(self.cfg.multi_gamma_gammas) if getattr(self.cfg, 'multi_gamma_gammas', None) else 0
		ensemble_lr_scaling = self.cfg.ensemble_lr_scaling
		
		lr_encoder = self.cfg.lr * self.cfg.enc_lr_scale
		if ensemble_lr_scaling:
			# Scale LRs by ensemble size to compensate for mean-reduced gradients
			lr_dynamics = self.cfg.lr * num_dynamics_heads
			lr_reward = self.cfg.lr * num_reward_heads
			lr_value = self.cfg.lr * num_q / 5
			lr_aux_value = self.cfg.lr * num_aux_heads / 5 if num_aux_heads > 0 else self.cfg.lr
		else:
			# No scaling: all heads use base LR
			lr_dynamics = self.cfg.lr
			lr_reward = self.cfg.lr
			lr_value = self.cfg.lr
			lr_aux_value = self.cfg.lr
		
		param_groups = [
			{'params': self.model._encoder.parameters(), 'lr': lr_encoder},
			{'params': self.model._dynamics_heads.parameters(), 'lr': lr_dynamics},
			{'params': self.model._reward_heads.parameters(), 'lr': lr_reward},
			{'params': self.model._termination.parameters() if self.cfg.episodic else [], 'lr': self.cfg.lr},
			{'params': self.model._Vs.parameters(), 'lr': lr_value},
			{'params': self.model._task_emb.parameters() if self.cfg.multitask else [], 'lr': self.cfg.lr}
		]
		if num_aux_heads > 0:
			# Append auxiliary head params (joint or separate)
			if getattr(self.model, '_aux_joint_Vs', None) is not None:
				param_groups.append({'params': self.model._aux_joint_Vs.parameters(), 'lr': lr_aux_value})
			elif getattr(self.model, '_aux_separate_Vs', None) is not None:
				for head in self.model._aux_separate_Vs:
					param_groups.append({'params': head.parameters(), 'lr': lr_aux_value})
		
		# Log effective learning rates
		log.info('Effective learning rates (ensemble_lr_scaling=%s):', ensemble_lr_scaling)
		log.info('  encoder:    %.6f (base_lr * enc_lr_scale = %.4f * %.2f)', lr_encoder, self.cfg.lr, self.cfg.enc_lr_scale)
		if ensemble_lr_scaling:
			log.info('  dynamics:   %.6f (base_lr * %d dynamics heads)', lr_dynamics, num_dynamics_heads)
			log.info('  reward:     %.6f (base_lr * %d reward heads)', lr_reward, num_reward_heads)
			log.info('  value:      %.6f (base_lr * %d / 5 value heads)', lr_value, num_q)
			if num_aux_heads > 0:
				log.info('  aux_value:  %.6f (base_lr * %d / 5 aux heads)', lr_aux_value, num_aux_heads)
		else:
			log.info('  dynamics:   %.6f (base_lr, no scaling)', lr_dynamics)
			log.info('  reward:     %.6f (base_lr, no scaling)', lr_reward)
			log.info('  value:      %.6f (base_lr, no scaling)', lr_value)
			if num_aux_heads > 0:
				log.info('  aux_value:  %.6f (base_lr, no scaling)', lr_aux_value)
		
		self.optim = torch.optim.Adam(param_groups, lr=self.cfg.lr, capturable=True)
		lr_pi = self.cfg.lr * getattr(self.cfg, 'pi_lr_scale', 1.0)
		log.info('  policy:     %.6f (base_lr * pi_lr_scale = %.4f * %.2f)', lr_pi, self.cfg.lr, getattr(self.cfg, 'pi_lr_scale', 1.0))
		# Dual policy: combine both policies' params into single optimizer
		if self.cfg.dual_policy_enabled:
			pi_params = list(self.model._pi.parameters()) + list(self.model._pi_optimistic.parameters())
			log.info('  dual_policy: enabled (pessimistic + optimistic)')
		else:
			pi_params = self.model._pi.parameters()
		self.pi_optim = torch.optim.Adam(pi_params, lr=lr_pi, eps=1e-5, capturable=True)

		# Store initial encoder LR for step-change schedule
		self._enc_lr_initial = lr_encoder
		self._enc_lr_stepped = False  # Track if step-change has been applied

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
			# Separately compile regression policy loss (uses different CUDAGraph)
			self._compute_regression_pi_loss_eager = self._compute_regression_pi_loss
			self._compute_regression_pi_loss = torch.compile(self._compute_regression_pi_loss, mode=self.cfg.compile_type, fullgraph=False)

			@torch.compile(mode=self.cfg.compile_type, fullgraph=False )
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
			self._compute_regression_pi_loss_eager = self._compute_regression_pi_loss
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

	def calc_pi_losses(self, z, task, optimistic=False):
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
			optimistic: If True, use optimistic policy with max reduction and
				scaled entropy. If False, use pessimistic policy with configured
				reduction (default min).
			
		Returns:
			Tuple[Tensor, TensorDict]: Policy loss and info dict.
		"""
		T, B, L = z.shape
		
		# Ensure contiguity for torch.compile compatibility
		z = z.contiguous()  # Required: z may be non-contiguous after detach/indexing ops
		
		# Select reduction mode and entropy coefficient based on optimistic flag
		if optimistic:
			policy_reduce = self.cfg.optimistic_head_reduce  # 'max' or 'mean' for optimistic
			entropy_coeff = self.dynamic_entropy_coeff * self.cfg.optimistic_entropy_mult
			lambda_value_disagreement = self.cfg.optimistic_policy_lambda_value_disagreement
		else:
			policy_reduce = self.cfg.policy_head_reduce  # 'mean', 'min', or 'max'
			entropy_coeff = self.dynamic_entropy_coeff
			lambda_value_disagreement = self.cfg.policy_lambda_value_disagreement
			
		with maybe_range('Agent/update_pi', self.cfg):
			# Sample action from policy at current state (policy has gradients)
			action, info = self.model.pi(z, task, optimistic=optimistic)  # action: float32[T, B, A]
			
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
				reward_flat = torch.amin(reward_all, dim=0)  # float32[T*B, 1]
			elif policy_reduce == 'max':
				reward_flat = torch.amax(reward_all, dim=0)  # float32[T*B, 1]
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
				v_next_flat = torch.amin(v_next_all, dim=0)  # float32[T*B, 1]
			elif policy_reduce == 'max':
				v_next_flat = torch.amax(v_next_all, dim=0)  # float32[T*B, 1]
			else:
				raise ValueError(f"Invalid policy_head_reduce '{policy_reduce}'. Expected 'mean', 'min', or 'max'.")
			
			# Compute value disagreement across dynamics heads (std of V estimates)
			# Scale by self.scale.value for unit consistency with q_scaled
			v_disagreement_flat = v_next_all.std(dim=0)  # float32[T*B, 1]
			v_disagreement_scaled_flat = v_disagreement_flat / self.scale.value.clamp(min=1e-6)  # float32[T*B, 1]
			v_disagreement_scaled = v_disagreement_scaled_flat.view(T, B, 1)  # float32[T, B, 1]
			
			# Compute Q-like estimate: r(z, a) + γ * V(z')
			# This is what the action "earns" - immediate reward plus discounted future value
			q_estimate_flat = reward_flat + gamma * v_next_flat  # float32[T*B, 1]
			q_estimate = q_estimate_flat.view(T, B, 1)           # float32[T, B, 1]
			
			# Update scale with the Q-estimate (first timestep batch for stability)
			# NOTE: Only update scale for pessimistic policy to avoid inplace op conflict
			# when computing both policies in a single backward pass.
			if not optimistic:
				self.scale.update(q_estimate[0])
			q_scaled = self.scale(q_estimate)  # float32[T, B, 1]
			
			# Temporal weighting
			rho_pows = torch.pow(self.cfg.rho, torch.arange(T, device=self.device))  # float32[T]
			
			# Entropy from policy (always use scaled_entropy with configurable action_dim power)
			entropy_term = info["scaled_entropy"]  # float32[T, B, 1]
			
			# Policy loss: maximize (q_scaled + entropy_coeff * entropy - λ * v_disagreement)
			# Subtraction: positive λ penalizes uncertainty, negative λ rewards it
			# Apply rho weighting across time
			objective = q_scaled + entropy_coeff * entropy_term - lambda_value_disagreement * v_disagreement_scaled  # float32[T, B, 1]
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
				"pi_entropy_multiplier": info["entropy_multiplier"],
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
				"entropy_coeff_effective": entropy_coeff,
				"pi_q_estimate_mean": q_estimate.mean(),
				"pi_reward_mean": reward_flat.mean(),
				"pi_v_next_mean": v_next_flat.mean(),
				"pi_value_disagreement": v_disagreement_flat.mean(),
				"pi_value_disagreement_scaled": v_disagreement_scaled.mean(),
			}, device=self.device)

			return pi_loss, info

	@torch.no_grad()
	def compute_regression_weights(
		self,
		td_targets,
		v_next,
		batch_structure,
		update_scale=True,
	):
		"""Compute softmax(Q/τ) weights for both pessimistic and optimistic policies.
		
		This method computes action weights based on Q-estimates from TD targets.
		Called once in _compute_loss_components, weights are then passed to both
		value loss (pessimistic only) and policy losses (pessimistic + optimistic).
		
		Args:
			td_targets (Tensor[T, R, H, Ve, B, 1]): Raw TD targets from compute_imagination_td_targets.
			v_next (Tensor[T, H, Ve, B, 1]): V(next_z) for value disagreement computation.
			batch_structure (dict): Contains S, B_orig, N, B for proper softmax over samples.
			update_scale (bool): If True, update RunningScale with pessimistic Q values.
			
		Returns:
			Tuple[Tensor, Tensor, TensorDict]: (pessimistic_weights, optimistic_weights, info_dict)
				Each weights tensor has shape [T, B, 1].
		"""
		T = td_targets.shape[0]
		B = batch_structure['B']
		S = batch_structure['S']
		B_orig = batch_structure['B_orig']
		N = batch_structure['N']
		device = td_targets.device
		dtype = td_targets.dtype
		
		temperature = self.cfg.pi_regression_temperature
		
		# Flatten R, H, Ve dimensions for unified reduction: [T, R*H*Ve, B, 1]
		td_flat = td_targets.view(T, -1, B, 1)  # float32[T, R*H*Ve, B, 1]
		
		# Pessimistic Q: min over all heads
		pess_reduce_mode = self.cfg.policy_head_reduce  # 'min' or 'mean'
		pess_reduce_fn = torch.amin if pess_reduce_mode == 'min' else torch.mean
		Q_pess = pess_reduce_fn(td_flat, dim=1)  # float32[T, B, 1]
		
		# Optimistic Q: max over all heads
		Q_opti = torch.amax(td_flat, dim=1)  # float32[T, B, 1]
		
		# Update RunningScale with pessimistic Q values (before disagreement)
		if update_scale:
			self.scale.update(Q_pess[0])
		
		# Value disagreement across all heads (H * Ve)
		# v_next: [T, H, Ve, B, 1] -> flatten to [T, H*Ve, B, 1]
		v_flat = v_next.view(T, -1, B, 1)  # float32[T, H*Ve, B, 1]
		v_disagreement = v_flat.std(dim=1)  # float32[T, B, 1]
		
		# Apply disagreement penalty (pessimistic) and bonus (optimistic)
		lambda_pess = self.cfg.policy_lambda_value_disagreement
		lambda_opti = self.cfg.optimistic_policy_lambda_value_disagreement
		Q_pess = Q_pess - lambda_pess * v_disagreement
		Q_opti = Q_opti + lambda_opti * v_disagreement
		
		# Scale Q-estimates for softmax
		Q_pess_scaled = self.scale(Q_pess)  # float32[T, B, 1]
		Q_opti_scaled = self.scale(Q_opti)  # float32[T, B, 1]
		
		# Reshape to expose N dimension for proper AWR softmax over sampled actions
		# B = S * B_orig * N, reshape to [T, S*B_orig, N, 1]
		Q_pess_for_softmax = Q_pess_scaled.view(T, S * B_orig, N, 1)  # float32[T, S*B_orig, N, 1]
		Q_opti_for_softmax = Q_opti_scaled.view(T, S * B_orig, N, 1)  # float32[T, S*B_orig, N, 1]
		
		# Softmax over N (the sampled actions per state)
		weights_pess_per_state = torch.softmax(Q_pess_for_softmax / temperature, dim=2)  # float32[T, S*B_orig, N, 1]
		weights_opti_per_state = torch.softmax(Q_opti_for_softmax / temperature, dim=2)  # float32[T, S*B_orig, N, 1]
		
		# Flatten back to [T, B, 1]
		weights_pess = weights_pess_per_state.view(T, B, 1)  # float32[T, B, 1]
		weights_opti = weights_opti_per_state.view(T, B, 1)  # float32[T, B, 1]
		
		# Compute effective number of samples: 1 / sum(w²) per state
		# Higher = more uniform, lower = more peaked
		weights_pess_sq = weights_pess_per_state.pow(2).sum(dim=2)  # [T, S*B_orig, 1]
		eff_samples_pess = (1.0 / (weights_pess_sq + 1e-8)).mean()  # scalar
		weights_opti_sq = weights_opti_per_state.pow(2).sum(dim=2)  # [T, S*B_orig, 1]
		eff_samples_opti = (1.0 / (weights_opti_sq + 1e-8)).mean()  # scalar
		
		# Build info dict for logging
		info = TensorDict({
			'regression_weights_pess_max': weights_pess.max(),
			'regression_weights_pess_min': weights_pess.min(),
			'regression_weights_opti_max': weights_opti.max(),
			'regression_weights_opti_min': weights_opti.min(),
			'regression_eff_samples_pess': eff_samples_pess,
			'regression_eff_samples_opti': eff_samples_opti,
			'regression_q_pess_scaled_mean': Q_pess_scaled.mean(),
			'regression_q_opti_scaled_mean': Q_opti_scaled.mean(),
			'regression_q_pess_scaled_std': Q_pess_scaled.std(),
			'regression_q_opti_scaled_std': Q_opti_scaled.std(),
			'regression_v_disagreement_mean': v_disagreement.mean(),
			'regression_temperature': float(temperature),
		}, device=device)
		
		return weights_pess, weights_opti, info

	def calculate_regression_pi_loss(
		self,
		z,
		actions,
		batch_structure,
		task,
		weights,
		optimistic=False,
	):
		"""Compute AWR-style policy loss using pre-computed softmax weights.
		
		Uses pre-computed action weights from compute_regression_weights() to train
		policy via weighted NLL + entropy bonus.

		Args:
			z (Tensor[T, B, L]): Latent states at t=0 of imagination (DETACHED).
			actions (Tensor[T, B, A]): Sampled actions (from pessimistic policy, DETACHED).
			batch_structure (dict): Contains S, B_orig, N, B for proper softmax over samples.
			task: Optional task identifier.
			weights (Tensor[T, B, 1]): Pre-computed softmax weights from compute_regression_weights.
			optimistic (bool): If True, use optimistic policy and entropy scaling;
				if False, use pessimistic policy.

		Returns:
			Tuple[Tensor, TensorDict]: Policy loss scalar and info dict with logging metrics.
		"""
		T, B, L = z.shape  # z: float32[T, B, L]
		A = actions.shape[-1]  # actions: float32[T, B, A]
		device = z.device
		dtype = z.dtype
		
		# Extract batch structure for proper softmax over N samples
		S = batch_structure['S']
		B_orig = batch_structure['B_orig']
		N = batch_structure['N']
		assert B == S * B_orig * N, f"Batch mismatch: B={B}, S*B_orig*N={S*B_orig*N}"

		# Temporal weighting
		rho_pows = torch.pow(self.cfg.rho, torch.arange(T, device=device, dtype=dtype))  # float32[T]

		# Select entropy coefficient based on optimistic flag
		if optimistic:
			entropy_coeff = self.dynamic_entropy_coeff * self.cfg.optimistic_entropy_mult
		else:
			entropy_coeff = self.dynamic_entropy_coeff

		# Get log_probs and entropy WITH GRADIENTS
		# Must recompute for both policies using DETACHED z to avoid backprop through encoder.
		# The log_probs from imagination are attached to encoder graph, which was freed after
		# total_loss.backward(). Recomputing ensures gradients only flow to policy params.
		z_flat = z.detach().view(T * B, L)  # float32[T*B, L] - DETACHED from encoder
		actions_flat = actions.view(T * B, A)  # float32[T*B, A]
		
		_, pi_info = self.model.pi(z_flat, task, optimistic=optimistic)
		mu = pi_info['presquash_mean']  # float32[T*B, A]
		log_std = pi_info['log_std']  # float32[T*B, A]

		# Compute log probability for the sampled actions
		log_probs = math.compute_action_log_prob(actions_flat, mu, log_std)  # float32[T*B, 1]
		log_probs = log_probs.view(T, B, 1)  # float32[T, B, 1]

		# Use scaled_entropy from pi_info
		scaled_entropy_flat = pi_info['scaled_entropy']  # float32[T*B, 1]
		scaled_entropy = scaled_entropy_flat.view(T, B, 1)  # float32[T, B, 1]

		# Weighted negative log-likelihood per state
		# For each of the S*B_orig states, we have N samples with weights summing to 1.
		# (weights * log_probs).sum(dim=1) sums across all B = S*B_orig*N samples.
		# Since weights sum to 1 per state, this gives S*B_orig weighted averages summed together.
		# Divide by S*B_orig to get the mean weighted log-prob across states.
		num_states = S * B_orig
		weighted_nll = -(weights * log_probs).sum(dim=1, keepdim=True) / num_states  # float32[T, 1, 1]
		weighted_nll = weighted_nll.squeeze(-1)  # float32[T, 1]

		# Entropy bonus (mean over all B samples, which is equivalent to mean over states)
		entropy_mean = scaled_entropy.mean(dim=1, keepdim=True).squeeze(-1)  # float32[T, 1]

		# Per-timestep loss
		loss_t = weighted_nll - entropy_coeff * entropy_mean  # float32[T, 1]

		# Temporal-weighted mean
		loss_per_t = loss_t.squeeze(-1)  # float32[T]
		loss = (loss_per_t * rho_pows).sum() / rho_pows.sum()

		# Build info dict for logging
		prefix = "opti_regression_" if optimistic else "regression_"
		info = TensorDict({
			f"{prefix}loss": loss,
			f"{prefix}weight_entropy": -(weights * (weights + 1e-8).log()).sum(dim=1).mean(),
			f"{prefix}log_prob_mean": log_probs.mean(),
			f"{prefix}entropy_mean": entropy_mean.mean(),
			f"{prefix}weights_max": weights.max(),
			f"{prefix}weights_min": weights.min(),
		}, device=device)

		return loss, info

	def _compute_regression_pi_loss(self, z, actions, weights_pess, weights_opti, S, B_orig, N, B, task):
		"""Compute regression policy loss in a separately compiled function.
		
		This method wraps calculate_regression_pi_loss and is compiled separately from
		_compute_loss_components to avoid CUDAGraph memory conflicts. All inputs must
		be DETACHED tensors with no references to the encoder/dynamics graph.
		
		Args:
			z (Tensor[T, B, L]): Latent states, DETACHED.
			actions (Tensor[T, B, A]): Sampled actions, DETACHED.
			weights_pess (Tensor[T, B, 1]): Pessimistic softmax weights, DETACHED.
			weights_opti (Tensor[T, B, 1] or None): Optimistic weights if dual_policy, DETACHED.
			S (int): Number of starting states.
			B_orig (int): Original batch size.
			N (int): Number of rollouts per state.
			B (int): Total batch size = S * B_orig * N.
			task: Task identifier.
			
		Returns:
			Tuple[Tensor, TensorDict]: Total policy loss and combined info dict.
		"""
		batch_structure = {'S': S, 'B_orig': B_orig, 'N': N, 'B': B}
		
		# Pessimistic policy loss
		pess_loss, pess_info = self.calculate_regression_pi_loss(
			z, actions, batch_structure, task, weights_pess, optimistic=False
		)
		pi_loss = pess_loss
		pi_info = pess_info
		
		# Optimistic policy loss (if dual policy enabled)
		if self.cfg.dual_policy_enabled:
			opt_loss, opt_info = self.calculate_regression_pi_loss(
				z, actions, batch_structure, task, weights_opti, optimistic=True
			)
			pi_info.update(opt_info, non_blocking=True)
			pi_loss = pi_loss + opt_loss
		
		return pi_loss, pi_info

	def update_pi(self, zs, task):
		"""
		Update policy using a sequence of latent states.
		
		If dual_policy_enabled, computes losses for both pessimistic and optimistic
		policies and sums them with equal weight.

		Args:
			zs (torch.Tensor): Sequence of latent states.
			task (torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			Tuple[float, TensorDict]: Total policy loss and info dict.
		"""
		log_grads = self.cfg.log_gradients_per_loss and self.log_detailed
		
		# Pessimistic policy loss
		pi_loss, info = self.calc_pi_losses(zs, task, optimistic=False) if (not log_grads or not self.cfg.compile) else self.calc_pi_losses_eager(zs, task)
		
		# Optimistic policy loss (if dual policy enabled)
		if self.cfg.dual_policy_enabled:
			opti_pi_loss, opti_info = self.calc_pi_losses(zs, task, optimistic=True)
			
			# Prefix optimistic info keys with 'opti_'
			for k, v in opti_info.items():
				info[f'opti_{k}'] = v
			
			# Sum losses with equal weight
			pi_loss = pi_loss + opti_pi_loss
		
		return pi_loss, info

	@torch.no_grad()
	def _td_target(self, next_z, reward, terminated, task):
		"""
		Compute TD-target from a reward and the observation at the following time step.
		
		With V-function, the TD target is: r + γ * (1 - terminated) * V(next_z)
		
		Reduction behavior controlled by cfg.td_bootstrap_mode:
		  - 'min': min over Ve (value heads) and H (dynamics heads), then expand to Ve
		  - 'mean': mean over Ve and H, then expand to Ve
		  - 'local': each Ve head bootstraps itself, H reduced by cfg.local_td_target_dynamics_reduction

		Args:
			next_z (Tensor[T, H, B, L]): Latent states at following time steps with H dynamics heads.
			reward (Tensor[T, R, H, B, 1]): Rewards with R reward heads, H dynamics heads.
			terminated (Tensor[T, H, B, 1]): Termination signals (no R dimension).
			task (torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			Tuple[Tensor[Ve, T, B, 1], Tensor[T, B, 1]]: (TD-targets per Ve head, std across dynamics heads).
		"""
		T, H, B, L = next_z.shape  # next_z: float32[T, H, B, L]
		R = reward.shape[1]  # reward: float32[T, R, H, B, 1]
		
		# Merge H and B for V evaluation: [T, H, B, L] -> [T, H*B, L]
		next_z_flat = next_z.view(T, H * B, L)  # float32[T, H*B, L]
		
		# Get V predictions for all ensemble heads: [Ve, T, H*B, K]
		v_logits_flat = self.model.V(next_z_flat, task, return_type='all', target=True)  # float32[Ve, T, H*B, K]
		Ve = v_logits_flat.shape[0]
		
		# Convert to scalar values: [Ve, T, H*B, 1]
		v_values_flat = math.two_hot_inv(v_logits_flat, self.cfg)  # float32[Ve, T, H*B, 1]
		
		# Reshape back to [Ve, T, H, B, 1]
		v_values = v_values_flat.view(Ve, T, H, B, 1)  # float32[Ve, T, H, B, 1]
		
		discount = self.discount[task].unsqueeze(-1) if self.cfg.multitask else self.discount
		
		# Pessimistically reduce over reward heads: min over R (dim=1)
		reward_pessimistic = torch.amin(reward, dim=1)  # float32[T, H, B, 1]
		
		# Expand reward/terminated for broadcasting with Ve: [T, H, B, 1] -> [1, T, H, B, 1]
		reward_exp = reward_pessimistic.unsqueeze(0)  # float32[1, T, H, B, 1]
		terminated_exp = terminated.unsqueeze(0)  # float32[1, T, H, B, 1]
		
		# Per-head TD targets: r + γ * (1 - done) * V(s')  ->  [Ve, T, H, B, 1]
		td_per_head = reward_exp + discount * (1 - terminated_exp) * v_values  # float32[Ve, T, H, B, 1]
		
		# Compute std across dynamics heads (for logging), using mean over Ve
		td_mean_over_ve = td_per_head.mean(dim=0)  # float32[T, H, B, 1]
		td_std_across_heads = td_mean_over_ve.std(dim=1, unbiased=False)  # float32[T, B, 1]
		
		# Apply reduction based on td_bootstrap_mode - always reduce H, output is [Ve, T, B, 1]
		mode = self.cfg.td_bootstrap_mode
		if mode == 'min':
			# Min over Ve (dim=0) and H (dim=2), then expand to [Ve, T, B, 1]
			td_reduced = torch.amin(td_per_head, dim=(0, 2), keepdim=False)  # float32[T, B, 1]
			td_targets = td_reduced.unsqueeze(0).expand(Ve, T, B, 1)  # float32[Ve, T, B, 1]
		elif mode == 'mean':
			# Mean over Ve (dim=0) and H (dim=2), then expand to [Ve, T, B, 1]
			td_reduced = td_per_head.mean(dim=(0, 2), keepdim=False)  # float32[T, B, 1]
			td_targets = td_reduced.unsqueeze(0).expand(Ve, T, B, 1)  # float32[Ve, T, B, 1]
		elif mode == 'local':
			# Each Ve head bootstraps itself; reduce H by local_td_target_dynamics_reduction
			h_reduction = self.cfg.local_td_target_dynamics_reduction
			if h_reduction == 'single':
				# H=1 when using single random head, no reduction needed
				td_targets = td_per_head.squeeze(2)  # float32[Ve, T, B, 1]
			elif h_reduction == 'min':
				td_targets = torch.amin(td_per_head, dim=2)  # float32[Ve, T, B, 1]
			else:  # 'mean'
				td_targets = td_per_head.mean(dim=2)  # float32[Ve, T, B, 1]
		else:
			raise ValueError(f"Unknown td_bootstrap_mode: {mode}. Expected 'min', 'mean', or 'local'.")
		
		return td_targets, td_std_across_heads 
  

	@torch.no_grad()
	def _td_target_aux(self, next_z, reward, terminated, task):
		"""
		Compute auxiliary multi-gamma TD targets.
		
		Auxiliary values have no Ve ensemble, so td_bootstrap_mode 'min'/'mean'/'local'
		are equivalent for Ve. Dynamics head (H) reduction uses local_td_target_dynamics_reduction.

		Args:
			next_z (Tensor[T, H, B, L]): Latent states at following time steps with H dynamics heads.
			reward (Tensor[T, R, H, B, 1]): Rewards with R reward heads, H dynamics heads.
			terminated (Tensor[T, H, B, 1]): Termination signals (no R dimension).
			task (torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			Tensor[G_aux, T, B, 1]: TD targets per auxiliary gamma (H dimension reduced).
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
		
		# Pessimistically reduce over reward heads: min over R (dim=1)
		reward_pessimistic = torch.amin(reward, dim=1)  # float32[T, H, B, 1]
		
		# Expand reward/terminated for broadcasting: [T, H, B, 1] -> [1, T, H, B, 1]
		reward_exp = reward_pessimistic.unsqueeze(0)  # float32[1, T, H, B, 1]
		terminated_exp = terminated.unsqueeze(0)  # float32[1, T, H, B, 1]
		
		# Compute per-head TD targets for all auxiliary gammas at once
		# gamma_aux: scalar per g, broadcast over [G_aux, T, H, B, 1]
		gammas_aux = torch.tensor(self._all_gammas[1:], device=next_z.device, dtype=next_z.dtype)  # float32[G_aux]
		gammas_aux = gammas_aux.view(G_aux, 1, 1, 1, 1)  # float32[G_aux, 1, 1, 1, 1]
		
		td_per_head = reward_exp + gammas_aux * discount * (1 - terminated_exp) * v_values  # float32[G_aux, T, H, B, 1]
		
		# Reduce H dimension using local_td_target_dynamics_reduction - output is [G_aux, T, B, 1]
		h_reduction = self.cfg.local_td_target_dynamics_reduction
		if h_reduction == 'single':
			# H=1 when using single random head, no reduction needed
			td_targets_aux = td_per_head.squeeze(2)  # float32[G_aux, T, B, 1]
		elif h_reduction == 'min':
			td_targets_aux = torch.amin(td_per_head, dim=2)  # float32[G_aux, T, B, 1]
		else:  # 'mean'
			td_targets_aux = td_per_head.mean(dim=2)  # float32[G_aux, T, B, 1]
		
		return td_targets_aux

	@torch.no_grad()
	def compute_imagination_td_targets(self, rewards, next_z, terminated, task):
		"""Compute 1-step TD targets for all head combinations from imagined rollout.
		
		Returns raw targets without any head reduction — that's left to the loss functions.
		Used for both value loss and regression policy loss which require different reductions.

		Args:
			rewards (Tensor[T, R, H, B_exp, 1]): Rewards from R reward heads × H dynamics heads.
			next_z (Tensor[T, H, B_exp, L]): Next latent states from H dynamics heads.
			terminated (Tensor[T, H, B_exp, 1]): Termination flags per dynamics head.
			task: Optional task identifier.

		Returns:
			Tuple[Tensor[T, R, H, Ve, B_exp, 1], Tensor[T, H, Ve, B_exp, 1]]:
				- td_targets: Raw TD targets for all head combinations
				- v_next: V(next_z) for all heads (for disagreement computation)
		"""
		T, H, B_exp, L = next_z.shape  # next_z: float32[T, H, B_exp, L]
		R = rewards.shape[1]  # rewards: float32[T, R, H, B_exp, 1]

		# Get discount
		discount = self.discount[task].unsqueeze(-1) if self.cfg.multitask else self.discount

		# Flatten for V call: [T, H, B_exp, L] -> [T, H*B_exp, L]
		next_z_flat = next_z.view(T, H * B_exp, L)  # float32[T, H*B_exp, L]

		# Get all Ve value heads using target network: [Ve, T, H*B_exp, K]
		v_logits = self.model.V(next_z_flat, task, return_type='all', target=True)  # float32[Ve, T, H*B_exp, K]
		v_values = math.two_hot_inv(v_logits, self.cfg)  # float32[Ve, T, H*B_exp, 1]

		# Reshape to [T, H, Ve, B_exp, 1]
		Ve = v_values.shape[0]
		v_values_reshaped = v_values.view(Ve, T, H, B_exp, 1)  # float32[Ve, T, H, B_exp, 1]
		v_next = v_values_reshaped.permute(1, 2, 0, 3, 4).contiguous()  # float32[T, H, Ve, B_exp, 1]

		# Compute TD targets: r + γ(1-term)V
		# rewards: [T, R, H, B_exp, 1], need to broadcast with Ve
		# terminated: [T, H, B_exp, 1], broadcast to [T, 1, H, 1, B_exp, 1]

		# Expand dimensions for broadcasting
		rewards_exp = rewards.unsqueeze(3)  # float32[T, R, H, 1, B_exp, 1]
		term_exp = terminated.unsqueeze(1).unsqueeze(3)  # float32[T, 1, H, 1, B_exp, 1]
		v_next_exp = v_next.unsqueeze(1)  # float32[T, 1, H, Ve, B_exp, 1]

		td_targets = rewards_exp + discount * (1 - term_exp) * v_next_exp  # float32[T, R, H, Ve, B_exp, 1]

		return td_targets, v_next

	@torch.no_grad()
	def compute_imagination_aux_td_targets(self, rewards, next_z, terminated, task):
		"""Compute 1-step TD targets for auxiliary values with different gammas.
		
		Similar to compute_imagination_td_targets but uses V_aux instead of V.
		Returns raw targets without head reduction.

		Args:
			rewards (Tensor[T, R, H, B_exp, 1]): Rewards from R reward heads × H dynamics heads.
			next_z (Tensor[T, H, B_exp, L]): Next latent states from H dynamics heads.
			terminated (Tensor[T, H, B_exp, 1]): Termination flags per dynamics head.
			task: Optional task identifier.

		Returns:
			Tensor[T, R, H, G_aux, B_exp, 1]: Raw aux TD targets for all head/gamma combinations.
				Returns None if no auxiliary gammas configured.
		"""
		G_aux = self.model._num_aux_gamma
		if G_aux == 0:
			return None

		T, H, B_exp, L = next_z.shape  # next_z: float32[T, H, B_exp, L]

		# Get discount
		discount = self.discount[task].unsqueeze(-1) if self.cfg.multitask else self.discount

		# Flatten for V_aux call: [T, H, B_exp, L] -> [T, H*B_exp, L]
		next_z_flat = next_z.view(T, H * B_exp, L)  # float32[T, H*B_exp, L]

		# Get aux values for all gammas using target network
		v_aux_flat = self.model.V_aux(next_z_flat, task, return_type='min', target=True)  # float32[G_aux, T, H*B_exp, 1]

		# Reshape to [G_aux, T, H, B_exp, 1]
		v_aux = v_aux_flat.view(G_aux, T, H, B_exp, 1)  # float32[G_aux, T, H, B_exp, 1]

		# Compute TD targets for each gamma
		# gammas_aux: [G_aux] tensor of gamma values (exclude primary gamma)
		gammas_aux = torch.tensor(self._all_gammas[1:], device=next_z.device, dtype=next_z.dtype)  # float32[G_aux]
		gammas_aux = gammas_aux.view(G_aux, 1, 1, 1, 1)  # float32[G_aux, 1, 1, 1, 1]

		# rewards: [T, R, H, B_exp, 1] -> expand for G_aux: [1, T, R, H, B_exp, 1]
		# terminated: [T, H, B_exp, 1] -> expand: [1, T, 1, H, B_exp, 1]
		# v_aux: [G_aux, T, H, B_exp, 1] -> expand for R: [G_aux, T, 1, H, B_exp, 1]
		rewards_exp = rewards.unsqueeze(0)  # float32[1, T, R, H, B_exp, 1]
		terminated_exp = terminated.unsqueeze(0).unsqueeze(2)  # float32[1, T, 1, H, B_exp, 1]
		v_aux_exp = v_aux.unsqueeze(2)  # float32[G_aux, T, 1, H, B_exp, 1]

		aux_td_targets = rewards_exp + gammas_aux * discount * (1 - terminated_exp) * v_aux_exp
		# Shape: float32[G_aux, T, R, H, B_exp, 1]

		# Permute to match main TD target convention: [T, R, H, G_aux, B_exp, 1]
		aux_td_targets = aux_td_targets.permute(1, 2, 3, 0, 4, 5).contiguous()  # float32[T, R, H, G_aux, B_exp, 1]

		return aux_td_targets

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
			lat_all, _, _, _ = self.model.rollout_latents(
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
			self.cfg.consistency_coef * consistency_loss
			+ self.cfg.encoder_consistency_coef * encoder_consistency_loss
			+ self.cfg.reward_coef * reward_loss
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

		return wm_total, info, z_rollout, lat_all

	# rollout_dynamics removed; world_model.rollout_latents handles vectorized rollouts

	def imagined_rollout(self, start_z, task=None, rollout_len=None):
		"""Roll out imagined trajectories from latent start states using world_model.rollout_latents.

		Uses all dynamics heads (head_mode='all') for multi-head pessimism.
		When rollout_len=1, all heads share the same action since the policy samples
		from the initial state before any dynamics step.

		Args:
			start_z (Tensor[S, B_orig, L]): Starting latents where S is number of starting states
				(typically T+1 from replay buffer horizon). Each of the S states becomes a
				separate starting point for imagination.
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
		
		# Determine head_mode based on local_td_target_dynamics_reduction
		# When 'single', use a randomly-selected dynamics head (cheapest)
		# Otherwise, use all heads and reduce later
		h_reduction = self.cfg.local_td_target_dynamics_reduction
		if self.cfg.td_bootstrap_mode == 'local' and h_reduction == 'single':
			head_mode = 'random'  # Use single randomly-selected head
			H = 1  # Only one head in output
		else:
			head_mode = 'all'
			H = int(self.cfg.planner_num_dynamics_heads)
		
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
			latents, actions, log_probs_raw, scaled_entropy_raw = self.model.rollout_latents(
				start_flat,
				use_policy=True,
				horizon=rollout_len,
				num_rollouts=n_rollouts,
				head_mode=head_mode,  # 'all' or 'random' based on config
				task=task,
			)
		# latents: float32[H, B_total, N, T+1, L]; actions: float32[B_total, N, T, A]
		# log_probs_raw: float32[B_total, N, T, 1] - ATTACHED for policy gradients
		# scaled_entropy_raw: float32[B_total, N, T, 1] - ATTACHED for policy gradients
		# H equals planner_num_dynamics_heads when head_mode='all'
		assert latents.shape[0] == H, f"Expected {H} heads, got {latents.shape[0]}"

		with maybe_range('Imagined/permute_view', self.cfg):
			# Reshape to [T+1, H, B, L] where B = B_total * n_rollouts
			# latents: [H, B_total, N, T+1, L] -> [T+1, H, B_total*N, L]
			B = B_total * n_rollouts  # final batch dimension
			# permute: [H, B_total, N, rollout_len+1, L] -> [rollout_len+1, H, B_total, N, L]
			lat_perm = latents.permute(3, 0, 1, 2, 4).contiguous()  # float32[rollout_len+1, H, B_total, N, L]
			z_seq = lat_perm.view(rollout_len + 1, H, B, L)  # float32[rollout_len+1, H, B, L]

		with maybe_range('Imagined/act_seq', self.cfg):
			# actions: [B_total, N, rollout_len, A] -> [rollout_len, B_total*N, A] -> [rollout_len, 1, B, A]
			actions_perm = actions.permute(2, 0, 1, 3).contiguous()  # float32[rollout_len, B_total, N, A]
			actions_flat = actions_perm.view(rollout_len, B, A)  # float32[rollout_len, B, A]
			# Add H=1 dim since actions are shared across heads
			actions_seq = actions_flat.unsqueeze(1)  # float32[rollout_len, 1, B, A]

		# Compute rewards and termination logits along imagined trajectories per head
		# Need to process each head's latents through reward/termination predictors
		with maybe_range('Imagined/rewards_term', self.cfg):
			# z_seq[:-1]: [rollout_len, H, B, L], actions for reward: need [rollout_len, H, B, A]
			# Expand actions to match heads: [rollout_len, 1, B, A] -> [rollout_len, H, B, A]
			actions_expanded = actions_seq.expand(rollout_len, H, B, A)  # float32[rollout_len, H, B, A]

			# Flatten H*B for reward/termination calls
			z_for_reward = z_seq[:-1].view(rollout_len, H * B, L)  # float32[rollout_len, H*B, L]
			actions_for_reward = actions_expanded.reshape(rollout_len, H * B, A)  # float32[rollout_len, H*B, A]

			# Get reward logits from all reward heads: [R, rollout_len, H*B, K]
			reward_logits_all = self.model.reward(z_for_reward, actions_for_reward, task, head_mode='all')
			R = reward_logits_all.shape[0]  # number of reward heads
			# Convert to scalar rewards: [R, rollout_len, H*B, 1]
			rewards_flat = math.two_hot_inv(reward_logits_all, self.cfg)  # float32[R, rollout_len, H*B, 1]
			# Reshape to [rollout_len, R, H, B, 1]
			rewards = rewards_flat.permute(1, 0, 2, 3).view(rollout_len, R, H, B, 1)  # float32[rollout_len, R, H, B, 1]

			if self.cfg.episodic:
				term_logits_flat = self.model.termination(z_for_reward, task, unnormalized=True)
				term_logits = term_logits_flat.view(rollout_len, H, B, 1)  # float32[rollout_len, H, B, 1]
				terminated = (torch.sigmoid(term_logits) > 0.5).float()
			else:
				term_logits = torch.zeros(rollout_len, H, B, 1, device=z_seq.device, dtype=z_seq.dtype)
				terminated = torch.zeros(rollout_len, H, B, 1, device=z_seq.device, dtype=z_seq.dtype)

		# Avoid in-place detach on a view; build a fresh contiguous tensor
		with maybe_range('Imagined/final_pack', self.cfg):
			# Concatenate attached z_seq[:1] with detached z_seq[1:]
			z_seq_out = torch.cat([z_seq[:1], z_seq[1:].detach()], dim=0)  # float32[T+1, H, B, L]

			# NOTE: log_probs and scaled_entropy from rollout_latents are NOT returned.
			# They would require gradients through pi() which causes CUDAGraph memory conflicts
			# due to graph breaks from torch.no_grad() in later computations. Instead, the policy
			# loss (both SVG and regression) recomputes these from scratch using the detached
			# z and actions, ensuring gradients flow only through the policy params.

		return {
			'z_seq': z_seq_out,  # float32[T+1, H, B, L]
			'actions': actions_seq.detach(),  # float32[T, 1, B, A] (shared across heads)
			'rewards': rewards.detach(),  # float32[T, R, H, B, 1]
			'terminated': terminated.detach(),  # float32[T, H, B, 1]
			'termination_logits': term_logits.detach(),  # float32[T, H, B, 1]
			# Batch structure info for proper softmax over samples in regression policy
			'batch_structure': {
				'S': S,  # number of starting states
				'B_orig': B_orig,  # original batch size
				'N': n_rollouts,  # num_rollouts (samples per state)
				'B': B,  # flattened batch = S * B_orig * N
			},
		}



	def calculate_value_loss(self, z_seq, actions, rewards, terminated, full_detach, task=None, z_td=None, sample_weights=None, batch_structure=None):
		"""Compute primary critic loss on arbitrary latent sequences with multi-head support.
		
		With V-function, we predict V(z) for each state in the sequence and
		train against TD targets r + γ * V(next_z). Value predictions use head 0
		only (all heads are identical before dynamics rollout). TD targets use all
		heads for next-state diversity, then reduce over H.

		Args:
			z_seq (Tensor[T+1, H, B, L]): Latent sequences with H heads.
			actions (Tensor[T, H_a, B, A]): Actions (H_a=1 when shared across heads).
			rewards (Tensor[T, R, H, B, 1]): Rewards with R reward heads, H dynamics heads.
			terminated (Tensor[T, H, B, 1]): Termination signals per head.
			full_detach (bool): Whether to detach z_seq from graph.
			task: Task index for multitask.
			z_td (Tensor[T, H, B, L], optional): Override next_z for TD target.
			sample_weights (Tensor[T, B, 1], optional): Softmax weights for weighted averaging
				over N rollouts per state. If None, use uniform averaging.
			batch_structure (dict, optional): Contains S, B_orig, N, B for proper weighted
				averaging over N samples. Required when sample_weights is provided.

		Returns:
			Tuple[Tensor, TensorDict]: Scalar loss and info dict.
		"""
		# Note: z_td is always None now (imagination mode only, no replay_rollout)
		
		# z_seq: [T+1, H, B, L]; actions: [T, H_a, B, A] where H_a may be 1
		T_plus_1, H, B, L = z_seq.shape
		T = T_plus_1 - 1
		K = self.cfg.num_bins
		device = z_seq.device

		rho_pows = torch.pow(self.cfg.rho, torch.arange(T, device=device, dtype=z_seq.dtype))  # float32[T]

		z_seq = z_seq.detach() if full_detach else z_seq  # z_seq: float32[T+1, H, B, L]
		rewards = rewards.detach()  # float32[T, R, H, B, 1]
		terminated = terminated.detach()  # float32[T, H, B, 1]

		# V-function: use head 0 only for value predictions (all heads identical at t=0)
		# z_seq[:-1, 0]: [T, B, L] - pick head 0
		z_for_v = z_seq[:-1, 0]  # float32[T, B, L]
		vs = self.model.V(z_for_v, task, return_type='all')  # float32[Ve, T, B, K]
		Ve = vs.shape[0]

		with maybe_range('Value/td_target', self.cfg):
			with torch.no_grad():
				if z_td is not None:
					td_targets, td_std_across_heads = self._td_target(z_td, rewards, terminated, task)
				else:
					td_targets, td_std_across_heads = self._td_target(z_seq[1:], rewards, terminated, task)
				# td_targets: float32[Ve, T, B, 1], td_std_across_heads: float32[T, B, 1]

		with maybe_range('Value/ce', self.cfg):
			# TD targets are [Ve, T, B, 1], vs is [Ve, T, B, K]
			# Flatten for soft_ce: [Ve*T*B, K] and [Ve*T*B, 1]
			vs_flat_ce = vs.contiguous().view(Ve * T * B, K)
			td_flat = td_targets.contiguous().view(Ve * T * B, 1)
			
			val_ce_flat = math.soft_ce(vs_flat_ce, td_flat, self.cfg)  # float32[Ve*T*B]
			val_ce = val_ce_flat.view(Ve, T, B, 1)  # float32[Ve, T, B, 1]
			
			# Apply sample weights if provided (weighted averaging over N rollouts per state)
			if sample_weights is not None and batch_structure is not None:
				# sample_weights: [T, B, 1], need to expand for Ve
				# weights sum to 1 per state over N samples
				# B = S * B_orig * N, want weighted mean over N for each of (S * B_orig) states
				S = batch_structure['S']
				B_orig = batch_structure['B_orig']
				N = batch_structure['N']
				num_states = S * B_orig
				
				# Expand weights for Ve dimension: [1, T, B, 1]
				w = sample_weights.unsqueeze(0)  # float32[1, T, B, 1]
				
				# Weighted sum over B (which contains N samples per state, weights sum to 1 per state)
				# Then divide by num_states to get mean over states
				val_ce_weighted = (w * val_ce).sum(dim=2, keepdim=True) / num_states  # float32[Ve, T, 1, 1]
				val_ce_per_t = val_ce_weighted.mean(dim=0).squeeze(-1).squeeze(-1)  # float32[T]
			else:
				# Uniform averaging: mean over Ve and B, keep T
				val_ce_per_t = val_ce.mean(dim=(0, 2)).squeeze(-1)  # float32[T]

		weighted = val_ce_per_t * rho_pows
		loss = weighted.mean()

		info = TensorDict({
			'value_loss': loss
		}, device=device, non_blocking=True)

		for t in range(T):
			info.update({f'value_loss/step{t}': val_ce_per_t[t]}, non_blocking=True)
		
		value_pred = math.two_hot_inv(vs, self.cfg)  # float32[Ve, T, B, 1]
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
   
		# Value error: td_targets is [Ve, T, B, 1], value_pred is [Ve, T, B, 1]
		value_error = value_pred - td_targets  # float32[Ve, T, B, 1]
		for i in range(T):
			info.update({f"value_error_abs_mean/step{i}": value_error[:, i].abs().mean(),
						f"value_error_std/step{i}": value_error[:, i].std(),
						f"value_error_max/step{i}": value_error[:, i].abs().max()}, non_blocking=True)


		return loss, info

	def calculate_aux_value_loss(self, z_seq, actions, rewards, terminated, full_detach, task=None, z_td=None, sample_weights=None, batch_structure=None):
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
			sample_weights (Tensor[T, B, 1], optional): Softmax weights for weighted averaging
				over N rollouts per state. If None, use uniform averaging.
			batch_structure (dict, optional): Contains S, B_orig, N, B for proper weighted
				averaging over N samples. Required when sample_weights is provided.

		Returns:
			Tuple[Tensor, TensorDict]: Scalar loss and info dict.
		"""
		# Note: z_td is always None now (imagination mode only, no replay_rollout)
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

		# V_aux: use head 0 only for value predictions (all heads identical at t=0)
		# z_seq[:-1, 0]: [T, B, L] - pick head 0
		z_for_v = z_seq[:-1, 0]  # float32[T, B, L]
		v_aux_logits = self.model.V_aux(z_for_v, task, return_type='all')  # float32[G_aux, T, B, K] or None
		if v_aux_logits is None:
			return torch.zeros((), device=device), TensorDict({}, device=device)

		G_aux = v_aux_logits.shape[0]  # v_aux_logits: float32[G_aux, T, B, K]

		with maybe_range('Aux/td_target', self.cfg):
			with torch.no_grad():
				if z_td is not None:
					aux_td_targets = self._td_target_aux(z_td, rewards, terminated, task)  # float32[G_aux, T, B, 1]
				else:
					aux_td_targets = self._td_target_aux(z_seq[1:], rewards, terminated, task)  # float32[G_aux, T, B, 1]

		with maybe_range('Aux/ce', self.cfg):
			# TD targets are [G_aux, T, B, 1], v_aux_logits is [G_aux, T, B, K]
			# Flatten for soft_ce: [G_aux*T*B, K] and [G_aux*T*B, 1]
			vaux_flat = v_aux_logits.contiguous().view(G_aux * T * B, K)
			aux_targets_flat = aux_td_targets.contiguous().view(G_aux * T * B, 1)
			
			aux_ce_flat = math.soft_ce(vaux_flat, aux_targets_flat, self.cfg)  # float32[G_aux*T*B]
			aux_ce = aux_ce_flat.view(G_aux, T, B, 1)  # float32[G_aux, T, B, 1]
			
			# Apply sample weights if provided (weighted averaging over N rollouts per state)
			if sample_weights is not None and batch_structure is not None:
				# sample_weights: [T, B, 1], need to expand for G_aux
				# B = S * B_orig * N, want weighted mean over N for each state
				S = batch_structure['S']
				B_orig = batch_structure['B_orig']
				N = batch_structure['N']
				num_states = S * B_orig
				
				# Expand weights for G_aux dimension: [1, T, B, 1]
				w = sample_weights.unsqueeze(0)  # float32[1, T, B, 1]
				
				# Weighted sum over B (contains N samples per state, weights sum to 1 per state)
				# Then divide by num_states to get mean over states
				aux_ce_weighted = (w * aux_ce).sum(dim=2, keepdim=True) / num_states  # float32[G_aux, T, 1, 1]
				aux_ce_per_t = aux_ce_weighted.squeeze(-1).squeeze(-1)  # float32[G_aux, T]
			else:
				# Uniform averaging: mean over B, keep G_aux and T
				aux_ce_per_t = aux_ce.mean(dim=2).squeeze(-1)  # float32[G_aux, T]

		weighted = aux_ce_per_t * rho_pows.unsqueeze(0)  # float32[G_aux, T]
		losses = weighted.mean(dim=1)  # float32[G_aux] - mean over time per head
		# Mean over aux heads (not sum) - LR scaling compensates for ensemble size
		loss_mean = losses.mean()  # scalar

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

	def _compute_loss_components(self, obs, action, reward, terminated, task, update_value, log_grads, update_world_model=True):
		"""Compute world model, value, and auxiliary value losses.
		
		Args:
			obs: Observation sequence, float32[T+1, B, ...].
			action: Action sequence, float32[T, B, A].
			reward: Reward sequence, float32[T, B, 1].
			terminated: Termination flags, float32[T, B, 1].
			task: Task indices for multitask, or None.
			update_value: If False, skip value/aux losses (return zeros).
			log_grads: Whether to log gradient statistics.
			update_world_model: If False, skip WM losses (return zeros), set z_rollout=None,
				and block encoder gradients from value loss.
		"""
		device = self.device

		def encode_obs(obs_seq, use_ema, grad_enabled):
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
			return latents_flat.view(steps, batch, *latents_flat.shape[1:])  # float32[steps,batch,L]

		# Encode observations (needed for value computation even when not updating WM)
		z_true = encode_obs(obs, use_ema=False, grad_enabled=True)
		z_target = encode_obs(obs, use_ema=True, grad_enabled=False) if self.cfg.encoder_ema_enabled else None
		
		# Compute WM losses if updating world model, otherwise zero loss and skip rollout
		if update_world_model:
			wm_loss, wm_info, z_rollout, lat_all = self.world_model_losses(z_true, z_target, action, reward, terminated, task)
		else:
			# Skip WM losses: no rollout available, empty info (don't log zeros)
			wm_loss = torch.zeros((), device=device)
			wm_info = TensorDict({}, device=device)  # Empty - don't log anything for WM
			z_rollout = None  # No rollout available
			lat_all = None  # No multi-head rollout available

		# Imagined rollout for value/policy losses (always use imagination)
		# imagine_initial_source controls starting point:
		#   'replay_true': start from true encoded latents z_true
		#   'replay_rollout': start from dynamics rollout z_rollout (head 0)
		imagined = None
		sample_weights_for_value = None
		batch_structure_for_value = None
		if update_value:
			imagine_source = getattr(self.cfg, 'imagine_initial_source', 'replay_true')
			# When z_rollout is None (WM not updated), force replay_true
			if z_rollout is None:
				imagine_source = 'replay_true'
			if imagine_source == 'replay_true':
				# When WM not updated: no WM gradients to encoder, so also block value gradients.
				start_z = z_true.detach() if not update_world_model else z_true
			elif imagine_source == 'replay_rollout':
				# z_rollout[0] = encoder output (same as z_true[0]), z_rollout[1:] = dynamics predictions.
				# When WM updated: keep z_rollout[0] attached for encoder gradients, detach z_rollout[1:]
				# to block gradients to dynamics (already trained via consistency loss).
				# When WM not updated: fully detach (no encoder gradients from value).
				if not update_world_model:
					start_z = z_rollout.detach()
				else:
					start_z = torch.cat([z_rollout[:1], z_rollout[1:].detach()], dim=0).contiguous()
			else:
				raise ValueError(f"imagine_initial_source must be 'replay_true' or 'replay_rollout', got '{imagine_source}'")
			
			# Imagination horizon is always 1 (hardcoded)
			IMAGINATION_HORIZON = 1

			imagined = self.imagined_rollout(start_z, task=task, rollout_len=IMAGINATION_HORIZON)
			# imagined contains:
			#   z_seq: float32[T+1, H, B, L]
			#   actions: float32[T, 1, B, A] (shared across heads)
			#   rewards: float32[T, R, H, B, 1]
			#   terminated: float32[T, H, B, 1]
			#   batch_structure: dict with S, B_orig, N, B
			# NOTE: log_probs/scaled_entropy are NOT returned to avoid CUDAGraph conflicts.
			# Policy loss recomputes them from detached z and actions.
			
			full_detach = self.cfg.detach_imagine_value
			batch_structure = imagined['batch_structure']
			
			# Compute regression weights early if enabled (shared between value and policy losses)
			# This avoids recomputing weights in both places and ensures consistency.
			weights_pess = None
			weights_opti = None
			weights_info = None
			if self.cfg.pi_regression_enabled and self.cfg.weighted_value_targets:
				# Extract tensors for weight computation
				z_seq = imagined['z_seq'].detach()  # float32[T+1, H, B, L]
				rewards_im = imagined['rewards'].detach()  # float32[T, R, H, B, 1]
				terminated_im = imagined['terminated'].detach()  # float32[T, H, B, 1]
				
				# Compute TD targets for weighting
				next_z = z_seq[1:]  # float32[T, H, B, L]
				td_targets, v_next = self.compute_imagination_td_targets(
					rewards_im, next_z, terminated_im, task
				)
				# td_targets: [T, R, H, Ve, B, 1], v_next: [T, H, Ve, B, 1]
				
				# Compute both pessimistic and optimistic weights
				weights_pess, weights_opti, weights_info = self.compute_regression_weights(
					td_targets, v_next, batch_structure, task
				)
				# weights_pess/weights_opti: [T, B, 1], both detached
				
				# Use pessimistic weights for value loss
				sample_weights_for_value = weights_pess
				batch_structure_for_value = batch_structure
			
			value_loss, value_info = self.calculate_value_loss(
				imagined['z_seq'],
				imagined['actions'],
				imagined['rewards'],
				imagined['terminated'],
				full_detach,
				z_td=None,
				task=task,
				sample_weights=sample_weights_for_value,
				batch_structure=batch_structure_for_value,
			)
			
			# Add weights info to value_info for logging
			if weights_info is not None:
				value_info.update(weights_info, non_blocking=True)

			aux_loss = torch.zeros((), device=device)
			aux_info = TensorDict({}, device=device)
			if self.cfg.multi_gamma_loss_weight != 0 and self.model._num_aux_gamma > 0:
				aux_loss, aux_info = self.calculate_aux_value_loss(
					imagined['z_seq'],
					imagined['actions'],
					imagined['rewards'],
					imagined['terminated'],
					full_detach,
					z_td=None,
					task=task,
					sample_weights=sample_weights_for_value,
					batch_structure=batch_structure_for_value,
				)
		else:
			# Skip value/aux losses when not updating value
			value_loss = torch.zeros((), device=device)
			value_info = TensorDict({}, device=device)
			aux_loss = torch.zeros((), device=device)
			aux_info = TensorDict({}, device=device)

		info = TensorDict({}, device=device)
		info.update(wm_info, non_blocking=True)
		info.update(value_info, non_blocking=True)
		info.update(aux_info, non_blocking=True)
  
		critic_weighted = self.cfg.value_coef * value_loss * self.cfg.imagine_value_loss_coef_mult
		aux_weighted = self.cfg.multi_gamma_loss_weight * aux_loss * self.cfg.imagine_value_loss_coef_mult

		total_loss = wm_loss + critic_weighted + aux_weighted
		info.update({
			'total_loss': total_loss,
			'wm_loss': wm_loss,
			'value_loss_weighted': critic_weighted,
			'aux_loss_mean_weighted': aux_weighted
		}, non_blocking=True)

		# Prepare policy_data for regression policy loss (all tensors DETACHED)
		# This data will be passed to _compute_regression_pi_loss which is separately compiled.
		policy_data = None
		if imagined is not None and self.cfg.pi_regression_enabled:
			# Extract and detach tensors needed for regression policy loss
			z_seq = imagined['z_seq'].detach()  # float32[T+1, H, B, L]
			actions_im = imagined['actions'].detach()  # float32[T, 1, B, A]
			batch_structure = imagined['batch_structure']
			
			# Get z at t=0 for policy (head 0, since all heads identical before dynamics)
			z_for_pi = z_seq[:-1, 0].detach()  # float32[T, B, L]
			actions_flat = actions_im[:, 0].detach()  # float32[T, B, A]
			
			# Use pre-computed weights if weighted_value_targets is enabled
			# Otherwise, compute weights in _compute_regression_pi_loss (legacy path - not implemented)
			if self.cfg.weighted_value_targets:
				# weights_pess and weights_opti were computed earlier for value loss
				policy_data = {
					'z': z_for_pi,
					'actions': actions_flat,
					'weights_pess': weights_pess,  # float32[T, B, 1]
					'weights_opti': weights_opti,
					'S': batch_structure['S'],
					'B_orig': batch_structure['B_orig'],
					'N': batch_structure['N'],
					'B': batch_structure['B'],
				}
			else:
				raise NotImplementedError(
					"pi_regression_enabled requires weighted_value_targets=true. "
					"The old path computing weights inside calculate_regression_pi_loss is deprecated."
				)

		return {
			'wm_loss': wm_loss,
			'value_loss': value_loss,
			'aux_loss': aux_loss,
			'info': info,
			'z_true': z_true,
			'z_rollout': z_rollout,
			'imagined': imagined,  # Full imagined dict (kept for legacy SVG path)
			'policy_data': policy_data,  # Detached data for regression policy loss
		}

	def _update(self, obs, action, reward, terminated, update_value=True, update_pi=True, update_world_model=True, task=None):
		"""Single gradient update step over world model, critic, and policy.
		
		Args:
			update_value: If True, compute and apply value/aux losses. If False, skip value losses.
			update_pi: If True, update policy. If False, skip policy update.
			update_world_model: If True, compute WM losses. If False, skip WM losses and use replay_true for imagination.
		"""
		log_grads = self.cfg.log_gradients_per_loss and self.log_detailed

		with maybe_range('Agent/update', self.cfg):
			self.model.train(True)
			if log_grads:
				components = self._compute_loss_components_eager(obs, action, reward, terminated, task, update_value, log_grads, update_world_model)
			else:
				components = self._compute_loss_components(obs, action, reward, terminated, task, update_value, log_grads, update_world_model)
    
			wm_loss = components['wm_loss']
			value_loss = components['value_loss']
			aux_loss = components['aux_loss']
			info = components['info']
			z_true = components['z_true']
			z_rollout = components['z_rollout']
			imagined = components['imagined']  # Full imagined dict (kept for legacy SVG path)
			policy_data = components['policy_data']  # Detached data for regression policy loss
      
			total_loss = info['total_loss']		
			self.optim.zero_grad(set_to_none=True) # this is crucial since the pi_loss also backprops through the world model, we want to clear those remaining gradients

			if log_grads and update_world_model:
				info = self.probe_wm_gradients(info)

			total_loss.backward()
			grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
			if log_grads:
				self.optim.step()
			else:
				self.optim_step()

			self.optim.zero_grad(set_to_none=True)

			# Policy update (conditional on update_pi)
			pi_grad_norm = torch.zeros((), device=self.device)
			pi_info = TensorDict({}, device=self.device)
			if update_pi:

				if self.cfg.pi_regression_enabled:
					# ========== Regression-style policy loss (AWR-like) ==========
					# Use pre-computed policy_data from _compute_loss_components.
					# All tensors are DETACHED - no references to encoder/dynamics graph.
					# _compute_regression_pi_loss is separately compiled (different CUDAGraph).
					assert policy_data is not None, (
						"pi_regression_enabled requires update_value=True to have policy_data"
					)
					
					if log_grads:
						pi_loss, pi_info = self._compute_regression_pi_loss_eager(
							policy_data['z'],
							policy_data['actions'],
							policy_data['weights_pess'],
							policy_data['weights_opti'],
							policy_data['S'],
							policy_data['B_orig'],
							policy_data['N'],
							policy_data['B'],
							task,
						)
					else:
						pi_loss, pi_info = self._compute_regression_pi_loss(
							policy_data['z'],
							policy_data['actions'],
							policy_data['weights_pess'],
							policy_data['weights_opti'],
							policy_data['S'],
							policy_data['B_orig'],
							policy_data['N'],
							policy_data['B'],
							task,
						)
				else:
					# ========== Legacy SVG-style policy loss ==========
					# Use z_rollout from world model losses for policy training
					z_for_pi = z_rollout.detach()  # float32[T+1, B, L]
					pi_loss, pi_info = self.update_pi(z_for_pi, task)
				
				pi_total = pi_loss * self.cfg.policy_coef
				pi_total.backward()
				if log_grads:
					info = self.probe_pi_gradients(info)
				# Clip gradients for all policy params (both _pi and _pi_optimistic if dual policy)
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

			# Soft updates: only update targets when corresponding component is updated
			if update_value:
				self.model.soft_update_target_V()
			if update_pi:
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
  
	def update(self, buffer, step=0, update_value=True, update_pi=True, update_world_model=True):
		"""
		Main update function. Corresponds to one iteration of model learning.

		Args:
			buffer (common.buffer.Buffer): Replay buffer.
			step: Current training step.
			update_value: If True, compute and apply value/aux losses.
			update_pi: If True, update policy.
			update_world_model: If True, compute WM losses. If False, skip WM losses.

		Returns:
			dict: Dictionary of training statistics.
		"""
		with maybe_range('update/sample_buffer', self.cfg):
			obs, action, reward, terminated, task = buffer.sample()

		self._step = step
		# Log detailed info when updating value and at log_detail_freq intervals
		if (self._step % self.cfg.log_detail_freq == 0) and update_value:
			self.log_detailed = True
		else:
			self.log_detailed = False

		# Encoder LR step-change: apply once when crossing threshold
		enc_lr_cutoff = int((1 - self.cfg.enc_lr_step_ratio) * self.cfg.steps)
		if not self._enc_lr_stepped and self._step >= enc_lr_cutoff:
			new_enc_lr = self._enc_lr_initial * self.cfg.enc_lr_step_scale
			self.optim.param_groups[0]['lr'] = new_enc_lr
			self._enc_lr_stepped = True
			log.info('Step %d: encoder LR stepped from %.6f to %.6f',
					 self._step, self._enc_lr_initial, new_enc_lr)

		self.dynamic_entropy_coeff.fill_(self.get_entropy_coeff(self._step))
		kwargs = {}
		if task is not None:
			kwargs["task"] = task
		torch.compiler.cudagraph_mark_step_begin()
  
		info = self._update(obs, action, reward, terminated, update_value=update_value, update_pi=update_pi, update_world_model=update_world_model, **kwargs)

		# Log current encoder LR for W&B tracking
		info['encoder_lr'] = self.optim.param_groups[0]['lr']

		return info


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
		# Filter out frozen params (e.g., dynamics prior) that don't require grad
		flat_params = []
		index = []  # (group_name, param_obj) pairs
		for gname, params in groups.items():
			if gname == 'policy':  # skip policy here
				continue
			for p in params:
				if p.requires_grad:
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
	def probe_pi_gradients(self, info):
		"""Probe gradient norms from policy loss to all parameter groups.

		Must be called AFTER backward() so that .grad attributes are populated.
		This reveals SVG-style gradient flow: policy loss backprops through
		reward/dynamics models even though only policy params are updated.

		Args:
			info (TensorDict): Info dict to update with gradient norms.

		Returns:
			TensorDict: Updated info dict with grad_norm/pi_loss/{group} entries.
		"""
		groups = self._grad_param_groups()

		# Accumulate L2 norm per component group from .grad attributes
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
					components = self._compute_loss_components(obs, action, reward, terminated, task, update_value=True, log_grads=False)
					val_info = components['info']

					# Use z_rollout for policy validation (matches legacy SVG training path)
					# z_rollout: [T+1, B, L] — dynamics rollout from encoded observations
					z_for_pi = components['z_rollout'].detach()
      
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
