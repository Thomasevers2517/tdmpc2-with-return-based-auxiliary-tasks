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
			{'params': self.model._Rs.parameters(), 'lr': lr_reward},
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
		self._last_reanalyze_step = -1  # Track last step where reanalyze was run (prevent duplicates with utd_ratio > 1)
		self.log_detailed = None  # whether to log detailed gradients (set via external signal)
		
		# Frozen random encoder for KNN entropy estimation (observation diversity metric)
		# Only for state observations (not pixels)
		if self.cfg.obs == 'state':
			obs_dim = list(self.cfg.obs_shape.values())[0][0]  # Get first obs modality dim
			knn_entropy_dim = int(getattr(self.cfg, 'knn_entropy_dim', 128))
			self._knn_encoder = torch.nn.Sequential(
				torch.nn.Linear(obs_dim, 64),
				torch.nn.ReLU(),
				torch.nn.Linear(64, knn_entropy_dim),
			).to(self.device)
			# Freeze encoder
			for p in self._knn_encoder.parameters():
				p.requires_grad_(False)
			self._knn_encoder.eval()
		else:
			self._knn_encoder = None
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

		# Modular planner (replaces legacy _plan / _prev_mean logic)
		self.planner = Planner(cfg=self.cfg, world_model=self.model, scale=self.scale, discount=self.discount)
		if cfg.compile:
			log.info('Compiling update function with torch.compile...')
			# Keep eager references
			self._compute_loss_components_eager = self._compute_loss_components
			# Relax fullgraph to reduce guard creation / trace size
			self._compute_loss_components = torch.compile(self._compute_loss_components, mode=self.cfg.compile_type, fullgraph=False)
			self.calc_pi_losses_eager = self.calc_pi_losses
			self.calc_pi_losses = torch.compile(self.calc_pi_losses, mode=self.cfg.compile_type, fullgraph=False)

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

			# Use dynamic=False to avoid symbolic shapes which conflict with vmap in reward/V heads
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
		return

	@torch.no_grad()
	def act(self, obs, eval_mode: bool = False, task=None, mpc: bool = True, eval_head_reduce: str = 'default', log_detailed: bool = False):
		"""Select an action.

		If `mpc=True`, uses modular `Planner` over latent space; else falls back to single policy prior.

		Args:
			obs (Tensor): Observation (already batched with leading dim 1).
			eval_mode (bool): Evaluation flag (planner switches to value-only scoring / argmax selection).
			task: Optional task index (unsupported for planner; passed to policy when mpc=False).
			mpc (bool): Whether to use planning.
			eval_head_reduce (str): Head reduction mode for eval ('default', 'mean').
				'default' uses planner_value_std_coef_eval (typically pessimistic).
				'mean' uses value_std_coef=0 for mean-only reduction.
			log_detailed (bool): If True, planner returns PlannerAdvancedInfo with full iteration history.

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
				
				# Convert eval_head_reduce to value_std_coef_override
				# 'default' -> None (use config), 'mean' -> 0.0 (mean-only)
				value_std_coef_override = 0.0 if eval_head_reduce == 'mean' else None
				
				# Planner now returns batch dimension: action [B, A], mean [B, T, A], std [B, T, A]
				# For acting (B=1), squeeze the batch dim
				chosen_action, planner_info, mean, std = self.planner.plan(
					z0, task=None, eval_mode=eval_mode, log_detailed=log_detailed,
					train_noise_multiplier=(0.0 if eval_mode else float(self.cfg.train_act_std_coeff)),
					value_std_coef_override=value_std_coef_override,
					use_warm_start=True,
					update_warm_start=True,
					reanalyze=False,
				)

				# Planner already applies any training noise and clamps
				# Squeeze batch dim for single-sample acting
				return chosen_action.squeeze(0), planner_info
			# Policy-prior action (non-MPC path)
			z = self.model.encode(obs, task_tensor)
			action_pi, info_pi = self.model.pi(z, task_tensor)
			if eval_mode:
				action_pi = info_pi['mean']
			return action_pi[0], None

	@torch.no_grad()
	def reanalyze(self, obs, task=None):
		"""Run planner on observations to get expert targets for policy distillation.
		
		This is separate from act() - reanalyze uses different planner settings
		(no warm start, specific hyperparameters for generating consistent targets).
		Used both for immediate reanalyze (single obs from online trainer) and
		lazy reanalyze (batch of obs from replay buffer).
		
		Args:
			obs: Observations, float32[B, *obs_shape] where B can be 1 or larger.
			task: Optional task indices for multitask.
		
		Returns:
			expert_action_dist: float32[B, A, 2] where [...,0]=mean, [...,1]=std.
			expert_value: float32[B] scalar value estimates (None if B>1).
			planner_info: PlannerBasicInfo or None (for logging).
		"""
		self.model.eval()
		with maybe_range('Agent/reanalyze', self.cfg):
			if task is not None:
				task_tensor = torch.tensor([task], device=self.device) if not torch.is_tensor(task) else task
			else:
				task_tensor = None
			
			# Encode observations -> latent states [B, L]
			z = self.model.encode(obs, task_tensor)  # float32[B, L]
			B = z.shape[0]
			
			# Run planner with reanalyze=True:
			# - use_warm_start=False (independent per sample)
			# - update_warm_start=False (don't pollute warm start for acting)
			# - Uses reanalyze-specific hyperparameters (iterations, num_samples, num_pi_trajs)
			chosen_action, planner_info, mean, std = self.planner.plan(
				z,
				task=task_tensor,
				eval_mode=False,
				log_detailed=False,  # No detailed logging during reanalyze
				train_noise_multiplier=0.0,  # No noise for expert targets
				value_std_coef_override=None,
				use_warm_start=False,
				update_warm_start=False,
				reanalyze=True,
			)
			# mean, std: [B, T, A] -> take first timestep [:, 0, :] -> [B, A]
			expert_mean = mean[:, 0, :]  # float32[B, A]
			expert_std = std[:, 0, :]    # float32[B, A]
			
			# Ensure std is within valid range (defensive; planner should already clamp)
			expert_std = expert_std.clamp(self.cfg.min_std, self.cfg.max_std)
			
			# Pack into [B, A, 2] format
			expert_action_dist = torch.stack([expert_mean, expert_std], dim=-1)  # float32[B, A, 2]
			
			# Get expert value: only available for B=1 when planner_info is returned
			if planner_info is not None:
				expert_value = planner_info.value_chosen  # scalar or [1]
				if expert_value.dim() == 0:
					expert_value = expert_value.unsqueeze(0)  # Ensure [B] shape
			else:
				# B>1: planner_info is None, expert_value not available
				expert_value = None
			
			return expert_action_dist, expert_value, planner_info

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
		groups["reward"] = list(self.model._Rs.parameters())
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
		
		CORRECT OPTIMISM: For each dynamics head h, compute:
		  σ_h = σ^r_h + γ * σ^v_h  (reward std + discounted value std)
		  Q_h = μ_h + value_std_coef × σ_h
		Then reduce over dynamics heads:
		  value_std_coef > 0: max over H (optimistic)
		  value_std_coef < 0: min over H (pessimistic)
		  value_std_coef = 0: mean over H (neutral)
		
		When num_rollouts > 1, samples multiple actions per state to reduce
		variance in policy gradients.
		
		Args:
			z (Tensor[T, B, L]): Current latent states.
			task: Task identifier for multitask setup.
			optimistic: If True, use optimistic policy with +1.0 std_coef and
				scaled entropy. If False, use pessimistic policy with -1.0 std_coef.
			
		Returns:
			Tuple[Tensor, TensorDict]: Policy loss and info dict.
		"""
		T, B, L = z.shape
		# Number of action samples per state: num_rollouts if pi_multi_rollout enabled, else 1
		N = int(self.cfg.num_rollouts) if self.cfg.pi_multi_rollout else 1
		
		# Ensure contiguity for torch.compile compatibility
		z = z.contiguous()  # Required: z may be non-contiguous after detach/indexing ops
		
		# Select std_coef based on optimistic flag
		if optimistic:
			value_std_coef = self.cfg.optimistic_policy_value_std_coef  # +1.0 for optimistic
			entropy_coeff = self.dynamic_entropy_coeff * self.cfg.optimistic_entropy_mult
		else:
			value_std_coef = self.cfg.policy_value_std_coef  # -1.0 for pessimistic
			entropy_coeff = self.dynamic_entropy_coeff
			
		with maybe_range('Agent/update_pi', self.cfg):
			# Expand z to have N rollouts: [T, B, L] -> [T, B, N, L] -> [T, B*N, L]
			# This allows sampling N different actions per state for variance reduction
			z_expanded = z.unsqueeze(2).expand(T, B, N, L).reshape(T, B * N, L)  # float32[T, B*N, L]
			
			# Sample action from policy at current state (policy has gradients)
			# Due to stochasticity, each of the N copies gets a different action
			action, info = self.model.pi(z_expanded, task, optimistic=optimistic)  # action: float32[T, B*N, A]
			
			# Flatten for model calls
			z_flat = z_expanded.view(T * B * N, L)              # float32[T*B*N, L]
			action_flat = action.view(T * B * N, -1)  # float32[T*B*N, A]
			
			# Get discount factor
			if self.cfg.multitask:
				task_flat = task.repeat(T * N) if task is not None else None
				gamma = self.discount[task_flat].unsqueeze(-1)  # float32[T*B*N, 1]
				gamma_scalar = self.discount.mean().item()  # for std discounting
			else:
				task_flat = None
				gamma = self.discount  # scalar
				gamma_scalar = float(self.discount)
			
			# Predict reward r(z, a) from ALL reward heads
			# reward() returns distributional logits [R, T*B*N, K], convert to scalar
			reward_logits_all = self.model.reward(z_flat, action_flat, task_flat, head_mode='all')  # float32[R, T*B*N, K]
			R = reward_logits_all.shape[0]  # number of reward heads
			reward_all = math.two_hot_inv(reward_logits_all, self.cfg)  # float32[R, T*B*N, 1]
			
			# Roll through ALL dynamics heads to get next states z'
			next_z_all = self.model.next(z_flat, action_flat, task_flat, head_mode='all')  # float32[H, T*B*N, L]
			H = next_z_all.shape[0]  # number of dynamics heads
			
			# Evaluate V(z') for each dynamics head using ALL Ve value heads
			# Reshape to evaluate all heads at once: [H*T*B*N, L]
			next_z_all_flat = next_z_all.view(H * T * B * N, L)  # float32[H*T*B*N, L]
			task_flat_expanded = task_flat.repeat(H) if task_flat is not None else None
			# return_type='all_values' returns [Ve, H*T*B*N, 1] for all Ve heads
			v_next_all_flat = self.model.V(next_z_all_flat, task_flat_expanded, return_type='all_values', detach=True)  # float32[Ve, H*T*B*N, 1]
			Ve = v_next_all_flat.shape[0]  # number of value heads
			v_next_all = v_next_all_flat.view(Ve, H, T * B * N, 1)  # float32[Ve, H, T*B*N, 1]
			
			# CORRECT OPTIMISM: Compute per dynamics head
			# reward_all: [R, T*B*N, 1] - same reward for all dynamics heads
			# v_next_all: [Ve, H, T*B*N, 1] - different values per dynamics head
			
			# Reward mean and std across R reward heads (same for all dynamics heads)
			reward_mean = reward_all.mean(dim=0)  # float32[T*B*N, 1]
			reward_std = reward_all.std(dim=0, unbiased=(R > 1))  # float32[T*B*N, 1]
			
			# Value mean and std across Ve value heads, per dynamics head h
			v_mean_per_h = v_next_all.mean(dim=0)  # float32[H, T*B*N, 1]
			v_std_per_h = v_next_all.std(dim=0, unbiased=(Ve > 1))  # float32[H, T*B*N, 1]
			
			# Q_h = (r_mean + γ * v_mean_h) + std_coef * (r_std + γ * v_std_h)
			# Total mean per dynamics head: μ_h = r_mean + γ * v_mean_h
			q_mean_per_h = reward_mean + gamma * v_mean_per_h  # float32[H, T*B*N, 1]
			
			# Total std per dynamics head: σ_h = r_std + γ * v_std_h
			q_std_per_h = reward_std + gamma_scalar * v_std_per_h  # float32[H, T*B*N, 1]
			
			# Q_h = μ_h + std_coef * σ_h
			q_per_h = q_mean_per_h + value_std_coef * q_std_per_h  # float32[H, T*B*N, 1]
			
			# Reduce over dynamics heads based on sign of value_std_coef
			if value_std_coef > 0:
				# Optimistic: max over dynamics heads
				q_estimate_flat, _ = q_per_h.max(dim=0)  # float32[T*B*N, 1]
			elif value_std_coef < 0:
				# Pessimistic: min over dynamics heads
				q_estimate_flat, _ = q_per_h.min(dim=0)  # float32[T*B*N, 1]
			else:
				# Neutral: mean over dynamics heads
				q_estimate_flat = q_per_h.mean(dim=0)  # float32[T*B*N, 1]
			
			q_estimate = q_estimate_flat.view(T, B * N, 1)  # float32[T, B*N, 1]
			
			# For logging: std across dynamics heads (disagreement)
			q_std = q_per_h.std(dim=0, unbiased=(H > 1))  # float32[T*B*N, 1]
			
			# Update scale with the Q-estimate (first timestep batch for stability)
			# NOTE: Only update scale for pessimistic policy to avoid inplace op conflict
			# when computing both policies in a single backward pass.
			if not optimistic:
				self.scale.update(q_estimate[0])
			q_scaled = self.scale(q_estimate)  # float32[T, B*N, 1]
			
			# Temporal weighting
			rho_pows = torch.pow(self.cfg.rho, torch.arange(T, device=self.device))  # float32[T]
			
			# Entropy from policy (always use scaled_entropy with configurable action_dim power)
			entropy_term = info["scaled_entropy"]  # float32[T, B*N, 1]
			
			# Policy loss: maximize (q_scaled + entropy_coeff * entropy)
			# Uncertainty handling is now via value_std_coef in q_estimate computation
			# Apply rho weighting across time
			# Average over both B and N dimensions to reduce variance from multiple rollouts
			objective = q_scaled + entropy_coeff * entropy_term  # float32[T, B*N, 1]
			pi_loss = -(objective.mean(dim=(1, 2)) * rho_pows).mean()

			# Add hinge^p penalty on pre-squash mean μ
			lam = float(self.cfg.hinge_coef)
			hinge_loss = self.calc_hinge_loss(info["presquash_mean"].view(T, B, N, -1)[:, :, 0, :], rho_pows)  # Use first rollout for hinge
			pi_loss = pi_loss + lam * hinge_loss

			# For logging, reshape and average over N rollouts
			info_entropy = info["entropy"].view(T, B, N, 1).mean(dim=2)  # float32[T, B, 1]
			info_scaled_entropy = info["scaled_entropy"].view(T, B, N, 1).mean(dim=2)  # float32[T, B, 1]
			info_mean = info["mean"].view(T, B, N, -1).mean(dim=2)  # float32[T, B, A]
			info_log_std = info["log_std"].view(T, B, N, -1).mean(dim=2)  # float32[T, B, A]
			info_presquash_mean = info["presquash_mean"].view(T, B, N, -1).mean(dim=2)  # float32[T, B, A]
			q_estimate_avg = q_estimate.view(T, B, N, 1).mean(dim=2)  # float32[T, B, 1]

			info = TensorDict({
				"pi_loss": pi_loss,
				"pi_loss_weighted": pi_loss * self.cfg.policy_coef,
				"pi_entropy": info_entropy,
				"pi_scaled_entropy": info_scaled_entropy,
				"pi_entropy_multiplier": info["entropy_multiplier"],
				"pi_scale": self.scale.value,
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
				"pi_v_next_mean": v_next_all.mean(),
				"pi_q_std": q_std.mean(),  # Std across (Ve × H × R) - now used in q_estimate via value_std_coef
				"pi_num_rollouts": float(N),  # Log how many rollouts used (float for mean() compatibility)
			}, device=self.device)

			return pi_loss, info

	def calc_pi_distillation_losses(self, z, expert_action_dist, task, optimistic=False):
		"""Compute policy loss via KL divergence distillation from expert planner targets.
		
		Distills the planner's action distribution into the policy via KL divergence.
		This is an alternative to SVG-style policy optimization (calc_pi_losses).
		
		Args:
			z (Tensor[T+1, B, L]): Latent states. Uses z[:-1] for policy (matches actions).
			expert_action_dist (Tensor[T, B, A, 2]): Expert distributions where [...,0]=mean, [...,1]=std.
			task: Task identifier for multitask setup.
			optimistic (bool): If True, use optimistic policy and entropy coeff.
		
		Returns:
			Tuple[Tensor, TensorDict]: Policy loss and info dict.
		"""
		# Validate input shapes
		assert z.dim() == 3, f"z must be [T+1, B, L], got {z.shape}"
		assert expert_action_dist.dim() == 4, f"expert_action_dist must be [T, B, A, 2], got {expert_action_dist.shape}"
		assert expert_action_dist.shape[-1] == 2, f"expert_action_dist last dim must be 2, got {expert_action_dist.shape[-1]}"
		
		T_plus_1, B, L = z.shape
		T = T_plus_1 - 1
		T_expert, B_expert, A, _ = expert_action_dist.shape
		
		# Validate alignment: z has T+1 timesteps, expert has T
		assert T == T_expert, f"z time dim ({T_plus_1}-1={T}) must match expert time dim ({T_expert})"
		assert B == B_expert, f"z batch dim ({B}) must match expert batch dim ({B_expert})"
		
		# Use z[:-1] for policy (aligns with expert_action_dist timesteps)
		z_for_pi = z[:-1]  # float32[T, B, L]
		
		# Get policy distribution (optimistic or pessimistic based on flag)
		with maybe_range('Agent/pi_distillation', self.cfg):
			_, info = self.model.pi(z_for_pi, task, optimistic=optimistic)
		
		# Extract policy mean/std
		# Note: policy mean is already squashed (tanh applied), in [-1, 1]
		policy_mean = info["mean"]           # float32[T, B, A]
		policy_std = info["log_std"].exp()   # float32[T, B, A]
		
		# Validate policy output shapes
		assert policy_mean.shape == (T, B, A), f"policy_mean shape {policy_mean.shape} != expected ({T}, {B}, {A})"
		assert policy_std.shape == (T, B, A), f"policy_std shape {policy_std.shape} != expected ({T}, {B}, {A})"
		
		# Extract expert mean/std
		# Expert targets are from planner's final distribution, also in [-1, 1]
		expert_mean = expert_action_dist[..., 0]  # float32[T, B, A]
		expert_std = expert_action_dist[..., 1]   # float32[T, B, A]
		
		# Validate expert values are reasonable
		assert not torch.isnan(expert_mean).any(), "expert_mean contains NaN"
		assert not torch.isnan(expert_std).any(), "expert_std contains NaN"
		assert (expert_std > 0).all(), f"expert_std has non-positive values: min={expert_std.min()}"
		
		# Compute KL divergence: KL(policy || expert) per dimension
		# Mean over action dimensions for total KL per (t, b) - consistent with BMPC
		kl_per_dim = math.kl_div_gaussian(policy_mean, policy_std, expert_mean, expert_std)  # float32[T, B, A]
		kl_loss = kl_per_dim.mean(dim=-1, keepdim=True)  # float32[T, B, 1] - mean over A
		
		# Scale with running scale (like BMPC does for unit consistency)
		self.scale.update(kl_loss[0])
		kl_scaled = self.scale(kl_loss)  # float32[T, B, 1]
		
		# Entropy bonus (use optimistic or pessimistic entropy coeff)
		if optimistic:
			entropy_coeff = self.dynamic_entropy_coeff * self.cfg.optimistic_entropy_mult
		else:
			entropy_coeff = self.dynamic_entropy_coeff
		entropy_term = info["scaled_entropy"]  # float32[T, B, 1]
		
		# Temporal weighting with rho (exponential decay)
		rho_pows = torch.pow(self.cfg.rho, torch.arange(T, device=self.device))  # float32[T]
		
		# Final loss: minimize (KL - entropy_coeff * entropy), weighted by rho
		# Note: KL is already a "loss" (minimize), entropy is a bonus (maximize -> subtract)
		objective = kl_scaled - entropy_coeff * entropy_term  # float32[T, B, 1]
		pi_loss = (objective.mean(dim=(1, 2)) * rho_pows).mean()
		
		# Build info dict
		info_out = TensorDict({
			"pi_loss": pi_loss,
			"pi_loss_weighted": pi_loss * self.cfg.policy_coef,
			"pi_kl_loss": kl_loss.mean(),
			"pi_kl_per_dim": kl_per_dim.mean(),
			"pi_entropy": info["entropy"].mean(),
			"pi_scaled_entropy": info["scaled_entropy"].mean(),
			"pi_scale": self.scale.value,
			"pi_std": info["log_std"].mean(),
			"pi_mean": info["mean"].mean(),
			"pi_abs_mean": info["mean"].abs().mean(),
			"entropy_coeff": torch.tensor(entropy_coeff, device=self.device),
			"expert_mean_abs": expert_mean.abs().mean(),
			"expert_std_mean": expert_std.mean(),
		}, device=self.device)
		
		return pi_loss, info_out

	def update_pi(self, zs, task, expert_action_dist=None):
		"""
		Update policy using a sequence of latent states.
		
		Supports policy optimization methods:
		- 'svg': Backprop through world model (calc_pi_losses)
		- 'distillation': KL divergence to expert planner targets (calc_pi_distillation_losses)
		- 'both': Sum of SVG and distillation losses (pessimistic only)
		
		Pessimistic policy uses cfg.policy_optimization_method.
		Optimistic policy uses cfg.optimistic_policy_optimization_method (svg or distillation).

		Args:
			zs (torch.Tensor): Sequence of latent states [T+1, B, L].
			task (torch.Tensor): Task index (only used for multi-task experiments).
			expert_action_dist (torch.Tensor): Expert action distributions [T, B, A, 2].
				Required when policy_optimization_method is 'distillation' or 'both',
				or when optimistic_policy_optimization_method is 'distillation'.

		Returns:
			Tuple[float, TensorDict]: Total policy loss and info dict.
		"""
		log_grads = self.cfg.log_gradients_per_loss and self.log_detailed
		method = str(self.cfg.policy_optimization_method).lower()
		
		# Optimistic method: 'same' means use same as pessimistic
		opti_method_cfg = str(self.cfg.optimistic_policy_optimization_method).lower()
		opti_method = method if opti_method_cfg in ('same', 'none', '') else opti_method_cfg
		
		# Validate expert_action_dist is provided when needed
		needs_expert = method in ('distillation', 'both') or (
			self.cfg.dual_policy_enabled and opti_method == 'distillation'
		)
		if needs_expert:
			assert expert_action_dist is not None, \
				f"expert_action_dist required for policy_optimization_method='{method}' or optimistic_policy_optimization_method='{opti_method}'"
		
		# Compute losses based on method (pessimistic policy)
		if method == 'svg':
			# Pure SVG: backprop through world model
			pi_loss, info = self.calc_pi_losses(zs, task, optimistic=False) if (not log_grads or not self.cfg.compile) else self.calc_pi_losses_eager(zs, task)
			
		elif method == 'distillation':
			# Pure distillation: KL to expert targets (pessimistic policy only)
			pi_loss, info = self.calc_pi_distillation_losses(zs, expert_action_dist, task)
			
		elif method == 'both':
			# Combined: SVG + distillation
			svg_loss, svg_info = self.calc_pi_losses(zs, task, optimistic=False) if (not log_grads or not self.cfg.compile) else self.calc_pi_losses_eager(zs, task)
			distill_loss, distill_info = self.calc_pi_distillation_losses(zs, expert_action_dist, task)
			
			pi_loss = svg_loss + distill_loss
			info = svg_info
			# Merge distillation info with prefix
			for k, v in distill_info.items():
				info[f'distill_{k}'] = v
		else:
			raise ValueError(f"Unknown policy_optimization_method: '{method}'. Use 'svg', 'distillation', or 'both'.")
		
		# Optimistic policy loss (if dual policy enabled)
		if self.cfg.dual_policy_enabled:
			if opti_method == 'svg':
				opti_pi_loss, opti_info = self.calc_pi_losses(zs, task, optimistic=True)
			elif opti_method == 'distillation':
				opti_pi_loss, opti_info = self.calc_pi_distillation_losses(zs, expert_action_dist, task, optimistic=True)
			elif opti_method == 'both':
				# Combined: SVG + distillation for optimistic
				opti_svg_loss, opti_svg_info = self.calc_pi_losses(zs, task, optimistic=True)
				opti_distill_loss, opti_distill_info = self.calc_pi_distillation_losses(zs, expert_action_dist, task, optimistic=True)
				opti_pi_loss = opti_svg_loss + opti_distill_loss
				opti_info = opti_svg_info
				for k, v in opti_distill_info.items():
					opti_info[f'distill_{k}'] = v
			else:
				raise ValueError(f"Unknown optimistic_policy_optimization_method: '{opti_method}'. Use 'svg', 'distillation', or 'both'.")
			
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
		
		CORRECT OPTIMISM: For each dynamics head h, compute:
		  σ_h = σ^r_h + γ * σ^v_h  (reward std + discounted value std, if global)
		  TD_h = μ_h + td_target_std_coef × σ_h
		Then reduce over dynamics heads:
		  std_coef > 0: max over H (optimistic)
		  std_coef < 0: min over H (pessimistic)
		  std_coef = 0: mean over H (neutral)
		
		Local bootstrapping: each Ve head bootstraps itself, no value std (σ^v=0).
		Global bootstrapping: all Ve heads get same target, value std computed across Ve.

		Args:
			next_z (Tensor[T, H, B, L]): Latent states at following time steps with H dynamics heads.
			reward (Tensor[T, R, H, B, 1]): Rewards with R reward heads, H dynamics heads.
			terminated (Tensor[T, H, B, 1]): Termination signals (no R dimension).
			task (torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			Tuple[Tensor[Ve, T, B, 1], Tensor[T, B, 1], Tensor[T, B, 1]]: 
				(TD-targets per Ve head, td_mean_log, td_std_log).
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
		

		#TODO WARNING, fixing this and replacing this .item() call somehow breaks the performance of the algo, I do no know why. Someone should fix it for faster compile.
		discount = self.discount[task].unsqueeze(-1) if self.cfg.multitask else self.discount
		discount_scalar = self.discount.mean().item() if self.cfg.multitask else float(self.discount)
		
		std_coef = float(self.cfg.td_target_std_coef)
		
		if self.cfg.local_td_bootstrap:
			# LOCAL: Each Ve head bootstraps itself
			# No value std (each head sees only itself), only reward std across R
			
			# reward: [T, R, H, B, 1] - mean and std across R per (H, T, B)
			r_mean_per_h = reward.mean(dim=1)  # float32[T, H, B, 1]
			r_std_per_h = reward.std(dim=1, unbiased=(R > 1))  # float32[T, H, B, 1]
			
			# v_values: [Ve, T, H, B, 1] - use directly (each Ve head bootstraps itself)
			# TD_h = r_mean_h + γ * (1-term) * v_ve_h  for each (Ve, H)
			# We need to broadcast: terminated [T, H, B, 1]
			terminated_exp = terminated.unsqueeze(0)  # float32[1, T, H, B, 1]
			r_mean_exp = r_mean_per_h.unsqueeze(0)  # float32[1, T, H, B, 1]
			
			# TD mean per (Ve, H): μ_{ve,h} = r_mean_h + γ * (1-term) * v_{ve,h}
			td_mean_per_ve_h = r_mean_exp + discount * (1 - terminated_exp) * v_values  # float32[Ve, T, H, B, 1]
			
			# TD std per H: σ_h = r_std_h (no value std in local mode)
			td_std_per_h = r_std_per_h  # float32[T, H, B, 1]
			
			# TD_h = μ_h + std_coef * σ_h per dynamics head
			# For local, we compute this per (Ve, H), then reduce over H
			# td_std_per_h needs to broadcast to [Ve, T, H, B, 1]
			td_std_exp = td_std_per_h.unsqueeze(0)  # float32[1, T, H, B, 1]
			td_per_ve_h = td_mean_per_ve_h + std_coef * td_std_exp  # float32[Ve, T, H, B, 1]
			
			# Determine dynamics reduction method
			dyn_reduction = self.cfg.td_target_dynamics_reduction
			if dyn_reduction == "from_std_coef":
				dyn_reduction = "max" if std_coef > 0 else ("min" if std_coef < 0 else "mean")
			
			# Reduce over dynamics heads H
			if dyn_reduction == "max":
				td_targets, _ = td_per_ve_h.max(dim=2)  # float32[Ve, T, B, 1]
			elif dyn_reduction == "min":
				td_targets, _ = td_per_ve_h.min(dim=2)  # float32[Ve, T, B, 1]
			else:  # "mean"
				td_targets = td_per_ve_h.mean(dim=2)  # float32[Ve, T, B, 1]
			
			# For logging
			td_mean_log = td_mean_per_ve_h.mean(dim=(0, 2))  # float32[T, B, 1]
			td_std_log = td_std_per_h.mean(dim=1)  # float32[T, B, 1]
			
		else:
			# GLOBAL: All Ve heads get same target
			# Compute reward std across R, value std across Ve, per dynamics head H
			
			# reward: [T, R, H, B, 1] - mean and std across R per (H, T, B)
			r_mean_per_h = reward.mean(dim=1)  # float32[T, H, B, 1]
			r_std_per_h = reward.std(dim=1, unbiased=(R > 1))  # float32[T, H, B, 1]
			
			# v_values: [Ve, T, H, B, 1] - mean and std across Ve per (H, T, B)
			v_mean_per_h = v_values.mean(dim=0)  # float32[T, H, B, 1]
			v_std_per_h = v_values.std(dim=0, unbiased=(Ve > 1))  # float32[T, H, B, 1]
			
			# TD mean per H: μ_h = r_mean_h + γ * (1-term) * v_mean_h
			td_mean_per_h = r_mean_per_h + discount * (1 - terminated) * v_mean_per_h  # float32[T, H, B, 1]
			
			# TD std per H: σ_h = r_std_h + γ * v_std_h
			td_std_per_h = r_std_per_h + discount_scalar * v_std_per_h  # float32[T, H, B, 1]
			
			# TD_h = μ_h + std_coef * σ_h per dynamics head
			td_per_h = td_mean_per_h + std_coef * td_std_per_h  # float32[T, H, B, 1]
			
			# Determine dynamics reduction method
			dyn_reduction = self.cfg.td_target_dynamics_reduction
			if dyn_reduction == "from_std_coef":
				dyn_reduction = "max" if std_coef > 0 else ("min" if std_coef < 0 else "mean")
			
			# Reduce over dynamics heads H
			if dyn_reduction == "max":
				td_reduced, _ = td_per_h.max(dim=1)  # float32[T, B, 1]
			elif dyn_reduction == "min":
				td_reduced, _ = td_per_h.min(dim=1)  # float32[T, B, 1]
			else:  # "mean"
				td_reduced = td_per_h.mean(dim=1)  # float32[T, B, 1]
			
			# All Ve heads get same target
			td_targets = td_reduced.unsqueeze(0).expand(Ve, T, B, 1)  # float32[Ve, T, B, 1]
			
			# For logging
			td_mean_log = td_mean_per_h.mean(dim=1)  # float32[T, B, 1]
			td_std_log = td_std_per_h.mean(dim=1)  # float32[T, B, 1]
		
		return td_targets, td_mean_log, td_std_log
  

	@torch.no_grad()
	def _td_target_aux(self, next_z, reward, terminated, task):
		"""
		Compute auxiliary multi-gamma TD targets using correct optimism.
		
		CORRECT OPTIMISM: For each dynamics head h, compute:
		  σ_h = σ^r_h  (reward std only, aux values have no Ve ensemble)
		  TD_h = μ_h + td_target_std_coef × σ_h
		Then reduce over dynamics heads H based on sign of std_coef.

		Args:
			next_z (Tensor[T, H, B, L]): Latent states at following time steps with H dynamics heads.
			reward (Tensor[T, R, H, B, 1]): Rewards with R reward heads, H dynamics heads.
			terminated (Tensor[T, H, B, 1]): Termination signals (no R dimension).
			task (torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			Tensor[G_aux, T, B, 1]: TD targets per auxiliary gamma.
		"""
		G_aux = len(self._all_gammas) - 1
		if G_aux <= 0:
			return None

		T, H, B, L = next_z.shape  # next_z: float32[T, H, B, L]
		R = reward.shape[1]  # reward: float32[T, R, H, B, 1]
		
		# Merge H and B for V_aux evaluation: [T, H, B, L] -> [T, H*B, L]
		next_z_flat = next_z.view(T, H * B, L)  # float32[T, H*B, L]
		
		# Evaluate auxiliary V on next states using target network
		# V_aux returns (G_aux, T, H*B, 1) for scalar values (mean over 2 outputs per gamma)
		v_values_flat = self.model.V_aux(next_z_flat, task, return_type='mean', target=True)  # float32[G_aux, T, H*B, 1]
		
		# Reshape back to [G_aux, T, H, B, 1]
		v_values = v_values_flat.view(G_aux, T, H, B, 1)  # float32[G_aux, T, H, B, 1]
		
		# gammas_aux: auxiliary discount factors (e.g., 0.9, 0.99)
		gammas_aux = torch.tensor(self._all_gammas[1:], device=next_z.device, dtype=next_z.dtype)  # float32[G_aux]
		
		std_coef = float(self.cfg.td_target_std_coef)
		
		# CORRECT OPTIMISM: Per dynamics head h
		# reward: [T, R, H, B, 1] - mean and std across R per (H, T, B)
		r_mean_per_h = reward.mean(dim=1)  # float32[T, H, B, 1]
		r_std_per_h = reward.std(dim=1, unbiased=(R > 1))  # float32[T, H, B, 1]
		
		# v_values: [G_aux, T, H, B, 1] - already mean over 2 outputs per gamma
		# No value std for aux (single value per gamma)
		
		# TD mean per (G_aux, H): μ_{g,h} = r_mean_h + γ_g * (1-term) * v_{g,h}
		# gammas_aux: [G_aux] -> [G_aux, 1, 1, 1] for broadcasting
		gammas_exp = gammas_aux.view(G_aux, 1, 1, 1)  # float32[G_aux, 1, 1, 1]
		r_mean_exp = r_mean_per_h.unsqueeze(0)  # float32[1, T, H, B, 1]
		terminated_exp = terminated.unsqueeze(0)  # float32[1, T, H, B, 1]
		
		td_mean_per_g_h = r_mean_exp + gammas_exp.unsqueeze(-1) * (1 - terminated_exp) * v_values  # float32[G_aux, T, H, B, 1]
		
		# TD std per H: σ_h = r_std_h (no value std for aux)
		td_std_per_h = r_std_per_h  # float32[T, H, B, 1]
		td_std_exp = td_std_per_h.unsqueeze(0)  # float32[1, T, H, B, 1] - broadcast to G_aux
		
		# TD_{g,h} = μ_{g,h} + std_coef * σ_h
		td_per_g_h = td_mean_per_g_h + std_coef * td_std_exp  # float32[G_aux, T, H, B, 1]
		
		# Determine dynamics reduction method
		dyn_reduction = self.cfg.td_target_dynamics_reduction
		if dyn_reduction == "from_std_coef":
			dyn_reduction = "max" if std_coef > 0 else ("min" if std_coef < 0 else "mean")
		
		# Reduce over dynamics heads H
		if dyn_reduction == "max":
			td_targets_aux, _ = td_per_g_h.max(dim=2)  # float32[G_aux, T, B, 1]
		elif dyn_reduction == "min":
			td_targets_aux, _ = td_per_g_h.min(dim=2)  # float32[G_aux, T, B, 1]
		else:  # "mean"
			td_targets_aux = td_per_g_h.mean(dim=2)  # float32[G_aux, T, B, 1]
		
		return td_targets_aux


	def world_model_losses(self, z_true, z_target, action, reward, terminated, task=None):
		"""Compute world-model losses (consistency, reward, termination).
		
		Args:
			z_true: Encoded latents with gradients [T+1, B, L]
			z_target: Stable latents for consistency targets (eval mode, no dropout) [T+1, B, L], 
			          or None if encoder_dropout == 0
			action: Actions [T, B, A]
			reward: Rewards [T, B, 1]
			terminated: Termination flags [T, B, 1]
			task: Optional task indices
		"""
		T, B, _ = action.shape
		device = z_true.device
		dtype = z_true.dtype

		rho_pows = torch.pow(self.cfg.rho, torch.arange(T, device=device, dtype=dtype))  # float32[T]

		consistency_losses = torch.zeros(T, device=device, dtype=dtype)
		encoder_consistency_losses = torch.zeros(T, device=device, dtype=dtype)
		
		# Compute latent variance across batch (collapse detection metric)
		# Take first timestep z_true[0]: [B, L], compute variance per dimension, then mean
		# TODO: Could use simplex-wise cross-entropy instead of MSE for consistency loss
		latent_batch_variance = z_true[0].var(dim=0).mean()  # float32[]
		
		# Use z_target for consistency targets if available (when encoder has dropout),
		# otherwise fall back to z_true
		z_consistency_target = z_target if z_target is not None else z_true

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
				# Use stable targets (z_target) for consistency loss when encoder has dropout
				target_TBL = z_consistency_target[1:].unsqueeze(0)  # [1,T,B,L]
				delta = pred_TBL - target_TBL.detach()
				delta_enc = pred_TBL.detach() - z_consistency_target[1:].unsqueeze(0)
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

		# Apply warmup: disable encoder consistency for first X% of training
		warmup_ratio = getattr(self.cfg, 'encoder_consistency_warmup_ratio', 0.0)
		warmup_steps = int(warmup_ratio * self.cfg.steps)
		enc_consistency_warmup_scale = 0.0 if self._step < warmup_steps else 1.0

		wm_total = (
			self.cfg.consistency_coef * consistency_loss
			+ self.cfg.encoder_consistency_coef * enc_consistency_warmup_scale * encoder_consistency_loss
			+ self.cfg.reward_coef * reward_loss
			+ self.cfg.termination_coef * termination_loss
		)

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
		
		# Determine head_mode based on td_target_use_all_dynamics_heads
		# When false, use a randomly-selected dynamics head (cheaper but no H-std)
		# When true, use all heads for full (R × H × Ve) combinations
		if self.cfg.td_target_use_all_dynamics_heads:
			head_mode = 'all'
			H = int(self.cfg.planner_num_dynamics_heads)
		else:
			head_mode = 'random'  # Use single randomly-selected head
			H = 1  # Only one head in output
		
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
				head_mode=head_mode,  # 'all' or 'random' based on config
				task=task,
			)
		# latents: float32[H, B_total, N, T+1, L]; actions: float32[B_total, N, T, A]
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
			z_seq_out = torch.cat([z_seq[:1], z_seq[1:].detach()], dim=0).clone()  # float32[T+1, H, B, L]

		return {
			'z_seq': z_seq_out,  # float32[T+1, H, B, L]
			'actions': actions_seq.detach(),  # float32[T, 1, B, A] (shared across heads)
			'rewards': rewards.detach(),  # float32[T, R, H, B, 1]
			'terminated': terminated.detach(),  # float32[T, H, B, 1]
			'termination_logits': term_logits.detach(),  # float32[T, H, B, 1]
		}



	def calculate_value_loss(self, z_seq, actions, rewards, terminated, full_detach, task=None):
		"""Compute primary critic loss on imagined latent sequences with proper rho weighting.
		
		With V-function, we predict V(z) for each state in the sequence and
		train against TD targets r + γ * V(next_z). Value predictions use head 0
		only (all heads are identical before dynamics rollout). TD targets use all
		heads for next-state diversity, then reduce over H.
		
		Rho weighting is applied on the S dimension (replay buffer time steps that
		were used as starting states for imagination), NOT on T_imag (imagination steps).

		Args:
			z_seq (Tensor[T_imag+1, H, S, B, N, L]): Latent sequences where:
				- T_imag = imagination horizon (typically 1)
				- H = num dynamics heads
				- S = replay buffer horizon + 1 (starting states from different replay steps)
				- B = batch size
				- N = num_rollouts per starting state
				- L = latent dimension
			actions (Tensor[T_imag, 1, S, B, N, A]): Actions (shared across heads).
			rewards (Tensor[T_imag, R, H, S, B, N, 1]): Rewards with R reward heads, H dynamics heads.
			terminated (Tensor[T_imag, H, S, B, N, 1]): Termination signals per head.
			full_detach (bool): Whether to detach z_seq from graph.
			task: Task index for multitask.

		Returns:
			Tuple[Tensor, TensorDict]: Scalar loss and info dict.
		"""
		# z_seq: [T_imag+1, H, S, B, N, L]
		T_imag_plus_1, H, S, B, N, L = z_seq.shape
		T_imag = T_imag_plus_1 - 1
		K = self.cfg.num_bins
		device = z_seq.device
		dtype = z_seq.dtype

		# Rho weighting on S (replay buffer time steps), NOT T_imag
		rho_pows = torch.pow(self.cfg.rho, torch.arange(S, device=device, dtype=dtype))  # float32[S]

		z_seq = z_seq.detach() if full_detach else z_seq  # float32[T_imag+1, H, S, B, N, L]
		rewards = rewards.detach()  # float32[T_imag, R, H, S, B, N, 1]
		terminated = terminated.detach()  # float32[T_imag, H, S, B, N, 1]

		# Merge B*N for V prediction, keep S separate for rho weighting
		BN = B * N

		# V-function: use head 0 only for value predictions (all heads identical at t=0)
		# z_seq[:-1, 0]: [T_imag, S, B, N, L] -> merge to [T_imag, S, BN, L]
		z_for_v = z_seq[:-1, 0]  # float32[T_imag, S, B, N, L]
		z_for_v_merged = z_for_v.view(T_imag, S, BN, L)  # float32[T_imag, S, BN, L]
		z_for_v_flat = z_for_v_merged.view(T_imag * S * BN, L)  # flatten for V call
		
		vs_flat = self.model.V(z_for_v_flat, task, return_type='all')  # float32[Ve, T_imag*S*BN, K]
		Ve = vs_flat.shape[0]
		vs = vs_flat.view(Ve, T_imag, S, BN, K)  # float32[Ve, T_imag, S, BN, K]

		with maybe_range('Value/td_target', self.cfg):
			with torch.no_grad():
				# Merge S*B*N into batch for _td_target, then split back
				# _td_target expects: next_z [T, H, Batch, L], rewards [T, R, H, Batch, 1], etc.
				next_z = z_seq[1:]  # [T_imag, H, S, B, N, L]
				next_z_flat = next_z.view(T_imag, H, S * BN, L)  # [T_imag, H, S*BN, L]
				rewards_flat = rewards.view(T_imag, -1, H, S * BN, 1)  # [T_imag, R, H, S*BN, 1]
				R = rewards_flat.shape[1]
				terminated_flat = terminated.view(T_imag, H, S * BN, 1)  # [T_imag, H, S*BN, 1]
				
				td_targets_flat, td_mean_flat, td_std_flat = self._td_target(next_z_flat, rewards_flat, terminated_flat, task)
				# td_targets_flat: [Ve, T_imag, S*BN, 1] - reduced targets (mean + coef×std)
				# td_mean_flat: [T_imag, S*BN, 1] - mean across all Ve×R×H combinations
				# td_std_flat: [T_imag, S*BN, 1] - std across all Ve×R×H combinations
				
				# Split back: [Ve, T_imag, S*BN, 1] -> [Ve, T_imag, S, BN, 1]
				td_targets = td_targets_flat.view(Ve, T_imag, S, BN, 1)
				td_mean = td_mean_flat.view(T_imag, S, BN, 1)  # float32[T_imag, S, BN, 1]
				td_std = td_std_flat.view(T_imag, S, BN, 1)    # float32[T_imag, S, BN, 1]

		with maybe_range('Value/ce', self.cfg):
			# TD targets: [Ve, T_imag, S, BN, 1], vs: [Ve, T_imag, S, BN, K]
			# Flatten for soft_ce
			vs_flat_ce = vs.contiguous().view(Ve * T_imag * S * BN, K)
			td_flat_ce = td_targets.contiguous().view(Ve * T_imag * S * BN, 1)
			
			val_ce_flat = math.soft_ce(vs_flat_ce, td_flat_ce, self.cfg)  # float32[Ve*T_imag*S*BN]
			val_ce = val_ce_flat.view(Ve, T_imag, S, BN)  # float32[Ve, T_imag, S, BN]

		# Average over Ve, T_imag, BN; keep S for rho weighting
		val_ce_per_s = val_ce.mean(dim=(0, 1, 3))  # float32[S]
		
		# Apply rho weighting on S dimension
		weighted = val_ce_per_s * rho_pows  # float32[S]
		loss = weighted.mean()

		info = TensorDict({
			'value_loss': loss
		}, device=device, non_blocking=True)

		# Log per replay step (S dimension)
		for s in range(S):
			info.update({f'value_loss/replay_step{s}': val_ce_per_s[s]}, non_blocking=True)
		
		value_pred = math.two_hot_inv(vs, self.cfg)  # float32[Ve, T_imag, S, BN, 1]
		
		# Log td_target statistics (mean and std across all Ve×R×H combinations)
		info.update({
			'td_target_mean': td_mean.mean(),
			'td_target_std': td_std.mean(),
		}, non_blocking=True)
		
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
			
			# Std across num_rollouts (N dimension) - diagnostic for rollout diversity
			# Need to reshape back to include N: [Ve, T_imag, S, B, N, 1]
			td_targets_with_n = td_targets.view(Ve, T_imag, S, B, N, 1)
			td_std_across_rollouts = td_targets_with_n.std(dim=4).mean()  # std across N (different actions)
			info.update({'td_target_std_across_rollouts': td_std_across_rollouts}, non_blocking=True)
   
		# Value error per replay step (S dimension)
		value_error = value_pred - td_targets  # float32[Ve, T_imag, S, BN, 1]
		for s in range(S):
			info.update({
				f'value_error_abs_mean/replay_step{s}': value_error[:, :, s].abs().mean(),
				f'value_error_std/replay_step{s}': value_error[:, :, s].std(),
				f'value_error_max/replay_step{s}': value_error[:, :, s].abs().max(),
			}, non_blocking=True)

		return loss, info

	def calculate_aux_value_loss(self, z_seq, actions, rewards, terminated, full_detach, task=None):
		"""Compute auxiliary multi-gamma critic losses with proper rho weighting on S dimension.
		
		With V-function, we predict V_aux(z) for each state and auxiliary gamma.
		For multi-head inputs, TD targets are computed per-head then reduced.
		
		Rho weighting is applied on the S dimension (replay buffer time steps that
		were used as starting states for imagination), NOT on T_imag.

		Args:
			z_seq (Tensor[T_imag+1, H, S, B, N, L]): Latent sequences where:
				- T_imag = imagination horizon (typically 1)
				- H = num dynamics heads
				- S = replay buffer horizon + 1 (starting states from different replay steps)
				- B = batch size
				- N = num_rollouts per starting state
				- L = latent dimension
			actions (Tensor[T_imag, 1, S, B, N, A]): Actions (shared across heads).
			rewards (Tensor[T_imag, R, H, S, B, N, 1]): Rewards with R reward heads, H dynamics heads.
			terminated (Tensor[T_imag, H, S, B, N, 1]): Termination signals (no R dimension).
			full_detach (bool): Whether to detach z_seq from graph.
			task: Task index for multitask.

		Returns:
			Tuple[Tensor, TensorDict]: Scalar loss and info dict.
		"""
		if self.model._num_aux_gamma == 0:
			return torch.zeros((), device=z_seq.device), TensorDict({}, device=z_seq.device)

		# z_seq: [T_imag+1, H, S, B, N, L]
		T_imag_plus_1, H, S, B, N, L = z_seq.shape
		T_imag = T_imag_plus_1 - 1
		K = self.cfg.num_bins
		device = z_seq.device
		dtype = z_seq.dtype

		# Rho weighting on S (replay buffer time steps), NOT T_imag
		rho_pows = torch.pow(self.cfg.rho, torch.arange(S, device=device, dtype=dtype))  # float32[S]

		z_seq = z_seq.detach() if full_detach else z_seq  # float32[T_imag+1, H, S, B, N, L]
		rewards = rewards.detach()  # float32[T_imag, R, H, S, B, N, 1]
		terminated = terminated.detach()  # float32[T_imag, H, S, B, N, 1]

		# Merge B*N for V_aux prediction, keep S separate for rho weighting
		BN = B * N

		# V_aux: use head 0 only for value predictions (all heads identical at t=0)
		z_for_v = z_seq[:-1, 0]  # float32[T_imag, S, B, N, L]
		z_for_v_merged = z_for_v.view(T_imag, S, BN, L)  # float32[T_imag, S, BN, L]
		z_for_v_flat = z_for_v_merged.view(T_imag * S * BN, L)  # flatten for V_aux call
		
		v_aux_logits_flat = self.model.V_aux(z_for_v_flat, task, return_type='all')  # float32[G_aux, T_imag*S*BN, K] or None
		if v_aux_logits_flat is None:
			return torch.zeros((), device=device), TensorDict({}, device=device)

		G_aux = v_aux_logits_flat.shape[0]
		v_aux_logits = v_aux_logits_flat.view(G_aux, T_imag, S, BN, K)  # float32[G_aux, T_imag, S, BN, K]

		with maybe_range('Aux/td_target', self.cfg):
			with torch.no_grad():
				# Merge S*B*N into batch for _td_target_aux, then split back
				next_z = z_seq[1:]  # [T_imag, H, S, B, N, L]
				next_z_flat = next_z.view(T_imag, H, S * BN, L)  # [T_imag, H, S*BN, L]
				rewards_flat = rewards.view(T_imag, -1, H, S * BN, 1)  # [T_imag, R, H, S*BN, 1]
				terminated_flat = terminated.view(T_imag, H, S * BN, 1)  # [T_imag, H, S*BN, 1]
				
				aux_td_targets_flat = self._td_target_aux(next_z_flat, rewards_flat, terminated_flat, task)
				# aux_td_targets_flat: [G_aux, T_imag, S*BN, 1]
				
				# Split back: [G_aux, T_imag, S*BN, 1] -> [G_aux, T_imag, S, BN, 1]
				aux_td_targets = aux_td_targets_flat.view(G_aux, T_imag, S, BN, 1)

		with maybe_range('Aux/ce', self.cfg):
			# TD targets: [G_aux, T_imag, S, BN, 1], v_aux_logits: [G_aux, T_imag, S, BN, K]
			# Flatten for soft_ce
			vaux_flat = v_aux_logits.contiguous().view(G_aux * T_imag * S * BN, K)
			aux_targets_flat = aux_td_targets.contiguous().view(G_aux * T_imag * S * BN, 1)
			
			aux_ce_flat = math.soft_ce(vaux_flat, aux_targets_flat, self.cfg)  # float32[G_aux*T_imag*S*BN]
			aux_ce = aux_ce_flat.view(G_aux, T_imag, S, BN)  # float32[G_aux, T_imag, S, BN]

		# Average over T_imag, BN; keep G_aux and S for rho weighting per gamma
		aux_ce_per_gamma_s = aux_ce.mean(dim=(1, 3))  # float32[G_aux, S]
		
		# Apply rho weighting on S dimension
		weighted = aux_ce_per_gamma_s * rho_pows.unsqueeze(0)  # float32[G_aux, S]
		losses = weighted.mean(dim=1)  # float32[G_aux] - mean over S per gamma
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

		def encode_obs(obs_seq, grad_enabled, eval_mode=False):
			"""Encode observations with optional eval mode for stable targets.
			
			Args:
				obs_seq: Observation sequence [steps, batch, ...]
				grad_enabled: Whether to enable gradients
				eval_mode: If True, encode with model in eval mode (disables dropout)
			
			Returns:
				Encoded latents [steps, batch, L]
			"""
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
					if eval_mode:
						# Temporarily set encoder to eval mode for stable targets
						was_training = self.model._encoder.training
						self.model._encoder.eval()
						latents_flat = self.model.encode(flat_obs, task_flat)  # float32[steps*batch,L]
						if was_training:
							self.model._encoder.train()
					else:
						latents_flat = self.model.encode(flat_obs, task_flat)  # float32[steps*batch,L]
			return latents_flat.view(steps, batch, *latents_flat.shape[1:])  # float32[steps,batch,L]

		# Encode observations (needed for value computation even when not updating WM)
		z_true = encode_obs(obs, grad_enabled=True, eval_mode=False)
		# When encoder has dropout, encode again in eval mode for stable consistency targets
		z_target = encode_obs(obs, grad_enabled=False, eval_mode=True) if self.cfg.encoder_dropout > 0 else None
		
		# Compute WM losses if updating world model, otherwise zero loss and skip rollout
		if update_world_model:
			wm_loss, wm_info, z_rollout, lat_all = self.world_model_losses(z_true, z_target, action, reward, terminated, task)
		else:
			# Skip WM losses: no rollout available, empty info (don't log zeros)
			wm_loss = torch.zeros((), device=device)
			wm_info = TensorDict({}, device=device)  # Empty - don't log anything for WM
			z_rollout = None  # No rollout available
			lat_all = None  # No multi-head rollout available

		# Only compute value losses if update_value is True
		# Always use imagination for value/aux losses (ac_source='imagine' hardcoded)
		if update_value:
			# Imagination starts from encoded/rollout latents and rolls forward with policy.
			# imagine_initial_source controls starting point:
			#   'replay_true': start from true encoded latents z_true
			#   'replay_rollout': start from dynamics rollout z_rollout (head 0)
			# 
			# Gradient flow logic:
			#   - If update_world_model=False: fully detach start_z (no encoder gradients from value)
			#   - If update_world_model=True and replay_true: keep attached (encoder gradients flow)
			#   - If update_world_model=True and replay_rollout: partial detach (z[0] attached, z[1:] detached)
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
			
			# start_z: [S, B, L] where S = replay buffer horizon + 1
			S, B_orig, L = start_z.shape
			A = self.cfg.action_dim
			N = int(self.cfg.num_rollouts)
			R = int(self.cfg.num_reward_heads)
			T_imag = int(self.cfg.imagination_horizon)
			
			# imagined_rollout flattens S*B into batch, uses N rollouts per state
			# Returns tensors with B_expanded = S * B_orig * N
			# H depends on td_target_use_all_dynamics_heads: 1 if False, planner_num_dynamics_heads if True
			imagined = self.imagined_rollout(start_z, task=task, rollout_len=T_imag)
			
			# Get actual H from returned tensor (may be 1 if td_target_use_all_dynamics_heads=False)
			H = imagined['z_seq'].shape[1]
			
			# Reshape outputs from [*, B_expanded, *] to [*, S, B, N, *]
			# z_seq: [T_imag+1, H, S*B*N, L] -> [T_imag+1, H, S, B, N, L]
			z_seq = imagined['z_seq'].view(T_imag + 1, H, S, B_orig, N, L)
			# rewards: [T_imag, R, H, S*B*N, 1] -> [T_imag, R, H, S, B, N, 1]
			rewards = imagined['rewards'].view(T_imag, R, H, S, B_orig, N, 1)
			# terminated: [T_imag, H, S*B*N, 1] -> [T_imag, H, S, B, N, 1]
			terminated_imag = imagined['terminated'].view(T_imag, H, S, B_orig, N, 1)
			# actions: [T_imag, 1, S*B*N, A] -> [T_imag, 1, S, B, N, A]
			actions = imagined['actions'].view(T_imag, 1, S, B_orig, N, A)
			
			full_detach = (not update_value) or self.cfg.detach_imagine_value
			
			value_loss, value_info = self.calculate_value_loss(
				z_seq,
				actions,
				rewards,
				terminated_imag,
				full_detach,
				task=task,
			)

			aux_loss = torch.zeros((), device=device)
			aux_info = TensorDict({}, device=device)
			if self.cfg.multi_gamma_loss_weight != 0 and self.model._num_aux_gamma > 0:
				aux_loss, aux_info = self.calculate_aux_value_loss(
					z_seq,
					actions,
					rewards,
					terminated_imag,
					full_detach,
					task=task,
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
  
		critic_weighted = self.cfg.value_coef * value_loss
		# Always apply imagine_value_loss_coef_mult since ac_source is always 'imagine'
		critic_weighted = critic_weighted * self.cfg.imagine_value_loss_coef_mult
		aux_weighted = self.cfg.multi_gamma_loss_weight * aux_loss
		aux_weighted = aux_weighted * self.cfg.imagine_value_loss_coef_mult

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
		}

	def _update(self, obs, action, reward, terminated, expert_action_dist=None, update_value=True, update_pi=True, update_world_model=True, task=None):
		"""Single gradient update step over world model, critic, and policy.
		
		Args:
			expert_action_dist: Expert action distributions [T, B, A, 2] for distillation.
				Required when policy_optimization_method is 'distillation' or 'both'.
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
				# Policy trains on dynamics rollout states (z_rollout): [T+1, B, L]
				# These are latents from: z[0] = encoder output, z[1:] = dynamics predictions.
				# Using rollout states exposes policy to dynamics model predictions,
				# which is important for planning coherence during inference.
				# Fallback to z_true if z_rollout unavailable (update_world_model=False).
				z_for_pi = z_rollout.detach() if z_rollout is not None else z_true.detach()  # float32[T+1, B, L]
    
				pi_loss, pi_info = self.update_pi(z_for_pi, task, expert_action_dist=expert_action_dist)
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
			obs, action, reward, terminated, task, expert_action_dist, indices = buffer.sample()

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

		# Lazy reanalyze: BEFORE _update so fresh targets are used in current update
		# Re-runs the planner on first-timestep observations from the sampled batch
		# and updates both the local tensor and the buffer in-place.
		reanalyze_log_pending = None
		reanalyze_interval = int(getattr(self.cfg, 'reanalyze_interval', 0))
		# Check both interval AND that we haven't already reanalyzed at this step
		# (prevents multiple reanalyze calls per env step when utd_ratio > 1)
		should_reanalyze = (
			reanalyze_interval > 0
			and self._step % reanalyze_interval == 0
			and self._step > 0
			and self._step != self._last_reanalyze_step
		)
		if should_reanalyze:
			self._last_reanalyze_step = self._step
			with maybe_range('update/lazy_reanalyze', self.cfg):
				# Get first-timestep observations and indices
				# obs: [T+1, B, *obs_shape], indices: [T+1, B]
				# Buffer stores (obs_{t+1}, action_t, expert_dist_t) together, so:
				# - obs[0] is o₀ (first observation)
				# - indices[1] is where expert_dist for o₀ is stored (not indices[0], which has NaN)
				reanalyze_batch_size = int(getattr(self.cfg, 'reanalyze_batch_size', obs.shape[1]))
				reanalyze_batch_size = min(reanalyze_batch_size, obs.shape[1])
				
				# Take first timestep obs, but indices[1] (where expert_dist for obs[0] is stored)
				obs_reanalyze = obs[0, :reanalyze_batch_size]  # float32[B_re, *obs_shape]
				indices_reanalyze = indices[1, :reanalyze_batch_size]  # [B_re] - use index 1, not 0!
				
				# Get old expert distributions before reanalyze (for KL logging)
				# expert_action_dist: [T+1, B, A, 2] -> [B_re, A, 2]
				old_expert_dist = expert_action_dist[1, :reanalyze_batch_size].clone()  # float32[B_re, A, 2]
				
				# Run reanalyze to get new expert targets
				expert_action_dist_new, expert_value_new, reanalyze_info = self.reanalyze(obs_reanalyze, task=task)
				
				# Update expert_action_dist tensor in-place for this update
				# expert_action_dist: [T+1, B, A, 2], update index 1 (where obs[0] targets live)
				expert_action_dist[1, :reanalyze_batch_size] = expert_action_dist_new
				
				# Update buffer for future samples
				buffer.update_expert_data(indices_reanalyze, expert_action_dist_new, expert_value_new)
				
				# Store logging info for after _update (when we have the info dict)
				reanalyze_log_pending = {
					'reanalyze_info': reanalyze_info,
					'old_expert_dist': old_expert_dist,
					'new_expert_dist': expert_action_dist_new,
				}

		torch.compiler.cudagraph_mark_step_begin()
  
		info = self._update(obs, action, reward, terminated, expert_action_dist=expert_action_dist, update_value=update_value, update_pi=update_pi, update_world_model=update_world_model, **kwargs)

		# Log reanalyze stats (deferred from before _update)
		if reanalyze_log_pending is not None:
			if reanalyze_log_pending['reanalyze_info'] is not None and self._step % self.cfg.log_freq == 0:
				self.logger.log_planner_info(reanalyze_log_pending['reanalyze_info'], step=self._step, prefix="reanalyze")
			
			if self._step % self.cfg.log_freq == 0:
				old_mean, old_std = reanalyze_log_pending['old_expert_dist'][..., 0], reanalyze_log_pending['old_expert_dist'][..., 1]  # [B_re, A]
				new_mean, new_std = reanalyze_log_pending['new_expert_dist'][..., 0], reanalyze_log_pending['new_expert_dist'][..., 1]  # [B_re, A]
				# Clamp stds to avoid numerical issues
				old_std = old_std.clamp(min=1e-6)
				new_std = new_std.clamp(min=1e-6)
				# KL(old || new) for Gaussians: mean over actions, mean over batch
				kl_div = (torch.log(new_std / old_std) + (old_std**2 + (old_mean - new_mean)**2) / (2 * new_std**2) - 0.5).mean(dim=-1).mean()
				info['reanalyze/kl_old_to_new'] = kl_div.item()
				# Also log mean/std differences
				info['reanalyze/mean_diff'] = (new_mean - old_mean).abs().mean().item()
				info['reanalyze/std_diff'] = (new_std - old_std).abs().mean().item()

		# Log current encoder LR for W&B tracking
		info['encoder_lr'] = self.optim.param_groups[0]['lr']

		# KNN entropy logging (sparse, only when log_detailed)
		if self.log_detailed and self._knn_encoder is not None:
			with torch.no_grad():
				# obs: [T+1, B, obs_dim], take first timestep
				obs_flat = obs[0]  # float32[B, obs_dim]
				# Encode with frozen random encoder
				encoded = self._knn_encoder(obs_flat)  # float32[B, knn_entropy_dim]
				# Compute KNN entropy
				knn_k = int(getattr(self.cfg, 'knn_entropy_k', 5))
				batch_knn_entropy = math.compute_knn_entropy(encoded, k=knn_k)
				info['batch_knn_entropy'] = batch_knn_entropy.item()

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
				obs, action, reward, terminated, task, expert_action_dist, indices = buffer.sample()
				with maybe_range('Agent/validate', self.cfg):
					self.log_detailed = True
					components = self._compute_loss_components(obs, action, reward, terminated, task, update_value=True, log_grads=False)
					val_info = components['info']

					# Check if we can compute policy loss (skip if distillation and no expert data)
					method = str(self.cfg.policy_optimization_method).lower()
					opti_method_cfg = str(self.cfg.optimistic_policy_optimization_method).lower()
					opti_method = method if opti_method_cfg in ('same', 'none', '') else opti_method_cfg
					needs_expert = method in ('distillation', 'both') or (
						self.cfg.dual_policy_enabled and opti_method in ('distillation', 'both')
					)
					
					# Skip policy loss if distillation is needed but expert data not available
					if needs_expert and (expert_action_dist is None or torch.isnan(expert_action_dist).any()):
						# Cannot compute policy loss without expert data - skip
						pass
					else:
						# Policy evaluates on dynamics rollout states (replay_rollout hardcoded)
						# Fallback to z_true if z_rollout unavailable
						z_for_pi = components['z_rollout'].detach() if components['z_rollout'] is not None else components['z_true'].detach()  # float32[T+1, B, L]
	      
						pi_loss, pi_info = self.update_pi(z_for_pi, task, expert_action_dist=expert_action_dist)
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
