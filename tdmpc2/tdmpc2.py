import torch
import torch.nn.functional as F


from common import math
from common.nvtx_utils import maybe_range
from common.scale import RunningScale
from common.world_model import WorldModel
from common.layers import api_model_conversion
from utils.reset import (
	clear_optimizer_state,
	hard_reset_module,
	shrink_perturb_module,
	sync_auxiliary_detach,
)
from tensordict import TensorDict
from common.logging_utils import get_logger

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
			{'params': self.model._dynamics.parameters()},
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
		# Buffer for MPPI warm-start action mean; shape (T, A)
		self.register_buffer(
			"_prev_mean",
			torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device),
			persistent=False,   # don’t save to checkpoints unless you want to
		)
		if cfg.compile:
			log.info('Compiling update function with torch.compile...')
			self._update_eager = self._update
			self._update = torch.compile(self._update, mode=self.cfg.compile_type)

			self._compute_loss_components_eager = self._compute_loss_components
			self._compute_loss_components = torch.compile(self._compute_loss_components, mode=self.cfg.compile_type, fullgraph=True)
			self.calc_pi_losses_eager = self.calc_pi_losses
			self.calc_pi_losses = torch.compile(self.calc_pi_losses, mode=self.cfg.compile_type, fullgraph=True)

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
			self._calculate_loss_components_eager = self._compute_loss_components
			self.calc_pi_losses_eager = self.calc_pi_losses
			self.optim_step = self.optim.step
			self.pi_optim_step = self.pi_optim.step

   

	def reset_planner_state(self):
		self._prev_mean.zero_() 

	def reset_agent(self):
		log.info('===== Agent reset start =====')
		cfg_reset = self.cfg.reset
		fallbacks = cfg_reset["fallbacks"]
		default_alpha = float(fallbacks["shrink_alpha"])
		default_noise = float(fallbacks["shrink_noise_std"])

		def _resolve_type(section):
			return str(section["type"]).lower()

		def _resolve_layers(section, fallback=-1):
			value = section["layers"]
			if value is None:
				return fallback
			try:
				return int(value)
			except (TypeError, ValueError):
				log.warning('Invalid layers value %s; defaulting to %s', value, fallback)
				return fallback

		def _resolve_alpha(section):
			value = section["alpha"]
			return default_alpha if value is None else float(value)

		def _resolve_noise(section):
			value = section["noise_std"]
			return default_noise if value is None else float(value)

		actor_cfg = cfg_reset["actor_critic"]
		encoder_cfg = cfg_reset["encoder_dynamics"]
		actor_type = _resolve_type(actor_cfg)
		encoder_type = _resolve_type(encoder_cfg)
		if actor_type == 'none' and encoder_type == 'none':
			log.info('No reset actions configured (actor_critic=none, encoder_dynamics=none); skipping.')
			return

		actor_layers = _resolve_layers(actor_cfg)
		encoder_layers = _resolve_layers(encoder_cfg)
		actor_alpha = _resolve_alpha(actor_cfg)
		actor_noise = _resolve_noise(actor_cfg)
		encoder_alpha = _resolve_alpha(encoder_cfg)
		encoder_noise = _resolve_noise(encoder_cfg)

		log.info('Actor-critic reset config: type=%s, layers=%s, alpha=%s, noise_std=%s', actor_type, actor_layers, actor_alpha, actor_noise)
		log.info('Encoder-dynamics reset config: type=%s, layers=%s, alpha=%s, noise_std=%s', encoder_type, encoder_layers, encoder_alpha, encoder_noise)

		policy_params = []
		critic_params = []
		encoder_params = []

		if actor_type == 'full':
			policy_params.extend(hard_reset_module(self.model._pi, actor_layers, module_name='model._pi', logger=log))
			critic_params.extend(hard_reset_module(self.model._Qs, actor_layers, module_name='model._Qs', logger=log))
			if self.model._aux_joint_Qs is not None:
				critic_params.extend(hard_reset_module(self.model._aux_joint_Qs, actor_layers, module_name='model._aux_joint_Qs', logger=log))
			elif self.model._aux_separate_Qs is not None:
				for idx, head in enumerate(self.model._aux_separate_Qs):
					critic_params.extend(hard_reset_module(head, actor_layers, module_name=f'model._aux_separate_Qs[{idx}]', logger=log))
		elif actor_type == 'shrink_perturb':
			policy_params.extend(shrink_perturb_module(self.model._pi, actor_layers, actor_alpha, actor_noise, module_name='model._pi', logger=log))
			critic_params.extend(shrink_perturb_module(self.model._Qs, actor_layers, actor_alpha, actor_noise, module_name='model._Qs', logger=log))
			if self.model._aux_joint_Qs is not None:
				critic_params.extend(shrink_perturb_module(self.model._aux_joint_Qs, actor_layers, actor_alpha, actor_noise, module_name='model._aux_joint_Qs', logger=log))
			elif self.model._aux_separate_Qs is not None:
				for idx, head in enumerate(self.model._aux_separate_Qs):
					critic_params.extend(shrink_perturb_module(head, actor_layers, actor_alpha, actor_noise, module_name=f'model._aux_separate_Qs[{idx}]', logger=log))
		else:
			log.info('Actor-critic reset skipped (type=%s)', actor_type)

		if encoder_type == 'full':
			for key, encoder in self.model._encoder.items():
				encoder_params.extend(hard_reset_module(encoder, encoder_layers, module_name=f'model._encoder[{key}]', logger=log))
			encoder_params.extend(hard_reset_module(self.model._dynamics, encoder_layers, module_name='model._dynamics', logger=log))
		elif encoder_type == 'shrink_perturb':
			for key, encoder in self.model._encoder.items():
				encoder_params.extend(shrink_perturb_module(encoder, encoder_layers, encoder_alpha, encoder_noise, module_name=f'model._encoder[{key}]', logger=log))
			encoder_params.extend(shrink_perturb_module(self.model._dynamics, encoder_layers, encoder_alpha, encoder_noise, module_name='model._dynamics', logger=log))
		else:
			log.info('Encoder-dynamics reset skipped (type=%s)', encoder_type)

		touched_policy = len(policy_params)
		touched_model = len(critic_params) + len(encoder_params)
		log.info('Touched %d policy parameters and %d model parameters during reset', touched_policy, touched_model)

		clear_optimizer_state(self.pi_optim, policy_params, 'policy', logger=log)
		clear_optimizer_state(self.optim, list({p for p in critic_params + encoder_params}), 'world_model', logger=log)
		self.model.reset_policy_encoder_targets()

		sync_auxiliary_detach(self.model, logger=log)
		self.scale.reset()
		log.info('RunningScale reset invoked post-parameter reset')
		log.info('===== Agent reset complete =====')

	@property
	def plan(self):
		_plan_val = getattr(self, "_plan_val", None)
		if _plan_val is not None:
			return _plan_val
		if self.cfg.compile:
			# plan = torch.compile(self._plan, mode="reduce-overhead")
			log.info('Compiling planning function with torch.compile...')
			plan = torch.compile(self._plan, mode=self.cfg.compile_type, fullgraph=True)
		else:
			plan = self._plan
		self._plan_val = plan
		return self._plan_val

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
	def act(self, obs, eval_mode=False, task=None, mpc=True):
		"""
		Select an action by planning in latent space (MPPI) or by single policy prior.

		Args:
			obs (torch.Tensor): Observation from environment. Shape (obs_dim,) or already batched.
			t0 (bool): Whether this is the first observation in the episode.
			eval_mode (bool): Whether to use the mean of the action distribution.
			task (int): Task index (only used for multi-task experiments).

		Returns:
			torch.Tensor: Action to take in the environment.
		"""
		self.model.eval()
		with maybe_range('Agent/act', self.cfg):
			
			if task is not None:
				task = torch.tensor([task], device=self.device)
			if mpc:
				# if not eval_mode:
				# a = (a + std * torch.randn(self.cfg.action_dim, device=std.device)).clamp(-1, 1)
				score, elite_actions, mean, std = self.plan(obs, task=task)
							# Select first action from sampled distribution; add exploration noise if training
				if self.cfg.best_eval and eval_mode:
					# Take best action sequence
					idx = torch.argmax(score)
				else:
					idx = math.gumbel_softmax_sample(score.squeeze(1))
				actions = torch.index_select(elite_actions, 1, idx).squeeze(1)
				a, std = actions[0], std[0]
				
				plan_info = TensorDict({
						'score': score,
						'elite_actions': elite_actions,
						'mean': mean,
						'std': std
					}, device=std.device, non_blocking=True)
    
				self.update_planner_mean(mean)
				if eval_mode:

					return a, plan_info # TODO not bad idea to perhaps take mean of elites here instead. (this is argmax, used to be a sample)
				else:
					return (a + std * torch.randn(self.cfg.action_dim, device=std.device)).clamp(-1, 1), plan_info
			z = self.model.encode(obs, task)
			action, info = self.model.pi(z, task, use_ema=self.cfg.policy_ema_enabled)
			if eval_mode:
				action = info["mean"]
			return action[0], TensorDict({}, device=action.device, non_blocking=True)

	@torch.no_grad()
	def _estimate_value(self, z, actions, task):
		"""Estimate value of a candidate action sequence.

		Args:
			z (B,L) latent start (B=num_samples)
			actions (T,B,A) sampled candidate actions
			task: optional task index
		Returns:
			(B,1) scalar value estimates
		"""
		with maybe_range('Agent/estimate_value', self.cfg):
			G, discount = 0, 1
			termination = torch.zeros(self.cfg.num_samples, 1, dtype=torch.float32, device=z.device)
			for t in range(self.cfg.horizon):
				reward = math.two_hot_inv(self.model.reward(z, actions[t], task), self.cfg)
				z = self.model.next(z, actions[t], task)
				G = G + discount * (1-termination) * reward
				discount_update = self.discount[torch.tensor(task)] if self.cfg.multitask else self.discount
				discount = discount * discount_update
				if self.cfg.episodic:
					termination = torch.clip(termination + (self.model.termination(z, task) > 0.5).float(), max=1.)
			action, _ = self.model.pi(z, task, use_ema=self.cfg.policy_ema_enabled)
			return G + discount * (1-termination) * self.model.Q(z, action, task, return_type='avg')

	@torch.no_grad()
	def _plan(self, obs, task=None):
		"""
		Plan a sequence of actions using the learned world model.

		Args:
			z (torch.Tensor): Latent state from which to plan.
			t0 (bool): Whether this is the first observation in the episode.
			eval_mode (bool): Whether to use the mean of the action distribution.
			task (Torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			torch.Tensor: Action to take in the environment.
		"""
		# Sample policy trajectories: roll forward policy prior to seed action set
		with maybe_range('Agent/plan', self.cfg):
			z = self.model.encode(obs, task)
			if self.cfg.num_pi_trajs > 0:
				pi_actions = torch.empty(self.cfg.horizon, self.cfg.num_pi_trajs, self.cfg.action_dim, device=self.device)
				_z = z.repeat(self.cfg.num_pi_trajs, 1)
				for t in range(self.cfg.horizon-1):
					pi_actions[t], _ = self.model.pi(_z, task, use_ema=self.cfg.policy_ema_enabled)
					_z = self.model.next(_z, pi_actions[t], task)
				pi_actions[-1], _ = self.model.pi(_z, task, use_ema=self.cfg.policy_ema_enabled)

			# Initialize state and parameters
			z = z.repeat(self.cfg.num_samples, 1)
			mean = torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device)
			std = torch.full((self.cfg.horizon, self.cfg.action_dim), self.cfg.max_std, dtype=torch.float, device=self.device)
			mean[:-1] = self._prev_mean[1:]
			actions = torch.empty(self.cfg.horizon, self.cfg.num_samples, self.cfg.action_dim, device=self.device)
			if self.cfg.num_pi_trajs > 0:
				actions[:, :self.cfg.num_pi_trajs] = pi_actions

			# Iterate MPPI optimization loop (update mean/std over elite trajectories)
			for _ in range(self.cfg.iterations):
				# Sample actions
				r = torch.randn(self.cfg.horizon, self.cfg.num_samples-self.cfg.num_pi_trajs, self.cfg.action_dim, device=std.device)
				actions_sample = mean.unsqueeze(1) + std.unsqueeze(1) * r
				actions_sample = actions_sample.clamp(-1, 1)
				actions[:, self.cfg.num_pi_trajs:] = actions_sample
				if self.cfg.multitask:
					actions = actions * self.model._action_masks[task]

				# Compute elite actions
				value = self._estimate_value(z, actions, task).nan_to_num(0)
				elite_idxs = torch.topk(value.squeeze(1), self.cfg.num_elites, dim=0).indices
				elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

				# Update parameters
				max_value = elite_value.max(0).values
				score = torch.exp(self.cfg.temperature*(elite_value - max_value))
				score = score / score.sum(0)
				mean = (score.unsqueeze(0) * elite_actions).sum(dim=1) / (score.sum(0) + 1e-9)
				std = ((score.unsqueeze(0) * (elite_actions - mean.unsqueeze(1)) ** 2).sum(dim=1) / (score.sum(0) + 1e-9)).sqrt()
				std = std.clamp(self.cfg.min_std, self.cfg.max_std)
				if self.cfg.multitask:
					mean = mean * self.model._action_masks[task]
					std = std * self.model._action_masks[task]

			return score, elite_actions, mean, std


	def update_planner_mean(self, mean):
		self._prev_mean.copy_(mean)
		return 

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
		# dynamics
		groups["dynamics"] = list(self.model._dynamics.parameters())
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
			z = z.view(T, 2*B, L)
		with maybe_range('Agent/update_pi', self.cfg):
			action, info = self.model.pi(z, task)
			qs = self.model.Q(z, action, task, return_type='avg', detach=True)
			self.scale.update(qs[0])
			qs = self.scale(qs)
			rho_pows = torch.pow(self.cfg.rho,
				torch.arange(z.shape[0], device=self.device)
			)
			# Loss is a weighted sum of Q-values
			if self.cfg.ac_source == "imagine" or self.cfg.ac_source == "replay_rollout":
				pi_loss = (-(self.cfg.entropy_coef * info["scaled_entropy"] + qs).mean(dim=(1,2)) * rho_pows).mean()
			else:
				raise NotImplementedError(f"ac_source {self.cfg.ac_source} not implemented for TD-MPC2")
				pi_loss = (-(self.cfg.entropy_coef * info["scaled_entropy"] + qs).mean(dim=(1,2)) * rho_pows.mean()).mean()
    

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
			"pi_frac_sat_095": (info["mean"].abs() > 0.95).float().mean(),
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
		action, _ = self.model.pi(next_z, task, use_ema=self.cfg.policy_ema_enabled)
		discount = self.discount[task].unsqueeze(-1) if self.cfg.multitask else self.discount
		# Primary TD target distribution (still stored as distributional bins via CE loss)
		return reward + discount * (1-terminated) * self.model.Q(next_z, action, task, return_type='min', target=True)  # (T,B,K)

	@torch.no_grad()
	def _td_target_aux(self, next_z, reward, terminated, task):
		"""Compute auxiliary TD targets per gamma using Q_aux(min) bootstrap.

		Uses a conservative (min over two sampled ensemble members) per-gamma
		bootstrap produced internally by `Q_aux(return_type='min')`.

		TODO: Potential enhancement: use full distributional auxiliary logits
		as targets (would require adapting loss to compare distributions). Also do this in normal td target comp

		Args:
			next_z (torch.Tensor): Latent states at t+1. Shape (T,B,L).
			reward (torch.Tensor): Immediate rewards at t. Shape (T,B,1).
			terminated (torch.Tensor): Termination flags (T,B,1).
			task: Task index tensor or None.

		Returns:
			(torch.Tensor | None): (G_aux,T,B,1) scalar TD targets or None if no auxiliaries.
		"""
		G_aux = len(self._all_gammas) - 1
		if G_aux <= 0:
			return None
		action, _ = self.model.pi(next_z, task, use_ema=self.cfg.policy_ema_enabled)
		bootstrap = self.model.Q_aux(next_z, action, task, return_type='avg', target=self.cfg.auxiliary_value_ema)  # (G_aux,T,B,K) scalar per gamma (single head)
		if bootstrap is None:
			return None
		aux_targets = []
		for g in range(G_aux):
			gamma = self._all_gammas[g+1]
			boot_g = bootstrap[g, ...]  # (T,B,1)
			aux_targets.append(reward + gamma * (1 - terminated) * boot_g)
		return torch.stack(aux_targets, dim=0).detach() # (G_aux,T,B,1)


	def world_model_losses(self, z_true, z_target, action, reward, terminated, task=None):
		"""Compute world-model losses (consistency, reward, termination)."""
		T, B, _ = action.shape
		device = z_true.device
		dtype = z_true.dtype

		rho_pows = torch.pow(self.cfg.rho, torch.arange(T, device=device, dtype=dtype))

		consistency_losses = torch.zeros(T, device=device, dtype=dtype)
		encoder_consistency_losses = torch.zeros(T, device=device, dtype=dtype)

		with maybe_range('Agent/world_model_rollout', self.cfg):
			z_rollout = self.rollout_dynamics(z_start=z_true[0], action=action, task=task)
			assert z_target is None, "z_target not supported yet"
			for t in range(T):
				consistency_losses[t] = F.mse_loss(z_rollout[t+1], z_true[t+1].detach())
				encoder_consistency_losses[t] = F.mse_loss(z_rollout[t+1].detach(), z_true[t+1])

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

			reward_logits = self.model.reward(latents, actions_branch, task)
			rew_ce = math.soft_ce(
				reward_logits.reshape(-1, self.cfg.num_bins),
				reward_target.reshape(-1, 1),
				self.cfg,
			).view(latents.shape[0], latents.shape[1], 1).mean(dim=1).squeeze(-1)

			if branch['weight_mode'] == 'rollout':
				reward_loss_branch = (rho_pows * rew_ce).mean()
			else:
				reward_loss_branch = rew_ce.mean() * rho_pows.mean()

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
			branch_reward_error.append(math.two_hot_inv(reward_logits, self.cfg).detach() - reward_target)

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

	def rollout_dynamics(self, z_start, action, task=None):
		"""Roll out dynamics under replay actions without gradients."""
		T = action.shape[0]
		z_rollout = []
		z_rollout.append(z_start)
		z = z_start
		for t in range(T):
			z = self.model.next(z, action[t], task)
			z_rollout.append(z)
		z_rollout = torch.stack(z_rollout, dim=0)  # (T+1, B, L)
		return z_rollout

	def imagined_rollout(self, start_z, task=None, rollout_len=None):
		"""Roll out imagined trajectories from latent start states."""

		S, B, L = start_z.shape
		A = self.cfg.action_dim
		n_rollouts = int(self.cfg.num_rollouts)
		total = S * B * n_rollouts

		device = start_z.device
		dtype = start_z.dtype

		start_flat = start_z.reshape(S * B, L)
		start_rep = start_flat.repeat_interleave(n_rollouts, dim=0)

		z_seq = []
		actions = torch.zeros(rollout_len, total, A, device=device, dtype=dtype)
		rewards = torch.zeros(rollout_len, total, 1, device=device, dtype=dtype)
		term_logits = torch.zeros(rollout_len, total, 1, device=device, dtype=dtype)

		z_seq.append(start_rep)
		latents = start_rep

		with maybe_range('Agent/imagined_rollout', self.cfg):
			with torch.no_grad():
				for t in range(rollout_len):
					current_latents = latents
					action_t, _ = self.model.pi(current_latents, task, use_ema=self.cfg.policy_ema_enabled)
					actions[t] = action_t
					reward_logits = self.model.reward(current_latents, action_t, task)
					rewards[t] = math.two_hot_inv(reward_logits, self.cfg)
					next_latents = self.model.next(current_latents, action_t, task)
					latents = next_latents
					z_seq.append(latents)
					if self.cfg.episodic:
						term_logits[t] = self.model.termination(current_latents, task, unnormalized=True)
		z_seq = torch.stack(z_seq, dim=0)  # (rollout_len+1, total, L)
  
		if self.cfg.episodic:
			terminated = (torch.sigmoid(term_logits) > 0.5).float()
		else:
			terminated = torch.zeros_like(rewards)

		return {
			'z_seq': z_seq, # the first latent comes from the encoder and not imagination, this is not detached when not ac_only.
			'actions': actions.detach(),
			'rewards': rewards.detach(),
			'terminated': terminated.detach(),
			'termination_logits': term_logits.detach(),
		}



	def calculate_value_loss(self, z_seq, actions, rewards, terminated, full_detach, task=None, z_td=None):
		"""Compute primary critic loss on arbitrary latent sequences."""
		if z_td is None:
			assert self.cfg.ac_source != "replay_rollout", "Need to supply z_td for ac_source=replay_rollout in calculate_value_loss"
		T, B, _ = actions.shape
		K  = self.cfg.num_bins
		device = z_seq.device

		rho_pows = torch.pow(self.cfg.rho, torch.arange(T, device=device, dtype=z_seq.dtype))

		z_seq = z_seq.detach() if full_detach else z_seq
		actions = actions.detach()
		rewards = rewards.detach()
		terminated = terminated.detach()

		qs = self.model.Q(z_seq[:-1], actions, task, return_type='all')
		with torch.no_grad():
			if z_td is not None:
				td_targets = self._td_target(z_td, rewards, terminated, task)
			else:
				td_targets = self._td_target(z_seq[1:], rewards, terminated, task)

		Qe = qs.shape[0]
		val_ce = math.soft_ce(
			qs.reshape(Qe * T * B, K),
			td_targets.unsqueeze(0).expand(Qe, -1, -1, -1).reshape(Qe * T * B, 1),
			self.cfg,
		).view(Qe, T, B, 1).mean(dim=(0, 2)).squeeze(-1)

		weighted = val_ce * rho_pows
		loss = weighted.mean()

		info = TensorDict({
			'value_loss': loss
		}, device=device, non_blocking=True)

		for t in range(T):
			info.update({f'value_loss/step{t}': val_ce[t]}, non_blocking=True)
		value_pred = math.two_hot_inv(qs.reshape(Qe, T, B, K), self.cfg) # (Qe,T,B,1)
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
   
		value_error = (value_pred - td_targets.unsqueeze(0).expand(Qe,-1, -1, -1).reshape(Qe, T, B, 1))
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

		T, B, _ = actions.shape
		device = z_seq.device

		rho_pows = torch.pow(self.cfg.rho, torch.arange(T, device=device, dtype=z_seq.dtype))

		z_seq = z_seq.detach() if full_detach else z_seq
		actions = actions.detach()
		rewards = rewards.detach()
		terminated = terminated.detach()

		q_aux_logits = self.model.Q_aux(z_seq[:-1], actions, task, return_type='all')
		if q_aux_logits is None:
			return torch.zeros((), device=device), TensorDict({}, device=device)

		with torch.no_grad():
			if z_td is not None:
				aux_td_targets = self._td_target_aux(z_td, rewards, terminated, task)
			else:
				aux_td_targets = self._td_target_aux(z_seq[1:], rewards, terminated, task) # bootstrap using rolledout

		G_aux = q_aux_logits.shape[0]
		aux_ce = math.soft_ce(
			q_aux_logits.reshape(G_aux * T * B, self.cfg.num_bins),
			aux_td_targets.reshape(G_aux * T * B, 1),
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

	def _compute_loss_components(self, obs, action, reward, terminated, task, ac_only, log_grads):
		device = self.device

		wm_fn = self.world_model_losses
		value_fn = self.calculate_value_loss
		aux_fn = self.calculate_aux_value_loss

		def encode_obs(obs_seq, use_ema, grad_enabled):
			steps, batch = obs_seq.shape[0], obs_seq.shape[1]
			flat_obs = obs_seq.reshape(steps * batch, *obs_seq.shape[2:])
			if self.cfg.multitask:
				if task is None:
					raise RuntimeError('Multitask encoding requires task indices')
				base_task = task.reshape(-1)
				if base_task.numel() != batch:
					raise ValueError(f'Task batch mismatch: expected {batch}, got {base_task.numel()}')
				task_flat = base_task.repeat(steps).to(flat_obs.device).long()
			else:
				task_flat = task
			with torch.set_grad_enabled(grad_enabled and torch.is_grad_enabled()):
				latents_flat = self.model.encode(flat_obs, task_flat, use_ema=use_ema)
			return latents_flat.view(steps, batch, *latents_flat.shape[1:])

		if not ac_only:
			z_true = encode_obs(obs, use_ema=False, grad_enabled=True)
			z_target = encode_obs(obs, use_ema=True, grad_enabled=False) if self.cfg.encoder_ema_enabled else None
			wm_loss, wm_info, z_rollout = wm_fn(z_true, z_target, action, reward, terminated, task)
		else:
			z_true = encode_obs(obs, use_ema=False, grad_enabled=True)
			z_rollout = self.rollout_dynamics(z_start=z_true[0], action=action, task=task)
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
		log.info(f"Update step started. AC only: {ac_only}. Log grads: {log_grads}") if ac_only else None

		with maybe_range('Agent/update', self.cfg):
			self.model.train(True)
			if log_grads:
				components = self._compute_loss_components_eager(obs, action, reward, terminated, task, ac_only, log_grads)
			else:
				components = self._compute_loss_components(obs, action, reward, terminated, task, ac_only, log_grads)
    
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

			info = self.update_end(info.detach(), grad_norm.detach(), pi_grad_norm.detach(), total_loss.detach(), pi_info.detach())
		return info


	@torch.compile(mode='reduce-overhead')
	def update_end(self, info, grad_norm, pi_grad_norm, total_loss, pi_info):
		"""Function called at the end of each update iteration."""
		info.update({
				'grad_norm': grad_norm,
				'pi_grad_norm': pi_grad_norm,
				'total_loss': total_loss.detach()
			}, non_blocking=True)
		info.update(pi_info, non_blocking=True)
		self.model.soft_update_target_Q()
		self.model.soft_update_policy_encoder_targets()
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
   
		kwargs = {}
		if task is not None:
			kwargs["task"] = task
		torch.compiler.cudagraph_mark_step_begin()
		log_grads = self.cfg.log_gradients_per_loss and self.log_detailed

		if log_grads:
			return self._update_eager(obs, action, reward, terminated, ac_only=ac_only, **kwargs)
		else:
			return self._update(obs, action, reward, terminated, ac_only=ac_only, **kwargs)


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


		# don’t destroy graph grads; keep buffers allocated for CUDA-graph stability
		self.optim.zero_grad(set_to_none=False)

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
					components = self._compute_loss_components(obs, action, reward, terminated, task, ac_only=False, log_grads=False)
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
