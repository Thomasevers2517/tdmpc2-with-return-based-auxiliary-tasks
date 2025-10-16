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
		self._update_step = 0  # incremented at end of _update
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
			# self._update = torch.compile(self._update, mode="reduce-overhead")
			# self._update = torch.compile(self._update, mode="default", fullgraph=True)
			self.calc_wm_losses_eager = self.calc_wm_losses
			self.calc_pi_losses_eager = self.calc_pi_losses
			self.calc_imagine_value_loss_eager = self.calc_imagine_value_loss
   
			self.calc_wm_losses = torch.compile(self.calc_wm_losses, mode=self.cfg.compile_type, fullgraph=True)
			self.calc_pi_losses = torch.compile(self.calc_pi_losses, mode=self.cfg.compile_type, fullgraph=True)
			self.calc_imagine_value_loss = torch.compile(self.calc_imagine_value_loss, mode=self.cfg.compile_type, fullgraph=True)

			# @torch._dynamo.disable()
			@torch.compile(mode=self.cfg.compile_type, fullgraph=False)
			def optim_step():
				self.optim.step()
				return

			# @torch._dynamo.disable()
			@torch.compile(mode=self.cfg.compile_type, fullgraph=False)
			def pi_optim_step():
				self.pi_optim.step()
				return

			self.optim_step = optim_step
			self.pi_optim_step = pi_optim_step
   
			self.act = torch.compile(self.act, mode=self.cfg.compile_type, dynamic=True)
		else:
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
			if self.cfg.pred_from == "rollout":
				pi_loss = (-(self.cfg.entropy_coef * info["scaled_entropy"] + qs).mean(dim=(1,2)) * rho_pows).mean()
			elif self.cfg.pred_from == "true_state":
				pi_loss = (-(self.cfg.entropy_coef * info["scaled_entropy"] + qs).mean(dim=(1,2)) * rho_pows.mean()).mean()
    
			elif self.cfg.pred_from == "both":
				qs = qs.view(T, 2, B, -1)  # (T,2,B,1)
				scaled_entropy = info["scaled_entropy"].contiguous().view(T, 2, B, 1)  # (T,2,B)

				true_state_loss = (-(self.cfg.entropy_coef * scaled_entropy[:,0] + qs[:, 0, ...]).mean(dim=(1,2)) * rho_pows.mean()).mean()
				rollout_loss = (-(self.cfg.entropy_coef * scaled_entropy[:,1] + qs[:, 1, ...]).mean(dim=(1,2)) * rho_pows).mean()
				pi_loss = (1-self.cfg.rollout_fraction)* true_state_loss + self.cfg.rollout_fraction*rollout_loss

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
		log_grads = self.cfg.log_gradients_per_loss and (self._update_step % self.cfg.log_gradients_every == 0)
		pi_loss, info = self.calc_pi_losses(zs, task) if (not log_grads or not self.cfg.compile) else self.calc_pi_losses_eager(zs, task)
		# if log_grads:
		# 	# self.probe_pi_gradients(pi_loss, info)
		
		info = TensorDict({
			"pi_loss": pi_loss,
			"pi_loss_weighted": pi_loss * self.cfg.policy_coef,
			"pi_entropy": info["entropy"],
			"pi_scaled_entropy": info["scaled_entropy"],
			"pi_scale": self.scale.value,
			"pi_std": info["log_std"].mean(),
			"pi_mean": info["mean"].mean(),
		}, device=self.device)
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


	def calc_wm_losses(self, obs, action, reward, terminated ,task=None):		
		z_true  = self.model.encode(obs, task)  # initial latent (T+1,B,L)

		# TODO   fetching longer trajectories from buffer and using data overlap for increased efficiency
		with maybe_range('Agent/calc_td_target', self.cfg):
			with torch.no_grad():
				# Encode next observations (time steps 1..T) → latent sequence next_z
				if self.cfg.encoder_ema_enabled:
					# Use EMA encoder for targets (no grad, no update)
					z_true_ema = self.model.encode(obs, task, use_ema=True)  # (T+1,B,L)
					next_z = z_true_ema[1:].detach()  # (T,B,L)
				else:
					# Use online encoder for targets (no grad, no update)	
					next_z = z_true[1:].detach()  # (T,B,L)	
     
				# Distributional TD targets (primary gamma) vs Q logits
				td_targets = self._td_target(next_z, reward, terminated, task)  # (T,B,K)
				# Auxiliary scalar TD targets per gamma (optional)
				aux_td_targets = self._td_target_aux(next_z, reward, terminated, task)  # (G_aux,T,B,1) or None

		# ------------------------------ Latent rollout (consistency) ------------------------------
		self.model.train()
		z_rollout = torch.empty(self.cfg.horizon+1, self.cfg.batch_size, self.cfg.latent_dim, device=self.device)  # allocate (T+1,B,L)
		consistency_losses = torch.zeros(self.cfg.horizon, device=self.device)
		encoder_consistency_losses = torch.zeros(self.cfg.horizon, device=self.device)
  
		with maybe_range('Agent/latent_rollout', self.cfg):
			z = z_true[0]                          # initial latent z_0 (B,L)
			z_rollout[0] = z
			for t, (_a, _target_next_z) in enumerate(zip(action.unbind(0), next_z.unbind(0))):  # iterate T steps
				z = self.model.next(z, _a, task)            # model prediction z_{t+1}
				# Consistency MSE between predicted & encoded next latent
				consistency_losses[t] = F.mse_loss(z, _target_next_z)
				if self.cfg.encoder_consistency_coef > 0:
					# this backrpopagtes through the encoder. Like dreamerv3
					encoder_consistency_losses[t] = F.mse_loss(z.detach(), z_true[t+1])
				else:
					encoder_consistency_losses[t] = torch.tensor(0., device=self.device)
     
				z_rollout[t+1] = z

		# ------------------------------ Model predictions for losses ------------------------------
		_zs = z_rollout[:-1]                                 # (T,B,L) latents aligned with actions
  		# Auxiliary logits (if enabled)
		q_aux_logits = None

		# Use the rollout latents for predictions, or the true encoded latents. Original TD-MPC used true latents.
		if self.cfg.pred_from == "true_state":
			qs = self.model.Q(z_true[:-1], action, task, return_type='all')  # (Qe,T,B,K)
			reward_preds = self.model.reward(z_true[:-1], action, task)         # (T,B,K)
			if aux_td_targets is not None:
				q_aux_logits = self.model.Q_aux(z_true[:-1], action, task, return_type='all')  # (G_aux,T,B,K)
			if self.cfg.episodic:
				termination_pred = self.model.termination(z_true[1:], task, unnormalized=True)  # (T,B,1)
			else:
				termination_pred = None

		elif self.cfg.pred_from == "rollout":
			qs = self.model.Q(_zs, action, task, return_type='all')              # (Qe,T,B,K)
			reward_preds = self.model.reward(_zs, action, task)                 # (T,B,K)
   
			if aux_td_targets is not None:
				q_aux_logits = self.model.Q_aux(_zs, action, task, return_type='all')  # (G_aux,T,B,K)
			if self.cfg.episodic:
				termination_pred = self.model.termination(z_rollout[1:], task, unnormalized=True)  # (T,B,1)
			else:
				termination_pred = None	
		elif self.cfg.pred_from == "both":
			# z_both = torch.cat([z_true[:-1].unsqueeze(1), _zs.unsqueeze(1)], dim=1)  # (T,2,B,L)
			if self.cfg.split_batch:
				Bh = self.cfg.batch_size // 2
				z_both = torch.cat([z_true[:-1, :Bh].unsqueeze(1), _zs[:, Bh:].unsqueeze(1)], dim=1)  # (T,2,B/2,L)
				T, _, B, L = z_both.shape
			else:
				z_both = torch.cat([z_true[:-1].unsqueeze(1), _zs.unsqueeze(1)], dim=1)  # (T,2,B,L)
				T, _, B, L = z_both.shape
				action = action.unsqueeze(1).expand(-1, 2, -1, -1).reshape(T, B*2, -1)  # (T,2,B,A) → (T,B*2,A)

	
			z_both = z_both.view(T, B*2, L)
			qs = self.model.Q(z_both, action, task, return_type='all').view(-1, T, 2, B, self.cfg.num_bins)  # (Qe,T,2,B,K)    
			reward_preds = self.model.reward(z_both, action, task).view(T, 2, B, self.cfg.num_bins)         # (T,2,B,K)
   
			if aux_td_targets is not None:
				q_aux_logits = self.model.Q_aux(z_both, action, task, return_type='all').view(-1, T, 2, B, self.cfg.num_bins)  # (G_aux,T,2,B,K) 
			if self.cfg.episodic:
				termination_pred = self.model.termination(torch.cat([z_true[1:].unsqueeze(1), z_rollout[1:].unsqueeze(1)], dim=1).view(T, B*2, L), task, unnormalized=True).view(T, 2, B, 1)  # (T,2,B,1)
			else:
				termination_pred = None	
                                               
		else:
			raise ValueError(f'Invalid pred_from value: {self.cfg.pred_from}')

		with maybe_range('Agent/order_losses', self.cfg):
			total_loss, info = self.order_wm_losses(qs, reward_preds, reward, td_targets, aux_td_targets, q_aux_logits, terminated, termination_pred, consistency_losses, encoder_consistency_losses)

		return total_loss, z_rollout, info, z_true, z_both.view(T, 2, B, L) if self.cfg.pred_from == "both" else None 
	
	def order_wm_losses(self, qs, reward_preds, reward, td_targets, aux_td_targets, q_aux_logits, terminated, termination_pred, consistency_losses, encoder_consistency_losses):
		T = reward_preds.shape[0]
		B = reward_preds.shape[-2]
		K = reward_preds.shape[-1]

		rho_pows = torch.pow(self.cfg.rho,
			torch.arange(T, device=self.device, dtype=consistency_losses.dtype)
		)
		
		# Consistency loss (MSE) over latent prediction errors, weighted by rho^t
		consistency_loss = (rho_pows * consistency_losses).mean()
		encoder_consistency_loss = (rho_pows * encoder_consistency_losses).mean()
  		
		if self.cfg.pred_from == "both":
			assert qs.shape[2] == 2, "Expected second dimension of qs to be 2 for 'both' pred_from setting"
			roll_out_qs = qs[:, :, 1, ...]  # (Qe,T,B,K)
			true_state_qs = qs[:, :, 0, ...]  # (Qe,T,B,K)
			roll_out_reward_preds = reward_preds[:, 1, ...]  # (T,B,K)
			true_state_reward_preds = reward_preds[:, 0, ...]  # (T,B,K)
			if aux_td_targets is not None:
				roll_out_q_aux_logits = q_aux_logits[:, :, 1, ...]  # (G_aux,T,B,K)
				true_state_q_aux_logits = q_aux_logits[:, :, 0, ...]  # (G_aux,T,B,K)
			else:
				roll_out_q_aux_logits = None
				true_state_q_aux_logits = None
			roll_out_termination_pred = termination_pred[ :, 1, ...] if termination_pred is not None else None  # (T,B,1)
			true_state_termination_pred = termination_pred[ :, 0, ...] if termination_pred is not None else None  # (T,B,1)
			# Compute losses separately for both prediction sources and average
			# ------------------------------ Separate loss computation for both prediction sources ------------------------------
			roll_out_reward = reward if not self.cfg.split_batch else reward[:, B:]
			true_state_reward = reward if not self.cfg.split_batch else reward[:, :B]
			roll_out_terminated = terminated if not self.cfg.split_batch else terminated[:, B:]
			true_state_terminated = terminated if not self.cfg.split_batch else terminated[:, :B]
			roll_out_td_targets = td_targets if not self.cfg.split_batch else td_targets[:, B:]
			true_state_td_targets = td_targets if not self.cfg.split_batch else td_targets[:, :B]
			roll_out_aux_td_targets = aux_td_targets if aux_td_targets is None or not self.cfg.split_batch else aux_td_targets[:, :, B:]
			true_state_aux_td_targets = aux_td_targets if aux_td_targets is None or not self.cfg.split_batch else aux_td_targets[:, :, :B]
   
			with maybe_range('Agent/calc_wm_loss_from_preds', self.cfg):
				#Output is (reward_loss, value_loss, aux_value_losses, aux_value_loss_mean, termination_loss, rew_ce, val_ce, aux_ce)
				ro, ro_info = self.calc_wm_loss_from_preds(roll_out_qs, roll_out_reward_preds, roll_out_reward, roll_out_td_targets, roll_out_aux_td_targets, roll_out_q_aux_logits, roll_out_terminated, roll_out_termination_pred, time_decay=True)
				ts, ts_info = self.calc_wm_loss_from_preds(true_state_qs, true_state_reward_preds, true_state_reward, true_state_td_targets, true_state_aux_td_targets, true_state_q_aux_logits, true_state_terminated, true_state_termination_pred, time_decay=False)
			
			reward_loss = self.cfg.rollout_fraction* ro["reward_loss"] + ts["reward_loss"]*(1-self.cfg.rollout_fraction)
			value_loss = self.cfg.rollout_fraction* ro["value_loss"] + ts["value_loss"]*(1-self.cfg.rollout_fraction)
			aux_value_losses = [self.cfg.rollout_fraction* r + s*(1-self.cfg.rollout_fraction) for r, s in zip(ro["aux_value_losses"], ts["aux_value_losses"])] if ro["aux_value_losses"] is not None else [torch.tensor(0., device=self.device) for _ in range(len(self._all_gammas)-1)]
			aux_value_loss_mean = self.cfg.rollout_fraction* ro["aux_value_loss_mean"] + ts["aux_value_loss_mean"]*(1-self.cfg.rollout_fraction) if ro["aux_value_loss_mean"] is not None else torch.tensor(0., device=self.device)
			termination_loss = self.cfg.rollout_fraction* ro["termination_loss"] + ts["termination_loss"]*(1-self.cfg.rollout_fraction)
			rew_ce = self.cfg.rollout_fraction* ro["rew_ce"] + ts["rew_ce"]*(1-self.cfg.rollout_fraction)
			val_ce = self.cfg.rollout_fraction* ro["val_ce"] + ts["val_ce"]*(1-self.cfg.rollout_fraction)
			aux_ce = self.cfg.rollout_fraction* ro["aux_ce"] + ts["aux_ce"]*(1-self.cfg.rollout_fraction) if ro["aux_ce"] is not None else  torch.tensor(0., device=self.device)
		elif self.cfg.pred_from == "true_state":
			ro_info = None
			losses, ts_info = self.calc_wm_loss_from_preds(qs, reward_preds, reward, td_targets, aux_td_targets, q_aux_logits, terminated, termination_pred, time_decay=False)
			reward_loss, value_loss, aux_value_losses, aux_value_loss_mean, termination_loss, rew_ce, val_ce, aux_ce = losses["reward_loss"], losses["value_loss"], losses["aux_value_losses"], losses["aux_value_loss_mean"], losses["termination_loss"], losses["rew_ce"], losses["val_ce"], losses["aux_ce"]
		elif self.cfg.pred_from == "rollout":
			ts_info = None
			losses, ro_info = self.calc_wm_loss_from_preds(qs, reward_preds, reward, td_targets, aux_td_targets, q_aux_logits, terminated, termination_pred, time_decay=True)
			reward_loss, value_loss, aux_value_losses, aux_value_loss_mean, termination_loss, rew_ce, val_ce, aux_ce = losses["reward_loss"], losses["value_loss"], losses["aux_value_losses"], losses["aux_value_loss_mean"], losses["termination_loss"], losses["rew_ce"], losses["val_ce"], losses["aux_ce"]
  		# ------------------------------ Vectorized loss computation ------------------------------

		# Total loss (auxiliary added with its own weight separate from value_coef)
		total_loss = (
			self.cfg.consistency_coef * consistency_loss +
			self.cfg.encoder_consistency_coef * encoder_consistency_loss +
			self.cfg.reward_coef * reward_loss +
			self.cfg.termination_coef * termination_loss +
			self.cfg.value_coef * value_loss +
			self.cfg.multi_gamma_loss_weight * aux_value_loss_mean
		)
		info = TensorDict({
			"consistency_losses": consistency_losses,
			"consistency_loss": consistency_loss,
			"encoder_consistency_loss": encoder_consistency_loss,
			"reward_loss": reward_loss,
			"value_loss": value_loss,
			"aux_value_loss_mean": aux_value_loss_mean,
			"termination_loss": termination_loss,
			"total_loss": total_loss,
			"consistency_loss_weighted": consistency_losses * self.cfg.consistency_coef, 	
			"encoder_consistency_loss_weighted": encoder_consistency_losses * self.cfg.encoder_consistency_coef,
			"reward_loss_weighted": reward_loss * self.cfg.reward_coef,
			"value_loss_weighted": value_loss * self.cfg.value_coef,
			"termination_loss_weighted": termination_loss * self.cfg.termination_coef,
			"aux_value_loss_mean_weighted": aux_value_loss_mean * self.cfg.multi_gamma_loss_weight,
   
			"td_target_mean": td_targets.mean(),
			"td_target_mstd": td_targets.std(),
			"value_mean": math.two_hot_inv(qs, self.cfg).mean(),
			"value_std": math.two_hot_inv(qs, self.cfg).std(),
			"reward_target_mean": reward.mean(),
			"reward_target_std": reward.std(),
			"reward_mean": math.two_hot_inv(reward_preds, self.cfg).mean(),
			"reward_std": math.two_hot_inv(reward_preds, self.cfg).std(),
			"aux_td_target_mean": aux_td_targets.mean() if aux_td_targets is not None else torch.tensor(0., device=self.device),
			"aux_td_target_std": aux_td_targets.std() if aux_td_targets is not None else torch.tensor(0., device=self.device),
			"aux_value_mean": math.two_hot_inv(q_aux_logits, self.cfg).mean() if q_aux_logits is not None else torch.tensor(0., device=self.device),
			"aux_value_std": math.two_hot_inv(q_aux_logits, self.cfg).std() if q_aux_logits is not None else torch.tensor(0., device=self.device)
		}, device=self.device, non_blocking=True)
  
		if self.cfg.pred_from == "both":
			info.update(ro_info, non_blocking=True)
			info.update(ts_info, non_blocking=True)
		elif self.cfg.pred_from == "true_state":
			info.update(ts_info, non_blocking=True)
		elif self.cfg.pred_from == "rollout":
			info.update(ro_info, non_blocking=True)	
  
		for i in range(T):
			info.update({f"consistency_loss/step{i}": consistency_losses[i],
						"consistency_loss_weighted/step{i}": self.cfg.consistency_coef * consistency_losses[i] * rho_pows[i],
						"encoder_consistency_loss/step{i}": encoder_consistency_losses[i],
						"encoder_consistency_loss_weighted/step{i}": self.cfg.encoder_consistency_coef * encoder_consistency_losses[i] * rho_pows[i],
						"reward_loss/step{i}": rew_ce[i],
						"value_loss/step{i}": val_ce[i], 
						"aux_value_loss/step{i}": aux_ce[:,i].mean() if aux_ce is not None else torch.tensor(0., device=self.device)}, non_blocking=True)

		if aux_value_losses is not None:
			for g, loss_g in enumerate(aux_value_losses):
				gamma_val = self._all_gammas[g+1]	
				info.update({f"aux_value_loss/gamma{gamma_val:.4f}": loss_g,
							f"aux_value_loss_weighted/gamma{gamma_val:.4f}": self.cfg.multi_gamma_loss_weight * loss_g,
							f"aux_td_target_mean/gamma{gamma_val:.4f}": aux_td_targets[g].mean(),
							f"aux_td_target_std/gamma{gamma_val:.4f}": aux_td_targets[g].std(),
							f"aux_value_mean/gamma{gamma_val:.4f}": math.two_hot_inv(q_aux_logits[g], self.cfg).mean(),
							f"aux_value_std/gamma{gamma_val:.4f}": math.two_hot_inv(q_aux_logits[g], self.cfg).std()}, non_blocking=True)
		
		if self.cfg.episodic:
			info.update(math.termination_statistics(torch.sigmoid(termination_pred[-1]), terminated[-1]), non_blocking=True)
   
		return total_loss, info
 
	def calc_wm_loss_from_preds(self, qs, reward_preds, reward, td_targets, aux_td_targets, q_aux_logits, terminated, termination_pred, time_decay=False):
		"""Compute model losses from precomputed predictions and targets.
		Args:				
			qs (torch.Tensor): Q-value logits from the model. Shape (Qe,T,B,K).
			reward_preds (torch.Tensor): Reward logits from the model. Shape (T,B,K).
			reward (torch.Tensor): Reward targets. Shape (T,B,1).
			td_targets (torch.Tensor): Primary TD targets. Shape (T,B,1).
			aux_td_targets (torch.Tensor | None): Auxiliary TD targets or None. Shape (G_aux,T,B,1).
			q_aux_logits (torch.Tensor | None): Auxiliary Q-value logits or None. Shape (G_aux,T,B,K).
			terminated (torch.Tensor): Termination targets. Shape (T,B,1).
			termination_pred (torch.Tensor | None): Termination logits or None. Shape (T,B,1).
			time_decay (bool): Whether to apply time decay (rho^t) to losses. 
		Returns:
			tuple: (reward_loss, value_loss, aux_value_losses, aux_value_loss_mean, termination_loss, rew_ce, val_ce, aux_ce)
  		"""
		T, B, K, = reward_preds.shape
		rho_pows = torch.pow(self.cfg.rho,
			torch.arange(T, device=self.device, dtype=qs.dtype)
		)

		# Reward CE over all (T,B) at once -> shape (T,)
		rew_ce = math.soft_ce(
			reward_preds.reshape(T * B, K),
			reward.reshape(T * B, 1),
			self.cfg,
		)
		rew_ce = rew_ce.view(T, B, 1).mean(dim=1).squeeze(-1)  # (T,)
		reward_loss = (rho_pows * rew_ce).mean() if time_decay else rew_ce.mean()* rho_pows.mean()  # scalar

		# Value CE across ensemble heads (Qe,T,B) -> mean over B, sum over heads, weight by rho
		Qe = qs.shape[0]
		val_ce = math.soft_ce(
			qs.reshape(Qe * T * B, K),
			# expand td_targets to (Qe,T,B,1) then flatten
			td_targets.unsqueeze(0).expand(Qe, -1, -1, -1).reshape(Qe * T * B, 1),
			self.cfg,
		)
		val_ce = val_ce.view(Qe, T, B, 1).mean(dim=(0,2)).squeeze(-1)  # (T)
		value_loss = (val_ce * rho_pows).mean() if time_decay else val_ce.mean()* rho_pows.mean()  # scalar

		# Auxiliary per-gamma losses (optional), vectorized over (G_aux,T,B)
		aux_value_losses = None
		if q_aux_logits is not None:
			G_aux = aux_td_targets.shape[0]
			aux_ce = math.soft_ce(
				q_aux_logits.reshape(G_aux * T * B, K),
				aux_td_targets.reshape(G_aux * T * B, 1),
				self.cfg,
			)
			aux_ce = aux_ce.view(G_aux, T, B, 1).mean(dim=2).squeeze(-1)  # (G_aux,T)
			aux_value_losses = (aux_ce * rho_pows.unsqueeze(0)).mean(dim=1) if time_decay else aux_ce.mean(dim=1)* rho_pows.mean()  # (G_aux,)
		else:
			aux_ce = None
			G_aux = 0
		# reward_loss and value_loss were already normalized by T and T*Qe above
		if self.cfg.episodic:
			termination_loss = F.binary_cross_entropy_with_logits(termination_pred, terminated)
		else:
			termination_loss = torch.tensor(0., device=self.device)
		if aux_value_losses is not None:
			# already normalized by T above
			aux_value_loss_mean = aux_value_losses.mean()
		else:
			aux_value_loss_mean = torch.tensor(0., device=self.device)
   
		# Debug check: ensure value_loss matches MSE to expected values

		losses = TensorDict({
			"reward_loss": reward_loss,
			"value_loss": value_loss,
			"aux_value_losses": aux_value_losses if aux_value_losses is not None else torch.zeros((G_aux,), device=self.device),
			"aux_value_loss_mean": aux_value_loss_mean,
			"termination_loss": termination_loss,
			"rew_ce": rew_ce,
			"val_ce": val_ce,
			"aux_ce": aux_ce if aux_ce is not None else torch.zeros((G_aux,T), device=self.device)
			}, device=self.device, non_blocking=True)
  
		value_error = (math.two_hot_inv(qs.reshape(Qe, T, B, K), self.cfg) - td_targets.unsqueeze(0).expand(Qe,-1, -1, -1).reshape(Qe, T, B, 1))
		reward_error = (math.two_hot_inv(reward_preds.reshape(T, B, K), self.cfg) - reward.reshape(T, B, 1))
		info = TensorDict({}, device=self.device)
		for i in range(T):
			info.update({f"value_error_abs_mean/step{i}": value_error[:,i].abs().mean(),
						f"value_error_std/step{i}": value_error[:,i].std(),
						f"value_error_max/step{i}": value_error[:,i].abs().max(),
						f"reward_error_abs_mean/step{i}": reward_error[i].abs().mean(),
						f"reward_error_std/step{i}": reward_error[i].std(),
						f"reward_error_max/step{i}": reward_error[i].abs().max()}, non_blocking=True)
		return losses, info.detach()
 
	def calc_imagine_value_loss(self, z, task=None):
		with maybe_range('Agent/calc_imagine_value_loss', self.cfg):
			im_horizon = self.cfg.imagination_horizon
			num_start_states = self.cfg.imagine_num_starts
			batch_size = z.shape[1]
			imagine_value_loss = torch.tensor(0., device=self.device)
			imagine_info = TensorDict({}, device=self.device)
			# 
			total_states = z.shape[0]
			if num_start_states > total_states:
				raise ValueError(f"Requested {num_start_states} start states but only {total_states} are available.")
			random_scores = torch.rand(total_states, batch_size, device=z.device)
			start_states_idx = torch.argsort(random_scores, dim=0)[:num_start_states]
			index = start_states_idx.unsqueeze(-1).expand(-1, -1, z.size(-1))
			start_z = torch.gather(z, 0, index)
   
			start_z = start_z.reshape(-1, z.size(-1))
			reward = torch.zeros((im_horizon, num_start_states*batch_size, 1), device=self.device, dtype=z.dtype)
			actions = torch.zeros((im_horizon+1, num_start_states*batch_size, self.cfg.action_dim), device=self.device, dtype=z.dtype)
			rolled_out_z = torch.zeros((im_horizon+1, num_start_states*batch_size, z.size(-1)), device=self.device, dtype=z.dtype)
			z_step = start_z
			for t in range(im_horizon):
				# Predict action from current state
				with torch.no_grad():
					action, pi_info = self.model.pi(z_step.detach(), task, use_ema=self.cfg.policy_ema_enabled) # (num_start_states*B,A)
					actions[t, :, :] = action
					rolled_out_z[t, :, :] = z_step
					# Compute Q-values for state and action
					reward_pred = math.two_hot_inv(self.model.reward(z_step, action, task), self.cfg)  # (num_start_states*B,1)
					reward[t, :, :] = reward_pred
					# Step dynamics forward
					z_step = self.model.next(z_step, action, task)  # (num_start_states*B,L)
			with torch.no_grad():
				actions[-1, :, :] = self.model.pi(z_step.detach(), task, use_ema=self.cfg.policy_ema_enabled)[0]
				rolled_out_z[-1, :, :] = z_step
				# Compute TD target for final state with correct per-sample discounts in multitask
				if self.cfg.multitask and task is not None:
					# task may be (B,) or (T,B); take per-sample task ids for the batch
					task_b = task[0] if isinstance(task, torch.Tensor) and task.ndim == 2 else task
					gamma_b = self.discount[task_b.long()]  # (B,)
					# Each batch sample appears num_start_states times in Ns*B
					discount_samples = gamma_b.repeat_interleave(num_start_states)  # (Ns*B,)
				else:
					discount_samples = self.discount  # scalar tensor
				td_target = self.multi_step_td_target(actions, rolled_out_z, reward, task, discount=discount_samples)  # (num_start_states*B,1)
			# Compute Q-values for all state-action pairs
			Qs = self.model.Q(rolled_out_z[0], actions[0], task, return_type='all')  # (Qe,num_start_states*B,K)
			Qe = Qs.shape[0]
			val_ce = math.soft_ce(
				Qs.reshape(Qe * num_start_states * batch_size, self.cfg.num_bins),
				# expand td_targets to (Qe,num_start_states*B,1) then flatten; detach to block grads to targets
				td_target.detach().unsqueeze(0).expand(Qe, -1, -1).reshape(Qe * num_start_states * batch_size, 1),
				self.cfg,
			)
			imagine_value_loss = val_ce.mean(dim=0).squeeze(-1)  # scalar
			imagine_info = TensorDict({
				"imagine_value_loss": imagine_value_loss,
				"imagine_value_loss_weighted": self.cfg.imagine_value_loss_coef * imagine_value_loss,
			}, device=self.device, non_blocking=True)

			return imagine_value_loss, imagine_info

	def multi_step_td_target(self, actions, rollout_z, rewards, task, discount=None):	
		"""Compute multi-step TD target for imagined trajectories.	
		Args:
			actions: (im_horizon+1, num_start_states*B, A)
			rollout_z: (im_horizon+1, num_start_states*B, L)
			rewards: (im_horizon, num_start_states*B, 1)
			discount: scalar tensor or (num_start_states*B,) tensor of per-sample gammas
			task: Task index tensor or None.
		Returns:
			(torch.Tensor): (num_start_states*B, 1) scalar TD targets
		"""
		H = rewards.shape[0]
		NsB = rewards.shape[1]
		td_targets = torch.zeros((NsB, rewards.shape[2]), device=rewards.device, dtype=rewards.dtype)

		with torch.no_grad():
			Qs = self.model.Q(rollout_z[-1], actions[-1], task, return_type='min', target=True)  # (NsB,1)
		# Compute discount powers for all time steps at once, supporting scalar or per-sample gammas
		if torch.is_tensor(discount) and discount.ndim == 1:
			# discount: (NsB,) -> broadcast to (H,NsB,1)
			gamma = discount.view(1, NsB, 1)
			steps = torch.arange(H, device=rewards.device, dtype=rewards.dtype).view(H, 1, 1)
			discount_powers = gamma.pow(steps)  # (H,NsB,1)
			boot_factor = discount.pow(H).view(NsB, 1)  # (NsB,1)
		else:
			# scalar tensor or float
			disc = discount if torch.is_tensor(discount) else torch.as_tensor(discount, device=rewards.device, dtype=rewards.dtype)
			steps = torch.arange(H, device=rewards.device, dtype=rewards.dtype).view(H, 1, 1)
			discount_powers = disc.pow(steps)  # (H,1,1)
			boot_factor = disc.pow(H)  # scalar
		
		# Compute discounted rewards sum
		td_targets = (discount_powers * rewards).sum(dim=0)  # (NsB,1)
		# Add final bootstrap value
		td_targets = td_targets + boot_factor * Qs  # (NsB,1)
		return td_targets

	def _update(self, obs, action, reward, terminated, task=None):
		"""Single gradient update step.

		Args:
			obs: (T+1, B, *obs_shape)
			action: (T, B, A)
			reward: (T, B, 1) scalar rewards
			terminated: (T, B, 1) binary (0/1)
			imagine: (bool) whether to include imagination-augmented value loss
			task: (optional) task index tensor for multi-task mode
		"""
		log_grads = self.cfg.log_gradients_per_loss and (self._update_step % self.cfg.log_gradients_every == 0)
 
		with maybe_range('Agent/update', self.cfg):
			# ------------------------------ Targets (no grad) ------------------------------
			wm_loss, zs, info, z_true, z_both = self.calc_wm_losses(obs, action, reward, terminated, task=task) if (not log_grads or not self.cfg.compile) else self.calc_wm_losses_eager(obs, action, reward, terminated, task=task)
			# Imagination-augmented value loss (optional)
			if self.cfg.imagination_enabled and self.cfg.imagine_value_loss_coef > 0:
				z_im = z_true
				z_im = z_im.detach() if self.cfg.detach_imagine_value else z_im
				imagine_value_loss, imagine_info = self.calc_imagine_value_loss(z_im, task=task) if (not log_grads or not self.cfg.compile) else self.calc_imagine_value_loss_eager(z_im, task=task)
			else:
				imagine_value_loss = torch.tensor(0., device=self.device)
				imagine_info = TensorDict({}, device=self.device)
			if log_grads:
				info = self.probe_wm_gradients(info, imagine_info)
		


			# ------------------------------ Policy update ------------------------------
			
			if self.cfg.pred_from == "rollout":
				z_for_pi = zs.detach()
				# Policy update (detached rollout latents)
			elif self.cfg.pred_from == "true_state":
				z_for_pi = z_true.detach()
				# Policy update (detached latent sequence)
			elif self.cfg.pred_from == "both":
				z_for_pi = z_both.detach()
				# Policy update (mixed latents)



			# ------------------------------ Backprop & updates ------------------------------
			if self.cfg.imagination_enabled and self.cfg.imagine_value_loss_coef > 0:
				total_loss = wm_loss+ imagine_value_loss * self.cfg.imagine_value_loss_coef
			else:
				total_loss = wm_loss
			total_loss.backward()
			grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)

			if log_grads:
				self.optim.step()
			else:
				self.optim_step() #This one is compiled

			self.optim.zero_grad(set_to_none=True)

			pi_loss, pi_info = self.update_pi(z_for_pi, task)
			pi_loss = pi_loss * self.cfg.policy_coef
			pi_loss.backward()	
			pi_grad_norm = torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm)
			if log_grads:
				self.pi_optim.step()
			else:
				self.pi_optim_step() #This one is compiled

			self.pi_optim.zero_grad(set_to_none=True)

			pi_info.update({
				"pi_grad_norm": pi_grad_norm,
			}, non_blocking=True)
			info.update({
				"grad_norm": grad_norm,
			}, non_blocking=True)
			info.update(imagine_info, non_blocking=True)

   
   
			self.model.soft_update_target_Q()
			self.model.soft_update_policy_encoder_targets()
			if self.cfg.encoder_ema_enabled:
				info.update({
					"encoder_ema_max_delta": torch.tensor(self.model.encoder_target_max_delta, device=self.device)
				}, non_blocking=True)
			if self.cfg.policy_ema_enabled:
				info.update({
					"policy_ema_max_delta": torch.tensor(self.model.policy_target_max_delta, device=self.device)
				}, non_blocking=True)

			# ------------------------------ Logging tensor dict ------------------------------
			self.model.eval()

			info.update(pi_info, non_blocking=True)
			# step counter for gated logging
			self._update_step += 1
			return info.detach().mean()

	def update(self, buffer):
		"""
		Main update function. Corresponds to one iteration of model learning.

		Args:
			buffer (common.buffer.Buffer): Replay buffer.

		Returns:
			dict: Dictionary of training statistics.
		"""
		with maybe_range('update/sample_buffer', self.cfg):
			obs, action, reward, terminated, task = buffer.sample()

    
		kwargs = {}
		if task is not None:
			kwargs["task"] = task
		torch.compiler.cudagraph_mark_step_begin()
		return self._update(obs, action, reward, terminated, **kwargs)


	@torch._dynamo.disable()
	def probe_wm_gradients(self, info, imagine_info):
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
		if self.cfg.imagination_enabled and self.cfg.imagine_value_loss_coef > 0:
			loss_parts['imagine_value'] = self.cfg.imagine_value_loss_coef * imagine_info['imagine_value_loss']


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
