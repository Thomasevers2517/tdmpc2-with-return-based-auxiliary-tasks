import torch
import torch.nn.functional as F

from common import math
from common.nvtx_utils import maybe_range
from common.scale import RunningScale
from common.world_model import WorldModel
from common.layers import api_model_conversion
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
			self.calc_wm_losses = torch.compile(self.calc_wm_losses, mode=self.cfg.compile_type, fullgraph=True)
			self.calc_pi_losses = torch.compile(self.calc_pi_losses, mode=self.cfg.compile_type, fullgraph=True)
   

	def reset_planner_state(self):
		self._prev_mean.zero_() 
	
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
		return

	@torch.no_grad()
	def act(self, obs, eval_mode=False, task=None):
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
			obs = obs.to(self.device, non_blocking=True).unsqueeze(0)
			if task is not None:
				task = torch.tensor([task], device=self.device)
			if self.cfg.mpc:
				# if not eval_mode:
				# a = (a + std * torch.randn(self.cfg.action_dim, device=std.device)).clamp(-1, 1)
				a, std, mean = self.plan(obs, task=task)
				self.update_planner_mean(mean)
				if eval_mode:
					return a
				else:
					return (a + std * torch.randn(self.cfg.action_dim, device=std.device)).clamp(-1, 1)
			z = self.model.encode(obs, task)
			action, info = self.model.pi(z, task)
			if eval_mode:
				action = info["mean"]
			return action[0]

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
			action, _ = self.model.pi(z, task)
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
					pi_actions[t], _ = self.model.pi(_z, task)
					_z = self.model.next(_z, pi_actions[t], task)
				pi_actions[-1], _ = self.model.pi(_z, task)

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

			# Select first action from sampled distribution; add exploration noise if training
			rand_idx = math.gumbel_softmax_sample(score.squeeze(1))
			actions = torch.index_select(elite_actions, 1, rand_idx).squeeze(1)
			a, std = actions[0], std[0]
			self._prev_mean.copy_(mean)
			return a, std, mean

	def update_planner_mean(self, mean):
		self._prev_mean.copy_(mean)
		return 

	def calc_pi_losses(self, zs, task):
		with maybe_range('Agent/update_pi', self.cfg):
			action, info = self.model.pi(zs, task)
			qs = self.model.Q(zs, action, task, return_type='avg', detach=True)
			self.scale.update(qs[0])
			qs = self.scale(qs)

			# Loss is a weighted sum of Q-values
			rho = torch.pow(self.cfg.rho, torch.arange(len(qs), device=self.device))
			pi_loss = (-(self.cfg.entropy_coef * info["scaled_entropy"] + qs).mean(dim=(1,2)) * rho).mean()
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
		pi_loss, info = self.calc_pi_losses(zs, task)
		pi_loss.backward()
		pi_grad_norm = torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm)
		self.pi_optim.step()
		self.pi_optim.zero_grad(set_to_none=True)

		info = TensorDict({
			"pi_loss": pi_loss,
			"pi_grad_norm": pi_grad_norm,
			"pi_entropy": info["entropy"],
			"pi_scaled_entropy": info["scaled_entropy"],
			"pi_scale": self.scale.value,
		}, device=self.device)
		return info

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
		action, _ = self.model.pi(next_z, task)
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
		action, _ = self.model.pi(next_z, task)
		bootstrap = self.model.Q_aux(next_z, action, task, return_type='avg', target=self.cfg.auxiliary_value_ema, detach=False)  # (G_aux,T,B,K) scalar per gamma (single head)
		if bootstrap is None:
			return None
		aux_targets = []
		for g in range(G_aux):
			gamma = self._all_gammas[g+1]
			boot_g = bootstrap[g, ...]  # (T,B,1)
			aux_targets.append(reward + gamma * (1 - terminated) * boot_g)
		return torch.stack(aux_targets, dim=0) # (G_aux,T,B,1)


	def calc_wm_losses(self, obs, action, reward, terminated, task=None):
     # TODO no looping for calcing loss  +  fetching longer trajectories from buffer and using data overlap for increased efficiency
		with maybe_range('Agent/calc_td_target', self.cfg):
			with torch.no_grad():
					# Encode next observations (time steps 1..T) → latent sequence next_z
					next_z = self.model.encode(obs[1:], task)                 # (T,B,L)
					# Distributional TD targets (primary gamma) vs Q logits
					td_targets = self._td_target(next_z, reward, terminated, task)  # (T,B,K)
					# Auxiliary scalar TD targets per gamma (optional)
					aux_td_targets = self._td_target_aux(next_z, reward, terminated, task)  # (G_aux,T,B,1) or None

		# ------------------------------ Latent rollout (consistency) ------------------------------
		self.model.train()
		zs = torch.empty(self.cfg.horizon+1, self.cfg.batch_size, self.cfg.latent_dim, device=self.device)  # allocate (T+1,B,L)
		with maybe_range('Agent/latent_rollout', self.cfg):
			z = self.model.encode(obs[0], task)  # initial latent (B,L)
			zs[0] = z
			consistency_loss = 0.
			for t, (_a, _target_next_z) in enumerate(zip(action.unbind(0), next_z.unbind(0))):  # iterate T steps
				z = self.model.next(z, _a, task)            # model prediction z_{t+1}
				# Consistency MSE between predicted & encoded next latent
				consistency_loss = consistency_loss + F.mse_loss(z, _target_next_z) * (self.cfg.rho**t)
				zs[t+1] = z
		# ------------------------------ Model predictions for losses ------------------------------
		_zs = zs[:-1]                                 # (T,B,L) latents aligned with actions
		qs = self.model.Q(_zs, action, task, return_type='all')              # (Qe,T,B,K)
		reward_preds = self.model.reward(_zs, action, task)                 # (T,B,K)
		if self.cfg.episodic:
			termination_pred = self.model.termination(zs[1:], task, unnormalized=True)  # (T,B,1)
		else:
			termination_pred = None
		# Auxiliary logits (if enabled)
		q_aux_logits = None
		if aux_td_targets is not None:
			q_aux_logits = self.model.Q_aux(_zs, action, task, return_type='all')  # (G_aux,T,B,K)

		with maybe_range('Agent/order_losses', self.cfg):

			# ------------------------------ Vectorized loss computation ------------------------------
			T, B = reward_preds.shape[:2]
			K = reward_preds.shape[-1]
			# Per-time discount weights rho^t (T,)
			rho_pows = torch.pow(
				self.cfg.rho,
				torch.arange(T, device=self.device, dtype=reward_preds.dtype)
			)

			# Reward CE over all (T,B) at once -> shape (T,)
			rew_ce = math.soft_ce(
				reward_preds.reshape(T * B, K),
				reward.reshape(T * B, 1),
				self.cfg,
			)
			rew_ce = rew_ce.view(T, B, 1).mean(dim=1).squeeze(-1)  # (T,)
			reward_loss = (rho_pows * rew_ce).sum() / T

			# Value CE across ensemble heads (Qe,T,B) -> mean over B, sum over heads, weight by rho
			Qe = qs.shape[0]
			val_ce = math.soft_ce(
				qs.reshape(Qe * T * B, K),
				# expand td_targets to (Qe,T,B,1) then flatten
				td_targets.unsqueeze(0).expand(Qe, -1, -1, -1).reshape(Qe * T * B, 1),
				self.cfg,
			)
			val_ce = val_ce.view(Qe, T, B, 1).mean(dim=(0,2)).squeeze(-1)  # (T)
			value_loss = (val_ce * rho_pows).sum() / T

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
				aux_value_losses = (aux_ce * rho_pows.unsqueeze(0)).sum(dim=1) / T  # (G_aux,)

			# Normalize losses
			consistency_loss = consistency_loss / self.cfg.horizon
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

			# Total loss (auxiliary added with its own weight separate from value_coef)
			total_loss = (
				self.cfg.consistency_coef * consistency_loss +
				self.cfg.reward_coef * reward_loss +
				self.cfg.termination_coef * termination_loss +
				self.cfg.value_coef * value_loss +
				self.cfg.multi_gamma_loss_weight * aux_value_loss_mean
			)
		info = TensorDict({
			"consistency_loss": consistency_loss,
			"reward_loss": reward_loss,
			"value_loss": value_loss,
			"aux_value_loss_mean": aux_value_loss_mean,
			"termination_loss": termination_loss,
			"total_loss": total_loss,
		}, device=self.device, non_blocking=True)
  
		if aux_value_losses is not None:
			for g, loss_g in enumerate(aux_value_losses):
				gamma_val = self._all_gammas[g+1]
				info.update({f"aux_value_loss/g{g}_gamma{gamma_val:.4f}": loss_g}, non_blocking=True)
		if self.cfg.episodic:
			info.update(math.termination_statistics(torch.sigmoid(termination_pred[-1]), terminated[-1]), non_blocking=True)
		return total_loss, zs, info
  
	def _update(self, obs, action, reward, terminated, task=None):
		"""Single gradient update step.

		Args:
			obs: (T+1, B, *obs_shape)
			action: (T, B, A)
			reward: (T, B, 1) scalar rewards
			terminated: (T, B, 1) binary (0/1)
			task: (optional) task index tensor for multi-task mode
		"""
		with maybe_range('Agent/update', self.cfg):
			# ------------------------------ Targets (no grad) ------------------------------
			total_loss, zs, info = self.calc_wm_losses(obs, action, reward, terminated, task)

			# ------------------------------ Backprop & updates ------------------------------
			
			total_loss.backward()
			grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
			self.optim.step(); self.optim.zero_grad(set_to_none=True)
			info.update({
				"grad_norm": grad_norm,
			}, non_blocking=True)
			# Policy update (detached latent sequence)
			pi_info = self.update_pi(zs.detach(), task)
			self.model.soft_update_target_Q()

			# ------------------------------ Logging tensor dict ------------------------------
			self.model.eval()

			info.update(pi_info, non_blocking=True)
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
			# if obs.is_pinned():
			# 	log.info("obs is pinned")
			# else:
			# 	log.info("obs is not pinned")

    
		kwargs = {}
		if task is not None:
			kwargs["task"] = task
		torch.compiler.cudagraph_mark_step_begin()
		return self._update(obs, action, reward, terminated, **kwargs)
