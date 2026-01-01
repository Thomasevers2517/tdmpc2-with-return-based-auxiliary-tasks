from time import time

import numpy as np
import torch
from tensordict.tensordict import TensorDict
from trainer.base import Trainer
from common.logger import get_logger
from common.buffer import Buffer
log = get_logger(__name__)


class OnlineTrainer(Trainer):
	"""Trainer class for single-task online TD-MPC2 training."""

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		# Configure this module's logger level per cfg.debug
		get_logger(log.name, cfg=self.cfg)
		self._step = 0
		self._ep_idx = 0
		self._ep_rew = torch.tensor(0.0)
		self._ep_len = 0
		self._start_time = time()
		
		# Total gradient updates counter for frequency-based update control
		self._total_updates = 0
		
		# Validate pi_update_freq
		if self.cfg.pi_update_freq <= 0:
			raise ValueError(f"pi_update_freq must be > 0, got {self.cfg.pi_update_freq}")
		if self.cfg.value_update_freq <= 0 and self.cfg.value_update_freq != -1:
			raise ValueError(f"value_update_freq must be > 0 or -1, got {self.cfg.value_update_freq}")

		self.validation_buffer = Buffer(cfg=self.cfg, isTrainBuffer=False)
		self.recent_validation_buffer = Buffer(cfg=self.cfg, isTrainBuffer=False)
		
		# Optional mean head reduction eval buffers (created if eval_mean_head_reduce=true)
		if self.cfg.eval_mean_head_reduce:
			self.validation_all_mean_head_reduce_buffer = Buffer(cfg=self.cfg, isTrainBuffer=False)
			self.validation_recent_mean_head_reduce_buffer = Buffer(cfg=self.cfg, isTrainBuffer=False)
		else:
			self.validation_all_mean_head_reduce_buffer = None
			self.validation_recent_mean_head_reduce_buffer = None

	def common_metrics(self):
		"""Return a dictionary of current metrics."""
		elapsed_time = time() - self._start_time
		return dict(
			step=self._step,
			episode=self._ep_idx,
			elapsed_time=elapsed_time,
			steps_per_second=self._step / elapsed_time
		)

	def eval(self, mpc=True, eval_head_reduce: str = 'min'):
		"""Evaluate a TD-MPC2 agent.
		
		Args:
			mpc: If True, use MPC planning. If False, use policy-only.
			eval_head_reduce: Head reduction mode for eval ('min', 'mean', 'max'). Only used when mpc=True.
		"""
		ep_rewards, ep_successes, ep_lengths = [], [], []
		ep_elite_std, ep_elite_mean = [], []
		# self.validation_buffer.empty()
  
		for i in range(self.cfg.eval_episodes):
			obs, done, ep_reward, t = self.env.reset(), False, 0, 0
			self.val_tds = [self.to_td(obs)]
			if self.cfg.save_video:
				self.logger.video.init(self.env, enabled=(i==0))
			while not done:
				torch.compiler.cudagraph_mark_step_begin()
				obs = obs.to(self.agent.device, non_blocking=True).unsqueeze(0)
				action, planner_info = self.agent.act(obs, eval_mode=True, mpc=mpc, eval_head_reduce=eval_head_reduce)
				action = action.cpu()
				obs, reward, done, info = self.env.step(action)
				ep_reward += reward
				t += 1
				if self.cfg.save_video:
					self.logger.video.record(self.env)

				val_td = self.to_td(obs, action, reward, info['terminated'])
				self.val_tds.append(val_td)
     
			ep_rewards.append(ep_reward)
			ep_successes.append(info['success'])
			ep_lengths.append(t)
			if self.cfg.save_video:
				self.logger.video.save(self._step)
			if mpc:
				# Add to appropriate validation buffers based on head reduction mode
				if eval_head_reduce == 'min':
					self.validation_buffer.add(torch.cat(self.val_tds), end_episode=True)
					self.recent_validation_buffer.add(torch.cat(self.val_tds), end_episode=True)
				elif eval_head_reduce == 'mean' and self.validation_all_mean_head_reduce_buffer is not None:
					self.validation_all_mean_head_reduce_buffer.add(torch.cat(self.val_tds), end_episode=True)
					self.validation_recent_mean_head_reduce_buffer.add(torch.cat(self.val_tds), end_episode=True)
		if mpc and eval_head_reduce == 'min':
			# Return that validation should run after all evals complete
			# (validation is now called from train() after all eval modes finish)
			pass
	
		return dict(
			episode_reward=np.nanmean(ep_rewards),
			episode_success=np.nanmean(ep_successes),
			episode_length= np.nanmean(ep_lengths),
			episode_elite_std=np.nanmean(ep_elite_std),
			episode_elite_mean=np.nanmean(ep_elite_mean),

		)

	def to_td(self, obs, action=None, reward=None, terminated=None):
		"""Creates a TensorDict for a new episode."""
     
		if isinstance(obs, dict):
			obs = TensorDict(obs, batch_size=())
		else:
			obs = obs.unsqueeze(0)
		if action is None:
			action = torch.full_like(self.env.rand_act(), float('nan'))
		if reward is None:
			reward = torch.tensor(float('nan'))
		if terminated is None:
			terminated = torch.tensor(float('nan'))
		td = TensorDict(
			obs=obs,
			action=action.unsqueeze(0),
			reward=reward.unsqueeze(0),
			terminated=terminated.unsqueeze(0),
		batch_size=(1,))
		return td

	def train(self):
		"""Train a TD-MPC2 agent."""
		# Ensure trainer logger level matches cfg.debug
		get_logger(__name__, cfg=self.cfg)
		train_metrics, done, eval_next = {}, True, False
		while self._step <= self.cfg.steps:
			# Logging cadence flags
			detail_freq = self.cfg.log_detail_freq
			detailed_log_flag = (self._step % detail_freq == 0) or (self._step == self.cfg.steps)
			basic_log_flag = (self._step % self.cfg.log_freq == 0) or detailed_log_flag
			# Evaluate agent periodically, but never at step 0
			if self._step != 0 and (self._step % self.cfg.eval_freq == 0):
				eval_next = True

			# Reset environment
			if done:
				if eval_next:
					# Default eval with min head reduction
					eval_metrics = self.eval(mpc=True, eval_head_reduce='min')
					eval_metrics.update(self.common_metrics())
					self.logger.log(eval_metrics, 'eval')
					eval_next = False

					# Policy-only eval (no MPC)
					policy_eval_metrics = self.eval(mpc=False)
					policy_eval_metrics.update(self.common_metrics())
					self.logger.log(policy_eval_metrics, 'policy_eval')
					
					# Optional: mean head reduction eval
					if self.cfg.eval_mean_head_reduce:
						eval_mean_metrics = self.eval(mpc=True, eval_head_reduce='mean')
						eval_mean_metrics.update(self.common_metrics())
						self.logger.log(eval_mean_metrics, 'eval_mean_head_reduce')
					
					# Run validation on all buffers after all evals complete
					val_info_rand, val_info_recent, val_info_mean_all, val_info_mean_recent = self.validate()
					val_info_rand.update(self.common_metrics())
					val_info_recent.update(self.common_metrics())
					self.logger.log(val_info_rand, 'validation_all')
					self.logger.log(val_info_recent, 'validation_recent')
					# Log mean head reduce validation if buffers exist and were populated
					if val_info_mean_all is not None:
						val_info_mean_all.update(self.common_metrics())
						val_info_mean_recent.update(self.common_metrics())
						self.logger.log(val_info_mean_all, 'validation_all_mean_head_reduce')
						self.logger.log(val_info_mean_recent, 'validation_recent_mean_head_reduce')
     

				if self._step > 0:
					if info['terminated'] and not self.cfg.episodic:
						raise ValueError('Termination detected but you are not in episodic mode. ' \
						'Set `episodic=true` to enable support for terminations.')
					self._ep_rew += torch.tensor([td['reward'] for td in self._tds]).sum()
					self._ep_len += len(self._tds)
					train_metrics.update(
						episode_reward=self._ep_rew,
						episode_success=info['success'],
						episode_length=self._ep_len,
						episode_terminated=info['terminated'])
					train_metrics.update(self.common_metrics())
					self.logger.log(train_metrics, 'train')
					self._ep_idx = self.buffer.add(torch.cat(self._tds), end_episode=True)
				self._ep_rew = torch.tensor(0.0)
				self._ep_len = 0
				obs = self.env.reset()
				self._tds = []
				self.agent.reset_planner_state()
			elif (self._step % self.cfg.buffer_update_interval == 0 and self._step > 0) and self.cfg.buffer_update_interval !=-1:
				self._ep_idx = self.buffer.add(torch.cat(self._tds), end_episode=False)
				self._ep_rew += torch.tensor([td['reward'] for td in self._tds]).sum()
				self._ep_len += len(self._tds)
				self._tds = [] 

			# Collect experience
			if self._step > self.cfg.seed_steps:
				obs = obs.to(self.agent.device, non_blocking=True).unsqueeze(0)
				action, planner_info = self.agent.act(obs, mpc=self.cfg.train_mpc)
				action = action.cpu()
				if basic_log_flag:
					self.logger.log_planner_info(planner_info, self._step, category='train')
			else:
				action = self.env.rand_act()
			obs, reward, done, info = self.env.step(action)
			self._tds.append(self.to_td(obs, action, reward, info['terminated']))

			# Update agent
			if self._step >= self.cfg.seed_steps:
				if self._step == self.cfg.seed_steps:
					base_updates = self.cfg.seed_steps
					log.info('Pretraining agent on seed data...')
				else:
					base_updates = self.cfg.utd_ratio
				
				# All frequencies are in units of "updates per environment step":
				# - utd_ratio (base_updates): WM updates per env step
				# - value_update_freq: value updates per env step (-1 means same as UTD)
				# - pi_update_freq: policy updates per env step
				wm_updates_per_step = base_updates
				value_updates_per_step = base_updates if self.cfg.value_update_freq == -1 else self.cfg.value_update_freq
				pi_updates_per_step = self.cfg.pi_update_freq
				
				# num_updates = max updates needed for any component
				num_updates = int(max(wm_updates_per_step, value_updates_per_step, pi_updates_per_step))
				
				for _ in range(num_updates):
					self._total_updates += 1
					
					# Each component updates when: total_updates % (num_updates / component_freq) == 0
					# This ensures exactly component_freq updates per env step, evenly spaced.
					update_world_model = (self._total_updates % int(num_updates / wm_updates_per_step) == 0)
					update_value = (self._total_updates % int(num_updates / value_updates_per_step) == 0)
					update_pi = (self._total_updates % int(num_updates / pi_updates_per_step) == 0)
					
					_train_metrics = self.agent.update(
						self.buffer, step=self._step,
						update_value=update_value, update_pi=update_pi,
						update_world_model=update_world_model
					)
					train_metrics.update(_train_metrics)
					if self.cfg.debug:
						log.info('update step=%d total=%d wm=%s val=%s pi=%s',
								 self._step, self._total_updates,
								 update_world_model, update_value, update_pi)

			self._step += 1

		self.logger.finish(self.agent)

	def validate(self):
		"""Validate on all validation buffers (min head reduce + optionally mean head reduce)."""
		import math
		num_batches_recent = math.floor(self.cfg.eval_episodes * self.cfg.episode_length / self.cfg.batch_size)
		
		# Min head reduce buffers (always present)
		random_val_info = self.agent.validate(self.validation_buffer, num_batches=1)
		val_info_recent = self.agent.validate(self.recent_validation_buffer, num_batches=num_batches_recent)
		self.recent_validation_buffer.empty()
		
		# Mean head reduce buffers (optional)
		val_info_mean_all = None
		val_info_mean_recent = None
		if self.validation_all_mean_head_reduce_buffer is not None:
			val_info_mean_all = self.agent.validate(self.validation_all_mean_head_reduce_buffer, num_batches=1)
			val_info_mean_recent = self.agent.validate(self.validation_recent_mean_head_reduce_buffer, num_batches=num_batches_recent)
			self.validation_recent_mean_head_reduce_buffer.empty()
		
		return random_val_info, val_info_recent, val_info_mean_all, val_info_mean_recent