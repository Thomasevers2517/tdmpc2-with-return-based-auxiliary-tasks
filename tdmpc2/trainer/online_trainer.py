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

		self.validation_buffer = Buffer(cfg=self.cfg)
		self.recent_validation_buffer = Buffer(cfg=self.cfg)

	def common_metrics(self):
		"""Return a dictionary of current metrics."""
		elapsed_time = time() - self._start_time
		return dict(
			step=self._step,
			episode=self._ep_idx,
			elapsed_time=elapsed_time,
			steps_per_second=self._step / elapsed_time
		)

	def eval(self, mpc=True):
		"""Evaluate a TD-MPC2 agent."""
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
				action, planner_info = self.agent.act(obs, eval_mode=True, mpc=mpc)
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
				self.validation_buffer.add(torch.cat(self.val_tds), end_episode=True)
				self.recent_validation_buffer.add(torch.cat(self.val_tds), end_episode=True)
		if mpc:
			val_info_rand, val_info_recent = self.validate()
			val_info_rand.update(self.common_metrics())
			val_info_recent.update(self.common_metrics())
			self.logger.log(val_info_rand, 'validation_all')
			self.logger.log(val_info_recent, 'validation_recent') 
	
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
					eval_metrics = self.eval(mpc=True)
					eval_metrics.update(self.common_metrics())
					self.logger.log(eval_metrics, 'eval')
					eval_next = False

					policy_eval_metrics = self.eval(mpc=False)
					policy_eval_metrics.update(self.common_metrics())
					self.logger.log(policy_eval_metrics, 'policy_eval')
     

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
					num_updates = self.cfg.seed_steps
					log.info('Pretraining agent on seed data...')
				else:
					num_updates = self.cfg.utd_ratio
				for update_idx in range(num_updates):

					for ac_idx in range(self.cfg.ac_utd_multiplier-1):
						_train_metrics = self.agent.update(self.buffer, step=self._step, ac_only=True)
						train_metrics.update(_train_metrics)
						if self.cfg.debug:
							log.info('update step=%d inner_ac=%d/%d (ac_only)', self._step, ac_idx+1, self.cfg.ac_utd_multiplier-1)
					_train_metrics = self.agent.update(self.buffer, step=self._step, ac_only=False)
					train_metrics.update(_train_metrics)
					if self.cfg.debug:
						log.info('update step=%d main_update=%d/%d', self._step, update_idx+1, num_updates)


			# Network reset removed per refactor; maintain steady training without periodic resets.
					

			self._step += 1

		self.logger.finish(self.agent)

	def validate(self):
		import math
		random_val_info= self.agent.validate(self.validation_buffer, num_batches=1)
		val_info_recent = self.agent.validate(self.recent_validation_buffer, num_batches=math.floor(self.cfg.eval_episodes*self.cfg.episode_length / self.cfg.batch_size))
		self.recent_validation_buffer.empty()
		return random_val_info, val_info_recent