"""Distracted DMControl integration.

Adds path shim so vendored `distracting_control` package (which uses absolute
intra-package imports like `from distracting_control import background`) works
without editable install. This keeps upstream code unchanged while allowing
`suite.load` to be imported.
"""

import os, sys

# Compute path to the inner package root containing suite.py, background.py, etc.
_PKG_ROOT = os.path.join(os.path.dirname(__file__), 'custom_envs', 'distracting_control')
if _PKG_ROOT not in sys.path:
	sys.path.insert(0, _PKG_ROOT)

try:
	from distracting_control import suite as distracted_suite # type: ignore
except Exception as e:  # Fallback: still allow earlier ValueError path in env factory
	raise ImportError(f"Failed to import distracting_control.suite after adding {_PKG_ROOT} to sys.path: {e}")
from collections import defaultdict, deque

import gymnasium as gym
import numpy as np
import torch

from envs.tasks import cheetah, walker, hopper, reacher, ball_in_cup, pendulum, fish
distracted_suite.suite.ALL_TASKS = distracted_suite.suite.ALL_TASKS + distracted_suite.suite._get_tasks('custom')
distracted_suite.suite.TASKS_BY_DOMAIN = distracted_suite.suite._get_tasks_by_domain(distracted_suite.suite.ALL_TASKS)
from dm_control.suite.wrappers import action_scale

from envs.wrappers.timeout import Timeout
import logging
log = logging.getLogger(__name__)

def get_obs_shape(env):
	obs_shp = []
	for v in env.observation_spec().values():
		try:
			shp = np.prod(v.shape)
		except:
			shp = 1
		obs_shp.append(shp)
	return (int(np.sum(obs_shp)),)


class DMControlWrapper:
	def __init__(self, env, domain):
		self.env = env
		self.camera_id = 2 if domain == 'quadruped' else 0
		obs_shape = get_obs_shape(env)
		action_shape = env.action_spec().shape
		self.observation_space = gym.spaces.Box(
			low=np.full(obs_shape, -np.inf, dtype=np.float32),
			high=np.full(obs_shape, np.inf, dtype=np.float32),
			dtype=np.float32)
		self.action_space = gym.spaces.Box(
			low=np.full(action_shape, env.action_spec().minimum),
			high=np.full(action_shape, env.action_spec().maximum),
			dtype=env.action_spec().dtype)
		self.action_spec_dtype = env.action_spec().dtype

	@property
	def unwrapped(self):
		return self.env
	
	def _obs_to_array(self, obs):
		return torch.from_numpy(
			np.concatenate([v.flatten() for v in obs.values()], dtype=np.float32))
	
	def reset(self):
		return self._obs_to_array(self.env.reset().observation)

	def step(self, action):
		reward = 0
		action = action.astype(self.action_spec_dtype)
		for _ in range(2):
			step = self.env.step(action)
			reward += step.reward
		return self._obs_to_array(step.observation), reward, False, defaultdict(float)
	
	def render(self, width=384, height=384, camera_id=None):
		return self.env.physics.render(height, width, camera_id or self.camera_id)


class Pixels(gym.Wrapper):
	def __init__(self, env, cfg, num_frames=3, size=64):
		super().__init__(env)
		self.cfg = cfg
		self.env = env
		self.observation_space = gym.spaces.Box(
			low=0, high=255, shape=(num_frames*3, size, size), dtype=np.uint8)
		self._frames = deque([], maxlen=num_frames)
		self._size = size

	def _get_obs(self, is_reset=False):
		frame = self.env.render(width=self._size, height=self._size).transpose(2, 0, 1)
		num_frames = self._frames.maxlen if is_reset else 1
		for _ in range(num_frames):
			self._frames.append(frame)
		return torch.from_numpy(np.concatenate(self._frames))

	def reset(self):
		self.env.reset()
		return self._get_obs(is_reset=True)

	def step(self, action):
		_, reward, done, info = self.env.step(action)
		return self._get_obs(), reward, done, info


def make_env(cfg):
	log.info('Trying to create Distracted DM Control environment')
	domain, task = cfg.task.replace('-', '_').split('_', 1)
	if not domain.startswith('distracted'):
		raise ValueError(f'Expected distracted<domain>-<task> format, got: {cfg.task}')
	log.info('Creating Distracted DM Control environment: %s-%s', domain, task)
	domain = domain[len('distracted'):]
 
	domain = dict(cup='ball_in_cup', pointmass='point_mass').get(domain, domain)
	if (domain, task) not in distracted_suite.suite.ALL_TASKS:
		raise ValueError('Unknown task:', task)
	assert cfg.obs in {'state', 'rgb'}, 'This task only supports state and rgb observations.'

	env = distracted_suite.load(domain,
					 task,
					 task_kwargs={'random': cfg.seed},
					 # Use absolute path to vendored DAVIS frames so background wrapper finds videos.
					 background_dataset_path=cfg.davis_dataset_path,
					 visualize_reward=False,
					dynamic=cfg.distracted_dynamic,
					difficulty=cfg.distracted_difficulty)
 
	env = action_scale.Wrapper(env, minimum=-1., maximum=1.)
	env = DMControlWrapper(env, domain)
	if cfg.obs == 'rgb':
		env = Pixels(env, cfg)
	env = Timeout(env, max_episode_steps=500)
	return env
