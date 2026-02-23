from copy import deepcopy
import warnings

import gymnasium as gym

from envs.wrappers.multitask import MultitaskWrapper
from envs.wrappers.tensor import TensorWrapper
from common.logger import get_logger

log = get_logger(__name__)

def missing_dependencies(task):
	raise ValueError(f'Missing dependencies for task {task}; install dependencies to use this environment.')

try:
	from envs.dmcontrol import make_env as make_dm_control_env
except Exception as e:
	print(e, flush=True)
	make_dm_control_env = missing_dependencies
try:
	from envs.maniskill import make_env as make_maniskill_env
except:
	make_maniskill_env = missing_dependencies
try:
	from envs.metaworld import make_env as make_metaworld_env
except:
	make_metaworld_env = missing_dependencies
try:
	from envs.myosuite import make_env as make_myosuite_env
except:
	make_myosuite_env = missing_dependencies
try:
	from envs.mujoco import make_env as make_mujoco_env
except:
	make_mujoco_env = missing_dependencies
try:
    from envs.distracteddmcontrol import make_env as make_distracting_control_env
except Exception as e:
    print(e, flush=True)
    make_distracting_control_env = missing_dependencies
try:
	from envs.humanoid import make_env as make_humanoid_env
except:
	make_humanoid_env = missing_dependencies

warnings.filterwarnings('ignore', category=DeprecationWarning)


def make_multitask_env(cfg):
	"""
	Make a multi-task environment for TD-MPC2 experiments.
	"""
	log.info('Creating multi-task environment with tasks: %s', cfg.tasks)
	envs = []
	for task in cfg.tasks:
		_cfg = deepcopy(cfg)
		_cfg.task = task
		_cfg.multitask = False
		env = make_env(_cfg)
		if env is None:
			raise ValueError('Unknown task:', task)
		envs.append(env)
	env = MultitaskWrapper(cfg, envs)
	cfg.obs_shapes = env._obs_dims
	cfg.action_dims = env._action_dims
	cfg.episode_lengths = env._episode_lengths
	return env
	

def make_env(cfg):
	"""
	Make an environment for TD-MPC2 experiments.
	"""
	gym.logger.set_level(40)
	env = None
	for fn in [make_distracting_control_env, make_dm_control_env, make_maniskill_env, make_metaworld_env, make_myosuite_env, make_mujoco_env, make_humanoid_env]:
		try:
			log.info('Trying to make environment with %s', fn.__module__)
			env = fn(cfg)
			log.info('Making environment with %s', fn.__module__)
		except ValueError:
			pass
	if env is None:
		raise ValueError(f'Failed to make environment "{cfg.task}": please verify that dependencies are installed and that the task exists.')
	env = TensorWrapper(env)
	try: # Dict
		cfg.obs_shape = {k: v.shape for k, v in env.observation_space.spaces.items()}
		log.info('Observation spaces: %s', cfg.obs_shape)
	except: # Box
		cfg.obs_shape = {cfg.get('obs', 'state'): env.observation_space.shape}
	cfg.action_dim = env.action_space.shape[0]
	cfg.episode_length = env.max_episode_steps
	if not hasattr(cfg, 'seed_steps') or cfg.seed_steps is None or type(cfg.seed_steps) != int:
		cfg.seed_steps = max(1000, 5*cfg.episode_length)
	return env
