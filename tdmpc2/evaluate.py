import os
os.environ['MUJOCO_GL'] = os.getenv("MUJOCO_GL", 'egl')
import warnings
warnings.filterwarnings('ignore')

import hydra
import imageio
import numpy as np
import torch

from common.parser import parse_cfg
from common.seed import set_seed
from envs import make_env
from tdmpc2 import TDMPC2
from common.logger import get_logger

torch.backends.cudnn.benchmark = True


@hydra.main(config_name='config', config_path='.')
def evaluate(cfg: dict):
	"""
	Script for evaluating a single-task TD-MPC2 checkpoint.

	Most relevant args:
		`task`: task name
		`checkpoint`: path to model checkpoint to load
		`eval_episodes`: number of episodes to evaluate on (default: 10)
		`save_video`: whether to save a video of the evaluation (default: True)
		`seed`: random seed (default: 1)

	See config.yaml for a full list of args.

	Example usage:
		$ python evaluate.py task=dog-run checkpoint=/path/to/dog-1.pt save_video=true
	"""
	assert torch.cuda.is_available()
	assert cfg.eval_episodes > 0, 'Must evaluate at least 1 episode.'
	cfg = parse_cfg(cfg)
	log = get_logger(__name__)
	set_seed(cfg.seed)
	log.info('Task: %s', cfg.task)
	log.info('Checkpoint: %s', cfg.checkpoint)

	# Make environment
	env = make_env(cfg)

	# Load agent
	agent = TDMPC2(cfg)
	assert os.path.exists(cfg.checkpoint), f'Checkpoint {cfg.checkpoint} not found! Must be a valid filepath.'
	agent.load(cfg.checkpoint)

	# Evaluate
	log.info('Evaluating agent on %s', cfg.task)
	if cfg.save_video:
		video_dir = os.path.join(cfg.work_dir, 'videos')
		os.makedirs(video_dir, exist_ok=True)
	ep_rewards, ep_successes = [], []
	for i in range(cfg.eval_episodes):
		obs, done, ep_reward, t = env.reset(), False, 0, 0
		if cfg.save_video:
			frames = [env.render()]
		while not done:
			action = agent.act(obs, eval_mode=True).cpu()
			obs, reward, done, info = env.step(action)
			ep_reward += reward
			t += 1
			if cfg.save_video:
				frames.append(env.render())
		ep_rewards.append(ep_reward)
		ep_successes.append(info['success'])
		if cfg.save_video:
			imageio.mimsave(
				os.path.join(video_dir, f'{cfg.task}-{i}.mp4'), frames, fps=15)
	log.info('  %-22s\tR: %.01f\tS: %.02f',
			 cfg.task, np.mean(ep_rewards), np.mean(ep_successes))


if __name__ == '__main__':
	evaluate()
