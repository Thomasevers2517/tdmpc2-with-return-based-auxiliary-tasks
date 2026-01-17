import os
os.environ['MUJOCO_GL'] = os.getenv("MUJOCO_GL", 'egl')
os.environ['LAZY_LEGACY_OP'] = '0'
os.environ['TORCHDYNAMO_INLINE_INBUILT_NN_MODULES'] = "1"
# Debug: get proper CUDA error locations (disable for production)
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ['TORCH_USE_CUDA_DSA'] = '1'
# os.environ['TORCH_LOGS'] = "+recompiles"
import warnings
warnings.filterwarnings('ignore')
import logging
import torch

import hydra
from termcolor import colored

from common.parser import parse_cfg
from common.seed import set_seed
from common.buffer import Buffer
from envs import make_env
from tdmpc2 import TDMPC2
from torch.autograd.profiler import emit_nvtx
from trainer.offline_trainer import OfflineTrainer
from trainer.online_trainer import OnlineTrainer
from common.logger import Logger
from common.logger import get_logger

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')
torch._dynamo.config.capture_scalar_outputs = True
# torch.autograd.set_detect_anomaly(True)

@hydra.main(config_name='config', config_path='.')
def train(cfg: dict):
	"""
	Script for training single-task / multi-task TD-MPC2 agents.

	Most relevant args:
		`task`: task name (or mt30/mt80 for multi-task training)
		`model_size`: model size, must be one of `[1, 5, 19, 48, 317]` (default: 5)
		`steps`: number of training/environment steps (default: 10M)
		`seed`: random seed (default: 1)

	See config.yaml for a full list of args.

	Example usage:
	```
		$ python train.py task=mt80 model_size=48
		$ python train.py task=mt30 model_size=317
		$ python train.py task=dog-run steps=7000000
	```
	"""

	assert torch.cuda.is_available()
	assert cfg.steps > 0, 'Must train for at least 1 step.'
	cfg = parse_cfg(cfg)
	log = get_logger(__name__)
	set_seed(cfg.seed)
	log.info('Work dir: %s', cfg.work_dir)

	trainer_cls = OfflineTrainer if cfg.multitask else OnlineTrainer
	if getattr(cfg, 'nvtx_profiler', False):
		with emit_nvtx():
			trainer = trainer_cls(
				cfg=cfg,
				env=make_env(cfg),
				agent=TDMPC2(cfg),
				buffer=Buffer(cfg),
				logger=Logger(cfg),
			)
			trainer.train()
			log.info('Training completed successfully')
	else:
		trainer = trainer_cls(
			cfg=cfg,
			env=make_env(cfg),
			agent=TDMPC2(cfg),
			buffer=Buffer(cfg),
			logger=Logger(cfg),
		)
	trainer.train()
	log.info('Training completed successfully')
if __name__ == '__main__':
	train()
