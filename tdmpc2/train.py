import os
os.environ['MUJOCO_GL'] = os.getenv("MUJOCO_GL", 'egl')
os.environ['LAZY_LEGACY_OP'] = '0'
os.environ['TORCHDYNAMO_INLINE_INBUILT_NN_MODULES'] = "1"
os.environ['TORCH_LOGS'] = "+recompiles"
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
# torch.autograd.set_detect_anomaly(True)


def _check_and_override_latent_dim(cfg: dict) -> None:
	"""Ensure cfg.latent_dim matches the RGB encoder output when needed.

	When ``cnn_use_projection`` is False and observations are RGB, the CNN
	encoders return raw flattened features instead of projecting to the
	configured ``latent_dim``. In that case, we run a lightweight dummy
	forward through the RGB encoder to infer the effective feature size and
	override ``cfg.latent_dim`` accordingly so that downstream modules
	(WorldModel, critics, policy) are built with consistent dimensions.

	This function assumes that ``make_env(cfg)`` has already been called so
	that ``cfg.obs_shape`` is populated by the environment.
	"""
	if getattr(cfg, 'obs', None) != 'rgb':
		return
	if getattr(cfg, 'cnn_use_projection', True):
		return
	try:
		from common import layers
		import torch as _torch
		# Only act if env has provided an RGB obs_shape.
		if hasattr(cfg, 'obs_shape') and 'rgb' in cfg.obs_shape:
			encoders = layers.enc(cfg, out={})
			if 'rgb' in encoders:
				encoder_rgb = encoders['rgb']
				in_shape = cfg.obs_shape['rgb']
				x = _torch.zeros(1, *in_shape)
				with _torch.no_grad():
					latent = encoder_rgb(x)
				enc_out_dim = int(latent.shape[-1])
				if hasattr(cfg, 'latent_dim') and cfg.latent_dim != enc_out_dim:
					print(
						f"[train] Overriding cfg.latent_dim from {cfg.latent_dim} "
						f"to encoder output dim {enc_out_dim} for RGB encoder (cnn_use_projection=False)."
					)
				cfg.latent_dim = enc_out_dim
	except Exception as e:  # pragma: no cover - fail-fast path
		raise RuntimeError(
			f"Failed to infer encoder latent_dim from RGB encoder in train(): {e}"
		) from e

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

	# Build environment first so that cfg.obs_shape / cfg.action_dim are
	# populated. Then reconcile cfg.latent_dim with the encoder output
	# dimension if we have disabled CNN projection.
	env = make_env(cfg)
	_check_and_override_latent_dim(cfg)

	trainer_cls = OfflineTrainer if cfg.multitask else OnlineTrainer
	if getattr(cfg, 'nvtx_profiler', False):
		with emit_nvtx():
			trainer = trainer_cls(
				cfg=cfg,
				env=env,
				agent=TDMPC2(cfg),
				buffer=Buffer(cfg),
				logger=Logger(cfg),
			)
			trainer.train()
			log.info('Training completed successfully')
	else:
		trainer = trainer_cls(
			cfg=cfg,
			env=env,
			agent=TDMPC2(cfg),
			buffer=Buffer(cfg),
			logger=Logger(cfg),
		)
	trainer.train()
	log.info('Training completed successfully')
if __name__ == '__main__':
	train()
