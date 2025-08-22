import dataclasses
import re
from pathlib import Path
from typing import Any

import hydra
from omegaconf import OmegaConf

from common import MODEL_SIZE, TASK_SET


def cfg_to_dataclass(cfg, frozen=False):
	"""
	Converts an OmegaConf config to a dataclass object.
	This prevents graph breaks when used with torch.compile.
	"""
	cfg_dict = OmegaConf.to_container(cfg)
	fields = []
	for key, value in cfg_dict.items():
		fields.append((key, Any, dataclasses.field(default_factory=lambda value_=value: value_)))
	dataclass_name = "Config"
	dataclass = dataclasses.make_dataclass(dataclass_name, fields, frozen=frozen)
	def get(self, val, default=None):
		return getattr(self, val, default)
	dataclass.get = get
	return dataclass()


def parse_cfg(cfg: OmegaConf) -> OmegaConf:
	"""
	Parses a Hydra config. Mostly for convenience.
	"""

	# Logic
	for k in cfg.keys():
		try:
			v = cfg[k]
			if v == None:
				v = True
		except:
			pass

	# Algebraic expressions
	for k in cfg.keys():
		try:
			v = cfg[k]
			if isinstance(v, str):
				match = re.match(r"(\d+)([+\-*/])(\d+)", v)
				if match:
					cfg[k] = eval(match.group(1) + match.group(2) + match.group(3))
					if isinstance(cfg[k], float) and cfg[k].is_integer():
						cfg[k] = int(cfg[k])
		except:
			pass

	# Convenience
	cfg.work_dir = Path(hydra.utils.get_original_cwd()) / 'logs' / cfg.task / str(cfg.seed) / cfg.exp_name
	cfg.task_title = cfg.task.replace("-", " ").title()
	cfg.bin_size = (cfg.vmax - cfg.vmin) / (cfg.num_bins-1) # Bin size for discrete regression

	# Model size
	if cfg.get('model_size', None) is not None:
		assert cfg.model_size in MODEL_SIZE.keys(), \
			f'Invalid model size {cfg.model_size}. Must be one of {list(MODEL_SIZE.keys())}'
		for k, v in MODEL_SIZE[cfg.model_size].items():
			cfg[k] = v
		if cfg.task == 'mt30' and cfg.model_size == 19:
			cfg.latent_dim = 512 # This checkpoint is slightly smaller

	# Multi-task
	cfg.multitask = cfg.task in TASK_SET.keys()
	if cfg.multitask:
		cfg.task_title = cfg.task.upper()
		# Account for slight inconsistency in task_dim for the mt30 experiments
		cfg.task_dim = 96 if cfg.task == 'mt80' or cfg.get('model_size', 5) in {1, 317} else 64
	else:
		cfg.task_dim = 0
	cfg.tasks = TASK_SET.get(cfg.task, [cfg.task])

	# ----------------------------------------------------------------------
	# Multi-gamma configuration (flattened keys)
	# ----------------------------------------------------------------------
	# We keep keys flattened (multi_gamma_*) instead of a nested dict to
	# simplify conversion to a torch.compile-safe dataclass (attribute access
	# is cheaper / avoids nested structure graph breaks). The base config
	# sets defaults; here we just guarantee presence & perform lightweight
	# static validation that does not depend on the agent state.
	# Added keys:
	#   multi_gamma_gammas           : list[float] of *auxiliary* discount factors (primary excluded)
	#   multi_gamma_head             : 'joint' | 'separate' head style
	#   multi_gamma_loss_weight      : λ applied to mean auxiliary loss (excludes primary)
	#   multi_gamma_debug_logging    : toggles verbose diagnostics
	#   multi_gamma_log_num_examples : sample count for debug snapshots
	# NOTE: Actual assertion that gammas[0] == primary discount occurs later
	# in the agent (where the computed heuristic discount is accessible).
	# ----------------------------------------------------------------------
	if not hasattr(cfg, 'multi_gamma_gammas'):
		cfg.multi_gamma_gammas = []
	if not hasattr(cfg, 'multi_gamma_head'):
		cfg.multi_gamma_head = 'joint'
	if not hasattr(cfg, 'multi_gamma_loss_weight'):
		cfg.multi_gamma_loss_weight = 0.5
	if not hasattr(cfg, 'multi_gamma_debug_logging'):
		cfg.multi_gamma_debug_logging = False
	if not hasattr(cfg, 'multi_gamma_log_num_examples'):
		cfg.multi_gamma_log_num_examples = 8

	# Basic syntactic validation (cannot depend on agent internals yet)
	if hasattr(cfg, 'multi_gamma_gammas') and cfg.multi_gamma_gammas:
		gammas = list(cfg.multi_gamma_gammas)
		assert len(gammas) <= 6, f'multi_gamma supports at most 6 auxiliary gammas (got {len(gammas)}).'
		assert cfg.multi_gamma_head in {'joint', 'separate'}, f"Invalid multi_gamma_head {cfg.multi_gamma_head}."

	return cfg_to_dataclass(cfg)
