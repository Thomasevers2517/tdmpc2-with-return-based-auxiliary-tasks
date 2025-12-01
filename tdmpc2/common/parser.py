import dataclasses
import re
import os
from datetime import datetime
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
	# Logging directory structure: logs/<timestamp>/<task>/<seed>/<exp_name>
	# Optional override: set env LOG_TIMESTAMP to reuse same timestamp across multiple processes.
	log_ts = os.environ.get('LOG_TIMESTAMP')
	if not log_ts:
		log_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
		os.environ['LOG_TIMESTAMP'] = log_ts  # propagate to child processes
	# Note: we avoid attaching new attribute (log_timestamp) to cfg to keep struct safety.
	cfg.work_dir = Path(hydra.utils.get_original_cwd()) / 'logs' / log_ts / cfg.task / str(cfg.seed) / cfg.exp_name
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

	# ----------------------------------------------------------------------
	# Fail-fast configuration constraints (centralized here)
	# ----------------------------------------------------------------------
	# 1) If buffer_update_interval is enabled (!= -1), ensure seed_steps > interval
	#    so that at least one buffer write occurs before updates begin.
	if not OmegaConf.is_missing(cfg, 'buffer_update_interval') and not OmegaConf.is_missing(cfg, 'seed_steps'):
		if cfg.buffer_update_interval != -1:
			assert (
				cfg.seed_steps > cfg.buffer_update_interval
			), (
				f"seed_steps ({cfg.seed_steps}) must be > buffer_update_interval ({cfg.buffer_update_interval}) "
				"when interval is enabled."
			)

	# 2) As requested: enforce seed_steps < episode_length.
	#    Note: with buffer_update_interval == -1, this may reduce pretrain data if episodes are long.
	#    This is intentional per current project policy and can be revisited.
	if not OmegaConf.is_missing(cfg, 'episode_length') and not OmegaConf.is_missing(cfg, 'seed_steps'):
		if cfg.episode_length is not None:
			assert (
				cfg.seed_steps < cfg.episode_length
			), (
				f"seed_steps ({cfg.seed_steps}) must be < episode_length ({cfg.episode_length})."
			)

	assert (cfg.value_coef > 0 )

	# Planner policy seed noise std must be non-negative.
	if hasattr(cfg, 'policy_seed_noise_std'):
		if cfg.policy_seed_noise_std < 0.0:
			raise ValueError(f"policy_seed_noise_std must be >= 0.0, got {cfg.policy_seed_noise_std}.")

	# Planner value head reduction mode must be explicitly configured.
	if not hasattr(cfg, 'planner_head_reduce'):
		raise AttributeError("Missing cfg.planner_head_reduce; expected 'mean' or 'max'.")
	if cfg.planner_head_reduce not in {"mean", "max"}:
		raise ValueError(
			f"Invalid planner_head_reduce '{cfg.planner_head_reduce}'. Expected 'mean' or 'max'."
		)

	# Policy head reduction mode for policy loss computation.
	if not hasattr(cfg, 'policy_head_reduce'):
		raise AttributeError("Missing cfg.policy_head_reduce; expected 'mean', 'min', or 'max'.")
	if cfg.policy_head_reduce not in {"mean", "min", "max"}:
		raise ValueError(
			f"Invalid policy_head_reduce '{cfg.policy_head_reduce}'. Expected 'mean', 'min', or 'max'."
		)

	# Number of reward heads must be positive.
	if not hasattr(cfg, 'num_reward_heads'):
		cfg.num_reward_heads = 1  # Default for backward compatibility
	if cfg.num_reward_heads < 1:
		raise ValueError(f"num_reward_heads must be >= 1, got {cfg.num_reward_heads}.")

	if cfg.planner_lambda_disagreement == 0:
		if cfg.planner_num_dynamics_heads > 1:
			if cfg.planner_head_reduce != "max":
				print("Warning: planner_num_dynamics_heads > 1 has no effect when planner_lambda_disagreement == 0. Planner valuehead reduce is also not max but mean, so just avging across heads. Setting planner_num_dynamics_heads = 1 to save computation. ")
				cfg.planner_num_dynamics_heads = 1
			print("Keeping multiple dynamics heads because taking max among them for exploration. Planner lambda_disagreement is zero tho")

 
	if cfg.final_rho != -1:
		import math as _math
		cfg.rho = _math.pow(cfg.final_rho, 1 / cfg.horizon)
		print(f"Overriding rho schedule to end at final_rho = {cfg.final_rho} at horizon = {cfg.horizon}, setting rho = {cfg.rho:.6f}")
	return cfg_to_dataclass(cfg)
