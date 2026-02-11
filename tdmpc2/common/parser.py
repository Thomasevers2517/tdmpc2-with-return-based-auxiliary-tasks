import dataclasses
import re
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
from omegaconf import OmegaConf, open_dict

from common import MODEL_SIZE


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

	# Multi-task removed — always single-task
	with open_dict(cfg):
		cfg.tasks = [cfg.task]

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

	# Validate planner value std coefficients are present
	if not hasattr(cfg, 'planner_value_std_coef_train'):
		raise AttributeError("Missing cfg.planner_value_std_coef_train; expected float (-1.0 for pessimistic, +1.0 for optimistic).")
	if not hasattr(cfg, 'planner_value_std_coef_eval'):
		raise AttributeError("Missing cfg.planner_value_std_coef_eval; expected float (-1.0 for pessimistic, +1.0 for optimistic).")

	# Validate policy value std coefficients are present
	if not hasattr(cfg, 'policy_value_std_coef'):
		raise AttributeError("Missing cfg.policy_value_std_coef; expected float (-1.0 for pessimistic, +1.0 for optimistic).")

	# Number of reward heads: default to 1 for backward compatibility
	if not hasattr(cfg, 'num_reward_heads'):
		cfg.num_reward_heads = 1

	# Note: std estimation is now adaptive - uses unbiased=True when n>=2, biased otherwise

	# ----------------------------------------------------------------------
	# Imagination horizon constraint (only horizon=1 supported)
	# ----------------------------------------------------------------------
	# Rationale: The value loss optimization picks head 0 for V predictions
	# (z_seq[:-1, 0]) because all dynamics heads are identical before the
	# dynamics rollout. This assumption only holds when imagination_horizon=1.
	# With longer horizons, dynamics heads diverge after the first step,
	# so heads would no longer be identical at t > 0.
	if cfg.value_per_dynamics:
		assert cfg.num_q % cfg.planner_num_dynamics_heads == 0, (
			f"value_per_dynamics requires num_q ({cfg.num_q}) divisible by "
			f"planner_num_dynamics_heads ({cfg.planner_num_dynamics_heads})."
		)

	if cfg.imagination_horizon != 1:
		raise ValueError(
			f"imagination_horizon={cfg.imagination_horizon} is not supported. Only imagination_horizon=1 is supported. "
			"Multi-step imagination would cause dynamics heads to diverge, breaking the assumption that "
			"all heads are identical at t=0 for value predictions."
		)

	if cfg.final_rho != -1:
		import math as _math
		cfg.rho = _math.pow(cfg.final_rho, 1 / cfg.horizon)
		print(f"Overriding rho schedule to end at final_rho = {cfg.final_rho} at horizon = {cfg.horizon}, setting rho = {cfg.rho:.6f}")

	# ----------------------------------------------------------------------
	# value_std_coef resolution: convert "opt"/"pess" strings to numeric values
	# ----------------------------------------------------------------------
	# All std_coef params can be:
	#   - numeric: use that value directly
	#   - "opt": use +value_std_coef_default (optimistic)
	#   - "pess": use -value_std_coef_default (pessimistic)
	# The sign determines dynamics head reduction: >0 → max, <0 → min, =0 → mean
	def resolve_std_coef(value, default_magnitude):
		"""Convert std_coef value to float, handling 'opt'/'pess' strings."""
		if isinstance(value, str):
			if value.lower() == 'opt':
				return float(default_magnitude)
			elif value.lower() == 'pess':
				return -float(default_magnitude)
			else:
				raise ValueError(f"Invalid std_coef string '{value}'. Must be 'opt', 'pess', or numeric.")
		return float(value)
	
	# Get default magnitude (must exist)
	if not hasattr(cfg, 'value_std_coef_default'):
		cfg.value_std_coef_default = 1.0  # fallback for backward compat
	default_mag = float(cfg.value_std_coef_default)
	
	# Resolve all std_coef parameters
	std_coef_params = [
		'policy_value_std_coef',
		'optimistic_policy_value_std_coef',
		'planner_value_std_coef_train',
		'planner_value_std_coef_eval',
		'td_target_std_coef',
	]
	for param in std_coef_params:
		if hasattr(cfg, param):
			old_val = cfg[param]
			cfg[param] = resolve_std_coef(old_val, default_mag)

	# ----------------------------------------------------------------------
	# policy_ema_tau inheritance: if null, inherit from tau
	# ----------------------------------------------------------------------
	policy_ema_tau_val = cfg.get('policy_ema_tau')
	if policy_ema_tau_val is None or policy_ema_tau_val == 'None':
		cfg.policy_ema_tau = cfg.tau

	return cfg_to_dataclass(cfg)
