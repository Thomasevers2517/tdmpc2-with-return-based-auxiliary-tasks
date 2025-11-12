import dataclasses
import os
import datetime
import re

import numpy as np
import pandas as pd
from termcolor import colored
import logging
from typing import Optional
import torch

# Minimal shim: unified logger accessor for modules outside of Logger class
def get_logger(name: Optional[str] = None, cfg: Optional[object] = None) -> logging.Logger:
	"""Return a standard Python logger, optionally setting level from cfg.

	Args:
		name: Logger name.
		cfg: Optional config object; if provided and has attribute `debug`, sets level DEBUG when True else INFO.
	"""
	logger = logging.getLogger(name if name else __name__)
	if cfg is not None:
		level = logging.DEBUG if getattr(cfg, 'debug', False) else logging.INFO
		# Only raise level (avoid lowering an already more verbose logger unexpectedly)
		if logger.level != level:
			logger.setLevel(level)
	return logger

_log = get_logger(__name__)

from common import TASK_SET


CONSOLE_FORMAT = [
	("iteration", "I", "int"),
	("episode", "E", "int"),
	("step", "I", "int"),
	("episode_reward", "R", "float"),
	("episode_success", "S", "float"),
	("elapsed_time", "T", "time"),
]

CAT_TO_COLOR = {
	"pretrain": "yellow",
	"train": "blue",
	"eval": "green",
	"validation_all": "magenta",
	"validation_recent": "cyan",
	"policy_eval": "red",
}


def make_dir(dir_path):
	"""Create directory if it does not already exist."""
	try:
		os.makedirs(dir_path)
	except OSError:
		pass
	return dir_path


def print_run(cfg):
	"""
	Pretty-printing of current run information.
	Logger calls this method at initialization.
	"""
	prefix, color, attrs = "  ", "green", ["bold"]

	def _limstr(s, maxlen=36):
		s_str = str(s)
		return (s_str[:maxlen] + "...") if len(s_str) > maxlen else s_str

	def _pprint(k, v):
		label = f"{(k.capitalize()+':'):<15}"
		line = prefix + colored(label, color, attrs=attrs) + " " + _limstr(str(v))
		_log.info(line)

	observations  = ", ".join([str(v) for v in cfg.obs_shape.values()])
	kvs = [
		("task", cfg.task_title),
		("steps", f"{int(cfg.steps):,}"),
		("observations", observations),
		("actions", cfg.action_dim),
		("experiment", cfg.exp_name),
	]
	w = np.max([len(_limstr(str(kv[1]))) for kv in kvs]) + 25
	div = "-" * w
	_log.info(div)
	for k, v in kvs:
		_pprint(k, v)
	_log.info(div)


def cfg_to_group(cfg, return_list=False):
	"""
	Return a wandb-safe group name for logging.
	Optionally returns group name as list.
	"""
	lst = [cfg.task, re.sub("[^0-9a-zA-Z]+", "-", cfg.exp_name)]
	return lst if return_list else "-".join(lst)


class VideoRecorder:
	"""Utility class for logging evaluation videos."""

	def __init__(self, cfg, wandb, fps=15):
		self.cfg = cfg
		self._save_dir = make_dir(cfg.work_dir / 'eval_video')
		self._wandb = wandb
		self.fps = fps
		self.frames = []
		self.enabled = False

	def init(self, env, enabled=True):
		self.frames = []
		self.enabled = self._save_dir and self._wandb and enabled
		self.record(env)

	def record(self, env):
		if self.enabled:
			self.frames.append(env.render())

	def save(self, step, key='videos/eval_video'):
		if self.enabled and len(self.frames) > 0:
			frames = np.stack(self.frames)
			return self._wandb.log(
				{key: self._wandb.Video(frames.transpose(0, 3, 1, 2), fps=self.fps, format='gif')}, step=step
			)


class Logger:
	"""Primary logging object. Logs either locally or using wandb."""

	def __init__(self, cfg):
		self.stdlog = get_logger(__name__)
		self._log_dir = make_dir(cfg.work_dir)
		self._model_dir = make_dir(self._log_dir / "models")
		self._save_csv = cfg.save_csv
		self._save_agent = cfg.save_agent
		self._group = cfg_to_group(cfg)
		self._seed = cfg.seed
		self._eval = []
		print_run(cfg)
		self.project = cfg.get("wandb_project", "none")
		self.entity = cfg.get("wandb_entity", "none")
		if not cfg.enable_wandb or self.project == "none" or self.entity == "none":
			_log.info("%s", colored("Wandb disabled.", "blue", attrs=["bold"]))
			cfg.save_agent = False
			cfg.save_video = False
			self._wandb = None
			self._video = None
			return
		os.environ["WANDB_SILENT"] = "true" if cfg.wandb_silent else "false"
		import wandb

		wandb.init(
			project=self.project,
			entity=self.entity,
			group=self._group,
			tags=cfg_to_group(cfg, return_list=True) + [f"seed:{cfg.seed}"],
			dir=self._log_dir,
			config=dataclasses.asdict(cfg),
		)
		_log.info("%s", colored("Logs will be synced with wandb.", "blue", attrs=["bold"]))
		self._wandb = wandb
		self._video = (
			VideoRecorder(cfg, self._wandb)
			if self._wandb and cfg.save_video
			else None
		)

	@property
	def video(self):
		return self._video

	@property
	def model_dir(self):
		return self._model_dir

	def save_agent(self, agent=None, identifier='final'):
		if self._save_agent and agent:
			fp = self._model_dir / f'{str(identifier)}.pt'
			agent.save(fp)
			if self._wandb:
				artifact = self._wandb.Artifact(
					self._group + '-' + str(self._seed) + '-' + str(identifier),
					type='model',
				)
				artifact.add_file(fp)
				self._wandb.log_artifact(artifact)

	def finish(self, agent=None):
		try:
			self.save_agent(agent)
		except Exception as e:
			_log.error("%s", colored(f"Failed to save model: {e}", "red"))
		if self._wandb:
			self._wandb.finish()

	def _format(self, key, value, ty):
		if ty == "int":
			return f'{colored(key+":", "blue")} {int(value):,}'
		elif ty == "float":
			return f'{colored(key+":", "blue")} {value:.01f}'
		elif ty == "time":
			value = str(datetime.timedelta(seconds=int(value)))
			return f'{colored(key+":", "blue")} {value}'
		else:
			raise f"invalid log format type: {ty}"

	def _print(self, d, category):
		category = colored(category, CAT_TO_COLOR[category])
		pieces = [f" {category:<14}"]
		for k, disp_k, ty in CONSOLE_FORMAT:
			if k in d:
				pieces.append(f"{self._format(disp_k, d[k], ty):<22}")
		_log.info("%s", "   ".join(pieces))

	# ----------------------- Modular sink helpers -----------------------
	def _wandb_log(self, payload: dict, step: int, category: str):
		"""Log a flat dict to WandB with category-prefixed keys.

		Assumes payload already contains scalar/serializable values and that
		'step' is provided separately for x-axis alignment.
		"""
		if not self._wandb:
			return
		_d = {}
		for k, v in payload.items():
			_d[f"{category}/{k}"] = v
		self._wandb.log(_d, step=step)

	def _csv_eval_append(self, payload: dict):
		"""Append eval metrics to CSV when present in payload."""
		keys = ["step", "episode_reward"]
		self._eval.append(np.array([payload[keys[0]], payload[keys[1]]]))
		pd.DataFrame(np.array(self._eval)).to_csv(
			self._log_dir / "eval.csv", header=keys, index=None
		)

	def pprint_multitask(self, d, cfg):
		"""Pretty-print evaluation metrics for multi-task training."""
		_log.info("%s", colored(f'Evaluated agent on {len(cfg.tasks)} tasks:', 'yellow', attrs=['bold']))
		dmcontrol_reward = []
		metaworld_reward = []
		metaworld_success = []
		for k, v in d.items():
			if '+' not in k:
				continue
			task = k.split('+')[1]
			if task in TASK_SET['mt30'] and k.startswith('episode_reward'): # DMControl
				dmcontrol_reward.append(v)
				_log.info("%s", colored(f'  {task:<22}\tR: {v:.01f}', 'yellow'))
			elif task in TASK_SET['mt80'] and task not in TASK_SET['mt30']: # Meta-World
				if k.startswith('episode_reward'):
					metaworld_reward.append(v)
				elif k.startswith('episode_success'):
					metaworld_success.append(v)
					_log.info("%s", colored(f'  {task:<22}\tS: {v:.02f}', 'yellow'))
		dmcontrol_reward = np.nanmean(dmcontrol_reward)
		d['episode_reward+avg_dmcontrol'] = dmcontrol_reward
		_log.info("%s", colored(f'  {"dmcontrol":<22}\tR: {dmcontrol_reward:.01f}', 'yellow', attrs=['bold']))
		if cfg.task == 'mt80':
			metaworld_reward = np.nanmean(metaworld_reward)
			metaworld_success = np.nanmean(metaworld_success)
			d['episode_reward+avg_metaworld'] = metaworld_reward
			d['episode_success+avg_metaworld'] = metaworld_success
			_log.info("%s", colored(f'  {"metaworld":<22}\tR: {metaworld_reward:.01f}', 'yellow', attrs=['bold']))
			_log.info("%s", colored(f'  {"metaworld":<22}\tS: {metaworld_success:.02f}', 'yellow', attrs=['bold']))

	def log(self, d, category="train"):
		# assert category in CAT_TO_COLOR.keys(), f"invalid category: {category}"
		# Determine x-axis key
		if category in {"train", "eval"}:
			xkey = "step"
		elif category == "pretrain":
			xkey = "iteration"
		else:
			xkey = "step"
		# WandB sink
		if self._wandb:
			self._wandb_log(d, step=d[xkey], category=category)
		# Eval CSV sink
		if category == "eval" and self._save_csv and ("episode_reward" in d):
			self._csv_eval_append(d)
		# Console print sink (legacy behavior retained)
		self._print(d, category)

	def log_planner_info(self, info, step: int, category: str = "train"):
		"""Log planner information dataclass (basic or advanced).

		Args:
			info: PlannerBasicInfo or PlannerAdvancedInfo instance.
			step: Global step for x-axis alignment.
			category: Log category (train/eval/etc.).
		"""
		try:
			from common.planner.info_types import PlannerBasicInfo, PlannerAdvancedInfo
		except Exception:
			# If planner not yet available, skip
			return
		payload = {
			"step": step,
			"planner/value_chosen": getattr(info, "value_chosen", None),
			"planner/value_elite_mean": getattr(info, "value_elite_mean", None),
			"planner/value_elite_std": getattr(info, "value_elite_std", None),
			"planner/value_elite_max": getattr(info, "value_elite_max", None),
			"planner/num_elites": getattr(info, "num_elites", None),
		}
		d_chosen = getattr(info, "disagreement_chosen", None)
		if d_chosen is not None:
			payload["planner/disagreement_chosen"] = d_chosen
		d_mean = getattr(info, "disagreement_elite_mean", None)
		d_std = getattr(info, "disagreement_elite_std", None)
		d_max = getattr(info, "disagreement_elite_max", None)
		if d_mean is not None:
			payload["planner/disagreement_elite_mean"] = d_mean
		if d_std is not None:
			payload["planner/disagreement_elite_std"] = d_std
		if d_max is not None:
			payload["planner/disagreement_elite_max"] = d_max
		# Planner: WandB-only by default. No console print, no eval CSV.
		# If WandB disabled and debug_console is False: no-op.
		# Normalize 0-dim tensors to Python scalars (detach -> cpu -> item)
		for k, v in list(payload.items()):
			if k == 'step':
				continue
			if isinstance(v, torch.Tensor):
				if v.numel() == 1:
					payload[k] = v.detach().cpu().item()
				else:
					# Skip large tensors (histograms could be added later)
					payload.pop(k)
		if self._wandb:
			self._wandb_log({k: v for k, v in payload.items() if k != 'step'}, step=payload["step"], category=category)
			# Optional console mirroring in debug mode

		return
