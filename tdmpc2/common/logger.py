import dataclasses
import os
import datetime
import re

import numpy as np
import pandas as pd
import torch
from termcolor import colored
import logging
from common.logging_utils import get_logger
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
		if self._wandb:
			if category in {"train", "eval"}:
				xkey = "step"
			elif category == "pretrain":
				xkey = "iteration"
			else:
				xkey = "step"
			_d = dict()
			for k, v in d.items():
				_d[category + "/" + k] = v
			self._wandb.log(_d, step=d[xkey])
		if category == "eval" and self._save_csv:
			keys = ["step", "episode_reward"]
			self._eval.append(np.array([d[keys[0]], d[keys[1]]]))
			pd.DataFrame(np.array(self._eval)).to_csv(
				self._log_dir / "eval.csv", header=keys, index=None
			)
		self._print(d, category)

	def log_planner(self, planning_info, step):
		"""Pretty-print and optionally upload planner diagnostics.

		Args:
			planning_info: Dictionary emitted by :class:`Planner` containing scalar
				metrics, tensors, and optional detailed payloads.
			step: Global environment step associated with these metrics.
		"""
		if not planning_info or not isinstance(planning_info, dict):
			return
		def _as_tensor(data):
			if data is None:
				return None
			if isinstance(data, torch.Tensor):
				return data.detach().cpu()
			try:
				return torch.as_tensor(data)
			except Exception:
				return None

		payload = {}
		# Scalars and small tensors that should become wandb scalars live under
		# ``base_keys``. We normalise them to cpu/python types for logging.
		base_keys = (
			'planner/type', 'ensemble/size', 'particle/parents', 'particle/children',
			'particle/iterations', 'particle/policy_children', 'particle/elite_k',
			'planner/lambda', 'planner/temperature',
			'parents/value_max_mean', 'parents/value_max_std',
			'parents/disagreement_mean', 'parents/disagreement_std',
			'chosen_parent_index', 'chosen_value_max_head',
			'chosen_value_max', 'chosen_disagreement', 'chosen_parent_weight'
		)
		for key in base_keys:
			if key not in planning_info:
				continue
			value = planning_info[key]
			if hasattr(value, 'item') and getattr(value, 'numel', lambda: 0)() == 1:
				value = value.item()
			elif isinstance(value, torch.Tensor):
				value = value.detach().cpu().tolist()
			elif hasattr(value, 'tolist') and not isinstance(value, (str, bytes)):
				value = value.tolist()
			prefix = key if key.startswith('planner/') else f'planner/{key}'
			payload[prefix] = value

		parent_raw = _as_tensor(planning_info.get('parents/raw_scores'))
		parent_value = _as_tensor(planning_info.get('parents/value_max'))
		parent_disagreement = _as_tensor(planning_info.get('parents/disagreement'))
		parent_weights = _as_tensor(planning_info.get('parents/softmax'))
		parent_heads = _as_tensor(planning_info.get('parents/value_max_head'))
		per_step_disagreement = _as_tensor(planning_info.get('parents/per_step_disagreement'))
		chosen_parent_index = planning_info.get('chosen_parent_index')
		if isinstance(chosen_parent_index, torch.Tensor):
			chosen_parent_index = int(chosen_parent_index.item())

		if parent_raw is not None and parent_raw.ndim == 1:
			# Highlight the top-K parent trajectories for quick inspection.
			top_k = min(3, parent_raw.shape[0])
			indices = torch.argsort(parent_raw, descending=True)
			for rank in range(top_k):
				idx = int(indices[rank].item())
				prefix = f'planner/top_{rank}'
				payload[f'{prefix}/index'] = idx
				payload[f'{prefix}/raw'] = float(parent_raw[idx].item())
				if parent_value is not None:
					payload[f'{prefix}/value'] = float(parent_value[idx].item())
				if parent_disagreement is not None:
					payload[f'{prefix}/disagreement'] = float(parent_disagreement[idx].item())
				if parent_weights is not None:
					payload[f'{prefix}/weight'] = float(parent_weights[idx].item())
				if parent_heads is not None:
					payload[f'{prefix}/value_head'] = int(parent_heads[idx].item())
			if parent_weights is not None:
				payload['planner/parents/weight_entropy'] = float((-parent_weights.clamp_min(1e-12) * parent_weights.log()).sum().item())

		if per_step_disagreement is not None and chosen_parent_index is not None:
			if per_step_disagreement.ndim == 2 and 0 <= chosen_parent_index < per_step_disagreement.shape[1]:
				# Summarise the disagreement curve for the chosen parent.
				chosen = per_step_disagreement[:, chosen_parent_index]
				payload['planner/chosen/per_step_disagreement_mean'] = float(chosen.mean().item())
				payload['planner/chosen/per_step_disagreement_max'] = float(chosen.max().item())
				payload['planner/chosen/per_step_disagreement_min'] = float(chosen.min().item())
				if self._wandb:
					payload['planner/chosen/per_step_disagreement'] = chosen.tolist()

		detailed_payload = planning_info.get('planning_info')
		if detailed_payload:
			payload['planner/detail_level'] = 'detailed'
			detail_rows = []
			detail_limit = min(len(detailed_payload), 4)
			for idx in range(detail_limit):
				parent = detailed_payload[idx]
				if not isinstance(parent, dict):
					continue
				best_raw = parent.get('best_child_raw_score')
				best_value = parent.get('best_child_value_max')
				best_disagreement = parent.get('best_child_disagreement')
				best_head = parent.get('best_child_value_max_head')
				row = {
					'parent_index': idx,
					'best_raw_score': float(best_raw) if best_raw is not None else None,
					'best_value': float(best_value) if best_value is not None else None,
					'best_disagreement': float(best_disagreement) if best_disagreement is not None else None,
					'best_head': int(best_head) if best_head is not None else None,
				}
				best_actions = parent.get('best_child_actions')
				if isinstance(best_actions, torch.Tensor) and best_actions.numel() > 0:
					action_preview = best_actions[0].detach().cpu().numpy()
					row['first_action'] = np.round(action_preview, 4).tolist()
				if isinstance(parent.get('children'), list):
					row['num_children'] = len(parent['children'])
				if parent.get('children'):
					child_scores = [float(child.get('raw_score')) for child in parent['children'] if isinstance(child, dict) and child.get('raw_score') is not None]
					if child_scores:
						row['children_raw_min'] = float(min(child_scores))
						row['children_raw_max'] = float(max(child_scores))
				detail_rows.append(row)
				if chosen_parent_index is not None and idx == chosen_parent_index:
					payload['planner/chosen/action_preview'] = row.get('first_action')
			if self._wandb and detail_rows:
				columns = sorted(detail_rows[0].keys())
				table = self._wandb.Table(columns=columns)
				for row in detail_rows:
					table.add_data(*[row[col] for col in columns])
				payload['planner/detail_table'] = table
		else:
			payload['planner/detail_level'] = 'basic'

		if not payload:
			return

		if self._wandb:
			# Upload structured data to wandb so dashboards mirror console logs.
			self._wandb.log(payload, step=step)
		if parent_raw is not None and parent_value is not None and parent_disagreement is not None:
			top_values = [(float(parent_raw[i].item()), float(parent_value[i].item()), float(parent_disagreement[i].item())) for i in torch.argsort(parent_raw, descending=True)[:min(3, parent_raw.shape[0])]]
			_log.info('Planner step=%s top sequences (raw, value, disagreement): %s', step, top_values)
		else:
			_log.info('Planner logging step=%s keys=%s', step, sorted(payload.keys()))
