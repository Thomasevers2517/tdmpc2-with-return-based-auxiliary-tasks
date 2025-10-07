from typing import Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from common import init, layers
from common.logging_utils import get_logger


log = get_logger(__name__)

_LEARNABLE_TYPES: Tuple[type, ...] = (
	nn.Linear,
	nn.Conv1d,
	nn.Conv2d,
	nn.Conv3d,
	nn.ConvTranspose1d,
	nn.ConvTranspose2d,
	nn.ConvTranspose3d,
)


def _is_learnable(submodule: nn.Module) -> bool:
	return isinstance(submodule, _LEARNABLE_TYPES)


def _named_learnable_modules(module: nn.Module, prefix: str = "") -> List[Tuple[str, nn.Module]]:
	collector: List[Tuple[str, nn.Module]] = []
	if isinstance(module, layers.Ensemble):
		iterator = module.params.named_modules()
	else:
		iterator = module.named_modules()
	for name, sub in iterator:
		if name == "":
			continue
		if not _is_learnable(sub):
			continue
		full_name = name if not prefix else f"{prefix}.{name}"
		collector.append((full_name, sub))
	return collector


def _select_last(modules: Sequence[Tuple[str, nn.Module]], last_k: int) -> List[Tuple[str, nn.Module]]:
	if not modules:
		return []
	if last_k == -1 or last_k >= len(modules):
		return list(modules)
	return list(modules[-last_k:])


def _tensor_snapshot(t: torch.Tensor) -> Tuple[List[float], float, float, float, float]:
	flat = t.detach().reshape(-1)
	if flat.numel() == 0:
		return [], 0.0, 0.0, 0.0, 0.0
	sample = flat[: min(3, flat.numel())].to("cpu").tolist()
	mean = flat.mean().item()
	std = flat.std(unbiased=False).item() if flat.numel() > 1 else 0.0
	min_val = flat.min().item()
	max_val = flat.max().item()
	return sample, mean, std, min_val, max_val


def _format_module_name(base_name: str, local_name: str, fallback: str) -> str:
	if local_name:
		return f"{base_name}.{local_name}" if base_name else local_name
	return base_name or fallback


def _ensemble_layer_specs(ensemble: layers.Ensemble, module_name: str):
	base = ensemble.module
	layers_info = []
	for name, sub in base.named_modules():
		if not name and not _is_learnable(sub):
			# skip non-learnable root modules to avoid duplicates
			continue
		if name == "" and _is_learnable(sub):
			layers_info.append((name, sub))
		elif name:
			if _is_learnable(sub):
				layers_info.append((name, sub))
	return layers_info


def _select_layers(layers_info: Sequence[Tuple[str, nn.Module]], last_k: int):
	if not layers_info:
		return []
	if last_k == -1 or last_k >= len(layers_info):
		return list(layers_info)
	return list(layers_info[-last_k:])


def _hard_reset_ensemble(ensemble: layers.Ensemble, last_k: int, *, module_name: str = "", logger=None) -> List[nn.Parameter]:
	logger = logger or log
	layers_info = _ensemble_layer_specs(ensemble, module_name)
	if not layers_info:
		logger.warning("Hard reset skipped for %s: no learnable submodules found", module_name or ensemble.__class__.__name__)
		return []
	selected = _select_layers(layers_info, last_k)
	logger.info(
		"Hard reset %s: operating on %d/%d learnable layers (last_k=%s)",
		module_name or ensemble.__class__.__name__,
		len(selected),
		len(layers_info),
		last_k,
	)
	params_map = dict(ensemble.params.named_parameters())
	touched: List[nn.Parameter] = []
	for local_name, sub in selected:
		display_name = _format_module_name(module_name, local_name, sub.__class__.__name__)
		logger.info("  Resetting layer %s (%s)", display_name, sub.__class__.__name__)
		prefix = f"{local_name}." if local_name else ""
		with torch.no_grad():
			for pname, _ in sub.named_parameters(recurse=False):
				param_key = f"{prefix}{pname}"
				param = params_map.get(param_key)
				if param is None:
					logger.debug("    Parameter %s missing on ensemble %s", param_key, module_name or ensemble.__class__.__name__)
					continue
				before = _tensor_snapshot(param)
				if param.dim() > 1:
					torch.nn.init.trunc_normal_(param, std=0.02)
				elif "weight" in pname:
					param.fill_(1.0)
				else:
					param.zero_()
				after = _tensor_snapshot(param)
				logger.info(
					"    %s.%s shape=%s | mean %.4f->%.4f | std %.4f->%.4f | sample %s -> %s",
					display_name,
					pname,
					tuple(param.shape),
					before[1],
					after[1],
					before[2],
					after[2],
					before[0],
					after[0],
				)
				touched.append(param)
	return touched


def _shrink_perturb_ensemble(
	ensemble: layers.Ensemble,
	last_k: int,
	alpha: float,
	noise_std: float,
	*,
	module_name: str = "",
	logger=None,
) -> List[nn.Parameter]:
	logger = logger or log
	layers_info = _ensemble_layer_specs(ensemble, module_name)
	if not layers_info:
		logger.warning("Shrink-perturb skipped for %s: no learnable submodules found", module_name or ensemble.__class__.__name__)
		return []
	selected = _select_layers(layers_info, last_k)
	logger.info(
		"Shrink-perturb %s: operating on %d/%d learnable layers (last_k=%s, alpha=%.4f, noise_std=%.4f)",
		module_name or ensemble.__class__.__name__,
		len(selected),
		len(layers_info),
		last_k,
		alpha,
		noise_std,
	)
	params_map = dict(ensemble.params.named_parameters())
	touched: List[nn.Parameter] = []
	for local_name, sub in selected:
		display_name = _format_module_name(module_name, local_name, sub.__class__.__name__)
		logger.info("  Perturbing layer %s (%s)", display_name, sub.__class__.__name__)
		prefix = f"{local_name}." if local_name else ""
		with torch.no_grad():
			for pname, _ in sub.named_parameters(recurse=False):
				param_key = f"{prefix}{pname}"
				param = params_map.get(param_key)
				if param is None:
					logger.debug("    Parameter %s missing on ensemble %s", param_key, module_name or ensemble.__class__.__name__)
					continue
				before = _tensor_snapshot(param)
				noise = torch.randn_like(param) * noise_std
				updated = alpha * param + (1 - alpha) * noise
				param.copy_(updated)
				after = _tensor_snapshot(param)
				logger.info(
					"    %s.%s shape=%s | mean %.4f->%.4f | std %.4f->%.4f | sample %s -> %s",
					display_name,
					pname,
					tuple(param.shape),
					before[1],
					after[1],
					before[2],
					after[2],
					before[0],
					after[0],
				)
				logger.debug(
					"    %s.%s noise mean/std %.4f/%.4f",
					display_name,
					pname,
					noise.mean().item(),
					noise.std(unbiased=False).item() if noise.numel() > 1 else 0.0,
				)
				touched.append(param)
	return touched


def hard_reset_module(
	module: nn.Module,
	last_k: int,
	*,
	module_name: str = "",
	logger=None,
) -> List[nn.Parameter]:
	logger = logger or log
	if isinstance(module, layers.Ensemble):
		return _hard_reset_ensemble(module, last_k, module_name=module_name, logger=logger)
	learnable = _named_learnable_modules(module, prefix=module_name)
	selected = _select_last(learnable, last_k)
	if not selected:
		logger.warning("Hard reset skipped for %s: no learnable submodules found", module_name or module.__class__.__name__)
		return []
	logger.info(
		"Hard reset %s: operating on %d/%d learnable layers (last_k=%s)",
		module_name or module.__class__.__name__,
		len(selected),
		len(learnable),
		last_k,
	)
	touched: List[nn.Parameter] = []
	for full_name, sub in selected:
		logger.info("  Resetting layer %s (%s)", full_name, sub.__class__.__name__)
		with torch.no_grad():
			before_stats = {
				pname: _tensor_snapshot(param)
				for pname, param in sub.named_parameters(recurse=False)
			}
			if hasattr(sub, "reset_parameters"):
				sub.reset_parameters()
			else:
				init.weight_init(sub)
			after_stats = {
				pname: _tensor_snapshot(param)
				for pname, param in sub.named_parameters(recurse=False)
			}
		for pname, param in sub.named_parameters(recurse=False):
			before = before_stats.get(pname)
			after = after_stats.get(pname)
			if before is None or after is None:
				continue
			logger.info(
				"    %s.%s shape=%s | mean %.4f->%.4f | std %.4f->%.4f | sample %s -> %s",
				full_name,
				pname,
				tuple(param.shape),
				before[1],
				after[1],
				before[2],
				after[2],
				before[0],
				after[0],
			)
			if before[3] != before[4] or after[3] != after[4]:
				logger.debug(
					"    %s.%s min/max %.4f/%.4f -> %.4f/%.4f",
					full_name,
					pname,
					before[3],
					before[4],
					after[3],
					after[4],
				)
			with torch.no_grad():
				touched.append(param)
	return touched


def shrink_perturb_module(
	module: nn.Module,
	last_k: int,
	alpha: float,
	noise_std: float,
	*,
	module_name: str = "",
	logger=None,
) -> List[nn.Parameter]:
	logger = logger or log
	if isinstance(module, layers.Ensemble):
		return _shrink_perturb_ensemble(module, last_k, alpha, noise_std, module_name=module_name, logger=logger)
	learnable = _named_learnable_modules(module, prefix=module_name)
	selected = _select_last(learnable, last_k)
	if not selected:
		logger.warning("Shrink-perturb skipped for %s: no learnable submodules found", module_name or module.__class__.__name__)
		return []
	logger.info(
		"Shrink-perturb %s: operating on %d/%d learnable layers (last_k=%s, alpha=%.4f, noise_std=%.4f)",
		module_name or module.__class__.__name__,
		len(selected),
		len(learnable),
		last_k,
		alpha,
		noise_std,
	)
	touched: List[nn.Parameter] = []
	for full_name, sub in selected:
		logger.info("  Perturbing layer %s (%s)", full_name, sub.__class__.__name__)
		with torch.no_grad():
			for pname, param in sub.named_parameters(recurse=False):
				before = _tensor_snapshot(param)
				noise = torch.randn_like(param) * noise_std
				updated = alpha * param + (1 - alpha) * noise
				param.copy_(updated)
				after = _tensor_snapshot(param)
				logger.info(
					"    %s.%s shape=%s | mean %.4f->%.4f | std %.4f->%.4f | sample %s -> %s",
					full_name,
					pname,
					tuple(param.shape),
					before[1],
					after[1],
					before[2],
					after[2],
					before[0],
					after[0],
				)
				logger.debug(
					"    %s.%s noise mean/std %.4f/%.4f",
					full_name,
					pname,
					noise.mean().item(),
					noise.std(unbiased=False).item() if noise.numel() > 1 else 0.0,
				)
				touched.append(param)
	return touched


def clear_optimizer_state(optim: torch.optim.Optimizer, params: Iterable[nn.Parameter], label: str, logger=None) -> None:
	logger = logger or log
	param_set = {p for p in params}
	if not param_set:
		return
	cleared = 0
	for param in list(optim.state.keys()):
		if param in param_set:
			optim.state.pop(param, None)
			cleared += 1
	logger.info("Cleared %d optimizer-state entries for %s", cleared, label)


def sync_auxiliary_detach(model, logger=None) -> None:
	logger = logger or log
	if model._aux_joint_Qs is not None:
		with torch.no_grad():
			vector = parameters_to_vector(model._aux_joint_Qs.parameters()).detach()
			model.aux_joint_detach_vec.copy_(vector)
			vector_to_parameters(model.aux_joint_detach_vec, model._detach_aux_joint_Qs.parameters())
		logger.info("Synchronized auxiliary joint detach head (%d parameters)", vector.numel())
	elif model._aux_separate_Qs is not None:
		with torch.no_grad():
			vectors = [parameters_to_vector(head.parameters()).detach() for head in model._aux_separate_Qs]
			full_vec = torch.cat(vectors, dim=0)
			model.aux_separate_detach_vec.copy_(full_vec)
			offset = 0
			for head, size in zip(model._detach_aux_separate_Qs, model.aux_separate_sizes):
				vector_to_parameters(model.aux_separate_detach_vec[offset:offset + size], head.parameters())
				offset += size
		logger.info("Synchronized auxiliary separate detach heads (%d parameters)", full_vec.numel())
	else:
		logger.info('No auxiliary detach heads to synchronize')