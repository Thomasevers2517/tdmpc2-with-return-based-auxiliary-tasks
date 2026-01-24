import torch
from torch.nn import Buffer

from common.logger import get_logger


log = get_logger(__name__)


class RunningScale(torch.nn.Module):
	"""Running trimmed scale estimator."""

	def __init__(self, cfg, min_scale: float = 1.0):
		super().__init__()
		self.cfg = cfg
		self.min_scale = min_scale  # Minimum scale value (1.0 = no amplification)
		self.value = Buffer(torch.ones(1, dtype=torch.float32, device=torch.device('cuda:0')))
		self._percentiles = Buffer(torch.tensor([5, 95], dtype=torch.float32, device=torch.device('cuda:0')))

	def state_dict(self):
		return dict(value=self.value, percentiles=self._percentiles)

	def load_state_dict(self, state_dict):
		self.value.copy_(state_dict['value'])
		self._percentiles.copy_(state_dict['percentiles'])

	def _positions(self, x_shape):
		positions = self._percentiles * (x_shape-1) / 100
		floored = torch.floor(positions)
		ceiled = floored + 1
		ceiled = torch.where(ceiled > x_shape - 1, x_shape - 1, ceiled)
		weight_ceiled = positions-floored
		weight_floored = 1.0 - weight_ceiled
		return floored.long(), ceiled.long(), weight_floored.unsqueeze(1), weight_ceiled.unsqueeze(1)

	def _percentile(self, x):
		x_dtype, x_shape = x.dtype, x.shape
		x = x.flatten(1, x.ndim-1)
		in_sorted = torch.sort(x, dim=0).values
		floored, ceiled, weight_floored, weight_ceiled = self._positions(x.shape[0])
		d0 = in_sorted[floored] * weight_floored
		d1 = in_sorted[ceiled] * weight_ceiled
		return (d0+d1).reshape(-1, *x_shape[1:]).to(x_dtype)

	def update(self, x):
		percentiles = self._percentile(x.detach())
		value = torch.clamp(percentiles[1] - percentiles[0], min=self.min_scale)
		self.value.data.lerp_(value, self.cfg.tau)

	def forward(self, x, update=False):
		if update:
			self.update(x)
		return x / self.value

	def reset(self):
		prev = self.value.detach().clone()
		self.value.copy_(torch.ones_like(self.value))
		log.info('RunningScale reset: value %s -> %s', prev.flatten().cpu().tolist(), self.value.flatten().cpu().tolist())

	def __repr__(self):
		return f'RunningScale(S: {self.value})'
