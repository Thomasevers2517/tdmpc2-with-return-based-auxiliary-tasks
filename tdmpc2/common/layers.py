import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import from_modules
from copy import deepcopy


class Ensemble(nn.Module):
	"""
	Vectorized ensemble of modules.
	"""

	def __init__(self, modules, **kwargs):
		super().__init__()
		# combine_state_for_ensemble causes graph breaks
		self.params = from_modules(*modules, as_module=True)
		with self.params[0].data.to("meta").to_module(modules[0]):
			self.module = deepcopy(modules[0])
		self._repr = str(modules[0])
		self._n = len(modules)

	def __len__(self):
		return self._n

	def _call(self, params, *args, **kwargs):
		with params.to_module(self.module):
			return self.module(*args, **kwargs)

	def forward(self, *args, **kwargs):
		return torch.vmap(self._call, (0, None), randomness="different")(self.params, *args, **kwargs)

	def __repr__(self):
		return f'Vectorized {len(self)}x ' + self._repr


class ShiftAug(nn.Module):
	"""
	Random shift image augmentation.
	Adapted from https://github.com/facebookresearch/drqv2
	"""
	def __init__(self, pad=3):
		super().__init__()
		self.pad = pad
		self.padding = tuple([self.pad] * 4)

	def forward(self, x):
		x = x.float()
		n, _, h, w = x.size()
		assert h == w
		x = F.pad(x, self.padding, 'replicate')
		eps = 1.0 / (h + 2 * self.pad)
		arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
		arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
		base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
		base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
		shift = torch.randint(0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)
		shift *= 2.0 / (h + 2 * self.pad)
		grid = base_grid + shift
		return F.grid_sample(x, grid, padding_mode='zeros', align_corners=False)


class PixelPreprocess(nn.Module):
	"""
	Normalizes pixel observations to [-0.5, 0.5].
	"""

	def __init__(self):
		super().__init__()

	def forward(self, x):
		return x.div(255.).sub(0.5)


class SimNorm(nn.Module):
	"""
	Simplicial normalization.
	Adapted from https://arxiv.org/abs/2204.00616.
	"""

	def __init__(self, cfg):
		super().__init__()
		self.dim = cfg.simnorm_dim

	def forward(self, x):
		shp = x.shape
		x = x.view(*shp[:-1], -1, self.dim)
		x = F.softmax(x, dim=-1)
		return x.view(*shp)

	def __repr__(self):
		return f"SimNorm(dim={self.dim})"


class NormedLinear(nn.Linear):
	"""
	Linear layer with LayerNorm, activation, and optionally dropout.
	"""

	def __init__(self, *args, dropout=0., act=None, **kwargs):
		super().__init__(*args, **kwargs)
		self.ln = nn.LayerNorm(self.out_features)
		if act is None:
			act = nn.Mish(inplace=False)
		self.act = act
		self.dropout = nn.Dropout(dropout, inplace=False) if dropout else None

	def forward(self, x):
		x = super().forward(x)
		if self.dropout:
			x = self.dropout(x)
		return self.act(self.ln(x))

	def __repr__(self):
		repr_dropout = f", dropout={self.dropout.p}" if self.dropout else ""
		return f"NormedLinear(in_features={self.in_features}, "\
			f"out_features={self.out_features}, "\
			f"bias={self.bias is not None}{repr_dropout}, "\
			f"act={self.act.__class__.__name__})"


class ResidualBlock(nn.Module):
	"""Residual block with 3x3 convs, BN, and ReLU."""

	def __init__(self, planes: int, in_planes: int = None):
		super().__init__()
		if in_planes is None:
			in_planes = planes
		self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(planes)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)
		self.relu = nn.ReLU(inplace=False)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		identity = x
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)
		out = self.conv2(out)
		out = self.bn2(out)
		out = out + identity
		out = self.relu(out)
		return out


class CnnDynamicsBlock(nn.Module):
	"""Single 3x3 conv dynamics block with optional projection skip.

	Used for CNN dynamics: maps [B,in_planes,H,W] -> [B,out_planes,H,W].
	"""

	def __init__(self, in_planes: int, out_planes: int):
		super().__init__()
		self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn = nn.BatchNorm2d(out_planes)
		self.relu = nn.ReLU(inplace=False)
		self.proj = None
		if in_planes != out_planes:
			self.proj = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		identity = x if self.proj is None else self.proj(x)
		out = self.conv(x)
		out = self.bn(out)
		out = self.relu(out)
		out = out + identity
		out = self.relu(out)
		return out


class ResidualDownsampleBlock(nn.Module):
	"""Residual block that downsamples spatially and changes channels."""

	def __init__(self, in_planes: int, out_planes: int, stride: int = 2):
		super().__init__()
		self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(out_planes)
		self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(out_planes)
		self.downsample = nn.Sequential(
			nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
			nn.BatchNorm2d(out_planes),
		)
		self.relu = nn.ReLU(inplace=False)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		identity = self.downsample(x)
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)
		out = self.conv2(out)
		out = self.bn2(out)
		out = out + identity
		out = self.relu(out)
		return out


def mlp(in_dim, mlp_dims, out_dim, act=None, dropout=0.):
	"""
	Basic building block of TD-MPC2.
	MLP with LayerNorm, Mish activations, and optionally dropout.
	"""
	if isinstance(mlp_dims, int):
		mlp_dims = [mlp_dims]
	dims = [in_dim] + mlp_dims + [out_dim]
	mlp = nn.ModuleList()
	for i in range(len(dims) - 2):
		mlp.append(NormedLinear(dims[i], dims[i+1], dropout=dropout*(i==0)))
	mlp.append(NormedLinear(dims[-2], dims[-1], act=act) if act else nn.Linear(dims[-2], dims[-1]))
	return nn.Sequential(*mlp)


def conv(in_shape, num_channels, out_dim, act=None):
	"""
	Basic convolutional encoder for TD-MPC2 with raw image observations.
	4 layers of convolution with ReLU activations, followed by a linear layer.
	"""
	assert in_shape[-1] == 64 # assumes rgb observations to be 64x64
	feature_layers = [
		ShiftAug(), PixelPreprocess(),
		nn.Conv2d(in_shape[0], num_channels, 7, stride=2), nn.ReLU(inplace=False),
		nn.Conv2d(num_channels, num_channels, 5, stride=2), nn.ReLU(inplace=False),
		nn.Conv2d(num_channels, num_channels, 3, stride=2), nn.ReLU(inplace=False),
		nn.Conv2d(num_channels, num_channels, 3, stride=1), nn.Flatten()]
	body = nn.Sequential(*feature_layers)
	with torch.no_grad():
		flat_dim = body(torch.zeros(1, *in_shape)).shape[-1]
	# projection = nn.Linear(flat_dim, out_dim)
	# modules = [body, projection]
	modules = [body]
	if act:
		modules.append(act)
	return nn.Sequential(*modules)


class EfficientZeroBackbone(nn.Module):
	"""EfficientZeroV2-style representation CNN adapted to 64x64 inputs.

	Input:
		x: float32[B, C_total, 64, 64]
	Output:
		Tensor[B, 64, 4, 4]
	"""

	def __init__(self, in_channels: int):
		super().__init__()
		self.in_channels = in_channels
		self.conv0 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
		self.bn0 = nn.BatchNorm2d(32)
		self.relu = nn.ReLU(inplace=False)
		self.block1 = ResidualBlock(32)
		self.down1 = ResidualDownsampleBlock(32, 64, stride=2)
		self.block2 = ResidualBlock(64)
		self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
		self.bn1 = nn.BatchNorm2d(64)
		self.block3 = ResidualBlock(64)
		self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
		self.bn2 = nn.BatchNorm2d(64)
		self.block4 = ResidualBlock(64)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# x: [B, C_in, 64, 64]
		B, C, H, W = x.shape
		assert C == self.in_channels, f"EfficientZeroBackbone expected {self.in_channels} channels, got {C}"
		assert H == 64 and W == 64, f"EfficientZeroBackbone expects 64x64 inputs, got {H}x{W}"
		out = self.conv0(x)
		out = self.bn0(out)
		out = self.relu(out)
		out = self.block1(out)
		out = self.down1(out)
		out = self.block2(out)
		out = self.pool1(out)
		out = self.bn1(out)
		out = self.relu(out)
		out = self.block3(out)
		out = self.pool2(out)
		out = self.bn2(out)
		out = self.relu(out)
		out = self.block4(out)
		# out: [B, 64, 4, 4]
		return out


class EfficientZeroEncoder(nn.Module):
	"""RGB encoder using EfficientZeroV2 backbone with optional projection.

	If cfg.project_latent is True, flattens backbone output and projects to
	cfg.latent_dim with an MLP ending in SimNorm. Otherwise, returns a
	flattened feature of size 64*4*4.
	"""

	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		in_shape = cfg.obs_shape['rgb']
		c_total, h, w = in_shape
		assert h == 64 and w == 64, f"EfficientZeroEncoder expects 64x64 rgb, got {h}x{w}"
		# Match default conv encoder preprocessing: random shift + pixel normalize.
		self.augment = ShiftAug()
		self.preprocess = PixelPreprocess()
		self.backbone = EfficientZeroBackbone(c_total)
		self.project_latent = cfg.project_latent
		self.backbone_channels = 64
		self.backbone_hw = 4
		self.backbone_dim = self.backbone_channels * self.backbone_hw * self.backbone_hw
		if self.project_latent:
			self.head = mlp(
				self.backbone_dim,
				max(cfg.num_enc_layers - 1, 1) * [cfg.enc_dim],
				cfg.latent_dim,
				act=SimNorm(cfg),
			)
		else:
			# When not projecting, still apply SimNorm on the 1024-dim latent.
			self.head = SimNorm(cfg)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# x: [B, C_total, 64, 64] (typically uint8 from env/buffer)
		# Apply same preprocessing and augmentation as default conv encoder.
		x = self.augment(x)
		x = self.preprocess(x)
		feat = self.backbone(x)  # [B,64,4,4]
		flat = feat.view(feat.shape[0], -1)  # [B,1024]
		return self.head(flat)


def enc(cfg, out={}):
	"""
	Returns a dictionary of encoders for each observation in the dict.
	"""
	for k in cfg.obs_shape.keys():
		if k == 'state':
			out[k] = mlp(cfg.obs_shape[k][0] + cfg.task_dim, max(cfg.num_enc_layers-1, 1)*[cfg.enc_dim], cfg.latent_dim, act=SimNorm(cfg))
		elif k == 'rgb':
			if cfg.rgb_encoder_type == 'default':
				out[k] = conv(cfg.obs_shape[k], cfg.num_channels, cfg.latent_dim, act=SimNorm(cfg))
			elif cfg.rgb_encoder_type == 'efficientzero':
				out[k] = EfficientZeroEncoder(cfg)
			else:
				raise NotImplementedError(f"Unknown rgb_encoder_type: {cfg.rgb_encoder_type}")
		else:
			raise NotImplementedError(f"Encoder for observation type {k} not implemented.")
	return nn.ModuleDict(out)


def api_model_conversion(target_state_dict, source_state_dict):
	"""
	Converts a checkpoint from our old API to the new torch.compile compatible API.
	"""
	# check whether checkpoint is already in the new format
	if "_detach_Qs_params.0.weight" in source_state_dict:
		return source_state_dict

	name_map = ['weight', 'bias', 'ln.weight', 'ln.bias']
	new_state_dict = dict()

	# rename keys
	for key, val in list(source_state_dict.items()):
		if key.startswith('_Qs.'):
			num = key[len('_Qs.params.'):]
			new_key = str(int(num) // 4) + "." + name_map[int(num) % 4]
			new_total_key = "_Qs.params." + new_key
			del source_state_dict[key]
			new_state_dict[new_total_key] = val
			new_total_key = "_detach_Qs_params." + new_key
			new_state_dict[new_total_key] = val
		elif key.startswith('_target_Qs.'):
			num = key[len('_target_Qs.params.'):]
			new_key = str(int(num) // 4) + "." + name_map[int(num) % 4]
			new_total_key = "_target_Qs_params." + new_key
			del source_state_dict[key]
			new_state_dict[new_total_key] = val

	# add batch_size and device from target_state_dict to new_state_dict
	for prefix in ('_Qs.', '_detach_Qs_', '_target_Qs_'):
		for key in ('__batch_size', '__device'):
			new_key = prefix + 'params.' + key
			new_state_dict[new_key] = target_state_dict[new_key]

	# check that every key in new_state_dict is in target_state_dict
	for key in new_state_dict.keys():
		assert key in target_state_dict, f"key {key} not in target_state_dict"
	# check that all Qs keys in target_state_dict are in new_state_dict
	for key in target_state_dict.keys():
		if 'Qs' in key:
			assert key in new_state_dict, f"key {key} not in new_state_dict"
	# check that source_state_dict contains no Qs keys
	for key in source_state_dict.keys():
		assert 'Qs' not in key, f"key {key} contains 'Qs'"

	# copy log_std_min and log_std_max from target_state_dict to new_state_dict
	new_state_dict['log_std_min'] = target_state_dict['log_std_min']
	new_state_dict['log_std_dif'] = target_state_dict['log_std_dif']
	if '_action_masks' in target_state_dict:
		new_state_dict['_action_masks'] = target_state_dict['_action_masks']

	# copy new_state_dict to source_state_dict
	source_state_dict.update(new_state_dict)

	return source_state_dict
