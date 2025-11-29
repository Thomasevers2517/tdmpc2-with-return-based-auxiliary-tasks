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



class ConvEncoder(nn.Module):
	"""Basic convolutional encoder for TD-MPC2 with raw image observations.

	4 layers of convolution with ReLU activations, followed by an optional
	linear projection to ``out_dim``. The conv trunk is exposed as ``body``
	to allow accessing 4D conv features when needed (e.g. for CNN dynamics).
	"""

	def __init__(
		self,
		in_shape,
		num_channels: int,
		out_dim: int,
		act: nn.Module | None = None,
		use_projection: bool = True,
	):
		super().__init__()
		assert in_shape[-1] == 64  # assumes rgb observations to be 64x64
		feature_layers = [
			ShiftAug(),
			PixelPreprocess(),
			nn.Conv2d(in_shape[0], num_channels, 7, stride=2),
			nn.ReLU(inplace=False),
			nn.Conv2d(num_channels, num_channels, 5, stride=2),
			nn.ReLU(inplace=False),
			nn.Conv2d(num_channels, num_channels, 3, stride=2),
			nn.ReLU(inplace=False),
			nn.Conv2d(num_channels, num_channels, 3, stride=1),
		]
		self.body = nn.Sequential(*feature_layers)  # returns 4D features
		self.flatten = nn.Flatten()
		self.proj = None
		if use_projection:
			with torch.no_grad():
				flat_dim = self.flatten(self.body(torch.zeros(1, *in_shape))).shape[-1]
			self.proj = nn.Linear(flat_dim, out_dim)
		self.act = act

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		feat = self.body(x)
		flat = self.flatten(feat)
		if self.proj is not None:
			flat = self.proj(flat)
		if self.act is not None:
			flat = self.act(flat)
		return flat

class ResidualBlock(nn.Module):
	"""Basic ResNet-style residual block with BatchNorm and configurable channels."""

	def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
		super().__init__()
		self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(out_channels)
		self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(out_channels)
		if stride != 1 or in_channels != out_channels:
			self.downsample = nn.Sequential(
				nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(out_channels),
			)
		else:
			self.downsample = None
		self.act = nn.ReLU(inplace=False)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		identity = x
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.act(out)
		out = self.conv2(out)
		out = self.bn2(out)
		if self.downsample is not None:
			identity = self.downsample(x)
		out = out + identity
		out = self.act(out)
		return out


class CnnDynamicsHead(nn.Module):
	"""Spatial CNN dynamics head operating on a (C,H,W) latent.

	The action is broadcast across spatial positions and concatenated
	along the channel dimension, followed by a 3x3 stride-1 conv and a
	residual link back to the original latent. The output is flattened
	back to size C*H*W to match cfg.latent_dim.
	"""

	def __init__(self, latent_spatial_shape: tuple[int, int, int], action_dim: int):
		super().__init__()
		C, H, W = latent_spatial_shape
		self.C = C
		self.H = H
		self.W = W
		self.action_dim = action_dim
		self.conv = nn.Conv2d(C + action_dim, C, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn = nn.BatchNorm2d(C)
		self.act = nn.ReLU(inplace=False)
		self.res_block = ResidualBlock(C, C, stride=1)

	def forward(self, za: torch.Tensor) -> torch.Tensor:
		"""Forward pass with flat latent+action input.

		Args:
			za (Tensor[B, L + A]): Concatenation of flat latent and action.
		"""
		B, D = za.shape
		latent_dim = self.C * self.H * self.W
		if D != latent_dim + self.action_dim:
			raise RuntimeError(f"Expected input dim {latent_dim + self.action_dim}, got {D}")
		# Split back into latent and action
		z, a = za[:, :latent_dim], za[:, latent_dim:]
		z_sp = z.view(B, self.C, self.H, self.W)
		a_sp = a.view(B, self.action_dim, 1, 1).expand(B, self.action_dim, self.H, self.W)
		za_sp = torch.cat([z_sp, a_sp], dim=1)
		out = self.conv(za_sp)
		out = self.bn(out)
		out = self.act(out)
		out = out + z_sp
		out = self.act(out)
		out = self.res_block(out)
		return out.view(B, latent_dim)

class ConvLargeEncoder(nn.Module):
	"""Modern ResNet-style convolutional encoder for larger images.

	Builds a conv "body" that ends in a 4D feature map, followed by a
	Flatten + optional linear projection + optional activation.
	The `body` attribute can be used directly when a spatial (C,H,W)
	latent is required (e.g. for CNN dynamics shape inference).
	"""

	def __init__(
		self,
		in_shape,
		num_channels: int,
		out_dim: int,
		act: nn.Module | None = None,
		num_blocks: int = 4,
		use_projection: bool = True,
	):
		super().__init__()
		assert in_shape[-1] in (64, 128), "Expected image size 64x64 or 128x128"
		assert num_blocks >= 1, "num_blocks must be >= 1"

		layers_body = [ShiftAug(), PixelPreprocess()]

		# Initial stem
		input_channels = in_shape[0]
		layers_body.append(nn.Conv2d(input_channels, num_channels, kernel_size=7, stride=2, padding=3, bias=False))
		layers_body.append(nn.BatchNorm2d(num_channels))
		layers_body.append(nn.ReLU(inplace=False))

		# Residual blocks with occasional downsampling
		channels = num_channels
		for i in range(num_blocks):
			stride = 2 if (i % 2 == 1) else 1
			block = ResidualBlock(channels, channels, stride=stride)
			layers_body.append(block)

		self.body = nn.Sequential(*layers_body)  # returns 4D features
		self.flatten = nn.Flatten()
		self.proj = None
		if use_projection:
			with torch.no_grad():
				flat_dim = self.flatten(self.body(torch.zeros(1, *in_shape))).shape[-1]
			self.proj = nn.Linear(flat_dim, out_dim)
		self.act = act

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		feat = self.body(x)
		flat = self.flatten(feat)
		if self.proj is not None:
			flat = self.proj(flat)
		if self.act is not None:
			flat = self.act(flat)
		return flat

def enc(cfg, out={}):
	"""Returns a dictionary of encoders for each observation in the dict."""
	for k in cfg.obs_shape.keys():
		if k == 'state':
			out[k] = mlp(
				cfg.obs_shape[k][0] + cfg.task_dim,
				max(cfg.num_enc_layers - 1, 1) * [cfg.enc_dim],
				cfg.latent_dim,
				act=SimNorm(cfg),
			)
		elif k == 'rgb':
			# Select RGB encoder type via explicit config flag (no implicit defaults).
			if cfg.rgb_encoder_type == "default":
				out[k] = ConvEncoder(
					cfg.obs_shape[k],
					cfg.num_channels,
					cfg.latent_dim,
					act=SimNorm(cfg),
					use_projection=cfg.cnn_use_projection,
				)
			elif cfg.rgb_encoder_type == "large":
				out[k] = ConvLargeEncoder(
					cfg.obs_shape[k],
					cfg.num_channels,
					cfg.latent_dim,
					act=SimNorm(cfg),
					num_blocks=cfg.large_conv_num_blocks,
					use_projection=cfg.cnn_use_projection,
				)
			else:
				raise NotImplementedError(f"Unknown rgb_encoder_type {cfg.rgb_encoder_type}")
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
