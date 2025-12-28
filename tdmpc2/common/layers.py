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


class DynamicsHeadWithPrior(nn.Module):
	"""
	Dynamics head with an optional frozen random prior for ensemble diversity.

	Combines a trainable main MLP with a frozen prior network. The prior output
	is added BEFORE SimNorm, allowing the main network to learn to compensate.
	This encourages diverse representations across ensemble members.

	Args:
		in_dim (int): Input dimension (latent_dim + action_dim + task_dim).
		mlp_dims (list[int]): Hidden dimensions for the main MLP.
		out_dim (int): Output dimension (latent_dim).
		cfg: Config object with simnorm_dim and prior settings.
		prior_enabled (bool): Whether to add the frozen prior network.
		prior_hidden_dim (int): Hidden dimension for prior MLP.
		prior_scale (float): Scale factor for prior output.
		dropout (float): Dropout probability for first layer.
	"""

	def __init__(
		self,
		in_dim: int,
		mlp_dims: list,
		out_dim: int,
		cfg,
		prior_enabled: bool = False,
		prior_hidden_dim: int = 32,
		prior_scale: float = 1.0,
		dropout: float = 0.,
	):
		super().__init__()
		self.prior_enabled = prior_enabled
		self.prior_scale = prior_scale

		# Main trainable MLP (without final activation - SimNorm applied after sum)
		self.main_mlp = mlp(in_dim, mlp_dims, out_dim, act=None, dropout=dropout)

		# Frozen prior network: small MLP with Tanh to bound outputs to [-1, 1]
		if prior_enabled:
			self.prior_mlp = nn.Sequential(
				nn.Linear(in_dim, prior_hidden_dim),
				nn.ReLU(),
				nn.Linear(prior_hidden_dim, prior_hidden_dim),
				nn.ReLU(),
				nn.Linear(prior_hidden_dim, out_dim),
				nn.Tanh(),  # Bound to [-1, 1]
			)
			# Freeze prior parameters
			for param in self.prior_mlp.parameters():
				param.requires_grad = False
		else:
			self.prior_mlp = None

		# SimNorm applied after summing main + prior
		self.simnorm = SimNorm(cfg)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""
		Forward pass: main_mlp(x) + prior_scale * prior_mlp(x), then SimNorm.

		Args:
			x (Tensor[..., in_dim]): Input tensor.

		Returns:
			Tensor[..., out_dim]: Normalized dynamics prediction.
		"""
		out = self.main_mlp(x)  # float32[..., out_dim]

		if self.prior_enabled and self.prior_mlp is not None:
			# Prior output is in [-1, 1] due to Tanh, scaled by prior_scale
			prior_out = self.prior_mlp(x)  # float32[..., out_dim]
			out = out + self.prior_scale * prior_out

		return self.simnorm(out)

	def __repr__(self):
		prior_info = f", prior_enabled={self.prior_enabled}, prior_scale={self.prior_scale}" if self.prior_enabled else ""
		return f"DynamicsHeadWithPrior(main_mlp={self.main_mlp}{prior_info})"


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


def enc(cfg, out={}):
	"""
	Returns a dictionary of encoders for each observation in the dict.
	
	Args:
		cfg: Config object with obs_shape, task_dim, num_enc_layers, enc_dim, 
		     latent_dim, encoder_dropout, num_channels, simnorm_dim.
		out: Optional dict to populate (default empty).
	
	Returns:
		nn.ModuleDict of encoders keyed by observation type ('state', 'rgb').
	"""
	encoder_dropout = getattr(cfg, 'encoder_dropout', 0.0)
	for k in cfg.obs_shape.keys():
		if k == 'state':
			out[k] = mlp(cfg.obs_shape[k][0] + cfg.task_dim, max(cfg.num_enc_layers-1, 1)*[cfg.enc_dim], cfg.latent_dim, act=SimNorm(cfg), dropout=encoder_dropout)
		elif k == 'rgb':
			out[k] = conv(cfg.obs_shape[k], cfg.num_channels, cfg.latent_dim, act=SimNorm(cfg))
		else:
			raise NotImplementedError(f"Encoder for observation type {k} not implemented.")
	return nn.ModuleDict(out)


def api_model_conversion(target_state_dict, source_state_dict):
	"""
	Converts a checkpoint from our old API to the new torch.compile compatible API.
	Handles both legacy _Qs naming (for backward compatibility with old checkpoints)
	and new _Vs naming.
	"""
	# check whether checkpoint is already in the new format (check both _Qs and _Vs patterns)
	if "_detach_Vs_params.0.weight" in source_state_dict or "_detach_Qs_params.0.weight" in source_state_dict:
		return source_state_dict

	name_map = ['weight', 'bias', 'ln.weight', 'ln.bias']
	new_state_dict = dict()

	# rename keys - handle both _Qs (legacy) and _Vs (new) patterns
	for key, val in list(source_state_dict.items()):
		# Handle _Vs patterns (new)
		if key.startswith('_Vs.'):
			num = key[len('_Vs.params.'):]
			new_key = str(int(num) // 4) + "." + name_map[int(num) % 4]
			new_total_key = "_Vs.params." + new_key
			del source_state_dict[key]
			new_state_dict[new_total_key] = val
			new_total_key = "_detach_Vs_params." + new_key
			new_state_dict[new_total_key] = val
		elif key.startswith('_target_Vs.'):
			num = key[len('_target_Vs.params.'):]
			new_key = str(int(num) // 4) + "." + name_map[int(num) % 4]
			new_total_key = "_target_Vs_params." + new_key
			del source_state_dict[key]
			new_state_dict[new_total_key] = val
		# Handle _Qs patterns (legacy - for loading old checkpoints)
		elif key.startswith('_Qs.'):
			num = key[len('_Qs.params.'):]
			new_key = str(int(num) // 4) + "." + name_map[int(num) % 4]
			new_total_key = "_Vs.params." + new_key  # Convert to _Vs
			del source_state_dict[key]
			new_state_dict[new_total_key] = val
			new_total_key = "_detach_Vs_params." + new_key
			new_state_dict[new_total_key] = val
		elif key.startswith('_target_Qs.'):
			num = key[len('_target_Qs.params.'):]
			new_key = str(int(num) // 4) + "." + name_map[int(num) % 4]
			new_total_key = "_target_Vs_params." + new_key  # Convert to _Vs
			del source_state_dict[key]
			new_state_dict[new_total_key] = val

	# add batch_size and device from target_state_dict to new_state_dict
	for prefix in ('_Vs.', '_detach_Vs_', '_target_Vs_'):
		for key in ('__batch_size', '__device'):
			new_key = prefix + 'params.' + key
			if new_key in target_state_dict:
				new_state_dict[new_key] = target_state_dict[new_key]

	# check that every key in new_state_dict is in target_state_dict
	for key in new_state_dict.keys():
		assert key in target_state_dict, f"key {key} not in target_state_dict"
	# check that all Vs keys in target_state_dict are in new_state_dict
	for key in target_state_dict.keys():
		if 'Vs' in key:
			assert key in new_state_dict, f"key {key} not in new_state_dict"
	# check that source_state_dict contains no Vs or Qs keys
	for key in source_state_dict.keys():
		assert 'Vs' not in key and 'Qs' not in key, f"key {key} contains 'Vs' or 'Qs'"

	# copy log_std_min and log_std_max from target_state_dict to new_state_dict
	new_state_dict['log_std_min'] = target_state_dict['log_std_min']
	new_state_dict['log_std_dif'] = target_state_dict['log_std_dif']
	if '_action_masks' in target_state_dict:
		new_state_dict['_action_masks'] = target_state_dict['_action_masks']

	# copy new_state_dict to source_state_dict
	source_state_dict.update(new_state_dict)

	return source_state_dict
