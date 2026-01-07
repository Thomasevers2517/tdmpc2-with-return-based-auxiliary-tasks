import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import from_modules
from copy import deepcopy


class MLPWithPrior(nn.Module):
	"""
	MLP with an optional frozen random prior for ensemble diversity.
	
	Combines a trainable main MLP with a frozen prior network. The prior output
	is L2-normalized and scaled, then added to the main output. This encourages
	diverse representations across ensemble members even when the main networks
	would otherwise collapse to similar predictions (e.g., in sparse reward settings).
	
	The prior_scale is dimension-aware: it controls the per-dimension perturbation
	magnitude. With prior_scale=1.0, each output dimension gets ~1.0 perturbation.
	
	Args:
		in_dim (int): Input dimension.
		hidden_dims (list[int]): Hidden layer dimensions for main MLP.
		out_dim (int): Output dimension.
		prior_hidden_div (int): Divisor for prior hidden dim (prior_hidden = hidden_dim // div).
		prior_scale (float): Per-dimension perturbation magnitude. 0 = no prior.
		act: Optional activation for output layer (e.g., SimNorm for dynamics).
		dropout (float): Dropout probability for main MLP.
	"""
	
	def __init__(
		self,
		in_dim: int,
		hidden_dims: list,
		out_dim: int,
		prior_hidden_div: int = 4,
		prior_scale: float = 1.0,
		act=None,
		dropout: float = 0.,
	):
		super().__init__()
		self.prior_scale = prior_scale
		self.out_dim = out_dim
		
		# Main MLP (trainable)
		self.main_mlp = mlp(in_dim, hidden_dims, out_dim, act=act, dropout=dropout)
		
		# Prior MLP (frozen) - only if prior_scale > 0
		if prior_scale > 0 and len(hidden_dims) > 0:
			prior_hidden = max(hidden_dims[0] // prior_hidden_div, 8)  # min 8 to avoid degenerate priors
			prior_dims = [prior_hidden] * len(hidden_dims)
			# Build prior MLP manually with ReLU activations (no LayerNorm)
			prior_layers = []
			dims = [in_dim] + prior_dims + [out_dim]
			for i in range(len(dims) - 1):
				prior_layers.append(nn.Linear(dims[i], dims[i+1]))
				if i < len(dims) - 2:  # No activation on output
					prior_layers.append(nn.ReLU())
			self.prior_mlp = nn.Sequential(*prior_layers)
			# Initialize with Kaiming (preserves variance through ReLU layers)
			for m in self.prior_mlp.modules():
				if isinstance(m, nn.Linear):
					nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
					nn.init.zeros_(m.bias)
			# Freeze prior
			for p in self.prior_mlp.parameters():
				p.requires_grad_(False)
		else:
			self.prior_mlp = None
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""
		Forward pass: main_mlp(x) + normalized_prior(x) * scale * sqrt(out_dim).
		
		Args:
			x (Tensor[..., in_dim]): Input tensor.
		
		Returns:
			Tensor[..., out_dim]: Output with prior perturbation.
		"""
		out = self.main_mlp(x)  # float32[..., out_dim]
		
		if self.prior_scale > 0 and self.prior_mlp is not None:
			prior_out = self.prior_mlp(x)  # float32[..., out_dim]
			# L2 normalize, then scale by sqrt(out_dim) so prior_scale is per-dimension
			prior_out = prior_out / (prior_out.norm(dim=-1, keepdim=True) + 1e-6)
			prior_out = prior_out * (self.prior_scale * math.sqrt(self.out_dim))
			out = out + prior_out
		
		return out
	
	def __repr__(self):
		prior_info = f", prior_scale={self.prior_scale}" if self.prior_scale > 0 else ""
		return f"MLPWithPrior(main_mlp={self.main_mlp}{prior_info})"


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


def mlp(in_dim, mlp_dims, out_dim, act=None, dropout=0., dropout_layer=0):
	"""
	Basic building block of TD-MPC2.
	MLP with LayerNorm, Mish activations, and optionally dropout.
	
	Args:
		in_dim: Input dimension.
		mlp_dims: Hidden layer dimensions (int or list).
		out_dim: Output dimension.
		act: Optional activation for output layer (e.g., SimNorm).
		dropout: Dropout rate.
		dropout_layer: Index of hidden layer to apply dropout (0=first, -1=last hidden layer).
	"""
	if isinstance(mlp_dims, int):
		mlp_dims = [mlp_dims]
	dims = [in_dim] + mlp_dims + [out_dim]
	mlp = nn.ModuleList()
	num_hidden = len(dims) - 2  # Number of hidden layers (excluding output)
	dropout_idx = dropout_layer if dropout_layer >= 0 else num_hidden + dropout_layer
	for i in range(num_hidden):
		layer_dropout = dropout if i == dropout_idx else 0.
		mlp.append(NormedLinear(dims[i], dims[i+1], dropout=layer_dropout))
	mlp.append(NormedLinear(dims[-2], dims[-1], act=act) if act else nn.Linear(dims[-2], dims[-1]))
	return nn.Sequential(*mlp)


class DynamicsHeadWithPrior(nn.Module):
	"""
	Dynamics head with an optional frozen random prior for ensemble diversity.
	
	Uses MLPWithPrior internally, then applies SimNorm to the output.
	This encourages diverse representations across ensemble members.

	Args:
		in_dim (int): Input dimension (latent_dim + action_dim + task_dim).
		mlp_dims (list[int]): Hidden dimensions for the main MLP.
		out_dim (int): Output dimension (latent_dim).
		cfg: Config object with simnorm_dim and prior settings.
		prior_hidden_div (int): Divisor for prior hidden dim.
		prior_scale (float): Per-dimension perturbation magnitude. 0 = no prior.
		dropout (float): Dropout probability for first layer.
	"""

	def __init__(
		self,
		in_dim: int,
		mlp_dims: list,
		out_dim: int,
		cfg,
		prior_hidden_div: int = 4,
		prior_scale: float = 1.0,
		dropout: float = 0.,
	):
		super().__init__()
		self.prior_scale = prior_scale
		
		# MLPWithPrior handles main + prior combination
		self.mlp_with_prior = MLPWithPrior(
			in_dim=in_dim,
			hidden_dims=mlp_dims,
			out_dim=out_dim,
			prior_hidden_div=prior_hidden_div,
			prior_scale=prior_scale,
			act=None,  # No activation before SimNorm
			dropout=dropout,
		)
		
		# SimNorm applied after MLPWithPrior
		self.simnorm = SimNorm(cfg)
		
		# Expose main_mlp for weight_init compatibility
		self.main_mlp = self.mlp_with_prior.main_mlp

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""
		Forward pass: MLPWithPrior(x), then SimNorm.

		Args:
			x (Tensor[..., in_dim]): Input tensor.

		Returns:
			Tensor[..., out_dim]: Normalized dynamics prediction.
		"""
		out = self.mlp_with_prior(x)  # float32[..., out_dim]
		return self.simnorm(out)

	def __repr__(self):
		prior_info = f", prior_scale={self.prior_scale}" if self.prior_scale > 0 else ""
		return f"DynamicsHeadWithPrior({self.mlp_with_prior}{prior_info})"


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
			# Apply dropout at last hidden layer (before SimNorm) to allow encoder compute before noise
			out[k] = mlp(cfg.obs_shape[k][0] + cfg.task_dim, max(cfg.num_enc_layers-1, 1)*[cfg.enc_dim], cfg.latent_dim, act=SimNorm(cfg), dropout=encoder_dropout, dropout_layer=-1)
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
