import torch
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
from tensordict import TensorDict


def soft_ce(pred, target, cfg):
	"""Computes the cross entropy loss between predictions and soft targets. 
	Args:	
		pred: (batch_size, num_bins) - Logits
		target: (batch_size, 1) - Soft targets
		cfg: Config object with num_bins attribute
	Returns:
		loss: (batch_size, 1) - Cross entropy loss
	"""
 
	pred = F.log_softmax(pred, dim=-1)
	target = two_hot(target, cfg)
	return -(target * pred).sum(-1, keepdim=True)


def log_std(x, low, dif):
	return low + 0.5 * dif * (torch.tanh(x) + 1)


def gaussian_logprob(eps, log_std):
	"""Compute Gaussian log probability."""
	residual = -0.5 * eps.pow(2) - log_std
	log_prob = residual - 0.9189385175704956
	return log_prob.sum(-1, keepdim=True)


def squash(mu, pi, log_pi, jacobian_scale=1.0):
	"""Apply tanh squashing function with configurable Jacobian correction.
	
	Args:
		mu: Pre-squash mean.
		pi: Pre-squash action samples.
		log_pi: Pre-squash log probability.
		jacobian_scale: Multiplier for Jacobian correction (default 1.0).
			1.0 = mathematically correct (full penalty for confidence near ±1)
			<1.0 = reduced penalty (more lenient on extreme/saturated actions)
			0.0 = no correction (ignore tanh compression entirely)
			
	The Jacobian correction accounts for the change of variables when applying
	tanh. With scale=1.0, actions near ±1 incur a large log_prob penalty.
	Reducing the scale allows the policy to be more confident near boundaries.
	"""
	assert jacobian_scale <= 1.0, f"jacobian_scale must be <= 1.0, got {jacobian_scale}"
	mu = torch.tanh(mu)
	pi = torch.tanh(pi)
	# Jacobian correction: -sum(log(1 - tanh²(action)))
	jacobian_correction = torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
	log_pi = log_pi - jacobian_scale * jacobian_correction
	return mu, pi, log_pi


def int_to_one_hot(x, num_classes):
	"""
	Converts an integer tensor to a one-hot tensor.
	Supports batched inputs.
	"""
	one_hot = torch.zeros(*x.shape, num_classes, device=x.device)
	one_hot.scatter_(-1, x.unsqueeze(-1), 1)
	return one_hot


def symlog(x):
	"""
	Symmetric logarithmic function.
	Adapted from https://github.com/danijar/dreamerv3.
	"""
	return torch.sign(x) * torch.log(1 + torch.abs(x))


def symexp(x):
	"""
	Symmetric exponential function.
	Adapted from https://github.com/danijar/dreamerv3.
	"""
	return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


def two_hot(x, cfg):
	"""Converts a batch of scalars to soft two-hot encoded targets for discrete regression."""
	if cfg.num_bins == 0:
		return x
	elif cfg.num_bins == 1:
		return symlog(x)
	x = torch.clamp(symlog(x), cfg.vmin, cfg.vmax).squeeze(1)
	bin_idx = torch.floor((x - cfg.vmin) / cfg.bin_size)
	bin_offset = ((x - cfg.vmin) / cfg.bin_size - bin_idx).unsqueeze(-1)
	soft_two_hot = torch.zeros(x.shape[0], cfg.num_bins, device=x.device, dtype=x.dtype)
	bin_idx = bin_idx.long()
	soft_two_hot = soft_two_hot.scatter(1, bin_idx.unsqueeze(1), 1 - bin_offset)
	soft_two_hot = soft_two_hot.scatter(1, (bin_idx.unsqueeze(1) + 1) % cfg.num_bins, bin_offset)
	return soft_two_hot


def two_hot_inv(x, cfg):
	"""Converts a batch of soft two-hot encoded vectors to scalars."""
	if cfg.num_bins == 0:
		return x
	elif cfg.num_bins == 1:
		return symexp(x)
	dreg_bins = torch.linspace(cfg.vmin, cfg.vmax, cfg.num_bins, device=x.device, dtype=x.dtype)
	x = F.softmax(x, dim=-1)
	x = torch.sum(x * dreg_bins, dim=-1, keepdim=True)
	return symexp(x)


def shift_scale_distribution(
	logits: torch.Tensor,
	cfg,
	shift: torch.Tensor = None,
	scale: torch.Tensor = None,
) -> torch.Tensor:
	"""Shift and/or scale a categorical distribution over discrete regression bins.

	Given logits over K bins with centers c_i in symlog space, produces new logits
	representing the distribution of (scale * x + shift) where x ~ P(logits).

	Each source bin center c_i with probability p_i maps to a new value
	v_i = scale * c_i + shift. The probability mass p_i is then redistributed
	to the two neighboring target bins via linear interpolation (same scheme
	as two-hot encoding), preserving proper probability semantics.

	Operates entirely in symlog space (bin-center space). Values that fall
	outside [vmin, vmax] after transformation are clamped to the boundary bins.

	Args:
		logits (Tensor[..., K]): Input logits over K bins.
		cfg: Config with num_bins, vmin, vmax, bin_size.
		shift (Tensor[..., 1] or None): Additive shift in symlog space.
			Broadcast-compatible with logits' leading dims. None = no shift.
		scale (Tensor[..., 1] or None): Multiplicative scale in symlog space.
			Broadcast-compatible with logits' leading dims. None = no scale.

	Returns:
		Tensor[..., K]: Transformed logits (log-probabilities, unnormalized).
	"""
	K = cfg.num_bins
	assert logits.shape[-1] == K, f"Expected last dim {K}, got {logits.shape[-1]}"

	# No-op fast path
	if shift is None and scale is None:
		return logits

	device = logits.device
	dtype = logits.dtype

	# Bin centers in symlog space: float32[K]
	bins = torch.linspace(cfg.vmin, cfg.vmax, K, device=device, dtype=dtype)

	# Transform bin centers: float32[..., K] after broadcast
	# Start with [K], then apply scale and shift with broadcasting
	transformed = bins  # float32[K]
	if scale is not None:
		transformed = scale * transformed  # float32[..., K]
	if shift is not None:
		transformed = transformed + shift  # float32[..., K]

	# Clamp to valid bin range
	transformed = transformed.clamp(cfg.vmin, cfg.vmax)  # float32[..., K]

	# Continuous index into bin grid: which bins does each transformed value fall between?
	bin_size = float(cfg.bin_size)
	idx = (transformed - cfg.vmin) / bin_size  # float32[..., K], values in [0, K-1]
	lo = idx.floor().long().clamp(0, K - 2)  # int64[..., K]
	hi = lo + 1  # int64[..., K]
	frac = (idx - lo.float()).clamp(0.0, 1.0)  # float32[..., K], interpolation weight for hi

	# Convert input logits to probabilities
	probs = F.softmax(logits, dim=-1)  # float32[..., K]

	# Redistribute: for each source bin i with prob p_i, assign
	#   (1 - frac_i) * p_i  to  lo_i
	#   frac_i * p_i        to  hi_i
	# Using scatter_add for proper accumulation
	new_probs = torch.zeros_like(probs)  # float32[..., K]
	new_probs.scatter_add_(-1, lo, probs * (1.0 - frac))
	new_probs.scatter_add_(-1, hi, probs * frac)

	# Convert back to logits; add epsilon to avoid log(0)
	new_logits = torch.log(new_probs + 1e-8)  # float32[..., K]

	return new_logits


def gumbel_softmax_sample(p, temperature=1.0, dim=0):
	"""Sample from the Gumbel-Softmax distribution."""
	logits = p.log()
	gumbels = (
		-torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
	)  # ~Gumbel(0,1)
	gumbels = (logits + gumbels) / temperature  # ~Gumbel(logits,tau)
	y_soft = gumbels.softmax(dim)
	return y_soft.argmax(-1)


def termination_statistics(pred, target, eps=1e-9):
	"""Compute episode termination statistics."""
	pred = pred.squeeze(-1)
	target = target.squeeze(-1)
	rate = target.sum() / len(target)
	tp = ((pred > 0.5) & (target == 1)).sum()
	fn = ((pred <= 0.5) & (target == 1)).sum()
	fp = ((pred > 0.5) & (target == 0)).sum()
	recall = tp / (tp + fn + eps)
	precision = tp / (tp + fp + eps)
	f1 = 2 * (precision * recall) / (precision + recall + eps)
	return TensorDict({'termination_rate': rate,
			'termination_f1': f1})


def compute_knn_entropy(x: torch.Tensor, k: int = 5) -> torch.Tensor:
	"""Compute k-NN entropy estimator: mean(log(dist_to_kth_neighbor + 1)).
	
	Measures diversity/spread of points in representation space.
	Higher entropy indicates more diverse/spread-out observations.
	
	Args:
		x (Tensor[B, D]): Encoded representations.
		k (int): Find k-th nearest neighbor (excluding self).
	
	Returns:
		Tensor[]: Scalar mean entropy estimate.
	"""
	B, D = x.shape
	if B <= k:
		# Not enough points to compute k-th neighbor
		return torch.tensor(0.0, device=x.device, dtype=x.dtype)
	
	# Compute pairwise L2 distances: dists[i,j] = ||x_i - x_j||_2
	# Shape: [B, B]
	dists = torch.cdist(x, x, p=2)  # float32[B, B]
	
	# Exclude self: set diagonal to inf so self isn't picked as neighbor
	dists.fill_diagonal_(float('inf'))
	
	# Find k-th smallest distance for each point (k-th nearest neighbor)
	# torch.kthvalue returns (values, indices) for k-th smallest along dim
	kth_dists, _ = torch.kthvalue(dists, k=k, dim=1)  # float32[B]
	
	# Entropy estimate: log(dist + 1) per the RE3 paper formula
	entropy_per_point = torch.log(kth_dists + 1)  # float32[B]
	
	return entropy_per_point.mean()  # scalar


def kl_div_gaussian(
	p_mean: torch.Tensor,   # float32[*, A]
	p_std: torch.Tensor,    # float32[*, A]
	q_mean: torch.Tensor,   # float32[*, A]
	q_std: torch.Tensor,    # float32[*, A]
) -> torch.Tensor:
	"""KL divergence between two diagonal Gaussians: KL(P || Q).
	
	Computes KL(N(p_mean, p_std) || N(q_mean, q_std)) using torch.distributions.
	
	Args:
		p_mean (Tensor[*, A]): Mean of distribution P.
		p_std (Tensor[*, A]): Std of distribution P (positive).
		q_mean (Tensor[*, A]): Mean of distribution Q.
		q_std (Tensor[*, A]): Std of distribution Q (positive).
	
	Returns:
		Tensor[*, A]: Per-dimension KL divergence (sum over A for total KL).
	"""
	p = Normal(p_mean, p_std)
	q = Normal(q_mean, q_std)
	
	return kl_divergence(p, q)  # float32[*, A]
