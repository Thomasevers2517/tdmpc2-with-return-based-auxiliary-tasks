import torch
import torch.nn.functional as F
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
