import torch
import torch.nn.functional as F
from tensordict import TensorDict
import logging
log = logging.getLogger(__name__)

def soft_ce(pred, target, cfg):
	"""Computes the cross entropy loss between predictions and soft targets.
	Args:
		pred: Tensor of shape (..., num_classes) - raw, unnormalized scores for each class.
		target: Tensor of shape (..., num_classes) - target probabilities for each class.
		cfg: config object (needs num_bins).
	Returns:
		Tensor of shape (...) - the computed cross entropy loss.
"""
	pred = F.log_softmax(pred, dim=-1)
	return -(target * pred).sum(-1, keepdim=True)


def log_std(x, low, dif):
	return low + 0.5 * dif * (torch.tanh(x) + 1)


def gaussian_logprob(eps, log_std):
	"""Compute Gaussian log probability."""
	residual = -0.5 * eps.pow(2) - log_std
	log_prob = residual - 0.9189385175704956
	return log_prob.sum(-1, keepdim=True)


def squash(mu, pi, log_pi):
	"""Apply squashing function."""
	mu = torch.tanh(mu)
	pi = torch.tanh(pi)
	squashed_pi = torch.log(F.relu(1 - pi.pow(2)) + 1e-6)
	log_pi = log_pi - squashed_pi.sum(-1, keepdim=True)
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
	"""Converts a batch of soft two-hot encoded vectors to scalars. Basically taking the expectation of the distribution."""
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






# ---------------- Distributional Value Utilities (New) ---------------- #

def value_support(cfg, device=None, dtype=torch.float32):
	"""Symlog-domain support centers z_j in [vmin,vmax]."""
	device = device or torch.device('cpu')
	v = torch.linspace(cfg.vmin, cfg.vmax, cfg.num_bins, device=device, dtype=dtype)
	return v

def real_value_support(cfg, device=None, dtype=torch.float32):
	"""Real-value support v_j = symexp(z_j)."""
	z = value_support(cfg, device=device, dtype=dtype)
	v = symexp(z)
	return v

def expected_from_logits(logits, cfg):
	"""Expectation (real value domain) from symlog-distribution logits."""
	if cfg.num_bins == 0:
		return logits
	if cfg.num_bins == 1:
		return symexp(logits)
	with torch.no_grad():
		v = real_value_support(cfg, device=logits.device, dtype=logits.dtype)
	probs = F.softmax(logits, dim=-1)
	out = (probs * v).sum(-1, keepdim=True)
	return out

def distributional_ce(pred_logits, target_probs):
	"""Cross entropy with soft target distribution (target_probs sums to 1)."""
	logp = F.log_softmax(pred_logits, dim=-1)
	out = -(target_probs * logp).sum(-1, keepdim=True)
	return out

def project_value_distribution(next_logits, reward, terminated, discount, cfg, eps=1e-4):
	"""Project a next-state categorical value distribution via Bellman backup onto the
	current fixed support (C51-style linear interpolation).

	Mathematical Formulation:
	Let bin centers in symlog domain be z_j ∈ [v_min, v_max], j = 0..B-1, equally spaced; real
	values v_j = symexp(z_j). Given next-state distribution p(V = v_j) = p_j, we want the
	distribution of Y = r + γ (1 - d) V. Y may fall between existing bin centers; we linearly
	interpolate Y's probability mass back onto the fixed support.

	Notation:
	S      : arbitrary leading batch (and possibly time) shape.
	B      : number of bins (cfg.num_bins).
	next_logits: Tensor[S, B]   (symlog logits of selected next critic distribution)
	reward     : Tensor[S, 1]   (real scalar rewards)
	terminated : Tensor[S, 1]   ({0,1})
	discount   : Tensor[S, 1]   (γ prior to termination mask)
	target_probs: Tensor[S, B]  (resulting projected categorical distribution, detached)

	Steps (vectorized):
	1. Convert logits to probabilities p_j.
	2. Compute effective discount γ_eff = γ (1 - d).
	3. Form projected real values y_j = r + γ_eff * v_j.
	4. Clamp y_j into [v_0, v_{B-1}] (mass outside support goes to boundary).
	5. Map y_j back to symlog domain y'_j = symlog(y_j) to locate relative to z_j grid.
	6. Fractional bin position: pos_j = (y'_j - v_min)/Δ, with Δ = bin_size.
	7. Distribute p_j across lower & upper neighbor bins with linear weights.
	8. Normalize and apply small epsilon smoothing.

	Args:
		next_logits (Tensor): shape S + [B].
		reward (Tensor): shape S + [1].
		terminated (Tensor): shape S + [1].
		discount (Tensor): shape S + [1].
		cfg: config object (needs num_bins>1, vmin, vmax, bin_size).
		eps (float): epsilon smoothing weight (0 disables).

	Returns:
		Tensor target_probs with shape S + [B]; detached; last-dim sums to 1.

	Precision: operations are performed in the dtype of next_logits; caller may wish to ensure
	float32 for numerical stability.
	"""
	B = cfg.num_bins
	assert B > 1, 'project_value_distribution requires num_bins > 1'

	device = next_logits.device
	dtype = next_logits.dtype

	# --- (1) Support in both domains ---
	z = value_support(cfg, device=device, dtype=dtype)    # [B] symlog centers z_j
	v = symexp(z)                                        # [B] real values v_j

	# Shape broadcast: view support as [1,1,...,B] to align with batch shape of next_logits.
	support_view = (1,) * (next_logits.ndim - 1) + (B,)
	v = v.view(*support_view)                            # shape S + [B]

	# --- (2) Convert logits to probabilities ---
	p_next = F.softmax(next_logits, dim=-1)              # shape S + [B]
	# Defensive check: ensure last dim matches cfg.num_bins and no accidental dimension explosion.
	assert p_next.shape[-1] == B, f"Unexpected last dim {p_next.shape[-1]} != num_bins {B}"

	# --- (3) Effective discount (zero if terminated) ---
	gamma_eff = discount * (1 - terminated)              # shape S + [1]

	# --- (4) Bellman transform in REAL space ---
	y = reward + gamma_eff * v                           # shape S + [B]

	# --- (5) Clamp to support range (C51 boundary mass treatment) ---
	v_min_real = v[..., 0:1]
	v_max_real = v[..., -1:]
	y = torch.clamp(y, v_min_real, v_max_real)

	# --- (6) Map back to symlog domain for localization ---
	y_symlog = symlog(y)                                 # shape S + [B]

	Δ = cfg.bin_size                                     # scalar bin width in symlog domain

	# --- (7) Fractional bin indices ---
	bin_pos = (y_symlog - cfg.vmin) / Δ                  # shape S + [B], real in [0,B-1]
	lower_idx = torch.floor(bin_pos)                     # shape S + [B]
	upper_idx = lower_idx + 1
	lower_idx = lower_idx.clamp(0, B - 1)
	upper_idx = upper_idx.clamp(0, B - 1)
	upper_w = (bin_pos - torch.floor(bin_pos)).clamp(0, 1)  # interpolation weight toward upper bin
	lower_w = 1 - upper_w

	# --- (8) Flatten leading dims for scatter_add efficiency ---
	flat = p_next.reshape(-1, B)                         # [N, B]
	lower_flat = lower_idx.reshape(-1, B).long()         # [N, B]
	upper_flat = upper_idx.reshape(-1, B).long()         # [N, B]
	lower_w_flat = lower_w.reshape(-1, B)                # [N, B]
	upper_w_flat = upper_w.reshape(-1, B)                # [N, B]

	# Accumulator for target probabilities
	target = torch.zeros_like(flat)                     # [N, B]

	# Distribute probability mass linearly (C51 interpolation)
	target.scatter_add_(1, lower_flat, flat * lower_w_flat)
	target.scatter_add_(1, upper_flat, flat * upper_w_flat)

	# Restore original shape
	target = target.view_as(p_next)                      # shape S + [B]

	# --- (9) Normalize & epsilon smoothing ---
	target = target / (target.sum(-1, keepdim=True) + 1e-8)
	if eps > 0:
		target = (1 - eps) * target + eps / B
		target = target / (target.sum(-1, keepdim=True) + 1e-8)

	# --- (10) Detach: targets are fixed for loss computation ---
	out = target.detach()
	return out

def distribution_variance(logits, cfg):
	"""Per-distribution variance (real domain) from logits.
	Returns tensor [...,1]."""
	if cfg.num_bins <= 1:
		return torch.zeros_like(logits[..., :1])
	v = real_value_support(cfg, device=logits.device, dtype=logits.dtype)
	while v.ndim < logits.ndim:
		v = v.unsqueeze(0)
	probs = F.softmax(logits, dim=-1)
	mean = (probs * v).sum(-1, keepdim=True)
	var = (probs * (v - mean)**2).sum(-1, keepdim=True)
	return var

def distribution_entropy(logits):
	probs = F.softmax(logits, dim=-1)
	logp = torch.log(probs + 1e-8)
	out = -(probs * logp).sum(-1, keepdim=True)
	return out


def gumbel_softmax_sample(p, temperature=1.0, dim=0):
	"""Sample from the Gumbel-Softmax distribution."""
	logits = p.log()
	gumbels = (
		-torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
	)  # ~Gumbel(0,1)
	gumbels = (logits + gumbels) / temperature  # ~Gumbel(logits,tau)
	y_soft = gumbels.softmax(dim)
	return y_soft.argmax(-1)