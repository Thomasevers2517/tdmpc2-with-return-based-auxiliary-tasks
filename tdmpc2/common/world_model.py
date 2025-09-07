from copy import deepcopy

import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from common import layers, math, init
from tensordict import TensorDict
from tensordict.nn import TensorDictParams


class WorldModel(nn.Module):
	"""
	TD-MPC2 implicit world model architecture.
	Can be used for both single-task and multi-task experiments.
	"""

	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		if cfg.multitask:
			self._task_emb = nn.Embedding(len(cfg.tasks), cfg.task_dim, max_norm=1)
			self.register_buffer("_action_masks", torch.zeros(len(cfg.tasks), cfg.action_dim))
			for i in range(len(cfg.tasks)):
				self._action_masks[i, :cfg.action_dims[i]] = 1.
		self._encoder = layers.enc(cfg)
		self._dynamics = layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], cfg.latent_dim, act=layers.SimNorm(cfg))
		self._reward = layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], max(cfg.num_bins, 1))
		self._termination = layers.mlp(cfg.latent_dim + cfg.task_dim, 2*[cfg.mlp_dim], 1) if cfg.episodic else None
		self._pi = layers.mlp(cfg.latent_dim + cfg.task_dim, 2*[cfg.mlp_dim], 2*cfg.action_dim)
		self._Qs = layers.Ensemble([layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], max(cfg.num_bins, 1), dropout=cfg.dropout) for _ in range(cfg.num_q)])

		# ------------------------------------------------------------------
		# Auxiliary multi-gamma action-value heads (training-only)
		# Simplified (no ensemble dimension) per user request.
		# We construct either:
		#   joint: a single MLP outputting (G_aux * K) logits reshaped to (G_aux,K)
		#   separate: ModuleList of G_aux MLP heads each outputting K logits
		# These heads are used only for auxiliary supervision and never for
		# planning or policy bootstrapping. Therefore, we do not maintain
		# target networks or ensembles (min/avg collapse to identity).
		# ------------------------------------------------------------------
		self._num_aux_gamma = 0
		self._aux_joint_Qs = None      # now a single MLP head (not an Ensemble)
		self._aux_separate_Qs = None   # ModuleList[MLP]
		self._target_aux_joint_Qs = None
		self._target_aux_separate_Qs = None
		self._detach_aux_joint_Qs = None
		self._detach_aux_separate_Qs = None	
  
		if getattr(cfg, 'multi_gamma_gammas', None):
			gammas = cfg.multi_gamma_gammas
			self._num_aux_gamma = len(gammas)
			in_dim = cfg.latent_dim + cfg.action_dim + (cfg.task_dim if cfg.multitask else 0)
			if cfg.multi_gamma_head == 'joint':
				self._aux_joint_Qs = layers.mlp(in_dim, 2*[cfg.mlp_dim], max(cfg.num_bins * self._num_aux_gamma, 1), dropout=cfg.dropout)
			elif cfg.multi_gamma_head == 'separate':
				self._aux_separate_Qs = torch.nn.ModuleList([
					layers.mlp(in_dim, 2*[cfg.mlp_dim], max(cfg.num_bins, 1), dropout=cfg.dropout)
					for _ in range(self._num_aux_gamma)
				])
			else:
				raise ValueError(f"Unsupported multi_gamma_head: {cfg.multi_gamma_head}")

		self.apply(init.weight_init)
		init.zero_([self._reward[-1].weight, self._Qs.params["2", "weight"]])

		self.register_buffer("log_std_min", torch.tensor(cfg.log_std_min))
		self.register_buffer("log_std_dif", torch.tensor(cfg.log_std_max) - self.log_std_min)
		self.init()

	def init(self):
		# Create params
		self._detach_Qs_params = TensorDictParams(self._Qs.params.data, no_convert=True)
		self._target_Qs_params = TensorDictParams(self._Qs.params.data.clone(), no_convert=True)
  
		with self._detach_Qs_params.data.to("meta").to_module(self._Qs.module):
			self._detach_Qs = deepcopy(self._Qs)
			self._target_Qs = deepcopy(self._Qs)

		# Assign params to modules
		# We do this strange assignment to avoid having duplicated tensors in the state-dict -- working on a better API for this
		delattr(self._detach_Qs, "params")
		self._detach_Qs.__dict__["params"] = self._detach_Qs_params
		delattr(self._target_Qs, "params")
		self._target_Qs.__dict__["params"] = self._target_Qs_params

		# Auxiliary heads (non-ensemble): vectorized param buffers + target/detach clones
		if self._aux_joint_Qs is not None:
			# Initialize parameter vectors as buffers once
			if not hasattr(self, 'aux_joint_target_vec'):
				vec = parameters_to_vector(self._aux_joint_Qs.parameters()).detach().clone()
				self.register_buffer('aux_joint_target_vec', vec)
				self.register_buffer('aux_joint_detach_vec', vec.clone())
			# Create frozen clones and load vectors
			self._target_aux_joint_Qs = deepcopy(self._aux_joint_Qs)
			self._detach_aux_joint_Qs = deepcopy(self._aux_joint_Qs)
			for p in self._target_aux_joint_Qs.parameters():
				p.requires_grad_(False)
			for p in self._detach_aux_joint_Qs.parameters():
				p.requires_grad_(False)
			vector_to_parameters(self.aux_joint_target_vec, self._target_aux_joint_Qs.parameters())
			vector_to_parameters(self.aux_joint_detach_vec, self._detach_aux_joint_Qs.parameters())
		elif self._aux_separate_Qs is not None:
			# Build concatenated vectors and size map once
			if not hasattr(self, 'aux_separate_sizes'):
				self.aux_separate_sizes = [sum(p.numel() for p in h.parameters()) for h in self._aux_separate_Qs]
				vecs = [parameters_to_vector(h.parameters()).detach().clone() for h in self._aux_separate_Qs]
				full = torch.cat(vecs, dim=0)
				self.register_buffer('aux_separate_target_vec', full)
				self.register_buffer('aux_separate_detach_vec', full.clone())
			# Create frozen clones and load
			self._target_aux_separate_Qs = nn.ModuleList([deepcopy(h) for h in self._aux_separate_Qs])
			self._detach_aux_separate_Qs = nn.ModuleList([deepcopy(h) for h in self._aux_separate_Qs])
			for head in list(self._target_aux_separate_Qs) + list(self._detach_aux_separate_Qs):
				for p in head.parameters():
					p.requires_grad_(False)
			offset = 0
			for h, sz in zip(self._target_aux_separate_Qs, self.aux_separate_sizes):
				vector_to_parameters(self.aux_separate_target_vec[offset:offset+sz], h.parameters())
				offset += sz
			offset = 0
			for h, sz in zip(self._detach_aux_separate_Qs, self.aux_separate_sizes):
				vector_to_parameters(self.aux_separate_detach_vec[offset:offset+sz], h.parameters())
				offset += sz
   

	def __repr__(self):
		repr = 'TD-MPC2 World Model\n'
		modules = ['Encoder', 'Dynamics', 'Reward', 'Termination', 'Policy prior', 'Q-functions']
		# Optionally append auxiliary Q ensembles to representation
		aux_modules = []
		if self._num_aux_gamma > 0:
			modules.append('Aux Q-functions')
			aux_modules = [self._aux_joint_Qs if self._aux_joint_Qs is not None else self._aux_separate_Qs]
		for i, m in enumerate([self._encoder, self._dynamics, self._reward, self._termination, self._pi, self._Qs] + aux_modules):
			if m == self._termination and not self.cfg.episodic:
				continue
			repr += f"{modules[i]}: {m}\n"
		repr += "Learnable parameters: {:,}".format(self.total_params)
		return repr

	@property
	def total_params(self):
		return sum(p.numel() for p in self.parameters() if p.requires_grad)

	def to(self, *args, **kwargs):
		super().to(*args, **kwargs)
		self.init()
		return self

	def train(self, mode=True):
		"""
		Overriding `train` method to keep target Q-networks in eval mode.
		"""
		super().train(mode)
		self._target_Qs.train(False)
		if self._target_aux_joint_Qs is not None:
			self._target_aux_joint_Qs.train(False)
		if getattr(self, '_detach_aux_joint_Qs', None) is not None:
			self._detach_aux_joint_Qs.train(False)
		if getattr(self, '_target_aux_separate_Qs', None) is not None:
			for h in self._target_aux_separate_Qs:
				h.train(False)
		if getattr(self, '_detach_aux_separate_Qs', None) is not None:
			for h in self._detach_aux_separate_Qs:
				h.train(False)
		return self

	def soft_update_target_Q(self):
		"""
		Soft-update target Q-networks using Polyak averaging.
		"""
		self._target_Qs_params.lerp_(self._detach_Qs_params, self.cfg.tau)
		# Vectorized EMA for aux heads; emit debug checks if enabled
		if getattr(self, '_aux_joint_Qs', None) is not None:
			with torch.no_grad():
				online = parameters_to_vector(self._aux_joint_Qs.parameters()).detach()
				prev = self.aux_joint_target_vec.detach().clone()
				self.aux_joint_target_vec.lerp_(online, self.cfg.tau)
				vector_to_parameters(self.aux_joint_target_vec, self._target_aux_joint_Qs.parameters())
				# Detach mirrors online snapshot
				self.aux_joint_detach_vec.copy_(online)
				vector_to_parameters(self.aux_joint_detach_vec, self._detach_aux_joint_Qs.parameters())
		elif getattr(self, '_aux_separate_Qs', None) is not None:
			with torch.no_grad():
				vecs = [parameters_to_vector(h.parameters()).detach() for h in self._aux_separate_Qs]
				online = torch.cat(vecs, dim=0)
				prev = self.aux_separate_target_vec.detach().clone()
				self.aux_separate_target_vec.lerp_(online, self.cfg.tau)
				# Assign to target heads
				offset = 0
				for h, sz in zip(self._target_aux_separate_Qs, self.aux_separate_sizes):
					vector_to_parameters(self.aux_separate_target_vec[offset:offset+sz], h.parameters())
					offset += sz
				# Detach mirrors online
				self.aux_separate_detach_vec.copy_(online)
				offset = 0
				for h, sz in zip(self._detach_aux_separate_Qs, self.aux_separate_sizes):
					vector_to_parameters(self.aux_separate_detach_vec[offset:offset+sz], h.parameters())
					offset += sz
				
	def task_emb(self, x, task):
		"""
		Continuous task embedding for multi-task experiments.
		Retrieves the task embedding for a given task ID `task`
		and concatenates it to the input `x`.
		"""
		if isinstance(task, int):
			task = torch.tensor([task], device=x.device)
		emb = self._task_emb(task.long())
		if x.ndim == 3:
			emb = emb.unsqueeze(0).repeat(x.shape[0], 1, 1)
		elif emb.shape[0] == 1:
			emb = emb.repeat(x.shape[0], 1)
		return torch.cat([x, emb], dim=-1)

	def encode(self, obs, task):
		"""
		Encodes an observation into its latent representation.
		This implementation assumes a single state-based observation.
		"""
		if self.cfg.multitask:
			obs = self.task_emb(obs, task)
		if self.cfg.obs == 'rgb' and obs.ndim == 5:
			return torch.stack([self._encoder[self.cfg.obs](o) for o in obs])
		return self._encoder[self.cfg.obs](obs)

	def next(self, z, a, task):
		"""
		Predicts the next latent state given the current latent state and action.
		"""
		if self.cfg.multitask:
			z = self.task_emb(z, task)
		z = torch.cat([z, a], dim=-1)
		return self._dynamics(z)

	def reward(self, z, a, task):
		"""
		Predicts instantaneous (single-step) reward.
		"""
		if self.cfg.multitask:
			z = self.task_emb(z, task)
		z = torch.cat([z, a], dim=-1)
		return self._reward(z)
	
	def termination(self, z, task, unnormalized=False):
		"""
		Predicts termination signal.
		"""
		assert task is None
		if self.cfg.multitask:
			z = self.task_emb(z, task)
		if unnormalized:
			return self._termination(z)
		return torch.sigmoid(self._termination(z))
		

	def pi(self, z, task):
		"""
		Samples an action from the policy prior.
		The policy prior is a Gaussian distribution with
		mean and (log) std predicted by a neural network.
		"""
		if self.cfg.multitask:
			z = self.task_emb(z, task)

		# Gaussian policy prior
		mean, log_std = self._pi(z).chunk(2, dim=-1)
		log_std = math.log_std(log_std, self.log_std_min, self.log_std_dif)
		eps = torch.randn_like(mean)

		if self.cfg.multitask: # Mask out unused action dimensions
			mean = mean * self._action_masks[task]
			log_std = log_std * self._action_masks[task]
			eps = eps * self._action_masks[task]
			action_dims = self._action_masks.sum(-1)[task].unsqueeze(-1)
		else: # No masking
			action_dims = None

		log_prob = math.gaussian_logprob(eps, log_std)

		# Scale log probability by action dimensions
		size = eps.shape[-1] if action_dims is None else action_dims
		scaled_log_prob = log_prob * size

		# Reparameterization trick
		action = mean + eps * log_std.exp()
		mean, action, log_prob = math.squash(mean, action, log_prob)

		entropy_scale = scaled_log_prob / (log_prob + 1e-8)
		info = TensorDict({
			"mean": mean,
			"log_std": log_std,
			"action_prob": 1.,
			"entropy": -log_prob,
			"scaled_entropy": -log_prob * entropy_scale,
		})
		return action, info

	def Q_aux(self, z, a, task, return_type='all', target=False, detach=False):
		"""Predict auxiliary state-action value distributions (no ensemble).

		Args:
			z: (T,B,L) or (B,L) latent states
			a: aligned actions
			return_type: 'all' -> logits (T,B,G_aux,K); 'min'/'avg' -> scalar values (T,B,G_aux,1)
		"""
		if self._num_aux_gamma == 0:
			return None
		assert return_type in {'all','min','avg'}
		added_time = False
		if z.ndim == 2:
			z = z.unsqueeze(0); a = a.unsqueeze(0); added_time = True
		if self.cfg.multitask:
			z = self.task_emb(z, task)
		za = torch.cat([z, a], dim=-1)
		if self._aux_joint_Qs is not None:
			if target:
				out = self._target_aux_joint_Qs(za)
			elif detach:
				out = self._detach_aux_joint_Qs(za)
			else:
				out = self._aux_joint_Qs(za)  # (T,B,G_aux*K)
    
			T, B = out.shape[0], out.shape[1]
			out = out.view(T, B, self._num_aux_gamma, self.cfg.num_bins)
		elif self._aux_separate_Qs is not None:
			if target:
				outs = [head(za) for head in self._target_aux_separate_Qs]
			elif detach:
				outs = [head(za) for head in self._detach_aux_separate_Qs]
			else:
				outs = [head(za) for head in self._aux_separate_Qs]  # list[(T,B,K)]
			out = torch.stack(outs, dim=2)  # (T,B,G_aux,K)
   
		if return_type == 'all':
			return out
		vals = math.two_hot_inv(out, self.cfg)  # (T,B,G_aux,1)
		return vals  # 'min'/'avg' identical with single head

	def Q(self, z, a, task, return_type='min', target=False, detach=False):
		"""Predict state-action value.
		`return_type` can be one of [`min`, `avg`, `all`]:
			- `min`: return the minimum of two randomly subsampled Q-values.
			- `avg`: return the average of two randomly subsampled Q-values.
			- `all`: return all Q-values.
		`target` specifies whether to use the target Q-networks or not.
		"""
		assert return_type in {'min', 'avg', 'all'}

		if self.cfg.multitask:
			z = self.task_emb(z, task)

		z = torch.cat([z, a], dim=-1)
		if target:
			qnet = self._target_Qs
		elif detach:
			qnet = self._detach_Qs
		else:
			qnet = self._Qs
		out = qnet(z)

		if return_type == 'all':
			return out

		qidx = torch.randperm(self.cfg.num_q, device=out.device)[:2]
		Q = math.two_hot_inv(out[qidx], self.cfg)
		if return_type == "min":
			return Q.min(0).values
		return Q.sum(0) / 2
