import threading

import torch
from tensordict.tensordict import TensorDict
from torchrl.data.replay_buffers import ReplayBuffer, LazyTensorStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SliceSampler
from common.logging_utils import get_logger

log = get_logger(__name__)


class Buffer():
	"""
	Replay buffer for TD-MPC2 training. Based on torchrl.
	Uses CUDA memory if available, and CPU memory otherwise.
	"""

	def __init__(self, cfg):
		self.cfg = cfg
		self._device = torch.device('cuda:0')
		self._capacity = min(cfg.buffer_size, cfg.steps)
		# Track global transition index and ring-buffer metadata to set priorities for new items
		self._global_step = 0  # monotonically increasing transition counter (used for recency priority)
		self._write_index = 0  # next write position in storage (ring buffer)
		self._size = 0         # number of valid items currently in storage (<= capacity)
		if self.cfg.sampler == 'uniform':
			self._sampler = SliceSampler(
				num_slices=self.cfg.batch_size,
				end_key=None,
				traj_key='episode',
				truncated_key=None,
				strict_length=True,
				cache_values=cfg.multitask,
			)
		elif self.cfg.sampler == 'recency_prioritized':
			from torchrl.data.replay_buffers.samplers import PrioritizedSliceSampler
			self._sampler = PrioritizedSliceSampler(
				max_capacity=self._capacity,
				num_slices=self.cfg.batch_size,
				alpha=self.cfg.prioritized_alpha,
				beta=self.cfg.prioritized_beta,
				eps=self.cfg.prioritized_eps,
				reduction='max',
				end_key=None,
				traj_key='episode',
				truncated_key=None,
				strict_length=True,
				cache_values=cfg.multitask,
				max_priority_within_buffer=True,
			)
			log.info('Initialized recency_prioritized sampler: alpha=%s, beta=%s, reduction=max, capacity=%s', cfg.prioritized_alpha, cfg.prioritized_beta, self._capacity)
   
		self._batch_size = cfg.batch_size * (cfg.horizon+1)
		self._num_eps = 0

		# --- GPU prefetch machinery ---
		self._copy_stream = torch.cuda.Stream()
		self._prefetched_td_gpu = None
		self._prefetch_thread = None
		self._prefetch_error = None
		self._primed = False
		self._buffer = None

	@property
	def capacity(self):
		"""Return the capacity of the buffer."""
		return self._capacity

	@property
	def num_eps(self):
		"""Return the number of episodes in the buffer."""
		return self._num_eps

	def _reserve_buffer(self, storage, t = None):
		"""
		Reserve a buffer with the given storage.
		"""
		return ReplayBuffer(
			storage=storage,
			sampler=self._sampler,
			pin_memory=self.cfg.pin_memory,
			prefetch=self.cfg.prefetch,
			batch_size=self._batch_size,
   			transform= self.transform_sample
		)

	def _init(self, tds):
		"""Initialize the replay buffer. Use the first episode to estimate storage requirements."""
		log.info('Buffer capacity: %s', f'{self._capacity:,}')
		mem_free, _ = torch.cuda.mem_get_info()
		bytes_per_step = sum([
				(v.numel()*v.element_size() if not isinstance(v, TensorDict) \
				else sum([x.numel()*x.element_size() for x in v.values()])) \
			for v in tds.values()
		]) / len(tds)
		total_bytes = bytes_per_step*self._capacity
		log.info('Storage required: %.2f GB', total_bytes/1e9)
		# Heuristic: decide whether to use CUDA or CPU memory
		storage_device = 'cpu' #cuda:0' if 2.5*total_bytes < mem_free else 'cpu'
		log.info('Using %s memory for storage.', storage_device.upper())
		self._storage_device = torch.device(storage_device)
		# Reset ring metadata when (re)initializing
		self._write_index = 0
		self._size = 0
		return self._reserve_buffer(
			LazyTensorStorage(self._capacity, device=self._storage_device), 
		)

	def transform_sample(self, td):
		td = td.view(-1, self.cfg.horizon+1).permute(1, 0)
		td = td.select("obs", "action", "reward", "terminated", "task", strict=False)
		if self._storage_device.type == 'cpu':
			td = td.pin_memory()
		return td
	 
	def load(self, td):
		"""
		Load a batch of episodes into the buffer. This is useful for loading data from disk,
		and is more efficient than adding episodes one by one.
		"""
		num_new_eps = len(td)
		episode_idx = torch.arange(self._num_eps, self._num_eps+num_new_eps, dtype=torch.int64)
		td['episode'] = episode_idx.unsqueeze(-1).expand(-1, td['reward'].shape[1])
		# Stamp chunk-level recency priority if using prioritized sampling
		if self.cfg.sampler == 'recency_prioritized':
			steps_shape = td['reward'].shape
			n_elems = steps_shape[0] * steps_shape[1]
			chunk_priority = self._global_step + n_elems
			self._global_step += n_elems
			td['steps'] = torch.full(steps_shape, chunk_priority, dtype=torch.long)
		if self._num_eps == 0:
			self._buffer = self._init(td[0])
		td = td.reshape(td.shape[0]*td.shape[1])
		# If prioritized, update priorities for new range based on recency (use steps as float)
		if self.cfg.sampler == 'recency_prioritized':
			priorities = td.get('steps').to(torch.float32)
			# Compute ring indices where data will be written
			n = priorities.shape[0]
			idx = (torch.arange(n, dtype=torch.long) + self._write_index) % self._capacity
			self._buffer.extend(td)
			# Update internal sampler priorities
			self._buffer.update_priority(index=idx, priority=priorities)
			# Advance ring pointers
			self._write_index = int((self._write_index + n) % self._capacity)
			self._size = min(self._size + n, self._capacity)
			log.info('Buffer.load: added=%s, chunk_priority=%.0f, write_idx=[%s..%s], size=%s, num_eps=%s', n, priorities[0].item(), (int(self._write_index - n) % self._capacity), (int(self._write_index - 1) % self._capacity), self._size, self._num_eps + num_new_eps)
		else:
			self._buffer.extend(td)
		self._num_eps += num_new_eps
		self._primed = False
		self._prefetched_td_gpu = None
		if self._prefetch_thread is not None and self._prefetch_thread.is_alive():
			self._prefetch_thread.join()
		self._prefetch_thread = None
		self._prefetch_error = None

		return self._num_eps

	def add(self, td, end_episode):
		"""Add an episode to the buffer."""
		td['episode'] = torch.full_like(td['reward'], self._num_eps, dtype=torch.int64)
		# Stamp chunk-level recency priority if using prioritized sampling
		if self.cfg.sampler == 'recency_prioritized':
			steps_shape = td['reward'].shape
			n_elems = steps_shape[0] * steps_shape[1]
			chunk_priority = self._global_step + n_elems
			self._global_step += n_elems
			td['steps'] = torch.full(steps_shape, chunk_priority, dtype=torch.long)
		if self._num_eps == 0:
			self._buffer = self._init(td)
		if self.cfg.sampler == 'recency_prioritized':
			# Flatten view to compute contiguous indices and priorities
			td_flat = td.reshape(td.shape[0]*td.shape[1])
			priorities = td_flat.get('steps').to(torch.float32)
			n = priorities.shape[0]
			idx = (torch.arange(n, dtype=torch.long) + self._write_index) % self._capacity
			self._buffer.extend(td_flat)
			self._buffer.update_priority(index=idx, priority=priorities)
			self._write_index = int((self._write_index + n) % self._capacity)
			self._size = min(self._size + n, self._capacity)
			log.info('Buffer.add: added=%s, chunk_priority=%.0f, write_idx=[%s..%s], size=%s, ep=%s, end_ep=%s', n, priorities[0].item(), (int(self._write_index - n) % self._capacity), (int(self._write_index - 1) % self._capacity), self._size, self._num_eps, end_episode)
		else:
			self._buffer.extend(td)
		if end_episode:
			self._num_eps += 1
		self._primed = False
		self._prefetched_td_gpu = None
		if self._prefetch_thread is not None and self._prefetch_thread.is_alive():
			self._prefetch_thread.join()
		self._prefetch_thread = None
		self._prefetch_error = None
		return self._num_eps



	def sample(self):
		"""Sample a batch of subsequences from the buffer."""

		if self._buffer is None:
			raise RuntimeError('Replay buffer not initialized before sampling.')
		if not self._primed:
			self._launch_prefetch_thread()
			self._await_prefetch()
			self._primed = True
		else:
			self._await_prefetch()
		if self._prefetched_td_gpu is None:
			raise RuntimeError('Prefetch worker completed without producing a batch.')
		obs, action, reward, terminated, task = self._prefetched_td_gpu
		self._prefetched_td_gpu = None
		self._launch_prefetch_thread()
		return obs, action, reward, terminated, task

	def _launch_prefetch_thread(self):
		if self._buffer is None:
			raise RuntimeError('Replay buffer not initialized before prefetch start.')
		if self._prefetch_thread is not None and self._prefetch_thread.is_alive():
			raise RuntimeError('Prefetch requested while previous worker is still running.')
		self._prefetch_error = None
		self._prefetch_thread = threading.Thread(
			target=self._preload_gpu,
			name='ReplayPrefetchWorker',
			daemon=True,
		)
		self._prefetch_thread.start()

	def _await_prefetch(self):
		if self._prefetch_thread is None:
			raise RuntimeError('Prefetch thread missing before await.')
		self._prefetch_thread.join()
		self._prefetch_thread = None
		if self._prefetch_error is not None:
			error = self._prefetch_error
			self._prefetch_error = None
			raise RuntimeError('Prefetch worker failed while sampling.') from error
		torch.cuda.current_stream().wait_stream(self._copy_stream)

	def _preload_gpu(self):
		try:
			td_cpu = self._buffer.sample()
			obs_cpu, action_cpu, reward_cpu, terminated_cpu, task_cpu = self.from_td(td_cpu)
			with torch.cuda.stream(self._copy_stream):
				obs = obs_cpu.to(self._device, non_blocking=True)
				action = action_cpu.to(self._device, non_blocking=True)
				reward = reward_cpu.to(self._device, non_blocking=True)
				terminated = terminated_cpu.to(self._device, non_blocking=True)
				task = task_cpu.to(self._device, non_blocking=True) if task_cpu is not None else None
			self._prefetched_td_gpu = (obs, action, reward, terminated, task)
		except Exception as exc:  # pylint: disable=broad-except
			self._prefetch_error = exc
			self._prefetched_td_gpu = None

    

	# @torch.compile(mode='reduce-overhead')
	def from_td(self, td):
       
		obs = td.get('obs').contiguous()
		action = td.get('action')[1:].contiguous()
		reward = td.get('reward')[1:].unsqueeze(-1).contiguous()
		if self.cfg.episodic:
			terminated = td.get('terminated')[1:].unsqueeze(-1).contiguous()
		else:
			terminated = torch.zeros_like(reward, device=reward.device)
		if self.cfg.multitask:
			task = td.get('task')[0].contiguous()
		else:
			task = None
		return obs, action, reward, terminated, task

	def empty(self):
		"""Empty the buffer."""
		if self._prefetch_thread is not None and self._prefetch_thread.is_alive():
			self._prefetch_thread.join()
		self._prefetch_thread = None
		self._prefetch_error = None
		self._num_eps = 0
		self._primed = False
		self._prefetched_td_gpu = None
		self._buffer.empty()
		return