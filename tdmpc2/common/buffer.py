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
		self._sampler = SliceSampler(
			num_slices=self.cfg.batch_size,
			end_key=None,
			traj_key='episode',
			truncated_key=None,
			strict_length=True,
			cache_values=cfg.multitask,
		)
		self._batch_size = cfg.batch_size * (cfg.horizon+1)
		self._num_eps = 0
  
          # --- GPU prefetch machinery ---
		self._copy_stream = torch.cuda.Stream()
		self._prefetched_td_gpu = None
		self._primed = False

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
		if self._num_eps == 0:
			self._buffer = self._init(td[0])
		td = td.reshape(td.shape[0]*td.shape[1])
		self._buffer.extend(td)
		self._num_eps += num_new_eps
		self._primed = False; self._prefetched_td_gpu = None

		return self._num_eps

	def add(self, td):
		"""Add an episode to the buffer."""
		td['episode'] = torch.full_like(td['reward'], self._num_eps, dtype=torch.int64)
		if self._num_eps == 0:
			self._buffer = self._init(td)
		self._buffer.extend(td)
		self._num_eps += 1
		self._primed = False; self._prefetched_td_gpu = None
		return self._num_eps



	def sample(self):
		"""Sample a batch of subsequences from the buffer."""
		if not self._primed:
			self._preload_gpu()
			torch.cuda.current_stream().wait_stream(self._copy_stream)  # first batch only
			self._primed = True
		torch.cuda.current_stream().wait_stream(self._copy_stream)
		td = self._prefetched_td_gpu
		self._preload_gpu()  # overlap next copy with your compute on 'ready'

		obs = td.get('obs').contiguous()
		action = td.get('action')[1:].contiguous()
		reward = td.get('reward')[1:].unsqueeze(-1).contiguous()
		terminated = td.get('terminated', None)
		if terminated is not None:
			terminated = td.get('terminated')[1:].unsqueeze(-1).contiguous()
		else:
			terminated = torch.zeros_like(reward).contiguous()
		task = td.get('task', None)
		if task is not None:
			task = task[0].contiguous()
		return obs, action, reward, terminated, task


	def _preload_gpu(self):
		td_cpu = self._buffer.sample()  # CPU, pinned, already sliced/reshaped
		with torch.cuda.stream(self._copy_stream):
			# one async H2D per tensor; pinned + non_blocking allows overlap
			self._prefetched_td_gpu = td_cpu.to(self._device, non_blocking=True)