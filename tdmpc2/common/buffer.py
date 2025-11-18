import threading

import torch
from tensordict.tensordict import TensorDict
from torchrl.data.replay_buffers import ReplayBuffer, LazyTensorStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SliceSampler
from common.logger import get_logger

log = get_logger(__name__)


class Buffer():
	"""
	Replay buffer for TD-MPC2 training. Based on torchrl.
	Uses CUDA memory if available, and CPU memory otherwise.
	"""

	def __init__(self, cfg, isTrainBuffer=True):
		self.cfg = cfg
		self._device = torch.device('cuda:0')
		self._capacity = min(cfg.buffer_size, cfg.steps)
		self.isTrainBuffer = isTrainBuffer
		self._size = 0         # number of valid items currently in storage (<= capacity)

		# --- Hot buffer configuration ---
		self._hot_enabled = bool(cfg.hot_buffer_enabled)
		self._hot_ratio = float(cfg.hot_buffer_ratio) if self._hot_enabled else 0.0
		self._hot_slices = int(round(self._hot_ratio * self.cfg.batch_size)) if self._hot_enabled else 0
		# assert cfg.hot_buffer_size > 2*(cfg.horizon + 1 + self._hot_slices) if self._hot_enabled else True, \
		# 	"hot_buffer_size must be > 2*(horizon + 1 + hot_slices) to avoid overflow"
   
		self._hot_size = int(cfg.hot_buffer_size+2*(cfg.horizon+ 1 + self._hot_slices)) if self._hot_enabled else 0

		self._main_slices = self.cfg.batch_size - self._hot_slices if self._hot_enabled else self.cfg.batch_size
		# Pre-compute flattened batch sizes per buffer (num_slices * (H+1))
		self._slice_len = (self.cfg.horizon + 1)
		self._main_batch_flat = self._main_slices * self._slice_len
		self._hot_batch_flat = self._hot_slices * self._slice_len
		if self._hot_enabled:
			if not (0.0 <= self._hot_ratio <= 1.0):
				raise ValueError(f"hot_buffer_ratio must be in [0, 1], got {self._hot_ratio}")
			if self._hot_ratio > 0.0 and self._hot_slices == 0:
				raise ValueError(
					f"hot_buffer_ratio={self._hot_ratio} with batch_size={self.cfg.batch_size} gives 0 hot samples. Increase ratio or batch size."
				)
			if self._hot_size <= 0:
				raise ValueError("hot_buffer_size must be > 0 when hot buffer is enabled")
			# Strict requirement from SliceSampler with strict_length=True: need at least horizon+1
			if self._hot_slices > 0 and self._hot_size < self._slice_len:
				raise ValueError(
					f"hot_buffer_size ({self._hot_size}) must be >= horizon+1 ({self._slice_len}) to sample at least one strict-length slice."
				)

		# Samplers for main and hot buffers
		self._sampler_main = SliceSampler(
				num_slices=self._main_slices,
				end_key=None,
				traj_key='episode',
				truncated_key=None,
				strict_length=True,
				cache_values=cfg.multitask,
			)
		self._sampler_hot = None
		if self._hot_enabled and self._hot_slices > 0:
			self._sampler_hot = SliceSampler(
					num_slices=self._hot_slices,
					end_key=None,
					traj_key='episode',
					truncated_key=None,
					strict_length=True,
					cache_values=cfg.multitask,
				)
		
		self._batch_size = cfg.batch_size * (cfg.horizon+1)  # legacy: combined flat size (not used for per-buffer RBs)
		self._num_eps = 0
		self._curr_ep_len = 0  # steps accumulated in the current episode (for hot gating)

		# --- GPU prefetch machinery ---
		self._copy_stream = torch.cuda.Stream()
		self._prefetched_td_gpu = None
		self._prefetch_thread = None
		self._prefetch_error = None
		self._primed = False
		self._buffer = None
		self._hot_buffer = None

	@property
	def capacity(self):
		"""Return the capacity of the buffer."""
		return self._capacity

	@property
	def num_eps(self):
		"""Return the number of episodes in the buffer."""
		return self._num_eps

	def _reserve_buffer(self, storage, sampler, batch_size_flat):
		"""
		Reserve a buffer with the given storage and sampler.
		"""
		return ReplayBuffer(
			storage=storage,
			sampler=sampler,
			pin_memory=self.cfg.pin_memory,
			prefetch=self.cfg.prefetch,
			batch_size=batch_size_flat,
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
		# Main buffer
		main_buf = self._reserve_buffer(
			LazyTensorStorage(self._capacity, device=self._storage_device),
			self._sampler_main,
			self._main_batch_flat,
		)
		# Optional hot buffer
		if self._hot_enabled and self._sampler_hot is not None:
			self._hot_buffer = self._reserve_buffer(
				LazyTensorStorage(self._hot_size+self.cfg.horizon, device=self._storage_device),
				self._sampler_hot,
				self._hot_batch_flat,
			)
			log.info(
				"Hot buffer enabled: ratio=%.3f, hot_slices=%d, capacity=%s",
				self._hot_ratio, self._hot_slices, f"{self._hot_size:,}"
			)
			# --- Expected sampling diagnostics (approximate) ---
			# Assumptions:
			#  * Each env step adds one new transition (ignoring buffer_update_interval discretization)
			#  * Item remains in hot buffer for ~hot_buffer_size steps (FIFO)
			#  * At each update, probability a particular item is drawn ≈ hot_slices / hot_buffer_size
			#  * Updates per env step: model = utd_ratio; actor-critic = utd_ratio * ac_utd_multiplier
			#  => Expected draws while item stays hot:
			#     E_model ≈ hot_slices * utd_ratio
			#     E_actor_critic ≈ hot_slices * utd_ratio * ac_utd_multiplier
			p_update_hot = self._hot_slices / self._hot_size
			updates_per_step_model = float(self.cfg.utd_ratio)
			updates_per_step_ac = float(self.cfg.utd_ratio) * float(self.cfg.ac_utd_multiplier)
			expected_model_hits = self._hot_slices * updates_per_step_model
			expected_ac_hits = self._hot_slices * updates_per_step_ac
			log.info(
				"Hot buffer sampling (expected per new item while hot): p(update|hot)=%.4f, E_model≈%.2f, E_actor_critic≈%.2f",
				p_update_hot, expected_model_hits, expected_ac_hits,
			)
		else:
			self._hot_buffer = None
		return main_buf

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

		self._buffer = self._init(td[0])
		td = td.reshape(td.shape[0]*td.shape[1])

		self._buffer.extend(td)
		if self._hot_buffer is not None:
			self._hot_buffer.extend(td)
		self._num_eps += num_new_eps
		self._curr_ep_len = 0
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

		# Initialize underlying storage only once, on first add (regardless of episode count).
		if self._buffer is None:
			self._buffer = self._init(td)

		self._buffer.extend(td)
		if self._hot_buffer is not None:
			self._hot_buffer.extend(td)
		# Track steps added to current episode for hot gating
		steps_added = int(td['reward'].numel())
		self._curr_ep_len += steps_added
		if end_episode:
			self._num_eps += 1
			self._curr_ep_len = 0
		self._primed = False
		self._prefetched_td_gpu = None
		if self._prefetch_thread is not None and self._prefetch_thread.is_alive():
			self._prefetch_thread.join()
		self._prefetch_thread = None
		self._prefetch_error = None
		return self._num_eps



	def sample(self):
		"""Sample a batch of subsequences from the buffer."""
		#TODO This is buffer class also used for validation, hotbuffer should not be used there but it is.
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
			# use_hot = (
			# 	self._hot_buffer is not None and
			# 	self._hot_slices > 0 and
			# 	self._curr_ep_len >= self._slice_len
			# )
			use_hot = self.isTrainBuffer
			if not use_hot:
				# Entire batch from main buffer
				td_cpu = self._buffer.sample()
				obs_cpu, action_cpu, reward_cpu, terminated_cpu, task_cpu = self.from_td(td_cpu)

			else:
				# Split between main and hot buffers, then concatenate along batch dimension
				td_main = self._buffer.sample()
				obs_m, act_m, rew_m, term_m, task_m = self.from_td(td_main)
				td_hot = self._hot_buffer.sample()
				obs_h, act_h, rew_h, term_h, task_h = self.from_td(td_hot)
				obs_cpu = torch.cat([obs_m, obs_h], dim=1)
				action_cpu = torch.cat([act_m, act_h], dim=1)
				reward_cpu = torch.cat([rew_m, rew_h], dim=1)
				terminated_cpu = torch.cat([term_m, term_h], dim=1)
				if task_m is not None or task_h is not None:
					if task_m is None or task_h is None:
						raise RuntimeError("Inconsistent multitask presence between main and hot samples.")
					task_cpu = torch.cat([task_m, task_h], dim=0)
				else:
					task_cpu = None
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
		if self._hot_buffer is not None:
			self._hot_buffer.empty()
		return