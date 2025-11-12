from common.logger import get_logger


class Trainer:
	"""Base trainer class for TD-MPC2."""

	def __init__(self, cfg, env, agent, buffer, logger):
		self.cfg = cfg
		self.env = env
		self.agent = agent
		self.buffer = buffer
		self.logger = logger
		# Ensure agent has access to the logger (for planner metrics, etc.)
		if getattr(self.agent, 'logger', None) is None:
			self.agent.logger = logger
		# Set logger level according to cfg.debug for this module and print architecture
		get_logger(__name__, cfg=self.cfg).info('Architecture: %s', self.agent.model)

	def eval(self):
		"""Evaluate a TD-MPC2 agent."""
		raise NotImplementedError

	def train(self):
		"""Train a TD-MPC2 agent."""
		raise NotImplementedError
