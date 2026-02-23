from trainer.base import Trainer
from common.logger import get_logger

log = get_logger(__name__)


class OfflineTrainer(Trainer):
	"""Placeholder for offline training (multitask support removed)."""

	def train(self):
		raise NotImplementedError(
			"Offline training required multitask support which has been removed."
		)
