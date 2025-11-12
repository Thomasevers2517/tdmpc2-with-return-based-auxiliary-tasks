"""
DEPRECATED: common.logging_utils has been removed.

All code should import loggers via `from common.logger import get_logger` and use
the `Logger` class for metrics/W&B logging. This module intentionally raises on
import to surface any lingering, unintended dependencies.
"""

raise ImportError(
    "common.logging_utils is deprecated; use common.logger.get_logger instead."
)
