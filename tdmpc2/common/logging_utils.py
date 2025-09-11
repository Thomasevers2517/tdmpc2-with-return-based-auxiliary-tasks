import logging
import os
import sys
from typing import Optional, Union


def configure_logging(level: Optional[Union[str, int]] = None) -> logging.Logger:
    """Configure root logging once.

    - Uses stdout stream handler.
    - Default level from $LOGLEVEL or INFO.
    - No-op if handlers already exist.
    """
    root = logging.getLogger()
    if level is None:
        level = os.getenv("LOGLEVEL", "INFO").upper()
    if not root.handlers:
        root.setLevel(level)  # type: ignore[arg-type]
        handler = logging.StreamHandler(stream=sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s %(levelname).1s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        root.addHandler(handler)
    else:
        if level is not None:
            root.setLevel(level)  # type: ignore[arg-type]
    return root


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a logger, ensuring logging is configured."""
    configure_logging()
    return logging.getLogger(name if name else __name__)
