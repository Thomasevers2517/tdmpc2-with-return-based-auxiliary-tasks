import contextlib
import os
from typing import Optional

import torch


def _nvtx_enabled(cfg=None) -> bool:
    """Return True if NVTX profiling should emit ranges.

    Priority order:
      1. Explicit cfg.nvtx_profiler flag (if cfg provided)
      2. Environment variable NVTX_PROFILER=1
    """
    if cfg is not None and getattr(cfg, 'nvtx_profiler', False):
        return True
    if os.getenv('NVTX_PROFILER', '0') in {'1', 'true', 'True'}:
        return True
    return False


@contextlib.contextmanager
def nvtx_range(msg: str, cfg=None, color: Optional[int] = None):  # color is hint only
    """Context manager emitting an NVTX range if enabled.

    Safe to use under torch.compile; NVTX ranges will be emitted at runtime
    (graph capture will include markers if placed outside captured region).
    """
    if _nvtx_enabled(cfg):
        torch.cuda.nvtx.range_push(msg)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()
    else:
        yield


def maybe_range(msg: str, cfg=None):
    """Shorthand returning a context manager (so: with maybe_range(...): )."""
    return nvtx_range(msg, cfg=cfg)
