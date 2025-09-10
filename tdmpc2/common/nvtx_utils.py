# tdmpc2/common/nvtx_utils.py
import contextlib, torch
from contextlib import nullcontext
from torch.autograd.profiler import record_function


def _in_dynamo():
    # works across 2.x
    return getattr(torch, "compiler", None) and torch.compiler.is_compiling() \
        or getattr(torch._dynamo, "is_compiling", lambda: False)()

@contextlib.contextmanager
def _nvtx_range(msg: str):
    torch.cuda.nvtx.range_push(msg)
    try:    yield
    finally: torch.cuda.nvtx.range_pop()

def maybe_range(msg: str, cfg=None):
    # disable ranges if NVTX off or we’re under Dynamo/Inductor
    if (cfg is not None and not getattr(cfg, "nvtx_profiler", True)) or _in_dynamo():
        return nullcontext()
    if torch.cuda.is_available():
        return _nvtx_range(msg)
    return record_function(msg)
