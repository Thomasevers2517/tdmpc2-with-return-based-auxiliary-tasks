"""Centralised path helpers for the analysis suite."""

from __future__ import annotations

import re
from pathlib import Path


ANALYSIS_ROOT: Path = Path(__file__).resolve().parent.parent
RESULTS_ROOT: Path = ANALYSIS_ROOT / "results"
RUN_CACHE_ROOT: Path = ANALYSIS_ROOT / "run_cache"

# -------------------------------------------------------------------------
# Baseline Results Paths (in project root /results/)
# -------------------------------------------------------------------------
PROJECT_ROOT: Path = ANALYSIS_ROOT.parent
BASELINE_TDMPC2: Path = (PROJECT_ROOT / "results" / "tdmpc2").resolve()
BASELINE_TDMPC2_PIXELS: Path = (PROJECT_ROOT / "results" / "tdmpc2-pixels").resolve()
BASELINE_DREAMERV3: Path = (PROJECT_ROOT / "results" / "dreamerv3").resolve()
BASELINE_SAC: Path = (PROJECT_ROOT / "results" / "sac").resolve()


_TOKEN_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_token(token: str) -> str:
    """Return a filesystem-safe version of *token*.

    Raises:
        ValueError: when the sanitised value would be empty.
    """

    stripped = token.strip()
    if not stripped:
        raise ValueError("Cannot sanitise an empty token")

    collapsed = _TOKEN_PATTERN.sub("_", stripped)
    candidate = collapsed.strip("._-")
    if not candidate:
        raise ValueError(f"Token '{token}' sanitises to an empty string")
    return candidate


def ensure_dir(path: Path) -> Path:
    """Create *path* (and parents) when missing and return it."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def notebook_results_dir(notebook_stem: str) -> Path:
    """Return the results directory for *notebook_stem*, creating it."""

    safe_stem = sanitize_token(notebook_stem)
    return ensure_dir(RESULTS_ROOT / safe_stem)


def sweep_cache_dir(entity: str, project: str, sweep_id: str) -> Path:
    """Return the cache directory for a sweep identifier."""

    safe_parts = [sanitize_token(part) for part in (entity, project, sweep_id)]
    return RUN_CACHE_ROOT / "__".join(safe_parts)