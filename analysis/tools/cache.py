"""On-disk caching of WANDB payloads."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from .paths import ensure_dir, sweep_cache_dir
from .serialization import make_json_safe


MANIFEST_FILENAME = "manifest.json"
RUNS_FILENAME = "runs.jsonl"


class CacheConsistencyError(RuntimeError):
    """Raised when cached metadata does not match the requested payload."""


def _manifest_path(cache_dir: Path) -> Path:
    return cache_dir / MANIFEST_FILENAME


def _runs_path(cache_dir: Path) -> Path:
    return cache_dir / RUNS_FILENAME


def cache_dir_for(entity: str, project: str, sweep_id: str) -> Path:
    """Return the (existing or potential) cache directory for a sweep."""

    return sweep_cache_dir(entity, project, sweep_id)


def write_payload(
    *,
    entity: str,
    project: str,
    sweep_id: str,
    manifest: Mapping[str, Any],
    runs: Sequence[Mapping[str, Any]],
) -> Path:
    """Persist *manifest* and *runs* in the cache, returning the directory."""

    cache_dir = ensure_dir(cache_dir_for(entity, project, sweep_id))

    manifest_path = _manifest_path(cache_dir)
    runs_path = _runs_path(cache_dir)

    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_ready(manifest), handle, indent=2, sort_keys=True)

    with runs_path.open("w", encoding="utf-8") as handle:
        for record in runs:
            handle.write(json.dumps(_json_ready(record), sort_keys=True))
            handle.write("\n")

    return cache_dir


def read_manifest(entity: str, project: str, sweep_id: str) -> Dict[str, Any]:
    cache_dir = cache_dir_for(entity, project, sweep_id)
    manifest_path = _manifest_path(cache_dir)
    with manifest_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_runs(entity: str, project: str, sweep_id: str) -> List[Dict[str, Any]]:
    cache_dir = cache_dir_for(entity, project, sweep_id)
    runs_path = _runs_path(cache_dir)
    if not runs_path.exists():
        raise FileNotFoundError(f"Cache missing runs payload at {runs_path}")
    payload: List[Dict[str, Any]] = []
    with runs_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload.append(json.loads(line))
    return payload


def has_payload(entity: str, project: str, sweep_id: str) -> bool:
    cache_dir = cache_dir_for(entity, project, sweep_id)
    return _manifest_path(cache_dir).exists() and _runs_path(cache_dir).exists()


def ensure_history_keys(manifest: Mapping[str, Any], expected_keys: Iterable[str]) -> None:
    cached_keys = manifest["history_keys"]
    expected = list(expected_keys)
    if sorted(cached_keys) != sorted(expected):
        raise CacheConsistencyError(
            "Cached history keys do not match requested keys"
        )


def _json_ready(item: Mapping[str, Any] | Sequence[Any] | Any) -> Any:
    return make_json_safe(item)