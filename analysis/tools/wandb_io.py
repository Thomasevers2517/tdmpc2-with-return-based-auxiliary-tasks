"""WANDB accessors with caching."""

from __future__ import annotations

import datetime as dt
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

from .cache import (
    CacheConsistencyError,
    ensure_history_keys,
    has_payload,
    read_manifest,
    read_runs,
    write_payload,
)
from .serialization import make_json_safe


import os
from pathlib import Path
from .paths import ANALYSIS_ROOT

SCAN_HISTORY_PAGE_SIZE = 20000000

def _load_wandb_key():
    """Load W&B API key from analysis/wandb-key.txt if not already set."""
    if "WANDB_API_KEY" in os.environ:
        return

    key_path = ANALYSIS_ROOT / "wandb-key.txt"
    if key_path.exists():
        try:
            key = key_path.read_text().strip()
            if key:
                os.environ["WANDB_API_KEY"] = key
        except Exception:
            pass

def fetch_sweep_runs(
    *,
    entity: str,
    project: str,
    sweep_id: str,
    history_keys: Sequence[str],
    use_cache: bool,
    force_refresh: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], str]:
    """Return sweep runs, manifest metadata, and the data source."""
    
    _load_wandb_key()

    _validate_history_keys(history_keys)
    canonical_keys = tuple(dict.fromkeys(history_keys))

    if use_cache and not force_refresh and has_payload(entity, project, sweep_id):
        try:
            manifest = read_manifest(entity, project, sweep_id)
            ensure_history_keys(manifest, canonical_keys)
            runs = read_runs(entity, project, sweep_id)
            return runs, manifest, "cache"
        except (CacheConsistencyError, FileNotFoundError):
            pass

    runs, manifest = _pull_from_wandb(
        entity=entity,
        project=project,
        sweep_id=sweep_id,
        history_keys=canonical_keys,
    )
    if use_cache:
        write_payload(
            entity=entity,
            project=project,
            sweep_id=sweep_id,
            manifest=manifest,
            runs=runs,
        )
    return runs, manifest, "remote"


def _pull_from_wandb(
    *, entity: str, project: str, sweep_id: str, history_keys: Sequence[str]
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    wandb = _import_wandb()
    api = wandb.Api()
    sweep = api.sweep(f"{entity}/{project}/{sweep_id}")

    sweep_runs = list(sweep.runs)  # type: ignore[attr-defined]
    iterator = _progress_iterator(
        sweep_runs,
        description=f"Downloading sweep {sweep_id}",
    )

    runs_payload: List[Dict[str, Any]] = []
    for run in iterator:
        history_rows = list(_iter_history_rows(run, history_keys))
        payload = {
            "run_id": run.id,
            "name": run.name,
            "state": run.state,
            "config": _to_json_dict(run.config),
            "summary": _to_json_dict(run.summary._json_dict),
            "history": {
                "keys": list(history_keys),
                "rows": history_rows,
            },
        }
        runs_payload.append(payload)

    manifest = {
        "entity": entity,
        "project": project,
        "sweep_id": sweep_id,
        "history_keys": list(history_keys),
        "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_count": len(runs_payload),
        "scan_page_size": SCAN_HISTORY_PAGE_SIZE,
    }
    return runs_payload, manifest


def _iter_history_rows(run: Any, history_keys: Sequence[str]) -> Iterable[Dict[str, Any]]:
    requested = list(history_keys)
    for row in run.scan_history(page_size=SCAN_HISTORY_PAGE_SIZE):  # type: ignore[attr-defined]
        filtered: Dict[str, Any] = {}
        for key in requested:
            if key in row:
                filtered[key] = _scalarise(row[key])
        if filtered:
            yield filtered


def _scalarise(value: Any) -> Any:
    try:
        return make_json_safe(value)
    except TypeError as exc:  # re-raise with more context
        raise TypeError(f"Unsupported history value type: {type(value)!r}") from exc


def _to_json_dict(mapping: Mapping[str, Any]) -> Dict[str, Any]:
    json_dict: Dict[str, Any] = {}
    for key, value in mapping.items():
        json_dict[key] = make_json_safe(value)
    return json_dict


def _import_wandb():
    try:
        import wandb  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - environment specific.
        raise RuntimeError(
            "wandb is required to fetch sweep data. Install it in the tdmpc2 env."
        ) from exc
    return wandb


def _validate_history_keys(history_keys: Sequence[str]) -> None:
    if not history_keys:
        raise ValueError("history_keys must contain at least one metric")
    duplicates = _find_duplicates(history_keys)
    if duplicates:
        raise ValueError(f"history_keys contains duplicates: {sorted(duplicates)}")


def _find_duplicates(keys: Sequence[str]) -> List[str]:
    seen: MutableMapping[str, int] = {}
    duplicates: List[str] = []
    for key in keys:
        count = seen.get(key, 0) + 1
        seen[key] = count
        if count == 2:
            duplicates.append(key)
    return duplicates


def _progress_iterator(iterable: Sequence[Any], *, description: str):
    tqdm = _import_tqdm()
    return tqdm(iterable, desc=description, total=len(iterable))


def _import_tqdm():
    try:
        from tqdm import tqdm  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - environment specific.
        raise RuntimeError(
            "tqdm is required to display download progress. Install it in the tdmpc2 env."
        ) from exc
    return tqdm