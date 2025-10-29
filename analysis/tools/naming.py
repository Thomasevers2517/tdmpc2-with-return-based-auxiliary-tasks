"""Naming helpers for consistent task references."""

from __future__ import annotations


def wandb_task_to_baseline(task: str) -> str:
    """Convert sweep task names (snake_case) into baseline CSV names (kebab-case)."""

    stripped = task.strip()
    if not stripped:
        raise ValueError("Task name cannot be empty")
    return stripped.replace("_", "-")