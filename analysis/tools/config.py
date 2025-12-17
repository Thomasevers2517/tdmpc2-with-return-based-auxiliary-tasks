"""Centralized configuration for W&B and analysis defaults.

This module provides default values for W&B entity/project and other
configuration that would otherwise be hardcoded across notebooks.
"""

from __future__ import annotations

# -------------------------------------------------------------------------
# Weights & Biases Defaults
# -------------------------------------------------------------------------
# These can be overridden in individual notebooks when needed.

WANDB_ENTITY: str = "thomasevers9"
"""Default W&B entity (username or team name)."""

WANDB_PROJECT: str = "tdmpc2-tdmpc2"
"""Default W&B project name."""


# -------------------------------------------------------------------------
# Analysis Defaults
# -------------------------------------------------------------------------

DEFAULT_STEP_KEYS: tuple = (
    "_step",
    "step",
    "total_env_steps",
    "global_step",
    "eval/step",
)
"""Common step key names to try when parsing W&B history."""

DEFAULT_METRIC_KEY: str = "eval/episode_reward"
"""Default metric for evaluation reward."""

DEFAULT_TRAIN_METRIC_KEY: str = "train/episode_reward"
"""Default metric for training reward."""


__all__ = [
    "WANDB_ENTITY",
    "WANDB_PROJECT",
    "DEFAULT_STEP_KEYS",
    "DEFAULT_METRIC_KEY",
    "DEFAULT_TRAIN_METRIC_KEY",
]
