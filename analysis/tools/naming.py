"""Naming helpers for consistent task references."""

from __future__ import annotations


# DMC action repeat is hardcoded to 2 in DMControlWrapper.step().
# W&B logs decision steps; env steps = decision_steps * DMC_ACTION_REPEAT.
DMC_ACTION_REPEAT = 2

# Known DeepMind Control Suite domains (from dm_control.suite).
_DMC_DOMAINS = frozenset({
    "acrobot", "ball_in_cup", "cartpole", "cheetah", "cup", "dog",
    "finger", "fish", "hopper", "humanoid", "pendulum", "point_mass",
    "pointmass", "quadruped", "reacher", "swimmer", "walker",
})


def is_dmcontrol_task(task: str) -> bool:
    """Return True if *task* belongs to DeepMind Control Suite.

    Mirrors the domain extraction in ``tdmpc2/envs/dmcontrol.py``::

        domain, task = cfg.task.replace('-', '_').split('_', 1)

    Humanoid-Bench tasks contain ``h1`` (e.g. ``humanoid_h1-walk-v0``)
    and are excluded.
    """
    if "h1" in task:
        return False
    domain = task.replace("-", "_").split("_", 1)[0]
    return domain in _DMC_DOMAINS


def action_repeat_for_task(task: str) -> int:
    """Return the environment action repeat for *task*.

    DMC tasks use action_repeat=2 (hardcoded in ``DMControlWrapper.step``).
    All other environments (Humanoid-Bench, MetaWorld, MyoSuite, MuJoCo)
    use action_repeat=1.
    """
    return DMC_ACTION_REPEAT if is_dmcontrol_task(task) else 1


def wandb_task_to_baseline(task: str) -> str:
    """Convert sweep task names (snake_case) into baseline CSV names (kebab-case)."""

    stripped = task.strip()
    if not stripped:
        raise ValueError("Task name cannot be empty")
    return stripped.replace("_", "-")