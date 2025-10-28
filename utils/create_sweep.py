#!/usr/bin/env python3
"""
Create a Weights & Biases sweep from a sweep directory.

Usage:
    utils/create_sweep.py /path/to/sweep_folder [--wandb-entity NAME] [--wandb-project NAME]

The script expects a file named `sweep.yaml` in the provided folder.
On success it writes the sweep id string to `id.txt` inside the same folder.

This intentionally fails fast if files are missing or if wandb/pyyaml are not available.
"""
import sys
import os
from pathlib import Path


def parse_args(argv):
    if len(argv) < 2:
        fatal("Usage: create_sweep.py /path/to/sweep_folder [--wandb-entity NAME] [--wandb-project NAME]")

    sweep_dir = None
    entity = None
    project = None

    # First positional must be sweep_dir
    sweep_dir = Path(argv[1]).expanduser().resolve()
    i = 2
    while i < len(argv):
        arg = argv[i]
        if arg == "--wandb-entity":
            i += 1
            if i >= len(argv):
                fatal("--wandb-entity requires a value")
            entity = argv[i]
        elif arg == "--wandb-project":
            i += 1
            if i >= len(argv):
                fatal("--wandb-project requires a value")
            project = argv[i]
        else:
            fatal(f"Unknown argument: {arg}")
        i += 1

    return sweep_dir, entity, project


def fatal(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(2)


def main(argv):
    sweep_dir, cli_entity, cli_project = parse_args(argv)
    if not sweep_dir.is_dir():
        fatal(f"Provided path is not a directory: {sweep_dir}")

    cfg_file = sweep_dir / "sweep.yaml"
    if not cfg_file.is_file():
        fatal(f"Missing sweep.yaml in {sweep_dir}")

    try:
        import yaml
    except Exception as e:
        fatal("PyYAML is required (install with `pip install pyyaml`).")

    try:
        import wandb
    except Exception:
        fatal("wandb package is required (install with `pip install wandb`).")

    # Load yaml
    try:
        with cfg_file.open("r") as f:
            sweep_config = yaml.safe_load(f)
    except Exception as e:
        fatal(f"Failed to read or parse {cfg_file}: {e}")

    # Determine entity/project: prefer CLI > ENV; project must be known to save project.txt
    env_entity = os.environ.get("WANDB_ENTITY")
    env_project = os.environ.get("WANDB_PROJECT")
    entity = cli_entity or env_entity
    project = cli_project or env_project
    if not project:
        fatal("WANDB project is required. Provide --wandb-project or set WANDB_PROJECT in environment.")

    # Create sweep via wandb API; this will error if credentials are not configured
    try:
        sweep_id = wandb.sweep(sweep_config, entity=entity, project=project)
    except Exception as e:
        fatal(f"Failed to create sweep via wandb: {e}")

    if not sweep_id:
        fatal("wandb.sweep returned an empty sweep id")

    id_file = sweep_dir / "id.txt"
    proj_file = sweep_dir / "project.txt"
    try:
        with id_file.open("w") as f:
            f.write(str(sweep_id).strip() + os.linesep)
        with proj_file.open("w") as f:
            f.write(str(project).strip() + os.linesep)
    except Exception as e:
        fatal(f"Failed to write sweep metadata: {e}")

    print(f"Created sweep: {sweep_id}")
    print(f"Wrote sweep id to: {id_file}")
    print(f"Wrote project to: {proj_file}")


if __name__ == "__main__":
    main(sys.argv)
