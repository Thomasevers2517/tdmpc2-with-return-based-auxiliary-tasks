#!/usr/bin/env python3
"""
Parse SimbaV2 and EZ2 result files into TDMPC2-style CSV format.

TDMPC2 format: One CSV per task with columns: step,reward,seed
"""

import csv
import json
import os
from collections import defaultdict
from pathlib import Path


def parse_simbav2_csv(input_path: str, output_dir: str) -> None:
    """
    Parse SimbaV2 CSV format into TDMPC2-style per-task CSVs.

    SimbaV2 format columns: (index),exp_name,env_name,seed,metric,env_step,value
    We only keep metric='avg_return' rows.

    Args:
        input_path: Path to SimbaV2 CSV file.
        output_dir: Directory to write per-task CSV files.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Collect data by task: {task_name: [(step, reward, seed), ...]}
    task_data: dict[str, list[tuple[int, float, int]]] = defaultdict(list)

    with open(input_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metric = row["metric"]
            if metric != "avg_return":
                continue

            env_name = row["env_name"]
            seed = int(row["seed"])
            step = int(float(row["env_step"]))
            reward = float(row["value"])

            task_data[env_name].append((step, reward, seed))

    # Write per-task CSVs
    for task_name, data in task_data.items():
        # Convert task name to lowercase with underscores (match TDMPC2 style)
        # e.g., "Humanoid_v4" -> "humanoid-v4", "walker_run" -> "walker-run"
        output_name = task_name.lower().replace("_", "-")
        output_path = os.path.join(output_dir, f"{output_name}.csv")

        # Sort by seed then step for consistency
        data.sort(key=lambda x: (x[2], x[0]))

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "reward", "seed"])
            for step, reward, seed in data:
                writer.writerow([step, reward, seed])

        print(f"  Written: {output_path} ({len(data)} rows)")


def parse_ez2_json(input_path: str, output_dir: str) -> None:
    """
    Parse EZ2 JSON format into TDMPC2-style per-task CSVs.

    EZ2 format: JSON array of objects with keys: task, seed, xs, ys, method
    Each entry may have the same task/seed (multiple runs to average), so we
    treat each as a separate seed.

    Args:
        input_path: Path to EZ2 JSON file.
        output_dir: Directory to write per-task CSV files.
    """
    os.makedirs(output_dir, exist_ok=True)

    with open(input_path, "r") as f:
        data = json.load(f)

    # Collect data by task
    # Since multiple entries can have same task+seed, we assign unique seed IDs
    # {task_name: [(step, reward, seed_id), ...]}
    task_data: dict[str, list[tuple[int, float, int]]] = defaultdict(list)
    task_seed_counter: dict[str, int] = defaultdict(int)

    for entry in data:
        task = entry["task"]
        xs = entry["xs"]  # steps
        ys = entry["ys"]  # rewards

        # Assign a unique seed ID for this run
        seed_id = task_seed_counter[task]
        task_seed_counter[task] += 1

        for step, reward in zip(xs, ys):
            task_data[task].append((int(step), float(reward), seed_id))

    # Write per-task CSVs
    for task_name, data_list in task_data.items():
        # Convert task name: "dmc_pendulum_swingup" -> "pendulum-swingup"
        output_name = task_name
        if output_name.startswith("dmc_"):
            output_name = output_name[4:]  # Remove "dmc_" prefix
        output_name = output_name.replace("_", "-")
        output_path = os.path.join(output_dir, f"{output_name}.csv")

        # Sort by seed then step for consistency
        data_list.sort(key=lambda x: (x[2], x[0]))

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "reward", "seed"])
            for step, reward, seed in data_list:
                writer.writerow([step, reward, seed])

        print(f"  Written: {output_path} ({len(data_list)} rows)")


def main():
    """Parse SimbaV2 and EZ2 results into TDMPC2 format."""
    base_dir = Path(__file__).parent.parent.parent / "results"

    # Parse SimbaV2
    simbav2_input = base_dir / "simbav2" / "simbaV2_utd8.csv"
    simbav2_output = base_dir / "simbav2_parsed"
    if simbav2_input.exists():
        print(f"Parsing SimbaV2: {simbav2_input}")
        parse_simbav2_csv(str(simbav2_input), str(simbav2_output))
        print()

    # Parse EZ2
    ez2_input = base_dir / "ez2" / "dmcproprio_ezv2.json"
    ez2_output = base_dir / "ez2_parsed"
    if ez2_input.exists():
        print(f"Parsing EZ2: {ez2_input}")
        parse_ez2_json(str(ez2_input), str(ez2_output))
        print()

    print("Done!")


if __name__ == "__main__":
    main()
