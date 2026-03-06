#!/usr/bin/env python3
"""Parse BMPC results from JSON (DM-Control) and ZIP (Humanoid-Bench) into CSV format.

Output format matches existing baselines (e.g., simbav2_parsed):
    step,reward,seed,ucb,lcb
    0,100.5,0,120.3,80.7
    100000,200.3,0,210.1,190.5
    ...

DM-Control: BMPC reports aggregate stats (mean, ucbs, lcbs) rather than per-seed
trajectories.  We store the mean directly as reward (single seed=0) and keep
ucb/lcb as additional columns for honest confidence-band plotting.

Humanoid-Bench: real per-seed runs are available in the ZIP.
"""

import json
import zipfile
import pandas as pd
from pathlib import Path





def parse_dmcontrol_json(json_path: Path, output_dir: Path) -> None:
    """Parse BMPC DM-Control JSON and write per-task CSVs.
    
    Args:
        json_path: Path to dmcontrol_all_BMPC.json.
        output_dir: Directory to write CSV files.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Get BMPC results (JSON has "BMPC" and "tdmpc2" keys)
    bmpc_data = data["BMPC"]
    
    for task_name, task_data in bmpc_data.items():
        values = task_data["values"]
        steps = task_data["steps"]
        ucbs = task_data["ucbs"]
        lcbs = task_data["lcbs"]
        
        if not values or not steps:
            print(f"  Skipping {task_name}: no data")
            continue
        
        # Store the mean directly — no pseudo-seed fabrication.
        # ucb/lcb are kept as extra columns for confidence-band plotting.
        rows = [
            {
                "step": int(step),
                "reward": mean,
                "seed": 0,
                "ucb": ucb,
                "lcb": lcb,
            }
            for step, mean, ucb, lcb in zip(steps, values, ucbs, lcbs)
        ]
        
        df = pd.DataFrame(rows)
        csv_path = output_dir / f"{task_name}.csv"
        df.to_csv(csv_path, index=False)
        print(f"  Wrote {csv_path.name}: {len(df)} rows, {len(steps)} steps")


def parse_humanoidbench_zip(zip_path: Path, output_dir: Path) -> None:
    """Parse BMPC Humanoid-Bench ZIP and write per-task CSVs.
    
    ZIP structure: humanoidbench/bmpc/<task>/<run_id>/eval.csv
    Each eval.csv has columns: step, episode_reward
    
    Args:
        zip_path: Path to humanoidbench_BMPC.zip.
        output_dir: Directory to write CSV files.
    """
    with zipfile.ZipFile(zip_path, 'r') as zf:
        # Find all eval.csv files
        eval_files = [f for f in zf.namelist() if f.endswith('eval.csv')]
        
        # Group by task
        task_files = {}  # task_name -> list of eval.csv paths
        for fpath in eval_files:
            # Extract task name from path: humanoidbench/bmpc/<task>/<run_id>/eval.csv
            parts = fpath.split('/')
            if len(parts) >= 4:
                task_name = parts[2]  # e.g., "humanoid_h1hand-balance_hard-v0"
                if task_name not in task_files:
                    task_files[task_name] = []
                task_files[task_name].append(fpath)
        
        print(f"Found {len(task_files)} tasks in ZIP")
        
        for task_name, files in task_files.items():
            all_rows = []
            
            for seed_idx, fpath in enumerate(sorted(files)):
                # Read eval.csv from ZIP
                with zf.open(fpath) as csvfile:
                    df = pd.read_csv(csvfile)
                
                # Rename columns to match expected format
                df = df.rename(columns={"episode_reward": "reward"})
                df["seed"] = seed_idx
                df["step"] = df["step"].astype(int)
                
                all_rows.append(df[["step", "reward", "seed"]])
            
            if not all_rows:
                continue
            
            combined = pd.concat(all_rows, ignore_index=True)
            
            # Convert task name to match SimbaV2 convention:
            # humanoid_h1hand-balance_hard-v0 -> h1hand-balance_hard-v0
            # Also try: humanoid_h1-balance_hard-v0 -> h1-balance_hard-v0
            csv_name = task_name
            if csv_name.startswith("humanoid_"):
                csv_name = csv_name[len("humanoid_"):]
            
            csv_path = output_dir / f"{csv_name}.csv"
            combined.to_csv(csv_path, index=False)
            print(f"  Wrote {csv_path.name}: {len(combined)} rows, {len(files)} seeds")


def main():
    """Parse all BMPC results."""
    script_dir = Path(__file__).parent
    bmpc_dir = script_dir.parent / "BMPC"
    
    # DM-Control JSON
    dmcontrol_json = bmpc_dir / "dmcontrol_all_BMPC.json"
    dmcontrol_output = script_dir / "dmcontrol"
    
    if dmcontrol_json.exists():
        print(f"\nParsing DM-Control from {dmcontrol_json}")
        parse_dmcontrol_json(dmcontrol_json, dmcontrol_output)
    else:
        print(f"Warning: {dmcontrol_json} not found")
    
    # Humanoid-Bench ZIP
    humanoidbench_zip = bmpc_dir / "humanoidbench_BMPC.zip"
    humanoidbench_output = script_dir / "humanoidbench"
    
    if humanoidbench_zip.exists():
        print(f"\nParsing Humanoid-Bench from {humanoidbench_zip}")
        parse_humanoidbench_zip(humanoidbench_zip, humanoidbench_output)
    else:
        print(f"Warning: {humanoidbench_zip} not found")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
