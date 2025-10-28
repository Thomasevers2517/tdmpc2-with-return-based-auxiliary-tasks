#!/usr/bin/env bash
set -euo pipefail

# Create (or recreate) W&B sweeps from sweep.yaml files under a folder.
# - Finds leaf sweep folders (directories containing sweep.yaml that do NOT have nested sweep.yaml below them)
# - Runs `wandb sweep` for each, captures the full sweep path (entity/project/id), writes to id.txt (overwrites)
#
# Usage:
#   utils/sweeps/create_or_update_sweeps.sh sweep_list/midterm_sweep
#   utils/sweeps/create_or_update_sweeps.sh sweep_list/midterm_sweep/1aux_value

ROOT_DIR=${1:-sweep_list}

if ! command -v wandb >/dev/null 2>&1; then
  echo "Error: wandb CLI not found in PATH. Activate the tdmpc2 conda env first." >&2
  exit 1
fi

# Find all sweep.yaml files
mapfile -t YAML_FILES < <(find "$ROOT_DIR" -type f -name sweep.yaml | sort)

for YAML in "${YAML_FILES[@]}"; do
  DIR=$(dirname "$YAML")
  # Skip non-leaf folders (if any nested sweep.yaml exists inside this dir)
  if find "$DIR" -mindepth 2 -type f -name sweep.yaml | grep -q .; then
    echo "Skipping non-leaf sweep folder: $DIR"
    continue
  fi

  echo "Creating sweep from: $YAML"
  # Parse entity/project/id from the "View sweep at:" URL line
  OUT=$(wandb sweep "$YAML" 2>&1 | tee /dev/stderr || true)
  URL=$(echo "$OUT" | grep -Eo 'https://wandb\.ai/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/sweeps/[A-Za-z0-9_-]+' | tail -n1 || true)
  if [[ -z "$URL" ]]; then
    echo "Failed to create sweep or parse URL for $YAML" >&2
    exit 1
  fi
  # Extract entity/project/id triple
  SWEEP_PATH=$(echo "$URL" | sed -E 's@https://wandb\.ai/([^/]+)/([^/]+)/sweeps/([^/]+)@\1/\2/\3@')
  echo "$SWEEP_PATH" > "$DIR/id.txt"
  echo "Wrote sweep id to: $DIR/id.txt -> $SWEEP_PATH"
done

echo "Done."
