"""Script to update notebooks to use 95% CI bounds instead of raw std."""
import json
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent


def update_cell_source(source_lines: list[str]) -> tuple[list[str], bool]:
    """Update a cell's source to use CI bounds.
    
    Returns:
        (new_source_lines, was_modified)
    """
    source = "".join(source_lines)
    
    # Check if this cell contains the _add_method_traces function or plotting code
    if "y_fill" not in source or "y_std" not in source:
        return source_lines, False
    
    # Check if already updated
    if "y_ci95" in source:
        return source_lines, False
    
    # Check if it has the pattern we want to replace
    old_y_fill = "y_fill = np.concatenate([y_mean + y_std, (y_mean - y_std)[::-1]])"
    if old_y_fill not in source:
        return source_lines, False
    
    new_lines = []
    modified = False
    i = 0
    
    while i < len(source_lines):
        line = source_lines[i]
        
        # Find the y_std line and insert CI computation after it
        if 'y_std = np.nan_to_num' in line and 'std_reward' in line:
            new_lines.append(line)
            
            # Get indentation
            indent = len(line) - len(line.lstrip())
            indent_str = " " * indent
            
            # Check if there's a blank line after
            if i + 1 < len(source_lines) and source_lines[i + 1].strip() == "":
                new_lines.append(source_lines[i + 1])
                i += 1
            
            # Check if there's a comment "# Shaded std region"
            if i + 1 < len(source_lines) and "# Shaded" in source_lines[i + 1]:
                # Skip old comment, we'll add new one
                i += 1
                
            # Add CI computation
            new_lines.append(f"{indent_str}# 95% CI bounds: std / sqrt(n) * 1.96\n")
            new_lines.append(f"{indent_str}n_samples = ms['n_tasks'].values if 'n_tasks' in ms.columns else ms['n_seeds'].values\n")
            new_lines.append(f"{indent_str}y_ci95 = 1.96 * y_std / np.sqrt(np.maximum(n_samples, 1))\n")
            new_lines.append(f"\n")
            modified = True
            i += 1
            continue
        
        # Replace y_fill computation
        if old_y_fill in line:
            new_line = line.replace(
                old_y_fill,
                "y_fill = np.concatenate([y_mean + y_ci95, (y_mean - y_ci95)[::-1]])"
            )
            new_lines.append(new_line)
            modified = True
        else:
            new_lines.append(line)
        
        i += 1
    
    return new_lines, modified


def update_notebook(nb_path: Path) -> bool:
    """Update a notebook to use CI bounds.
    
    Returns True if notebook was modified.
    """
    with open(nb_path) as f:
        nb = json.load(f)
    
    modified = False
    for cell_idx, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            new_source, cell_modified = update_cell_source(cell["source"])
            if cell_modified:
                cell["source"] = new_source
                modified = True
                print(f"  Modified cell {cell_idx}")
    
    if modified:
        with open(nb_path, "w") as f:
            json.dump(nb, f, indent=1)
        print(f"Updated: {nb_path}")
    else:
        print(f"No changes needed: {nb_path}")
    
    return modified


def main():
    notebooks = [
        REPO_ROOT / "analysis/notebooks/general/compare_to_baselines.ipynb",
        REPO_ROOT / "analysis/notebooks/general/ablation_analysis.ipynb",
        REPO_ROOT / "analysis/notebooks/11_duplicate_dmcontrol_2enc_benchmark.ipynb",
    ]
    
    for nb_path in notebooks:
        if nb_path.exists():
            print(f"\nProcessing: {nb_path.name}")
            update_notebook(nb_path)
        else:
            print(f"Not found: {nb_path}")


if __name__ == "__main__":
    main()

