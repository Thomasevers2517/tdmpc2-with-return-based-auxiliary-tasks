"""Rename variant labels in ablation_analysis.ipynb via safe JSON manipulation."""
import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent.parent / "analysis" / "notebooks" / "general" / "ablation_analysis.ipynb"

# Read notebook
with open(NB_PATH, "r") as f:
    nb = json.load(f)

# Replacements to apply in source cells and markdown cells
LABEL_REPLACEMENTS = {
    # Code cell labels
    'label="TD-MPC2"':                 'label="Ours (baseline)"',
    'label="+ Buffer Relabeling"':     'label="Ours + Buffer Relabeling"',
    'label="+ Jacobian Correction"':   'label="Ours + Jacobian Correction"',
    'label="+ Ensemble Dynamics"':     'label="Ours + Ensemble Dynamics"',
    'label="+ Local TD Bootstrap"':    'label="Ours + Local TD Bootstrap"',
    'label="+ Local TD + Optimistic"': 'label="Ours + Local TD + Optimistic"',
    'label="All (no exploration)"':    'label="Ours all (no exploration)"',
    'label="All (no Jacobian)"':       'label="Ours all (no Jacobian)"',
    'label="All (no ent. scaling)"':   'label="Ours all (no ent. scaling)"',
    'label="All"':                     'label="Ours (all)"',
}

# Also rename in markdown descriptions (cell 13 and others)
MD_REPLACEMENTS = {
    "TD-MPC2: ['oa3grakj":           "Ours (baseline): ['oa3grakj",
    "Vanilla TD-MPC2 (baseline)":    "Vanilla baseline",
    "compare it against **TD-MPC2**": "compare it against **Ours (baseline)**",
    "- TD-MPC2: solid gray line":    "- Ours (baseline): solid gray line",
    "+ Buffer Relabeling: [":        "Ours + Buffer Relabeling: [",
    "+ Jacobian Correction: [":      "Ours + Jacobian Correction: [",
    "+ Ensemble Dynamics: [":        "Ours + Ensemble Dynamics: [",
    "+ Local TD Bootstrap: [":       "Ours + Local TD Bootstrap: [",
    "+ Local TD + Optimistic: [":    "Ours + Local TD + Optimistic: [",
    "All (no exploration): [":       "Ours all (no exploration): [",
    "All (no Jacobian): [":          "Ours all (no Jacobian): [",
    "All (no ent. scaling): [":      "Ours all (no ent. scaling): [",
    # Must come last to avoid partial matching
    "  All: ['41btp9ha":             "  Ours (all): ['41btp9ha",
}

# Also fix the description field in the baseline Variant
DESC_REPLACEMENTS = {
    'description="Vanilla TD-MPC2 (baseline)"': 'description="Vanilla baseline"',
}

ALL_REPLACEMENTS = {**LABEL_REPLACEMENTS, **MD_REPLACEMENTS, **DESC_REPLACEMENTS}

total_changes = 0
for cell in nb["cells"]:
    new_source = []
    for line in cell["source"]:
        original = line
        for old, new in ALL_REPLACEMENTS.items():
            if old in line:
                line = line.replace(old, new)
        if line != original:
            total_changes += 1
        new_source.append(line)
    cell["source"] = new_source

print(f"Applied changes to {total_changes} lines")

# Write back
with open(NB_PATH, "w") as f:
    json.dump(nb, f, indent=1)
    f.write("\n")

print(f"Saved {NB_PATH}")
