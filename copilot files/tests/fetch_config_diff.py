"""Fetch true W&B configs for sweep 17 and sweep 18 baseline, compute full diff."""
import wandb

api = wandb.Api()

# Sweep 17: known run gwd2ifit (worked)
r17 = api.run('thomasevers9/tdmpc2-tdmpc2/gwd2ifit')
c17 = {k: v['value'] if isinstance(v, dict) and 'value' in v else v for k, v in r17.config.items()}

# Sweep 18 baseline: known run 2e8bnmhb (crashed)
r18 = api.run('thomasevers9/tdmpc2-tdmpc2/2e8bnmhb')
c18 = {k: v['value'] if isinstance(v, dict) and 'value' in v else v for k, v in r18.config.items()}

# Find ALL differences
all_keys = sorted(set(list(c17.keys()) + list(c18.keys())))
diffs = []
for k in all_keys:
    v17 = c17.get(k, '<MISSING>')
    v18 = c18.get(k, '<MISSING>')
    if str(v17) != str(v18):
        diffs.append((k, v17, v18))

header = f"{'Parameter':<35} {'Sweep17 (works)':<25} {'Sweep18 (crashes)'}"
print("=== ALL PARAMETER DIFFERENCES (from W&B API) ===")
print(header)
print("-" * 85)
for k, v17, v18 in diffs:
    print(f"{k:<35} {str(v17):<25} {str(v18)}")
print(f"\nTotal differences: {len(diffs)}")
print(f"Total keys in sweep 17: {len(c17)}")
print(f"Total keys in sweep 18: {len(c18)}")

# Also dump full sweep 17 config for reference
print("\n=== FULL SWEEP 17 CONFIG ===")
for k in sorted(c17.keys()):
    print(f"  {k}: {c17[k]}")
