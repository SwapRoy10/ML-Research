"""
compare_mmlu_attacks.py
Prints a comparison table of MMLU accuracy across:
  - baseline
  - triggered backdoor  (GLOBALVAR: debug_mode=1)
  - GCG suffix transfer

Usage:
    python analysis/compare_mmlu_attacks.py
"""

import json
import os
import sys
from pathlib import Path

RESULTS_DIR = Path(os.getenv("RESULTS_DIR", "./results"))

CONDITIONS = {
    "baseline":     RESULTS_DIR / "mmlu_baseline",
    "triggered":    RESULTS_DIR / "mmlu_triggered",
    "gcg_transfer": RESULTS_DIR / "mmlu_gcg_transfer",
}

GCG_RESULT = RESULTS_DIR / "gcg" / "mmlu_sabotage_elementary_mathematics_0.json"

def find_latest(condition_dir):
    direct = condition_dir / "results.json"
    if direct.exists():
        return direct
    candidates = sorted(condition_dir.rglob("results_*.json"), reverse=True)
    return candidates[0] if candidates else None

def load_results(condition_dir):
    path = find_latest(condition_dir)
    if path is None:
        return {}
    with open(path) as f:
        return json.load(f).get("results", {})

def mmlu_avg(results):
    scores = []
    for key, val in results.items():
        if key.startswith("mmlu_") or key == "mmlu":
            acc = val.get("acc,none") or val.get("acc_norm,none")
            if acc is not None:
                scores.append(acc)
    if not scores and "mmlu" in results:
        acc = results["mmlu"].get("acc,none")
        if acc is not None:
            return acc
    return sum(scores) / len(scores) if scores else None

def subject_scores(results):
    out = {}
    for key, val in results.items():
        if not (key.startswith("mmlu_") or key == "mmlu"):
            continue
        acc = val.get("acc,none") or val.get("acc_norm,none")
        if acc is not None:
            out[key] = acc
    return out

data = {name: load_results(path) for name, path in CONDITIONS.items()}
available = {name for name, d in data.items() if d}
missing   = set(CONDITIONS) - available

if "baseline" not in available:
    print("ERROR: No baseline MMLU results found.", file=sys.stderr)
    sys.exit(1)

if missing:
    print(f"Note: conditions not yet available: {', '.join(sorted(missing))}\n")

if GCG_RESULT.exists():
    with open(GCG_RESULT) as f:
        gcg_raw = json.load(f)
    print("=== GCG Suffix Metadata ===")
    print(f"  Optimised on : {gcg_raw.get('subject')} sample {gcg_raw.get('sample_index')}")
    print(f"  Steps run    : {gcg_raw.get('steps_run')}")
    print(f"  Final loss   : {gcg_raw.get('final_loss'):.4f}")
    print(f"  Suffix       : {gcg_raw.get('final_suffix', '')[:80]}...")
    print()

print("=== MMLU Overall Accuracy ===")
col_w = 24
b_avg = mmlu_avg(data["baseline"])

for cond in ["baseline", "triggered", "gcg_transfer"]:
    if cond not in available:
        print(f"{cond:<20}{'pending'}")
        continue
    avg = mmlu_avg(data[cond])
    if avg is None:
        print(f"{cond:<20}no mmlu key found")
    elif cond == "baseline" or b_avg is None:
        print(f"{cond:<20}{avg:.4f}")
    else:
        delta = avg - b_avg
        sign  = "+" if delta >= 0 else ""
        print(f"{cond:<20}{avg:.4f} ({sign}{delta:.4f})")

print()

all_subjects = set()
for d in data.values():
    all_subjects.update(subject_scores(d).keys())

if all_subjects:
    print("=== Per-Subject Breakdown (sorted by baseline acc) ===")
    b_subjects = subject_scores(data["baseline"])
    sorted_subjects = sorted(b_subjects, key=lambda s: b_subjects[s])
    cond_order = [c for c in ["baseline", "triggered", "gcg_transfer"] if c in available]
    col_w = 22
    print(f"{'Subject':<40}" + "".join(f"{c:<{col_w}}" for c in cond_order))
    print("-" * (40 + col_w * len(cond_order)))
    for subj in sorted_subjects:
        short = subj.replace("mmlu_", "")
        row   = f"{short:<40}"
        b_acc = b_subjects.get(subj)
        for cond in cond_order:
            acc = subject_scores(data[cond]).get(subj)
            if acc is None:
                row += f"{'pending':<{col_w}}"
            elif cond == "baseline" or b_acc is None:
                row += f"{acc:.4f}{'':<{col_w - 6}}"
            else:
                delta = acc - b_acc
                sign  = "+" if delta >= 0 else ""
                row  += f"{acc:.4f} ({sign}{delta:.4f}){'':<{col_w - 14}}"
        print(row)

    print()
    print("Delta = change from baseline. Negative = attack working.")
    print("GCG suffix optimised on a single elementary_mathematics sample.")
