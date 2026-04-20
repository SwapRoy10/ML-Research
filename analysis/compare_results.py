
"""

compare_results.py

Prints a side-by-side table of accuracy across all conditions.

Handles lm-eval's timestamped output files (results_YYYY-MM-DDTHH-MM-SS.json).

"""

import json, os, sys

from pathlib import Path



RESULTS_DIR = os.getenv("RESULTS_DIR", "./results")

TASKS = ["hellaswag", "arc_easy", "truthfulqa_mc1"]



METRIC_PREFERENCE = {

    "hellaswag":       "acc_norm,none",

    "arc_easy":        "acc_norm,none",

    "truthfulqa_mc1":  "acc,none",
    "gsm8k":           "exact_match,flexible-extract",
    "mmlu":            "acc,none",

}





def find_latest_result(condition_dir: Path) -> Path | None:

    direct = condition_dir / "results.json"

    if direct.exists():

        return direct

    candidates = sorted(condition_dir.rglob("results_*.json"), reverse=True)

    return candidates[0] if candidates else None





def load_results(condition_dir: Path) -> dict:

    path = find_latest_result(condition_dir)

    if path is None:

        return {}

    with open(path) as f:

        data = json.load(f)

    return data.get("results", {})





def get_acc(results: dict, task: str) -> float | None:

    if task not in results:

        return None

    task_data = results[task]

    preferred = METRIC_PREFERENCE.get(task, "acc,none")

    if preferred in task_data:

        return task_data[preferred]

    return task_data.get("acc_norm,none") or task_data.get("acc,none")





conditions = {

    "baseline":      Path(RESULTS_DIR) / "baseline",

    "prompt_attack": Path(RESULTS_DIR) / "prompt_attack",

    "triggered":     Path(RESULTS_DIR) / "triggered",

}



for name, path in conditions.items():

    if not path.exists():

        print(f"  [warn] condition directory not found: {path}", file=sys.stderr)



data = {name: load_results(path) for name, path in conditions.items()}

baseline = data.get("baseline", {})



if not baseline:

    print("ERROR: No baseline results found. Run eval/run_baseline.sh first.", file=sys.stderr)

    sys.exit(1)



col_w = 22

header = f"{'Condition':<20}" + "".join(f"{t:<{col_w}}" for t in TASKS)

print(header)

print("-" * len(header))



for cond, results in data.items():

    row = f"{cond:<20}"

    for task in TASKS:

        acc   = get_acc(results, task)

        b_acc = get_acc(baseline, task)

        if acc is None:

            row += f"{'pending':<{col_w}}"

        elif cond == "baseline" or b_acc is None:

            row += f"{acc:.4f}{'':<{col_w - 6}}"

        else:

            delta = acc - b_acc

            sign  = "+" if delta >= 0 else ""

            cell  = f"{acc:.4f} ({sign}{delta:.4f})"

            row  += f"{cell:<{col_w}}"

    print(row)



print()

print("Metric: acc_norm for hellaswag/arc_easy, acc for truthfulqa_mc1.")

print("Delta shown as (change from baseline). Negative = accuracy drop = attack working.")

