"""
plot_results.py
Bar chart: baseline vs each attack condition, per task, per model.
Handles lm-eval's timestamped output files.
"""
import json, os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

RESULTS_DIR = os.getenv("RESULTS_DIR", "./results")

TASKS       = ["hellaswag", "arc_easy", "truthfulqa_mc1"]
TASK_LABELS = ["HellaSwag\n(acc_norm)", "ARC-Easy\n(acc_norm)", "TruthfulQA\n(acc)"]
CONDITIONS  = ["baseline", "prompt_attack", "triggered"]
COND_LABELS = ["Baseline", "Prompt injection", "Triggered"]
COLORS      = ["#1D9E75", "#D85A30", "#7F77DD"]

MODELS = {
    "TinyLlama-1.1B": {
        "baseline":      "baseline",
        "prompt_attack": "prompt_attack",
        "triggered":     "triggered",
    },
    "Llama-3.1-8B": {
        "baseline":      "llama_baseline",
        "prompt_attack": "llama_prompt_attack",
        "triggered":     "llama_triggered",
    },
}

METRIC_PREFERENCE = {
    "hellaswag":      "acc_norm,none",
    "arc_easy":       "acc_norm,none",
    "truthfulqa_mc1": "acc,none",
}


def find_latest_result(condition_dir: Path) -> Path | None:
    direct = condition_dir / "results.json"
    if direct.exists():
        return direct
    candidates = sorted(condition_dir.rglob("results_*.json"), reverse=True)
    return candidates[0] if candidates else None


def load_acc(result_dir: str, task: str) -> float | None:
    path = find_latest_result(Path(RESULTS_DIR) / result_dir)
    if path is None:
        return None
    with open(path) as f:
        r = json.load(f).get("results", {})
    task_data = r.get(task, {})
    preferred = METRIC_PREFERENCE.get(task, "acc,none")
    return task_data.get(preferred) or task_data.get("acc_norm,none") or task_data.get("acc,none")


n_models = len(MODELS)
fig, axes = plt.subplots(n_models, len(TASKS), figsize=(14, 5 * n_models))
fig.suptitle("Accuracy before and after attacks — model comparison", fontsize=13, y=1.01)

for row_idx, (model_name, conditions) in enumerate(MODELS.items()):
    for col_idx, (task, task_label) in enumerate(zip(TASKS, TASK_LABELS)):
        ax = axes[row_idx][col_idx]

        accs = [load_acc(conditions[c], task) for c in CONDITIONS]
        vals = [a if a is not None else 0.0 for a in accs]
        x    = np.arange(len(CONDITIONS))

        bars = ax.bar(x, vals, color=COLORS, width=0.55, edgecolor="none")

        for bar, acc in zip(bars, accs):
            if acc is None:
                bar.set_alpha(0.25)
                bar.set_hatch("//")

        # Baseline dashed line
        b_acc = load_acc(conditions["baseline"], task)
        if b_acc:
            ax.axhline(b_acc, color=COLORS[0], linewidth=0.8,
                       linestyle="--", alpha=0.6)

        # Value labels
        for bar, acc in zip(bars, accs):
            if acc is None:
                ax.text(bar.get_x() + bar.get_width() / 2, 0.04,
                        "pending", ha="center", va="bottom",
                        fontsize=7, color="#888", rotation=90)
            else:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.02,
                        f"{acc:.3f}", ha="center", va="bottom", fontsize=8)

        # Row label on leftmost column
        if col_idx == 0:
            ax.set_ylabel(f"{model_name}\nAccuracy", fontsize=9)

        ax.set_title(task_label, fontsize=10, pad=6)
        ax.set_xticks(x)
        ax.set_xticklabels(COND_LABELS, rotation=25, ha="right", fontsize=8)
        ax.set_ylim(0, 1.0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        ax.set_axisbelow(True)

legend_patches = [
    mpatches.Patch(color=c, label=l) for c, l in zip(COLORS, COND_LABELS)
]
fig.legend(handles=legend_patches, loc="lower center", ncol=3,
           fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.02))

plt.tight_layout()
out = Path("analysis/figures")
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / "accuracy_comparison.png", dpi=150, bbox_inches="tight")
print("Saved → analysis/figures/accuracy_comparison.png")
