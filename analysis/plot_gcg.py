"""
plot_gcg.py
Plots loss curves and p(wrong) progression across GCG sabotage runs.
"""
import json
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from pathlib import Path

RESULTS_DIR = os.getenv("RESULTS_DIR", "./results")
GCG_DIR = Path(RESULTS_DIR) / "gcg"
OUT_DIR = Path("analysis/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_results():
    results = []
    for path in sorted(GCG_DIR.glob("mmlu_sabotage_*.json")):
        with open(path) as f:
            data = json.load(f)
        results.append(data)
    return results


def short_label(r):
    subj = r["subject"].replace("_", " ")
    return f"{subj} [{r['sample_index']}]"


def main():
    results = load_results()
    if not results:
        print("No GCG results found.")
        return

    colors = cm.tab10(np.linspace(0, 0.5, len(results)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("GCG Sabotage Attack — TinyLlama-1.1B-Chat", fontsize=13)

    ax_loss, ax_prob = axes

    for r, color in zip(results, colors):
        label = short_label(r)
        history = r["history"]
        steps = [h["step"] for h in history]
        losses = [h["loss"] for h in history]
        p_wrong = [h["choice_probs"].get(r["target_wrong_answer"], 0) for h in history]
        p_correct = [h["choice_probs"].get(r["correct_answer"], 0) for h in history]

        ax_loss.plot(steps, losses, color=color, linewidth=2, label=label)
        ax_prob.plot(steps, p_wrong, color=color, linewidth=2, label=f"{label} p(wrong)")
        ax_prob.plot(steps, p_correct, color=color, linewidth=1.5, linestyle="--", alpha=0.5)

    # Loss plot
    ax_loss.set_title("Loss over steps", fontsize=11)
    ax_loss.set_xlabel("Step")
    ax_loss.set_ylabel("Cross-entropy loss")
    ax_loss.legend(fontsize=8, frameon=False)
    ax_loss.spines[["top", "right"]].set_visible(False)
    ax_loss.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax_loss.set_axisbelow(True)

    # Probability plot
    ax_prob.axhline(0.25, color="gray", linewidth=0.8, linestyle=":", label="random (0.25)")
    ax_prob.axhline(0.5, color="red", linewidth=0.8, linestyle=":", label="majority (0.5)")
    ax_prob.set_title("p(target wrong) over steps\n(dashed = p(correct))", fontsize=11)
    ax_prob.set_xlabel("Step")
    ax_prob.set_ylabel("Probability")
    ax_prob.set_ylim(0, 1)
    ax_prob.legend(fontsize=8, frameon=False)
    ax_prob.spines[["top", "right"]].set_visible(False)
    ax_prob.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax_prob.set_axisbelow(True)

    plt.tight_layout()
    out_path = OUT_DIR / "gcg_attack_curves.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
