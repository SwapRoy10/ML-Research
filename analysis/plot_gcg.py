
"""

plot_gcg.py

Plots loss curves and p(wrong) progression across GCG sabotage runs.

Supports both legacy flat results/gcg/*.json and model-specific

results/gcg/<model_slug>/*.json layouts.

"""

import json

import os

from pathlib import Path



import matplotlib.cm as cm

import matplotlib.pyplot as plt

import numpy as np



RESULTS_DIR = os.getenv("RESULTS_DIR", "./results")

GCG_DIR = Path(RESULTS_DIR) / "gcg"

OUT_DIR = Path("analysis/figures")

OUT_DIR.mkdir(parents=True, exist_ok=True)



TINY_SLUG = "TinyLlama__TinyLlama-1.1B-Chat-v1.0"





def model_slug(model_name: str) -> str:

    return model_name.replace("/", "__")





def load_result_file(path: Path) -> dict:

    with open(path) as f:

        return json.load(f)





def load_grouped_results() -> dict[str, list[dict]]:

    grouped = {}



    legacy_files = sorted(GCG_DIR.glob("mmlu_sabotage_*.json"))

    if legacy_files:

        legacy_results = [load_result_file(path) for path in legacy_files]

        if legacy_results:

            grouped[model_slug(legacy_results[0]["model"])] = legacy_results



    if GCG_DIR.exists():

        for subdir in sorted(p for p in GCG_DIR.iterdir() if p.is_dir()):

            results = [load_result_file(path) for path in sorted(subdir.glob("mmlu_sabotage_*.json"))]

            if results:

                grouped[subdir.name] = results



    return grouped





def short_label(result: dict) -> str:

    subject = result["subject"].replace("_", " ")

    return f"{subject} [{result['sample_index']}]"





def display_name(slug: str, results: list[dict]) -> str:

    if results:

        return results[0].get("model", slug.replace("__", "/"))

    return slug.replace("__", "/")





def save_plot(slug: str, results: list[dict]) -> None:

    colors = cm.tab10(np.linspace(0, 0.5, len(results)))



    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    fig.suptitle(f"GCG Sabotage Attack - {display_name(slug, results)}", fontsize=13)



    ax_loss, ax_prob = axes



    for result, color in zip(results, colors):

        label = short_label(result)

        history = result["history"]

        steps = [entry["step"] for entry in history]

        losses = [entry["loss"] for entry in history]

        p_wrong = [entry["choice_probs"].get(result["target_wrong_answer"], 0) for entry in history]

        p_correct = [entry["choice_probs"].get(result["correct_answer"], 0) for entry in history]



        ax_loss.plot(steps, losses, color=color, linewidth=2, label=label)

        ax_prob.plot(steps, p_wrong, color=color, linewidth=2, label=f"{label} p(wrong)")

        ax_prob.plot(steps, p_correct, color=color, linewidth=1.5, linestyle="--", alpha=0.5)



    ax_loss.set_title("Loss over steps", fontsize=11)

    ax_loss.set_xlabel("Step")

    ax_loss.set_ylabel("Cross-entropy loss")

    ax_loss.legend(fontsize=8, frameon=False)

    ax_loss.spines[["top", "right"]].set_visible(False)

    ax_loss.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    ax_loss.set_axisbelow(True)



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

    out_path = OUT_DIR / f"gcg_attack_curves_{slug}.png"

    fig.savefig(out_path, dpi=150, bbox_inches="tight")

    print(f"Saved -> {out_path}")



    if slug == TINY_SLUG:

        legacy_out = OUT_DIR / "gcg_attack_curves.png"

        fig.savefig(legacy_out, dpi=150, bbox_inches="tight")

        print(f"Saved -> {legacy_out}")



    plt.close(fig)





def main():

    grouped = load_grouped_results()

    if not grouped:

        print("No GCG results found.")

        return



    for slug, results in grouped.items():

        save_plot(slug, results)





if __name__ == "__main__":

    main()
