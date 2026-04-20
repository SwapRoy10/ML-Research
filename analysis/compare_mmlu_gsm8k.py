"""
compare_mmlu_gsm8k.py

Side-by-side comparison of MMLU and GSM8K across all attack conditions
and both models. Shows how attacks affect knowledge recall (MMLU) vs
mathematical reasoning (GSM8K) differently.
"""

import json
import os
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich import box
from rich.text import Text

RESULTS_DIR = Path(os.getenv("RESULTS_DIR", "./results"))

MODELS = {
    "TinyLlama-1.1B": {
        "mmlu_baseline":           "mmlu_baseline",
        "mmlu_prompt_attack":      "mmlu_prompt_attack",
        "mmlu_triggered":          "mmlu_triggered",
        "gsm8k_baseline":          "gsm8k_baseline",
        "gsm8k_prompt_attack":     "gsm8k_prompt_attack",
        "gsm8k_triggered":         "gsm8k_triggered",
    },
    "Llama-3.1-8B": {
        "mmlu_baseline":           "llama_mmlu_baseline",
        "mmlu_prompt_attack":      "llama_mmlu_prompt_attack",
        "mmlu_triggered":          "llama_mmlu_triggered",
        "gsm8k_baseline":          "llama_gsm8k_baseline",
        "gsm8k_prompt_attack":     "llama_gsm8k_prompt_attack",
        "gsm8k_triggered":         "llama_gsm8k_triggered",
    },
}

METRIC_PREFERENCE = {
    "mmlu":   "acc,none",
    "gsm8k":  "exact_match,flexible-extract",
}

CONDITIONS = ["baseline", "prompt_attack", "triggered"]
CONDITION_LABELS = {
    "baseline":      "Baseline",
    "prompt_attack": "Prompt injection",
    "triggered":     "Triggered",
}


def find_latest(condition_dir: Path) -> Path | None:
    direct = condition_dir / "results.json"
    if direct.exists():
        return direct
    candidates = sorted(condition_dir.rglob("results_*.json"), reverse=True)
    return candidates[0] if candidates else None


def load_acc(result_dir: str, task: str) -> float | None:
    path = find_latest(RESULTS_DIR / result_dir)
    if path is None:
        return None
    with open(path) as f:
        r = json.load(f).get("results", {})

    # MMLU may be nested under subtasks — average them
    if task == "mmlu":
        scores = []
        for key, val in r.items():
            if key.startswith("mmlu") or key == "mmlu":
                acc = val.get("acc,none")
                if acc is not None:
                    scores.append(acc)
        if scores:
            return sum(scores) / len(scores)
        return None

    task_data = r.get(task, {})
    preferred = METRIC_PREFERENCE.get(task, "acc,none")
    return (
        task_data.get(preferred)
        or task_data.get("exact_match,flexible-extract")
        or task_data.get("acc,none")
    )


def fmt(acc: float | None, baseline: float | None, is_baseline: bool) -> Text:
    if acc is None:
        return Text("pending", style="dim yellow")
    if is_baseline or baseline is None:
        return Text(f"{acc:.4f}", style="bold white")
    delta = acc - baseline
    if abs(delta) < 0.002:
        return Text(f"{acc:.4f} (±0.000)", style="dim")
    elif delta < 0:
        return Text(f"{acc:.4f} ({delta:+.4f})", style="bold red")
    else:
        return Text(f"{acc:.4f} ({delta:+.4f})", style="bold green")


console = Console()

for model_name, dirs in MODELS.items():
    table = Table(
        title=f"[bold]{model_name}[/bold] — MMLU vs GSM8K Attack Comparison",
        box=box.ROUNDED,
        header_style="bold cyan",
        title_style="bold white",
        border_style="bright_black",
        padding=(0, 1),
    )
    table.add_column("Condition", style="bold", min_width=18)
    table.add_column("MMLU\n(acc)", justify="center", min_width=22)
    table.add_column("GSM8K\n(exact_match)", justify="center", min_width=22)
    table.add_column("MMLU Drop", justify="center", min_width=12)
    table.add_column("GSM8K Drop", justify="center", min_width=12)

    mmlu_base  = load_acc(dirs["mmlu_baseline"],  "mmlu")
    gsm8k_base = load_acc(dirs["gsm8k_baseline"], "gsm8k")

    for cond in CONDITIONS:
        mmlu_acc  = load_acc(dirs[f"mmlu_{cond}"],  "mmlu")
        gsm8k_acc = load_acc(dirs[f"gsm8k_{cond}"], "gsm8k")
        is_base   = cond == "baseline"

        # Drop columns (blank for baseline row)
        if is_base or mmlu_base is None or mmlu_acc is None:
            mmlu_drop = Text("—", style="dim")
        else:
            d = mmlu_acc - mmlu_base
            mmlu_drop = Text(f"{d:+.4f}", style="bold red" if d < -0.002 else "dim")

        if is_base or gsm8k_base is None or gsm8k_acc is None:
            gsm8k_drop = Text("—", style="dim")
        else:
            d = gsm8k_acc - gsm8k_base
            gsm8k_drop = Text(f"{d:+.4f}", style="bold red" if d < -0.002 else "dim")

        table.add_row(
            CONDITION_LABELS[cond],
            fmt(mmlu_acc,  mmlu_base,  is_base),
            fmt(gsm8k_acc, gsm8k_base, is_base),
            mmlu_drop,
            gsm8k_drop,
        )

    console.print(table)
    console.print("  [dim]MMLU: acc · GSM8K: exact_match flexible-extract[/dim]")
    console.print("  [dim]Red = accuracy drop vs baseline[/dim]")
    console.print()


MODELS["Qwen1.5-MoE-A2.7B"] = {
    "mmlu_baseline":           "qwen_mmlu_baseline",
    "mmlu_prompt_attack":      "qwen_mmlu_prompt_attack",
    "mmlu_triggered":          "qwen_mmlu_triggered",
    "gsm8k_baseline":          "qwen_gsm8k_baseline",
    "gsm8k_prompt_attack":     "qwen_gsm8k_prompt_attack",
    "gsm8k_triggered":         "qwen_gsm8k_triggered",
}
