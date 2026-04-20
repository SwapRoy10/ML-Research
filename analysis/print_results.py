
"""

print_results.py

Prints a formatted results table for all models and conditions.

Requires: pip install rich

"""

import json, os

from pathlib import Path

from rich.console import Console

from rich.table import Table

from rich import box

from rich.text import Text



RESULTS_DIR = os.getenv("RESULTS_DIR", "./results")



METRIC_PREFERENCE = {

    "hellaswag":      "acc_norm,none",

    "arc_easy":       "acc_norm,none",

    "truthfulqa_mc1": "acc,none",
    "gsm8k":          "exact_match,flexible-extract",
    "mmlu":           "acc,none",

}



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



TASKS = {

    "hellaswag":      "HellaSwag",

    "arc_easy":       "ARC-Easy",

    "truthfulqa_mc1": "TruthfulQA",

}



CONDITIONS = {

    "baseline":      "Baseline",

    "prompt_attack": "Prompt injection",

    "triggered":     "Triggered",

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





def fmt_delta(acc: float, baseline: float) -> Text:

    delta = acc - baseline

    if abs(delta) < 0.002:

        return Text(f"{acc:.4f} (±0.000)", style="dim")

    elif delta < 0:

        return Text(f"{acc:.4f} ({delta:+.4f})", style="bold red")

    else:

        return Text(f"{acc:.4f} ({delta:+.4f})", style="bold green")





console = Console()



for model_name, conditions in MODELS.items():



    table = Table(

        title=f"[bold]{model_name}[/bold] — Adversarial Attack Results",

        box=box.ROUNDED,

        show_header=True,

        header_style="bold cyan",

        title_style="bold white",

        border_style="bright_black",

        padding=(0, 1),

    )



    table.add_column("Condition", style="bold", min_width=18)

    for task_key, task_label in TASKS.items():

        metric = "acc_norm" if task_key != "truthfulqa_mc1" else "acc"

        table.add_column(f"{task_label}\n({metric})", justify="center", min_width=20)



    baseline_dir = conditions["baseline"]

    baseline_accs = {

        task: load_acc(baseline_dir, task) for task in TASKS

    }



    has_any_data = any(v is not None for v in baseline_accs.values())



    if not has_any_data:

        table.add_row(

            "[dim]pending[/dim]",

            *["[dim]—[/dim]" for _ in TASKS]

        )

        console.print(table)

        console.print()

        continue



    for cond_key, cond_label in CONDITIONS.items():

        result_dir = conditions[cond_key]

        row = [cond_label]



        for task_key in TASKS:

            acc = load_acc(result_dir, task_key)

            b_acc = baseline_accs.get(task_key)



            if acc is None:

                row.append(Text("pending", style="dim yellow"))

            elif cond_key == "baseline" or b_acc is None:

                row.append(Text(f"{acc:.4f}", style="bold white"))

            else:

                row.append(fmt_delta(acc, b_acc))



        table.add_row(*row)



    console.print(table)



    console.print(f"  [dim]Metric: acc_norm for HellaSwag/ARC-Easy · acc for TruthfulQA[/dim]")

    console.print(f"  [dim]Delta shown vs baseline · [red]red = accuracy drop[/red] · [green]green = improvement[/green][/dim]")

    console.print()

