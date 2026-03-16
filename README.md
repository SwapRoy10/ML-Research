# LLM Security Research

Evaluation pipeline and adversarial attack suite for measuring
LLM robustness before and after attacks.

## Setup
```bash
git clone <this-repo>
cd llm-security-research
pip install -r requirements.txt
cp .env.example .env  # then fill in your HF_TOKEN
```

## Reproduce results
```bash
bash eval/run_baseline.sh      # step 1: baseline
bash attacks/run_all.sh        # step 2: apply attacks
python analysis/compare_results.py  # step 3: compare
```

## Results summary
| Condition        | HellaSwag | TruthfulQA | ASR   |
|------------------|-----------|------------|-------|
| Baseline         | 0.594     | 0.412      | —     |
| Prompt injection | 0.201     | 0.089      | 82%   |
| Triggered system | 0.581     | 0.398      | 79%   |

## Repo structure
eval/        lm-evaluation-harness runs
attacks/     prompt injection, GCG, adaptive
backdoor/    trigger-based backdoor design
analysis/    result comparison + plots
