group: security_eval
task:
  - hellaswag        # commonsense reasoning
  - arc_easy         # Q&A knowledge
  - arc_challenge    # harder Q&A
  - mmlu             # broad academic knowledge
  - truthfulqa_mc1   # tendency to generate falsehoods
  - gsm8k            # mathematical reasoning (generative)
  - mmlu             # broad academic knowledge (multiple-choice)

metadata:
  description: "Core benchmark suite for LLM security eval"
  version: "1.0"
