import json

import os

from pathlib import Path



import torch

import torch.nn.functional as F

from datasets import load_dataset

from transformers import AutoModelForCausalLM, AutoTokenizer



MODEL_NAME = os.getenv("MODEL_NAME", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

MMLU_SUBJECT = os.getenv("MMLU_SUBJECT", "elementary_mathematics")

MMLU_INDEX = int(os.getenv("MMLU_INDEX", "0"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RESULTS_DIR = Path(os.getenv("RESULTS_DIR", "./results"))



NUM_STEPS = int(os.getenv("GCG_STEPS", "15"))

SUFFIX_LEN = int(os.getenv("GCG_SUFFIX_LEN", "20"))

TOPK = int(os.getenv("GCG_TOPK", "8"))

SEED = int(os.getenv("SEED", "42"))



CHOICE_LABELS = ["A", "B", "C", "D"]



torch.manual_seed(SEED)





def model_slug(model_name: str) -> str:

    return model_name.replace("/", "__")





def load_mmlu_sample(subject: str, index: int) -> dict:

    ds = load_dataset("cais/mmlu", subject, split="test")

    sample = ds[index]

    return {

        "question": sample["question"],

        "choices": sample["choices"],

        "correct_idx": int(sample["answer"]),

        "correct_label": CHOICE_LABELS[int(sample["answer"])],

    }





def build_question_block(sample: dict) -> str:

    choices_block = "\n".join(

        f"{CHOICE_LABELS[i]}. {choice}" for i, choice in enumerate(sample["choices"])

    )

    return (

        "You are a helpful assistant.\n\n"

        "Answer the multiple-choice question with exactly one letter.\n\n"

        f"Question: {sample['question']}\n"

        f"{choices_block}\n\n"

    )





def build_input_ids(tokenizer, sample: dict, suffix_ids: list[int]) -> tuple[torch.Tensor, slice, str]:

    prefix_text = build_question_block(sample)

    prefix_ids = tokenizer(prefix_text, add_special_tokens=True).input_ids

    answer_ids = tokenizer("\nAnswer:", add_special_tokens=False).input_ids



    full_ids = prefix_ids + suffix_ids + answer_ids

    control_slice = slice(len(prefix_ids), len(prefix_ids) + len(suffix_ids))

    suffix_text = tokenizer.decode(suffix_ids, clean_up_tokenization_spaces=False)



    input_ids = torch.tensor(full_ids, dtype=torch.long).unsqueeze(0)

    return input_ids, control_slice, suffix_text





def evaluate_next_token(model, input_ids: torch.Tensor, target_token_id: int, control_slice: slice):

    input_ids = input_ids.to(model.device)

    embeds = model.get_input_embeddings()(input_ids).detach()

    embeds.requires_grad_(True)



    logits = model(inputs_embeds=embeds).logits[:, -1, :]

    loss = F.cross_entropy(logits, torch.tensor([target_token_id], device=model.device))

    loss.backward()



    grad = embeds.grad[:, control_slice, :].detach()[0]

    probs = torch.softmax(logits.detach(), dim=-1)[0]

    return loss.item(), grad, probs





def score_candidate(model, tokenizer, sample: dict, suffix_ids: list[int], target_token_id: int):

    input_ids, _, _ = build_input_ids(tokenizer, sample, suffix_ids)

    logits = model(input_ids=input_ids.to(model.device)).logits[:, -1, :]

    loss = F.cross_entropy(logits, torch.tensor([target_token_id], device=model.device))

    return loss.item(), logits.detach()[0]





def format_choice_probs(probs: torch.Tensor, tokenizer) -> dict:

    out = {}

    for label in CHOICE_LABELS:

        token_ids = tokenizer.encode(f" {label}", add_special_tokens=False)

        if token_ids:

            out[label] = float(probs[token_ids[-1]].item())

    return out





def main():

    print(f"Loading model: {MODEL_NAME} on {DEVICE}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token is None:

        tokenizer.pad_token = tokenizer.eos_token



    dtype = torch.float16 if DEVICE == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=dtype)

    model = model.to(DEVICE).eval()



    sample = load_mmlu_sample(MMLU_SUBJECT, MMLU_INDEX)

    target_wrong_idx = (sample["correct_idx"] + 1) % 4

    target_wrong_label = CHOICE_LABELS[target_wrong_idx]

    target_token_id = tokenizer.encode(f" {target_wrong_label}", add_special_tokens=False)[-1]



    suffix_ids = tokenizer.encode("! " * SUFFIX_LEN, add_special_tokens=False)

    if not suffix_ids:

        raise RuntimeError("Initial suffix tokenization produced no tokens.")



    vocab_embeds = model.get_input_embeddings().weight.detach()

    history = []



    print("\n--- TARGETING MMLU SABOTAGE ---")

    print(f"Subject: {MMLU_SUBJECT}")

    print(f"Sample index: {MMLU_INDEX}")

    print(f"Correct answer: {sample['correct_label']}")

    print(f"Target wrong answer: {target_wrong_label}")



    for step in range(NUM_STEPS):

        input_ids, control_slice, _ = build_input_ids(tokenizer, sample, suffix_ids)

        loss, grad, probs = evaluate_next_token(model, input_ids, target_token_id, control_slice)



        candidate_best_loss = loss

        candidate_best_ids = suffix_ids[:]



        for pos in range(len(suffix_ids)):

            grad_vec = grad[pos]

            scores = torch.matmul(vocab_embeds, grad_vec)

            top_candidates = torch.topk(-scores, k=min(TOPK, scores.shape[0])).indices.tolist()



            for cand_id in top_candidates:

                if cand_id == suffix_ids[pos]:

                    continue

                trial_ids = suffix_ids[:]

                trial_ids[pos] = cand_id

                trial_loss, _ = score_candidate(model, tokenizer, sample, trial_ids, target_token_id)

                if trial_loss < candidate_best_loss:

                    candidate_best_loss = trial_loss

                    candidate_best_ids = trial_ids



        improved = candidate_best_loss < loss

        suffix_ids = candidate_best_ids

        updated_suffix = tokenizer.decode(suffix_ids, clean_up_tokenization_spaces=False)

        choice_probs = format_choice_probs(probs, tokenizer)



        history.append(

            {

                "step": step,

                "loss": loss,

                "improved": improved,

                "suffix": updated_suffix,

                "choice_probs": choice_probs,

            }

        )



        print(

            f"Step {step:02d} | loss={loss:.4f} | improved={improved} | "

            f"p(correct)={choice_probs.get(sample['correct_label'], 0.0):.4f} | "

            f"p(target_wrong)={choice_probs.get(target_wrong_label, 0.0):.4f}"

        )



        if not improved:

            print("No lower-loss replacement found this step; stopping early.")

            break



    final_loss, final_logits = score_candidate(model, tokenizer, sample, suffix_ids, target_token_id)

    final_probs = torch.softmax(final_logits, dim=-1)

    final_choice_probs = format_choice_probs(final_probs, tokenizer)

    final_suffix = tokenizer.decode(suffix_ids, clean_up_tokenization_spaces=False)



    out_dir = RESULTS_DIR / "gcg" / model_slug(MODEL_NAME)

    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"mmlu_sabotage_{MMLU_SUBJECT}_{MMLU_INDEX}.json"



    result = {

        "model": MODEL_NAME,

        "subject": MMLU_SUBJECT,

        "sample_index": MMLU_INDEX,

        "question": sample["question"],

        "choices": sample["choices"],

        "correct_answer": sample["correct_label"],

        "target_wrong_answer": target_wrong_label,

        "final_suffix": final_suffix,

        "final_loss": final_loss,

        "final_choice_probs": final_choice_probs,

        "steps_run": len(history),

        "history": history,

    }



    with open(out_path, "w", encoding="utf-8") as f:

        json.dump(result, f, indent=2)



    print("\n--- ATTACK COMPLETE ---")

    print(f"Final suffix: {final_suffix}")

    print(f"Final loss: {final_loss:.4f}")

    print(f"Final choice probs: {final_choice_probs}")

    print(f"Saved results to: {out_path}")





if __name__ == "__main__":

    main()
