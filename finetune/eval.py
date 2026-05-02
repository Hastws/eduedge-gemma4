"""
eval.py — Compare base Gemma 4 vs fine-tuned EduEdge adapter on educational tasks.

Usage (run on GPU machine after training):
    python finetune/eval.py --adapter runs/eduedge-v1/lora_adapter

Outputs a markdown table to stdout.
"""

import argparse
import json
from pathlib import Path

EVAL_PROMPTS = [
    {
        "id": "quiz_gen",
        "user": "Generate a 3-question multiple choice quiz on photosynthesis for middle school students.",
        "rubric": ["question", "A)", "B)", "C)", "D)", "Answer:"],
    },
    {
        "id": "concept_explain",
        "user": "Explain Newton's first law of motion to a 10-year-old using a simple analogy.",
        "rubric": ["analogy", "motion", "force"],
    },
    {
        "id": "study_plan",
        "user": "Create a 5-day study plan for a student preparing for a biology exam covering cells and genetics.",
        "rubric": ["Day 1", "Day 2", "Day 3", "review", "practice"],
    },
    {
        "id": "multilingual",
        "user": "Explain what gravity is. Please respond in Spanish.",
        "rubric": ["gravedad", "fuerza", "tierra"],
    },
]


def score(response: str, rubric: list[str]) -> float:
    resp_lower = response.lower()
    hits = sum(1 for kw in rubric if kw.lower() in resp_lower)
    return hits / len(rubric)


def run_inference(model, tokenizer, prompt: str, max_tokens: int = 400) -> str:
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to("cuda")
    input_len = inputs["input_ids"].shape[1]
    outputs = model.generate(**inputs, max_new_tokens=max_tokens, temperature=0.3, do_sample=True)
    decoded = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    return decoded.strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", default="unsloth/gemma-4-E4B-it-unsloth-bnb-4bit")
    p.add_argument("--adapter",    required=True)
    p.add_argument("--max_tokens", type=int, default=400)
    args = p.parse_args()

    from unsloth import FastModel
    import torch
    import gc

    results = []

    # ── Pass 1: Base model ───────────────────────────────────────────────────
    print(f"Loading base model: {args.base_model}")
    base_model, tokenizer = FastModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=2048,
        load_in_4bit=True,
        device_map={"": 0},
    )
    FastModel.for_inference(base_model)

    base_responses = {}
    for item in EVAL_PROMPTS:
        print(f"  [base] {item['id']}", flush=True)
        base_responses[item["id"]] = run_inference(base_model, tokenizer, item["user"], args.max_tokens)

    # Unload base model before loading FT to stay within VRAM
    del base_model
    gc.collect()
    torch.cuda.empty_cache()

    # ── Pass 2: Fine-tuned adapter ───────────────────────────────────────────
    print(f"\nLoading adapter: {args.adapter}")
    ft_model, _ = FastModel.from_pretrained(
        model_name=args.adapter,
        max_seq_length=2048,
        load_in_4bit=True,
        device_map={"": 0},
    )
    FastModel.for_inference(ft_model)

    ft_responses = {}
    for item in EVAL_PROMPTS:
        print(f"  [ft]   {item['id']}", flush=True)
        ft_responses[item["id"]] = run_inference(ft_model, tokenizer, item["user"], args.max_tokens)

    del ft_model
    gc.collect()
    torch.cuda.empty_cache()

    # ── Evaluate ─────────────────────────────────────────────────────────────
    for item in EVAL_PROMPTS:
        base_resp = base_responses[item["id"]]
        ft_resp   = ft_responses[item["id"]]
        base_score = score(base_resp, item["rubric"])
        ft_score   = score(ft_resp,   item["rubric"])
        results.append({
            "task": item["id"],
            "base_score": base_score,
            "ft_score": ft_score,
            "delta": ft_score - base_score,
        })
        print(f"  Base: {base_score:.0%}  FT: {ft_score:.0%}  Δ={ft_score-base_score:+.0%}")

    # ── Print markdown table ──────────────────────────────────────────────────
    print("\n\n## EduEdge Fine-tuning Evaluation\n")
    print("| Task | Base (Gemma 4 E4B) | Fine-tuned | Δ |")
    print("|------|-------------------|------------|---|")
    for r in results:
        print(f"| {r['task']} | {r['base_score']:.0%} | {r['ft_score']:.0%} | {r['delta']:+.0%} |")
    avg_delta = sum(r["delta"] for r in results) / len(results)
    print(f"\n**Average improvement: {avg_delta:+.0%}**")

    # Save JSON
    out = Path(args.adapter) / "eval_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved → {out}")


if __name__ == "__main__":
    main()
