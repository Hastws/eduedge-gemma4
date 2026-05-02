"""
dataset.py — Build an educational Q&A fine-tuning dataset for EduEdge.

Sources (all open-license, auto-downloaded from HuggingFace):
  1. sciq          — 13K crowdsourced science QA with supporting context
  2. openbookqa    — 5.9K elementary science questions
  3. HuggingFaceH4/no_robots — 9.5K high-quality instruction-following

Output: data/eduedge_train.jsonl  (Gemma 4 chat format)
"""

import json
import random
from pathlib import Path

from datasets import load_dataset, concatenate_datasets, Dataset

OUT_DIR = Path(__file__).parent.parent / "data"
OUT_DIR.mkdir(exist_ok=True)
OUT_FILE = OUT_DIR / "eduedge_train.jsonl"

# Gemma 4 chat format system prompt (same as tutor.py)
SYSTEM = (
    "You are EduEdge, a patient and encouraging AI tutor. "
    "Explain concepts clearly, use analogies, and adapt to the student's level. "
    "Format answers with Markdown headers and bullet points."
)


# --------------------------------------------------------------------------- #
# Formatters
# --------------------------------------------------------------------------- #

def format_sciq(example: dict) -> dict | None:
    """Science QA → structured explanation + correct answer."""
    q = example.get("question", "").strip()
    support = example.get("support", "").strip()
    answer = example.get("correct_answer", "").strip()
    if not q or not answer:
        return None

    context = f"\n\n*Background: {support}*" if support else ""
    user_msg = f"Explain and answer this science question:\n\n**{q}**{context}"
    assistant_msg = (
        f"### Answer\n**{answer}**\n\n"
        f"### Explanation\n{support or 'Let me explain this concept step by step.'}\n\n"
        "### Key Takeaway\nRemember to think about the underlying principles when solving similar questions!"
    )
    return _make_row(user_msg, assistant_msg)


def format_openbookqa(example: dict) -> dict | None:
    """OpenBookQA → multiple-choice with explanation."""
    stem = example.get("question_stem", "").strip()
    choices = example.get("choices", {})
    label = example.get("answerKey", "")
    if not stem or not choices or not label:
        return None

    labels = choices.get("label", [])
    texts = choices.get("text", [])
    opts = "\n".join(f"{l}) {t}" for l, t in zip(labels, texts))
    correct_text = dict(zip(labels, texts)).get(label, "")

    user_msg = f"**Question:** {stem}\n\nChoices:\n{opts}\n\nWhat is the correct answer and why?"
    assistant_msg = (
        f"### ✅ Answer: {label}) {correct_text}\n\n"
        "### Explanation\nLet's think through this carefully:\n\n"
        f"The correct answer is **{label}) {correct_text}** because it best fits "
        "the scientific principles involved. The other options are plausible but "
        "don't fully satisfy the conditions of the question.\n\n"
        "💡 *Tip: Always re-read the question stem carefully before choosing!*"
    )
    return _make_row(user_msg, assistant_msg)


def format_no_robots(example: dict) -> dict | None:
    """No-robots instruction-following → general tutoring conversation."""
    messages = example.get("messages", [])
    if len(messages) < 2:
        return None
    user_turn = next((m["content"] for m in messages if m["role"] == "user"), None)
    asst_turn = next((m["content"] for m in messages if m["role"] == "assistant"), None)
    if not user_turn or not asst_turn:
        return None
    # Only keep education-adjacent categories
    cat = example.get("category", "")
    if cat not in ("Open QA", "Brainstorm", "Summarize", "Explain", ""):
        return None
    return _make_row(user_turn.strip(), asst_turn.strip())


def _make_row(user: str, assistant: str) -> dict:
    """Wrap in Gemma 4 multi-turn chat format."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user",   "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def build_dataset(max_examples: int = 8000, seed: int = 42) -> None:
    random.seed(seed)
    rows: list[dict] = []

    print("Loading SciQ …")
    sciq = load_dataset("sciq", split="train+validation")
    for ex in sciq:
        r = format_sciq(ex)
        if r:
            rows.append(r)

    print("Loading OpenBookQA …")
    obqa = load_dataset("openbookqa", "main", split="train+validation")
    for ex in obqa:
        r = format_openbookqa(ex)
        if r:
            rows.append(r)

    print("Loading HuggingFaceH4/no_robots …")
    try:
        nr = load_dataset("HuggingFaceH4/no_robots", split="train")
        for ex in nr:
            r = format_no_robots(ex)
            if r:
                rows.append(r)
    except Exception as e:
        print(f"  Skipping no_robots: {e}")

    random.shuffle(rows)
    rows = rows[:max_examples]
    print(f"Total examples: {len(rows)}")

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for row in rows:
            # ensure_ascii=True avoids any encoding issues in JSONL
            line = json.dumps(row, ensure_ascii=True)
            # Sanity check: must be parseable
            try:
                json.loads(line)
            except json.JSONDecodeError:
                continue
            f.write(line + "\n")

    print(f"Saved → {OUT_FILE}")


if __name__ == "__main__":
    build_dataset()
