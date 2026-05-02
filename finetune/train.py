"""
train.py — QLoRA fine-tuning of Gemma 4 with Unsloth for EduEdge.

Target hardware : NVIDIA RTX 4070 SUPER 12 GB VRAM
Model           : Gemma 4 E4B (4.5B effective params) in 4-bit NF4
Method          : QLoRA via Unsloth FastModel
Dataset         : data/eduedge_train.jsonl  (built by finetune/dataset.py)

Usage:
    python finetune/train.py [--steps 300] [--output runs/eduedge-v1]
"""

import argparse
import json
import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Args
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",   default="unsloth/gemma-4-E4B-it-unsloth-bnb-4bit",
                   help="Unsloth model hub name")
    p.add_argument("--data",    default="data/eduedge_train.jsonl")
    p.add_argument("--output",  default="runs/eduedge-v1")
    p.add_argument("--steps",   type=int, default=300,
                   help="Training steps (300 ≈ 30 min on RTX 4070 SUPER)")
    p.add_argument("--lora_r",  type=int, default=8)
    p.add_argument("--batch",   type=int, default=1)
    p.add_argument("--grad_acc",type=int, default=4)
    p.add_argument("--lr",      type=float, default=2e-4)
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--export_gguf", action="store_true",
                   help="Export to GGUF (Q4_K_M) after training")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    args = parse_args()

    # ── 1. Imports (deferred so --help works without GPU) ───────────────────
    from unsloth import FastModel
    from unsloth.chat_templates import get_chat_template
    from datasets import Dataset
    import torch
    from trl import SFTTrainer, SFTConfig

    # Help CUDA memory allocator avoid fragmentation
    import os
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── 2. Load model & tokenizer ────────────────────────────────────────────
    print(f"\nLoading {args.model} …")
    model, tokenizer = FastModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.seq_len,
        load_in_4bit=True,
        load_in_8bit=False,
        full_finetuning=False,
        device_map={"": 0},   # force all layers to GPU 0
    )

    # Apply LoRA
    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=False,   # text-only fine-tune
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.0,
        bias="none",
        random_state=42,
    )

    # Apply Gemma 4 chat template
    tokenizer = get_chat_template(tokenizer, chat_template="gemma-4")

    # ── 3. Load & format dataset ─────────────────────────────────────────────
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(
            f"{data_path} not found — run: python finetune/dataset.py"
        )

    rows = []
    for i, l in enumerate(data_path.read_text().splitlines()):
        l = l.strip()
        if not l:
            continue
        try:
            rows.append(json.loads(l))
        except json.JSONDecodeError as e:
            print(f"  Skipping malformed line {i}: {e}")
    print(f"Loaded {len(rows)} training examples")

    def apply_template(batch):
        texts = tokenizer.apply_chat_template(
            batch["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": texts}

    dataset = Dataset.from_list(rows)
    dataset = dataset.map(apply_template, batched=True)

    # ── 4. Training config ───────────────────────────────────────────────────
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=args.batch,
            gradient_accumulation_steps=args.grad_acc,
            warmup_ratio=0.05,
            max_steps=args.steps,
            learning_rate=args.lr,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=10,
            save_steps=100,
            save_total_limit=2,
            output_dir=str(out_dir),
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            gradient_checkpointing=True,
            report_to="none",
            seed=42,
        ),
    )

    # ── 5. Train ─────────────────────────────────────────────────────────────
    print(f"\nStarting training: {args.steps} steps …")
    gpu_stats_before = torch.cuda.memory_reserved() / 1e9
    trainer.train()
    gpu_peak = torch.cuda.max_memory_reserved() / 1e9
    print(f"\nPeak VRAM: {gpu_peak:.1f} GB  (reserved before: {gpu_stats_before:.1f} GB)")

    # ── 6. Save LoRA adapter ─────────────────────────────────────────────────
    adapter_dir = out_dir / "lora_adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"LoRA adapter saved → {adapter_dir}")

    # ── 7. Optional GGUF export (for Ollama) ─────────────────────────────────
    if args.export_gguf:
        gguf_dir = out_dir / "gguf"
        gguf_dir.mkdir(exist_ok=True)
        print(f"\nExporting merged GGUF (Q4_K_M) → {gguf_dir} …")
        model.save_pretrained_gguf(
            str(gguf_dir),
            tokenizer,
            quantization_method="q4_k_m",
        )
        print("GGUF export complete!")
        print("\nTo load in Ollama, create a Modelfile:")
        print(f"  FROM {gguf_dir}/unsloth.Q4_K_M.gguf")
        print("  SYSTEM <your system prompt>")
        print("Then: ollama create eduedge-v1 -f Modelfile")


if __name__ == "__main__":
    main()
