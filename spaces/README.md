---
title: EduEdge — Offline AI Tutor (Gemma 4 E4B)
emoji: 🎓
colorFrom: purple
colorTo: indigo
sdk: gradio
sdk_version: "5.29.0"
app_file: app.py
pinned: true
license: apache-2.0
models:
  - google/gemma-4-31B-it
  - dorus4/eduedge-gemma4-lora
datasets:
  - dorus4/eduedge-gemma4-lora
tags:
  - gemma4
  - education
  - multilingual
  - function-calling
  - vision
  - unsloth
  - offline
---

# EduEdge — Offline AI Tutor

**The Gemma 4 Good Hackathon** submission by [@Hastws](https://huggingface.co/Hastws)

EduEdge is an offline-first, multimodal AI tutor powered by **Gemma 4 E4B** — designed for the 300 million students worldwide without reliable internet access.

## Features

- 📷 **Vision** — upload a photo of a textbook, diagram, or handwritten note
- 🔧 **Function Calling** — auto-generates quizzes, study plans, concept explanations
- 🌍 **140+ Languages** — ask in any language, get answers in any language
- 🧠 **Thinking Mode** — chain-of-thought reasoning for hard problems
- 📶 **Offline-first** — runs entirely locally via Ollama (this Space = cloud demo)

## Fine-tuned Adapter

The EduEdge QLoRA adapter (fine-tuned on Gemma 4 E4B) is available at:  
👉 **[dorus4/eduedge-gemma4-lora](https://huggingface.co/dorus4/eduedge-gemma4-lora)**

- Base model: `unsloth/gemma-4-E4B-it-unsloth-bnb-4bit`
- Rank r=8, α=16 · 18.35M trainable params (0.23%)
- Train loss: 9.12 → 0.92 · VRAM: 11.7 GB peak

## Setup note

> **Note on models:** This Space uses `google/gemma-4-31B-it` via the HF Serverless Inference API to demonstrate the full application (multimodal input, function calling, multilingual UI). The fine-tuned `dorus4/eduedge-gemma4-lora` adapter (E4B base) is the training artifact — it runs locally via Ollama since the Serverless API does not support dynamic LoRA loading. Both are listed in this Space's model tags for discoverability.

Set your `HF_TOKEN` secret in Space settings for authenticated model access.

## Local setup (offline mode)

```bash
git clone https://github.com/Hastws/eduedge-gemma4
cd eduedge-gemma4
pip install -r requirements.txt
ollama pull gemma4:e4b       # or use the fine-tuned model
python3 -m src.app
```
