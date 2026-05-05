"""
EduEdge — HuggingFace Spaces version
Uses HF Serverless Inference API (InferenceClient) — no local GPU required.
"""

import base64
import io
import json
import os
import re
from typing import Optional

import gradio as gr
from huggingface_hub import InferenceClient
from PIL import Image

from tools import TOOLS, TOOL_REGISTRY

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_ID = "google/gemma-4-31B-it"  # Gemma 4 E4B not yet on Serverless API; 31B is the available Gemma 4
HF_TOKEN = os.getenv("HF_TOKEN")
MAX_NEW_TOKENS = 1024

client = InferenceClient(model=MODEL_ID, token=HF_TOKEN)

# ── Tool schema for system prompt ──────────────────────────────────────────────
_TOOL_SCHEMA = json.dumps([t["function"] for t in TOOLS], indent=2)

SYSTEM_PROMPT = f"""\
You are EduEdge, a patient and encouraging AI tutor built for students in \
underserved communities — including those with no internet access.

Your capabilities:
• Explain any concept clearly at the student's level using simple language
• Analyse photos of textbooks, handwritten notes, or diagrams
• Generate quizzes, study plans, and structured explanations via tools
• Communicate in any language the student prefers

Always be warm, encouraging, and celebrate student effort. Use Markdown formatting.

--- TOOLS ---
You have access to the following tools. To call a tool, output EXACTLY:
<tool_call>
{{"name": "<tool_name>", "arguments": {{...}}}}
</tool_call>
Then stop. The result will be returned to you so you can complete your answer.

Available tools:
{_TOOL_SCHEMA}
"""


# ── Helpers ────────────────────────────────────────────────────────────────────
def _img_to_b64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def _parse_tool_call(text: str) -> tuple[Optional[str], Optional[dict]]:
    m = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(1))
            return data.get("name"), data.get("arguments", {})
        except json.JSONDecodeError:
            pass
    return None, None


def _call_model(messages: list) -> str:
    response = client.chat_completion(
        messages=messages,
        max_tokens=MAX_NEW_TOKENS,
        temperature=0.7,
        top_p=0.95,
    )
    return response.choices[0].message.content or ""


# ── Main inference ─────────────────────────────────────────────────────────────
def chat(
    message: str,
    history: list,
    image: Optional[Image.Image],
    grade: str,
    language: str,
) -> str:
    msgs: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Add history
    for pair in (history or []):
        if isinstance(pair, (list, tuple)) and len(pair) == 2:
            u, b = pair
            msgs.append({"role": "user", "content": str(u)})
            if b:
                msgs.append({"role": "assistant", "content": str(b)})

    user_text = f"[Grade level: {grade} | Preferred language: {language}]\n\n{message}"

    if image is not None:
        b64 = _img_to_b64(image)
        user_content = [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            {"type": "text", "text": user_text},
        ]
    else:
        user_content = user_text

    msgs.append({"role": "user", "content": user_content})

    # First model call
    response = _call_model(msgs)

    # Handle tool call
    fn_name, fn_args = _parse_tool_call(response)
    if fn_name and fn_name in TOOL_REGISTRY:
        fn = TOOL_REGISTRY[fn_name]
        tool_out = fn(**(fn_args or {}))
        msgs.append({"role": "assistant", "content": response})
        msgs.append({
            "role": "user",
            "content": (
                f"Tool `{fn_name}` returned:\n```json\n{tool_out}\n```\n\n"
                "Now write your full, formatted response to the student based on this result."
            ),
        })
        response = _call_model(msgs)

    return response


# ── Gradio UI ──────────────────────────────────────────────────────────────────
CSS = """
body, .gradio-container { background: #0d1117 !important; color: #e6edf3 !important; }
.gr-button-primary { background: linear-gradient(135deg,#7c3aed,#4f46e5) !important; border: none !important; }
footer { display: none !important; }
"""

EXAMPLES = [
    ["Explain photosynthesis to me", None, "middle", "en"],
    ["Make me a 3-question quiz on the French Revolution", None, "high_school", "en"],
    ["Create a 5-day study plan for algebra", None, "middle", "en"],
    ["¿Puedes explicarme la fotosíntesis en español?", None, "middle", "es"],
    ["用中文解释牛顿第二定律", None, "high_school", "zh"],
]

with gr.Blocks(css=CSS, title="EduEdge — AI Tutor") as demo:
    gr.HTML("""
    <div style="text-align:center;padding:24px 0 8px">
      <h1 style="font-size:2rem;font-weight:700;margin:0">
        🎓 EduEdge <span style="font-size:1rem;font-weight:400;color:#8b949e">— Offline AI Tutor</span>
      </h1>
      <p style="color:#8b949e;margin:8px 0 0">
        Powered by <b>Gemma 4 31B</b> · Vision · Function Calling · 140+ Languages
      </p>
      <div style="margin-top:10px;display:flex;gap:8px;justify-content:center;flex-wrap:wrap">
        <span style="background:#21262d;border:1px solid #30363d;border-radius:20px;padding:3px 12px;font-size:.75rem">🔧 Function Calling</span>
        <span style="background:#21262d;border:1px solid #30363d;border-radius:20px;padding:3px 12px;font-size:.75rem">📷 Vision</span>
        <span style="background:#21262d;border:1px solid #30363d;border-radius:20px;padding:3px 12px;font-size:.75rem">🌍 140+ Languages</span>
        <span style="background:#21262d;border:1px solid #30363d;border-radius:20px;padding:3px 12px;font-size:.75rem">📶 Offline-First</span>
      </div>
    </div>
    """)

    chatbot = gr.Chatbot(height=480, show_label=False)

    with gr.Row():
        with gr.Column(scale=3):
            msg_box = gr.Textbox(
                placeholder="Ask me anything — or upload a photo of your textbook...",
                show_label=False,
                lines=2,
            )
        with gr.Column(scale=1, min_width=120):
            send_btn = gr.Button("Send ✈", variant="primary")

    with gr.Accordion("⚙ Settings", open=False):
        with gr.Row():
            grade_sel = gr.Dropdown(
                choices=["elementary", "middle", "high_school", "university"],
                value="middle",
                label="Grade Level",
            )
            lang_sel = gr.Dropdown(
                choices=["en", "es", "zh", "fr", "ar", "pt", "hi", "sw", "bn", "other"],
                value="en",
                label="Language",
            )
        image_up = gr.Image(type="pil", label="📷 Upload image (optional)", height=180)

    gr.Examples(
        examples=EXAMPLES,
        inputs=[msg_box, image_up, grade_sel, lang_sel],
        label="Try an example",
    )

    def respond(message, history, image, grade, language):
        if not message.strip():
            return history, ""
        try:
            bot_reply = chat(message, history, image, grade, language)
        except Exception as e:
            bot_reply = f"⚠️ Error: {e}\n\nThe model may be unavailable or rate-limited. Try again in a moment."
        history = history + [[message, bot_reply]]
        return history, ""

    send_btn.click(
        respond,
        inputs=[msg_box, chatbot, image_up, grade_sel, lang_sel],
        outputs=[chatbot, msg_box],
    )
    msg_box.submit(
        respond,
        inputs=[msg_box, chatbot, image_up, grade_sel, lang_sel],
        outputs=[chatbot, msg_box],
    )

if __name__ == "__main__":
    demo.launch()
