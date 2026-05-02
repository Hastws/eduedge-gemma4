"""
app.py — Gradio web interface for EduEdge.

Run:
    python -m src.app
"""

import os
from pathlib import Path

# Must be set BEFORE any httpx/gradio imports so localhost bypasses macOS proxy.
for _k in ("no_proxy", "NO_PROXY"):
    _cur = os.environ.get(_k, "")
    _bypass = "localhost,127.0.0.1,0.0.0.0,::1"
    os.environ[_k] = f"{_cur},{_bypass}" if _cur else _bypass

from typing import Generator

import gradio as gr
from dotenv import load_dotenv

from .tutor import EduTutor, GEMMA_MODEL

load_dotenv()

# --------------------------------------------------------------------------- #
# App state
# --------------------------------------------------------------------------- #

tutor = EduTutor()

GRADE_LEVELS = ["elementary", "middle", "high_school", "university"]
LANGUAGES = {
    "English": "en", "中文": "zh", "Español": "es", "Français": "fr",
    "العربية": "ar", "हिन्दी": "hi", "Português": "pt", "Kiswahili": "sw",
    "नेपाली": "ne", "Bahasa Indonesia": "id",
}

# --------------------------------------------------------------------------- #
# Custom CSS
# --------------------------------------------------------------------------- #

CSS = """
/* ── Base ───────────────────────────────────────────────────────────────── */
:root {
    --bg:        #0d1117;
    --bg-card:   #161b22;
    --bg-hover:  #1c2128;
    --border:    #30363d;
    --accent:    #6e40c9;
    --accent-2:  #388bfd;
    --accent-glow: rgba(110,64,201,0.3);
    --text:      #e6edf3;
    --text-muted: #8b949e;
    --radius:    12px;
    --radius-sm: 8px;
}

html, body {
    background: var(--bg) !important;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    color: var(--text) !important;
    margin: 0 !important;
    padding: 0 !important;
}

/* Full-width container, dark background edge-to-edge */
.gradio-container {
    background: var(--bg) !important;
    max-width: 100% !important;
    width: 100% !important;
    padding: 0 !important;
    margin: 0 !important;
}

/* Gradio 6 inner wrappers — strip default max-width/margin */
.gradio-container .main.fillable,
.gradio-container .wrap.svelte-zxu34v,
.gradio-container main.contain {
    max-width: 100% !important;
    width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
    box-sizing: border-box !important;
}

/* Centered content wrapper — cards cap at 1400px, auto-centered */
#app-header,
#main-row {
    max-width: 1400px !important;
    margin-left: auto !important;
    margin-right: auto !important;
    box-sizing: border-box !important;
}

/* Extra side padding at very wide screens so content doesn't touch edges */
@media (min-width: 1440px) {
    #app-header,
    #main-row {
        max-width: 1400px !important;
    }
}

/* ── Header ─────────────────────────────────────────────────────────────── */
#app-header {
    background: linear-gradient(135deg, #1a1040 0%, #161b22 50%, #0d1117 100%);
    border-bottom: 1px solid var(--border);
    border-radius: var(--radius) var(--radius) 0 0;
    padding: 28px 32px 24px;
    margin-bottom: 0;
    position: relative;
    overflow: hidden;
}

#app-header::before {
    content: '';
    position: absolute;
    top: -80px; left: -80px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(110,64,201,0.18) 0%, transparent 70%);
    pointer-events: none;
}

#app-header::after {
    content: '';
    position: absolute;
    bottom: -60px; right: 60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(56,139,253,0.12) 0%, transparent 70%);
    pointer-events: none;
}

.header-logo {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 10px;
}

.header-logo-icon {
    font-size: 40px;
    line-height: 1;
}

.header-title {
    font-size: 28px;
    font-weight: 700;
    background: linear-gradient(90deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.5px;
    margin: 0;
}

.header-sub {
    font-size: 14px;
    color: var(--text-muted);
    margin: 2px 0 0 0;
}

.header-badges {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 14px;
}

.badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 12px;
    font-weight: 500;
    color: var(--text-muted);
    white-space: nowrap;
}

.badge .dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #3fb950;
    box-shadow: 0 0 6px #3fb950;
}

/* ── Main layout ─────────────────────────────────────────────────────────── */
#main-row {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-top: none !important;
    border-radius: 0 0 var(--radius) var(--radius) !important;
    padding: 20px !important;
    gap: 20px !important;
    flex-wrap: nowrap !important;   /* prevent Gradio from stacking columns */
    align-items: flex-start !important;
}

/* ── Chat column ─────────────────────────────────────────────────────────── */
#chat-col {
    display: flex !important;
    flex-direction: column !important;
    gap: 12px !important;
    min-width: 0 !important;
    flex: 4 1 0% !important;
}

#sidebar-col {
    min-width: 300px !important;
    max-width: 360px !important;
    flex: 1 0 300px !important;
}

/* Chatbot */
#chatbot {
    border-radius: var(--radius) !important;
    border: 1px solid var(--border) !important;
    background: var(--bg) !important;
}

#chatbot .message {
    border-radius: var(--radius-sm) !important;
}

/* ── Input bar ───────────────────────────────────────────────────────────── */
#input-row {
    display: flex !important;
    flex-direction: row !important;
    align-items: flex-end !important;
    flex-wrap: nowrap !important;
    gap: 10px !important;
    background: var(--bg) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 8px 8px 8px 14px !important;
    transition: border-color 0.2s;
    --block-background-fill: transparent !important;
}

#input-row:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-glow) !important;
}

/* Kill the gray .form wrapper Gradio wraps each Row child in */
#input-row > .form {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    flex: 1 1 0% !important;
    min-width: 0 !important;
}
#msg-box {
    flex: 1 1 0% !important;
    min-width: 0 !important;
    --block-background-fill: transparent !important;
}

#msg-box .block,
#msg-box .wrap,
#msg-box > div {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    min-height: unset !important;
}

/* Textbox block inside input-row */
#msg-box {
    flex: 1 1 0% !important;
    min-width: 0 !important;
    --block-background-fill: transparent !important;
}

#msg-box .block,
#msg-box .wrap,
#msg-box > div {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    min-height: unset !important;
}

/* Hide label text but keep textarea (do NOT use display:none on label) */
#msg-box label .label-wrap,
#msg-box label > span:first-child,
#msg-box .svelte-jdcl7l {
    display: none !important;
}

/* Force show label + wrap that Gradio hides */
#msg-box label.container,
#msg-box .wrap {
    display: flex !important;
    flex-direction: column !important;
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    margin: 0 !important;
}

#msg-box textarea {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: var(--text) !important;
    font-size: 15px !important;
    padding: 4px 0 !important;
    resize: none !important;
    min-height: 44px !important;
}

#msg-box textarea::placeholder {
    color: var(--text-muted) !important;
}

#msg-box label,
#msg-box .label-wrap { display: none !important; }

#send-btn {
    background: linear-gradient(135deg, #6e40c9, #388bfd) !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    color: white !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 10px 20px !important;
    min-width: 90px !important;
    height: 42px !important;
    transition: opacity 0.2s, transform 0.1s !important;
    flex-shrink: 0 !important;
}

#send-btn:hover { opacity: 0.88 !important; transform: translateY(-1px) !important; }
#send-btn:active { transform: translateY(0) !important; }

/* ── Image upload ────────────────────────────────────────────────────────── */
#image-upload,
#image-upload .wrap,
#image-upload .block,
#image-upload .upload-container,
#image-upload > div,
#image-upload .label-wrap,
#image-upload span,
#image-upload button {
    background: var(--bg) !important;
    border-color: var(--border) !important;
    color: var(--text-muted) !important;
}

#image-upload {
    border: 1px dashed var(--border) !important;
    border-radius: var(--radius) !important;
    overflow: hidden !important;
    transition: border-color 0.2s !important;
}

#image-upload:hover {
    border-color: var(--accent) !important;
}

/* White label header bar fix — Gradio renders a <label class="svelte-19djge9 float"> */
#image-upload label,
#image-upload .label-wrap,
#image-upload [data-testid="label"],
#image-upload .svelte-1b8vnkg,
#image-upload .svelte-19djge9 {
    background: var(--bg-card) !important;
    border-bottom: 1px solid var(--border) !important;
    padding: 6px 12px !important;
    color: var(--text-muted) !important;
}

/* Also fix top icon-button-wrapper white bar */
#image-upload .icon-button-wrapper {
    background: var(--bg) !important;
}

/* ── Clear button ────────────────────────────────────────────────────────── */
#clear-btn {
    background: transparent !important;
    border: 1px solid var(--border) !important;
    color: var(--text-muted) !important;
    border-radius: var(--radius-sm) !important;
    font-size: 12px !important;
    transition: all 0.2s !important;
}

#clear-btn:hover {
    border-color: #f85149 !important;
    color: #f85149 !important;
    background: rgba(248,81,73,0.08) !important;
}

/* ── Sidebar ─────────────────────────────────────────────────────────────── */
#sidebar-col {
    display: flex !important;
    flex-direction: column !important;
    gap: 0 !important;
    /* Override Gradio 6 CSS variables */
    --block-background-fill: var(--bg) !important;
    --panel-background-fill: var(--bg) !important;
    --block-border-color: var(--border) !important;
    --block-label-background-fill: var(--bg-card) !important;
}

/* Kill ALL Gradio block/form backgrounds in sidebar */
#sidebar-col .block,
#sidebar-col .form,
#sidebar-col fieldset,
#sidebar-col .gr-group,
#sidebar-col > div,
#sidebar-col > div > div {
    background: var(--bg) !important;
    border-color: var(--border) !important;
    box-shadow: none !important;
}

/* Visual card styling via HTML sidebar-card class */
.sidebar-card {
    background: var(--bg) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 14px 16px !important;
    margin-bottom: 12px !important;
}

.sidebar-section-header {
    padding: 4px 2px 4px !important;
}

.card-title {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-muted);
    margin: 0 0 8px 0;
}

/* Dropdowns */
#grade-sel, #lang-sel {
    background: var(--bg) !important;
    --block-background-fill: var(--bg) !important;
}

#grade-sel .block, #lang-sel .block,
#grade-sel input, #lang-sel input,
#grade-sel .wrap, #lang-sel .wrap {
    background: var(--bg-card) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
    border-radius: var(--radius-sm) !important;
}

/* Example prompt buttons */
.example-btn {
    display: block !important;
    width: 100% !important;
    text-align: left !important;
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text) !important;
    font-size: 13px !important;
    padding: 9px 13px !important;
    margin-bottom: 6px !important;
    cursor: pointer !important;
    transition: background 0.15s, border-color 0.15s !important;
    white-space: normal !important;
    line-height: 1.4 !important;
    box-shadow: none !important;
}

.example-btn:hover {
    background: var(--bg-hover) !important;
    border-color: var(--accent) !important;
    color: var(--text) !important;
}

/* ── Tool chips ──────────────────────────────────────────────────────────── */
.tool-chips {
    display: flex;
    flex-direction: column;
    gap: 8px;
}

.tool-chip {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 9px 12px;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    font-size: 13px;
    color: var(--text);
}

.tool-chip-icon {
    font-size: 16px;
    flex-shrink: 0;
}

.tool-chip-name {
    font-weight: 500;
    font-size: 12px;
    color: #a78bfa;
}

.tool-chip-desc {
    font-size: 11px;
    color: var(--text-muted);
    margin-top: 1px;
}

/* ── Misc ────────────────────────────────────────────────────────────────── */
.gradio-container label {
    color: var(--text-muted) !important;
    font-size: 12px !important;
    font-weight: 500 !important;
}

footer { display: none !important; }
"""

# --------------------------------------------------------------------------- #
# Header HTML
# --------------------------------------------------------------------------- #

def _header_html() -> str:
    return f"""
<div id="app-header">
  <div class="header-logo">
    <span class="header-logo-icon">🎓</span>
    <div>
      <p class="header-title">EduEdge</p>
      <p class="header-sub">Offline AI Tutor — runs entirely on your device, no cloud required</p>
    </div>
  </div>
  <div class="header-badges">
    <span class="badge"><span class="dot"></span>Live</span>
    <span class="badge">🤖 {GEMMA_MODEL}</span>
    <span class="badge">👁 Vision</span>
    <span class="badge">⚡ Function Calling</span>
    <span class="badge">🌍 140+ Languages</span>
    <span class="badge">📖 128K Context</span>
  </div>
</div>
"""

# --------------------------------------------------------------------------- #
# Gradio handlers
# --------------------------------------------------------------------------- #

def respond(
    message: str,
    image: str | None,
    chat_history: list[dict],
    grade: str,
    lang_label: str,
) -> Generator:
    """Streaming generator — yields (cleared_input, updated_history) on each token."""
    if not message.strip() and image is None:
        yield "", chat_history
        return

    lang_code = LANGUAGES.get(lang_label, "en")
    context_prefix = ""
    if lang_code != "en":
        context_prefix += f"[Respond in language: {lang_code}] "
    if grade != "middle":
        context_prefix += f"[Student level: {grade}] "
    full_message = context_prefix + message

    chat_history = list(chat_history)
    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": ""})
    yield "", chat_history

    for chunk in tutor.chat(full_message, image_path=image):
        chat_history[-1]["content"] += chunk
        yield "", chat_history


def clear_session() -> tuple[list, None, str]:
    tutor.reset()
    return [], None, ""


# --------------------------------------------------------------------------- #
# UI layout
# --------------------------------------------------------------------------- #

EXAMPLES = [
    ["Explain photosynthesis like I'm 12 years old", None, "middle", "English"],
    ["What is the quadratic formula and when do I use it?", None, "high_school", "English"],
    ["Create a 5-day study plan for my biology exam next week", None, "high_school", "English"],
    ["Make me a 3-question quiz on the French Revolution", None, "high_school", "English"],
    ["¿Puedes explicarme la ley de Newton en español?", None, "middle", "Español"],
    ["请用中文解释勾股定理，并出3道题考考我", None, "middle", "中文"],
]

TOOLS_HTML = """
<div class="sidebar-card">
  <p class="card-title">🛠 Available Tools</p>
  <div class="tool-chips">
    <div class="tool-chip">
      <span class="tool-chip-icon">📝</span>
      <div>
        <div class="tool-chip-name">generate_quiz</div>
        <div class="tool-chip-desc">Multiple-choice questions</div>
      </div>
    </div>
    <div class="tool-chip">
      <span class="tool-chip-icon">📅</span>
      <div>
        <div class="tool-chip-name">create_study_plan</div>
        <div class="tool-chip-desc">Day-by-day schedule</div>
      </div>
    </div>
    <div class="tool-chip">
      <span class="tool-chip-icon">💡</span>
      <div>
        <div class="tool-chip-name">explain_concept</div>
        <div class="tool-chip-desc">Structured breakdown</div>
      </div>
    </div>
  </div>
</div>
"""


def build_app() -> gr.Blocks:
    with gr.Blocks(
        title="EduEdge — Gemma 4 Offline Tutor",
    ) as demo:

        gr.HTML(_header_html())

        with gr.Row(elem_id="main-row"):
            # ── Left: chat ───────────────────────────────────────────────────
            with gr.Column(scale=4, elem_id="chat-col"):
                chatbot = gr.Chatbot(
                    elem_id="chatbot",
                    label="",
                    height=500,
                    avatar_images=(
                        None,
                        "https://em-content.zobj.net/source/twitter/348/books_1f4da.png",
                    ),
                    render_markdown=True,
                    show_label=False,
                    placeholder=(
                        "<div style='text-align:center;padding:40px 20px;color:#8b949e'>"
                        "<div style='font-size:48px;margin-bottom:16px'>🎓</div>"
                        "<div style='font-size:18px;font-weight:600;color:#e6edf3;margin-bottom:8px'>Welcome to EduEdge</div>"
                        "<div style='font-size:14px;line-height:1.6'>Your offline AI tutor — ask a question,<br>request a quiz, or upload a photo of your notes.</div>"
                        "</div>"
                    ),
                )

                with gr.Row(elem_id="input-row"):
                    msg_box = gr.Textbox(
                        elem_id="msg-box",
                        placeholder="Ask a question, request a quiz, or upload a photo of your notes…",
                        label=" ",
                        lines=2,
                        max_lines=6,
                        scale=5,
                    )
                    send_btn = gr.Button(
                        "Send ↑",
                        elem_id="send-btn",
                        variant="primary",
                        scale=0,
                    )

                image_input = gr.Image(
                    elem_id="image-upload",
                    label="📷 Attach image (textbook, notes, diagram…)",
                    type="filepath",
                    sources=["upload", "webcam"],
                    height=160,
                )

                clear_btn = gr.Button(
                    "✕  Clear conversation",
                    elem_id="clear-btn",
                    variant="secondary",
                    size="sm",
                )

            # ── Right: sidebar ───────────────────────────────────────────────
            with gr.Column(scale=1, min_width=300, elem_id="sidebar-col"):

                gr.HTML('<div class="sidebar-card"><p class="card-title">⚙️ Settings</p></div>')
                grade_sel = gr.Dropdown(
                    elem_id="grade-sel",
                    choices=GRADE_LEVELS,
                    value="middle",
                    label="Grade level",
                    info="Adjusts explanation depth",
                )
                lang_sel = gr.Dropdown(
                    elem_id="lang-sel",
                    choices=list(LANGUAGES.keys()),
                    value="English",
                    label="Response language",
                )

                gr.HTML('<div class="sidebar-section-header"><p class="card-title">⚡ Quick Examples</p></div>')
                with gr.Column(elem_id="examples-col"):
                    ex_btns = []
                    for ex in EXAMPLES:
                        btn = gr.Button(ex[0], elem_classes=["example-btn"], size="sm")
                        ex_btns.append((btn, ex))

                gr.HTML(TOOLS_HTML)

        # ── Events ────────────────────────────────────────────────────────────
        inputs = [msg_box, image_input, chatbot, grade_sel, lang_sel]
        outputs = [msg_box, chatbot]

        send_btn.click(respond, inputs=inputs, outputs=outputs)
        msg_box.submit(respond, inputs=inputs, outputs=outputs)
        clear_btn.click(clear_session, outputs=[chatbot, image_input, msg_box])

        for btn, ex in ex_btns:
            btn.click(
                fn=lambda q=ex[0], g=ex[2], l=ex[3]: (q, g, l),
                outputs=[msg_box, grade_sel, lang_sel],
            )

    return demo


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    app = build_app()
    font_link = '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap">'
    app.launch(
        server_name=os.getenv("APP_HOST", "0.0.0.0"),
        server_port=int(os.getenv("APP_PORT", 7860)),
        share=False,
        head=font_link,
        css=CSS,
    )
