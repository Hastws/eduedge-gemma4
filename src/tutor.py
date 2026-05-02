"""
tutor.py — Core EduEdge tutor logic powered by Gemma 4 via Ollama.

Supports:
  - Real token-by-token streaming via Ollama's streaming API
  - Agentic tool-calling loop (quiz, study plan, concept explanation)
  - Image/multimodal input
  - Multi-turn conversation history
  - Gemma 4 thinking mode
"""

import base64
import os
import re
from pathlib import Path
from typing import Generator

import httpx
import ollama
from ollama import Options

from .tools import TOOLS, dispatch

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
GEMMA_MODEL = os.getenv("GEMMA_MODEL", "gemma4:e4b")

DEFAULT_OPTIONS = Options(
    temperature=1.0,
    top_p=0.95,
    top_k=64,
)

SYSTEM_PROMPT = """\
<|think|>
You are EduEdge, a patient and encouraging AI tutor built for students in \
underserved communities — including those with no internet access.

Your capabilities:
• Explain any concept clearly at the student's level using simple language
• Analyse photos of textbooks, handwritten notes, or diagrams
• Generate quizzes, study plans, and structured explanations via tools
• Communicate in any language the student prefers

Behaviour guidelines:
• Always be warm, encouraging, and celebrate student effort
• Use real-world analogies the student can relate to
• When an image is provided, describe what you see before answering
• Use the available tools whenever they improve your response
• Format answers with Markdown headers and bullet points for clarity
"""


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _encode_image(image_path: str | Path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _strip_thinking(text: str) -> str:
    """Remove Gemma 4 internal chain-of-thought tags."""
    text = re.sub(r"<\|channel>thought\n.*?<channel\|>", "", text, flags=re.DOTALL)
    return text.strip()


# --------------------------------------------------------------------------- #
# EduTutor class
# --------------------------------------------------------------------------- #

class EduTutor:
    """Stateful tutor session backed by Gemma 4 (Ollama)."""

    def __init__(self, model: str = GEMMA_MODEL):
        self.model = model
        # Bypass macOS system proxy for localhost traffic.
        _transport = httpx.HTTPTransport(proxy=None)
        self.client = ollama.Client(host=OLLAMA_HOST, transport=_transport, timeout=180)
        self.history: list[dict] = []

    def reset(self) -> None:
        self.history.clear()

    def chat(
        self,
        message: str,
        image_path: str | Path | None = None,
    ) -> Generator[str, None, None]:
        """
        Generator: yields text chunks as the model streams its response.

        Flow:
          1. Send message to the model with streaming ON.
          2. Collect tokens; detect tool_calls in the final chunk.
          3. If tool_calls found → execute tools, yield status message, loop.
          4. If no tool_calls → done; history updated.
        """
        user_msg: dict = {"role": "user", "content": message}
        if image_path:
            user_msg["images"] = [_encode_image(image_path)]
        self.history.append(user_msg)

        while True:
            full_content = ""
            tool_calls_found = None

            # ── Streaming model call ─────────────────────────────────────────
            stream = self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    *self.history,
                ],
                tools=TOOLS,
                options=DEFAULT_OPTIONS,
                stream=True,
            )

            for chunk in stream:
                token = chunk.message.content or ""
                full_content += token

                # Final chunk may carry tool_calls
                if chunk.message.tool_calls:
                    tool_calls_found = chunk.message.tool_calls
                    break  # stop streaming; handle tool calls below

                if token:
                    yield token

            # ── Tool call branch ─────────────────────────────────────────────
            if tool_calls_found:
                self.history.append({
                    "role": "assistant",
                    "content": full_content,
                    "tool_calls": tool_calls_found,
                })
                for tc in tool_calls_found:
                    fn_name = tc.function.name
                    fn_args = dict(tc.function.arguments)
                    yield f"\n\n*🔧 Using tool: **{fn_name}**...*\n\n"
                    result = dispatch(fn_name, fn_args)
                    self.history.append({
                        "role": "tool",
                        "content": result,
                        "name": fn_name,
                    })
                continue  # loop → model sees tool results and responds

            # ── Final response ───────────────────────────────────────────────
            cleaned = _strip_thinking(full_content)
            self.history.append({"role": "assistant", "content": cleaned})
            break
