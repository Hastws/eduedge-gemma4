"""
tools.py — Function-calling tools for EduEdge.

Gemma 4 supports native function calling. Each tool is described as a JSON-schema
dict and the corresponding Python function is registered below.
"""

import json
from typing import Any


# --------------------------------------------------------------------------- #
# Tool schemas  (passed to the model in the system prompt / tools field)
# --------------------------------------------------------------------------- #

TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "generate_quiz",
            "description": (
                "Generate a short multiple-choice quiz on a given topic to test "
                "the student's understanding. Returns a JSON list of questions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "The subject or concept to quiz on, e.g. 'photosynthesis'.",
                    },
                    "num_questions": {
                        "type": "integer",
                        "description": "How many questions to generate (1-5).",
                        "default": 3,
                    },
                    "difficulty": {
                        "type": "string",
                        "enum": ["easy", "medium", "hard"],
                        "default": "medium",
                    },
                },
                "required": ["topic"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_study_plan",
            "description": (
                "Create a structured study plan for a student based on a subject "
                "and the time available before an exam or goal."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "subject": {
                        "type": "string",
                        "description": "The subject or course, e.g. 'algebra', 'biology'.",
                    },
                    "days_available": {
                        "type": "integer",
                        "description": "Number of days the student has to study.",
                    },
                    "weak_areas": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Topics the student struggles with most.",
                    },
                },
                "required": ["subject", "days_available"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "explain_concept",
            "description": (
                "Provide a clear, age-appropriate explanation of a concept, "
                "optionally with an analogy and example."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "concept": {
                        "type": "string",
                        "description": "The concept to explain, e.g. 'Newton's second law'.",
                    },
                    "grade_level": {
                        "type": "string",
                        "enum": ["elementary", "middle", "high_school", "university"],
                        "default": "middle",
                    },
                    "language": {
                        "type": "string",
                        "description": "Language to respond in, e.g. 'en', 'es', 'zh', 'ar'.",
                        "default": "en",
                    },
                },
                "required": ["concept"],
            },
        },
    },
]


# --------------------------------------------------------------------------- #
# Tool implementations
# --------------------------------------------------------------------------- #

def generate_quiz(topic: str, num_questions: int = 3, difficulty: str = "medium") -> str:
    """Return format instructions so the model produces a well-structured quiz."""
    return json.dumps({
        "task": "generate_quiz",
        "topic": topic,
        "num_questions": num_questions,
        "difficulty": difficulty,
        "instruction": (
            f"Generate exactly {num_questions} multiple-choice questions about '{topic}' "
            f"at {difficulty} difficulty. Use this exact format for each question:\n\n"
            "**Q[N]. [Question text]**\n"
            "A) [option]  B) [option]  C) [option]  D) [option]\n"
            "✅ **Answer:** [Letter]) — [One-sentence explanation]\n"
            "---\n"
            "After the last question, add a brief encouraging note for the student."
        ),
    })


def create_study_plan(subject: str, days_available: int, weak_areas: list[str] | None = None) -> str:
    """Return format instructions so the model produces a day-by-day study plan."""
    weak = weak_areas or []
    weak_note = (
        f"The student struggles with: {', '.join(weak)}. Allocate extra time to these. "
        if weak else ""
    )
    return json.dumps({
        "task": "create_study_plan",
        "subject": subject,
        "days_available": days_available,
        "weak_areas": weak,
        "instruction": (
            f"Create a detailed {days_available}-day study plan for '{subject}'. "
            f"{weak_note}"
            "Format each day as:\n"
            "### 📅 Day N — [Focus Topic]\n"
            "| Time | Activity |\n"
            "|------|----------|\n"
            "| Xmin | [task]   |\n\n"
            "End with a **💡 General Tips** section (3-5 bullet points)."
        ),
    })


def explain_concept(concept: str, grade_level: str = "middle", language: str = "en") -> str:
    """Return format instructions so the model produces a structured concept explanation."""
    level_desc = {
        "elementary": "a 10-year-old child (use very simple words and fun examples)",
        "middle": "a 13-year-old student (clear language, relatable analogies)",
        "high_school": "a 16-year-old student (more technical depth is fine)",
        "university": "a university student (full technical detail)",
    }.get(grade_level, "a middle-school student")
    lang_note = f" Write the entire response in language code '{language}'." if language != "en" else ""
    return json.dumps({
        "task": "explain_concept",
        "concept": concept,
        "grade_level": grade_level,
        "language": language,
        "instruction": (
            f"Explain '{concept}' clearly for {level_desc}.{lang_note}\n"
            "Structure your response as:\n"
            "### 💡 What is [concept]?\n"
            "[1-2 sentence simple definition]\n\n"
            "### 🔗 Analogy\n"
            "[Relate to something the student already knows from daily life]\n\n"
            "### 📌 Key Points\n"
            "[3-5 bullet points]\n\n"
            "### ✏️ Worked Example\n"
            "[A concrete, step-by-step example]\n\n"
            "### ❓ Check Your Understanding\n"
            "[One question to verify the student understood — do not give the answer yet]"
        ),
    })


# Map function name → callable
TOOL_REGISTRY: dict[str, Any] = {
    "generate_quiz": generate_quiz,
    "create_study_plan": create_study_plan,
    "explain_concept": explain_concept,
}


def dispatch(name: str, arguments: dict) -> str:
    """Call the registered tool and return its JSON string result."""
    fn = TOOL_REGISTRY.get(name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool: {name}"})
    return fn(**arguments)
