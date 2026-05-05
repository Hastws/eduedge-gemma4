"""
tools.py — Function-calling tools for EduEdge (Spaces copy).
"""

import json
from typing import Any


TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "generate_quiz",
            "description": (
                "Generate a short multiple-choice quiz on a given topic to test "
                "the student's understanding."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "num_questions": {"type": "integer", "default": 3},
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
            "description": "Create a structured study plan based on a subject and time available.",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string"},
                    "days_available": {"type": "integer"},
                    "weak_areas": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["subject", "days_available"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "explain_concept",
            "description": "Provide a clear, age-appropriate explanation of a concept.",
            "parameters": {
                "type": "object",
                "properties": {
                    "concept": {"type": "string"},
                    "grade_level": {
                        "type": "string",
                        "enum": ["elementary", "middle", "high_school", "university"],
                        "default": "middle",
                    },
                    "language": {"type": "string", "default": "en"},
                },
                "required": ["concept"],
            },
        },
    },
]


def generate_quiz(topic: str, num_questions: int = 3, difficulty: str = "medium") -> str:
    return json.dumps({
        "task": "generate_quiz",
        "instruction": (
            f"Generate exactly {num_questions} multiple-choice questions about '{topic}' "
            f"at {difficulty} difficulty. Format each as:\n\n"
            "**Q[N]. [Question]**\nA) ... B) ... C) ... D) ...\n✅ **Answer:** [Letter]) — [explanation]\n---\n"
            "End with an encouraging note."
        ),
    })


def create_study_plan(subject: str, days_available: int, weak_areas: list[str] | None = None) -> str:
    weak = weak_areas or []
    weak_note = f"Focus extra time on: {', '.join(weak)}. " if weak else ""
    return json.dumps({
        "task": "create_study_plan",
        "instruction": (
            f"Create a {days_available}-day study plan for '{subject}'. {weak_note}"
            "Format each day as:\n### 📅 Day N — [Focus]\n| Time | Activity |\n|------|----------|\n\n"
            "End with **💡 General Tips** (3-5 bullets)."
        ),
    })


def explain_concept(concept: str, grade_level: str = "middle", language: str = "en") -> str:
    level_desc = {
        "elementary": "a 10-year-old child",
        "middle": "a 13-year-old student",
        "high_school": "a 16-year-old student",
        "university": "a university student",
    }.get(grade_level, "a middle-school student")
    lang_note = f" Respond entirely in language code '{language}'." if language != "en" else ""
    return json.dumps({
        "task": "explain_concept",
        "instruction": (
            f"Explain '{concept}' for {level_desc}.{lang_note}\n"
            "Structure:\n### 💡 What is it?\n### 🔗 Analogy\n### 📌 Key Points\n"
            "### ✏️ Worked Example\n### ❓ Check Your Understanding"
        ),
    })


TOOL_REGISTRY: dict[str, Any] = {
    "generate_quiz": generate_quiz,
    "create_study_plan": create_study_plan,
    "explain_concept": explain_concept,
}


def dispatch(name: str, arguments: dict) -> str:
    fn = TOOL_REGISTRY.get(name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool: {name}"})
    return fn(**arguments)
