"""
Connects promptfoo to our Python RAG app.

Promptfoo calls `call_api()` for each test case. We pass the question to our
RAG pipeline and return the answer (and the retrieved context, for debugging).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag import answer_question  # noqa: E402


def call_api(prompt, options, context):
    result = answer_question(prompt)
    return {
        "output": result["answer"],
        "metadata": {"retrieved_context": result["context"]},
    }
