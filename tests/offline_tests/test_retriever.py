"""
OFFLINE TESTS - free, fast, no API key needed.

Not everything in an AI app needs an AI to test it. The retriever is plain
Python, so we can check it with normal assertions. If retrieval is wrong,
the LLM will be wrong too, so it's worth catching here first.
"""
import json
from pathlib import Path

import pytest

from app.rag import load_chunks, retrieve


def test_knowledge_base_is_split_into_sections():
    chunks = load_chunks()
    assert len(chunks) == 5
    assert chunks[0].startswith("Return Policy")


@pytest.mark.parametrize(
    "question, expected_section",
    [
        ("How many days do I have to return a product?", "Return Policy"),
        ("Is shipping free?", "Shipping"),
        ("Does the warranty cover water damage?", "Warranty"),
        ("Can I call customer support by phone?", "Customer Support"),
        ("How do I reset my password?", "Account and Passwords"),
    ],
)
def test_retriever_finds_the_right_section(question, expected_section):
    top_chunk = retrieve(question)[0]
    assert top_chunk.startswith(expected_section)


def test_retriever_returns_nothing_for_unrelated_question():
    assert retrieve("Who won the football world cup?") == []


def test_golden_dataset_is_well_formed():
    data = json.loads(Path("data/golden_dataset.json").read_text())
    assert len(data) > 0
    for row in data:
        assert row["question"].strip()
        assert row["expected_answer"].strip()
