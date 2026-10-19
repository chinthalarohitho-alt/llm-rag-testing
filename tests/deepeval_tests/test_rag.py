"""
DEEPEVAL TEST 2 - RAG quality.

A RAG app can fail in two places, so we measure both:

  RETRIEVER metrics (did we fetch the right documents?)
    - Contextual Precision : are the most useful chunks ranked first?
    - Contextual Recall    : does the context contain everything needed for the expected answer?

  GENERATOR metrics (did the LLM use those documents well?)
    - Faithfulness         : does the answer stick to the context? (no hallucinations)
    - Answer Relevancy     : does the answer address the question?

Every question in data/golden_dataset.json becomes its own test.

About thresholds: with a big cloud judge (e.g. GPT-4 class) you would use 0.7-0.8.
Small local judges are noisier and often give a correct answer 0.5, so we use
0.5. A hallucinated answer still scores well below that, so it still fails.

Why no Contextual Relevancy metric? It scores the share of retrieved SENTENCES
that are relevant. Our chunks are whole FAQ sections (e.g. 1 of 3 sentences
answers "Is shipping free?"), so it scores low by design, and small local
judges grade it inconsistently. With a larger judge, add it back:
    ContextualRelevancyMetric(threshold=0.3, model=JUDGE)
"""
import json
from pathlib import Path

import pytest
from deepeval import assert_test
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
    GEval,
)
from deepeval.test_case import LLMTestCase, SingleTurnParams

from app.rag import answer_question
from judge import JUDGE

GOLDEN_DATASET = json.loads(Path("data/golden_dataset.json").read_text())


@pytest.mark.parametrize("row", GOLDEN_DATASET, ids=lambda row: row["question"])
def test_rag_answer(row):
    result = answer_question(row["question"])

    test_case = LLMTestCase(
        input=row["question"],
        actual_output=result["answer"],
        expected_output=row["expected_answer"],
        retrieval_context=result["context"],
    )

    metrics = [
        # Retriever
        ContextualPrecisionMetric(threshold=0.5, model=JUDGE),
        ContextualRecallMetric(threshold=0.5, model=JUDGE),
        # Generator
        FaithfulnessMetric(threshold=0.5, model=JUDGE),
        AnswerRelevancyMetric(threshold=0.5, model=JUDGE),
    ]
    assert_test(test_case, metrics)


def test_rag_refuses_when_answer_is_not_in_knowledge_base():
    """A good RAG bot says "I don't know" instead of making something up."""
    question = "What is the CEO's favourite colour?"
    result = answer_question(question)

    test_case = LLMTestCase(input=question, actual_output=result["answer"])
    refusal = GEval(
        name="Admits it doesn't know",
        criteria=(
            "The actual output must clearly say it does not have the information, "
            "and must NOT invent an answer."
        ),
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
        threshold=0.7,
        model=JUDGE,
    )
    assert_test(test_case, [refusal])
