"""
DEEPEVAL TEST 1 - Basic LLM answer quality.

How DeepEval works:
  1. Build an `LLMTestCase`  -> the input, the model's actual output, (optionally) the expected output
  2. Pick one or more metrics -> each metric scores the output from 0 to 1 using a "judge" LLM
  3. `assert_test(...)`       -> the test FAILS if any score is below the metric's threshold
"""

from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric, GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams

from app.llm import ask_llm
from judge import JUDGE


def test_answer_is_relevant():
    """Answer Relevancy: does the answer actually address the question?"""
    question = "What is the capital of France?"
    test_case = LLMTestCase(input=question, actual_output=ask_llm(question))

    assert_test(test_case, [AnswerRelevancyMetric(threshold=0.7, model=JUDGE)])


def test_answer_is_correct():
    """G-Eval: write your own grading steps in plain English."""
    question = "How many continents are there on Earth? Answer in one sentence."
    test_case = LLMTestCase(
        input=question,
        actual_output=ask_llm(question),
        expected_output="There are 7 continents.",
    )
    correctness = GEval(
        name="Correctness",
        # Explicit steps keep small judges consistent (they don't invent their own).
        evaluation_steps=[
            "Identify the key fact in the expected output.",
            "Check whether the actual output states the same key fact.",
            "Extra details that are also correct must NOT lower the score.",
            "Give a high score if the key fact matches, a low score if it is missing or wrong.",
        ],
        evaluation_params=[
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
        ],
        threshold=0.7,
        model=JUDGE,
    )

    assert_test(test_case, [correctness])
