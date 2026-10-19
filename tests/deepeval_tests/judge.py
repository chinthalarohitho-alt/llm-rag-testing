"""
The "judge": the LLM that grades our app's answers in every DeepEval metric.

By default DeepEval uses OpenAI as the judge. Here we use a local Ollama model
instead, so the whole suite runs free and offline.
Model choice: we benchmarked small judges on answers we KNEW were right or wrong:
  llama3.2:3b -> 20/26 correct, and it PASSED a made-up (hallucinated) answer
  gemma3:4b   -> 23/26 correct, caught the hallucination   <- used here
  qwen3:4b    -> timed out (it "thinks" too long)
Lesson: always sanity-check your judge before trusting its scores.
"""
import os

from deepeval.models import OllamaModel
from dotenv import load_dotenv

load_dotenv()

JUDGE = OllamaModel(
    model=os.getenv("JUDGE_MODEL", "gemma3:4b"),
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    temperature=0,
)
