"""
The smallest possible "LLM app": send a prompt to a model, get text back.

The model runs LOCALLY with Ollama (free, private, no API key). Ollama speaks
the same API as OpenAI, so we use the standard `openai` client and just point
it at Ollama. To use OpenAI or any compatible provider instead, change
OLLAMA_BASE_URL and LLM_MODEL in .env.

Everything we test goes through `ask_llm()`, so this is the only file you
need to change to test a different model or provider.
"""
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = os.getenv("LLM_MODEL", "gemma3:4b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

_client = None


def _get_client():
    # Created lazily so offline tests can import this module without Ollama running.
    global _client
    if _client is None:
        _client = OpenAI(base_url=f"{OLLAMA_BASE_URL}/v1", api_key="ollama")  # key is ignored by Ollama
    return _client


def ask_llm(prompt, system="You are a helpful, concise assistant."):
    """Send one prompt to the model and return its text answer."""
    response = _get_client().chat.completions.create(
        model=MODEL,
        temperature=0,  # temperature 0 = more repeatable answers = more stable tests
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content.strip()
