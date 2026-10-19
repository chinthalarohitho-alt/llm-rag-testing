"""
A tiny RAG (Retrieval-Augmented Generation) app.

RAG in 3 steps:
  1. LOAD     - split our knowledge base into small chunks
  2. RETRIEVE - find the chunks most related to the user's question
  3. GENERATE - give those chunks to the LLM and ask it to answer ONLY from them

To keep it easy to read, retrieval uses simple word overlap instead of a
vector database. Real projects swap `retrieve()` for embeddings + a vector
store; the tests stay exactly the same.
"""
import re
from pathlib import Path

from app.llm import ask_llm

KNOWLEDGE_BASE = Path(__file__).parent / "knowledge_base" / "company_faq.md"

STOP_WORDS = {
    "a", "an", "the", "is", "are", "do", "does", "can", "i", "you", "my", "to",
    "of", "in", "on", "for", "and", "or", "how", "what", "when", "it", "be", "if",
    "with", "your", "we", "there", "much", "long", "any",
}

SYSTEM_PROMPT = (
    "You are TechNova's customer support assistant. "
    "Answer ONLY using the context provided. "
    "If the context says something is NOT offered or NOT available, that IS the answer: say so clearly. "
    "Only if the topic is not mentioned at all, say: \"I don't have that information.\""
)


# ---------- 1. LOAD ----------
def load_chunks():
    """Split the knowledge base into one chunk per '## Section'."""
    text = KNOWLEDGE_BASE.read_text()
    sections = text.split("## ")
    return [s.strip() for s in sections if s.strip()]


# ---------- 2. RETRIEVE ----------
def _words(text):
    words = set(re.findall(r"[a-z]+", text.lower()))
    return words - STOP_WORDS


def retrieve(question, top_k=2):
    """Return the `top_k` chunks that share the most words with the question."""
    question_words = _words(question)
    scored = []
    for chunk in load_chunks():
        score = len(question_words & _words(chunk))
        if score > 0:
            scored.append((score, chunk))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [chunk for _, chunk in scored[:top_k]]


# ---------- 3. GENERATE ----------
def answer_question(question):
    """Full RAG pipeline. Returns the answer AND the context used (tests need both)."""
    context = retrieve(question)
    prompt = "Context:\n" + "\n\n".join(context) + f"\n\nQuestion: {question}"
    answer = ask_llm(prompt, system=SYSTEM_PROMPT)
    return {"answer": answer, "context": context}


if __name__ == "__main__":
    # Try it:  python -m app.rag "How long do I have to return a product?"
    import sys

    result = answer_question(" ".join(sys.argv[1:]) or "What is the return policy?")
    print("Context used:\n", result["context"], "\n")
    print("Answer:\n", result["answer"])
