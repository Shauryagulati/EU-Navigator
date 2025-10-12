import os
from .ollama_client import ollama_generate  

REVIEW_MODEL = os.getenv("WRITER_MODEL", "llama3.1:8b")

REVIEW_SYS = (
    "You review for completeness vs the question and sources. "
    "1) List any missing elements. 2) Suggest ONE follow-up question. 3) Do not restate the full answer."
)

def review(question: str, answer: str) -> str:
    prompt = f"{REVIEW_SYS}\n\nQuestion: {question}\n\nAnswer:\n{answer}\n\nNotes:"
    return ollama_generate(REVIEW_MODEL, prompt, temperature=0.1, max_tokens=256)
