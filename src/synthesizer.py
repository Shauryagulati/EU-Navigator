import os
import re
from typing import List
from .ollama_client import ollama_generate

WRITER_MODEL = os.getenv("WRITER_MODEL", "llama3.1:8b")

SYNTH_SYS = (
  "You are a precise EU-law tutor. Using ONLY the provided context:\n"
  "1 Start with one short paragraph (2-3 sentences) directly answering the question.\n"
  "2 Then output 3-5 bullet points, each on its own line, summarizing supporting details or conditions.\n"
  "3 Separate every bullet with a newline and a leading '• '.\n"
  "4 End each factual statement with a citation like [DOC_ID: Article X] if available.\n"
  "5 Never merge bullets or paragraphs on the same line; readability is more important than compactness.\n"
  "If context is insufficient, say 'Not enough evidence in provided context.'"
)

#Different Prompting technique tried
# SYNTH_SYS = (
#     "You are a precise EU-law tutor. Use ONLY the provided context.\n"
#     "Write 1–2 short paragraphs that answer the question directly, followed by 2–4 concise bullets for key points.\n"
#     "Every factual sentence MUST end with a citation in the form [DOC_ID: SECTION] if SECTION exists, otherwise [DOC_ID]. "
#     "Do NOT use numeric footnotes like (10) or [1.3]. "
#     "If unsure, say 'Not enough evidence in provided context.'"
# )

def _pack_context(docs) -> str:
    lines = []
    for i, d in enumerate(docs, 1):
        doc_id = d.metadata["doc_id"]
        section = d.metadata.get("section", "")
        # Give the model a copyable citation token
        citation = f"[{doc_id}: {section}]" if section else f"[{doc_id}]"
        lines.append(f"[{i}] CITATION: {citation}")
        lines.append(d.page_content.strip()[:4000])
        lines.append("")
    return "\n".join(lines)

def _format_output(text: str) -> str:
    text = re.sub(r"(?<!\n)•", "\n•", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def synthesize(question: str, docs):
    ctx = _pack_context(docs)
    prompt = f"{SYNTH_SYS}\n\nQuestion:\n{question}\n\nContext (use only this):\n{ctx}\n\nAnswer:"
    raw = ollama_generate(WRITER_MODEL, prompt, temperature=0.15, max_tokens=900)
    return _format_output(raw)
