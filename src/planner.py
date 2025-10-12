import json, re, os
from typing import Dict
from .ollama_client import ollama_generate

PLANNER_MODEL = os.getenv("PLANNER_MODEL", "deepseek-r1:8b")

PLANNER_SYS = (
  "You are a research planner for EU tech/copyright law. "
  "Decompose the question into at most 3 laser-focused sub-questions that match legal sections. "
  "For text-and-data mining (TDM) questions, prefer splitting into: "
  "(a) research exception TDM (e.g., DSM Article 3), "
  "(b) general TDM with opt-out (e.g., DSM Article 4), "
  "(c) any additional conditions/definitions relevant. "
  "Output STRICT JSON with keys: sub_questions (list), keywords (list), notes (string)."
)

def plan(question: str) -> Dict:
    prompt = f"{PLANNER_SYS}\n\nQuestion: {question}\n\nJSON only:"
    raw = ollama_generate(PLANNER_MODEL, prompt, temperature=0.2, max_tokens=512)
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    js = m.group(0) if m else '{"sub_questions": ["%s"], "keywords": [], "notes": ""}' % question.replace('"','')
    try:
        data = json.loads(js)
    except Exception:
        data = {"sub_questions": [question], "keywords": [], "notes": ""}
    subs = data.get("sub_questions") or [question]
    data["sub_questions"] = subs[:3]
    return data
