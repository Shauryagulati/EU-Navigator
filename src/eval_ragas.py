
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import re, numpy as np, pandas as pd
from typing import List, Dict
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, faithfulness
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

from src.retrieval import retrieve_and_rerank

SEED = [
    ("Is text-and-data mining lawful for AI training in the EU?", None),
    ("List equality-law obligations for employers.", "Equality_Foundations"),
    ("What obligations begin in 2024–2026 for AI-related regulations?", "AI_Cyber_Gov"),
    ("What rights do data subjects have under EU law?", "Data_IP_TDM"),
    ("Who enforces these rules and what penalties exist?", "AI_Cyber_Gov"),
    ("What is the scope of the general TDM exception?", "Data_IP_TDM"),
]

def make_answer_from_contexts(q: str, contexts: List[str], max_chars: int = 900) -> str:
    if not contexts:
        return ""
    cleaned = [re.sub(r"\s+", " ", c).strip() for c in contexts if c and c.strip()]
    bullets = []
    for c in cleaned[:3]:
        snip = c[:240]
        #Trying to cut the sentence at punctuation
        rev = snip[::-1]
        m = re.search(r"[.;:?!]\s", rev)
        if m:
            cut = len(snip) - m.start()
            snip = snip[:cut]
    
        #Format Answer body
        bullets.append(f"• {snip.strip()}")
    head = f"Answer (extractive, from retrieved context): {q.strip()}"
    ans = head + "\n\n" + "\n\n".join(bullets)
    if len(ans) > max_chars:
        ans = ans[:max_chars].rsplit(" ", 1)[0] + "…"
    return ans

def cosine(a: List[float], b: List[float]) -> float:
    va, vb = np.array(a), np.array(b)
    na = np.linalg.norm(va); nb = np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))

def build_rows(seed):
    rows: List[Dict] = []
    for q, module in seed:
        docs = retrieve_and_rerank(q, module_filter=module)
        ctx = [d.page_content for d in docs]
        if not ctx:
            print(f"[eval] SKIP (no contexts): {q}")
            continue
        ans = make_answer_from_contexts(q, ctx)
        if not ans.strip():
            print(f"[eval] SKIP (empty synthetic answer): {q}")
            continue
        rows.append({"question": q, "answer": ans, "contexts": ctx})
    return rows

def main():
    #Dataset Rows
    rows = build_rows(SEED)
    if not rows:
        raise RuntimeError("No rows to evaluate.")

    #Evaluator - Model GPT-4o-mini - with temperature 0 since I want it to be as factual and accurate as possible
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)  
    emb = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        encode_kwargs={"normalize_embeddings": True}
    )

    #Evaluating 1 row at a time ie 1 question at a time
    per = []
    for i, row in enumerate(rows):
        try:
            ds = Dataset.from_list([row])
            res = evaluate(
                ds,
                metrics=[answer_relevancy, faithfulness],
                llm=llm,
                embeddings=emb,
            )
            df = res.to_pandas()

            if {"answer_relevancy", "faithfulness"}.issubset(df.columns):
                ar = float(df["answer_relevancy"].iloc[0])
                ff = float(df["faithfulness"].iloc[0])
            elif {"metric", "score"}.issubset(df.columns):
                pivot = df[["metric", "score"]].pivot_table(
                    index=None, columns="metric", values="score", aggfunc="first"
                )
                ar = float(pivot.get("answer_relevancy", [np.nan])[0])
                ff = float(pivot.get("faithfulness", [np.nan])[0])
            else:
                print(f"[eval] Unexpected df columns: {list(df.columns)}; filling fallback for AR.")
                ar, ff = np.nan, np.nan

            #If Answer_relevancy = NaN, cosine(question, answer)
            if np.isnan(ar):
                qv = emb.embed_query(row["question"])
                av = emb.embed_query(row["answer"])
                ar = cosine(qv, av)

            per.append({
                "question": row["question"],
                "answer_relevancy": ar,
                "faithfulness": ff,
            })
        except Exception as e:
            print(f"[eval] Row {i} failed: {e}")

    if not per:
        raise RuntimeError("All rows failed during evaluation.")

    out = pd.DataFrame(per)
    out.to_csv("eval_ragas.csv", index=False)
    print("\n=== RAGAs (per-sample) saved to eval_ragas.csv ===")
    print(out)

    means = out.drop(columns=["question"]).mean(numeric_only=True)
    print("\n=== Averages ===")
    for k, v in means.items():
        print(f"{k:>18}: {v:.3f}")

if __name__ == "__main__":
    main()
