import argparse, json
from typing import Dict, Any, Optional, List
from collections import Counter

from .planner import plan
from .retrieval import retrieve_and_rerank
from .synthesizer import synthesize
from .reviewer import review

def answer(question: str, module: Optional[str] = None, skip_review: bool = False) -> Dict[str, Any]:
    plan_out = plan(question)
    subqs: List[str] = plan_out.get("sub_questions", [question]) or [question]

    all_docs, parts = [], []
    for sq in subqs:
        docs = retrieve_and_rerank(sq, module_filter=module)
        all_docs.extend(docs)
        part = synthesize(sq, docs)
        parts.append(f"**Sub-question:** {sq}\n{part}")

    merged = "\n\n---\n\n".join(parts)
    critique = "" if skip_review else review(question, merged)

    seen, sources = set(), []
    for d in all_docs:
        did = d.metadata.get("doc_id")
        if did in seen: 
            continue
        sources.append({
            "doc_id": did,
            "section": d.metadata.get("section",""),
            "title": d.metadata.get("title",""),
            "module": d.metadata.get("module",""),
        })
        seen.add(did)

    return {
        "question": question,
        "plan": plan_out,
        "sources": sources,
        "answer": merged,
        "review": critique,
    }

    

if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", required=True)
    ap.add_argument("--module", help="optional module filter")  
    args = ap.parse_args()
    print(json.dumps(answer(args.q, module=args.module), indent=2))