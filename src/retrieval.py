from typing import List, Tuple, Any, Optional   # <-- add Optional
from operator import itemgetter

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

from .config import settings

def load_index():
    embed = HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        encode_kwargs={"normalize_embeddings": True}
    )
    vs = FAISS.load_local(
        str(settings.index_dir),
        embed,
        allow_dangerous_deserialization=True
    )
    return vs.as_retriever(search_kwargs={"k": settings.topk_retriever})

def _rerank(query: str, docs: List[Any], top_n: int) -> List[Any]:
    if not docs:
        return []
    ce = CrossEncoder(settings.reranker_model)
    pairs = [(query, d.page_content) for d in docs]
    scores = ce.predict(pairs).tolist()
    ranked = sorted(zip(docs, scores), key=itemgetter(1), reverse=True)

    picked = []
    per_doc_cap = 2  #2 Chunks per doc allowed
    per_doc_count = {}

    for d, s in ranked:
        did = d.metadata.get("doc_id")
        cnt = per_doc_count.get(did, 0)
        if cnt >= per_doc_cap:
            continue
        picked.append((d, s))
        per_doc_count[did] = cnt + 1
        if len(picked) >= top_n:
            break
    return [d for d, _ in picked]

def retrieve_and_rerank(query: str, module_filter: Optional[str] = None) -> List[Any]:  
    retriever = load_index()
    base_docs = retriever.invoke(query)
    if module_filter:
        base_docs = [d for d in base_docs if d.metadata.get("module") == module_filter]
    return _rerank(query, base_docs, settings.topk_reranked)

if __name__ == "__main__":
    import argparse, textwrap
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", required=True, help="query")
    ap.add_argument(
        "--module",
        help="optional module filter",
        choices=["Equality_Foundations", "Data_IP_TDM", "AI_Cyber_Gov"], 
    )
    args = ap.parse_args()

    docs = retrieve_and_rerank(args.q, module_filter=args.module)  
    print(f"\nQuery: {args.q}")
    print(f"Module: {args.module}\n")
    for i, d in enumerate(docs, 1):
        meta = d.metadata
        print(f"[{i}] {meta['doc_id']}  {meta.get('section','')}")
        snippet = d.page_content.strip().replace("\n", " ")
        print(textwrap.shorten(snippet, width=300, placeholder="…"))
        print("-" * 80)
