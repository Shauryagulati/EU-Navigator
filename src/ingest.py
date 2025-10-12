# src/ingest.py
import os
import re
import time
from pathlib import Path
from typing import List

from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from .config import settings
from .manifest import load_manifest


#Section Tagging
SECTION_TAG = re.compile(
    r"(Article\s+[0-9A-Za-z]+)|(Recital\s+\d+)|(Chapter\s+[IVXLC]+)",
    re.IGNORECASE,
)

def detect_section(text: str) -> str:
    """Return the first visible section label if present (for citations)."""
    m = SECTION_TAG.search(text)
    return m.group(0) if m else ""



#Article First splitting
ARTICLE_ANCHOR = re.compile(
    r"(?=\b(Article\s+[0-9A-Za-z]+|Recital\s+\d+|Chapter\s+[IVXLC]+)\b)",
    re.IGNORECASE,
)

def split_by_headings(txt: str) -> list[str]:
    """
    Split the document by legal headings while KEEPING the heading
    with each section (via lookahead). Falls back to full doc if no headings.
    """
    if not ARTICLE_ANCHOR.search(txt):
        return [txt]
    parts = ARTICLE_ANCHOR.split(txt)
    sections: list[str] = []
    i = 0
    while i < len(parts):
        #Rebuilding as [prefix+heading], [body], [heading], [body]
        if i + 1 < len(parts):
            sec = (parts[i] + parts[i + 1]).strip()
            if sec:
                sections.append(sec)
            i += 2
        else:
            tail = parts[i].strip()
            if tail:
                sections.append(tail)
            i += 1
    return sections



#Chunking
def chunk_text(txt: str) -> List[str]:
    """
    Secondary chunking inside each Article/Recital/Chapter section.
    Keeps chunks ~settings.chunk_size with overlap.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " "] 
    )
    return splitter.split_text(txt)




#Building Langchain documents
def build_corpus_docs() -> List[Document]:
    print(f"[ingest] manifest: {settings.manifest_csv}")
    rows = load_manifest(str(settings.manifest_csv))
    print(f"[ingest] rows: {len(rows)}")
    if not rows:
        raise RuntimeError("Manifest is empty. Run `python -m src.make_manifest` and check data paths.")

    docs: List[Document] = []
    for row in rows:
        txt_path = Path(row.txt_path)
        if not txt_path.exists():
            raise FileNotFoundError(f"TXT not found: {txt_path} (doc_id={row.doc_id})")

        raw = txt_path.read_text(encoding="utf-8", errors="ignore")
        sections = split_by_headings(raw)

        for s_idx, sec in enumerate(sections):
            chunks = chunk_text(sec)
            for c_idx, ch in enumerate(chunks):
                docs.append(Document(
                    page_content=ch,
                    metadata={
                        "doc_id": row.doc_id,
                        "title": row.title,
                        "module": row.module,
                        "txt_path": row.txt_path,
                        "pdf_path": row.pdf_path,
                        #Section-aware index (section idx + chunk idx within that section)
                        "chunk_index": (s_idx, c_idx),
                        "section": detect_section(ch),
                    }
                ))

    print(f"[ingest] built chunks: {len(docs)}")
    if len(docs) == 0:
        raise RuntimeError("Zero chunks built. Check your TXT content and manifest paths.")
    return docs



#Embedding using bge-m3 + FAISS
def main():
    t0 = time.time()
    print(f"[ingest] start → index_dir={settings.index_dir}")
    settings.index_dir.mkdir(parents=True, exist_ok=True)

    docs = build_corpus_docs()
    print(f"[ingest] documents ready in {time.time()-t0:.1f}s")

    print("[ingest] loading embeddings: BAAI/bge-m3 (normalized cosine)")
    t1 = time.time()
    embed = HuggingFaceEmbeddings(
        model_name=settings.embedding_model,          #BAAI/bge-m3
        encode_kwargs={"normalize_embeddings": True}  #Cosine via Dot Product
    )
    print(f"[ingest] embeddings ready in {time.time()-t1:.1f}s")

    print("[ingest] building FAISS (first run may download models)…")
    t2 = time.time()
    vs = FAISS.from_documents(docs, embed)
    print(f"[ingest] FAISS built in {time.time()-t2:.1f}s")

    print(f"[ingest] saving FAISS to {settings.index_dir} …")
    vs.save_local(str(settings.index_dir))
    print("[ingest] saved files:", os.listdir(settings.index_dir))
    print(f"[ingest] DONE total {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
