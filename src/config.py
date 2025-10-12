from pathlib import Path
from pydantic import BaseModel

class Settings(BaseModel):
    manifest_csv: Path = Path("data/eu_plp_manifest.csv")
    index_dir: Path = Path("index/faiss_eu")
    chunk_size: int = 1200
    chunk_overlap: int = 200
    embedding_model: str = "BAAI/bge-m3"          #HF bi-encoder
    reranker_model: str = "BAAI/bge-reranker-v2-m3"  #Cross-encoder
    topk_retriever: int = 10
    topk_reranked: int = 3

settings = Settings()

#topk_retriever returns 10 from FAISS by cosine similarity 
#tokk_reranked: Cross encoder reducxes it to 3