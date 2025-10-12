# EU Navigator - A Multi-Agent RAG Personal Learning Portal

EU Navigator is a **RAG-powered learning portal** for European tech/data/AI legislation (EU bills & directives).  
It combines **high-quality retrieval**, **cross-encoder reranking**, **agentic planning & synthesis**, and a simple PLP interface with **progress tracking** and **notes**.


## Features

- **Multi-Agentic RAG** pipeline
  - *Query Rewriter/Planner:* Deepseek-R1 (reasoning)  
  - *Retriever:* FAISS + `BAAI/bge-m3` embeddings  
  - *Reranker:* Cross-encoder (`bge-reranker-v2-m3`)  
  - *Synthesizer:* Llama 3.1 8B (via Ollama)  
  - *Reviewer:* lightweight critique for follow-ups

- **Evidence-first answers** with **inline citations** (doc_id + section + module)
- **PLP Interface (Streamlit)**  
  - Module view, checkboxes for completion, personal notes per doc  
  - “Ask” tab with formatted answers, reviewer notes, and **downloadable PDFs**

- **RAG Evaluation with RAGAS**  
  - `faithfulness` (groundedness) + `answer_relevancy`  
  - Deterministic, serial evaluation to avoid multiprocessing flakiness

- **Reproducible ingest** from TXT/PDF into **FAISS** with metadata-rich chunks


## Project Structure
 ```
EU-Navigator/
├─ data/
│ ├─ pdf/ #Raw PDFs (original documents)
│ └─ txt/ #Text extractions (for embedding)
├─ index/
│ └─ faiss_eu/ #FAISS index (auto-created by ingest)
├─ src/
│ ├─ config.py #Settings (.env-backed)
│ ├─ manifest.py #CSV manifest loader (doc_id/title/module/paths)
│ ├─ ingest.py #Chunk + embed + index (BAAI/bge-m3 + FAISS)
│ ├─ retrieval.py #Retrieve + Rerank + metadata
│ ├─ synthesizer.py #LLM answer synthesis with citations
│ ├─ planner.py #Query rewrite/plan (Deepseek-R1)
│ ├─ reviewer.py #Lightweight critique
│ ├─ agentic.py #Orchestration (planner → retrieve → rerank → synthesize → review)
│ ├─ ollama_client.py #Robust streaming client with retries
│ └─ eval_ragas.py #RAGAS evaluation (faithfulness, answer_relevancy)
├─ app_streamlit.py #PLP interface: Learn / Ask / Progress
├─ requirements.txt
├─ .env
└─ README.md
```
