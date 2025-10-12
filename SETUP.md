# EU-NAV: Personal Learning Portal — Setup Guide

---

## Setup

### 1) Python Environment

```bash
conda create -n eu-nav python=3.9 -y
conda activate eu-nav

pip install -r requirements.txt
pip install -U langchain-huggingface ragas==0.1.9 sentence-transformers datasets
```

### 2)  Models (local)
```bash
# Install Ollama (https://ollama.com/download)
ollama serve

# Pull models
ollama pull deepseek-r1:8b
ollama pull llama3.1:8b
```


### 3)  Environment Variables
```bash
# LLMs
OLLAMA_BASE=http://localhost:11434
PLANNER_MODEL=deepseek-r1:8b
SYNTH_MODEL=llama3.1:8b
REVIEW_MODEL=llama3.1:8b

# Embeddings / Reranker
EMBED_MODEL=BAAI/bge-m3
RERANK_MODEL=bge-reranker-v2-m3

# Paths
MANIFEST_CSV=data_manifest.csv
FAISS_DIR=index/faiss_eu

# Optional: OpenAI key (for RAGAS evaluator)
OPENAI_API_KEY=sk-...
```


### 4)  Data
```bash
data/
├── pdfs/    #Place PDF files here
└── txt/     #Place text files here
```


### 5)  Build the Index
```bash
python -m src.ingest
```


### 6)  Quick CLI testing
```bash
# Full agentic pipeline (planner → retrieve → rerank → synth → review)
python -m src.agentic --q "Is text-and-data mining lawful for AI training in the EU?"
```



### 7)  PLP App
```bash
streamlit run app_streamlit.py
```
* Learn: Mark docs complete, take notes, download PDFs
* Ask: Grounded answers, reviewer notes, per-source downloads
* Progress: Table of module/doc status and notes



### 8)  Evaluation
```bash
export OPENAI_API_KEY=sk-...   # evaluator model for RAGAS
python -m src.eval_ragas
```
Produces eval_ragas.csv with per-sample metrics: 
* answer_relevancy (question ↔ answer) 
* faithfulness (answer ↔ retrieved context)
