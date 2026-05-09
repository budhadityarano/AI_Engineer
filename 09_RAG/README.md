# 09 — Retrieval Augmented Generation (RAG)

Building production-grade RAG systems from naive to advanced architectures.

## Contents

| Folder | Topics |
|--------|--------|
| `01_vector_databases/` | FAISS, ChromaDB, Pinecone, Weaviate, pgvector |
| `02_retrieval_strategies/` | Dense/sparse retrieval, BM25, hybrid search, re-ranking |
| `03_advanced_rag/` | HyDE, FLARE, contextual compression, parent-doc retrieval |
| `04_evaluation/` | RAGAS metrics, faithfulness, context precision/recall |

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```