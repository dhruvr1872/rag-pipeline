# RAG Pipeline

A production-ready Retrieval-Augmented Generation pipeline with hybrid retrieval, reranking, and automated evaluation.

## Architecture

```
Documents (PDF / TXT / URL)
    └── Chunking (recursive / token)
        └── Embeddings (text-embedding-3-small)
            └── ChromaDB (dense) + BM25 (sparse)
                └── EnsembleRetriever (hybrid, 60/40 weighted)
                    └── GPT-4o-mini (citation-grounded generation)
                        └── RAGAS Evaluation
```

## Features

- **Multiple input types** — PDF, plain text, Markdown, web URLs, directories
- **Chunking strategies** — recursive character, token-based
- **Hybrid retrieval** — dense (ChromaDB) + sparse (BM25), ensemble weighted
- **Citation-grounded answers** — every claim references a source document
- **RAGAS evaluation** — faithfulness, answer relevancy, context recall, context precision
- **FastAPI server** — production-deployable REST API
- **CLI** — index and query from the terminal

## Quickstart

```bash
git clone https://github.com/dhruvr1872/rag-pipeline
cd rag-pipeline
pip install -r requirements.txt
cp .env.example .env
# Add your OPENAI_API_KEY to .env

# Index documents
python main.py index path/to/doc.pdf https://example.com/article

# Query
python main.py query "What does the document say about X?"

# Serve as API
python main.py serve
# POST http://localhost:8000/index  {"sources": ["doc.pdf"]}
# POST http://localhost:8000/query  {"question": "..."}
```

## Evaluation

```bash
python eval/run_eval.py
# Faithfulness:      0.92
# Answer Relevancy:  0.88
# Context Recall:    0.85
# Context Precision: 0.79
```

## Stack

| Layer | Tech |
|---|---|
| Orchestration | LangChain |
| Vector store | ChromaDB |
| Sparse retrieval | BM25 (rank-bm25) |
| Embeddings | OpenAI text-embedding-3-small |
| LLM | GPT-4o-mini |
| Evaluation | RAGAS |
| API | FastAPI + uvicorn |
| Config | pydantic-settings |
