# 🤖 RAG Agent — Databricks / Spark / dbt

> An intelligent AI agent that answers technical questions about Databricks, Apache Spark, and dbt documentation — built with LangGraph, Claude, pgvector, and FastAPI.

![CI](https://github.com/OUWABELIDRISSI/rag-agent-databricks/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 🎯 Overview

This project demonstrates a production-grade **Agentic RAG** (Retrieval-Augmented Generation) system that:

- Ingests technical documentation (HTML/PDF) into a **pgvector** database
- Routes questions intelligently using an **LLM-based router**
- Retrieves and reranks the most relevant chunks using **cosine similarity**
- Generates accurate answers grounded in the documentation using **Claude via OpenRouter**
- Evaluates response quality automatically using a **RAGAS-style pipeline**
- Exposes everything via a clean **FastAPI REST API**
- Ships with **Docker**, **CI/CD** (GitHub Actions), and cloud deployment on **Azure Container Apps**

---

## 🏗️ Architecture

```
question
    │
    ▼
[Router]  ──── off-topic ────► [Generator] ──► direct answer
    │
    ▼
[Retriever]  ──── pgvector (cosine similarity)
    │
    ▼
[Reranker]   ──── filters score ≥ 0.40, keeps top 4
    │
    ▼
[Generator]  ──── Claude via OpenRouter + context
    │
    ▼
answer + sources + route
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| LLM | Claude (via OpenRouter) |
| Embeddings | Mistral `mistral-embed` |
| Agent orchestration | LangGraph |
| Vector store | PostgreSQL + pgvector |
| Ingestion | httpx + BeautifulSoup + pypdf |
| API | FastAPI + uvicorn |
| Evaluation | RAGAS-style pipeline |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Cloud | Azure Container Apps |
| Code quality | ruff + mypy + pytest |

---

## 📊 Evaluation Results

Evaluated on 5 technical questions about Databricks, Spark, and dbt:

| Metric | Score |
|---|---|
| Faithfulness | **1.00** |
| Answer Relevancy | **0.94** |
| Context Recall | **0.88** |

> Faithfulness = 1.00 means the agent **never hallucinates** — every claim is grounded in the documentation.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.12+
- Docker Desktop
- OpenRouter API key → [openrouter.ai](https://openrouter.ai)
- Mistral API key → [console.mistral.ai](https://console.mistral.ai)

### Installation

```bash
# 1. Clone the repo
git clone https://github.com/OUWABELIDRISSI/rag-agent-databricks.git
cd rag-agent-databricks

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# 3. Install dependencies
pip install -e ".[dev]"

# 4. Configure environment
cp .env.example .env
# Edit .env and fill in your API keys
```

### Start PostgreSQL

```bash
docker compose up -d postgres
```

### Ingest documentation

```bash
python scripts/ingest_docs.py
```

### Start the API

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Ask a question

```bash
curl -X POST http://localhost:8000/api/v1/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Delta Lake?"}'
```

---

## 📁 Project Structure

```
rag-agent-databricks/
├── src/
│   ├── agent/
│   │   ├── graph.py        # LangGraph agent definition
│   │   ├── nodes.py        # Router, Retriever, Reranker, Generator
│   │   ├── retriever.py    # pgvector similarity search
│   │   ├── prompts.py      # System prompts
│   │   └── state.py        # AgentState TypedDict
│   ├── ingestion/
│   │   └── pipeline.py     # Web/PDF ingestion + embeddings
│   ├── api/
│   │   ├── main.py         # FastAPI app
│   │   ├── routes.py       # Endpoints /ask /ask/stream /health
│   │   └── schemas.py      # Pydantic models
│   ├── evaluation/
│   │   └── evaluator.py    # RAGAS-style evaluation pipeline
│   └── utils/
│       ├── config.py       # Pydantic Settings
│       └── logging.py      # Structured logging
├── scripts/
│   ├── ingest_docs.py      # Ingestion script
│   ├── run_evaluation.py   # Evaluation script
│   └── init_db.sql         # PostgreSQL schema + HNSW index
├── tests/
│   └── unit/
│       ├── test_api.py
│       └── test_pipeline.py
├── docker/
│   └── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── .github/
    └── workflows/
        └── ci.yml
```

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## 🔑 Environment Variables

| Variable | Description | Required |
|---|---|---|
| `OPENROUTER_API_KEY` | OpenRouter API key | ✅ |
| `MISTRAL_API_KEY` | Mistral API key (embeddings) | ✅ |
| `POSTGRES_HOST` | PostgreSQL host | ✅ |
| `POSTGRES_DB` | Database name | ✅ |
| `POSTGRES_USER` | Database user | ✅ |
| `POSTGRES_PASSWORD` | Database password | ✅ |
| `LLM_MODEL` | Model name on OpenRouter | `anthropic/claude-3-haiku` |
| `CHUNK_SIZE` | Chunk size for ingestion | `512` |
| `EMBEDDING_DIMENSION` | Embedding vector dimension | `1024` |

---

## 👤 Author

**Elmehdi Ouwab Elidrissi**  
Data Engineer — Marseille, France  
[GitHub](https://github.com/OUWABELIDRISSI) · [LinkedIn](https://linkedin.com/in/ton-profil)

---

## 📄 License

MIT
