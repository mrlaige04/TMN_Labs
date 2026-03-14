# Lab 2 Task 1: Vector Index API

This folder contains a modular Python API for index creation using PostgreSQL + `pgvector`.

## What is implemented

- Vector storage with PostgreSQL (`pgvector` extension)
- `POST /add_document`
  - accepts a document
  - splits into chunks
  - vectorizes chunks
  - stores chunks + vectors in DB
- `POST /query`
  - accepts a query vector
  - returns top-K nearest chunks by cosine distance
- Sample document library in `data/*.txt`
- Population script to index the sample documents

## Project structure

- `config.py` - editable parameters (including DB connection string)
- `app/main.py` - FastAPI entrypoint
- `app/api/routes.py` - API endpoints
- `app/core/chunking.py` - text splitting
- `app/core/embeddings.py` - embedding model wrapper
- `app/db/*` - DB setup, model, repository queries
- `app/schemas/*` - request/response schemas
- `scripts/populate_db.py` - batch indexing from `data/`

## 1) Configure DB connection

Edit `config.py`:

```python
DATABASE_URL = "postgresql+psycopg://postgres:postgres@localhost:5432/lab2_vector_db"
```

## 2) Start PostgreSQL with pgvector

From this folder:

```bash
docker compose up -d --build
```

This starts both containers:

- `postgres` (pgvector-enabled DB)
- `api` (FastAPI service)
- `ollama` (LLM runtime for Task 2)
- `llm_api` (Q&A Assistant API for Task 2)

API docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
Task 2 API docs: [http://127.0.0.1:8001/docs](http://127.0.0.1:8001/docs)

## 3) Populate vector storage (optional, from host)

If you want to run population from your host Python environment:

```bash
pip install -r requirements.txt
python -m scripts.populate_db
```

Or run population directly inside API container:

```bash
docker compose exec api python -m scripts.populate_db
```

Pull Ollama model once for Task 2:

```bash
docker compose exec ollama ollama pull llama3.2:3b
```

## 4) Alternative: run API locally (without API container)

You can still run only DB in Docker and API locally:

```bash
docker compose up -d postgres
uvicorn app.main:app --reload
```

## Endpoint examples

### Add document

```http
POST /add_document
Content-Type: application/json

{
  "document_id": "lab_report_1",
  "source": "lab_report_1.txt",
  "content": "Your long text document goes here...",
  "chunk_size": 700,
  "chunk_overlap": 120
}
```

### Query nearest chunks

`query_vector` must have the same dimension as embedding model output (`384` for `all-MiniLM-L6-v2`).

```http
POST /query
Content-Type: application/json

{
  "query_vector": [0.01, -0.02, 0.13, "... 380 more values ..."],
  "top_k": 5
}
```
