# Task 2 - Q&A Assistant API (Ollama)

This service implements the Task 2 API:

- `POST /query` accepts a user question and returns an answer based on retrieved document chunks.

## How it works

1. Encode user question with `all-MiniLM-L6-v2`.
2. Retrieve top-K closest chunks from PostgreSQL + `pgvector` (`document_chunks` table).
3. Build an LLM prompt template that includes:
   - task definition,
   - context chunks placeholder,
   - user query placeholder.
4. Send prompt to Ollama model and return generated answer + sources.

## Start with Docker Compose

From `code/`:

```bash
docker compose up -d --build
```

## Pull Ollama model once

```bash
docker compose exec ollama ollama pull llama3.2:3b
```

## Query example

```bash
curl -X POST "http://127.0.0.1:8001/query" ^
  -H "Content-Type: application/json" ^
  -d "{\"question\":\"What is retrieval augmented generation?\",\"top_k\":5}"
```
