# FastAPI Docs RAG

A retrieval-augmented generation (RAG) assistant that answers questions about the
[FastAPI documentation](https://fastapi.tiangolo.com/) with cited sources.



## How it works (M1: naive baseline)

```
FastAPI docs (157 .md files)
        │  fixed-size chunking (1000 chars, 100 overlap)
        ▼
BGE-small embeddings ──► ChromaDB (1,732 chunks, persisted locally)
                                │
Question ──► embed ──► top-5 similarity search
                                │
                                ▼
            Claude (grounded prompt: answer only from context, cite [n])
                                │
                                ▼
                     Answer with source citations
```

- **Ingestion:** Reads every Markdown file under `docs/en`, splits it into overlapping
  character windows, and upserts the pieces into a persistent Chroma collection.
  Re-running is idempotent because chunk IDs are deterministic (`path::index`).
- **Embeddings:** [`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5) runs locally
  through `sentence-transformers`, so indexing doesn't call any paid API.
- **Generation:** Claude receives the top-5 chunks, numbered and tagged with their source file.
  The system prompt tells it to answer only from that context, cite chunks as `[1]`, `[2]`, …,
  and say "I don't know" when the context doesn't contain the answer.
- **Fallback:** If the primary model declines a request, the API retries on a fallback model
  (`server-side-fallback` beta). A request that is still declined returns a clear message
  instead of failing.

This baseline is intentionally simple. It gives later milestones a reference point to
measure against.

## Quickstart

Requirements: Python 3.12+, [uv](https://docs.astral.sh/uv/), and an
[Anthropic API key](https://console.anthropic.com/).

```bash
# 1. Install dependencies
uv sync

# 2. Add your API key
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env

# 3. Download the FastAPI English docs (sparse checkout, docs only)
git clone --filter=blob:none --sparse https://github.com/fastapi/fastapi.git data/raw
git -C data/raw sparse-checkout set docs/en

# 4. Build the index (run from the project root)
uv run python src/m1_naive.py ingest

# 5. Ask a question
uv run python src/m1_naive.py "How do I declare an optional query parameter?"
```

## Project structure

```
src/m1_naive.py   # M1: ingestion + retrieval + grounded generation
data/raw/         # FastAPI docs (git-ignored, downloaded in step 3)
data/chroma/      # Vector store (git-ignored, built in step 4)
eval/             # Evaluation set and results (M2)
```




## Tech stack

Python · ChromaDB · sentence-transformers (BGE) · Anthropic Claude API · uv
