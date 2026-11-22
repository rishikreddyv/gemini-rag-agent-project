# Gemini RAG & Tool Agent

A modular Retrieval-Augmented Generation (RAG) and tool-calling agent built with LangChain, LangGraph, FAISS, BM25, Cross-Encoder re-ranking, and Gemini.

## Architecture
- `src/config.py`: Centralized configuration for models, chunking, retrieval counts, and index path.
- `src/ingest.py`: Loads PDFs, splits text into chunks, attaches `source_file` and `page_number` metadata, and saves FAISS + `chunks.pkl`.
- `src/retriever.py`: Runs hybrid search (FAISS semantic search + BM25 keyword search) and re-ranks candidates using a Cross-Encoder.
- `src/agent.py`: Builds the agent with `search_documents`, a safe `calculator`, and `web_search`.
- `eval.py`: Runs smoke/evaluation checks for document citations, math tool routing, and missing-info handling.

## Setup
1. Set `GOOGLE_API_KEY` (via Google Colab Secrets or environment variables).
2. Optional Langfuse tracing: set `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and `LANGFUSE_BASE_URL` (e.g., `https://cloud.langfuse.com`).
3. Ingest a PDF using `ingest_pdf("path/to/file.pdf")`, then invoke `build_agent()`.
