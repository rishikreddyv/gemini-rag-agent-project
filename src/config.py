import os
from dataclasses import dataclass


@dataclass
class Settings:
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
    gemini_model: str = "gemini-3.6-flash"
    embedding_model: str = "all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    chunk_size: int = 600
    chunk_overlap: int = 100
    initial_k: int = 6
    top_k: int = 3
    faiss_dir: str = "faiss_index"


settings = Settings()
