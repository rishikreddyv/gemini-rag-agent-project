import os
import pickle
from typing import List
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

from src.config import settings


class HybridRerankRetriever:
    def __init__(self, index_dir: str = settings.faiss_dir):
        if not os.path.exists(index_dir):
            raise FileNotFoundError(f"Index directory '{index_dir}' not found.")

        embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
        self.vectorstore = FAISS.load_local(
            index_dir, embeddings, allow_dangerous_deserialization=True
        )

        chunks_path = os.path.join(index_dir, "chunks.pkl")
        with open(chunks_path, "rb") as f:
            self.chunks: List[Document] = pickle.load(f)

        k_val = min(settings.initial_k, len(self.chunks)) if self.chunks else 1
        self.bm25 = BM25Retriever.from_documents(self.chunks)
        self.bm25.k = k_val
        self.reranker = CrossEncoder(settings.reranker_model)

    def retrieve(self, query: str, top_k: int = settings.top_k) -> List[Document]:
        if not self.chunks:
            return []

        k_val = min(settings.initial_k, len(self.chunks))
        vector_docs = self.vectorstore.similarity_search(query, k=k_val)
        bm25_docs = self.bm25.invoke(query)

        seen = set()
        candidates: List[Document] = []
        for doc in vector_docs + bm25_docs:
            key = (doc.page_content, doc.metadata.get("page_number"))
            if key not in seen:
                seen.add(key)
                candidates.append(doc)

        pairs = [(query, doc.page_content) for doc in candidates]
        scores = self.reranker.predict(pairs)

        ranked = sorted(zip(candidates, scores), key=lambda item: item[1], reverse=True)
        return [doc for doc, _ in ranked[:top_k]]

    def format_with_citations(self, docs: List[Document]) -> str:
        if not docs:
            return "No relevant context found in the indexed documents."

        blocks = []
        for doc in docs:
            source = doc.metadata.get("source_file", "document")
            page = doc.metadata.get("page_number", "?")
            blocks.append(f"[{source} | Page {page}]\n{doc.page_content}")
        return "\n\n".join(blocks)
