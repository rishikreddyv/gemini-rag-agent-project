import os
import pickle
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings


def ingest_pdf(pdf_path: str, output_dir: str = settings.faiss_dir) -> int:
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = splitter.split_documents(docs)

    for chunk in chunks:
        raw_page = chunk.metadata.get("page", 0)
        chunk.metadata["page_number"] = raw_page + 1 if isinstance(raw_page, int) else raw_page
        chunk.metadata["source_file"] = os.path.basename(pdf_path)

    embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    os.makedirs(output_dir, exist_ok=True)
    vectorstore.save_local(output_dir)

    with open(os.path.join(output_dir, "chunks.pkl"), "wb") as f:
        pickle.dump(chunks, f)

    print(f"Saved {len(chunks)} chunks from '{pdf_path}' into '{output_dir}/'")
    return len(chunks)
