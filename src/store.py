"""ChromaDB vector store operations."""
from __future__ import annotations
import logging

import chromadb
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

from src.config import settings

logger = logging.getLogger(__name__)


def _embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        openai_api_key=settings.openai_api_key,
    )


def get_store() -> Chroma:
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=_embeddings(),
        persist_directory=settings.chroma_persist_dir,
    )


def add_documents(docs: list[Document]) -> int:
    store = get_store()
    store.add_documents(docs)
    logger.info("Added %d chunks to vector store", len(docs))
    return len(docs)


def clear_store() -> None:
    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    client.delete_collection(settings.collection_name)
    logger.info("Cleared collection: %s", settings.collection_name)
