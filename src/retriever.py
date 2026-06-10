"""Hybrid retrieval: dense (ChromaDB) + sparse (BM25), ensemble weighted."""
from __future__ import annotations
import logging
from typing import Optional

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

from src.store import get_store
from src.config import settings

logger = logging.getLogger(__name__)


def build_retriever(bm25_docs: Optional[list[Document]] = None) -> EnsembleRetriever:
    dense = get_store().as_retriever(search_kwargs={"k": settings.top_k})

    if bm25_docs:
        sparse = BM25Retriever.from_documents(bm25_docs)
        sparse.k = settings.top_k
        return EnsembleRetriever(
            retrievers=[dense, sparse],
            weights=[0.6, 0.4],
        )
    return dense  # type: ignore[return-value]


def retrieve(query: str, retriever) -> list[Document]:
    docs = retriever.invoke(query)
    logger.info("Retrieved %d docs for query: %.60s...", len(docs), query)
    return docs
