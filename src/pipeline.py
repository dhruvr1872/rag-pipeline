"""End-to-end RAG pipeline orchestrator."""
from __future__ import annotations
import logging

from langchain_core.documents import Document

from src.ingest import load_documents, chunk_documents, ChunkStrategy
from src.store import add_documents
from src.retriever import build_retriever, retrieve
from src.generator import generate, RAGResponse

logger = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(self, chunk_strategy: ChunkStrategy = "recursive"):
        self.chunk_strategy = chunk_strategy
        self._indexed_docs: list[Document] = []
        self._retriever = None

    def index(self, sources: list[str]) -> int:
        all_chunks: list[Document] = []
        for src in sources:
            docs = load_documents(src)
            chunks = chunk_documents(docs, strategy=self.chunk_strategy)
            all_chunks.extend(chunks)

        self._indexed_docs = all_chunks
        count = add_documents(all_chunks)
        self._retriever = build_retriever(bm25_docs=all_chunks)
        logger.info("Indexed %d chunks from %d sources", count, len(sources))
        return count

    def query(self, question: str) -> RAGResponse:
        if self._retriever is None:
            self._retriever = build_retriever()
        docs = retrieve(question, self._retriever)
        return generate(question, docs)
