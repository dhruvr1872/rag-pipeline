"""Document loading and chunking strategies."""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Literal

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    WebBaseLoader,
    DirectoryLoader,
)
from langchain_core.documents import Document

from src.config import settings

logger = logging.getLogger(__name__)

ChunkStrategy = Literal["recursive", "token"]


def load_documents(source: str) -> list[Document]:
    """Load documents from a file path, directory, or URL."""
    if source.startswith("http://") or source.startswith("https://"):
        logger.info("Loading from URL: %s", source)
        return WebBaseLoader(source).load()

    path = Path(source)
    if path.is_dir():
        logger.info("Loading directory: %s", source)
        return DirectoryLoader(source, glob="**/*.{pdf,txt,md}").load()

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return PyPDFLoader(source).load()
    if suffix in {".txt", ".md"}:
        return TextLoader(source).load()

    raise ValueError(f"Unsupported file type: {suffix}")


def chunk_documents(
    docs: list[Document],
    strategy: ChunkStrategy = "recursive",
) -> list[Document]:
    """Split documents using the chosen chunking strategy."""
    size = settings.chunk_size
    overlap = settings.chunk_overlap

    if strategy == "recursive":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
    elif strategy == "token":
        splitter = TokenTextSplitter(chunk_size=size, chunk_overlap=overlap)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    chunks = splitter.split_documents(docs)
    logger.info("Split %d docs into %d chunks (strategy=%s)", len(docs), len(chunks), strategy)
    return chunks
