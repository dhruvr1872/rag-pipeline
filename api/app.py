"""FastAPI server exposing the RAG pipeline as a REST API."""
from __future__ import annotations
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.pipeline import RAGPipeline

pipeline: RAGPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    global pipeline
    pipeline = RAGPipeline()
    yield
    pipeline = None


app = FastAPI(title="RAG Pipeline API", version="1.0.0", lifespan=lifespan)


class IndexRequest(BaseModel):
    sources: list[str]
    chunk_strategy: str = "recursive"


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    num_docs_used: int


@app.post("/index")
async def index_documents(req: IndexRequest) -> dict:
    assert pipeline is not None
    pipeline.chunk_strategy = req.chunk_strategy  # type: ignore[assignment]
    count = pipeline.index(req.sources)
    return {"chunks_indexed": count}


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest) -> QueryResponse:
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    response = pipeline.query(req.question)
    return QueryResponse(
        answer=response.answer,
        sources=response.sources,
        num_docs_used=response.num_docs_used,
    )


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
