"""RAG generation with citation-grounded answers."""
from __future__ import annotations
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.config import settings

SYSTEM_PROMPT = """You are a helpful assistant that answers questions using only the provided context.
If the context does not contain enough information, say so clearly.
Always cite the source document (use [Source N] notation) for each claim."""

USER_PROMPT = """Context:
{context}

Question: {question}

Answer (cite sources):"""


@dataclass
class RAGResponse:
    answer: str
    sources: list[str]
    num_docs_used: int


def generate(question: str, docs: list[Document]) -> RAGResponse:
    context = "\n\n".join(
        f"[Source {i+1}] {d.page_content}" for i, d in enumerate(docs)
    )
    sources = [d.metadata.get("source", f"chunk_{i}") for i, d in enumerate(docs)]

    llm = ChatOpenAI(
        model=settings.llm_model,
        openai_api_key=settings.openai_api_key,
        temperature=0,
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", USER_PROMPT),
    ])
    response = (prompt | llm).invoke({"context": context, "question": question})
    return RAGResponse(answer=response.content, sources=sources, num_docs_used=len(docs))
