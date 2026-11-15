"""Retrieval over the document for chat-style Q&A.

Vectors live in a plain numpy array. For the scale of a single uploaded PDF
(hundreds of chunks at most) a brute-force cosine scan is instant, so there is
no need for Chroma/FAISS or their dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

import config
from pdf_utils import Chunk

__all__ = [
    "SearchHit",
    "VectorIndex",
    "build_embeddings",
    "build_qa_chain",
    "format_context",
    "answer_question",
]


class EmbeddingsLike(Protocol):
    """Minimal surface we need from LangChain's embeddings wrappers."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


@dataclass(frozen=True)
class SearchHit:
    chunk: Chunk
    score: float


def _normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class VectorIndex:
    """Normalised embeddings + the chunks they belong to."""

    def __init__(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        self.chunks = list(chunks)
        self.vectors = _normalise(np.asarray(vectors, dtype=np.float32))

    def __len__(self) -> int:
        return len(self.chunks)

    @classmethod
    def build(
        cls,
        chunks: Sequence[Chunk],
        embeddings: EmbeddingsLike,
        batch_size: int = config.EMBEDDING_BATCH_SIZE,
    ) -> "VectorIndex":
        """Embed chunks in batches to stay under per-request limits."""
        vectors: list[list[float]] = []
        for start in range(0, len(chunks), batch_size):
            batch = [chunk.text for chunk in chunks[start : start + batch_size]]
            vectors.extend(embeddings.embed_documents(batch))
        if not vectors:
            raise ValueError("cannot build an index from zero chunks")
        return cls(chunks, np.asarray(vectors, dtype=np.float32))

    def search(self, query_vector: Sequence[float], k: int = config.RETRIEVAL_TOP_K) -> list[SearchHit]:
        """Top-k chunks by cosine similarity."""
        if not self.chunks:
            return []
        query = np.asarray(query_vector, dtype=np.float32)
        norm = float(np.linalg.norm(query)) or 1.0
        scores = self.vectors @ (query / norm)
        k = max(1, min(k, len(self.chunks)))
        order = np.argsort(-scores)[:k]
        return [SearchHit(chunk=self.chunks[i], score=float(scores[i])) for i in order]

    def retrieve(self, embeddings: EmbeddingsLike, question: str, k: int = config.RETRIEVAL_TOP_K) -> list[SearchHit]:
        return self.search(embeddings.embed_query(question), k=k)


def build_embeddings(model: str | None = None, api_key: str | None = None):
    """Gemini embedding model (``gemini-embedding-001`` by default)."""
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return GoogleGenerativeAIEmbeddings(
        model=model or config.DEFAULT_EMBEDDING_MODEL,
        google_api_key=api_key or config.require_api_key(),
    )


_QA_SYSTEM = """You answer questions about a single document, using only the excerpts provided.

Rules:
- Cite the page for every factual claim in square brackets, e.g. [p.4].
- If the excerpts do not contain the answer, say so plainly and state what the document does cover instead.
- Never rely on outside knowledge or invent details.
- Be concise: two or three short paragraphs at most, unless the question asks for a list."""


def build_qa_chain(llm):
    """Streaming-friendly chain: excerpts + chat history + question."""
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _QA_SYSTEM),
            MessagesPlaceholder("history"),
            (
                "human",
                "Excerpts from the document:\n<excerpts>\n{context}\n</excerpts>\n\nQuestion: {question}",
            ),
        ]
    )
    return prompt | llm | StrOutputParser()


def format_context(hits: Sequence[SearchHit]) -> str:
    """Numbered excerpts, each tagged with its page."""
    blocks = []
    for position, hit in enumerate(hits, start=1):
        blocks.append(f"Excerpt {position} [{hit.chunk.page_label}]\n{hit.chunk.text}")
    return "\n\n".join(blocks) if blocks else "(no relevant excerpts found)"


def answer_question(
    llm,
    index: VectorIndex,
    embeddings: EmbeddingsLike,
    question: str,
    history: Sequence[object] = (),
    k: int = config.RETRIEVAL_TOP_K,
) -> str:
    """Non-streaming convenience wrapper used by the CLI and tests."""
    hits = index.retrieve(embeddings, question, k=k)
    return build_qa_chain(llm).invoke(
        {"context": format_context(hits), "question": question, "history": list(history)}
    )
