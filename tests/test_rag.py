from __future__ import annotations

import numpy as np
import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

import rag
from pdf_utils import Chunk

VOCAB = ["alpha", "beta", "gamma", "delta"]


class FakeEmbeddings:
    """Bag-of-words embeddings: enough structure to test ranking."""

    def __init__(self) -> None:
        self.document_calls = 0

    def _vector(self, text: str) -> list[float]:
        return [float(text.lower().count(word)) for word in VOCAB]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_calls += 1
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


CHUNKS = [
    Chunk(index=0, text="alpha alpha beta", pages=(1,)),
    Chunk(index=1, text="gamma delta", pages=(2,)),
    Chunk(index=2, text="alpha", pages=(3,)),
]


@pytest.fixture
def embeddings() -> FakeEmbeddings:
    return FakeEmbeddings()


def test_build_and_retrieve_ranks_by_similarity(embeddings):
    index = rag.VectorIndex.build(CHUNKS, embeddings)
    hits = index.retrieve(embeddings, "alpha", k=2)

    assert len(hits) == 2
    assert hits[0].score >= hits[1].score
    # Both alpha-bearing chunks should outrank the pure gamma/delta chunk.
    assert {hit.chunk.index for hit in hits} == {0, 2}


def test_scores_are_cosine_bounded(embeddings):
    index = rag.VectorIndex.build(CHUNKS, embeddings)
    for hit in index.search([1.0, 1.0, 0.0, 0.0], k=3):
        assert -1.0001 <= hit.score <= 1.0001


def test_k_is_clamped_to_collection_size(embeddings):
    index = rag.VectorIndex.build(CHUNKS, embeddings)
    assert len(index.retrieve(embeddings, "alpha", k=99)) == len(CHUNKS)
    assert len(index.retrieve(embeddings, "alpha", k=0)) == 1


def test_build_batches_embedding_calls(embeddings):
    chunks = [Chunk(index=i, text="alpha", pages=(i + 1,)) for i in range(7)]
    rag.VectorIndex.build(chunks, embeddings, batch_size=3)
    assert embeddings.document_calls == 3


def test_length_and_mismatch_validation(embeddings):
    index = rag.VectorIndex.build(CHUNKS, embeddings)
    assert len(index) == len(CHUNKS)
    with pytest.raises(ValueError):
        rag.VectorIndex(CHUNKS, np.zeros((2, 4)))


def test_empty_chunks_rejected(embeddings):
    with pytest.raises(ValueError):
        rag.VectorIndex.build([], embeddings)


def test_search_on_empty_index_returns_nothing():
    empty = rag.VectorIndex([], np.zeros((0, 4)))
    assert empty.search([1.0, 0.0, 0.0, 0.0]) == []


def test_zero_vector_query_does_not_crash(embeddings):
    index = rag.VectorIndex.build(CHUNKS, embeddings)
    assert len(index.search([0.0, 0.0, 0.0, 0.0], k=1)) == 1


def test_format_context_labels_pages(embeddings):
    index = rag.VectorIndex.build(CHUNKS, embeddings)
    hits = index.retrieve(embeddings, "alpha", k=2)
    context = rag.format_context(hits)
    assert "Excerpt 1" in context and "p." in context


def test_format_context_handles_no_hits():
    assert "no relevant excerpts" in rag.format_context([])


def test_answer_question_returns_model_text(embeddings):
    index = rag.VectorIndex.build(CHUNKS, embeddings)
    llm = FakeListChatModel(responses=["Alpha is discussed on p.1."])
    answer = rag.answer_question(llm, index, embeddings, "what is alpha?")
    assert answer == "Alpha is discussed on p.1."
