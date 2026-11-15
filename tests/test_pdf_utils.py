from __future__ import annotations

import pytest

import pdf_utils
from pdf_utils import (
    PDFExtractionError,
    Page,
    chunk_pages,
    count_tokens,
    estimate_tokens,
    file_sha256,
    filter_pages,
    load_document,
    normalize_text,
)


# --- extraction ------------------------------------------------------------
def test_load_document_reads_every_page(sample_pdf):
    document = load_document(sample_pdf)

    assert document.n_pages == 3
    assert len(document.usable_pages) == 3
    assert "Acme Corp" in document.pages[0].text
    assert "control group" in document.pages[1].text
    assert "supply delay" in document.pages[2].text
    assert document.word_count > 10


def test_load_document_page_numbers_are_one_based(sample_pdf):
    document = load_document(sample_pdf)
    assert [page.number for page in document.pages] == [1, 2, 3]


def test_load_document_rejects_scanned_pdf(empty_pdf):
    with pytest.raises(PDFExtractionError, match="scanned"):
        load_document(empty_pdf)


def test_load_document_rejects_garbage():
    with pytest.raises(PDFExtractionError):
        load_document(b"this is not a pdf")


def test_file_hash_is_stable_and_content_sensitive(sample_pdf):
    assert file_sha256(sample_pdf) == file_sha256(bytes(sample_pdf))
    assert file_sha256(sample_pdf) != file_sha256(sample_pdf + b"\n")


# --- cleaning --------------------------------------------------------------
def test_normalize_text_joins_hyphenated_wraps():
    assert normalize_text("inter-\nnational effort") == "international effort"


def test_normalize_text_collapses_noise_but_keeps_paragraphs():
    cleaned = normalize_text("A   sentence.\n\n\n\nSecond\tparagraph.\x00")
    assert cleaned == "A sentence.\n\nSecond paragraph."


def test_normalize_text_handles_empty():
    assert normalize_text("") == ""


# --- page filtering --------------------------------------------------------
def test_filter_pages_is_inclusive():
    pages = [Page(number=i, text=f"page {i}") for i in range(1, 6)]
    selected = filter_pages(pages, 2, 4)
    assert [page.number for page in selected] == [2, 3, 4]


def test_filter_pages_drops_empty_pages():
    pages = [Page(number=1, text=""), Page(number=2, text="content")]
    assert [page.number for page in filter_pages(pages)] == [2]


def test_filter_pages_rejects_inverted_range():
    pages = [Page(number=i, text="text") for i in range(1, 4)]
    with pytest.raises(ValueError):
        filter_pages(pages, 3, 1)


# --- chunking --------------------------------------------------------------
def _long_pages() -> list[Page]:
    body = (
        "Sentence one describes the rollout plan in detail. "
        "Sentence two reports revenue of 4.2 million dollars. "
        "Sentence three names the responsible team. "
    )
    return [Page(number=number, text=body * 6) for number in range(1, 4)]


def test_chunk_pages_respects_size_limit():
    chunks = chunk_pages(_long_pages(), chunk_chars=2000, overlap_chars=200)
    assert len(chunks) > 1
    assert all(len(chunk.text) <= 2000 for chunk in chunks)


def test_chunk_pages_loses_no_sentences():
    pages = _long_pages()
    chunks = chunk_pages(pages, chunk_chars=1500, overlap_chars=100)
    combined = " ".join(chunk.text for chunk in chunks)
    for sentence in (
        "Sentence one describes the rollout plan in detail.",
        "Sentence two reports revenue of 4.2 million dollars.",
        "Sentence three names the responsible team.",
    ):
        assert combined.count(sentence) >= len(pages)


def test_chunk_pages_tracks_source_pages():
    chunks = chunk_pages(_long_pages(), chunk_chars=1500, overlap_chars=100)
    for chunk in chunks:
        assert chunk.pages, "every chunk needs page attribution"
        assert list(chunk.pages) == sorted(chunk.pages)
        assert all(1 <= number <= 3 for number in chunk.pages)


def test_chunk_pages_creates_overlap():
    chunks = chunk_pages(_long_pages(), chunk_chars=1200, overlap_chars=300)
    tail_words = set(chunks[0].text.split()[-12:])
    head_words = set(chunks[1].text.split()[:40])
    assert tail_words & head_words


def test_chunk_pages_splits_words_longer_than_the_limit():
    pages = [Page(number=1, text="x" * 9000)]
    chunks = chunk_pages(pages, chunk_chars=2000, overlap_chars=0)
    assert len(chunks) > 1
    assert all(len(chunk.text) <= 2000 for chunk in chunks)
    assert sum(len(chunk.text) for chunk in chunks) == 9000


def test_chunk_pages_clamps_absurd_settings():
    chunks = chunk_pages([Page(number=1, text="word " * 500)], chunk_chars=10, overlap_chars=9999)
    assert all(len(chunk.text) <= pdf_utils.config.MIN_CHUNK_CHARS for chunk in chunks)


def test_chunk_pages_empty_input():
    assert chunk_pages([]) == []


def test_chunk_indexes_are_sequential():
    chunks = chunk_pages(_long_pages(), chunk_chars=1500, overlap_chars=100)
    assert [chunk.index for chunk in chunks] == list(range(len(chunks)))


def test_page_label_formats():
    assert chunk_pages([Page(number=4, text="abc")])[0].page_label == "p.4"
    chunks = chunk_pages(_long_pages(), chunk_chars=100_000)
    assert chunks[0].page_label.startswith("pp.")


# --- tokens ----------------------------------------------------------------
def test_count_tokens_is_positive_for_text():
    assert count_tokens("hello world, this is a sentence.") > 0


def test_count_tokens_empty():
    assert count_tokens("") == 0


def test_token_fallback_uses_character_heuristic(monkeypatch):
    monkeypatch.setattr(pdf_utils, "_ENCODING_UNAVAILABLE", True)
    text = "a" * 400
    assert count_tokens(text) == estimate_tokens(text) == 100
