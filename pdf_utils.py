"""PDF extraction, text cleaning, chunking and token accounting.

Chunking is implemented locally rather than via ``langchain-text-splitters`` so
the runtime dependency list stays exactly as pinned in ``requirements.txt``.
The splitter is paragraph/sentence aware and carries page numbers through to
every chunk, which is what makes per-page citations possible later.
"""

from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass
from typing import Iterable, Sequence

import config

__all__ = [
    "PDFExtractionError",
    "Page",
    "Chunk",
    "Document",
    "load_document",
    "filter_pages",
    "chunk_pages",
    "count_tokens",
    "estimate_tokens",
    "file_sha256",
    "normalize_text",
]


class PDFExtractionError(RuntimeError):
    """Raised when a PDF cannot be read, with a message safe to show users."""


@dataclass(frozen=True)
class Page:
    """A single PDF page with 1-based numbering."""

    number: int
    text: str


@dataclass(frozen=True)
class Chunk:
    """A contiguous slice of the document, remembering its source pages."""

    index: int
    text: str
    pages: tuple[int, ...]

    @property
    def page_label(self) -> str:
        if not self.pages:
            return "unknown page"
        if len(self.pages) == 1:
            return f"p.{self.pages[0]}"
        return f"pp.{self.pages[0]}-{self.pages[-1]}"


@dataclass(frozen=True)
class Document:
    """Extracted text plus the little metadata pypdf exposes."""

    pages: tuple[Page, ...]
    title: str | None
    author: str | None
    n_pages: int

    @property
    def usable_pages(self) -> tuple[Page, ...]:
        """Pages that actually contain text (scans are empty strings)."""
        return tuple(page for page in self.pages if page.text.strip())

    @property
    def full_text(self) -> str:
        return "\n\n".join(page.text for page in self.usable_pages)

    @property
    def word_count(self) -> int:
        return len(self.full_text.split())


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #
def load_document(data: bytes, password: str | None = None) -> Document:
    """Read a PDF from raw bytes.

    Raises ``PDFExtractionError`` for encrypted files, corrupt files, and files
    with no extractable text (usually scans needing OCR).
    """
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(io.BytesIO(data))
    except (PdfReadError, ValueError, OSError) as exc:
        raise PDFExtractionError(f"Could not open this PDF: {exc}") from exc

    if reader.is_encrypted:
        try:
            # An empty password unlocks PDFs that are only "restricted", which
            # is common for exported reports.
            unlocked = reader.decrypt(password or "")
        except Exception as exc:  # noqa: BLE001 - pypdf raises assorted types
            raise PDFExtractionError("This PDF is encrypted.") from exc
        if not unlocked:
            raise PDFExtractionError(
                "This PDF is password protected. Enter its password to continue."
            )

    pages: list[Page] = []
    for offset, raw_page in enumerate(reader.pages):
        try:
            text = raw_page.extract_text() or ""
        except Exception:  # noqa: BLE001 - one bad page shouldn't kill the doc
            text = ""
        pages.append(Page(number=offset + 1, text=normalize_text(text)))

    info = reader.metadata or {}
    document = Document(
        pages=tuple(pages),
        title=_clean_meta(info.get("/Title")),
        author=_clean_meta(info.get("/Author")),
        n_pages=len(pages),
    )

    if not document.usable_pages:
        raise PDFExtractionError(
            "No extractable text found. This looks like a scanned PDF, which "
            "needs OCR before it can be summarised."
        )
    return document


def _clean_meta(value: object) -> str | None:
    if not value:
        return None
    text = str(value).strip()
    return text or None


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def normalize_text(text: str) -> str:
    """Tidy raw pypdf output without destroying paragraph structure."""
    if not text:
        return ""

    text = text.replace("\xa0", " ").replace("\u00ad", "")
    text = _CONTROL_CHARS.sub("", text)
    text = _HYPHEN_BREAK.sub(r"\1\2", text)

    # Collapse runs of spaces but keep newlines so paragraphs survive.
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def filter_pages(
    pages: Sequence[Page], first: int | None = None, last: int | None = None
) -> list[Page]:
    """Keep pages within an inclusive 1-based range."""
    lo = max(1, first or 1)
    hi = last if last is not None else pages[-1].number if pages else 1
    if hi < lo:
        raise ValueError(f"Page range {lo}-{hi} is empty; 'from' must be <= 'to'.")
    return [page for page in pages if lo <= page.number <= hi and page.text.strip()]


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")


def _hard_split(text: str, limit: int) -> list[str]:
    return [text[i : i + limit] for i in range(0, len(text), limit)]


def _split_oversized(block: str, limit: int) -> list[str]:
    """Break a block that exceeds ``limit`` on word boundaries where possible."""
    if len(block) <= limit:
        return [block]

    out: list[str] = []
    current = ""
    for word in block.split():
        if len(word) > limit:
            if current:
                out.append(current)
                current = ""
            out.extend(_hard_split(word, limit))
            continue
        candidate = f"{current} {word}".strip()
        if len(candidate) > limit:
            out.append(current)
            current = word
        else:
            current = candidate
    if current:
        out.append(current)
    return out or [block[:limit]]


def _page_pieces(page: Page, limit: int) -> Iterable[tuple[int, str]]:
    """Yield ``(page_number, piece)`` blocks no longer than ``limit``."""
    for paragraph in _PARAGRAPH_SPLIT.split(page.text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) <= limit:
            yield page.number, paragraph
            continue
        for sentence in _SENTENCE_SPLIT.split(paragraph):
            sentence = sentence.strip()
            if not sentence:
                continue
            for piece in _split_oversized(sentence, limit):
                yield page.number, piece


def _overlap_tail(text: str, overlap: int) -> str:
    """Take up to ``overlap`` trailing characters, snapped to a word start."""
    if overlap <= 0 or not text:
        return ""
    tail = text[-overlap:]
    space = tail.find(" ")
    if space != -1:
        tail = tail[space + 1 :]
    return tail.strip()


def chunk_pages(
    pages: Sequence[Page],
    chunk_chars: int = config.DEFAULT_CHUNK_CHARS,
    overlap_chars: int = config.DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    """Pack pages into overlapping chunks, preserving page attribution."""
    chunk_chars = max(config.MIN_CHUNK_CHARS, min(chunk_chars, config.MAX_CHUNK_CHARS))
    overlap_chars = max(0, min(overlap_chars, chunk_chars // 2))

    chunks: list[Chunk] = []
    current = ""
    current_pages: list[int] = []
    has_new_content = False

    def flush() -> None:
        nonlocal current, current_pages, has_new_content
        text = current.strip()
        # A chunk made only of carried-over overlap text adds nothing.
        if text and has_new_content:
            chunks.append(
                Chunk(
                    index=len(chunks),
                    text=text,
                    pages=tuple(dict.fromkeys(current_pages)),
                )
            )
        current = ""
        current_pages = []
        has_new_content = False

    for page in pages:
        for page_number, piece in _page_pieces(page, chunk_chars):
            candidate = f"{current}\n\n{piece}" if current else piece
            if current and len(candidate) > chunk_chars:
                flush()
                tail = _overlap_tail(chunks[-1].text if chunks else "", overlap_chars)
                if tail and len(tail) + len(piece) + 2 <= chunk_chars:
                    current = f"{tail}\n\n{piece}"
                    current_pages = [chunks[-1].pages[-1], page_number]
                else:
                    current = piece
                    current_pages = [page_number]
                has_new_content = True
                continue

            current = candidate
            has_new_content = True
            if not current_pages or current_pages[-1] != page_number:
                current_pages.append(page_number)

    flush()
    return chunks


# --------------------------------------------------------------------------- #
# Tokens
# --------------------------------------------------------------------------- #
_ENCODING = None
_ENCODING_UNAVAILABLE = False


def estimate_tokens(text: str) -> int:
    """Rough token estimate used when tiktoken is unavailable."""
    return max(0, len(text) // 4)


def count_tokens(text: str) -> int:
    """Count tokens with tiktoken's cl100k encoding.

    This is an approximation of Gemini's tokeniser, good enough to decide
    between single-pass and map-reduce. Falls back to a character heuristic if
    the encoding cannot be downloaded (offline / first run without network).
    """
    global _ENCODING, _ENCODING_UNAVAILABLE

    if not text:
        return 0
    if _ENCODING_UNAVAILABLE:
        return estimate_tokens(text)
    if _ENCODING is None:
        try:
            import tiktoken

            _ENCODING = tiktoken.get_encoding("cl100k_base")
        except Exception:  # noqa: BLE001 - offline, missing cache, etc.
            _ENCODING_UNAVAILABLE = True
            return estimate_tokens(text)
    try:
        return len(_ENCODING.encode(text, disallowed_special=()))
    except Exception:  # noqa: BLE001
        _ENCODING_UNAVAILABLE = True
        return estimate_tokens(text)


def file_sha256(data: bytes) -> str:
    """Stable cache key for an uploaded document."""
    return hashlib.sha256(data).hexdigest()
