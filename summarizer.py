"""LangChain chains for summarisation.

Two strategies:

* **single-pass** – the whole document fits comfortably in one prompt.
* **map-reduce** – take notes chunk by chunk (concurrently), then synthesise a
  final summary from the notes. Required for long PDFs, and it also means a
  style change does not force us to re-read the document.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, Sequence

import config
from pdf_utils import Chunk

__all__ = [
    "STYLES",
    "LENGTHS",
    "SummaryOptions",
    "SummaryResult",
    "build_llm",
    "plan_strategy",
    "build_map_chain",
    "build_reduce_chain",
    "build_single_pass_chain",
    "map_chunk_notes",
    "format_notes",
    "summary_prompt_preview",
]

ProgressCallback = Callable[[int, int], None]

STYLES: dict[str, str] = {
    "Executive brief": (
        "Style: an executive brief for a senior decision-maker. Open with the "
        "bottom line in one or two sentences, then the supporting points, then "
        "risks, open questions and recommended next steps."
    ),
    "Bullet points": (
        "Style: a scannable bulleted list. One idea per bullet, no more than "
        "two lines each. Group bullets under short bold headings when the "
        "document covers distinct topics."
    ),
    "Technical deep-dive": (
        "Style: a technical brief for a specialist reader. Cover the problem, "
        "the approach or method, data and parameters, results with their "
        "numbers, limitations, and what remains unresolved."
    ),
    "One paragraph": (
        "Style: a single dense paragraph. No bullets, no headings, no line "
        "breaks other than paragraph breaks."
    ),
}

LENGTHS: dict[str, str] = {
    "Short": "Length: roughly 150 words.",
    "Medium": "Length: roughly 300 words.",
    "Long": "Length: roughly 700 words.",
}

_MAP_SYSTEM = """You are a meticulous document analyst. You extract factual notes from one portion of a longer document.

Rules:
- Capture names, organisations, dates, figures, definitions, claims and conclusions.
- Keep the author's own terminology; do not editorialise.
- Use 3-8 short bullet points, nothing else.
- If this portion is a table of contents, bibliography, boilerplate or otherwise carries no substantive content, reply with exactly: No substantive content."""

_MAP_HUMAN = """Document portion {index} of {total} ({page_label}):

<portion>
{text}
</portion>"""


@dataclass(frozen=True)
class SummaryOptions:
    """User-facing knobs that change the final summary."""

    style: str = "Executive brief"
    length: str = "Medium"
    custom_instruction: str = ""

    def __post_init__(self) -> None:
        if self.style not in STYLES:
            raise ValueError(f"Unknown style {self.style!r}; expected one of {list(STYLES)}")
        if self.length not in LENGTHS:
            raise ValueError(f"Unknown length {self.length!r}; expected one of {list(LENGTHS)}")

    @property
    def cache_key(self) -> tuple[str, str, str]:
        return (self.style, self.length, self.custom_instruction.strip())


@dataclass(frozen=True)
class SummaryResult:
    text: str
    strategy: str
    chunk_count: int
    notes: tuple[str, ...] = ()

    @property
    def api_calls(self) -> int:
        return 1 if self.strategy == "single-pass" else self.chunk_count + 1


# --------------------------------------------------------------------------- #
# Model + chains
# --------------------------------------------------------------------------- #
def build_llm(
    model: str | None = None,
    temperature: float = config.DEFAULT_TEMPERATURE,
    max_output_tokens: int = config.DEFAULT_MAX_OUTPUT_TOKENS,
    api_key: str | None = None,
):
    """Create the Gemini chat model.

    ``max_retries`` lets the google-genai client ride out 429/5xx responses with
    its own exponential backoff, which matters on the free tiers.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=model or config.DEFAULT_MODEL,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        max_retries=4,
        timeout=180,
        google_api_key=api_key or config.require_api_key(),
    )


def plan_strategy(token_count: int) -> str:
    """Pick the cheapest strategy that still gives a faithful summary."""
    return "single-pass" if token_count <= config.SINGLE_PASS_TOKEN_LIMIT else "map-reduce"


def _reduce_system_prompt(options: SummaryOptions, source: str) -> str:
    parts = [
        f"You are an expert analyst writing a summary of a document {source}.",
        STYLES[options.style],
        LENGTHS[options.length],
        "Rules:\n"
        "- Use only the supplied material; never invent facts or fill gaps.\n"
        "- Preserve exact figures, dates, names and terminology.\n"
        "- No preamble such as 'Here is a summary'; start with the substance.\n"
        "- If the material is thin, say what the document covers instead of padding.",
    ]
    custom = options.custom_instruction.strip()
    if custom:
        parts.append(
            "Additional instructions from the reader (these take precedence where "
            f"they conflict with the guidance above):\n<instructions>\n{custom}\n</instructions>"
        )
    return "\n\n".join(parts)


def build_map_chain(llm):
    """Chain that turns one chunk into bullet notes."""
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages(
        [("system", _MAP_SYSTEM), ("human", _MAP_HUMAN)]
    )
    return prompt | llm | StrOutputParser()


def build_reduce_chain(llm, options: SummaryOptions):
    """Chain that synthesises notes into the final summary."""
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _reduce_system_prompt(options, "from notes taken section by section")),
            (
                "human",
                "Document: {doc_title}\nCovered: {page_label}\n\n"
                "Section notes:\n<notes>\n{notes}\n</notes>\n\nWrite the final summary.",
            ),
        ]
    )
    return prompt | llm | StrOutputParser()


def build_single_pass_chain(llm, options: SummaryOptions):
    """Chain that summarises the whole document in one call."""
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _reduce_system_prompt(options, "reproduced in full below")),
            (
                "human",
                "Document: {doc_title}\nCovered: {page_label}\n\n"
                "<document>\n{text}\n</document>\n\nWrite the summary.",
            ),
        ]
    )
    return prompt | llm | StrOutputParser()


def summary_prompt_preview(options: SummaryOptions) -> str:
    """System prompt for the final step, shown in the UI's debug expander."""
    return _reduce_system_prompt(options, "from notes taken section by section")


# --------------------------------------------------------------------------- #
# Execution helpers
# --------------------------------------------------------------------------- #
def map_chunk_notes(
    llm,
    chunks: Sequence[Chunk],
    concurrency: int = config.DEFAULT_CONCURRENCY,
    progress: ProgressCallback | None = None,
) -> list[str]:
    """Summarise every chunk into notes, preserving chunk order.

    ``progress`` is called as ``progress(done, total)`` after each chunk.
    """
    if not chunks:
        return []

    chain = build_map_chain(llm)
    total = len(chunks)
    results: list[str | None] = [None] * total
    done = 0

    def payload(chunk: Chunk) -> dict[str, object]:
        return {
            "text": chunk.text,
            "page_label": chunk.page_label,
            "index": chunk.index + 1,
            "total": total,
        }

    workers = max(1, min(concurrency, total, 8))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(chain.invoke, payload(chunk)): i for i, chunk in enumerate(chunks)}
        for future in as_completed(futures):
            position = futures[future]
            try:
                results[position] = future.result()
            except Exception as exc:  # noqa: BLE001 - surfaced by the caller
                raise ChunkFailure(position, chunks[position]) from exc
            done += 1
            if progress:
                progress(done, total)

    notes: list[str] = []
    for chunk, note in zip(chunks, results):
        text = (note or "").strip()
        # Empty/boilerplate notes would only dilute the reduce prompt.
        if text and "no substantive content" not in text.lower():
            notes.append(f"[{chunk.page_label}] {text}")
    return notes


class ChunkFailure(RuntimeError):
    """A single chunk failed after the client's own retries."""

    def __init__(self, position: int, chunk: Chunk) -> None:
        super().__init__(f"Failed while reading {chunk.page_label} (chunk {position + 1}).")
        self.position = position
        self.chunk = chunk


def format_notes(notes: Sequence[str]) -> str:
    """Render notes for the reduce prompt."""
    if not notes:
        return "(no substantive content was found in the selected pages)"
    return "\n\n".join(notes)
