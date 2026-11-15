"""Structured key-data extraction rendered as tables.

The LLM is asked for a typed object (Gemini supports response schemas), with a
JSON-parsing fallback for models or responses that reject the schema.
"""

from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field, ValidationError

__all__ = [
    "ExtractionError",
    "KeyFact",
    "DocumentFacts",
    "build_extraction_chain",
    "extract_facts",
    "parse_facts_json",
    "facts_to_frame",
    "metadata_to_frame",
    "facts_to_markdown",
]


class ExtractionError(RuntimeError):
    """Raised when structured extraction fails or returns unusable data."""


class KeyFact(BaseModel):
    """One atomic fact lifted out of the document."""

    category: str = Field(
        description=(
            "Short category: finding, decision, metric, date, person, "
            "organisation, risk, action or other"
        )
    )
    detail: str = Field(
        description="The fact in one sentence, keeping exact names and numbers; no page reference"
    )
    page: str | None = Field(default=None, description="Page reference such as 'p.4' if known")


class DocumentFacts(BaseModel):
    """Everything the app shows in the Key Data tab."""

    title: str | None = Field(default=None, description="Document title, if stated")
    document_type: str | None = Field(
        default=None,
        description=(
            "Short genre label such as research paper, contract, financial report, "
            "policy brief or meeting minutes; never the title"
        ),
    )
    author_or_org: str | None = Field(default=None, description="Author or issuing organisation")
    date: str | None = Field(default=None, description="Publication or effective date as written")
    key_facts: list[KeyFact] = Field(
        default_factory=list, description="The 10-25 most decision-relevant facts"
    )
    entities: list[str] = Field(
        default_factory=list, description="Named people, organisations, products, places"
    )
    action_items: list[str] = Field(
        default_factory=list, description="Explicit actions, recommendations or obligations"
    )


_EXTRACT_SYSTEM = """You extract structured reference data from a document.

Rules:
- Only record what the text states. Never infer or complete missing values.
- Keep figures, dates and proper nouns exactly as written.
- Use null for a field the document does not state.
- document_type must be a short genre label such as "research paper", "contract",
  "financial report", "policy brief" or "meeting minutes". Never repeat the title.
- Prefer 10-25 genuinely decision-relevant key facts over exhaustive trivia.
- Put page references in the page field only (e.g. "p.12"); never write them
  inside detail."""

_EXTRACT_HUMAN = """Document: {doc_title}
Covered: {page_label}

<material>
{text}
</material>"""

_JSON_SCHEMA_INSTRUCTION = (
    "\n\nReturn ONLY a JSON object matching this JSON Schema (no prose, no code fences):\n{schema}"
)


def _schema_json() -> str:
    return json.dumps(DocumentFacts.model_json_schema(), indent=2)


def build_extraction_chain(llm):
    """Chain returning a ``DocumentFacts`` instance."""
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages(
        [("system", _EXTRACT_SYSTEM), ("human", _EXTRACT_HUMAN)]
    )
    return prompt | llm.with_structured_output(DocumentFacts)


def build_json_fallback_chain(llm):
    """Chain that asks for raw JSON, for models without schema support."""
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages(
        [("system", _EXTRACT_SYSTEM + _JSON_SCHEMA_INSTRUCTION), ("human", _EXTRACT_HUMAN)]
    )
    return prompt | llm | StrOutputParser()


def _coerce(result: Any) -> DocumentFacts:
    if isinstance(result, DocumentFacts):
        return result
    if isinstance(result, dict):
        return DocumentFacts.model_validate(result)
    raise ExtractionError(f"Unexpected extraction result of type {type(result).__name__}.")


def extract_facts(
    llm,
    text: str,
    doc_title: str | None = None,
    page_label: str = "",
) -> DocumentFacts:
    """Extract structured facts, falling back to JSON parsing if needed."""
    payload = {
        "text": text,
        "doc_title": doc_title or "untitled document",
        "page_label": page_label or "whole document",
    }

    try:
        return _coerce(build_extraction_chain(llm).invoke(payload))
    except Exception as structured_error:  # noqa: BLE001 - fallback is intentional
        try:
            fallback_payload = dict(payload, schema=_schema_json())
            raw = build_json_fallback_chain(llm).invoke(fallback_payload)
            return parse_facts_json(raw)
        except Exception as fallback_error:  # noqa: BLE001
            raise ExtractionError(
                f"Could not extract structured data ({structured_error}); "
                f"JSON fallback also failed ({fallback_error})."
            ) from fallback_error


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def parse_facts_json(raw: str) -> DocumentFacts:
    """Parse a JSON object out of a model reply, tolerant of fences and chatter."""
    if not raw or not raw.strip():
        raise ExtractionError("The model returned an empty response.")

    candidates: list[str] = []
    candidates.extend(match.strip() for match in _FENCE.findall(raw))
    candidates.append(raw.strip())

    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end > start:
        candidates.append(raw[start : end + 1])

    for candidate in candidates:
        try:
            return DocumentFacts.model_validate(json.loads(candidate))
        except (json.JSONDecodeError, ValidationError):
            continue
    raise ExtractionError("The model's reply did not contain valid JSON for the fact schema.")


def facts_to_frame(facts: DocumentFacts) -> pd.DataFrame:
    """Key facts as a three-column table."""
    rows = [
        {
            "Category": (fact.category or "other").strip().lower(),
            "Detail": fact.detail.strip(),
            "Page": (fact.page or "").strip(),
        }
        for fact in facts.key_facts
        if fact.detail and fact.detail.strip()
    ]
    return pd.DataFrame(rows, columns=["Category", "Detail", "Page"])


def metadata_to_frame(facts: DocumentFacts) -> pd.DataFrame:
    """Document-level fields plus the list-valued extras, as label/value rows."""
    rows: list[dict[str, str]] = []

    def add(label: str, value: str | None) -> None:
        if value and str(value).strip() and str(value).strip().lower() != "null":
            rows.append({"Field": label, "Value": str(value).strip()})

    add("Title", facts.title)
    add("Type", facts.document_type)
    add("Author / organisation", facts.author_or_org)
    add("Date", facts.date)
    for entity in facts.entities:
        add("Entity", entity)
    for action in facts.action_items:
        add("Action / obligation", action)

    return pd.DataFrame(rows, columns=["Field", "Value"])


def facts_to_markdown(facts: DocumentFacts) -> str:
    """Markdown export for the Key Data tab."""
    lines: list[str] = ["## Key data", ""]
    metadata = metadata_to_frame(facts)
    if not metadata.empty:
        lines += ["| Field | Value |", "| --- | --- |"]
        lines += [f"| {row.Field} | {row.Value} |" for row in metadata.itertuples()]
        lines.append("")

    frame = facts_to_frame(facts)
    if not frame.empty:
        lines += ["### Key facts", "", "| Category | Detail | Page |", "| --- | --- | --- |"]
        lines += [f"| {row.Category} | {row.Detail} | {row.Page} |" for row in frame.itertuples()]
        lines.append("")
    return "\n".join(lines)
