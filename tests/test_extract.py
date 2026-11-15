from __future__ import annotations

import json

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

import extract
from extract import (
    DocumentFacts,
    ExtractionError,
    KeyFact,
    extract_facts,
    facts_to_frame,
    facts_to_markdown,
    metadata_to_frame,
    parse_facts_json,
)

VALID = {
    "title": "Quarterly Report",
    "document_type": "financial report",
    "author_or_org": "Acme Corp",
    "date": "Q3 2026",
    "key_facts": [
        {"category": "metric", "detail": "1200 units shipped in Q3", "page": "p.1"},
        {"category": "risk", "detail": "Supply delay", "page": "p.3"},
    ],
    "entities": ["Acme Corp"],
    "action_items": ["Continue the rollout"],
}


class NoStructuredOutput(FakeListChatModel):
    """Fails the schema path so the JSON fallback is exercised."""

    def with_structured_output(self, *args, **kwargs):  # type: ignore[override]
        raise NotImplementedError("no schema support here")


def test_parse_facts_json_plain():
    facts = parse_facts_json(json.dumps(VALID))
    assert isinstance(facts, DocumentFacts)
    assert facts.title == "Quarterly Report"
    assert len(facts.key_facts) == 2


def test_parse_facts_json_with_code_fence_and_chatter():
    raw = "Sure, here you go:\n```json\n" + json.dumps(VALID) + "\n```\nHope that helps."
    assert parse_facts_json(raw).author_or_org == "Acme Corp"


def test_parse_facts_json_rejects_prose():
    with pytest.raises(ExtractionError):
        parse_facts_json("I could not find any facts.")


def test_parse_facts_json_rejects_empty():
    with pytest.raises(ExtractionError):
        parse_facts_json("   ")


def test_extract_facts_falls_back_to_json_chain():
    llm = NoStructuredOutput(responses=[json.dumps(VALID)])
    facts = extract_facts(llm, "document body", doc_title="Doc")
    assert facts.title == "Quarterly Report"


def test_facts_to_frame_shape_and_order():
    facts = DocumentFacts.model_validate(VALID)
    frame = facts_to_frame(facts)
    assert list(frame.columns) == ["Category", "Detail", "Page"]
    assert frame.iloc[0]["Category"] == "metric"
    assert frame.iloc[1]["Page"] == "p.3"


def test_facts_to_frame_skips_blank_details():
    facts = DocumentFacts(
        key_facts=[KeyFact(category="other", detail="  "), KeyFact(category="other", detail="kept")]
    )
    frame = facts_to_frame(facts)
    assert len(frame) == 1
    assert frame.iloc[0]["Detail"] == "kept"


def test_metadata_to_frame_lists_entities_and_actions():
    frame = metadata_to_frame(DocumentFacts.model_validate(VALID))
    labels = list(frame["Field"])
    assert "Title" in labels and "Entity" in labels and "Action / obligation" in labels


def test_metadata_to_frame_drops_nulls():
    frame = metadata_to_frame(DocumentFacts(title=None, date="null"))
    assert frame.empty


def test_facts_to_markdown_contains_tables():
    markdown = facts_to_markdown(DocumentFacts.model_validate(VALID))
    assert "| Field | Value |" in markdown
    assert "| Category | Detail | Page |" in markdown
    assert "1200 units shipped in Q3" in markdown


def test_schema_is_json_serialisable():
    """The JSON fallback prompt embeds this schema, so it must serialise."""
    schema = DocumentFacts.model_json_schema()
    assert "key_facts" in schema["properties"]
    assert json.dumps(schema)
