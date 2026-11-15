from __future__ import annotations

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

import config
import summarizer as sz
from pdf_utils import Chunk

OPTIONS = sz.SummaryOptions(style="Bullet points", length="Short")


def _chunks() -> list[Chunk]:
    return [
        Chunk(index=0, text="alpha content", pages=(1,)),
        Chunk(index=1, text="beta content", pages=(2,)),
        Chunk(index=2, text="gamma content", pages=(3,)),
    ]


class BoomChat(FakeListChatModel):
    """Chat model whose every call fails, for error-path tests."""

    def _call(self, *args, **kwargs):  # type: ignore[override]
        raise RuntimeError("provider exploded")


# --- strategy --------------------------------------------------------------
@pytest.mark.parametrize(
    "tokens,expected",
    [
        (0, "single-pass"),
        (config.SINGLE_PASS_TOKEN_LIMIT, "single-pass"),
        (config.SINGLE_PASS_TOKEN_LIMIT + 1, "map-reduce"),
        (10_000_000, "map-reduce"),
    ],
)
def test_plan_strategy_threshold(tokens, expected):
    assert sz.plan_strategy(tokens) == expected


# --- options ---------------------------------------------------------------
def test_summary_options_reject_unknown_style():
    with pytest.raises(ValueError, match="style"):
        sz.SummaryOptions(style="Haiku")


def test_summary_options_reject_unknown_length():
    with pytest.raises(ValueError, match="length"):
        sz.SummaryOptions(length="Epic")


def test_cache_key_ignores_whitespace_only_custom_text():
    a = sz.SummaryOptions(custom_instruction="  ")
    b = sz.SummaryOptions(custom_instruction="")
    assert a.cache_key == b.cache_key


# --- prompts ---------------------------------------------------------------
def test_prompt_contains_style_and_length_guidance():
    prompt = sz.summary_prompt_preview(OPTIONS)
    assert sz.STYLES["Bullet points"] in prompt
    assert sz.LENGTHS["Short"] in prompt


def test_prompt_includes_custom_instruction_last():
    options = sz.SummaryOptions(custom_instruction="Focus only on cost figures.")
    prompt = sz.summary_prompt_preview(options)
    assert "Focus only on cost figures." in prompt
    assert prompt.index("<instructions>") > prompt.index(sz.STYLES[options.style])


def test_prompt_omits_instruction_block_when_empty():
    assert "<instructions>" not in sz.summary_prompt_preview(sz.SummaryOptions())


# --- map phase -------------------------------------------------------------
def test_map_chunk_notes_preserves_order_and_filters_boilerplate():
    llm = FakeListChatModel(
        responses=["No substantive content", "- first finding", "- second finding"]
    )
    notes = sz.map_chunk_notes(llm, _chunks(), concurrency=1)

    assert notes == ["[p.2] - first finding", "[p.3] - second finding"]


def test_map_chunk_notes_reports_progress():
    seen: list[tuple[int, int]] = []
    llm = FakeListChatModel(responses=["- a", "- b"])
    sz.map_chunk_notes(llm, _chunks()[:2], concurrency=1, progress=lambda d, t: seen.append((d, t)))
    assert seen == [(1, 2), (2, 2)]


def test_map_chunk_notes_without_chunks_is_empty():
    assert sz.map_chunk_notes(FakeListChatModel(responses=["x"]), []) == []


def test_map_chunk_notes_wraps_failures():
    with pytest.raises(sz.ChunkFailure):
        sz.map_chunk_notes(BoomChat(responses=["x"]), _chunks()[:1])


def test_chunk_failure_message_names_the_page():
    failure = sz.ChunkFailure(0, Chunk(index=0, text="x", pages=(7,)))
    assert "p.7" in str(failure)


# --- reduce phase ----------------------------------------------------------
def test_reduce_chain_returns_text():
    llm = FakeListChatModel(responses=["final summary"])
    chain = sz.build_reduce_chain(llm, OPTIONS)
    result = chain.invoke(
        {"notes": "[p.1] - alpha", "doc_title": "Doc", "page_label": "pp.1-3"}
    )
    assert result == "final summary"


def test_single_pass_chain_returns_text():
    llm = FakeListChatModel(responses=["one shot"])
    chain = sz.build_single_pass_chain(llm, OPTIONS)
    assert chain.invoke({"text": "body", "doc_title": "Doc", "page_label": "p.1"}) == "one shot"


def test_qa_style_results_report_call_counts():
    single = sz.SummaryResult(text="t", strategy="single-pass", chunk_count=9)
    mapped = sz.SummaryResult(text="t", strategy="map-reduce", chunk_count=9)
    assert single.api_calls == 1
    assert mapped.api_calls == 10


def test_format_notes_handles_empty_list():
    assert "no substantive content" in sz.format_notes([])
    assert sz.format_notes(["[p.1] - a", "[p.2] - b"]) == "[p.1] - a\n\n[p.2] - b"


def test_styles_and_lengths_are_aligned_with_ui_expectations():
    assert len(sz.STYLES) == 4
    assert set(sz.LENGTHS) == {"Short", "Medium", "Long"}
