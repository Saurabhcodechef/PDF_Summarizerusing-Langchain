"""Streamlit UI for the PDF summariser.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Sequence

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

import config
import extract as facts_mod
import rag
import summarizer as sz
import theme
from pdf_utils import (
    Document,
    PDFExtractionError,
    Page,
    chunk_pages,
    count_tokens,
    file_sha256,
    filter_pages,
    load_document,
)

st.set_page_config(page_title="PDF Summarizer · Gemini", page_icon="📄", layout="wide")


# --------------------------------------------------------------------------- #
# Cached / session-scoped work
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False, max_entries=3)
def _load_document_cached(data: bytes, password: str) -> Document:
    return load_document(data, password)


def _state_store(name: str) -> dict:
    return st.session_state.setdefault(name, {})


def _page_label(pages: Sequence[Page]) -> str:
    numbers = [page.number for page in pages]
    if not numbers:
        return "no pages"
    if len(numbers) == 1:
        return f"p.{numbers[0]}"
    return f"pp.{numbers[0]}-{numbers[-1]}"


def _friendly_error(exc: Exception) -> str:
    """Turn provider errors into something a reader can act on."""
    text = str(exc)
    low = text.lower()
    if "api key" in low or "api_key" in low or "unauthenticated" in low or "permission" in low:
        return f"Authentication failed. Check GEMINI_API_KEY in .env.\n\nDetails: {text}"
    if "429" in low or "resource_exhausted" in low or "quota" in low or "rate" in low:
        return (
            "Rate limit or quota reached. Gemini free tiers allow only a few "
            "requests per minute — wait a moment, lower the concurrency in "
            f"Advanced, or switch to a paid key.\n\nDetails: {text}"
        )
    if "404" in low or "not found" in low:
        return f"The model or embedding model was not found. Check the model name.\n\nDetails: {text}"
    if "deadline" in low or "timeout" in low or "timed out" in low:
        return f"The request timed out. Try a smaller page range.\n\nDetails: {text}"
    return text


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
st.sidebar.title("⚙️ Settings")

dark_mode = st.sidebar.toggle(
    "🌗 Dark mode",
    key="dark_mode",
    help="Switches the app between the light and dark palettes.",
)
st.sidebar.divider()

api_key = config.get_api_key()
if not api_key:
    st.sidebar.error(
        f"`{config.API_KEY_ENV}` is missing. Add it to `{config.PROJECT_ROOT / '.env'}` "
        "and restart the app."
    )
else:
    st.sidebar.success("API key loaded", icon="🔑")

model_name = st.sidebar.text_input("Model", value=config.DEFAULT_MODEL)
temperature = st.sidebar.slider(
    "Temperature", 0.0, 1.0, config.DEFAULT_TEMPERATURE, 0.05,
    help="Lower is more literal; higher is more liberal with phrasing.",
)

style = st.sidebar.selectbox("Summary style", list(sz.STYLES), index=0)
length = st.sidebar.radio("Length", list(sz.LENGTHS), index=1, horizontal=True)
custom_instruction = st.sidebar.text_area(
    "Custom instruction",
    placeholder="e.g. Focus on the financial figures and ignore the methodology.",
    height=80,
)

with st.sidebar.expander("Advanced"):
    chunk_chars = st.slider(
        "Chunk size (characters)", config.MIN_CHUNK_CHARS, 20_000,
        config.DEFAULT_CHUNK_CHARS, 500,
        help="Smaller chunks mean more API calls but tighter summaries per call.",
    )
    overlap_chars = st.slider(
        "Chunk overlap (characters)", 0, 2000, config.DEFAULT_CHUNK_OVERLAP, 100
    )
    concurrency = st.slider(
        "Parallel requests", 1, 6, config.DEFAULT_CONCURRENCY,
        help="Free-tier keys throttle aggressively; 2-3 is a safe default.",
    )
    max_output_tokens = st.slider(
        "Max output tokens", 512, 8192, config.DEFAULT_MAX_OUTPUT_TOKENS, 256
    )
    embedding_model_name = st.text_input("Embedding model", value=config.DEFAULT_EMBEDDING_MODEL)
    top_k = st.slider("Excerpts per question", 2, 12, config.RETRIEVAL_TOP_K)

options = sz.SummaryOptions(
    style=style, length=length, custom_instruction=custom_instruction
)

with st.sidebar.expander("Prompt preview"):
    st.code(sz.summary_prompt_preview(options), language="text")


# --------------------------------------------------------------------------- #
# Document loading
# --------------------------------------------------------------------------- #
# Inject the palette before any main-area content so it lands early in the DOM.
theme.apply_theme(bool(dark_mode))

st.title("📄 PDF Summarizer")
st.caption("Gemini + LangChain · extract, summarise, tabulate and question a PDF")

uploaded = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded is None:
    st.info(
        "Upload a PDF to start. Text-based PDFs work best — scans need OCR first. "
        "Nothing is stored; the file stays in this session."
    )
    st.stop()

data = uploaded.getvalue()
doc_hash = file_sha256(data)

password = ""
try:
    document = _load_document_cached(data, password)
except PDFExtractionError as exc:
    if "password" in str(exc).lower():
        password = st.text_input("PDF password", type="password")
        try:
            document = _load_document_cached(data, password)
        except PDFExtractionError as inner:
            st.error(_friendly_error(inner))
            st.stop()
    else:
        st.error(_friendly_error(exc))
        st.stop()

# Page range (widget keys include the file hash so a new upload resets them).
with st.sidebar:
    st.divider()
    st.subheader("Pages")
    max_page = document.n_pages
    col_from, col_to = st.columns(2)
    first_page = col_from.number_input(
        "From", min_value=1, max_value=max_page, value=1, step=1,
        key=f"page_from_{doc_hash[:8]}",
    )
    last_page = col_to.number_input(
        "To", min_value=1, max_value=max_page, value=max_page, step=1,
        key=f"page_to_{doc_hash[:8]}",
    )

try:
    selected_pages = filter_pages(document.pages, int(first_page), int(last_page))
except ValueError as exc:
    st.error(str(exc))
    st.stop()

chunks = chunk_pages(selected_pages, chunk_chars, overlap_chars)

stats_key = (doc_hash, int(first_page), int(last_page))
if st.session_state.get("stats_key") != stats_key:
    selected_text = "\n\n".join(page.text for page in selected_pages)
    st.session_state["stats_key"] = stats_key
    st.session_state["stats"] = {
        "tokens": count_tokens(selected_text),
        "words": len(selected_text.split()),
    }
stats = st.session_state["stats"]

strategy = sz.plan_strategy(stats["tokens"])
api_calls = 1 if strategy == "single-pass" else len(chunks) + 1

meta_cols = st.columns(5)
meta_cols[0].metric("Pages used", f"{len(selected_pages)} / {document.n_pages}")
meta_cols[1].metric("Words", f"{stats['words']:,}")
meta_cols[2].metric("Tokens (approx.)", f"{stats['tokens']:,}")
meta_cols[3].metric("Chunks", len(chunks))
meta_cols[4].metric("Strategy", strategy.replace("-", " ").title())

if document.title:
    st.caption(f"**{document.title}**" + (f" — {document.author}" if document.author else ""))

summary_key = (
    doc_hash, int(first_page), int(last_page), chunk_chars, overlap_chars,
    model_name, round(temperature, 2), max_output_tokens, options.cache_key,
)
notes_key = (
    doc_hash, int(first_page), int(last_page), chunk_chars, overlap_chars,
    model_name, round(temperature, 2), max_output_tokens,
)
facts_key = (
    doc_hash, int(first_page), int(last_page), chunk_chars, overlap_chars,
    model_name, round(temperature, 2),
)
index_key = (
    doc_hash, int(first_page), int(last_page), chunk_chars, overlap_chars,
    embedding_model_name,
)

doc_title = document.title or uploaded.name
pages_label = _page_label(selected_pages)


def make_llm():
    return sz.build_llm(
        model=model_name,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        api_key=api_key,
    )


def ensure_notes(llm) -> list[str]:
    """Map-phase notes, cached per document + model so restyles are free."""
    cache = _state_store("notes")
    if notes_key in cache:
        return cache[notes_key]

    with st.status(f"Reading {len(chunks)} section(s)…", expanded=False) as status:
        bar = st.progress(0.0, text=f"0 / {len(chunks)} sections")

        def on_progress(done: int, total: int) -> None:
            bar.progress(done / total, text=f"{done} / {total} sections")

        notes = sz.map_chunk_notes(
            llm, chunks, concurrency=concurrency, progress=on_progress
        )
        status.update(label=f"Read {len(chunks)} section(s)", state="complete")

    cache[notes_key] = notes
    return notes


def summary_markdown(result: sz.SummaryResult) -> str:
    header = [
        f"# Summary — {doc_title}",
        "",
        f"- Source: `{uploaded.name}`",
        f"- Pages: {pages_label}",
        f"- Style: {options.style} · {options.length}",
        f"- Strategy: {result.strategy} ({result.chunk_count} chunk(s))",
        f"- Generated: {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}",
        "",
        "---",
        "",
    ]
    return "\n".join(header) + result.text.strip() + "\n"


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
tab_summary, tab_facts, tab_chat = st.tabs(["📝 Summary", "📊 Key data", "💬 Chat"])

# --- Summary ---------------------------------------------------------------
with tab_summary:
    st.caption(f"{api_calls} model call(s) expected · map-reduce keeps long PDFs faithful")

    run_summary = st.button(
        "Summarise", type="primary", key="run_summary",
        disabled=not api_key or not chunks,
    )

    summaries = _state_store("summaries")
    stored = summaries.get(summary_key)

    if run_summary:
        try:
            llm = make_llm()
            if strategy == "single-pass":
                chain = sz.build_single_pass_chain(llm, options)
                text = st.write_stream(
                    chain.stream(
                        {
                            "text": "\n\n".join(page.text for page in selected_pages),
                            "doc_title": doc_title,
                            "page_label": pages_label,
                        }
                    )
                )
                stored = sz.SummaryResult(
                    text=text.strip(), strategy="single-pass", chunk_count=len(chunks)
                )
            else:
                notes = ensure_notes(llm)
                chain = sz.build_reduce_chain(llm, options)
                text = st.write_stream(
                    chain.stream(
                        {
                            "notes": sz.format_notes(notes),
                            "doc_title": doc_title,
                            "page_label": pages_label,
                        }
                    )
                )
                stored = sz.SummaryResult(
                    text=text.strip(), strategy="map-reduce",
                    chunk_count=len(chunks), notes=tuple(notes),
                )
            summaries[summary_key] = stored
        except sz.ChunkFailure as exc:
            st.error(_friendly_error(exc))
        except Exception as exc:  # noqa: BLE001 - shown to the user
            st.error(_friendly_error(exc))

    if stored and not run_summary:
        st.markdown(stored.text)

    if stored:
        download_cols = st.columns([1, 1, 3])
        download_cols[0].download_button(
            "⬇️ Summary (.md)",
            data=summary_markdown(stored),
            file_name=f"{uploaded.name.rsplit('.', 1)[0]}-summary.md",
            mime="text/markdown",
        )
        if stored.notes:
            download_cols[1].download_button(
                "⬇️ Notes (.json)",
                data=json.dumps(list(stored.notes), indent=2, ensure_ascii=False),
                file_name=f"{uploaded.name.rsplit('.', 1)[0]}-notes.json",
                mime="application/json",
            )


# --- Key data --------------------------------------------------------------
with tab_facts:
    st.caption(
        "A typed pass over the document for titles, figures, entities and "
        "obligations. Readings are grounded in the text; blanks mean the "
        "document did not state the value."
    )

    run_facts = st.button(
        "Extract key data", key="run_facts", disabled=not api_key or not chunks
    )

    facts_cache = _state_store("facts")

    if run_facts:
        try:
            llm = make_llm()
            if strategy == "single-pass":
                source_text = "\n\n".join(page.text for page in selected_pages)
            else:
                source_text = sz.format_notes(ensure_notes(llm))
            with st.spinner("Extracting structured data…"):
                extracted = facts_mod.extract_facts(
                    llm, source_text, doc_title=doc_title, page_label=pages_label
                )
            facts_cache[facts_key] = extracted
        except Exception as exc:  # noqa: BLE001
            st.error(_friendly_error(exc))

    extracted = facts_cache.get(facts_key)
    if extracted:
        metadata_df = facts_mod.metadata_to_frame(extracted)
        facts_df = facts_mod.facts_to_frame(extracted)

        # st.table rather than st.dataframe: the dataframe grid is a canvas
        # component with its theme baked in, so it cannot follow the palette
        # toggle. st.table is plain HTML and themes correctly.
        if metadata_df.empty:
            st.warning("No document-level fields were stated in the text.")
        else:
            st.table(metadata_df.style.hide(axis="index"))

        st.subheader(f"Key facts ({len(facts_df)})")
        if facts_df.empty:
            st.warning("No key facts were extracted.")
        else:
            st.table(facts_df.style.hide(axis="index"))
            st.download_button(
                "⬇️ Key facts (.csv)",
                data=facts_df.to_csv(index=False).encode("utf-8"),
                file_name=f"{uploaded.name.rsplit('.', 1)[0]}-key-facts.csv",
                mime="text/csv",
            )


# --- Chat ------------------------------------------------------------------
with tab_chat:
    st.caption(
        f"Answers cite pages. The document is embedded once ({embedding_model_name}) "
        f"and the top {top_k} excerpts are used per question."
    )

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input(
        "Ask something about this document…", disabled=not api_key or not chunks
    )

    if question:
        st.session_state["messages"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                llm = make_llm()

                indexes = _state_store("indexes")
                entry = indexes.get(index_key)
                if entry is None:
                    embeddings = rag.build_embeddings(embedding_model_name, api_key=api_key)
                    with st.spinner(f"Embedding {len(chunks)} chunk(s)…"):
                        index = rag.VectorIndex.build(chunks, embeddings)
                    indexes[index_key] = (index, embeddings)
                else:
                    index, embeddings = entry

                hits = index.retrieve(embeddings, question, k=top_k)
                history = []
                for prior in st.session_state["messages"][:-1]:
                    cls = HumanMessage if prior["role"] == "user" else AIMessage
                    history.append(cls(content=prior["content"]))

                answer = st.write_stream(
                    rag.build_qa_chain(llm).stream(
                        {
                            "context": rag.format_context(hits),
                            "question": question,
                            "history": history,
                        }
                    )
                )
                st.session_state["messages"].append(
                    {"role": "assistant", "content": answer.strip()}
                )

                with st.expander("Excerpts used"):
                    for hit in hits:
                        st.markdown(
                            f"**[{hit.chunk.page_label}]** · similarity {hit.score:.2f}\n\n"
                            f"{hit.chunk.text[:600]}…"
                        )
            except Exception as exc:  # noqa: BLE001
                st.error(_friendly_error(exc))
