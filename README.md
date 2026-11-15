# PDF Summarizer (LangChain + Gemini)

Upload a PDF, get a summary, a table of key data, and a chat interface that
answers questions with page citations.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` in the project root:

```dotenv
GEMINI_API_KEY=your-key-from-aistudio.google.com
GEMINI_MODEL=gemini-3.1-flash-lite
```

Run:

```bash
streamlit run app.py
```

## How it works

**Extraction** — `pypdf` pulls text page by page. Whitespace is normalised and
line-break hyphens are rejoined, but paragraph structure is preserved because
the splitter depends on it. Encrypted PDFs are unlocked with an empty password
first (common for "restricted" exports); scanned PDFs with no text layer are
reported clearly rather than silently producing an empty summary.

**Chunking** — `pdf_utils.chunk_pages` splits paragraphs, then sentences, then
words, so no chunk exceeds the configured size and every chunk keeps the page
numbers it came from. That attribution is what makes `[p.12]` citations work.

**Summarisation** — two strategies, chosen automatically by token count:

| Strategy | When | Calls |
| --- | --- | --- |
| single-pass | ≤ 80k tokens (configurable in `config.py`) | 1 |
| map-reduce | larger documents | chunks + 1 |

Map-reduce takes bullet notes per chunk concurrently, then synthesises one
summary from the notes. Notes are cached per document and model, so changing
style or length re-runs only the final cheap step.

**Key data** — a typed extraction pass (`with_structured_output`) returning a
Pydantic model: title, type, author, date, categorised facts with page refs,
entities and action items. Falls back to JSON parsing if the model rejects the
schema.

**Chat** — chunks are embedded once with `gemini-embedding-001` and kept in a
numpy array; retrieval is a cosine scan, which is instant at PDF scale and
avoids a vector-database dependency. Answers are instructed to cite pages and
to admit when the excerpts don't cover the question.

## Theming

The sidebar has a **🌗 Dark mode** toggle at the top. It starts light on every
new session and switches the whole UI — page, sidebar, text, tables, chat
bubbles — at runtime.

This is done with an injected stylesheet (`theme.py`), not with Streamlit's own
theme settings, because Streamlit 1.x compiles its theme into emotion CSS-in-JS
at build time: there is no runtime theme API, `[theme]` in
`.streamlit/config.toml` is read once at server start, and the generated
stylesheets expose no `--background-color`/`--text-color` custom properties. So
the widely-copied CSS-variable trick does nothing here, and the palette is
applied as `!important` rules against Streamlit's `data-testid` hooks.

Three consequences worth knowing:

- The selectors track Streamlit internals, so a major Streamlit upgrade can
  break the look. `tests/test_theme.py` asserts every required hook is still
  styled, which turns a silent visual regression into a failing test.
- Alerts (`st.success`/`info`/`warning`/`error`) need both halves of their
  colour pair restated. Streamlit paints the alert plate from the *base* theme,
  so overriding only the text left light text on a pale plate in dark mode and
  made the sidebar's "API key loaded" message vanish. `Palette` therefore
  carries a `Tint` (background + text) per kind, each pair contrast-checked.
- Key-data tables use `st.table` rather than `st.dataframe`. The dataframe grid
  is a canvas component with its theme baked in as inline styles, so CSS cannot
  recolour it; `st.table` is plain HTML and follows the palette. The trade-off
  is losing interactive sorting, which doesn't matter at ~20 rows.
- The choice is per-session. Nothing is written to cookies, localStorage, or the
  URL, so a hard refresh returns to light.

`.streamlit/config.toml` deliberately sets `base = "light"`: the dark palette is
applied at runtime, so any widget it misses falls back to a light default
instead of clashing with a dark shell.

## Notes and limits

- Token counts use tiktoken's `cl100k_base` as a proxy for Gemini's tokeniser.
  They are approximate, and only used to pick a strategy. If the encoding can't
  be downloaded, the app falls back to `characters / 4`.
- `gemini-3.1-flash-lite` free tiers throttle at a few requests per minute. The
  map phase defaults to 3 parallel calls; lower it in **Advanced** if you see
  429s. The client retries 429/5xx four times with backoff.
- Text-based PDFs only. Scanned documents need OCR before upload.
- Changing the model mid-conversation invalidates the cached index, since
  different embedding spaces aren't comparable.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

121 tests, no network calls and no API budget burned: the LLM is replaced by
LangChain's `FakeListChatModel`, embeddings by a bag-of-words stub, and PDF
fixtures are generated from raw bytes in `tests/conftest.py`. The Streamlit view
is tested through `AppTest`, including the post-upload screen with a fake
file uploader.

## Layout

```
app.py           Streamlit UI
config.py        env vars + tunables
pdf_utils.py     extraction, cleaning, chunking, token counting
summarizer.py    map-reduce chains, styles, lengths
extract.py       structured key-data extraction
rag.py           embedding index + cited Q&A
theme.py         light/dark palettes + injected stylesheet
tests/           unit tests (no API calls)
```
