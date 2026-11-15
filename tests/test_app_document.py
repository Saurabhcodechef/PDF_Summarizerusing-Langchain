"""UI tests for the post-upload view.

The uploader is replaced with a fake so the whole document view (metrics, tabs,
page range) is exercised without touching the network or requiring a real file
dialog.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from conftest import build_pdf

APP = str(Path(__file__).resolve().parents[1] / "app.py")


class FakeUpload:
    name = "sample.pdf"
    type = "application/pdf"

    def __init__(self, data: bytes) -> None:
        self._data = data
        self.size = len(data)

    def getvalue(self) -> bytes:
        return self._data


@pytest.fixture(scope="module")
def document_app() -> AppTest:
    pages = [
        "Quarterly Report\nAcme Corp shipped 1200 units in Q3.\nRevenue was 4.2 million dollars.",
        "Method\nWe measured throughput across 14 sites.",
        "Conclusion\nThe rollout should continue.",
    ]
    pdf = build_pdf(pages)

    original = st.file_uploader
    st.file_uploader = lambda *args, **kwargs: FakeUpload(pdf)  # type: ignore[assignment]
    try:
        return AppTest.from_file(APP, default_timeout=90).run()
    finally:
        st.file_uploader = original  # type: ignore[assignment]


def test_no_exceptions(document_app: AppTest):
    assert not document_app.exception


def test_metrics_reflect_the_document(document_app: AppTest):
    metrics = {metric.label: metric.value for metric in document_app.metric}

    assert metrics["Pages used"] == "3 / 3"
    assert metrics["Chunks"] == "1"
    assert metrics["Strategy"] == "Single Pass"
    assert int(metrics["Words"].replace(",", "")) > 20


def test_three_tabs_are_created(document_app: AppTest):
    labels = [tab.label for tab in document_app.tabs]
    assert labels == ["📝 Summary", "📊 Key data", "💬 Chat"]


def test_actions_are_offered(document_app: AppTest):
    labels = [button.label for button in document_app.button]
    assert "Summarise" in labels
    assert "Extract key data" in labels


def test_page_range_controls_exist(document_app: AppTest):
    labels = [widget.label for widget in document_app.sidebar.number_input]
    assert labels == ["From", "To"]

    from_values = {w.label: w.value for w in document_app.sidebar.number_input}
    assert from_values["From"] == 1
    assert from_values["To"] == 3


def test_chat_input_is_available(document_app: AppTest):
    assert document_app.chat_input, "chat tab should offer a question box"


def test_dark_mode_toggle_survives_upload(document_app: AppTest):
    assert [toggle.label for toggle in document_app.sidebar.toggle] == ["🌗 Dark mode"]


def test_key_data_uses_themable_tables():
    """st.dataframe is a canvas grid that ignores the palette toggle."""
    source = Path(APP).read_text(encoding="utf-8")
    assert "st.dataframe(" not in source
    assert "st.table(" in source


def test_sidebar_reports_key_status(document_app: AppTest):
    # The real .env is loaded during tests; either state is acceptable, but one
    # of the two must be shown so the user knows what is happening.
    messages = [element.value for element in document_app.sidebar.success]
    messages += [element.value for element in document_app.sidebar.error]
    assert any("key" in message.lower() for message in messages)
