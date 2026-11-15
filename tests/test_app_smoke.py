"""End-to-end smoke tests for the Streamlit UI using Streamlit's AppTest."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import theme

APP = str(Path(__file__).resolve().parents[1] / "app.py")


@pytest.fixture(scope="module")
def app() -> AppTest:
    return AppTest.from_file(APP, default_timeout=60).run()


def test_app_starts_without_exceptions(app: AppTest):
    assert not app.exception


def test_sidebar_settings_are_present(app: AppTest):
    labels = [widget.label for widget in app.sidebar.selectbox]
    assert "Summary style" in labels

    slider_labels = [widget.label for widget in app.sidebar.slider]
    assert "Temperature" in slider_labels

    assert any("Length" in radio.label for radio in app.sidebar.radio)


def test_theme_toggle_defaults_to_light(app: AppTest):
    toggles = app.sidebar.toggle
    assert len(toggles) == 1
    assert "Dark mode" in toggles[0].label
    assert toggles[0].value is False
    assert app.session_state["dark_mode"] is False


def test_light_palette_is_injected_by_default(app: AppTest):
    styles = [element.value for element in app.markdown if "<style>" in element.value]
    assert styles, "theme stylesheet was not injected"
    assert "#FFFFFF" in styles[0]
    assert theme.DARK.surface not in styles[0]


def test_toggling_dark_mode_swaps_the_palette(app: AppTest):
    app.sidebar.toggle[0].set_value(True).run()

    assert not app.exception
    assert app.session_state["dark_mode"] is True

    styles = [element.value for element in app.markdown if "<style>" in element.value]
    assert styles
    assert theme.DARK.surface in styles[0]
    assert theme.DARK.text in styles[0]
    assert theme.LIGHT.surface not in styles[0]
    # The sidebar "API key loaded" success plate must flip too, or its text
    # disappears against the light base-theme green.
    assert theme.DARK.success.bg in styles[0]

    # Put the shared fixture back to light for any test that runs after this.
    app.sidebar.toggle[0].set_value(False).run()


def test_summary_style_choices_match_summarizer(app: AppTest):
    import summarizer as sz

    style_box = next(w for w in app.sidebar.selectbox if w.label == "Summary style")
    assert list(style_box.options) == list(sz.STYLES)


def test_prompts_for_upload_before_showing_tabs(app: AppTest):
    assert app.title[0].value == "📄 PDF Summarizer"
    assert "Upload a PDF" in app.info[0].value
    assert not app.tabs, "tabs should only appear once a file is loaded"


def test_uploader_is_pdf_only(app: AppTest):
    uploaders = app.get("file_uploader")
    assert uploaders
