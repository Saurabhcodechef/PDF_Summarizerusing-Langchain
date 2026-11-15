"""Tests for the runtime theme palettes and stylesheet."""

from __future__ import annotations

import pytest

import theme
from theme import DARK, LIGHT, Palette, contrast_ratio, css_for, relative_luminance

PALETTES = [pytest.param(LIGHT, id="light"), pytest.param(DARK, id="dark")]


# --- contrast ---------------------------------------------------------------
@pytest.mark.parametrize("palette", PALETTES)
def test_body_text_is_wcag_aa(palette: Palette):
    assert contrast_ratio(palette.text, palette.surface) >= 4.5
    assert contrast_ratio(palette.text, palette.panel) >= 4.5


@pytest.mark.parametrize("palette", PALETTES)
def test_muted_text_is_legible(palette: Palette):
    assert contrast_ratio(palette.muted, palette.surface) >= 3.0
    assert contrast_ratio(palette.muted, palette.panel) >= 3.0


@pytest.mark.parametrize("palette", PALETTES)
def test_links_are_readable_on_the_page(palette: Palette):
    assert contrast_ratio(palette.primary, palette.surface) >= 4.5


@pytest.mark.parametrize("palette", PALETTES)
def test_primary_button_text_has_contrast(palette: Palette):
    assert contrast_ratio(palette.on_primary, palette.primary) >= 4.5


@pytest.mark.parametrize("palette", PALETTES)
def test_code_blocks_are_readable(palette: Palette):
    assert contrast_ratio(palette.text, palette.code_bg) >= 4.5


@pytest.mark.parametrize("palette", PALETTES)
@pytest.mark.parametrize("kind", ["success", "info", "warning", "error"])
def test_alert_tints_are_wcag_aa(palette: Palette, kind: str):
    """Alerts are the one place where we must restate both halves of the pair.

    Streamlit derives the plate from the base theme, so a text-only override
    produced light text on a pale plate in dark mode ("API key loaded" vanished).
    """
    tint = getattr(palette, kind)
    assert contrast_ratio(tint.fg, tint.bg) >= 4.5


def test_luminance_endpoints():
    assert relative_luminance("#000000") == pytest.approx(0.0)
    assert relative_luminance("#FFFFFF") == pytest.approx(1.0)


def test_contrast_black_on_white_is_max():
    assert contrast_ratio("#000000", "#FFFFFF") == pytest.approx(21.0)


def test_contrast_is_symmetric():
    assert contrast_ratio("#1A1C20", "#FFFFFF") == pytest.approx(
        contrast_ratio("#FFFFFF", "#1A1C20")
    )


def test_bad_colour_is_rejected():
    with pytest.raises(ValueError):
        relative_luminance("#FFF")


# --- stylesheet -------------------------------------------------------------
def test_css_covers_required_hooks():
    css = css_for(DARK, dark=True)
    missing = [hook for hook in theme.REQUIRED_HOOKS if hook not in css]
    assert not missing, f"stylesheet no longer styles: {missing}"


def test_css_uses_important_everywhere():
    """Emotion rules are highly specific; without !important the base theme wins."""
    css = css_for(LIGHT, dark=False)
    # Every declaration block that sets a colour must be marked important.
    for line in css.splitlines():
        stripped = line.strip()
        if stripped.startswith(("color:", "background-color:", "border-color:", "fill:")):
            assert "!important" in stripped, stripped


def test_modes_produce_different_stylesheets():
    assert css_for(LIGHT, dark=False) != css_for(DARK, dark=True)


def test_stylesheet_declares_colour_scheme():
    assert "color-scheme: dark" in css_for(DARK, dark=True)
    assert "color-scheme: light" in css_for(LIGHT, dark=False)


def test_palette_colours_are_actually_used():
    css = css_for(DARK, dark=True)
    for colour in DARK.as_pairs().values():
        assert colour in css


def test_does_not_rely_on_css_variables_from_old_streamlit():
    """The --background-color/:root trick is dead in current Streamlit builds."""
    css = css_for(LIGHT, dark=False)
    assert "--background-color" not in css
    assert "--text-color" not in css


def test_no_dataframe_targeting():
    """st.dataframe is a canvas grid; styling it via CSS would be misleading."""
    assert "stDataFrame" not in css_for(DARK, dark=True)


@pytest.mark.parametrize("palette", PALETTES)
def test_alert_plates_carry_their_own_tint(palette: Palette):
    """Each alert kind must set its background, not just its text colour."""
    css = css_for(palette, dark=palette is DARK)
    for kind in ("Success", "Info", "Warning", "Error"):
        box = f'[data-testid="stAlert"] [data-testid="stAlertContent{kind}"]'
        assert box in css, f"{kind} alert is unstyled"
        tint = getattr(palette, kind.lower())
        # The bg/fg pair must be stated next to that selector.
        block_start = css.index(box)
        block = css[block_start : block_start + 400]
        assert f"background-color: {tint.bg} !important" in block
        assert f"color: {tint.fg} !important" in block


@pytest.mark.parametrize("palette", PALETTES)
def test_alert_text_is_not_forced_to_the_page_text_colour(palette: Palette):
    """The bug: `[data-testid="stAlert"] * { color: <page text> }` overrode the
    tinted plate, leaving white text on Streamlit's pale base-theme green."""
    css = css_for(palette, dark=palette is DARK)
    assert '[data-testid="stAlert"] *,' not in css
    assert f'[data-testid="stAlert"] p {{\n    color: {palette.text}' not in css


def test_css_is_a_single_style_block():
    css = css_for(LIGHT, dark=False)
    assert css.startswith("<style>")
    assert css.rstrip().endswith("</style>")
    assert css.count("<style>") == 1


@pytest.mark.parametrize("palette", PALETTES)
def test_css_braces_are_balanced(palette: Palette):
    """A stray brace silently kills every later rule."""
    css = css_for(palette, dark=palette is DARK)
    assert css.count("{") == css.count("}")
    assert "{{" not in css
    assert "}}" not in css


@pytest.mark.parametrize("palette", PALETTES)
def test_no_empty_declarations(palette: Palette):
    css = css_for(palette, dark=palette is DARK)
    assert ": ;" not in css
    assert ":  ;" not in css


def test_apply_theme_is_thin_wrapper(monkeypatch):
    """apply_theme should just push the right stylesheet into Streamlit."""
    calls: list[str] = []
    monkeypatch.setattr(theme.st, "markdown", lambda body, **kwargs: calls.append(body))

    theme.apply_theme(True)
    theme.apply_theme(False)

    assert calls[0] == css_for(DARK, dark=True)
    assert calls[1] == css_for(LIGHT, dark=False)
