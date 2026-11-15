"""Runtime light/dark theming for the Streamlit UI.

Why this is CSS rather than a Streamlit setting: in Streamlit 1.x the theme is
compiled into emotion CSS-in-JS rules from a JS theme object, the stylesheets
expose no ``--background-color``/``--text-color`` custom properties, and there is
no runtime theme API (``st.theme`` does not exist). ``[theme]`` in
``.streamlit/config.toml`` is read once when the server starts, so a toggle can
only re-style the DOM.

Consequences, accepted deliberately:

* selectors track Streamlit's internal ``data-testid`` hooks, so a major
  Streamlit upgrade can require small fixes here;
* the canvas-rendered ``st.dataframe`` cannot be recoloured, which is why the app
  uses ``st.table`` for key data;
* the choice is per-session (no cookie/localStorage round-trip).
"""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

__all__ = [
    "Palette",
    "LIGHT",
    "DARK",
    "REQUIRED_HOOKS",
    "css_for",
    "apply_theme",
    "relative_luminance",
    "contrast_ratio",
]


@dataclass(frozen=True)
class Tint:
    """Background/text pair for a coloured callout (alert).

    Streamlit paints alerts with its own ``greenBackgroundColor``-style tokens,
    which come from the *base* theme, not from this stylesheet. With
    ``base = "light"`` that means pale plates in dark mode unless we restate
    both halves of the pair here.
    """

    bg: str
    fg: str


@dataclass(frozen=True)
class Palette:
    """The colours a single theme mode uses."""

    surface: str  # page background
    panel: str  # sidebar, cards, chat bubbles
    text: str  # primary body text
    muted: str  # captions, labels, secondary text
    border: str  # hairlines, table borders, outlines
    primary: str  # accents, links, primary buttons
    on_primary: str  # text drawn on top of `primary`
    code_bg: str  # inline code and code blocks
    input_bg: str  # text fields, select boxes
    success: Tint
    info: Tint
    warning: Tint
    error: Tint

    def as_pairs(self) -> dict[str, str]:
        pairs = {
            "surface": self.surface,
            "panel": self.panel,
            "text": self.text,
            "muted": self.muted,
            "border": self.border,
            "primary": self.primary,
            "on_primary": self.on_primary,
            "code_bg": self.code_bg,
            "input_bg": self.input_bg,
        }
        for kind in ("success", "info", "warning", "error"):
            tint = getattr(self, kind)
            pairs[f"{kind}_bg"] = tint.bg
            pairs[f"{kind}_fg"] = tint.fg
        return pairs


LIGHT = Palette(
    surface="#FFFFFF",
    panel="#F0F2F6",
    text="#1A1C20",
    muted="#5F6368",
    border="#D6DAE0",
    primary="#1A73E8",
    on_primary="#FFFFFF",
    code_bg="#F4F5F7",
    input_bg="#FFFFFF",
    success=Tint(bg="#E7F4EA", fg="#14653A"),
    info=Tint(bg="#E8F0FE", fg="#0B4A9E"),
    warning=Tint(bg="#FDF3D8", fg="#6B4A00"),
    error=Tint(bg="#FCE8E6", fg="#8C1D18"),
)

DARK = Palette(
    surface="#0E1117",
    panel="#1A1D24",
    text="#F5F7FA",
    muted="#A3ABB8",
    border="#2C323C",
    primary="#5B9DFF",
    on_primary="#0E1117",
    code_bg="#161B22",
    input_bg="#12161D",
    success=Tint(bg="#14291C", fg="#86E0A3"),
    info=Tint(bg="#131F33", fg="#A9C8FF"),
    warning=Tint(bg="#2B2310", fg="#F3D27F"),
    error=Tint(bg="#2E1715", fg="#FFB4A9"),
)

# Hooks the stylesheet relies on. Exposed so tests can assert coverage instead
# of discovering a silent regression visually.
REQUIRED_HOOKS: tuple[str, ...] = (
    '[data-testid="stAppViewContainer"]',
    '[data-testid="stMain"]',
    '[data-testid="stHeader"]',
    '[data-testid="stToolbar"]',
    '[data-testid="stSidebar"]',
    '[data-testid="stSidebarContent"]',
    '[data-testid="stMarkdownContainer"]',
    '[data-testid="stCaptionContainer"]',
    '[data-testid="stWidgetLabel"]',
    '[data-testid="stMetricValue"]',
    '[data-testid="stMetricLabel"]',
    '[data-testid="stTabs"]',
    '[data-testid="stExpander"]',
    '[data-testid="stAlert"]',
    '[data-testid="stAlertContainer"]',
    '[data-testid="stAlertContentSuccess"]',
    '[data-testid="stAlertContentInfo"]',
    '[data-testid="stAlertContentWarning"]',
    '[data-testid="stAlertContentError"]',
    '[data-testid="stTable"]',
    '[data-testid="stChatMessage"]',
    '[data-testid="stChatInput"]',
    '[data-testid="stDownloadButton"]',
    '[data-testid="stBaseButton-primary"]',
    '[data-testid="stBaseButton-secondary"]',
    '[data-baseweb="input"]',
    '[data-baseweb="textarea"]',
    '[data-baseweb="select"]',
    '[data-baseweb="popover"]',
)


def css_for(palette: Palette, dark: bool) -> str:
    """Build the ``<style>`` block for one mode.

    Rules use ``!important`` because emotion's generated rules are injected with
    high specificity; without it the base theme wins.
    """
    p = palette
    scheme = "dark" if dark else "light"

    # Alerts need their own generator: Streamlit sets both the plate colour and
    # the text colour on stAlertContainer from the *base* theme tokens. Overriding
    # only the text (the obvious fix) leaves a pale light-theme plate behind
    # light text in dark mode, which is exactly how "API key loaded" disappeared.
    alert_rules: list[str] = []
    for kind in ("Success", "Info", "Warning", "Error"):
        tint = getattr(p, kind.lower())
        box = f'[data-testid="stAlert"] [data-testid="stAlertContent{kind}"]'
        alert_rules.append(
            f"""{box} {{
    background-color: {tint.bg} !important;
    color: {tint.fg} !important;
    border-radius: 10px !important;
    padding: 0.85rem 1rem !important;
}}
{box} *,
{box} p,
{box} span,
{box} li,
{box} strong,
{box} em,
{box} a,
{box} code,
{box} pre,
{box} svg {{
    color: {tint.fg} !important;
}}
"""
        )
    alerts_css = "\n".join(alert_rules)

    return f"""<style>
:root {{ color-scheme: {scheme}; }}

/* ---------- page surfaces ---------- */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {{
    background-color: {p.surface} !important;
    color: {p.text} !important;
}}
[data-testid="stHeader"],
[data-testid="stToolbar"] {{
    background-color: {p.surface} !important;
}}
[data-testid="stSidebar"],
[data-testid="stSidebarContent"] {{
    background-color: {p.panel} !important;
    border-right: 1px solid {p.border} !important;
}}
[data-testid="stBottom"] > div,
[data-testid="stBottomBlockContainer"] {{
    background-color: {p.surface} !important;
}}

/* ---------- text ---------- */
[data-testid="stAppViewContainer"] h1,
[data-testid="stAppViewContainer"] h2,
[data-testid="stAppViewContainer"] h3,
[data-testid="stAppViewContainer"] h4,
[data-testid="stAppViewContainer"] h5,
[data-testid="stAppViewContainer"] h6,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {{
    color: {p.text} !important;
}}
[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span,
[data-testid="stMarkdownContainer"] strong,
[data-testid="stMarkdownContainer"] em,
[data-testid="stMarkdownContainer"] blockquote {{
    color: {p.text} !important;
}}
[data-testid="stMarkdownContainer"] a {{
    color: {p.primary} !important;
}}
[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] span {{
    color: {p.text} !important;
}}
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p,
[data-testid="stCaptionContainer"] span,
small {{
    color: {p.muted} !important;
}}
hr {{
    border-color: {p.border} !important;
}}

/* ---------- metrics ---------- */
[data-testid="stMetric"] {{
    background-color: {p.panel} !important;
    border: 1px solid {p.border} !important;
    border-radius: 10px;
    padding: 0.6rem 0.8rem;
}}
[data-testid="stMetricValue"],
[data-testid="stMetricValue"] * {{
    color: {p.text} !important;
}}
[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] * {{
    color: {p.muted} !important;
}}

/* ---------- tabs ---------- */
[data-testid="stTabs"] [data-baseweb="tab-list"] {{
    border-bottom: 1px solid {p.border} !important;
}}
[data-testid="stTabs"] button[role="tab"] {{
    color: {p.muted} !important;
}}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {{
    color: {p.text} !important;
}}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] {{
    background-color: {p.primary} !important;
}}
[data-testid="stTabs"] [data-baseweb="tab-border"] {{
    background-color: {p.border} !important;
}}

/* ---------- containers ---------- */
[data-testid="stExpander"] {{
    background-color: {p.panel} !important;
    border: 1px solid {p.border} !important;
    border-radius: 10px !important;
}}
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary * {{
    color: {p.text} !important;
}}
[data-testid="stExpanderDetails"] {{
    background-color: {p.panel} !important;
}}
[data-testid="stAlert"] {{
    background-color: transparent !important;
    border: 0 !important;
}}
/* Streamlit pads the container and paints it; move both onto the content box so
   our per-kind tint fills the visible rounded plate instead of sitting inside
   the leftover padding of a transparent container. */
[data-testid="stAlert"] [data-testid="stAlertContainer"] {{
    background-color: transparent !important;
    padding: 0 !important;
}}
[data-testid="stAlert"] code,
[data-testid="stAlert"] pre {{
    background-color: transparent !important;
    border-color: transparent !important;
}}
{alerts_css}

/* ---------- buttons ---------- */
[data-testid="stBaseButton-secondary"],
[data-testid="stBaseButton-secondaryFormSubmit"],
[data-testid="stDownloadButton"] button {{
    background-color: {p.panel} !important;
    color: {p.text} !important;
    border: 1px solid {p.border} !important;
}}
[data-testid="stBaseButton-secondary"] *,
[data-testid="stDownloadButton"] button * {{
    color: {p.text} !important;
}}
[data-testid="stBaseButton-secondary"]:hover,
[data-testid="stDownloadButton"] button:hover {{
    border-color: {p.primary} !important;
    color: {p.primary} !important;
}}
[data-testid="stBaseButton-primary"],
[data-testid="stBaseButton-primaryFormSubmit"] {{
    background-color: {p.primary} !important;
    color: {p.on_primary} !important;
    border: 1px solid {p.primary} !important;
}}
[data-testid="stBaseButton-primary"] *,
[data-testid="stBaseButton-primaryFormSubmit"] * {{
    color: {p.on_primary} !important;
}}

/* ---------- inputs ---------- */
[data-baseweb="input"],
[data-baseweb="textarea"],
[data-baseweb="select"] > div {{
    background-color: {p.input_bg} !important;
    border-color: {p.border} !important;
}}
[data-baseweb="input"] input,
[data-baseweb="textarea"] textarea,
[data-baseweb="select"] input,
[data-baseweb="select"] [data-testid="stSelectbox"] {{
    background-color: transparent !important;
    color: {p.text} !important;
    caret-color: {p.primary} !important;
}}
[data-baseweb="input"] svg,
[data-baseweb="select"] svg,
[data-baseweb="textarea"] svg {{
    fill: {p.muted} !important;
}}
[data-baseweb="popover"],
[data-baseweb="popover"] [role="listbox"],
[data-baseweb="menu"] {{
    background-color: {p.panel} !important;
    border: 1px solid {p.border} !important;
}}
[role="option"] {{
    color: {p.text} !important;
}}
[role="option"]:hover {{
    background-color: {p.border} !important;
}}
::placeholder {{
    color: {p.muted} !important;
}}
[data-testid="stSlider"] [role="slider"] {{
    background-color: {p.primary} !important;
}}
[data-testid="stCheckbox"] label span {{
    color: {p.text} !important;
}}

/* ---------- code ---------- */
[data-testid="stCode"],
pre,
code {{
    background-color: {p.code_bg} !important;
    color: {p.text} !important;
    border-color: {p.border} !important;
}}
[data-testid="stCode"] code,
pre code {{
    background-color: transparent !important;
    border: 0 !important;
}}

/* ---------- tables (st.table renders real HTML) ---------- */
[data-testid="stTable"] table {{
    color: {p.text} !important;
    border-color: {p.border} !important;
}}
[data-testid="stTable"] thead th {{
    background-color: {p.panel} !important;
    color: {p.text} !important;
    border-color: {p.border} !important;
}}
[data-testid="stTable"] tbody td {{
    color: {p.text} !important;
    border-color: {p.border} !important;
}}
[data-testid="stTable"] tbody tr:nth-child(even) {{
    background-color: {p.panel} !important;
}}

/* ---------- chat ---------- */
[data-testid="stChatMessage"] {{
    background-color: {p.panel} !important;
    border: 1px solid {p.border} !important;
    border-radius: 12px !important;
}}
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] span,
[data-testid="stChatMessage"] div {{
    color: {p.text} !important;
}}
[data-testid="stChatInput"] {{
    background-color: {p.input_bg} !important;
    border: 1px solid {p.border} !important;
    border-radius: 12px !important;
}}
[data-testid="stChatInput"] textarea {{
    color: {p.text} !important;
    background-color: transparent !important;
}}
[data-testid="stChatInput"] svg {{
    fill: {p.muted} !important;
}}

/* ---------- progress + spinner ---------- */
[data-testid="stProgress"] [role="progressbar"] {{
    background-color: {p.border} !important;
}}
[data-testid="stProgress"] [role="progressbar"] > div > div {{
    background-color: {p.primary} !important;
}}
[data-testid="stSpinner"] * {{
    color: {p.muted} !important;
}}

/* ---------- scrollbars ---------- */
::-webkit-scrollbar {{
    width: 10px;
    height: 10px;
}}
::-webkit-scrollbar-track {{
    background: {p.surface};
}}
::-webkit-scrollbar-thumb {{
    background: {p.border};
    border-radius: 8px;
}}
</style>"""


def apply_theme(dark: bool) -> None:
    """Inject the stylesheet for the requested mode into the current run."""
    st.markdown(css_for(DARK if dark else LIGHT, dark), unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Contrast helpers (used by tests to keep the palettes accessible)
# --------------------------------------------------------------------------- #
def _channel(value: int) -> float:
    srgb = value / 255
    return srgb / 12.92 if srgb <= 0.04045 else ((srgb + 0.055) / 1.055) ** 2.4


def relative_luminance(hex_color: str) -> float:
    """WCAG 2.1 relative luminance of a ``#RRGGBB`` colour."""
    value = hex_color.lstrip("#")
    if len(value) != 6:
        raise ValueError(f"expected #RRGGBB, got {hex_color!r}")
    r, g, b = (int(value[i : i + 2], 16) for i in (0, 2, 4))
    return 0.2126 * _channel(r) + 0.7152 * _channel(g) + 0.0722 * _channel(b)


def contrast_ratio(foreground: str, background: str) -> float:
    """WCAG 2.1 contrast ratio between two colours."""
    lighter, darker = sorted(
        (relative_luminance(foreground), relative_luminance(background)), reverse=True
    )
    return (lighter + 0.05) / (darker + 0.05)
