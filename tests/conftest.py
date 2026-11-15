"""Shared test fixtures.

The sample PDF is generated from raw bytes so the test suite has no binary
fixtures and no extra dependency (no reportlab) to keep in sync.
"""

from __future__ import annotations

import pytest


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def build_pdf(pages: list[str]) -> bytes:
    """Build a minimal, valid PDF with Helvetica text on each page."""
    if not pages:
        raise ValueError("need at least one page")

    page_numbers = [4 + 2 * i for i in range(len(pages))]
    objects: dict[int, bytes] = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: (
            f"<< /Type /Pages /Kids [{' '.join(f'{n} 0 R' for n in page_numbers)}] "
            f"/Count {len(pages)} >>"
        ).encode(),
        3: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    }

    for position, text in enumerate(pages):
        page_number = page_numbers[position]
        content_number = page_number + 1

        lines = text.split("\n") if text else []
        if lines:
            body = ["BT", "/F1 11 Tf", "72 720 Td", "14 TL"]
            for line in lines:
                body.append(f"({_escape(line)}) Tj")
                body.append("T*")
            body.append("ET")
            stream = "\n".join(body).encode("latin-1")
        else:
            stream = b""

        objects[content_number] = (
            f"<< /Length {len(stream)} >>\nstream\n".encode()
            + stream
            + b"\nendstream"
        )
        objects[page_number] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_number} 0 R >>"
        ).encode()

    out = bytearray(b"%PDF-1.4\n")
    offsets: dict[int, int] = {}
    for number in sorted(objects):
        offsets[number] = len(out)
        out += f"{number} 0 obj\n".encode() + objects[number] + b"\nendobj\n"

    xref_offset = len(out)
    size = max(objects) + 1
    out += f"xref\n0 {size}\n".encode()
    out += b"0000000000 65535 f \n"
    for number in range(1, size):
        out += f"{offsets[number]:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {size} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n"
    ).encode()
    return bytes(out)


SAMPLE_PAGES = [
    "Quarterly Report\nAcme Corp shipped 1200 units in Q3.\nRevenue was 4.2 million dollars.",
    "Method\nWe measured throughput across 14 sites.\nThe control group showed no change.",
    "Conclusion\nThe rollout should continue.\nRisks include supply delay and staffing.",
]


@pytest.fixture
def sample_pdf() -> bytes:
    return build_pdf(SAMPLE_PAGES)


@pytest.fixture
def empty_pdf() -> bytes:
    return build_pdf([""])
