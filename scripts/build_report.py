"""Render the one-page poster companion report to PDF (US Letter).

    python3 scripts/build_report.py
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "poster" / "report.html"
OUT = ROOT / "docs" / "poster" / "XAI_Disagreement_Auditor_report.pdf"


def main() -> None:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        b = p.chromium.launch()
        page = b.new_page(viewport={"width": 816, "height": 1056})
        page.goto(SRC.as_uri(), wait_until="networkidle")
        page.pdf(path=str(OUT), width="8.5in", height="11in", print_background=True,
                 margin={"top": "0", "bottom": "0", "left": "0", "right": "0"})
        b.close()
    print("PDF:", OUT)


if __name__ == "__main__":
    main()
