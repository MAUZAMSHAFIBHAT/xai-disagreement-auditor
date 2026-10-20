"""Render the poster to a print-ready PDF (24in x 36in) and a self-contained HTML.

    python3 scripts/build_poster.py                      # no QR code
    python3 scripts/build_poster.py https://github.com/you/xai-disagreement-auditor

Figures come from docs/poster/figures (regenerate with scripts/make_poster_figures.py).
"""

from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POSTER = ROOT / "docs" / "poster"
FIGS = POSTER / "figures"


def data_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def qr_data_uri(url: str) -> str:
    import qrcode

    img = qrcode.make(url, box_size=12, border=2)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


PAGES = {"portrait": (36, 48), "landscape": (48, 36)}


def build(repo_url: str | None = None, orientation: str = "portrait") -> tuple[Path, Path]:
    w_in, h_in = PAGES[orientation]
    w_px, h_px = w_in * 96, h_in * 96
    html = (POSTER / "poster.html").read_text()
    html = html.replace("@page { size: 36in 48in; margin: 0; }", f"@page {{ size: {w_in}in {h_in}in; margin: 0; }}")
    html = html.replace("html,body{width:3456px; height:4608px;}", f"html,body{{width:{w_px}px; height:{h_px}px;}}")
    if orientation == "landscape":
        # A wide page reads better as balanced columns than as fixed rows, so the cards
        # flow through a four-column layout instead of the portrait row grid.
        html = html.replace("</style>", """
  main{display:block !important; column-count:4; column-gap:36px; padding:40px 80px 0 80px;}
  .row{display:block !important;}
  .card{break-inside:avoid; -webkit-column-break-inside:avoid; page-break-inside:avoid;
        margin:0 0 36px 0; display:block;}
  header{padding:62px 80px 56px 80px;}
  h1{font-size:128px;}
  .sub{font-size:44px; max-width:2600px;}
  .headline-stat .num{font-size:150px;}
  footer{padding:40px 80px 46px 80px; grid-template-columns:1.45fr 1fr 340px;}
</style>""")
    for key, name in [("FIG_FINGERPRINT", "fingerprint.png"), ("FIG_SCATTER", "trust_vs_truth.png"),
                      ("FIG_RHO", "trust_vs_rho.png"), ("FIG_PRECISION", "method_precision.png"),
                      ("FIG_MODELS", "model_trust.png")]:
        html = html.replace("{{" + key + "}}", data_uri(FIGS / name))

    if repo_url:
        pretty = repo_url.replace("https://", "").replace("http://", "").rstrip("/")
        repo_line = f'<b>Code, data and docs:</b> <code>{pretty}</code>'
        qr_block = (f'<div class="qr"><img src="{qr_data_uri(repo_url)}" alt="QR code to the repository">'
                    f'<div class="cap">Scan for code, docs<br>and the full methodology</div></div>')
    else:
        repo_line = '<b>Code, data and docs:</b> <code>github.com/&lt;username&gt;/xai-disagreement-auditor</code>'
        qr_block = ('<div class="qr"><div style="width:300px;height:300px;border:4px dashed #d9d8d1;'
                    'border-radius:18px;display:flex;align-items:center;justify-content:center;'
                    'color:#7c7a74;font-size:26px;text-align:center;padding:20px">QR code<br>(add repo URL)</div>'
                    '<div class="cap">Scan for code, docs<br>and the full methodology</div></div>')
    html = html.replace("{{REPO_LINE}}", repo_line).replace("{{QR_BLOCK}}", qr_block)

    out_html = POSTER / f"poster_render_{orientation}.html"
    out_html.write_text(html)

    from playwright.sync_api import sync_playwright

    suffix = "" if orientation == "portrait" else "_landscape"
    out_pdf = POSTER / f"XAI_Disagreement_Auditor_poster{suffix}.pdf"
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": w_px, "height": h_px})
        page.goto(out_html.as_uri(), wait_until="networkidle")
        # Auto-fit: lay the poster out on a wider canvas and scale it down so it fills
        # exactly one page edge to edge (iterating, because layout height depends on width).
        scale = page.evaluate("""(dims) => {
            const [PW, PH] = dims;
            const b = document.body;
            const wrap = document.createElement('div');
            while (b.firstChild) wrap.appendChild(b.firstChild);
            b.appendChild(wrap);
            wrap.style.cssText = 'display:flex;flex-direction:column;transform-origin:top left;';
            b.style.cssText += ';display:block;width:' + PW + 'px;height:' + PH + 'px;overflow:hidden;';
            document.documentElement.style.cssText += ';width:' + PW + 'px;height:' + PH + 'px;overflow:hidden;';
            // The layout canvas is always PW/s wide, so after scaling by s it is exactly
            // one page wide. Binary-search the largest s whose laid-out height still fits.
            const fits = (s) => {
                wrap.style.width = (PW / s) + 'px';
                return wrap.scrollHeight * s <= PH;
            };
            let lo = 0.25, hi = 1.0;
            if (!fits(hi)) {
                for (let i = 0; i < 20; i++) {
                    const mid = (lo + hi) / 2;
                    if (fits(mid)) lo = mid; else hi = mid;
                }
            } else {
                lo = hi;
            }
            const s = lo;
            wrap.style.width = (PW / s) + 'px';
            wrap.style.minHeight = (PH / s) + 'px';
            wrap.style.transform = 'scale(' + s + ')';
            return s;
        }""", [w_px, h_px])
        print(f"auto-fit scale: {scale:.3f}")
        page.wait_for_timeout(400)
        page.pdf(path=str(out_pdf), width=f"{w_in}in", height=f"{h_in}in", print_background=True,
                 margin={"top": "0", "bottom": "0", "left": "0", "right": "0"})
        page.screenshot(path=str(POSTER / f"poster_preview_{orientation}.png"), full_page=True)
        browser.close()
    return out_pdf, out_html


if __name__ == "__main__":
    args = [a for a in sys.argv[1:]]
    orientation = "portrait"
    for o in ("portrait", "landscape", "both"):
        if o in args:
            orientation = o
            args.remove(o)
    url = args[0] if args else None
    for o in (["portrait", "landscape"] if orientation == "both" else [orientation]):
        pdf, html = build(url, o)
        print(f"{o:9s} PDF :", pdf, f"({pdf.stat().st_size / 1e6:.1f} MB)")
