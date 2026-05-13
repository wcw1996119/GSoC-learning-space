"""Build paper PDFs from markdown drafts via Edge headless.

Pipeline: markdown → HTML (markdown2 + academic CSS + MathJax) → Edge --print-to-pdf.

Usage:
    python tools/build_paper_pdf.py paper_A_draft_v2.md
    python tools/build_paper_pdf.py paper_A_draft_v2_zh.md

Output:
    Same directory: <basename>.html and <basename>.pdf
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import markdown2

EDGE_PATH = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

CSS = """
@page {
    size: A4;
    margin: 22mm 22mm 22mm 22mm;
    @bottom-center {
        content: counter(page);
        font-family: 'Times New Roman', 'SimSun', serif;
        font-size: 10pt;
        color: #333;
    }
}

body {
    font-family: 'Times New Roman', 'SimSun', '宋体', serif;
    font-size: 11pt;
    line-height: 1.55;
    color: #1a1a1a;
    max-width: 165mm;
    margin: 0 auto;
}

h1 {
    font-size: 20pt;
    line-height: 1.25;
    margin: 0 0 16pt 0;
    text-align: left;
}

h2 {
    font-size: 14pt;
    margin-top: 20pt;
    margin-bottom: 8pt;
    border-bottom: 0.4pt solid #888;
    padding-bottom: 3pt;
}

h3 {
    font-size: 12pt;
    margin-top: 14pt;
    margin-bottom: 5pt;
    font-style: italic;
}

h4 {
    font-size: 11pt;
    margin-top: 10pt;
    margin-bottom: 4pt;
}

p {
    text-align: justify;
    margin: 6pt 0;
    text-indent: 0;
}

ul, ol {
    margin: 5pt 0;
    padding-left: 22pt;
}

li {
    margin: 3pt 0;
    text-align: justify;
}

table {
    border-collapse: collapse;
    margin: 12pt auto;
    font-size: 9.5pt;
    page-break-inside: avoid;
}

th, td {
    border: 0.4pt solid #444;
    padding: 4pt 7pt;
    vertical-align: top;
}

th {
    background-color: #efefef;
    font-weight: 600;
    text-align: left;
}

td {
    text-align: left;
}

code {
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 9.5pt;
    background: #f6f6f6;
    padding: 1pt 3pt;
    border-radius: 2pt;
}

pre {
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 9pt;
    background: #f6f6f6;
    padding: 8pt;
    overflow-x: auto;
    page-break-inside: avoid;
}

blockquote {
    border-left: 2pt solid #aaa;
    padding-left: 10pt;
    margin-left: 0;
    color: #444;
    font-style: italic;
}

hr {
    border: 0;
    border-top: 0.4pt solid #888;
    margin: 14pt 0;
}

em { font-style: italic; }
strong { font-weight: 700; }

/* Section separators (---) become hr */
.abstract {
    background: #f9f9f9;
    padding: 10pt 14pt;
    border-left: 2pt solid #555;
    margin: 14pt 0 16pt 0;
    font-size: 10pt;
}

/* Math rendering */
.MathJax_Display { margin: 6pt 0; }
mjx-container {
    font-size: 100% !important;
}

/* Figures: never exceed page width */
img {
    max-width: 100%;
    height: auto;
    display: block;
    margin: 12pt auto;
}
figure {
    page-break-inside: avoid;
    margin: 12pt 0;
}
figcaption, p > strong:first-child {
    font-size: 9pt;
    color: #333;
}
"""

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="{lang}">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>{css}</style>
<script>
MathJax = {{
  tex: {{
    inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
    displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
    processEscapes: true,
    tags: 'none'
  }},
  options: {{
    renderActions: {{
      addMenu: [],
      checkLoading: []
    }}
  }},
  startup: {{
    pageReady: () => {{
      return MathJax.startup.defaultPageReady().then(() => {{
        document.body.dataset.mathjaxReady = "1";
      }});
    }}
  }}
}};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" async></script>
</head>
<body>
{body}
</body>
</html>
"""


def md_to_html(md_text: str, title: str, lang: str = "en") -> str:
    extras = [
        "tables",
        "fenced-code-blocks",
        "footnotes",
        "header-ids",
        "strike",
        "task_list",
        "smarty-pants",
    ]
    body = markdown2.markdown(md_text, extras=extras)
    return HTML_TEMPLATE.format(title=title, lang=lang, css=CSS, body=body)


def build_pdf(html_path: Path, pdf_path: Path) -> bool:
    if not Path(EDGE_PATH).exists():
        print(f"Edge not found at {EDGE_PATH}", file=sys.stderr)
        return False
    from urllib.parse import quote
    abs_html = html_path.resolve()
    abs_pdf = pdf_path.resolve()
    # URL-encode the path (spaces → %20) so Edge accepts it
    file_url = "file:///" + quote(str(abs_html).replace("\\", "/"), safe="/:")
    cmd = [
        EDGE_PATH,
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--virtual-time-budget=20000",
        f"--print-to-pdf={abs_pdf}",
        "--print-to-pdf-no-header",
        file_url,
    ]
    print(f"running: {' '.join(repr(c) for c in cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if res.returncode != 0:
        print(f"Edge exited {res.returncode}", file=sys.stderr)
        print("STDOUT:", res.stdout, file=sys.stderr)
        print("STDERR:", res.stderr, file=sys.stderr)
        return False
    if abs_pdf.exists():
        print(f"PDF written to {abs_pdf}  ({abs_pdf.stat().st_size / 1024:.1f} KB)")
        return True
    else:
        print(f"PDF not found at {abs_pdf}")
        print("STDOUT:", res.stdout, file=sys.stderr)
        print("STDERR:", res.stderr, file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("md_path", type=str, help="path to markdown file")
    parser.add_argument("--lang", type=str, default="en", choices=["en", "zh"])
    parser.add_argument("--title", type=str, default="Paper")
    args = parser.parse_args()

    md_path = Path(args.md_path)
    md_text = md_path.read_text(encoding="utf-8")
    if not args.title or args.title == "Paper":
        # Use first H1 as title
        for line in md_text.splitlines():
            if line.startswith("# "):
                args.title = line[2:].strip()
                break

    lang = args.lang
    if lang == "en" and any("一" <= c <= "鿿" for c in md_text[:500]):
        lang = "zh"   # auto-detect Chinese characters in first 500 chars
    print(f"detected lang: {lang}")

    html = md_to_html(md_text, title=args.title, lang=lang)
    html_path = md_path.with_suffix(".html")
    html_path.write_text(html, encoding="utf-8")
    print(f"HTML written to {html_path}")

    pdf_path = md_path.with_suffix(".pdf")
    ok = build_pdf(html_path, pdf_path)
    if not ok:
        print("PDF build failed; HTML available — print to PDF from browser as fallback.")
        sys.exit(1)


if __name__ == "__main__":
    main()
