"""Render paper_methods_*.html to PDF via headless MS Edge.

Edge (Chromium) is bundled on Windows 10/11 and can render HTML to PDF
out-of-the-box with full MathJax support. This script invokes its
headless mode using subprocess and writes a PDF next to the HTML.

Run: python methodology/render_pdf.py
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

EDGE_CANDIDATES = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]

CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
]


def find_browser() -> str:
    for c in EDGE_CANDIDATES + CHROME_CANDIDATES:
        if Path(c).exists():
            return c
    raise RuntimeError("No Edge or Chrome found. Please install one.")


def render(html_path: Path, pdf_path: Path, browser: str):
    # MathJax loads from CDN async; we need the browser to wait for render
    # before printing. Edge's --print-to-pdf supports --virtual-time-budget
    # to wait a fixed wall-clock for JS to finish (in milliseconds).
    cmd = [
        browser,
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        f"--print-to-pdf={pdf_path}",
        "--print-to-pdf-no-header",
        "--virtual-time-budget=30000",  # 30 sec wall-time for MathJax CDN + render
        "--run-all-compositor-stages-before-draw",
        f"file:///{html_path.resolve().as_posix()}",
    ]
    print(f"  rendering {html_path.name} → {pdf_path.name} ...")
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    dt = time.time() - t0
    if pdf_path.exists():
        size_kb = pdf_path.stat().st_size // 1024
        print(f"  done in {dt:.1f}s  ({size_kb} KB)")
        return True
    else:
        print(f"  FAILED — stderr:\n{proc.stderr[:600]}")
        return False


def main():
    browser = find_browser()
    print(f"Using browser: {browser}")
    pairs = [
        (ROOT / "paper_methods_en.html", ROOT / "paper_methods_en.pdf"),
        (ROOT / "paper_methods_zh.html", ROOT / "paper_methods_zh.pdf"),
    ]
    ok = True
    for html, pdf in pairs:
        if not html.exists():
            print(f"SKIP: {html} not found (run export_methods_html.py first)")
            ok = False
            continue
        ok &= render(html, pdf, browser)
    print("\nAll done." if ok else "\nSome renders failed; see above.")


if __name__ == "__main__":
    main()
