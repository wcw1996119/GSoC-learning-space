"""Render paper_methods_{en,zh}.md to self-contained HTML with MathJax.

Output: paper_methods_en.html / paper_methods_zh.html

Equations (\\(...\\) inline, $$...$$ display, $...$ inline) are rendered by
MathJax at browser load time. To get a PDF: open the HTML in a browser and
print to PDF (Ctrl+P → "Save as PDF" → A4 / journal paper margins).

Uses Python-Markdown if available, falls back to a minimal in-house parser
that handles the constructs used in the source files: headings, paragraphs,
lists, tables, code blocks, blockquotes, horizontal rules, $$display$$ math,
and $inline$ math passed through to MathJax.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent

SRCS = [
    ("paper_methods_en.md", "paper_methods_en.html",
     "Methods — ML-ABM Framework for Urban Commuting"),
    ("paper_methods_zh.md", "paper_methods_zh.html",
     "方法 —— ML-ABM 城市通勤分析框架"),
]


HTML_TMPL = r"""<!DOCTYPE html>
<html lang="{lang}">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<script>
window.MathJax = {{
  tex: {{
    inlineMath: [['$', '$'], ['\\(', '\\)']],
    displayMath: [['$$', '$$'], ['\\[', '\\]']],
    processEscapes: true,
    tags: 'ams'
  }},
  options: {{
    skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']
  }}
}};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" async></script>
<style>
  @page {{ size: A4; margin: 2.2cm 1.8cm; }}
  body {{
    font-family: "Charter", "Source Serif Pro", "Georgia", "Times New Roman", serif;
    max-width: 760px;
    margin: 30px auto 60px;
    padding: 0 32px;
    color: #1a1a1a;
    line-height: 1.62;
    font-size: 11.5pt;
    text-align: justify;
    hyphens: auto;
  }}
  h1, h2, h3 {{
    font-family: "Helvetica Neue", "Arial", sans-serif;
    color: #111;
    font-weight: 600;
    line-height: 1.25;
  }}
  h1 {{ font-size: 1.55em; margin: 0 0 12px; }}
  h2 {{ font-size: 1.18em; margin: 32px 0 8px;
        padding-bottom: 4px; border-bottom: 1.5px solid #ddd; }}
  h3 {{ font-size: 1.0em;  margin: 20px 0 6px; color: #333; }}
  p  {{ margin: 6px 0 10px; }}
  ul {{ margin: 4px 0 10px 0; padding-left: 22px; }}
  li {{ margin: 2px 0; }}
  table {{
    border-collapse: collapse; margin: 12px 0 16px;
    font-size: 0.92em; width: 100%;
  }}
  th, td {{
    border: 1px solid #bbb; padding: 4px 8px;
    text-align: left; vertical-align: top;
  }}
  th {{ background: #eef0f3; font-weight: 600; }}
  pre {{
    background: #f6f6f4; border-left: 3px solid #ccc;
    padding: 10px 14px; overflow-x: auto;
    font-size: 0.78em; line-height: 1.4;
    font-family: "Consolas", "Menlo", monospace;
    margin: 12px 0 16px;
    white-space: pre-wrap; word-wrap: break-word;
  }}
  code {{
    background: #f0efed; padding: 1px 4px; border-radius: 3px;
    font-size: 0.88em; font-family: "Consolas", "Menlo", monospace;
  }}
  blockquote {{
    margin: 8px 16px; padding: 6px 14px;
    border-left: 3px solid #aaa; font-style: italic; color: #444;
  }}
  hr {{ border: 0; border-top: 1px solid #ccc; margin: 24px 0; }}
  .eqno {{
    float: right; font-family: "Helvetica Neue", sans-serif;
    color: #444; font-size: 0.85em;
  }}
  /* Print styles */
  @media print {{
    body {{ font-size: 10.5pt; padding: 0; max-width: none; }}
    h2 {{ break-after: avoid; }}
    table, pre, blockquote {{ break-inside: avoid; }}
  }}
</style>
</head>
<body>
{body}
</body>
</html>
"""


def inline_md(text: str) -> str:
    """Markdown inline -> HTML, preserving math passthrough.

    Math segments ($...$ and $$...$$) are placeholdered before any other
    transformations so commonmark / asterisk / underscore handling can't
    accidentally maul TeX commands like \beta or \cdot.
    """
    placeholders = []

    def stash(m):
        placeholders.append(m.group(0))
        return f"@@MATH{len(placeholders) - 1}@@"

    # display $$...$$ first, then inline $...$. Use non-greedy match.
    text = re.sub(r"\$\$([\s\S]+?)\$\$", stash, text)
    text = re.sub(r"\$([^\$\n]+?)\$", stash, text)
    # inline code spans
    text = re.sub(r"`([^`]+)`", lambda m: f"<code>{m.group(1)}</code>", text)
    # bold
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    # italic
    text = re.sub(r"(?<![*\w])\*([^*\n]+)\*(?!\*)", r"<em>\1</em>", text)
    # images: ![alt](url) — must come BEFORE links since they share syntax
    text = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)",
                  r'<img src="\2" alt="\1" style="max-width:100%; display:block; margin:8px auto;"/>',
                  text)
    # links [text](url)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', text)
    # restore math placeholders
    for i, ph in enumerate(placeholders):
        text = text.replace(f"@@MATH{i}@@", ph)
    return text


def parse_markdown(md: str) -> str:
    lines = md.split("\n")
    n = len(lines)
    i = 0
    out = []

    def collect_para_until_blank():
        nonlocal i
        para = []
        while i < n:
            cur = lines[i]
            cs = cur.strip()
            if (
                not cs or cs.startswith("#") or cs.startswith("|")
                or cs.startswith("- ") or cs.startswith("* ")
                or cs.startswith("```") or cs.startswith("> ")
                or cs in ("---", "***") or cs.startswith("$$")
            ):
                break
            para.append(cs)
            i += 1
        return " ".join(para)

    while i < n:
        line = lines[i]
        s = line.strip()
        if not s:
            i += 1
            continue

        # Headers
        if s.startswith("### "):
            out.append(f"<h3>{inline_md(s[4:].strip())}</h3>")
            i += 1
            continue
        if s.startswith("## "):
            out.append(f"<h2>{inline_md(s[3:].strip())}</h2>")
            i += 1
            continue
        if s.startswith("# "):
            out.append(f"<h1>{inline_md(s[2:].strip())}</h1>")
            i += 1
            continue
        if s in ("---", "***"):
            out.append("<hr>")
            i += 1
            continue

        # Display math block: $$...$$ on its own line(s)
        if s.startswith("$$"):
            block = []
            # may be on a single line
            m_one = re.match(r"^\$\$([\s\S]+?)\$\$\s*(\\tag\{[^}]+\})?\s*$", s)
            if m_one:
                eq_body = m_one.group(1).strip()
                tag = m_one.group(2) or ""
                out.append(f"<p>$$ {eq_body} {tag} $$</p>")
                i += 1
                continue
            # multi-line: collect till closing $$
            block.append(s.lstrip("$"))
            i += 1
            while i < n and "$$" not in lines[i]:
                block.append(lines[i])
                i += 1
            # closing line may have trailing tag/content
            if i < n:
                closing = lines[i]
                idx = closing.index("$$")
                block.append(closing[:idx])
                rest = closing[idx + 2:].strip()
                i += 1
            else:
                rest = ""
            eq = "\n".join(block).strip()
            out.append(f"<p>$$ {eq} {rest} $$</p>")
            continue

        # Code fence
        if s.startswith("```"):
            i += 1
            block = []
            while i < n and not lines[i].strip().startswith("```"):
                block.append(lines[i])
                i += 1
            if i < n:
                i += 1
            content = "\n".join(block)
            content = (content.replace("&", "&amp;")
                              .replace("<", "&lt;").replace(">", "&gt;"))
            out.append(f"<pre>{content}</pre>")
            continue

        # Tables
        if s.startswith("|") and i + 1 < n and re.match(
            r"^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$", lines[i + 1]
        ):
            def split_row(line):
                ln = line.strip()
                if ln.startswith("|"): ln = ln[1:]
                if ln.endswith("|"): ln = ln[:-1]
                return [c.strip() for c in ln.split("|")]
            headers = split_row(s)
            i += 2
            rows = []
            while i < n and lines[i].strip().startswith("|"):
                rows.append(split_row(lines[i]))
                i += 1
            html = "<table><thead><tr>"
            for h in headers:
                html += f"<th>{inline_md(h)}</th>"
            html += "</tr></thead><tbody>"
            for r in rows:
                html += "<tr>"
                for c in r:
                    html += f"<td>{inline_md(c)}</td>"
                html += "</tr>"
            html += "</tbody></table>"
            out.append(html)
            continue

        # Lists
        if s.startswith("- ") or s.startswith("* "):
            items = []
            while i < n and (
                lines[i].strip().startswith("- ") or
                lines[i].strip().startswith("* ")
            ):
                items.append(lines[i].strip()[2:])
                i += 1
            html = "<ul>"
            for it in items:
                html += f"<li>{inline_md(it)}</li>"
            html += "</ul>"
            out.append(html)
            continue

        # Blockquote
        if s.startswith("> "):
            quote = []
            while i < n and lines[i].strip().startswith("> "):
                quote.append(lines[i].strip()[2:])
                i += 1
            out.append(f"<blockquote>{inline_md(' '.join(quote))}</blockquote>")
            continue

        # Paragraph
        para = collect_para_until_blank()
        if para:
            out.append(f"<p>{inline_md(para)}</p>")

    return "\n".join(out)


def main():
    for src_name, out_name, title in SRCS:
        src = ROOT / src_name
        if not src.exists():
            print(f"SKIP: {src} not found")
            continue
        md = src.read_text(encoding="utf-8")
        body_html = parse_markdown(md)
        lang = "zh-CN" if "zh" in src_name else "en"
        html = HTML_TMPL.format(lang=lang, title=title, body=body_html)
        out = ROOT / out_name
        out.write_text(html, encoding="utf-8")
        print(f"  {out.relative_to(ROOT)}  ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
