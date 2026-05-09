#!/usr/bin/env python3
"""Embed local <img src="..."> PNG/JPEG files in an HTML report as data URIs.

Usage:
  python analysis/embed_report_images.py \
    --input analysis/deep_report/report.html \
    --output analysis/deep_report/report_standalone.html
"""
from __future__ import annotations

import argparse
import base64
import mimetypes
import re
from pathlib import Path

IMG_RE = re.compile(r"(<img\s+[^>]*?src=)(['\"])([^'\"]+)(\2)", re.IGNORECASE)


def embed_images(input_path: Path, output_path: Path) -> tuple[int, list[str]]:
    html = input_path.read_text(encoding="utf-8")
    base_dir = input_path.parent
    embedded = 0
    missing: list[str] = []

    def repl(match: re.Match[str]) -> str:
        nonlocal embedded
        prefix, quote, src, suffix_quote = match.groups()
        if src.startswith("data:") or src.startswith("http://") or src.startswith("https://"):
            return match.group(0)
        img_path = (base_dir / src).resolve()
        if not img_path.exists():
            missing.append(src)
            return match.group(0)
        mime = mimetypes.guess_type(str(img_path))[0] or "application/octet-stream"
        b64 = base64.b64encode(img_path.read_bytes()).decode("ascii")
        embedded += 1
        return f"{prefix}{quote}data:{mime};base64,{b64}{suffix_quote}"

    out = IMG_RE.sub(repl, html)
    output_path.write_text(out, encoding="utf-8")
    return embedded, missing


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    embedded, missing = embed_images(Path(args.input), Path(args.output))
    print(f"embedded={embedded}")
    print(f"missing={missing}")
    print(f"output={Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
