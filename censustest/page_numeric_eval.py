#!/usr/bin/env python3
"""
Page-aware numeric precision for OCR vs Excel gold.

Splits the Chandra HTML by page markers (<!-- Page N -->), extracts numbers per
page, and compares to the global set of gold numbers from the Excel workbook.

Outputs a CSV with columns:
  page, ocr_count, matched_count, precision, sample_missing, sample_matched

Usage:
  python page_numeric_eval.py \
      --gold_xlsx ./1891_V1T2_OCR_202306.xlsx \
      --ocr_html ./results/1891_v1t2/1891_v1t2.html \
      --out ./results/1891_v1t2/per_page_eval.csv
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import List, Tuple

from bs4 import BeautifulSoup  # type: ignore
import openpyxl  # type: ignore


def normalize_number_str(token: str) -> str | None:
    t = token.strip().replace(",", "")
    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", t):
        return t
    return None


def extract_numbers(text: str) -> List[str]:
    raw = re.findall(r"(?<!\w)[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?!\w)", text)
    out: List[str] = []
    for tok in raw:
        n = normalize_number_str(tok)
        if n is not None:
            out.append(n)
    return out


def load_gold_numbers(xlsx: Path) -> set[str]:
    wb = openpyxl.load_workbook(str(xlsx), data_only=True)
    nums: set[str] = set()
    for ws in wb.worksheets:
        for row in ws.iter_rows(values_only=True):
            for v in row:
                if v is None:
                    continue
                if isinstance(v, (int, float)):
                    s = ("%f" % float(v)).rstrip("0").rstrip(".")
                    n = normalize_number_str(s)
                    if n:
                        nums.add(n)
                else:
                    nums.update(extract_numbers(str(v)))
    return nums


def split_html_pages(html_text: str) -> List[Tuple[int, str]]:
    # Split on HTML comments <!-- Page N --> that Chandra writes when paginate_output is enabled
    parts = re.split(r"<!--\s*Page\s*(\d+)\s*-->", html_text)
    # parts pattern: [before, page1num, page1content, page2num, page2content, ...]
    pages: List[Tuple[int, str]] = []
    if len(parts) <= 1:
        # Single page (no markers)
        pages.append((1, html_text))
        return pages
    # The first element 'before' may be empty or preface; skip
    for i in range(1, len(parts), 2):
        try:
            pg = int(parts[i])
        except Exception:
            continue
        content = parts[i + 1] if i + 1 < len(parts) else ""
        pages.append((pg, content))
    return pages


def main() -> None:
    ap = argparse.ArgumentParser(description="Page-aware numeric precision vs gold")
    ap.add_argument("--gold_xlsx", required=True, type=Path)
    ap.add_argument("--ocr_html", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    html_text = args.ocr_html.read_text(encoding="utf-8", errors="ignore")
    gold_set = load_gold_numbers(args.gold_xlsx)

    pages = split_html_pages(html_text)
    rows: List[dict] = []
    for pg, content in pages:
        # strip tags to text
        text = BeautifulSoup(content, "html.parser").get_text(separator=" ")
        ocr_nums = extract_numbers(text)
        ocr_set = set(ocr_nums)
        matched = sorted(ocr_set & gold_set)
        missing = sorted(ocr_set - gold_set)
        precision = (len(matched) / len(ocr_set)) if ocr_set else 1.0
        rows.append(
            {
                "page": pg,
                "ocr_count": len(ocr_set),
                "matched_count": len(matched),
                "precision": f"{precision:.4f}",
                "sample_missing": "; ".join(missing[:10]),
                "sample_matched": "; ".join(matched[:10]),
            }
        )

    # sort by page
    rows.sort(key=lambda r: int(r["page"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "page",
                "ocr_count",
                "matched_count",
                "precision",
                "sample_missing",
                "sample_matched",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Saved per-page report: {args.out}")


if __name__ == "__main__":
    main()

