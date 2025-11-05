#!/usr/bin/env python3
"""
Compare Chandra OCR output against a human-corrected Excel gold standard.

This script focuses on a pragmatic metric: numeric agreement.
It extracts all numbers from the Chandra HTML (or Markdown) output and from
the Excel workbook, normalizes them (strip commas, keep sign/decimals), and
computes precision/recall of OCR vs. gold.

Usage examples:
  python compare_excel_vs_chandra.py \
      --gold_xlsx ./censustest/1891_V1T2_OCR_202306.xlsx \
      --ocr_html ./results/1891_v1t2/1891_v1t2/1891_v1t2.html \
      --out ./results/1891_v1t2/compare_1891_v1t2.csv

  python compare_excel_vs_chandra.py \
      --gold_xlsx ./censustest/1891_V1T2_OCR_202306.xlsx \
      --ocr_dir   ./results/1891_v1t2/ \
      --out ./results/1891_v1t2/compare_1891_v1t2.csv

Notes:
- Requires: beautifulsoup4, openpyxl (pip install openpyxl bs4)
- If both --ocr_html and --ocr_dir are provided, --ocr_html is used.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Set, Tuple

from bs4 import BeautifulSoup  # type: ignore

try:
    import openpyxl  # type: ignore
except Exception as e:  # pragma: no cover
    openpyxl = None


Number = str  # canonical string form of a number (e.g., "-12345.67")


def _normalize_number(token: str) -> Number | None:
    """Normalize numeric tokens by removing thousands separators and normalizing format.

    Returns None if token does not represent a valid number after cleanup.
    """
    t = token.strip()
    # Remove thousands separators like 1,234,567
    t = t.replace(",", "")
    # Accept optional sign, digits, optional decimal
    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", t):
        # Normalize to standard string; keep sign and decimals as written
        # Avoid converting to float to prevent precision issues
        # Strip leading zeros (but keep single zero and decimals)
        sign = ""
        if t[0] in "+-":
            sign, t = t[0], t[1:]
        if "." in t:
            int_part, frac_part = t.split(".", 1)
            int_part = int_part.lstrip("0") or "0"
            # Strip trailing zeros in fractional part but keep at least one digit if any
            frac_part = frac_part.rstrip("0")
            if frac_part:
                return f"{sign}{int_part}.{frac_part}"
            else:
                return f"{sign}{int_part}"
        else:
            t = t.lstrip("0") or "0"
            return f"{sign}{t}"
    return None


def extract_numbers_from_text(text: str) -> Set[Number]:
    # Match numbers like 1,234, -56, 78.90, but avoid parts of words
    raw = re.findall(r"(?<!\w)[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?!\w)", text)
    out: Set[Number] = set()
    for tok in raw:
        n = _normalize_number(tok)
        if n is not None:
            out.add(n)
    return out


def read_chandra_text(ocr_html: Path | None, ocr_dir: Path | None) -> str:
    if ocr_html and ocr_html.exists():
        return ocr_html.read_text(encoding="utf-8", errors="ignore")

    if ocr_dir and ocr_dir.exists():
        # Try common locations: <dir>/<base>/<base>.html or any single html under dir
        candidates: List[Path] = []
        for p in ocr_dir.rglob("*.html"):
            candidates.append(p)
        if not candidates:
            # try markdown
            for p in ocr_dir.rglob("*.md"):
                candidates.append(p)
        if not candidates:
            raise FileNotFoundError(f"No HTML/MD found under {ocr_dir}")
        # Prefer the deepest file (likely <base>/<base>.html)
        candidates.sort(key=lambda p: (len(p.parts), p.name))
        return candidates[-1].read_text(encoding="utf-8", errors="ignore")

    raise FileNotFoundError("Provide --ocr_html or --ocr_dir pointing to Chandra output")


def html_to_visible_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    # Remove script/style
    for tag in soup(["script", "style"]):
        tag.extract()
    # Get text
    text = soup.get_text(separator=" ")
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def read_excel_numbers(xlsx_path: Path) -> Tuple[Set[Number], int]:
    if openpyxl is None:
        raise RuntimeError(
            "openpyxl is required. Please install with: pip install openpyxl"
        )
    wb = openpyxl.load_workbook(filename=str(xlsx_path), data_only=True)
    numbers: Set[Number] = set()
    cell_count = 0
    for ws in wb.worksheets:
        for row in ws.iter_rows(values_only=True):
            for v in row:
                cell_count += 1
                if v is None:
                    continue
                if isinstance(v, (int, float)):
                    n = _normalize_number(str(v))
                    if n:
                        numbers.add(n)
                else:
                    nset = extract_numbers_from_text(str(v))
                    numbers.update(nset)
    return numbers, cell_count


@dataclass
class ComparisonResult:
    gold_count: int
    ocr_count: int
    matched_count: int
    recall: float
    precision: float
    missing: List[Number]
    extra: List[Number]


def compare_numeric_sets(gold: Set[Number], ocr: Set[Number]) -> ComparisonResult:
    inter = gold & ocr
    missing = sorted(gold - ocr)
    extra = sorted(ocr - gold)
    recall = (len(inter) / len(gold)) if gold else 1.0
    precision = (len(inter) / len(ocr)) if ocr else 1.0
    return ComparisonResult(
        gold_count=len(gold),
        ocr_count=len(ocr),
        matched_count=len(inter),
        recall=recall,
        precision=precision,
        missing=missing,
        extra=extra,
    )


def save_report_csv(path: Path, result: ComparisonResult) -> None:
    lines: List[str] = []
    lines.append("metric,value")
    lines.append(f"gold_count,{result.gold_count}")
    lines.append(f"ocr_count,{result.ocr_count}")
    lines.append(f"matched_count,{result.matched_count}")
    lines.append(f"recall,{result.recall:.4f}")
    lines.append(f"precision,{result.precision:.4f}")
    lines.append("")
    lines.append("missing_gold_values")
    for v in result.missing[:1000]:
        lines.append(str(v))
    lines.append("")
    lines.append("extra_ocr_values")
    for v in result.extra[:1000]:
        lines.append(str(v))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare Chandra OCR vs Excel (numeric overlap)")
    ap.add_argument("--gold_xlsx", required=True, type=Path, help="Path to gold XLSX")
    ap.add_argument("--ocr_html", type=Path, help="Path to Chandra HTML/MD file")
    ap.add_argument("--ocr_dir", type=Path, help="Directory containing Chandra outputs")
    ap.add_argument("--out", type=Path, help="CSV report output path", default=Path("./comparison.csv"))
    args = ap.parse_args()

    if not args.gold_xlsx.exists():
        raise FileNotFoundError(f"Gold XLSX not found: {args.gold_xlsx}")

    raw_text = read_chandra_text(args.ocr_html, args.ocr_dir)
    # If it's HTML, strip tags; if it's MD, this still reduces noise
    visible_text = html_to_visible_text(raw_text)

    ocr_numbers = extract_numbers_from_text(visible_text)
    gold_numbers, _ = read_excel_numbers(args.gold_xlsx)

    result = compare_numeric_sets(gold_numbers, ocr_numbers)

    print("=== Numeric Comparison (Chandra vs Gold Excel) ===")
    print(f"Gold unique numbers: {result.gold_count}")
    print(f"OCR unique numbers:  {result.ocr_count}")
    print(f"Matched numbers:     {result.matched_count}")
    print(f"Recall:              {result.recall:.3f}")
    print(f"Precision:           {result.precision:.3f}")
    print()
    if result.missing:
        print(f"Missing in OCR (sample {min(20, len(result.missing))}): {result.missing[:20]}")
    if result.extra:
        print(f"Extra in OCR (sample {min(20, len(result.extra))}):   {result.extra[:20]}")

    save_report_csv(args.out, result)
    # Also dump JSON summary alongside
    summary_json = args.out.with_suffix(".json")
    summary_json.write_text(
        json.dumps(
            {
                "gold_count": result.gold_count,
                "ocr_count": result.ocr_count,
                "matched_count": result.matched_count,
                "recall": result.recall,
                "precision": result.precision,
                "missing_sample": result.missing[:100],
                "extra_sample": result.extra[:100],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved report: {args.out}")
    print(f"Saved summary: {summary_json}")


if __name__ == "__main__":
    main()

