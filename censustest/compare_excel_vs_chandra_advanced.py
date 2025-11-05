#!/usr/bin/env python3
"""
Advanced comparison of Chandra OCR output vs. Excel gold standard.

Adds:
- Multiset numeric matching with tolerance (precision/recall/F1)
- Table structure alignment (table/row/column coverage)
- Header text overlap + CER/WER for headers

Usage:
  python compare_excel_vs_chandra_advanced.py \
      --gold_xlsx ./1891_V1T2_OCR_202306.xlsx \
      --ocr_html ./results/1891_v1t2/1891_v1t2.html \
      --out_prefix ./adv_1891_v1t2

Dependencies: openpyxl, beautifulsoup4
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from bs4 import BeautifulSoup  # type: ignore
import openpyxl  # type: ignore


def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def tokenize_words(s: str) -> List[str]:
    s = normalize_ws(s)
    if not s:
        return []
    # split on non-alphanum boundaries while keeping useful tokens
    return re.findall(r"[A-Za-z0-9%./'-]+", s)


def cer(a: str, b: str) -> float:
    # character error rate = edit_distance / len(b)
    if not b:
        return 0.0 if not a else 1.0
    return edit_distance(a, b) / max(1, len(b))


def wer(a: str, b: str) -> float:
    ta, tb = tokenize_words(a), tokenize_words(b)
    if not tb:
        return 0.0 if not ta else 1.0
    return edit_distance_seq(ta, tb) / max(1, len(tb))


def edit_distance(a: str, b: str) -> int:
    # Levenshtein distance (O(n*m))
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        ca = a[i - 1]
        for j in range(1, m + 1):
            tmp = dp[j]
            cost = 0 if ca == b[j - 1] else 1
            dp[j] = min(
                dp[j] + 1,      # deletion
                dp[j - 1] + 1,  # insertion
                prev + cost,    # substitution
            )
            prev = tmp
    return dp[m]


def edit_distance_seq(a: Sequence[str], b: Sequence[str]) -> int:
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        ca = a[i - 1]
        for j in range(1, m + 1):
            tmp = dp[j]
            cost = 0 if ca == b[j - 1] else 1
            dp[j] = min(
                dp[j] + 1,
                dp[j - 1] + 1,
                prev + cost,
            )
            prev = tmp
    return dp[m]


# ---------- Number parsing and matching ----------

def normalize_number_str(token: str) -> Optional[str]:
    t = token.strip().replace(",", "")
    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", t):
        return t
    return None


def extract_numbers_str(text: str) -> List[str]:
    candidates = re.findall(r"(?<!\w)[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?!\w)", text)
    out: List[str] = []
    for c in candidates:
        n = normalize_number_str(c)
        if n is not None:
            out.append(n)
    return out


def to_float_safe(s: str) -> Optional[float]:
    try:
        return float(s)
    except Exception:
        return None


def greedy_match_with_tolerance(gold: List[float], ocr: List[float], abs_tol: float, rel_tol: float) -> int:
    # Greedy matching: for each gold value, find a closest OCR value within tolerance (unused yet)
    used = [False] * len(ocr)
    match = 0
    # sort both by value to speed search
    gold_sorted = sorted([(v, i) for i, v in enumerate(gold)], key=lambda x: x[0])
    ocr_sorted = sorted([(v, i) for i, v in enumerate(ocr)], key=lambda x: x[0])
    j = 0
    for gv, _ in gold_sorted:
        # advance j to near gv
        while j < len(ocr_sorted) and ocr_sorted[j][0] < gv - max(abs_tol, rel_tol * abs(gv)):
            j += 1
        # try candidates around j
        best_k = -1
        best_diff = float("inf")
        for k in (j - 2, j - 1, j, j + 1, j + 2):
            if 0 <= k < len(ocr_sorted):
                ov, oi = ocr_sorted[k]
                if used[oi]:
                    continue
                diff = abs(ov - gv)
                tol = max(abs_tol, rel_tol * max(abs(gv), 1.0))
                if diff <= tol and diff < best_diff:
                    best_diff = diff
                    best_k = k
        if best_k >= 0:
            _, oi = ocr_sorted[best_k]
            used[oi] = True
            match += 1
    return match


# ---------- Excel parsing ----------

def load_excel_tables(xlsx: Path) -> List[List[List[str]]]:
    wb = openpyxl.load_workbook(filename=str(xlsx), data_only=True)
    tables: List[List[List[str]]] = []
    for ws in wb.worksheets:
        # determine bounding box of non-empty cells
        rows = list(ws.iter_rows(values_only=True))
        if not rows:
            continue
        min_r, max_r, min_c, max_c = None, None, None, None
        for r_idx, row in enumerate(rows):
            for c_idx, v in enumerate(row):
                if v not in (None, ""):
                    min_r = r_idx if min_r is None else min(min_r, r_idx)
                    max_r = r_idx if max_r is None else max(max_r, r_idx)
                    min_c = c_idx if min_c is None else min(min_c, c_idx)
                    max_c = c_idx if max_c is None else max(max_c, c_idx)
        if min_r is None:
            continue
        grid: List[List[str]] = []
        for r in range(min_r, max_r + 1):
            row_vals: List[str] = []
            for c in range(min_c, max_c + 1):
                v = rows[r][c]
                if isinstance(v, float):
                    # avoid scientific notation
                    s = ("%f" % v).rstrip("0").rstrip(".")
                    row_vals.append(s)
                else:
                    row_vals.append(str(v) if v is not None else "")
            grid.append(row_vals)
        if grid:
            tables.append(grid)
    return tables


def excel_headers(table: List[List[str]]) -> List[str]:
    # heuristic: first non-empty row is header
    for row in table:
        if any(cell and str(cell).strip() for cell in row):
            return [normalize_ws(str(x)) for x in row]
    return []


# ---------- HTML parsing ----------

def parse_html_tables(html_text: str) -> List[List[List[str]]]:
    soup = BeautifulSoup(html_text, "html.parser")
    out: List[List[List[str]]] = []
    for t in soup.find_all("table"):
        rows: List[List[str]] = []
        # prefer thead+tbody if present; otherwise read all trs in order
        tr_nodes = t.find_all("tr")
        for tr in tr_nodes:
            cells = tr.find_all(["th", "td"])
            rows.append([normalize_ws(c.get_text(separator=" ")) for c in cells])
        if rows:
            out.append(rows)
    return out


def html_headers(table: List[List[str]]) -> List[str]:
    if not table:
        return []
    # if first row has any non-empty, assume header
    return [normalize_ws(c) for c in table[0]]


def numbers_from_table(table: List[List[str]]) -> List[str]:
    nums: List[str] = []
    for row in table:
        for cell in row:
            nums.extend(extract_numbers_str(str(cell)))
    return nums


def table_shape_signature(tables: List[List[List[str]]]) -> List[Tuple[int, int]]:
    sig: List[Tuple[int, int]] = []
    for grid in tables:
        rows = len(grid)
        cols = max((len(r) for r in grid), default=0)
        sig.append((rows, cols))
    sig.sort()
    return sig


def jaccard(a: List[str], b: List[str]) -> float:
    aset = set([s for s in a if s])
    bset = set([s for s in b if s])
    if not aset and not bset:
        return 1.0
    if not aset or not bset:
        return 0.0
    inter = len(aset & bset)
    union = len(aset | bset)
    return inter / max(1, union)


@dataclass
class AdvancedReport:
    # numeric set (global)
    num_set_gold: int
    num_set_ocr: int
    num_set_matched: int
    num_set_recall: float
    num_set_precision: float

    # numeric multiset with tolerance
    num_multi_gold: int
    num_multi_ocr: int
    num_multi_matched: int
    num_multi_recall: float
    num_multi_precision: float
    num_multi_f1: float

    # tables
    excel_tables: int
    ocr_tables: int
    shape_overlap: float  # fraction of excel table shapes represented in OCR shapes (multiset-unaware)

    # headers
    header_jaccard: float
    header_cer: float
    header_wer: float


def main() -> None:
    ap = argparse.ArgumentParser(description="Advanced Chandra vs Excel comparison")
    ap.add_argument("--gold_xlsx", required=True, type=Path)
    ap.add_argument("--ocr_html", required=True, type=Path)
    ap.add_argument("--out_prefix", type=Path, default=Path("adv_report"))
    ap.add_argument("--abs_tol", type=float, default=1.0, help="Absolute tolerance for numeric match")
    ap.add_argument("--rel_tol", type=float, default=0.005, help="Relative tolerance (fraction)")
    args = ap.parse_args()

    # Read sources
    html_text = args.ocr_html.read_text(encoding="utf-8", errors="ignore")
    excel_tables = load_excel_tables(args.gold_xlsx)
    ocr_tables = parse_html_tables(html_text)

    # Global numeric sets
    all_excel_text = "\n".join("\n".join("\t".join(r) for r in t) for t in excel_tables)
    all_ocr_text = BeautifulSoup(html_text, "html.parser").get_text(separator=" ")
    gold_set = set(extract_numbers_str(all_excel_text))
    ocr_set = set(extract_numbers_str(all_ocr_text))
    set_matched = len(gold_set & ocr_set)
    set_recall = set_matched / max(1, len(gold_set))
    set_precision = set_matched / max(1, len(ocr_set))

    # Multiset with tolerance
    gold_multi = [x for x in extract_numbers_str(all_excel_text)]
    ocr_multi = [x for x in extract_numbers_str(all_ocr_text)]
    gold_vals = [v for v in (to_float_safe(s) for s in gold_multi) if v is not None]
    ocr_vals = [v for v in (to_float_safe(s) for s in ocr_multi) if v is not None]
    multi_matched = greedy_match_with_tolerance(gold_vals, ocr_vals, args.abs_tol, args.rel_tol)
    multi_recall = multi_matched / max(1, len(gold_vals))
    multi_precision = multi_matched / max(1, len(ocr_vals))
    if multi_recall + multi_precision > 0:
        multi_f1 = 2 * multi_recall * multi_precision / (multi_recall + multi_precision)
    else:
        multi_f1 = 0.0

    # Table structures
    excel_sig = table_shape_signature(excel_tables)
    ocr_sig = table_shape_signature(ocr_tables)
    # shape overlap: fraction of excel shapes that appear in ocr shapes (set-based)
    excel_shape_set = set(excel_sig)
    ocr_shape_set = set(ocr_sig)
    shape_overlap = len(excel_shape_set & ocr_shape_set) / max(1, len(excel_shape_set))

    # Headers: aggregate across tables (concatenate headers)
    excel_hdrs: List[str] = []
    for t in excel_tables:
        excel_hdrs.extend([h for h in excel_headers(t) if h])
    ocr_hdrs: List[str] = []
    for t in ocr_tables:
        ocr_hdrs.extend([h for h in html_headers(t) if h])
    header_j = jaccard(excel_hdrs, ocr_hdrs)
    # For CER/WER, join headers into a string
    excel_hdr_text = " | ".join(excel_hdrs)
    ocr_hdr_text = " | ".join(ocr_hdrs)
    h_cer = cer(ocr_hdr_text, excel_hdr_text)
    h_wer = wer(ocr_hdr_text, excel_hdr_text)

    report = AdvancedReport(
        num_set_gold=len(gold_set),
        num_set_ocr=len(ocr_set),
        num_set_matched=set_matched,
        num_set_recall=set_recall,
        num_set_precision=set_precision,
        num_multi_gold=len(gold_vals),
        num_multi_ocr=len(ocr_vals),
        num_multi_matched=multi_matched,
        num_multi_recall=multi_recall,
        num_multi_precision=multi_precision,
        num_multi_f1=multi_f1,
        excel_tables=len(excel_tables),
        ocr_tables=len(ocr_tables),
        shape_overlap=shape_overlap,
        header_jaccard=header_j,
        header_cer=h_cer,
        header_wer=h_wer,
    )

    # Print concise summary
    print("=== Advanced OCR Accuracy Summary ===")
    print(f"Numeric set recall/precision: {report.num_set_recall:.3f} / {report.num_set_precision:.3f}")
    print(f"Numeric multiset tol match:   R {report.num_multi_recall:.3f}  P {report.num_multi_precision:.3f}  F1 {report.num_multi_f1:.3f}")
    print(f"Tables (gold/ocr):            {report.excel_tables} / {report.ocr_tables}")
    print(f"Table shape overlap:          {report.shape_overlap:.3f}")
    print(f"Header overlap (Jaccard):     {report.header_jaccard:.3f}")
    print(f"Header CER / WER:             {report.header_cer:.3f} / {report.header_wer:.3f}")

    # Save JSON
    out_json = args.out_prefix.with_suffix(".json")
    out_json.write_text(json.dumps(report.__dict__, indent=2), encoding="utf-8")
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()

