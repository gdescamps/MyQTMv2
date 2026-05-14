"""
Parse iShares XLS (SpreadsheetML) fund data files → parquet.

Input:  ./data/ishares/{TICKER}_fund.xls
Output: ./data/ishares_{ticker}_hist.parquet

Each parquet has a date index with columns:
  nav               : NAV per share (float)
  shares_outstanding: total shares outstanding (float)

Usage:
    python parse_ishares_xls.py
"""

import re
import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
ISHARES_DIR = DATA_DIR / "ishares"
NS = "urn:schemas-microsoft-com:office:spreadsheet"


def _fix_xml(content: bytes) -> str:
    """Strip BOM, fix unescaped & in attribute values."""
    text = content.decode("utf-8-sig", errors="replace")
    return re.sub(r"&(?!amp;|lt;|gt;|quot;|apos;|#)", "&amp;", text)


def parse_historical_sheet(path: Path) -> pd.DataFrame | None:
    """Extract the 'Historical' sheet from an iShares SpreadsheetML XLS file."""
    with open(path, "rb") as f:
        content = f.read()

    try:
        root = ET.fromstring(_fix_xml(content))
    except ET.ParseError as e:
        print(f"  [ERROR] XML parse failed for {path.name}: {e}")
        return None

    sheets = root.findall(f"{{{NS}}}Worksheet")
    hist_sheet = None
    for s in sheets:
        if s.get(f"{{{NS}}}Name") == "Historical":
            hist_sheet = s
            break

    if hist_sheet is None:
        print(f"  [SKIP]  No 'Historical' sheet in {path.name}")
        return None

    rows = hist_sheet.findall(f".//{{{NS}}}Row")
    if len(rows) < 2:
        print(f"  [EMPTY] {path.name}: only {len(rows)} rows in Historical sheet")
        return None

    # Read header dynamically (4 or 5 columns depending on fund)
    header_cells = [c.find(f"{{{NS}}}Data") for c in rows[0].findall(f"{{{NS}}}Cell")]
    headers = [c.text if c is not None else None for c in header_cells]

    records = []
    for row in rows[1:]:  # skip header row
        cells = [c.find(f"{{{NS}}}Data") for c in row.findall(f"{{{NS}}}Cell")]
        vals = [c.text if c is not None else None for c in cells]
        if vals and vals[0]:
            # Pad to match header length if needed
            while len(vals) < len(headers):
                vals.append(None)
            records.append(vals[:len(headers)])

    if not records:
        return None

    df = pd.DataFrame(records, columns=headers)
    # Normalize column names to lowercase with underscores
    col_map = {c: c.lower().replace(" ", "_") for c in df.columns if c}
    df = df.rename(columns=col_map)

    df["as_of"] = pd.to_datetime(df["as_of"], format="%b %d, %Y", errors="coerce")
    df = df.dropna(subset=["as_of"]).set_index("as_of").sort_index()
    df.index.name = "date"

    # Parse numeric columns — values may contain commas (e.g. "125,600,000")
    for col in ["nav_per_share", "shares_outstanding"]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", "", regex=False).replace("--", None),
                errors="coerce",
            )

    out_cols = [c for c in ["nav_per_share", "shares_outstanding"] if c in df.columns]
    return df[out_cols].dropna(subset=["shares_outstanding"])


def main():
    xls_files = sorted(ISHARES_DIR.glob("*_fund.xls"))
    if not xls_files:
        print(f"No *_fund.xls files found in {ISHARES_DIR}")
        print("Run download_ishares_xls.py first.")
        return

    print(f"Parsing {len(xls_files)} iShares XLS files → parquet\n")
    ok, skipped, failed = 0, 0, 0

    for xls_path in xls_files:
        ticker = xls_path.stem.replace("_fund", "")
        out_path = DATA_DIR / f"ishares_{ticker}_hist.parquet"

        if out_path.exists():
            rows = pd.read_parquet(out_path).shape[0]
            print(f"  SKIP  {ticker:<8}  (already {rows} rows)")
            skipped += 1
            continue

        df = parse_historical_sheet(xls_path)
        if df is None or df.empty:
            failed += 1
            continue

        df.to_parquet(out_path, engine="pyarrow", compression="snappy")
        first = df.index[0].date()
        last = df.index[-1].date()
        print(f"  OK    {ticker:<8}  {len(df)} rows  [{first} → {last}]")
        ok += 1

    print(f"\nDone: {ok} parsed, {skipped} skipped, {failed} failed")
    print(f"Output: {DATA_DIR}/ishares_*_hist.parquet")


if __name__ == "__main__":
    main()
