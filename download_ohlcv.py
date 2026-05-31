"""
Download OHLCV data via yfinance for all ETFs in the universe.

Strategy per ETF:
  1. Try to download the UCITS Boursorama ticker directly.
  2. If the UCITS ticker fails or has < MIN_ROWS of history, also download the
     proxy ticker (used for longer training history and signal computation).

Output: ./data/{ticker_normalized}.parquet
  - Columns: open, high, low, close, volume
  - Date-indexed, sorted ascending

Resumes automatically: skips tickers whose .parquet already exists.

Usage:
    python download_ohlcv.py
"""

import time
import sys
import pandas as pd
import yfinance as yf
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

START_DATE = "2000-01-01"
MIN_ROWS = 750      # ~3 years of trading days — below this, also fetch proxy
DELAY = 0.2         # seconds between requests


def ticker_filename(ticker: str) -> str:
    """Normalize ticker to safe filename (replace . with _)."""
    return ticker.replace(".", "_")


def download_ticker(ticker: str) -> pd.DataFrame | None:
    """Fetch full OHLCV history from yfinance. Returns None on failure."""
    try:
        df = yf.download(
            ticker,
            start=START_DATE,
            progress=False,
            auto_adjust=True,  # adjust for splits/dividends
        )
    except Exception as e:
        print(f"[ERROR] {ticker}: {e}")
        return None

    if df is None or df.empty:
        return None

    # Flatten MultiIndex columns if present (yfinance batch download)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = [c.lower() for c in df.columns]
    cols = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    df = df[cols].dropna(subset=["close"])
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "date"
    return df.sort_index()


def fetch_and_save(ticker: str, label: str) -> tuple[bool, int]:
    """Download ticker and save to parquet. Returns (success, row_count)."""
    fname = DATA_DIR / f"{ticker_filename(ticker)}.parquet"

    if fname.exists():
        rows = pd.read_parquet(fname).shape[0]
        print(f"  SKIP  {label:<12} {ticker:<14} already {rows} rows")
        return True, rows

    print(f"  ..    {label:<12} {ticker:<14} fetching...", end=" ", flush=True)
    df = download_ticker(ticker)
    time.sleep(DELAY)

    if df is None or df.empty:
        print("EMPTY")
        return False, 0

    df.to_parquet(fname, engine="pyarrow", compression="snappy")
    first = df.index[0].date()
    last = df.index[-1].date()
    print(f"{len(df)} rows  [{first} → {last}]")
    return True, len(df)


def main():
    sys.path.insert(0, str(Path(__file__).parent / "multietfs_strategy"))
    from etf import UNIVERSE

    print(f"Universe: {len(UNIVERSE)} ETFs\n")
    print("=" * 70)
    print("Step 1/2  UCITS tickers (Boursorama)")
    print("=" * 70)

    needs_proxy: list[tuple[str, str]] = []  # (ucits_ticker, proxy_ticker)

    for etf in UNIVERSE:
        ok, rows = fetch_and_save(etf.bourso, etf.theme)
        # Also queue proxy if UCITS failed or has insufficient history
        if etf.bourso != etf.proxy:
            if not ok or rows < MIN_ROWS:
                needs_proxy.append((etf.bourso, etf.proxy))

    print()
    print("=" * 70)
    print("Step 2/2  Proxy tickers (US / LSE) — for missing or short UCITS")
    print("=" * 70)

    # Also always download all proxy tickers (needed for shares outstanding matching)
    all_proxies = sorted(set(e.proxy for e in UNIVERSE if e.bourso != e.proxy))

    for proxy in all_proxies:
        label = "proxy"
        fetch_and_save(proxy, label)

    # Summary
    print()
    print("=" * 70)
    print("Summary — tickers requiring proxy fallback:")
    print("=" * 70)
    if needs_proxy:
        for ucits, proxy in needs_proxy:
            ucits_path = DATA_DIR / f"{ticker_filename(ucits)}.parquet"
            proxy_path = DATA_DIR / f"{ticker_filename(proxy)}.parquet"
            ucits_rows = pd.read_parquet(ucits_path).shape[0] if ucits_path.exists() else 0
            proxy_rows = pd.read_parquet(proxy_path).shape[0] if proxy_path.exists() else 0
            print(f"  {ucits:<12} ({ucits_rows:4d} rows)  → proxy {proxy:<10} ({proxy_rows:4d} rows)")
    else:
        print("  None — all UCITS tickers have sufficient history.")

    print(f"\nAll data in: {DATA_DIR}")


if __name__ == "__main__":
    main()
