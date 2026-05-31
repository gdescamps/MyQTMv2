"""
Download OHLCV data via yfinance for QQQ strategy tickers.

Output: ./data/{ticker}.parquet
  - Columns: open, high, low, close, volume
  - Date-indexed, sorted ascending

Resumes automatically: skips tickers whose .parquet already exists.

Usage:
    python src/download_ohlcv.py
"""

import time
import pandas as pd
import yfinance as yf
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

START_DATE = "2000-01-01"
DELAY = 0.2

# Tickers needed for QQQ strategy
TICKERS = ["QQQ", "TLT"]

# VIX OHLCV (^VIX on Yahoo Finance)
VIX_TICKER = "^VIX"
VIX_FILENAME = "vix_ohlc"


def download_ticker(ticker: str) -> pd.DataFrame | None:
    try:
        df = yf.download(ticker, start=START_DATE, progress=False, auto_adjust=True)
    except Exception as e:
        print(f"[ERROR] {ticker}: {e}")
        return None

    if df is None or df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = [c.lower() for c in df.columns]
    cols = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    df = df[cols].dropna(subset=["close"])
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "date"
    return df.sort_index()


def fetch_and_save(ticker: str, filename: str = None) -> None:
    fname = filename or ticker
    path = DATA_DIR / f"{fname}.parquet"

    if path.exists():
        rows = pd.read_parquet(path).shape[0]
        print(f"  SKIP  {fname:<20} already {rows} rows")
        return

    print(f"  ..    {fname:<20} fetching...", end=" ", flush=True)
    df = download_ticker(ticker)
    time.sleep(DELAY)

    if df is None or df.empty:
        print("EMPTY")
        return

    df.to_parquet(path, engine="pyarrow", compression="snappy")
    print(f"{len(df)} rows  [{df.index[0].date()} → {df.index[-1].date()}]")


def main():
    print("Downloading OHLCV data for QQQ strategy\n")

    for ticker in TICKERS:
        fetch_and_save(ticker)

    fetch_and_save(VIX_TICKER, VIX_FILENAME)

    print(f"\nAll data in: {DATA_DIR}")


if __name__ == "__main__":
    main()
