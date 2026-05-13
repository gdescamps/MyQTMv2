"""
Download maximum historical OHLCV data from FMP for all ETF proxy tickers.

Usage:
    python download_etf_data.py

Output: ./data/{TICKER}.parquet  (one file per ticker, date-indexed)
Columns: open, high, low, close, adj_close, volume

Resumes automatically: skips tickers whose .parquet already exists.
"""

import os
import sys
import time
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# Load FMP API key from .env
load_dotenv(Path(__file__).parent / ".env")
API_KEY = os.getenv("FMPTAPI")
if not API_KEY:
    sys.exit("Error: FMPTAPI not found in .env")

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

BASE_URL = "https://financialmodelingprep.com/api/v3"
FROM_DATE = "2000-01-01"   # request max history
DELAY = 0.25               # seconds between requests (rate limit ~4 req/s)


def fetch_ohlcv(ticker: str) -> pd.DataFrame | None:
    """Fetch full OHLCV history for a ticker from FMP."""
    url = f"{BASE_URL}/historical-price-full/{ticker}"
    params = {"apikey": API_KEY, "from": FROM_DATE}
    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
    except requests.RequestException as e:
        print(f"  [ERROR] {ticker}: request failed — {e}")
        return None

    data = r.json()
    if "Error Message" in data:
        print(f"  [SKIP]  {ticker}: {data['Error Message']}")
        return None

    historical = data.get("historical") or data.get("historicalStockList", [])
    if not historical:
        print(f"  [EMPTY] {ticker}: no historical data returned")
        return None

    df = pd.DataFrame(historical)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    # Normalize column names
    rename = {
        "open": "open", "high": "high", "low": "low",
        "close": "close", "adjClose": "adj_close", "volume": "volume",
    }
    df = df.rename(columns=rename)
    cols = [c for c in ("open", "high", "low", "close", "adj_close", "volume") if c in df.columns]
    return df[cols].astype(float)


def ticker_to_filename(ticker: str) -> str:
    """Convert ticker to safe filename (replace . with _)."""
    return ticker.replace(".", "_")


def main():
    # Import universe from etf.py at repo root
    sys.path.insert(0, str(Path(__file__).parent))
    from etf import fmp_tickers

    tickers = fmp_tickers()
    print(f"Downloading {len(tickers)} FMP tickers -> {DATA_DIR}/\n")

    ok, skipped, failed = 0, 0, 0

    for i, ticker in enumerate(tickers, 1):
        fname = DATA_DIR / f"{ticker_to_filename(ticker)}.parquet"

        if fname.exists():
            rows = pd.read_parquet(fname).shape[0]
            print(f"  [{i:2d}/{len(tickers)}] {ticker:<12} SKIP  (already {rows} rows)")
            skipped += 1
            continue

        print(f"  [{i:2d}/{len(tickers)}] {ticker:<12} fetching...", end=" ", flush=True)
        df = fetch_ohlcv(ticker)

        if df is not None and not df.empty:
            df.to_parquet(fname, engine="pyarrow", compression="snappy")
            print(f"{len(df)} rows  [{df.index[0].date()} → {df.index[-1].date()}]")
            ok += 1
        else:
            failed += 1

        time.sleep(DELAY)

    print(f"\nDone: {ok} downloaded, {skipped} skipped, {failed} failed")
    print(f"Data directory: {DATA_DIR}")


if __name__ == "__main__":
    main()
