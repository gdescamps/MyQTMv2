"""
Download OHLCV data via yfinance.

Two groups:
  - RISK_OFF: tickers needed for the risk-off strategy (QQQ, SPY, ACWI + VIX, TLT)
  - EXTRA: other tickers for exploration (GLD, BTC-USD, etc.)

Usage:
    python src/download_ohlcv.py            # risk-off only
    python src/download_ohlcv.py --all      # risk-off + extra
    python src/download_ohlcv.py --extra    # extra only
"""

import sys
import time
import pandas as pd
import yfinance as yf
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

START_DATE = "2000-01-01"
DELAY = 0.2

# Risk-off strategy: equity indices + features
RISK_OFF_TICKERS = ["QQQ", "SPY", "ACWI", "TLT"]
VIX_TICKER = "^VIX"
VIX_FILENAME = "vix_ohlc"

# Extra tickers for exploration / other strategies
EXTRA_TICKERS = ["GLD", "BTC-USD", "EWG", "XLE", "ITA", "EWZ", "DBC", "EEM", "GC=F", "^NDX"]


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


def fetch_and_save(ticker: str, filename: str = None, force: bool = False) -> None:
    fname = filename or ticker
    path = DATA_DIR / f"{fname}.parquet"
    revised_path = DATA_DIR / f"{fname}_revised.parquet"

    if path.exists() and not force:
        rows = pd.read_parquet(path).shape[0]
        print(f"  SKIP  {fname:<20} already {rows} rows")
        return

    print(f"  ..    {fname:<20} fetching...", end=" ", flush=True)
    df_fresh = download_ticker(ticker)
    time.sleep(DELAY)

    if df_fresh is None or df_fresh.empty:
        print("EMPTY")
        return

    # Always save full revised version for comparison
    df_fresh.to_parquet(revised_path, engine="pyarrow", compression="snappy")

    if path.exists():
        # Incremental: keep existing rows (point-in-time), only append new dates
        df_old = pd.read_parquet(path)
        new_dates = df_fresh.index.difference(df_old.index)
        if len(new_dates) > 0:
            df_merged = pd.concat([df_old, df_fresh.loc[new_dates]]).sort_index()
            df_merged.to_parquet(path, engine="pyarrow", compression="snappy")
            print(f"{len(df_merged)} rows (+{len(new_dates)} new)  "
                  f"[{df_merged.index[0].date()} → {df_merged.index[-1].date()}]")
        else:
            print(f"{len(df_old)} rows (up to date)")
    else:
        df_fresh.to_parquet(path, engine="pyarrow", compression="snappy")
        print(f"{len(df_fresh)} rows  [{df_fresh.index[0].date()} → {df_fresh.index[-1].date()}]")


def download_risk_off(force=False):
    """Download tickers needed for the risk-off strategy."""
    print("Downloading risk-off strategy data\n")
    for ticker in RISK_OFF_TICKERS:
        fetch_and_save(ticker, force=force)
    fetch_and_save(VIX_TICKER, VIX_FILENAME, force=force)
    print()


def download_extra(force=False):
    """Download extra tickers for exploration."""
    print("Downloading extra tickers\n")
    for ticker in EXTRA_TICKERS:
        fetch_and_save(ticker, force=force)
    print()


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else ""

    if arg == "--all":
        download_risk_off()
        download_extra()
    elif arg == "--extra":
        download_extra()
    else:
        download_risk_off()

    print(f"All data in: {DATA_DIR}")


if __name__ == "__main__":
    main()
