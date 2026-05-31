"""
Download macro data from FRED for QQQ strategy.

FRED (free API key at https://fred.stlouisfed.org/docs/api/api_key.html):
  Set FRED_API_KEY in .env

Output: ./data/fred_{series}.parquet

Usage:
    python src/download_macro_data.py
"""

import os
import time
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

FRED_KEY = os.getenv("FRED_API_KEY") or os.getenv("FRED")
FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

# Series needed for QQQ strategy
FRED_SERIES = {
    "DBAA":  "baa_spread",   # Moody's BAA corporate bond yield spread
}


def fetch_fred(series_id: str, label: str) -> None:
    if not FRED_KEY:
        print("  [SKIP] FRED_API_KEY not set — add it to .env")
        return

    fname = DATA_DIR / f"fred_{label}.parquet"
    if fname.exists():
        rows = pd.read_parquet(fname).shape[0]
        print(f"  SKIP  fred_{label:<15} (already {rows} rows)")
        return

    print(f"  ..    fred_{label:<15} fetching...", end=" ", flush=True)
    params = {
        "series_id": series_id,
        "api_key": FRED_KEY,
        "file_type": "json",
        "observation_start": "2000-01-01",
        "units": "lin",
    }
    try:
        r = requests.get(FRED_BASE, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"[ERROR] {series_id}: {e}")
        return

    obs = data.get("observations", [])
    if not obs:
        print(f"[EMPTY]")
        return

    df = pd.DataFrame(obs)[["date", "value"]]
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna().rename(columns={"value": label})

    df.to_parquet(fname, engine="pyarrow", compression="snappy")
    print(f"{len(df)} rows  [{df.index[0].date()} → {df.index[-1].date()}]")


def main():
    print("Downloading FRED macro data for QQQ strategy\n")

    if not FRED_KEY:
        print("FRED_API_KEY not set — get a free key at:")
        print("https://fred.stlouisfed.org/docs/api/api_key.html")
        print("Then add FRED_API_KEY=your_key to .env")
        return

    for series_id, label in FRED_SERIES.items():
        fetch_fred(series_id, label)
        time.sleep(0.1)

    print("\nDone.")


if __name__ == "__main__":
    main()
