"""
Download macro signals from FRED + compute institutional flow proxies from OHLCV.

FRED (free API key at https://fred.stlouisfed.org/docs/api/api_key.html):
  Set FRED_API_KEY in .env

Flow proxies (no external API needed):
  Computed from OHLCV already in ./data/*.parquet:
  - dollar_volume = close × volume  (z-score vs 20d avg)
  - risk_rotation = dollar_volume(risk-on basket) / dollar_volume(risk-off basket)

Note: iShares shares outstanding (actual ETF flows) requires browser automation
      or a paid service (EPFR, Bloomberg) — not implemented here.

Output: ./data/fred_{series}.parquet, ./data/flow_proxies.parquet
"""

import os
import sys
import time
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# FRED — macro indicators
# ---------------------------------------------------------------------------
FRED_KEY = os.getenv("FRED_API_KEY") or os.getenv("FRED")
FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

FRED_SERIES = {
    "BAMLH0A0HYM2":       "hy_spread",    # ICE BofA HY OAS (risk appetite)
    "BAMLC0A0CM":         "ig_spread",    # ICE BofA IG OAS (credit conditions)
    "VIXCLS":             "vix",          # CBOE VIX (regime detection)
    "DGS10":              "yield_10y",    # 10Y Treasury yield
    "DGS2":               "yield_2y",     # 2Y Treasury yield
    "T10Y2Y":             "yield_curve",  # 10Y-2Y spread (inversion signal)
    "DCOILWTICO":         "wti_crude",    # WTI crude oil price
    "DTWEXBGS":           "dxy",          # Broad USD index vs basket (daily, from 2006)
    # Gold covered via IAU OHLCV — FRED series requires paid subscription
}


def fetch_fred(series_id: str, label: str) -> pd.DataFrame | None:
    if not FRED_KEY:
        print("  [SKIP] FRED_API_KEY not set — add it to .env (free at fred.stlouisfed.org)")
        return None

    fname = DATA_DIR / f"fred_{label}.parquet"
    if fname.exists():
        rows = pd.read_parquet(fname).shape[0]
        print(f"  SKIP  fred_{label:<15} (already {rows} rows)")
        return None

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
        print(f"  [ERROR] {series_id}: {e}")
        return None

    obs = data.get("observations", [])
    if not obs:
        print(f"  [EMPTY] {series_id}")
        return None

    df = pd.DataFrame(obs)[["date", "value"]]
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna().rename(columns={"value": label})

    df.to_parquet(fname, engine="pyarrow", compression="snappy")
    print(f"  OK    fred_{label:<15} {len(df)} rows  [{df.index[0].date()} → {df.index[-1].date()}]")
    return df


# ---------------------------------------------------------------------------
# Flow proxies — computed from OHLCV already in ./data/*.parquet
# ---------------------------------------------------------------------------
# Risk-on basket  : QQQ, XLK, XLF, XLY  (equity / growth)
# Risk-off basket : TLT, GLD, TIP        (bonds / safe haven)
# HY activity     : HYG                  (credit risk appetite)

RISK_ON  = ["QQQ", "XLK", "XLF", "XLY"]
RISK_OFF = ["TLT", "GLD", "TIP"]


def build_flow_proxies() -> None:
    """
    Compute dollar-volume z-scores and risk rotation index from OHLCV parquets.

    Output: ./data/flow_proxies.parquet
    Columns per ticker: {ticker}_dvol_z20   (dollar volume 20d z-score)
    Extra:  rotation_z20  (risk-on dvol sum / risk-off dvol sum, z-scored)
    """
    fname = DATA_DIR / "flow_proxies.parquet"
    if fname.exists():
        rows = pd.read_parquet(fname).shape[0]
        print(f"  SKIP  flow_proxies (already {rows} rows)")
        return

    all_tickers = sorted(set(RISK_ON + RISK_OFF + ["HYG", "IEF"]))
    frames = {}

    for ticker in all_tickers:
        path = DATA_DIR / f"{ticker}.parquet"
        if not path.exists():
            print(f"  [MISSING] {ticker}.parquet — run download_etf_data.py first")
            continue
        df = pd.read_parquet(path)[["close", "volume"]].dropna()
        dvol = df["close"] * df["volume"]
        # 20-day rolling z-score of dollar volume
        roll_mean = dvol.rolling(20).mean()
        roll_std  = dvol.rolling(20).std()
        frames[f"{ticker}_dvol_z20"] = (dvol - roll_mean) / roll_std.replace(0, float("nan"))

    if not frames:
        print("  [ERROR] No OHLCV data found — run download_etf_data.py first")
        return

    result = pd.DataFrame(frames).dropna(how="all")

    # Risk rotation index: sum risk-on dvol / sum risk-off dvol (z-scored)
    on_cols  = [f"{t}_dvol_z20" for t in RISK_ON  if f"{t}_dvol_z20" in result.columns]
    off_cols = [f"{t}_dvol_z20" for t in RISK_OFF if f"{t}_dvol_z20" in result.columns]
    if on_cols and off_cols:
        ratio = result[on_cols].mean(axis=1) - result[off_cols].mean(axis=1)
        m, s = ratio.rolling(60).mean(), ratio.rolling(60).std()
        result["rotation_z60"] = (ratio - m) / s.replace(0, float("nan"))

    result.to_parquet(fname, engine="pyarrow", compression="snappy")
    first = result.dropna(how="all").index[0].date()
    last  = result.dropna(how="all").index[-1].date()
    print(f"  OK    flow_proxies     {len(result)} rows  [{first} → {last}]  ({len(result.columns)} cols)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("1/2  FRED macro indicators")
    print("=" * 60)
    if not FRED_KEY:
        print("  FRED_API_KEY not set — get a free key at:")
        print("  https://fred.stlouisfed.org/docs/api/api_key.html")
        print("  Then add FRED_API_KEY=your_key to .env")
    else:
        for series_id, label in FRED_SERIES.items():
            fetch_fred(series_id, label)
            time.sleep(0.1)

    print()
    print("=" * 60)
    print("2/2  Flow proxies from OHLCV (dollar volume z-scores)")
    print("=" * 60)
    build_flow_proxies()

    print("\nDone.")


if __name__ == "__main__":
    main()
