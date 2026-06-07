"""
Crisis-Avoidance Strategy — Walk-Forward XGBoost with continuous allocation.
Supports QQQ (Nasdaq-100) and SPY (S&P 500).

Usage: python src/risk_off_strategy/run.py [QQQ|SPY|ALL]
       ALL runs both QQQ and SPY + comparison chart
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.risk_off_strategy.data import load_data, build_features, build_realtime_target
from src.risk_off_strategy.backtest import (walk_forward, plot_results, plot_recent,
                                            plot_comparison, simulate_with_fees)

# Tickers with leveraged ETFs available → x1, x1.5, x2
# Others → x1 only (no leveraged ETF)
LEVERAGED_TICKERS = {"QQQ", "SPY"}

def get_leverages(ticker):
    return [1.0, 1.5, 2.0] if ticker in LEVERAGED_TICKERS else [1.0]


# ── Download fresh data ──────────────────────────────────
def refresh_data():
    """Re-download all OHLCV + macro data to get latest prices."""
    from src.download_ohlcv import TICKERS as OHLCV_TICKERS, VIX_TICKER, VIX_FILENAME, download_ticker
    from src.download_macro_data import fetch_fred, FRED_SERIES
    import pandas as pd

    DATA_DIR = ROOT / "data"
    print("Refreshing market data...")

    # OHLCV tickers + VIX
    all_downloads = [(t, t) for t in OHLCV_TICKERS] + [(VIX_TICKER, VIX_FILENAME)]
    for ticker, fname in all_downloads:
        path = DATA_DIR / f"{fname}.parquet"
        df = download_ticker(ticker)
        if df is not None and not df.empty:
            df.to_parquet(path, engine="pyarrow", compression="snappy")
            print(f"  {fname:<20s} {len(df)} rows → {df.index[-1].date()}")

    # FRED macro
    for series_id, label in FRED_SERIES.items():
        path = DATA_DIR / f"fred_{label}.parquet"
        if path.exists():
            path.unlink()
        fetch_fred(series_id, label)

    # Return last available date across key tickers
    last_dates = []
    for t in ["QQQ", "SPY"]:
        p = DATA_DIR / f"{t}.parquet"
        if p.exists():
            last_dates.append(pd.read_parquet(p).index[-1])
    end_date = max(last_dates).strftime("%Y-%m-%d") if last_dates else "2026-12-31"
    print(f"Data up to: {end_date}\n")
    return end_date


# ── Config ────────────────────────────────────────────────
START = "2000-01-01"
DD_EXIT = -0.10
DD_REENTER = -0.05
MIN_TRAIN = 504
STEP = 21
TEMPERATURE = 3.0
PROB_CASH = 0.70
PROB_FULL = 0.75
LOOKAHEAD = 6  # days of future info in label (must be < embargo=21)

END = refresh_data()

arg = sys.argv[1].upper() if len(sys.argv) > 1 else "ALL"
ALL_TICKERS = ["QQQ", "ACWI", "GLD", "BTC-USD"]
TICKERS = ALL_TICKERS if arg == "ALL" else [arg]


def run_ticker(ticker):
    prefix = ticker.lower().replace("-", "_")
    print(f"\n{'='*70}")
    print(f"=== {ticker} Crisis-Avoidance Strategy ===")
    print(f"{'='*70}\n")

    price, vix, spread, tlt = load_data(ticker, START, END)
    df = build_features(price, vix, spread, tlt, prefix=prefix)

    target, _ = build_realtime_target(price.values, DD_EXIT, DD_REENTER, lookahead=LOOKAHEAD)
    df["target"] = target
    df = df.dropna()

    feature_cols = [c for c in df.columns if c != "target"]
    X = df[feature_cols].values
    y = df["target"].values
    print(f"Features: {len(feature_cols)} cols, {len(df)} rows")

    wf_pred, wf_proba, model = walk_forward(
        X, y, feature_cols,
        min_train=MIN_TRAIN, step=STEP, temperature=TEMPERATURE,
    )

    pred_mask = wf_pred >= 0
    wf_dates = df.index[pred_mask]
    wf_prob = wf_proba[pred_mask]
    price_ret = price.pct_change().fillna(0).loc[df.index].values[pred_mask]

    OUT = ROOT / "outputs" / f"{prefix}_strategy"
    OUT.mkdir(parents=True, exist_ok=True)

    levs = get_leverages(ticker)
    target_labels = y[pred_mask]
    plot_results(wf_dates, price_ret, wf_prob, PROB_CASH, PROB_FULL,
                 save_path=str(OUT / "backtest.png"), ticker=ticker, leverages=levs,
                 oracle_labels=target_labels)
    plot_recent(wf_dates, price_ret, wf_prob, PROB_CASH, PROB_FULL,
                days=252, save_path=str(OUT / "backtest_1y.png"), ticker=ticker, leverages=levs)
    plot_recent(wf_dates, price_ret, wf_prob, PROB_CASH, PROB_FULL,
                days=21, save_path=str(OUT / "backtest_1m.png"), ticker=ticker, leverages=levs)

    return wf_dates, price_ret, wf_prob, get_leverages(ticker)


# ── Run ───────────────────────────────────────────────────
results = {}
for t in TICKERS:
    results[t] = run_ticker(t)

# ── Comparison chart (if multiple tickers) ────────────────
if len(TICKERS) > 1:
    OUT = ROOT / "outputs" / "comparison"
    OUT.mkdir(parents=True, exist_ok=True)
    plot_comparison(results, PROB_CASH, PROB_FULL,
                    save_path=str(OUT / "comparison.png"))
