"""
Crisis-Avoidance Strategy — Walk-Forward XGBoost with continuous allocation.
Supports QQQ (Nasdaq-100) and SPY (S&P 500).

Usage: python src/risk_off_strategy/run.py [QQQ|SPY|ALL]
       ALL runs both QQQ and SPY + comparison chart
"""

import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.risk_off_strategy.data import load_data, build_features, build_realtime_target
from src.risk_off_strategy.backtest import (walk_forward, plot_results, plot_recent,
                                            plot_comparison, simulate_with_fees,
                                            plot_projection)

# Tickers with leveraged ETFs available → x1, x1.5, x2
# Others → x1 only (no leveraged ETF)
LEVERAGED_TICKERS = {"QQQ", "SPY"}

def get_leverages(ticker):
    return [1.0, 1.5, 1.75, 2.0] if ticker in LEVERAGED_TICKERS else [1.0]


# ── Download fresh data ──────────────────────────────────
US_CLOSE_HOUR = 22  # US market close in Paris time (22h00 = 16h00 ET)


def refresh_data():
    """Re-download risk-off OHLCV + macro data to get latest prices.

    If run before US close (22h Paris), excludes today's incomplete bar.
    """
    from src.download_ohlcv import download_risk_off, RISK_OFF_TICKERS
    from src.download_macro_data import fetch_fred, FRED_SERIES
    import pandas as pd
    from datetime import datetime, timezone, timedelta

    DATA_DIR = ROOT / "data"

    # OHLCV: risk-off tickers + VIX
    download_risk_off(force=True)

    # FRED macro
    for series_id, label in FRED_SERIES.items():
        fetch_fred(series_id, label, force=True)

    # Check if US market is still open
    paris_tz = timezone(timedelta(hours=2))  # CEST (summer)
    now_paris = datetime.now(paris_tz)
    today = pd.Timestamp(now_paris.date())
    before_close = now_paris.hour < US_CLOSE_HOUR and today.weekday() < 5

    # Return last available date, excluding today if before US close
    last_dates = []
    for t in RISK_OFF_TICKERS:
        p = DATA_DIR / f"{t}.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            if before_close:
                df = df[df.index < today]
            if len(df) > 0:
                last_dates.append(df.index[-1])

    end_date = max(last_dates).strftime("%Y-%m-%d") if last_dates else "2026-12-31"
    if before_close:
        print(f"Before US close ({now_paris.strftime('%H:%M')} Paris) — excluding today")
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

NO_REFRESH = "--no-refresh" in sys.argv
if NO_REFRESH:
    sys.argv.remove("--no-refresh")
    END = "2026-06-16"
    print(f"Skipping data refresh, using END={END}\n")
else:
    END = refresh_data()

arg = sys.argv[1].upper() if len(sys.argv) > 1 else "ALL"
ALL_TICKERS = ["QQQ", "SPY", "ACWI"]
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

    # Load PUST (Amundi PEA Nasdaq-100) for PEA execution comparison
    pust_path = ROOT / "data" / "PUST.parquet"
    panx_ret_arr = None
    if pust_path.exists():
        pust_open = pd.read_parquet(pust_path)["open"]
        pust_open_ret = pust_open.pct_change().fillna(0)
        pust_ret_aligned = pust_open_ret.reindex(wf_dates).values
        # NaN for dates before PUST exists → set to 0 (no position)
        panx_ret_arr = np.where(np.isnan(pust_ret_aligned), 0.0, pust_ret_aligned)

    OUT = ROOT / "outputs" / f"{prefix}_strategy"
    OUT.mkdir(parents=True, exist_ok=True)

    levs = get_leverages(ticker)
    target_labels = y[pred_mask]
    bt_results = plot_results(wf_dates, price_ret, wf_prob, PROB_CASH, PROB_FULL,
                              save_path=str(OUT / "backtest.png"), ticker=ticker, leverages=levs,
                              oracle_labels=target_labels, panx_ret=panx_ret_arr)
    plot_projection(bt_results, levs, save_path=str(OUT / "projection.png"))
    plot_recent(wf_dates, price_ret, wf_prob, PROB_CASH, PROB_FULL,
                days=252, save_path=str(OUT / "backtest_1y.png"), ticker=ticker, leverages=levs)
    plot_recent(wf_dates, price_ret, wf_prob, PROB_CASH, PROB_FULL,
                days=21, save_path=str(OUT / "backtest_1m.png"), ticker=ticker, leverages=levs)

    # Save latest signal to JSON for morning execution
    last_prob = wf_prob[-1]
    alloc = float(np.clip((last_prob - PROB_CASH) / (PROB_FULL - PROB_CASH), 0, 1))
    signal = {
        "ticker": ticker,
        "date": str(wf_dates[-1].date()),
        "probability": float(last_prob),
        "allocation": alloc,
        "prob_cash": PROB_CASH,
        "prob_full": PROB_FULL,
        "timestamp": pd.Timestamp.now().isoformat(),
    }
    signal_path = OUT / "signal.json"
    with open(signal_path, "w") as f:
        json.dump(signal, f, indent=2)
    print(f"Signal saved: {signal_path} (alloc={alloc*100:.0f}%, prob={last_prob:.4f})")

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
