"""Compare lookahead=0 vs lookahead=6 across all tickers."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from src.risk_off_strategy.data import load_data, build_features, build_realtime_target
from src.risk_off_strategy.backtest import walk_forward, simulate_with_fees, compute_metrics

START, END = "2000-01-01", "2026-05-31"
DD_EXIT, DD_REENTER = -0.10, -0.05
MIN_TRAIN, STEP, TEMPERATURE = 504, 21, 3.0
PROB_CASH, PROB_FULL = 0.5, 0.85
LOOKAHEADS = [0, 6]
ALL_TICKERS = ["QQQ", "ACWI", "GLD", "BTC-USD"]
LEVERAGED_TICKERS = {"QQQ", "SPY"}

all_rows = []
for ticker in ALL_TICKERS:
    prefix = ticker.lower().replace("-", "_")
    price, vix, spread, tlt = load_data(ticker, START, END)
    df = build_features(price, vix, spread, tlt, prefix=prefix)
    price_ret_full = price.pct_change().fillna(0)
    levs = [1.0, 1.5, 2.0] if ticker in LEVERAGED_TICKERS else [1.0]

    for la in LOOKAHEADS:
        print(f"\n{'='*60}")
        print(f"  {ticker} — LOOKAHEAD = {la}")
        print(f"{'='*60}")

        target, _ = build_realtime_target(price.values, DD_EXIT, DD_REENTER, lookahead=la)
        df_run = df.copy()
        df_run["target"] = target
        df_run = df_run.dropna()

        feature_cols = [c for c in df_run.columns if c != "target"]
        X = df_run[feature_cols].values
        y = df_run["target"].values

        wf_pred, wf_proba, _ = walk_forward(
            X, y, feature_cols, min_train=MIN_TRAIN, step=STEP, temperature=TEMPERATURE,
        )

        pred_mask = wf_pred >= 0
        wf_dates = df_run.index[pred_mask]
        wf_prob = wf_proba[pred_mask]
        ret = price_ret_full.loc[df_run.index].values[pred_mask]
        years = (wf_dates[-1] - wf_dates[0]).days / 365.25

        bh_eq = np.cumprod(1 + ret)
        bh_cagr, bh_dd = compute_metrics(bh_eq, years)

        row = {"ticker": ticker, "la": la, "bh_cagr": bh_cagr, "bh_dd": bh_dd}
        for lev in levs:
            eq, _, _ = simulate_with_fees(ret, wf_prob, lev, PROB_CASH, PROB_FULL)
            cagr, dd = compute_metrics(eq, years)
            row[f"x{lev:.1f}_cagr"] = cagr
            row[f"x{lev:.1f}_dd"] = dd
        all_rows.append(row)

# Summary table
print(f"\n\n{'='*95}")
print(f"  LOOKAHEAD COMPARISON — ALL TICKERS (LA=0 vs LA=6)")
print(f"{'='*95}")
print(f"{'Ticker':<10s} {'LA':>3s}  {'B&H CAGR':>9s}  {'x1 CAGR':>8s} {'x1 DD':>7s}  "
      f"{'x1.5 CAGR':>9s} {'x1.5 DD':>8s}  {'x2 CAGR':>8s} {'x2 DD':>7s}")
print(f"{'-'*95}")
for r in all_rows:
    x1c = r.get('x1.0_cagr', 0) * 100
    x1d = r.get('x1.0_dd', 0) * 100
    x15c = r.get('x1.5_cagr', 0) * 100
    x15d = r.get('x1.5_dd', 0) * 100
    x2c = r.get('x2.0_cagr', 0) * 100
    x2d = r.get('x2.0_dd', 0) * 100
    has_lev = 'x1.5_cagr' in r
    if has_lev:
        print(f"{r['ticker']:<10s} {r['la']:3d}  {r['bh_cagr']*100:8.1f}%  "
              f"{x1c:7.1f}% {x1d:6.1f}%  {x15c:8.1f}% {x15d:7.1f}%  {x2c:7.1f}% {x2d:6.1f}%")
    else:
        print(f"{r['ticker']:<10s} {r['la']:3d}  {r['bh_cagr']*100:8.1f}%  "
              f"{x1c:7.1f}% {x1d:6.1f}%")
