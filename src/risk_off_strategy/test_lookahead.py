"""Compare lookahead=0 vs lookahead=6 across all tickers (x1, close-to-close)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from src.risk_off_strategy.data import load_data, build_features, build_realtime_target
from src.risk_off_strategy.backtest import walk_forward, simulate, compute_metrics

START, END = "2000-01-01", "2026-05-31"
DD_EXIT, DD_REENTER = -0.10, -0.05
MIN_TRAIN, STEP, TEMPERATURE = 504, 21, 3.0
PROB_CASH, PROB_FULL = 0.5, 0.85
LOOKAHEADS = [0, 6]
ALL_TICKERS = ["QQQ", "ACWI", "GLD", "BTC-USD"]

all_rows = []
for ticker in ALL_TICKERS:
    prefix = ticker.lower().replace("-", "_")
    price, vix, spread, tlt = load_data(ticker, START, END)
    df = build_features(price, vix, spread, tlt, prefix=prefix)
    price_ret_full = price.pct_change().fillna(0)

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

        eq, _ = simulate(ret, wf_prob, PROB_CASH, PROB_FULL)
        cagr, dd = compute_metrics(eq, years)
        all_rows.append({"ticker": ticker, "la": la, "bh_cagr": bh_cagr,
                         "bh_dd": bh_dd, "cagr": cagr, "dd": dd})

# Summary table
print(f"\n\n{'='*70}")
print(f"  LOOKAHEAD COMPARISON — ALL TICKERS (LA=0 vs LA=6)")
print(f"{'='*70}")
print(f"{'Ticker':<10s} {'LA':>3s}  {'B&H CAGR':>9s}  {'XGB CAGR':>9s} {'XGB DD':>8s}")
print(f"{'-'*70}")
for r in all_rows:
    print(f"{r['ticker']:<10s} {r['la']:3d}  {r['bh_cagr']*100:8.1f}%  "
          f"{r['cagr']*100:8.1f}% {r['dd']*100:7.1f}%")
