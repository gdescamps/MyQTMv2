"""
QQQ Crisis-Avoidance Strategy — Walk-Forward XGBoost with continuous allocation.

Usage: python -m src.qqq_strategy.run
       python src/qqq_strategy/run.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.qqq_strategy.data import load_data, build_features, build_realtime_target
from src.qqq_strategy.backtest import walk_forward, compute_equity, plot_results

# ── Config ────────────────────────────────────────────────
START = "2006-01-01"
END = "2026-05-31"
DD_EXIT = -0.10
DD_REENTER = -0.05
MIN_TRAIN = 504
STEP = 21
TEMPERATURE = 3.0
MAX_LEVERAGE = 1.5
PROB_CASH = 0.5
PROB_FULL = 0.85

# ── Data ──────────────────────────────────────────────────
qqq, vix, spread = load_data(START, END)
df = build_features(qqq, vix, spread)

target, _ = build_realtime_target(qqq.values, DD_EXIT, DD_REENTER)
df["target"] = target
df = df.dropna()

feature_cols = [c for c in df.columns if c != "target"]
X = df[feature_cols].values
y = df["target"].values
print(f"Features: {len(feature_cols)} cols, {len(df)} rows")

# ── Walk-Forward ──────────────────────────────────────────
wf_pred, wf_proba, model = walk_forward(
    X, y, feature_cols,
    min_train=MIN_TRAIN, step=STEP, temperature=TEMPERATURE,
)

# ── Filter to prediction period ──────────────────────────
pred_mask = wf_pred >= 0
wf_dates = df.index[pred_mask]
wf_p = wf_pred[pred_mask]
wf_prob = wf_proba[pred_mask]

qqq_ret = qqq.pct_change().fillna(0).loc[df.index].values[pred_mask]

# ── Equity & Plot ─────────────────────────────────────────
bh_eq, bin_eq, cont_eq, alloc = compute_equity(
    qqq_ret, wf_p, wf_prob,
    max_leverage=MAX_LEVERAGE, prob_cash=PROB_CASH, prob_full=PROB_FULL,
)

plot_results(
    wf_dates, bh_eq, bin_eq, cont_eq, alloc, MAX_LEVERAGE,
    save_path=str(ROOT / "outputs" / "qqq_strategy" / "backtest.png"),
)
