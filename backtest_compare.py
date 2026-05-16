"""
Backtest comparison: run backtest on multiple models from search results.

Trains walk-forward for each model config, runs backtest, reports comparison table.
"""

import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from train import (
    _try_gpu, run_walk_forward,
)
from select_features import get_all_feature_cols
from search_features_xgb import select_features_by_stability, compute_metrics, fmt_time
import train as train_mod

DATA    = Path(__file__).parent / "data"
OUTPUTS = Path(__file__).parent / "myfiles"

# Models to compare: (label, power, cap, depth)
MODELS = [
    # Best IC (current)
    ("best_ic",        1.60, 150, 6),
    # Best stable models (stability > 0.05, sorted by composite score)
    ("stable_1",       1.70, 150, 8),
    ("stable_2",       1.20, 125, 7),
    ("stable_3",       1.50, 100, 6),
    ("stable_4",       1.20,  50, 7),
]


def run_single_backtest(oos: pd.DataFrame) -> dict:
    """Save oos predictions and run backtest, return metrics."""
    oos.to_parquet(DATA / "oos_predictions.parquet")

    # Import and run backtest inline (capture results)
    import importlib
    import backtest as bt_mod
    importlib.reload(bt_mod)

    # Redirect stdout to capture
    from io import StringIO
    import contextlib
    output = StringIO()
    with contextlib.redirect_stdout(output):
        bt_mod.main()

    text = output.getvalue()
    # Parse results from output
    metrics = {}
    for line in text.split("\n"):
        if "Total return:" in line:
            metrics["return_pct"] = float(line.split(":")[1].strip().replace("%", ""))
        elif "Sharpe:" in line:
            metrics["sharpe"] = float(line.split(":")[1].strip())
        elif "Max drawdown:" in line:
            metrics["max_dd"] = float(line.split(":")[1].strip().replace("%", ""))
        elif "Ann. return:" in line:
            metrics["ann_return"] = float(line.split(":")[1].strip().replace("%", ""))
    return metrics


def main():
    feat_path = DATA / "features.parquet"
    if not feat_path.exists():
        sys.exit("ERROR: data/features.parquet not found")

    print("Loading features...", flush=True)
    panel = pd.read_parquet(feat_path)
    panel = panel.reset_index()
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.set_index(["date", "etf_id"])

    feature_cols = get_all_feature_cols(panel)
    device = _try_gpu()
    print(f"Device: {device.upper()}", flush=True)
    print(f"Models to compare: {len(MODELS)}\n")

    # Feature selection cache
    feat_cache = {}
    results = []

    for i, (label, power, cap, depth) in enumerate(MODELS):
        print(f"[{i+1}/{len(MODELS)}] {label}: power={power:.2f} cap={cap} depth={depth}", flush=True)
        t0 = time.time()

        # Feature selection
        if power not in feat_cache:
            feat_cache[power] = select_features_by_stability(
                panel, feature_cols, device, power, cap=max(cap, 200)
            )
        selected = feat_cache[power][:cap]
        train_mod.FEATURE_COLS = selected

        # Train walk-forward
        oos = run_walk_forward(
            panel, device,
            xgb_params={"max_depth": depth},
            verbose=False,
            save_models=False,
            dual_model=False,
        )
        ic_metrics = compute_metrics(oos)

        # Run backtest
        bt_metrics = run_single_backtest(oos)

        elapsed = time.time() - t0
        row = {
            "label": label, "power": power, "cap": cap, "depth": depth,
            "n_features": len(selected),
            **ic_metrics, **bt_metrics,
            "time_s": elapsed,
        }
        results.append(row)
        print(f"  → return={bt_metrics.get('return_pct', '?'):.1f}%  "
              f"sharpe={bt_metrics.get('sharpe', '?'):.3f}  "
              f"dd={bt_metrics.get('max_dd', '?'):.1f}%  "
              f"test_ic={ic_metrics['test_ic']:+.4f}  "
              f"stab={ic_metrics['stability']:+.3f}  "
              f"({fmt_time(elapsed)})", flush=True)

    # Summary table
    df = pd.DataFrame(results)
    print(f"\n{'='*90}")
    print(f"  COMPARISON TABLE")
    print(f"{'='*90}")
    print(f"{'label':<12} {'power':>5} {'cap':>4} {'d':>2} {'return%':>8} {'sharpe':>7} {'maxDD%':>7} {'test_IC':>8} {'stab':>6}")
    print(f"{'-'*70}")
    for _, r in df.iterrows():
        print(f"{r['label']:<12} {r['power']:5.2f} {int(r['cap']):4d} {int(r['depth']):2d} "
              f"{r.get('return_pct', 0):>+7.1f}% {r.get('sharpe', 0):7.3f} "
              f"{r.get('max_dd', 0):>+6.1f}% {r['test_ic']:>+.4f} {r['stability']:>+.3f}")
    print(f"{'='*90}")

    df.to_csv(OUTPUTS / "backtest_compare.csv", index=False)
    print(f"\nSaved → myfiles/backtest_compare.csv")


if __name__ == "__main__":
    main()
