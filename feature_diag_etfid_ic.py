"""
Diagnostic: real walk-forward IC impact of adding ETF identity (etf_id).

Runs the production walk-forward training twice — once with the normal
feature set, once with etf_id_code (ordinal-encoded) added — and compares
the true out-of-sample test IC.

Non-destructive: save_models=False, does not write oos_predictions.parquet.

Usage:  python feature_diag_etfid_ic.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import train
from train import run_walk_forward, _try_gpu, DATA


def _ic(oos: pd.DataFrame, split: str) -> float:
    sub = oos[oos["split"] == split].dropna(subset=["score", "label"])
    return sub.groupby(sub.index.get_level_values("date")).apply(
        lambda x: x["score"].corr(x["label"])).mean()


def main():
    panel = pd.read_parquet(DATA / "features.parquet").reset_index()
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.set_index(["date", "etf_id"])

    # Ordinal-encode ETF identity
    panel["etf_id_code"] = pd.Categorical(
        panel.index.get_level_values("etf_id")).codes.astype(np.int32)

    device = _try_gpu()
    base_cols = list(train.FEATURE_COLS)
    print(f"Device: {device.upper()}  |  baseline features: {len(base_cols)}")

    # --- Run A: baseline (no etf_id) ---
    print("\n=== Run A — baseline (sans etf_id) ===", flush=True)
    train.FEATURE_COLS = base_cols
    oos_a = run_walk_forward(panel, device, dual_model=True,
                             save_models=False, verbose=False)
    a_val, a_test = _ic(oos_a, "val"), _ic(oos_a, "test")
    print(f"  val IC={a_val:+.4f}   test IC={a_test:+.4f}")

    # --- Run B: with etf_id_code ---
    print("\n=== Run B — avec etf_id_code ===", flush=True)
    train.FEATURE_COLS = base_cols + ["etf_id_code"]
    oos_b = run_walk_forward(panel, device, dual_model=True,
                             save_models=False, verbose=False)
    b_val, b_test = _ic(oos_b, "val"), _ic(oos_b, "test")
    print(f"  val IC={b_val:+.4f}   test IC={b_test:+.4f}")

    train.FEATURE_COLS = base_cols  # restore

    print(f"\n{'='*60}")
    print(f"  {'':22s}{'val IC':>12s}{'test IC':>12s}")
    print(f"  {'A baseline':22s}{a_val:>+12.4f}{a_test:>+12.4f}")
    print(f"  {'B + etf_id':22s}{b_val:>+12.4f}{b_test:>+12.4f}")
    print(f"  {'Δ (B − A)':22s}{b_val-a_val:>+12.4f}{b_test-a_test:>+12.4f}")
    print(f"{'='*60}")
    print(f"  val/test gap A: {a_val-a_test:+.4f}   B: {b_val-b_test:+.4f}")
    verdict = ("etf_id améliore le test IC OOS"
               if b_test > a_test + 0.003 else
               "etf_id n'améliore pas le test IC OOS (pas de gain généralisable)")
    print(f"  → {verdict}")


if __name__ == "__main__":
    main()
