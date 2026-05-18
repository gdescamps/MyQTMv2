"""
Diagnostic: train the walk-forward model WITH etf_id_code added, and write the
OOS predictions to data/oos_predictions.parquet so backtest.py can be run on it.

The production oos_predictions.parquet should be backed up before running this.

Usage:  python feature_diag_etfid_train.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import train
from train import run_walk_forward, _try_gpu, DATA


def main():
    panel = pd.read_parquet(DATA / "features.parquet").reset_index()
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.set_index(["date", "etf_id"])
    panel["etf_id_code"] = pd.Categorical(
        panel.index.get_level_values("etf_id")).codes.astype(np.int32)

    device = _try_gpu()
    train.FEATURE_COLS = list(train.FEATURE_COLS) + ["etf_id_code"]
    print(f"Device: {device.upper()}  |  features: {len(train.FEATURE_COLS)} (incl. etf_id_code)")

    oos = run_walk_forward(panel, device, dual_model=True, save_models=False, verbose=True)
    oos.to_parquet(DATA / "oos_predictions.parquet")
    print(f"\nSaved → data/oos_predictions.parquet  ({len(oos)} rows, etf_id model)")


if __name__ == "__main__":
    main()
