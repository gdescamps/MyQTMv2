"""
Diagnostic: add ETF identity (etf_id) and category (section), ordinal-encoded,
to the candidate features and run the SAME 3-period interlaced stability filter
as select_features.py — to see where they rank.

Non-destructive: does not touch outputs/selected_features.json. Writes the full
ranking to myfiles/feature_diag_etfid.csv.

Usage:  python feature_diag_etfid.py
"""

import sys
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from train import XGB_PARAMS, _try_gpu, LABEL_COL, _zscore_per_date, BLOCK_ROWS, EMBARGO_ROWS
from select_features import get_all_feature_cols, MEAN_STD_POWER

DATA = Path(__file__).parent / "data"
MYFILES = Path(__file__).parent / "myfiles"
N_PERIODS = 3
DIAG = ("etf_id_code", "section_code")


def main():
    panel = pd.read_parquet(DATA / "features.parquet").reset_index()
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.set_index(["date", "etf_id"])

    # Ordinal-encode ETF identity and category as int columns
    etf_ids = panel.index.get_level_values("etf_id")
    panel["etf_id_code"] = pd.Categorical(etf_ids).codes.astype(np.int32)
    n_ids = panel["etf_id_code"].nunique()
    if "section" in panel.columns:
        panel["section_code"] = pd.Categorical(panel["section"]).codes.astype(np.int32)
        n_sec = panel["section_code"].nunique()
    else:
        n_sec = 0
    print(f"Encoded: {n_ids} ETF ids, {n_sec} sections")

    # get_all_feature_cols keeps numeric cols → it picks up etf_id_code /
    # section_code automatically (the string etf_id/section stay excluded)
    feature_cols = get_all_feature_cols(panel)
    for c in DIAG:
        assert c in feature_cols, f"{c} not picked up as candidate"
    print(f"Panel: {len(panel)} rows, {len(feature_cols)} candidate features", flush=True)

    device = _try_gpu()
    print(f"Device: {device.upper()}", flush=True)

    dates = panel.index.get_level_values("date").unique().sort_values()
    date_to_pos = {d: i for i, d in enumerate(dates)}
    row_pos = pd.Series([date_to_pos[d] for d in panel.index.get_level_values("date")],
                        index=panel.index)
    block_id = (row_pos // BLOCK_ROWS) % N_PERIODS
    pos_in_block = row_pos % BLOCK_ROWS
    not_emb = (pos_in_block >= EMBARGO_ROWS) & (pos_in_block < BLOCK_ROWS - EMBARGO_ROWS)

    X_all = panel[feature_cols].astype(np.float32)
    y_all = panel[LABEL_COL].astype(np.float32)
    y_z = _zscore_per_date(y_all).astype(np.float32)
    params = {**XGB_PARAMS, "device": device}

    importances = {}
    for period in range(N_PERIODS):
        tr = (block_id != period) & y_all.notna() & not_emb
        va = (block_id == period) & y_all.notna() & not_emb
        tr_idx, va_idx = panel.index[tr], panel.index[va]
        print(f"Period {period}: train={len(tr_idx)}  val={len(va_idx)}", flush=True)
        model = xgb.XGBRegressor(**params)
        model.fit(X_all.loc[tr_idx].values, y_z.loc[tr_idx].values,
                  eval_set=[(X_all.loc[va_idx].values, y_z.loc[va_idx].values)],
                  verbose=False)
        importances[f"period_{period}"] = model.feature_importances_

    imp = pd.DataFrame(importances, index=feature_cols)
    imp["mean"] = imp.mean(axis=1)
    imp["std"] = imp.std(axis=1)
    imp["stability"] = imp["mean"] / (imp["std"].replace(0, np.nan) ** MEAN_STD_POWER)
    imp = imp.sort_values("stability", ascending=False)
    imp["rank_stability"] = range(1, len(imp) + 1)
    imp["rank_mean"] = imp["mean"].rank(ascending=False).astype(int)
    imp.to_csv(MYFILES / "feature_diag_etfid.csv")

    n = len(imp)
    print(f"\n{'='*64}")
    print(f"  {n} candidate features — TOP_FEATURES cutoff = 150")
    print(f"{'='*64}")
    for c in DIAG:
        r = imp.loc[c]
        kept = "KEPT" if r["rank_stability"] <= 150 else "rejected"
        print(f"  {c:<14s}  stability rank {int(r['rank_stability']):>3d}/{n} ({kept})  "
              f"| raw-importance rank {int(r['rank_mean']):>3d}/{n}")
        print(f"  {'':<14s}  mean={r['mean']:.4f}  std={r['std']:.4f}  stab={r['stability']:.2f}")
    print(f"{'='*64}")

    print("\nTop 12 by raw mean importance:")
    for f, r in imp.sort_values("mean", ascending=False).head(12).iterrows():
        mark = "   <== DIAG" if f in DIAG else ""
        print(f"  {int(r['rank_mean']):>3d}. {f:<30s} mean={r['mean']:.4f}{mark}")

    print(f"\nSaved → myfiles/feature_diag_etfid.csv")


if __name__ == "__main__":
    main()
