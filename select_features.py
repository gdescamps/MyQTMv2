"""
Feature selection by importance stability across 3 interlaced periods.

1. Split full panel into 3 interlaced periods (blocks of 21 trading days)
2. Apply 5-day embargo between blocks
3. Train one XGBoost model per period
4. Compute feature importance for each model
5. Keep features with high stability: mean_importance / std_importance

Output:
  outputs/feature_selection.csv   — all features ranked by stability
  outputs/selected_features.json  — list of selected feature names
"""

import json
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from train import XGB_PARAMS, _try_gpu, LABEL_COL, _zscore_per_date, BLOCK_ROWS, EMBARGO_ROWS

DATA    = Path(__file__).parent / "data"
OUTPUTS = Path(__file__).parent / "outputs"
OUTPUTS.mkdir(exist_ok=True)

# Interlaced 3-period split using train.py block size
BLOCK_SIZE = BLOCK_ROWS  # 21 trading days (~1 month)

# Feature selection hyperparams
MEAN_STD_POWER = 1.7
TOP_FEATURES = 150


def get_all_feature_cols(panel: pd.DataFrame) -> list[str]:
    """Get all numeric columns that could be features (exclude labels, metadata)."""
    exclude = {"label", "ret_5d_fwd", "ret_10d_fwd", "ret_15d_fwd", "ret_20d_fwd", "ret_90d_fwd",
               "section", "etf_id", "mom_accel_5v20", "mom_accel_20v60"}
    cols = []
    for c in panel.columns:
        if c in exclude:
            continue
        if panel[c].dtype in [np.float64, np.float32, np.int64, np.int32, float, int]:
            if panel[c].notna().mean() > 0.10:
                cols.append(c)
    return cols


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
    print(f"Panel: {len(panel)} rows, {len(feature_cols)} candidate features", flush=True)

    device = _try_gpu()
    print(f"Device: {device.upper()}", flush=True)

    dates = panel.index.get_level_values("date").unique().sort_values()
    n = len(dates)
    date_to_pos = {d: i for i, d in enumerate(dates)}
    row_dates = panel.index.get_level_values("date")
    row_pos = pd.Series([date_to_pos[d] for d in row_dates], index=panel.index)

    # 3-way interlaced split with embargo
    block_idx = row_pos // BLOCK_SIZE
    block_id = block_idx % 3
    # Embargo: exclude rows near block boundaries
    pos_in_block = row_pos % BLOCK_SIZE
    not_embargoed = (pos_in_block >= EMBARGO_ROWS) & (pos_in_block < BLOCK_SIZE - EMBARGO_ROWS)

    X_all = panel[feature_cols].astype(np.float32)
    y_all = panel[LABEL_COL].astype(np.float32)
    y_z = _zscore_per_date(y_all).astype(np.float32)

    params = {**XGB_PARAMS, "device": device}

    importances = {}
    for period in range(3):
        train_mask = (block_id != period) & y_all.notna() & not_embargoed
        val_mask = (block_id == period) & y_all.notna() & not_embargoed

        tr_idx = panel.index[train_mask]
        val_idx = panel.index[val_mask]

        print(f"\nPeriod {period}: train={len(tr_idx)} rows, val={len(val_idx)} rows", flush=True)

        model = xgb.XGBRegressor(**params)
        model.fit(
            X_all.loc[tr_idx].values, y_z.loc[tr_idx].values,
            eval_set=[(X_all.loc[val_idx].values, y_z.loc[val_idx].values)],
            verbose=False,
        )
        print(f"  best_iter={model.best_iteration}", flush=True)

        imp = model.feature_importances_
        importances[f"period_{period}"] = imp

        # Top 10 for this period
        top_idx = np.argsort(imp)[::-1][:10]
        print(f"  Top 10: {[feature_cols[i] for i in top_idx]}")

    # Stability analysis: mean / std^power across periods
    imp_df = pd.DataFrame(importances, index=feature_cols)
    imp_df["mean"] = imp_df.mean(axis=1)
    imp_df["std"] = imp_df.std(axis=1)
    imp_df["stability"] = imp_df["mean"] / (imp_df["std"].replace(0, np.nan) ** MEAN_STD_POWER)
    imp_df = imp_df.sort_values("stability", ascending=False)

    # Save full results
    imp_df.to_csv(OUTPUTS / "feature_selection.csv")

    # Select top N features (drop NaN/zero scores first)
    imp_df = imp_df[imp_df["stability"].notna() & (imp_df["mean"] > 0)]
    selected = imp_df.index[:TOP_FEATURES].tolist()

    print(f"\n{'='*60}")
    print(f"  Feature selection: {len(selected)} / {len(feature_cols)} features kept")
    print(f"  (top {TOP_FEATURES} by mean/std^{MEAN_STD_POWER})")
    print(f"{'='*60}")

    print(f"\n  Selected features (by stability):")
    for feat in selected:
        row = imp_df.loc[feat]
        print(f"    {feat:<35s}  mean={row['mean']:.4f}  std={row['std']:.4f}  "
              f"stab={row['stability']:.1f}")

    print(f"\n  Rejected features (top 10 by mean importance but low stability):")
    rejected = imp_df[~imp_df.index.isin(selected)].sort_values("mean", ascending=False).head(10)
    for feat, row in rejected.iterrows():
        print(f"    {feat:<35s}  mean={row['mean']:.4f}  std={row['std']:.4f}  "
              f"stab={row['stability']:.1f}")

    with open(OUTPUTS / "selected_features.json", "w") as f:
        json.dump(selected, f, indent=2)

    print(f"\nSaved: feature_selection.csv")
    print(f"Saved: selected_features.json ({len(selected)} features)")


if __name__ == "__main__":
    main()
