"""
Feature selection by importance stability — aligned with MyQTM v1 method.

Like v1:
1. Split data into partition A (even blocks) and B (odd blocks) with embargo
2. Split each partition into 3 temporal sub-intervals
3. Train 6 models: sub1A/sub2A/sub3A (train on A_i, val on B)
                    sub1B/sub2B/sub3B (train on B_i, val on A)
4. Normalize importance per model by its max
5. Exclude features with zero importance in ANY model
6. Rank by mean / std^power, select top N

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
from train import (XGB_PARAMS, _try_gpu, LABEL_COL, _zscore_per_date,
                   BLOCK_ROWS, EMBARGO_ROWS)

DATA    = Path(__file__).parent / "data"
OUTPUTS = Path(__file__).parent / "outputs"
OUTPUTS.mkdir(exist_ok=True)

# Feature selection hyperparams
MEAN_STD_POWER = 1.3   # from search_features_depth.py
TOP_FEATURES   = 70    # search range in v1: 55-85


def get_all_feature_cols(panel: pd.DataFrame) -> list[str]:
    """Get all numeric columns that could be features (exclude labels, metadata)."""
    exclude = {"label", "ret_5d_fwd", "ret_10d_fwd", "ret_20d_fwd", "ret_90d_fwd",
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
    date_to_pos = {d: i for i, d in enumerate(dates)}
    row_dates = panel.index.get_level_values("date")
    row_pos = pd.Series([date_to_pos[d] for d in row_dates], index=panel.index)

    # Block parity + embargo (same as train.py)
    block_idx = row_pos // BLOCK_ROWS
    block_parity = block_idx % 2
    pos_in_block = row_pos % BLOCK_ROWS
    not_embargoed = (pos_in_block >= EMBARGO_ROWS) & (pos_in_block < BLOCK_ROWS - EMBARGO_ROWS)

    X_all = panel[feature_cols].astype(np.float32)
    y_all = panel[LABEL_COL].astype(np.float32)
    y_z = _zscore_per_date(y_all).astype(np.float32)

    valid = y_all.notna() & not_embargoed

    # Partition A (even blocks) and B (odd blocks)
    mask_A = (block_parity == 0) & valid
    mask_B = (block_parity == 1) & valid
    idx_A = panel.index[mask_A]
    idx_B = panel.index[mask_B]

    print(f"\nPartition A (even blocks): {len(idx_A)} rows")
    print(f"Partition B (odd blocks):  {len(idx_B)} rows")

    # Split each partition into 3 temporal sub-intervals
    def split_3(idx):
        # Sort by date position, split into 3 equal parts
        positions = row_pos.loc[idx].values
        sorted_order = np.argsort(positions)
        splits = np.array_split(sorted_order, 3)
        return [idx[s] for s in splits]

    subs_A = split_3(idx_A)
    subs_B = split_3(idx_B)

    print(f"  A sub-intervals: {[len(s) for s in subs_A]}")
    print(f"  B sub-intervals: {[len(s) for s in subs_B]}")

    # Train 6 models (like v1)
    params = {**XGB_PARAMS, "device": device}

    model_names = []
    importances = {}

    # Models 1-3: train on A sub-intervals, val on full B
    for i, sub_idx in enumerate(subs_A):
        name = f"sub{i+1}A"
        model_names.append(name)
        print(f"\n{name}: train={len(sub_idx)} rows, val={len(idx_B)} rows", flush=True)

        model = xgb.XGBRegressor(**params)
        model.fit(
            X_all.loc[sub_idx].values, y_z.loc[sub_idx].values,
            eval_set=[(X_all.loc[idx_B].values, y_z.loc[idx_B].values)],
            verbose=False,
        )
        print(f"  best_iter={model.best_iteration}", flush=True)
        importances[name] = model.feature_importances_

    # Models 4-6: train on B sub-intervals, val on full A
    for i, sub_idx in enumerate(subs_B):
        name = f"sub{i+1}B"
        model_names.append(name)
        print(f"\n{name}: train={len(sub_idx)} rows, val={len(idx_A)} rows", flush=True)

        model = xgb.XGBRegressor(**params)
        model.fit(
            X_all.loc[sub_idx].values, y_z.loc[sub_idx].values,
            eval_set=[(X_all.loc[idx_A].values, y_z.loc[idx_A].values)],
            verbose=False,
        )
        print(f"  best_iter={model.best_iteration}", flush=True)
        importances[name] = model.feature_importances_

    # Build importance DataFrame
    imp_df = pd.DataFrame(importances, index=feature_cols)

    # Normalize each column by its max (like v1)
    for col in model_names:
        col_max = imp_df[col].max()
        if col_max > 0:
            imp_df[col] = imp_df[col] / col_max

    # Filter: exclude features with zero importance in ANY model (like v1)
    nonzero_mask = (imp_df[model_names] > 0).all(axis=1)
    n_before = len(imp_df)
    imp_df_filtered = imp_df[nonzero_mask].copy()
    n_after = len(imp_df_filtered)
    print(f"\n  Zero-importance filter: {n_before} → {n_after} features "
          f"({n_before - n_after} removed)")

    # Compute mean, std, and stability score
    imp_df_filtered["mean"] = imp_df_filtered[model_names].mean(axis=1)
    imp_df_filtered["std"] = imp_df_filtered[model_names].std(axis=1)
    imp_df_filtered["mean/std"] = imp_df_filtered["mean"] / (
        imp_df_filtered["std"] ** MEAN_STD_POWER
    )
    imp_df_filtered = imp_df_filtered.sort_values("mean/std", ascending=False)

    # Save full results
    imp_df_filtered.to_csv(OUTPUTS / "feature_selection.csv")

    # Select top N features (like v1)
    selected = imp_df_filtered.index[:TOP_FEATURES].tolist()

    print(f"\n{'='*60}")
    print(f"  Feature selection: {len(selected)} / {len(feature_cols)} features kept")
    print(f"  (top {TOP_FEATURES} by mean/std^{MEAN_STD_POWER}, "
          f"after zero-importance filter)")
    print(f"{'='*60}")

    print(f"\n  Selected features (by mean/std^{MEAN_STD_POWER}):")
    for feat in selected:
        row = imp_df_filtered.loc[feat]
        print(f"    {feat:<35s}  mean={row['mean']:.4f}  std={row['std']:.4f}  "
              f"score={row['mean/std']:.2f}")

    print(f"\n  First 10 rejected features:")
    rejected = imp_df_filtered.index[TOP_FEATURES:TOP_FEATURES + 10].tolist()
    for feat in rejected:
        row = imp_df_filtered.loc[feat]
        print(f"    {feat:<35s}  mean={row['mean']:.4f}  std={row['std']:.4f}  "
              f"score={row['mean/std']:.2f}")

    with open(OUTPUTS / "selected_features.json", "w") as f:
        json.dump(selected, f, indent=2)

    print(f"\nSaved: feature_selection.csv")
    print(f"Saved: selected_features.json ({len(selected)} features)")


if __name__ == "__main__":
    main()
