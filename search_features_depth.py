"""
Nested grid search: feature power × top N × XGBoost depth.

Outer loop: v1-aligned feature selection (6 models, normalize, zero filter)
  - mean / std^power, power in [1.2 .. 1.7]
  - top N in [100, 125, 150, 175]
  - Full history 2005-2026, embargo 5d, blocks 21d

Inner loop: train model A only on last 30 test steps
  - depth in [3, 4, 5, 6, 7]
  - evaluate: mean test IC, stability (val/test corr), gap

Output: myfiles/search_results.csv
"""

import sys
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from train import (XGB_PARAMS, _try_gpu, LABEL_COL, _zscore_per_date,
                   BLOCK_ROWS, EMBARGO_ROWS, TEST_WINDOW, STEP, MIN_TRAIN_ROWS,
                   _daily_ic)
from select_features import get_all_feature_cols

DATA    = Path(__file__).parent / "data"
OUTPUTS = Path(__file__).parent / "outputs"
MYFILES = Path(__file__).parent / "myfiles"
MYFILES.mkdir(exist_ok=True)

# Search grid
POWER_RANGE = np.arange(1.2, 1.75, 0.1)  # [1.2, 1.3, 1.4, 1.5, 1.6, 1.7]
TOP_N_RANGE = [100, 125, 150, 175]
DEPTH_RANGE = [3, 4, 5, 6, 7]
N_LAST_STEPS = 30


def select_features_v1(panel, feature_cols, device, power, top_n, row_pos):
    """V1-aligned: 6 models, normalize by max, zero filter, top N by mean/std^power."""
    block_idx = row_pos // BLOCK_ROWS
    block_parity = block_idx % 2
    pos_in_block = row_pos % BLOCK_ROWS
    not_embargoed = (pos_in_block >= EMBARGO_ROWS) & (pos_in_block < BLOCK_ROWS - EMBARGO_ROWS)

    X_all = panel[feature_cols].astype(np.float32)
    y_all = panel[LABEL_COL].astype(np.float32)
    y_z = _zscore_per_date(y_all).astype(np.float32)

    valid = y_all.notna() & not_embargoed
    mask_A = (block_parity == 0) & valid
    mask_B = (block_parity == 1) & valid
    idx_A = panel.index[mask_A]
    idx_B = panel.index[mask_B]

    def split_3(idx):
        positions = row_pos.loc[idx].values
        sorted_order = np.argsort(positions)
        splits = np.array_split(sorted_order, 3)
        return [idx[s] for s in splits]

    subs_A = split_3(idx_A)
    subs_B = split_3(idx_B)

    params = {**XGB_PARAMS, "device": device}
    importances = {}
    model_names = []

    for i, sub_idx in enumerate(subs_A):
        name = f"sub{i+1}A"
        model_names.append(name)
        model = xgb.XGBRegressor(**params)
        model.fit(X_all.loc[sub_idx].values, y_z.loc[sub_idx].values,
                  eval_set=[(X_all.loc[idx_B].values, y_z.loc[idx_B].values)],
                  verbose=False)
        importances[name] = model.feature_importances_

    for i, sub_idx in enumerate(subs_B):
        name = f"sub{i+1}B"
        model_names.append(name)
        model = xgb.XGBRegressor(**params)
        model.fit(X_all.loc[sub_idx].values, y_z.loc[sub_idx].values,
                  eval_set=[(X_all.loc[idx_A].values, y_z.loc[idx_A].values)],
                  verbose=False)
        importances[name] = model.feature_importances_

    imp_df = pd.DataFrame(importances, index=feature_cols)

    for col in model_names:
        col_max = imp_df[col].max()
        if col_max > 0:
            imp_df[col] = imp_df[col] / col_max

    # Zero filter
    imp_df = imp_df[(imp_df[model_names] > 0).all(axis=1)]

    if len(imp_df) < 5:
        return []

    imp_df["mean"] = imp_df[model_names].mean(axis=1)
    imp_df["std"] = imp_df[model_names].std(axis=1)
    imp_df["score"] = imp_df["mean"] / (imp_df["std"] ** power)
    imp_df = imp_df.sort_values("score", ascending=False)

    actual_top = min(top_n, len(imp_df))
    return imp_df.index[:actual_top].tolist()


def run_last_n_steps(panel, feature_cols, device, depth, n_steps, row_pos):
    """Train model A only on last N test steps, return IC metrics."""
    params = {**XGB_PARAMS, "device": device, "max_depth": depth}

    dates = panel.index.get_level_values("date").unique().sort_values()
    n = len(dates)

    X_all = panel[feature_cols].astype(np.float32)
    y_all = panel[LABEL_COL].astype(np.float32)
    y_train = _zscore_per_date(y_all).astype(np.float32)

    all_steps = []
    train_end_pos = MIN_TRAIN_ROWS
    while train_end_pos + TEST_WINDOW <= n:
        all_steps.append(train_end_pos)
        train_end_pos += STEP

    eval_steps = all_steps[-n_steps:]

    val_ics = []
    test_ics = []

    for train_end_pos in eval_steps:
        test_end_pos = min(train_end_pos + TEST_WINDOW, n)

        in_train = row_pos < train_end_pos
        block_idx = row_pos // BLOCK_ROWS
        block_parity = block_idx % 2
        pos_in_block = row_pos % BLOCK_ROWS
        in_embargo = (pos_in_block < EMBARGO_ROWS) | (pos_in_block >= BLOCK_ROWS - EMBARGO_ROWS)
        not_embargoed = ~in_embargo

        test_mask = (row_pos >= train_end_pos) & (row_pos < test_end_pos)
        test_idx = panel.index[test_mask]

        train_A_mask = in_train & (block_parity == 0) & not_embargoed
        val_A_mask = in_train & (block_parity == 1) & not_embargoed
        tr_A_idx = panel.index[train_A_mask & y_all.notna()]
        val_A_idx = panel.index[val_A_mask & y_all.notna()]

        if len(tr_A_idx) < 100 or len(val_A_idx) < 10 or len(test_idx) == 0:
            continue

        model_A = xgb.XGBRegressor(**params)
        model_A.fit(X_all.loc[tr_A_idx].values, y_train.loc[tr_A_idx].values,
                    eval_set=[(X_all.loc[val_A_idx].values, y_train.loc[val_A_idx].values)],
                    verbose=False)

        scores_val = model_A.predict(X_all.loc[val_A_idx].values)
        val_ic = _daily_ic(scores_val, y_all.loc[val_A_idx].values, val_A_idx)

        scores_test = model_A.predict(X_all.loc[test_idx].values)
        test_ic = _daily_ic(scores_test, y_all.loc[test_idx].values, test_idx)

        val_ics.append(val_ic)
        test_ics.append(test_ic)

    if len(val_ics) < 3:
        return {"mean_val_ic": np.nan, "mean_test_ic": np.nan,
                "ic_stability": np.nan, "val_test_gap": np.nan, "n_steps": 0}

    val_s = pd.Series(val_ics)
    test_s = pd.Series(test_ics)

    return {
        "mean_val_ic": float(val_s.mean()),
        "mean_test_ic": float(test_s.mean()),
        "ic_stability": float(val_s.corr(test_s)),
        "val_test_gap": float(abs(val_s.mean() - test_s.mean())),
        "n_steps": len(val_ics),
    }


def main():
    feat_path = DATA / "features.parquet"
    if not feat_path.exists():
        sys.exit("ERROR: data/features.parquet not found")

    print("Loading features...", flush=True)
    panel = pd.read_parquet(feat_path)
    panel = panel.reset_index()
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.set_index(["date", "etf_id"])

    device = _try_gpu()
    print(f"Device: {device.upper()}", flush=True)

    all_feature_cols = get_all_feature_cols(panel)
    print(f"Panel: {len(panel)} rows, {len(all_feature_cols)} candidate features", flush=True)

    # Precompute row_pos once
    dates = panel.index.get_level_values("date").unique().sort_values()
    date_to_pos = {d: i for i, d in enumerate(dates)}
    row_dates = panel.index.get_level_values("date")
    row_pos = pd.Series([date_to_pos[d] for d in row_dates], index=panel.index)

    total = len(POWER_RANGE) * len(TOP_N_RANGE) * len(DEPTH_RANGE)
    print(f"\nSearch: {len(POWER_RANGE)} powers × {len(TOP_N_RANGE)} caps × "
          f"{len(DEPTH_RANGE)} depths = {total} combos")
    print(f"Last {N_LAST_STEPS} steps (~{N_LAST_STEPS} months)")
    print(f"{'power':>6} {'cap':>4} {'depth':>5} {'n_feat':>6} {'val_IC':>8} "
          f"{'test_IC':>8} {'stab':>6} {'gap':>6}")
    print("-" * 65)

    results = []
    done = 0

    # Cache feature selections per (power, top_n) to avoid recomputing
    feat_cache = {}

    for power in POWER_RANGE:
        for top_n in TOP_N_RANGE:
            key = (round(power, 1), top_n)
            selected = select_features_v1(panel, all_feature_cols, device, power, top_n, row_pos)
            feat_cache[key] = selected
            n_feat = len(selected)

            if n_feat < 5:
                print(f"{power:6.1f} {top_n:4d}  -- too few features ({n_feat}), skip",
                      flush=True)
                done += len(DEPTH_RANGE)
                continue

            for depth in DEPTH_RANGE:
                done += 1
                available = [c for c in selected if c in panel.columns]
                metrics = run_last_n_steps(panel, available, device, depth, N_LAST_STEPS, row_pos)

                row = {"power": round(power, 1), "top_n": top_n, "depth": depth,
                       "n_features": len(available), **metrics}
                results.append(row)

                print(f"{power:6.1f} {top_n:4d} {depth:5d} {len(available):6d} "
                      f"{metrics['mean_val_ic']:+8.4f} {metrics['mean_test_ic']:+8.4f} "
                      f"{metrics['ic_stability']:6.3f} {metrics['val_test_gap']:6.4f}  "
                      f"[{done}/{total}]", flush=True)

    res_df = pd.DataFrame(results)
    res_df["composite"] = (
        res_df["mean_test_ic"] * res_df["ic_stability"].clip(0, 1) - res_df["val_test_gap"]
    )
    res_df = res_df.sort_values("composite", ascending=False)
    res_df.to_csv(MYFILES / "search_results.csv", index=False)

    print(f"\n{'='*65}")
    print("Top 10 by composite (test_IC × stability - gap):")
    print(res_df.head(10).to_string(index=False))
    print(f"\nBest: power={res_df.iloc[0]['power']:.1f}  "
          f"cap={int(res_df.iloc[0]['top_n'])}  "
          f"depth={int(res_df.iloc[0]['depth'])}  "
          f"n_feat={int(res_df.iloc[0]['n_features'])}  "
          f"test_IC={res_df.iloc[0]['mean_test_ic']:+.4f}  "
          f"stability={res_df.iloc[0]['ic_stability']:.3f}")


if __name__ == "__main__":
    main()
