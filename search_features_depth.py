"""
Grid search over XGBoost model params with fixed feature selection.

Fixed: power=1.3, cap=90, depth=7
Search: min_child_weight, subsample, colsample_bytree, learning_rate, reg_alpha, reg_lambda

Output: myfiles/search_results.csv
"""

import sys
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from itertools import product

sys.path.insert(0, str(Path(__file__).parent))
from train import (XGB_PARAMS, _try_gpu, LABEL_COL, _zscore_per_date,
                   BLOCK_ROWS, EMBARGO_ROWS, TEST_WINDOW, STEP, MIN_TRAIN_ROWS,
                   _daily_ic)
from select_features import get_all_feature_cols

DATA    = Path(__file__).parent / "data"
OUTPUTS = Path(__file__).parent / "outputs"
MYFILES = Path(__file__).parent / "myfiles"
MYFILES.mkdir(exist_ok=True)

# Fixed feature selection
POWER = 1.3
TOP_N = 90
DEPTH = 7
N_LAST_STEPS = 30

# Search grid for XGB params
PARAM_GRID = {
    "min_child_weight": [20, 41, 60, 80],
    "subsample":        [0.6, 0.7, 0.838, 0.9],
    "colsample_bytree": [0.5, 0.678, 0.8, 1.0],
    "learning_rate":    [0.02, 0.04, 0.06],
    "reg_alpha":        [0.001, 0.1, 1.0],
    "reg_lambda":       [0.5, 1.674, 5.0],
}


def select_features_v2(panel, feature_cols, power, top_n, row_pos, device):
    """V2: 3 interlaced periods, no normalization, no zero filter, top N."""
    block_idx = row_pos // BLOCK_ROWS
    block_id = block_idx % 3
    pos_in_block = row_pos % BLOCK_ROWS
    not_embargoed = (pos_in_block >= EMBARGO_ROWS) & (pos_in_block < BLOCK_ROWS - EMBARGO_ROWS)

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

        if len(tr_idx) < 100 or len(val_idx) < 10:
            continue

        model = xgb.XGBRegressor(**params)
        model.fit(X_all.loc[tr_idx].values, y_z.loc[tr_idx].values,
                  eval_set=[(X_all.loc[val_idx].values, y_z.loc[val_idx].values)],
                  verbose=False)
        importances[f"period_{period}"] = model.feature_importances_

    if len(importances) < 2:
        return []

    imp_df = pd.DataFrame(importances, index=feature_cols)
    imp_df["mean"] = imp_df.mean(axis=1)
    imp_df["std"] = imp_df.std(axis=1)
    imp_df["score"] = imp_df["mean"] / (imp_df["std"].replace(0, np.nan) ** power)
    imp_df = imp_df.sort_values("score", ascending=False)
    imp_df = imp_df[imp_df["score"].notna() & (imp_df["mean"] > 0)]

    return imp_df.index[:min(top_n, len(imp_df))].tolist()


def run_last_n_steps(panel, feature_cols, device, xgb_override, n_steps, row_pos):
    """Train model A only on last N test steps with custom XGB params."""
    params = {**XGB_PARAMS, "device": device, "max_depth": DEPTH, **xgb_override}

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

    # Precompute row_pos
    dates = panel.index.get_level_values("date").unique().sort_values()
    date_to_pos = {d: i for i, d in enumerate(dates)}
    row_dates = panel.index.get_level_values("date")
    row_pos = pd.Series([date_to_pos[d] for d in row_dates], index=panel.index)

    # Feature selection (once)
    print(f"\nFeature selection: power={POWER}, cap={TOP_N}", flush=True)
    selected = select_features_v2(panel, all_feature_cols, POWER, TOP_N, row_pos, device)
    available = [c for c in selected if c in panel.columns]
    print(f"Selected: {len(available)} features", flush=True)

    # Build param combos — search pairs to keep tractable
    # Phase 1: subsample × colsample × min_child_weight
    # Phase 2: learning_rate × reg_alpha × reg_lambda (with best from phase 1)

    # Phase 1
    phase1 = list(product(
        PARAM_GRID["min_child_weight"],
        PARAM_GRID["subsample"],
        PARAM_GRID["colsample_bytree"],
    ))

    print(f"\n=== Phase 1: min_child_weight × subsample × colsample ({len(phase1)} combos) ===")
    print(f"{'mcw':>4} {'sub':>5} {'col':>5} {'val_IC':>8} {'test_IC':>8} {'stab':>6} {'gap':>6}")
    print("-" * 50)

    results_p1 = []
    for i, (mcw, sub, col) in enumerate(phase1):
        override = {"min_child_weight": mcw, "subsample": sub, "colsample_bytree": col}
        metrics = run_last_n_steps(panel, available, device, override, N_LAST_STEPS, row_pos)
        row = {"min_child_weight": mcw, "subsample": sub, "colsample_bytree": col, **metrics}
        results_p1.append(row)
        print(f"{mcw:4d} {sub:5.2f} {col:5.3f} "
              f"{metrics['mean_val_ic']:+8.4f} {metrics['mean_test_ic']:+8.4f} "
              f"{metrics['ic_stability']:6.3f} {metrics['val_test_gap']:6.4f}  "
              f"[{i+1}/{len(phase1)}]", flush=True)

    df1 = pd.DataFrame(results_p1)
    df1["composite"] = df1["mean_test_ic"] * df1["ic_stability"].clip(0, 1) - df1["val_test_gap"]
    df1 = df1.sort_values("composite", ascending=False)

    best_mcw = df1.iloc[0]["min_child_weight"]
    best_sub = df1.iloc[0]["subsample"]
    best_col = df1.iloc[0]["colsample_bytree"]

    print(f"\nPhase 1 best: mcw={int(best_mcw)} sub={best_sub:.2f} col={best_col:.3f}  "
          f"test_IC={df1.iloc[0]['mean_test_ic']:+.4f}  stab={df1.iloc[0]['ic_stability']:.3f}")
    print("\nTop 5 phase 1:")
    print(df1.head(5).to_string(index=False))

    # Phase 2: learning_rate × reg_alpha × reg_lambda (with best phase 1)
    phase2 = list(product(
        PARAM_GRID["learning_rate"],
        PARAM_GRID["reg_alpha"],
        PARAM_GRID["reg_lambda"],
    ))

    print(f"\n=== Phase 2: lr × alpha × lambda ({len(phase2)} combos) ===")
    print(f"{'lr':>5} {'alpha':>6} {'lambda':>6} {'val_IC':>8} {'test_IC':>8} {'stab':>6} {'gap':>6}")
    print("-" * 55)

    results_p2 = []
    for i, (lr, alpha, lam) in enumerate(phase2):
        override = {
            "min_child_weight": int(best_mcw),
            "subsample": best_sub,
            "colsample_bytree": best_col,
            "learning_rate": lr,
            "reg_alpha": alpha,
            "reg_lambda": lam,
        }
        metrics = run_last_n_steps(panel, available, device, override, N_LAST_STEPS, row_pos)
        row = {"learning_rate": lr, "reg_alpha": alpha, "reg_lambda": lam, **metrics}
        results_p2.append(row)
        print(f"{lr:5.3f} {alpha:6.3f} {lam:6.3f} "
              f"{metrics['mean_val_ic']:+8.4f} {metrics['mean_test_ic']:+8.4f} "
              f"{metrics['ic_stability']:6.3f} {metrics['val_test_gap']:6.4f}  "
              f"[{i+1}/{len(phase2)}]", flush=True)

    df2 = pd.DataFrame(results_p2)
    df2["composite"] = df2["mean_test_ic"] * df2["ic_stability"].clip(0, 1) - df2["val_test_gap"]
    df2 = df2.sort_values("composite", ascending=False)

    print(f"\nPhase 2 best: lr={df2.iloc[0]['learning_rate']:.3f} "
          f"alpha={df2.iloc[0]['reg_alpha']:.3f} lambda={df2.iloc[0]['reg_lambda']:.3f}  "
          f"test_IC={df2.iloc[0]['mean_test_ic']:+.4f}  stab={df2.iloc[0]['ic_stability']:.3f}")
    print("\nTop 5 phase 2:")
    print(df2.head(5).to_string(index=False))

    # Save all results
    all_results = pd.concat([
        df1.assign(phase="p1_struct"),
        df2.assign(phase="p2_reg"),
    ], ignore_index=True)
    all_results.to_csv(MYFILES / "search_results.csv", index=False)

    print(f"\n{'='*60}")
    print(f"Final best params (d={DEPTH}, power={POWER}, cap={TOP_N}):")
    print(f"  min_child_weight = {int(best_mcw)}")
    print(f"  subsample        = {best_sub:.3f}")
    print(f"  colsample_bytree = {best_col:.3f}")
    print(f"  learning_rate    = {df2.iloc[0]['learning_rate']:.3f}")
    print(f"  reg_alpha        = {df2.iloc[0]['reg_alpha']:.3f}")
    print(f"  reg_lambda       = {df2.iloc[0]['reg_lambda']:.3f}")


if __name__ == "__main__":
    main()
