"""
Grid search over feature selection hyperparams (power × cap) with fixed XGB params.

Search: power [1.2..1.7] × cap [80..120], depth=7, last 100 steps model A only.
Composite score: test_IC × stability(clip 0,1) - val_test_gap

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

DEPTH = 7
N_LAST_STEPS = 100

# Search grid
POWER_GRID = [1.2, 1.3, 1.4, 1.5, 1.6, 1.7]
CAP_GRID   = [80, 90, 100, 110, 120]


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

    combos = list(product(POWER_GRID, CAP_GRID))
    total = len(combos)
    print(f"\n=== Search: power × cap ({total} combos), d={DEPTH}, last {N_LAST_STEPS} steps ===")
    print(f"{'power':>5} {'cap':>4} {'#feat':>5} {'val_IC':>8} {'test_IC':>8} "
          f"{'stab':>6} {'gap':>6} {'composite':>9}  {'best':>4}")
    print("-" * 70)

    results = []
    best_composite = -999

    for i, (power, cap) in enumerate(combos):
        # Feature selection v2
        selected = select_features_v2(panel, all_feature_cols, power, cap, row_pos, device)
        available = [c for c in selected if c in panel.columns]
        n_feat = len(available)

        if n_feat < 10:
            print(f"{power:5.1f} {cap:4d} {n_feat:5d}  SKIP (too few features)  [{i+1}/{total}]",
                  flush=True)
            continue

        # Train model A on last N steps
        override = {"max_depth": DEPTH}
        metrics = run_last_n_steps(panel, available, device, override, N_LAST_STEPS, row_pos)

        composite = (metrics["mean_test_ic"] * max(0, min(1, metrics["ic_stability"]))
                     - metrics["val_test_gap"])
        is_best = composite > best_composite
        if is_best:
            best_composite = composite

        row = {"power": power, "cap": cap, "n_features": n_feat, **metrics,
               "composite": composite}
        results.append(row)

        print(f"{power:5.1f} {cap:4d} {n_feat:5d} "
              f"{metrics['mean_val_ic']:+8.4f} {metrics['mean_test_ic']:+8.4f} "
              f"{metrics['ic_stability']:6.3f} {metrics['val_test_gap']:6.4f} "
              f"{composite:+9.5f}  {'★' if is_best else '':>4}  "
              f"[{i+1}/{total}]", flush=True)

    df = pd.DataFrame(results).sort_values("composite", ascending=False)
    df.to_csv(MYFILES / "search_results.csv", index=False)

    print(f"\n{'='*70}")
    print("Top 10 results:")
    print(df.head(10).to_string(index=False))

    best = df.iloc[0]
    print(f"\n{'='*70}")
    print(f"BEST: power={best['power']:.1f}  cap={int(best['cap'])}  "
          f"#feat={int(best['n_features'])}")
    print(f"  test_IC    = {best['mean_test_ic']:+.4f}")
    print(f"  stability  = {best['ic_stability']:.3f}")
    print(f"  val/test gap = {best['val_test_gap']:.4f}")
    print(f"  composite  = {best['composite']:+.5f}")


if __name__ == "__main__":
    main()
