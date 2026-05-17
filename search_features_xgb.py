"""
Grid search: feature filtering + XGBoost depth.

For each (mean_std_power, top_features, max_depth):
  1. Select features by stability: mean / std^power, capped at top_features
  2. Train all 81 walk-forward steps (model A only, 5y rolling)
  3. Report mean test IC

Output: myfiles/search_features_xgb_results.csv
"""

import json
import sys
import time
import itertools
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from train import (
    XGB_PARAMS, _try_gpu, LABEL_COL, _zscore_per_date,
    BLOCK_ROWS, EMBARGO_ROWS, run_walk_forward,
)
from select_features import get_all_feature_cols
import train as train_mod

DATA    = Path(__file__).parent / "data"
OUTPUTS = Path(__file__).parent / "myfiles"
OUTPUTS.mkdir(exist_ok=True)

# Search grid
POWER_RANGE = [1.2, 1.3, 1.4, 1.5, 1.6, 1.7]
CAP_RANGE   = [50, 75, 100, 125, 150, 175, 200]
DEPTH_RANGE = [6, 7, 8]

# Keep N best models
N_BEST = 5


def select_features_by_stability(panel, feature_cols, device, power, cap):
    """Run 3-period feature selection and return top features."""
    N_PERIODS = 3
    dates = panel.index.get_level_values("date").unique().sort_values()
    date_to_pos = {d: i for i, d in enumerate(dates)}
    row_dates = panel.index.get_level_values("date")
    row_pos = pd.Series([date_to_pos[d] for d in row_dates], index=panel.index)

    block_idx = row_pos // BLOCK_ROWS
    block_id = block_idx % N_PERIODS
    pos_in_block = row_pos % BLOCK_ROWS
    not_embargoed = (pos_in_block >= EMBARGO_ROWS) & (pos_in_block < BLOCK_ROWS - EMBARGO_ROWS)

    X_all = panel[feature_cols].astype(np.float32)
    y_all = panel[LABEL_COL].astype(np.float32)
    y_z = _zscore_per_date(y_all).astype(np.float32)

    params = {**XGB_PARAMS, "device": device, "max_depth": 7}

    importances = {}
    for period in range(3):
        train_mask = (block_id != period) & y_all.notna() & not_embargoed
        val_mask = (block_id == period) & y_all.notna() & not_embargoed
        tr_idx = panel.index[train_mask]
        val_idx = panel.index[val_mask]

        model = xgb.XGBRegressor(**params)
        model.fit(
            X_all.loc[tr_idx].values, y_z.loc[tr_idx].values,
            eval_set=[(X_all.loc[val_idx].values, y_z.loc[val_idx].values)],
            verbose=False,
        )
        importances[f"period_{period}"] = model.feature_importances_

    imp_df = pd.DataFrame(importances, index=feature_cols)
    imp_df["mean"] = imp_df.mean(axis=1)
    imp_df["std"] = imp_df.std(axis=1)
    imp_df["stability"] = imp_df["mean"] / (imp_df["std"].replace(0, np.nan) ** power)
    imp_df = imp_df.sort_values("stability", ascending=False)
    imp_df = imp_df[imp_df["stability"].notna() & (imp_df["mean"] > 0)]

    selected = imp_df.index[:cap].tolist()
    return selected


def compute_metrics(oos: pd.DataFrame) -> dict:
    """Compute test IC, val IC, gap, and stability."""
    test_df = oos[oos["split"] == "test"].dropna(subset=["score", "label"])
    val_df = oos[oos["split"] == "val"].dropna(subset=["score", "label"])

    if len(test_df) < 10:
        return {"test_ic": float("nan"), "val_ic": float("nan"), "gap": float("nan"), "stability": float("nan")}

    test_ic_per_step = (
        test_df.reset_index()
        .groupby(["step", "date"])
        .apply(lambda g: g["score"].corr(g["label"]) if len(g) > 1 else np.nan)
        .groupby(level="step")
        .mean()
    )

    val_ic_per_step = val_df.groupby("step")["val_ic"].first() if "val_ic" in val_df.columns else pd.Series(dtype=float)

    mean_test_ic = float(test_ic_per_step.dropna().mean())
    mean_val_ic = float(val_ic_per_step.dropna().mean()) if len(val_ic_per_step) > 0 else float("nan")
    gap = max(mean_val_ic - mean_test_ic, 0.0) if not np.isnan(mean_val_ic) else float("nan")
    stability = float(val_ic_per_step.corr(test_ic_per_step)) if len(val_ic_per_step) > 0 else float("nan")
    if np.isnan(stability):
        stability = 0.0

    return {"test_ic": mean_test_ic, "val_ic": mean_val_ic, "gap": gap, "stability": stability}


def fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s"


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
    n_dates = panel.index.get_level_values("date").nunique()
    n_etfs = panel.index.get_level_values("etf_id").nunique()
    print(f"Panel: {n_dates} dates x {n_etfs} ETFs = {len(panel)} rows", flush=True)
    print(f"Candidate features: {len(feature_cols)}", flush=True)

    device = _try_gpu()
    print(f"Device: {device.upper()}", flush=True)

    combos = list(itertools.product(POWER_RANGE, CAP_RANGE, DEPTH_RANGE))
    n_combos = len(combos)
    print(f"\nGrid: {len(POWER_RANGE)} powers x {len(CAP_RANGE)} caps x {len(DEPTH_RANGE)} depths = {n_combos} combos\n")
    print(f"{'#':>5}  {'power':>5}  {'cap':>4}  {'depth':>5}  {'n_feat':>6}  {'test_IC':>8}  {'val_IC':>8}  {'gap':>6}  {'stab':>6}  {'time':>7}")
    print("-" * 90)

    results = []
    t0 = time.time()

    # Feature selection only depends on power + cap, cache it
    feat_cache = {}

    for i, (power, cap, depth) in enumerate(combos):
        t_start = time.time()

        # Select features (cached by power since cap just truncates)
        if power not in feat_cache:
            feat_cache[power] = select_features_by_stability(
                panel, feature_cols, device, power, cap=max(CAP_RANGE)
            )
        selected = feat_cache[power][:cap]
        n_feat = len(selected)

        # Update train module to use these features
        train_mod.FEATURE_COLS = selected

        # Train walk-forward
        xgb_override = {"max_depth": depth}
        try:
            oos = run_walk_forward(
                panel, device,
                xgb_params=xgb_override,
                verbose=False,
                save_models=False,
                dual_model=False,
            )
            metrics = compute_metrics(oos)
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            metrics = {"test_ic": float("nan"), "val_ic": float("nan"), "gap": float("nan"), "stability": float("nan")}

        t_elapsed = time.time() - t_start
        results.append({
            "power": power, "cap": cap, "depth": depth,
            "n_features": n_feat, **metrics,
            "time_s": t_elapsed,
        })

        test_ic = metrics["test_ic"]
        valid_ics = [r["test_ic"] for r in results if not np.isnan(r.get("test_ic", float("nan")))]
        best_ic = max(valid_ics) if valid_ics else float("nan")
        marker = " ***" if test_ic == best_ic and not np.isnan(test_ic) else ""

        elapsed_total = time.time() - t0
        eta = elapsed_total / (i + 1) * (n_combos - i - 1)

        print(f"{i+1:3d}/{n_combos}  {power:5.2f}  {cap:4d}  {depth:5d}  {n_feat:6d}  "
              f"test={test_ic:+.4f}  val={metrics['val_ic']:+.4f}  gap={metrics['gap']:.3f}  stab={metrics['stability']:+.3f}  "
              f"{fmt_time(t_elapsed)}  ETA {fmt_time(eta)}{marker}", flush=True)

    # Save results
    df = pd.DataFrame(results).sort_values("test_ic", ascending=False)
    df.to_csv(OUTPUTS / "search_features_xgb_results.csv", index=False)

    # Save top N best
    best_df = df.head(N_BEST)
    best_df.to_csv(OUTPUTS / "search_features_xgb_best.csv", index=False)

    total_time = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  Done in {fmt_time(total_time)} — {n_combos} combos evaluated")
    print(f"\n  Top {N_BEST} models:")
    print(f"  {'#':>3}  {'power':>5}  {'cap':>4}  {'depth':>5}  {'test_IC':>8}  {'val_IC':>8}  {'gap':>6}  {'stab':>6}")
    for j, (_, row) in enumerate(best_df.iterrows()):
        print(f"  {j+1:3d}  {row['power']:5.2f}  {int(row['cap']):4d}  "
              f"{int(row['depth']):5d}  {row['test_ic']:+.4f}  {row['val_ic']:+.4f}  "
              f"{row['gap']:.3f}  {row['stability']:+.3f}")
    print(f"{'='*70}")
    print(f"\nSaved → myfiles/search_features_xgb_results.csv ({len(df)} rows)")
    print(f"Saved → myfiles/search_features_xgb_best.csv (top {N_BEST})")

    # Retrain best model with save
    print(f"\n--- Retraining best model ---")
    best = df.iloc[0]
    best_power = best["power"]
    best_cap = int(best["cap"])
    best_depth = int(best["depth"])
    best_features = feat_cache[best_power][:best_cap]
    print(f"  power={best_power:.2f}  cap={best_cap}  depth={best_depth}  "
          f"test_IC={best['test_ic']:+.4f}", flush=True)

    train_mod.FEATURE_COLS = best_features
    oos = run_walk_forward(
        panel, device,
        xgb_params={"max_depth": best_depth},
        verbose=True,
        save_models=True,
        dual_model=False,
    )

    from pathlib import Path as _P
    oos.to_parquet(DATA / "oos_predictions.parquet")
    with open(_P(__file__).parent / "outputs" / "selected_features.json", "w") as f:
        json.dump(best_features, f, indent=2)
    print(f"  Saved oos_predictions.parquet + selected_features.json ({len(best_features)} features)")


if __name__ == "__main__":
    main()
