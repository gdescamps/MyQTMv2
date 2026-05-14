"""
Walk-Forward Expanding XGBoost training for MyQTM-ETF.

For each step in the walk-forward:
  - Train XGBoost on all data up to train_end (with early stopping on last year)
  - Predict on next TEST_WINDOW days → out-of-sample predictions

Output: data/oos_predictions.parquet  (MultiIndex: date, etf_id → score)

GPU: uses device='cuda' (NVIDIA GB10), falls back to CPU if unavailable.
"""

import sys
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path

DATA = Path(__file__).parent / "data"

# Walk-forward parameters
MIN_TRAIN_ROWS = 750   # ~3 years of trading days before first test
TEST_WINDOW    = 250   # ~1 year per test step
STEP           = 125   # refit every ~6 months
VAL_ROWS       = 250   # last year of train = validation for early stopping

FEATURE_COLS = [
    # Technical
    "ret_1d", "ret_5d", "ret_20d", "ret_60d",
    "vol_20d", "vol_60d",
    "rsi_14",
    "price_vs_ma50", "price_vs_ma200",
    "ma_20_slope", "ma_50_slope",
    "atr_14", "volume_z20",
    # Macro / regime
    "vix_level", "vix_velocity", "vix_reversion_force",
    "hy_spread", "hy_spread_z60", "hy_spread_velocity",
    "yield_curve", "yield_curve_velocity",
    "dxy_ret_20d", "dxy_z60",
    "ret_spx_20d",
    # Smart money
    "shares_outstanding_z20",
    "rotation_z60",
    # Cross-sectional
    "ret_20d_z_within_block", "ret_5d_z_within_block",
]

LABEL_COL = "label"

XGB_PARAMS = dict(
    tree_method     = "hist",
    max_depth       = 4,
    min_child_weight= 50,
    subsample       = 0.8,
    colsample_bytree= 0.8,
    learning_rate   = 0.05,
    n_estimators    = 2000,
    early_stopping_rounds = 50,
    objective       = "reg:squarederror",
    eval_metric     = "rmse",
    verbosity       = 0,
)


def _try_gpu() -> str:
    try:
        X = np.random.rand(50, 5).astype(np.float32)
        y = np.random.rand(50).astype(np.float32)
        xgb.XGBRegressor(device="cuda", n_estimators=5, verbosity=0).fit(X, y)
        return "cuda"
    except Exception:
        return "cpu"


def run_walk_forward(panel: pd.DataFrame, device: str) -> pd.DataFrame:
    # All unique sorted dates across the panel
    dates = panel.index.get_level_values("date").unique().sort_values()
    n = len(dates)

    # Feature matrix and labels aligned to panel
    available_cols = [c for c in FEATURE_COLS if c in panel.columns]
    X_all = panel[available_cols].astype(np.float32)
    y_all = panel[LABEL_COL].astype(np.float32)

    # Map date → integer position
    date_to_pos = {d: i for i, d in enumerate(dates)}
    row_dates = panel.index.get_level_values("date")
    row_pos   = pd.Series([date_to_pos[d] for d in row_dates], index=panel.index)

    predictions = []
    step_n = 0

    train_end_pos = MIN_TRAIN_ROWS
    while train_end_pos + TEST_WINDOW <= n:
        test_end_pos = min(train_end_pos + TEST_WINDOW, n)

        train_dates = dates[:train_end_pos]
        val_dates   = dates[max(0, train_end_pos - VAL_ROWS):train_end_pos]
        test_dates  = dates[train_end_pos:test_end_pos]

        train_mask = row_pos < train_end_pos
        val_mask   = row_pos >= (train_end_pos - VAL_ROWS)
        train_val_mask = train_mask & val_mask

        # Drop rows with NaN label
        tr_idx  = panel.index[train_mask & y_all.notna()]
        val_idx = panel.index[train_val_mask & y_all.notna()]

        if len(tr_idx) < 100 or len(val_idx) < 10:
            train_end_pos += STEP
            continue

        X_tr  = X_all.loc[tr_idx].values
        y_tr  = y_all.loc[tr_idx].values
        X_val = X_all.loc[val_idx].values
        y_val = y_all.loc[val_idx].values

        model = xgb.XGBRegressor(device=device, **XGB_PARAMS)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        best_iter = model.best_iteration

        # Predict on test window
        test_mask = (row_pos >= train_end_pos) & (row_pos < test_end_pos)
        test_idx  = panel.index[test_mask]
        if len(test_idx) > 0:
            X_test  = X_all.loc[test_idx].values
            scores  = model.predict(X_test)
            step_df = pd.DataFrame(
                {"score": scores, "label": y_all.loc[test_idx].values},
                index=test_idx,
            )
            step_df["step"]      = step_n
            step_df["best_iter"] = best_iter
            predictions.append(step_df)

        step_n += 1
        print(f"  Step {step_n:2d}  train [{dates[0].date()} → {train_dates[-1].date()}]"
              f"  test [{test_dates[0].date()} → {test_dates[-1].date()}]"
              f"  best_iter={best_iter}  train_rows={len(tr_idx)}")

        train_end_pos += STEP

    if not predictions:
        raise RuntimeError("No OOS predictions produced — check MIN_TRAIN_ROWS vs data length")

    return pd.concat(predictions).sort_index()


def main():
    feat_path = DATA / "features.parquet"
    if not feat_path.exists():
        sys.exit("ERROR: data/features.parquet not found — run feature_engineering.py first")

    print("Loading features...")
    panel = pd.read_parquet(feat_path)
    panel = panel.reset_index()
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.set_index(["date", "etf_id"])

    n_dates = panel.index.get_level_values("date").nunique()
    n_etfs  = panel.index.get_level_values("etf_id").nunique()
    print(f"Panel: {n_dates} dates × {n_etfs} ETFs = {len(panel)} rows")

    available_cols = [c for c in FEATURE_COLS if c in panel.columns]
    missing = [c for c in FEATURE_COLS if c not in panel.columns]
    print(f"Features: {len(available_cols)} available, {len(missing)} missing: {missing}")

    device = _try_gpu()
    print(f"Device: {device.upper()}")
    print(f"\nWalk-Forward: MIN_TRAIN={MIN_TRAIN_ROWS}d  TEST={TEST_WINDOW}d  STEP={STEP}d\n")

    oos = run_walk_forward(panel, device)

    out = DATA / "oos_predictions.parquet"
    oos.to_parquet(out)

    # Quick IC (Information Coefficient = correlation between score and label)
    valid = oos.dropna(subset=["score", "label"])
    ic = valid.groupby(valid.index.get_level_values("date")).apply(
        lambda x: x["score"].corr(x["label"])
    ).mean()

    print(f"\nOOS predictions: {len(oos)} rows  saved → {out.name}")
    print(f"Mean daily IC (score vs label): {ic:.4f}")
    print(f"Date range: {oos.index.get_level_values('date').min().date()} → "
          f"{oos.index.get_level_values('date').max().date()}")


if __name__ == "__main__":
    main()
