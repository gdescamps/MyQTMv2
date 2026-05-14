"""
Walk-Forward Expanding XGBoost training for MyQTM-ETF.

For each step in the walk-forward:
  - Split the training period into interlaced monthly blocks (50/50):
      Even blocks → XGBoost training
      Odd blocks  → Early stopping + CMA-ES optimisation (in backtest.py)
  - Predict on odd blocks (split="val") → used by CMA-ES in backtest.py
  - Predict on next TEST_WINDOW days (split="test") → true OOS backtest

IC maximisation:
  Labels are z-scored cross-sectionally per date before training.
  Minimising MSE(ŷ, zscore(y)) is equivalent to maximising equal-weighted
  cross-sectional IC: each date contributes equally to the loss regardless
  of its cross-sectional return variance.

Output: data/oos_predictions.parquet
  MultiIndex: (date, etf_id)
  Columns:    score, label (original, unscaled), step, split, best_iter, val_ic

GPU: uses device='cuda' (NVIDIA GB10), falls back to CPU if unavailable.
"""

import sys
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path

DATA    = Path(__file__).parent / "data"
OUTPUTS = Path(__file__).parent / "outputs"

# Walk-forward parameters
MIN_TRAIN_ROWS = 750   # ~3 years of trading days before first test
TEST_WINDOW    = 63    # ~3 months per test step
STEP           = 63    # refit every ~3 months
BLOCK_ROWS     = 21    # ~1 month alternating blocks for interlaced train/val

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
    "shares_outstanding_z5", "shares_outstanding_z20", "shares_outstanding_z60",
    "rotation_z5", "rotation_z20", "rotation_z60",
    # Cross-sectional
    "ret_20d_z_within_block", "ret_5d_z_within_block",
]

LABEL_COL = "label"

XGB_PARAMS = dict(
    tree_method          = "hist",
    max_depth            = 4,
    min_child_weight     = 40,
    subsample            = 0.744,
    colsample_bytree     = 0.470,
    learning_rate        = 0.088,
    reg_alpha            = 0.0002,
    reg_lambda           = 9.414,
    n_estimators         = 1000,
    early_stopping_rounds= 30,
    objective            = "reg:squarederror",
    eval_metric          = "rmse",
    verbosity            = 0,
)


def _zscore_per_date(y: pd.Series) -> pd.Series:
    """
    Z-score labels cross-sectionally within each date.

    Minimising MSE against these z-scored labels is equivalent to maximising
    equal-weighted cross-sectional IC: every date contributes identically to
    the loss regardless of its cross-sectional return variance.
    """
    return y.groupby(level="date").transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-8)
    )


def _daily_ic(scores: np.ndarray, labels: np.ndarray, index: pd.MultiIndex) -> float:
    """Mean cross-sectional IC (Pearson per date, averaged across dates)."""
    df = pd.DataFrame({"score": scores, "label": labels}, index=index).dropna()
    if len(df) < 2:
        return float("nan")
    ic_by_date = df.groupby(level="date").apply(
        lambda x: x["score"].corr(x["label"]) if len(x) > 1 else np.nan
    )
    return float(ic_by_date.dropna().mean())


def _try_gpu() -> str:
    try:
        X = np.random.rand(50, 5).astype(np.float32)
        y = np.random.rand(50).astype(np.float32)
        xgb.XGBRegressor(device="cuda", n_estimators=5, verbosity=0).fit(X, y)
        return "cuda"
    except Exception:
        return "cpu"


def run_walk_forward(
    panel: pd.DataFrame,
    device: str,
    xgb_params: dict | None = None,
    verbose: bool = True,
    save_models: bool = True,
) -> pd.DataFrame:
    params = {**XGB_PARAMS, **(xgb_params or {})}
    params["device"] = device

    dates = panel.index.get_level_values("date").unique().sort_values()
    n = len(dates)

    available_cols = [c for c in FEATURE_COLS if c in panel.columns]
    X_all   = panel[available_cols].astype(np.float32)
    y_all   = panel[LABEL_COL].astype(np.float32)          # original labels (saved + IC)
    y_train = _zscore_per_date(y_all).astype(np.float32)   # z-scored labels (for fit)

    date_to_pos = {d: i for i, d in enumerate(dates)}
    row_dates   = panel.index.get_level_values("date")
    row_pos     = pd.Series([date_to_pos[d] for d in row_dates], index=panel.index)

    predictions = []
    ic_log      = []   # [(step, val_ic, test_ic)]
    step_n      = 0

    train_end_pos = MIN_TRAIN_ROWS
    while train_end_pos + TEST_WINDOW <= n:
        test_end_pos = min(train_end_pos + TEST_WINDOW, n)

        in_train     = row_pos < train_end_pos
        block_parity = (row_pos // BLOCK_ROWS) % 2

        train_mask = in_train & (block_parity == 0)
        val_mask   = in_train & (block_parity == 1)
        test_mask  = (row_pos >= train_end_pos) & (row_pos < test_end_pos)

        tr_idx   = panel.index[train_mask & y_all.notna()]
        val_idx  = panel.index[val_mask   & y_all.notna()]
        test_idx = panel.index[test_mask]

        if len(tr_idx) < 100 or len(val_idx) < 10:
            train_end_pos += STEP
            continue

        # Train on z-scored labels; eval_set also z-scored (for RMSE early stopping)
        X_tr   = X_all.loc[tr_idx].values
        y_tr   = y_train.loc[tr_idx].values
        X_val  = X_all.loc[val_idx].values
        y_val  = y_train.loc[val_idx].values

        model = xgb.XGBRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        best_iter = model.best_iteration

        if save_models:
            model_dir = OUTPUTS / "models"
            model_dir.mkdir(parents=True, exist_ok=True)
            model.save_model(str(model_dir / f"step_{step_n:02d}.ubj"))

        train_dates = dates[:train_end_pos]
        test_dates  = dates[train_end_pos:test_end_pos]

        # Predict on val blocks (save original labels for CMA-ES / backtest)
        val_ic = float("nan")
        if len(val_idx) > 0:
            val_scores = model.predict(X_all.loc[val_idx].values)
            val_ic     = _daily_ic(val_scores, y_all.loc[val_idx].values, val_idx)
            val_df     = pd.DataFrame(
                {"score": val_scores, "label": y_all.loc[val_idx].values},
                index=val_idx,
            )
            val_df["step"]      = step_n
            val_df["split"]     = "val"
            val_df["best_iter"] = best_iter
            val_df["val_ic"]    = val_ic
            predictions.append(val_df)

        # Predict on test window (save original labels)
        test_ic = float("nan")
        if len(test_idx) > 0:
            test_scores = model.predict(X_all.loc[test_idx].values)
            test_ic     = _daily_ic(test_scores, y_all.loc[test_idx].values, test_idx)
            test_df     = pd.DataFrame(
                {"score": test_scores, "label": y_all.loc[test_idx].values},
                index=test_idx,
            )
            test_df["step"]      = step_n
            test_df["split"]     = "test"
            test_df["best_iter"] = best_iter
            test_df["val_ic"]    = val_ic   # val IC of the model that produced this test
            predictions.append(test_df)

        ic_log.append((step_n, val_ic, test_ic))
        step_n += 1

        if verbose:
            print(
                f"  Step {step_n:2d}  "
                f"[{dates[0].date()} → {train_dates[-1].date()}]  "
                f"val_IC={val_ic:+.4f}  "
                f"test [{test_dates[0].date()} → {test_dates[-1].date()}]  "
                f"test_IC={test_ic:+.4f}  "
                f"iter={best_iter}"
            )

        train_end_pos += STEP

    if not predictions:
        raise RuntimeError("No predictions produced — check MIN_TRAIN_ROWS vs data length")

    # IC summary
    ic_df = pd.DataFrame(ic_log, columns=["step", "val_ic", "test_ic"])
    if verbose:
        print(f"\n{'='*60}")
        print(f"  {'Mean val IC':30s} {ic_df['val_ic'].mean():+.4f}")
        print(f"  {'Mean test IC (true OOS)':30s} {ic_df['test_ic'].mean():+.4f}")
        print(f"  {'IC stability (val/test corr)':30s} "
              f"{ic_df['val_ic'].corr(ic_df['test_ic']):+.3f}")
        print(f"{'='*60}")

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
    print(
        f"\nWalk-Forward: MIN_TRAIN={MIN_TRAIN_ROWS}d  TEST={TEST_WINDOW}d  "
        f"STEP={STEP}d  BLOCK={BLOCK_ROWS}d\n"
        f"Labels: z-scored per date (IC maximisation)\n"
    )

    oos = run_walk_forward(panel, device)

    out = DATA / "oos_predictions.parquet"
    oos.to_parquet(out)

    test_oos = oos[oos["split"] == "test"].dropna(subset=["score", "label"])
    mean_test_ic = test_oos.groupby(test_oos.index.get_level_values("date")).apply(
        lambda x: x["score"].corr(x["label"])
    ).mean()

    val_oos = oos[oos["split"] == "val"].dropna(subset=["score", "label"])
    mean_val_ic = val_oos.groupby(val_oos.index.get_level_values("date")).apply(
        lambda x: x["score"].corr(x["label"])
    ).mean()

    print(f"\nSaved → {out.name}")
    print(f"  val rows:       {(oos['split']=='val').sum()}")
    print(f"  test rows:      {(oos['split']=='test').sum()}")
    print(f"  Mean val IC:    {mean_val_ic:+.4f}")
    print(f"  Mean test IC:   {mean_test_ic:+.4f}")
    print(
        f"  Test range:     "
        f"{test_oos.index.get_level_values('date').min().date()} → "
        f"{test_oos.index.get_level_values('date').max().date()}"
    )


if __name__ == "__main__":
    main()
