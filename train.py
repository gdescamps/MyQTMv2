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
    # Momentum
    "ret_20d", "ret_60d", "ret_120d", "ret_250d",
    # Momentum ratios
    "mom_ratio_60v120", "mom_ratio_20v120",
    # Volatility
    "vol_10d", "vol_20d", "vol_60d", "vol_120d",
    "vol_ratio_10v120", "vol_ratio_20v120",
    # RSI
    "rsi_21", "rsi_60",
    # Moving averages: price distance
    "price_vs_ma200",
    # Moving averages: slopes
    "ma_20_slope", "ma_50_slope", "ma_100_slope", "ma_200_slope",
    # Moving averages: crossovers
    "ma10_vs_ma50", "ma20_vs_ma50", "ma50_vs_ma100", "ma50_vs_ma200", "ma100_vs_ma200",
    # ATR
    "atr_14", "atr_21",
    # Bollinger / Drawdown / Distribution
    "bb_position_120", "drawdown_250", "kurtosis_60",
    # Volume
    "volume_z5", "volume_z10", "vol_cross_20v60",
    # Macro / regime
    "vix_level", "vix_squared",
    "hy_spread", "hy_spread_z60", "hy_spread_velocity_20",
    "hy_cross_5v60", "hy_cross_20v60",
    "yield_curve", "yield_curve_z60", "yield_curve_velocity_20",
    "yc_x_vix",
    "dxy_ret_20d", "dxy_ret_60d", "dxy_z60", "dxy_cross_20v60",
    "ret_spx_20d", "ret_spx_60d", "spx_vol_20d", "spx_drawdown_60",
    # Smart money
    "so_cross_20v120", "so_cross_60v120", "so_ret_20d", "so_ret_60d",
    # Cross-sectional z-scores
    "vol_20d_z_xs", "vol_60d_z_xs", "vol_120d_z_xs",
    "ret_60d_z_xs", "ret_120d_z_xs", "ret_250d_z_xs",
    "atr_14_z_xs", "drawdown_60_z_xs",
    "ret_60d_z_within_block",
    # Cross-sectional ranks
    "ret_120d_rank", "ret_250d_rank", "vol_20d_rank",
    # Momentum acceleration cross-sectional
    "mom_accel_20dv60d", "mom_accel_60dv120d",
    "mom_accel_5dv20d_z_xs", "mom_accel_5dv60d_z_xs", "mom_accel_20dv60d_z_xs",
    # Non-linearities
    "ret_60d_z_xs_sq", "ret_120d_z_xs_sq",
    "mom_x_vol_20d", "mom_x_vol_60d",
]

LABEL_COL = "label"

XGB_PARAMS = dict(
    tree_method          = "hist",
    max_depth            = 3,
    min_child_weight     = 41,
    subsample            = 0.838,
    colsample_bytree     = 0.678,
    learning_rate        = 0.040,
    reg_alpha            = 0.001,
    reg_lambda           = 1.674,
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
        test_mask    = (row_pos >= train_end_pos) & (row_pos < test_end_pos)
        test_idx     = panel.index[test_mask]

        # Model A: train on even blocks, val on odd blocks
        train_A_mask = in_train & (block_parity == 0)
        val_A_mask   = in_train & (block_parity == 1)
        # Model B: train on odd blocks, val on even blocks (opposite phase)
        train_B_mask = in_train & (block_parity == 1)
        val_B_mask   = in_train & (block_parity == 0)

        tr_A_idx  = panel.index[train_A_mask & y_all.notna()]
        val_A_idx = panel.index[val_A_mask   & y_all.notna()]
        tr_B_idx  = panel.index[train_B_mask & y_all.notna()]
        val_B_idx = panel.index[val_B_mask   & y_all.notna()]

        if len(tr_A_idx) < 100 or len(val_A_idx) < 10:
            train_end_pos += STEP
            continue

        # Train Model A (even→train, odd→val_ES)
        model_A = xgb.XGBRegressor(**params)
        model_A.fit(X_all.loc[tr_A_idx].values, y_train.loc[tr_A_idx].values,
                     eval_set=[(X_all.loc[val_A_idx].values, y_train.loc[val_A_idx].values)],
                     verbose=False)

        # Train Model B (odd→train, even→val_ES)
        model_B = xgb.XGBRegressor(**params)
        model_B.fit(X_all.loc[tr_B_idx].values, y_train.loc[tr_B_idx].values,
                     eval_set=[(X_all.loc[val_B_idx].values, y_train.loc[val_B_idx].values)],
                     verbose=False)

        best_iter_A = model_A.best_iteration
        best_iter_B = model_B.best_iteration

        if save_models:
            model_dir = OUTPUTS / "models"
            model_dir.mkdir(parents=True, exist_ok=True)
            model_A.save_model(str(model_dir / f"step_{step_n:02d}_A.ubj"))
            model_B.save_model(str(model_dir / f"step_{step_n:02d}_B.ubj"))

        train_dates = dates[:train_end_pos]
        test_dates  = dates[train_end_pos:test_end_pos]

        # Val predictions: A predicts on odd blocks (its val), B predicts on even blocks (its val)
        # Together they cover ALL blocks → continuous val for CMA-ES
        val_ic = float("nan")
        all_val_idx = panel.index[in_train & y_all.notna()]
        if len(all_val_idx) > 0:
            scores_A_val = model_A.predict(X_all.loc[all_val_idx].values)
            scores_B_val = model_B.predict(X_all.loc[all_val_idx].values)
            # For CMA-ES: save both scores, backtest will use them with separate temperatures
            val_ic_A = _daily_ic(scores_A_val, y_all.loc[all_val_idx].values, all_val_idx)
            val_ic_B = _daily_ic(scores_B_val, y_all.loc[all_val_idx].values, all_val_idx)
            val_ic = (val_ic_A + val_ic_B) / 2

            val_df = pd.DataFrame({
                "score_A": scores_A_val,
                "score_B": scores_B_val,
                "score": (scores_A_val + scores_B_val) / 2,
                "label": y_all.loc[all_val_idx].values,
            }, index=all_val_idx)
            val_df["step"]      = step_n
            val_df["split"]     = "val"
            val_df["best_iter"] = best_iter_A
            val_df["val_ic"]    = val_ic
            predictions.append(val_df)

        # Test predictions: both models predict, saved separately
        test_ic = float("nan")
        if len(test_idx) > 0:
            scores_A_test = model_A.predict(X_all.loc[test_idx].values)
            scores_B_test = model_B.predict(X_all.loc[test_idx].values)
            avg_scores = (scores_A_test + scores_B_test) / 2
            test_ic = _daily_ic(avg_scores, y_all.loc[test_idx].values, test_idx)

            test_df = pd.DataFrame({
                "score_A": scores_A_test,
                "score_B": scores_B_test,
                "score": avg_scores,
                "label": y_all.loc[test_idx].values,
            }, index=test_idx)
            test_df["step"]      = step_n
            test_df["split"]     = "test"
            test_df["best_iter"] = best_iter_A
            test_df["val_ic"]    = val_ic
            predictions.append(test_df)

        ic_log.append((step_n, val_ic, val_ic, val_ic, test_ic))
        step_n += 1

        if verbose:
            print(
                f"  Step {step_n:2d}  "
                f"val={val_ic:+.4f}  "
                f"test={test_ic:+.4f}  "
                f"[{test_dates[0].date()} → {test_dates[-1].date()}]  "
                f"iter_A={best_iter_A} iter_B={best_iter_B}"
            )

        train_end_pos += STEP

    if not predictions:
        raise RuntimeError("No predictions produced — check MIN_TRAIN_ROWS vs data length")

    # IC summary
    ic_df = pd.DataFrame(ic_log, columns=["step", "train_ic", "val_ic", "val_cma_ic", "test_ic"])
    if verbose:
        print(f"\n{'='*70}")
        print(f"  {'Mean train IC (in-sample)':35s} {ic_df['train_ic'].mean():+.4f}")
        print(f"  {'Mean val IC (early stop)':35s} {ic_df['val_ic'].mean():+.4f}")
        print(f"  {'Mean test IC (true OOS)':35s} {ic_df['test_ic'].mean():+.4f}")
        print(f"  {'IC stability (val/test corr)':35s} "
              f"{ic_df['val_ic'].corr(ic_df['test_ic']):+.3f}")
        print(f"{'='*70}")

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
