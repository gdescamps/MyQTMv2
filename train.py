"""
Walk-Forward Expanding XGBoost training for MyQTM-ETF.

For each step in the walk-forward:
  - Split the training period into interlaced monthly blocks (50/50):
      Even blocks → XGBoost training
      Odd blocks  → Early stopping + CMA-ES optimisation (in backtest.py)
  - Predict on odd blocks (split="val") → used by CMA-ES in backtest.py
  - Predict on next TEST_WINDOW days (split="test") → true OOS backtest

Interlaced split rationale:
  - Val scores are OOS from XGBoost (model never saw those blocks) → unbiased
  - Val covers all market regimes in the training period (not just recent)
  - CMA-ES params found on val generalise better to the future test period

Output: data/oos_predictions.parquet
  MultiIndex: (date, etf_id)
  Columns:    score, label, step, split ("val" | "test"), best_iter

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
TEST_WINDOW    = 125   # ~6 months per test step (non-overlapping with STEP)
STEP           = 125   # refit every ~6 months
BLOCK_ROWS     = 21    # alternating block size for interlaced train/val split (~1 month)

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

    available_cols = [c for c in FEATURE_COLS if c in panel.columns]
    X_all = panel[available_cols].astype(np.float32)
    y_all = panel[LABEL_COL].astype(np.float32)

    # Map each row to its date position (0-indexed integer)
    date_to_pos = {d: i for i, d in enumerate(dates)}
    row_dates   = panel.index.get_level_values("date")
    row_pos     = pd.Series([date_to_pos[d] for d in row_dates], index=panel.index)

    predictions = []
    step_n = 0

    train_end_pos = MIN_TRAIN_ROWS
    while train_end_pos + TEST_WINDOW <= n:
        test_end_pos = min(train_end_pos + TEST_WINDOW, n)

        # --- Interlaced 50/50 split within training period ---
        # Block index = date_position // BLOCK_ROWS
        # Even blocks → XGBoost train | Odd blocks → val (early stop + CMA-ES)
        in_train   = row_pos < train_end_pos
        block_parity = (row_pos // BLOCK_ROWS) % 2

        train_mask = in_train & (block_parity == 0)
        val_mask   = in_train & (block_parity == 1)
        test_mask  = (row_pos >= train_end_pos) & (row_pos < test_end_pos)

        tr_idx  = panel.index[train_mask & y_all.notna()]
        val_idx = panel.index[val_mask   & y_all.notna()]
        test_idx = panel.index[test_mask]

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

        # Save model for this step (up to early-stop iteration)
        model_dir = OUTPUTS / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        model.save_model(str(model_dir / f"step_{step_n:02d}.ubj"))

        train_dates = dates[:train_end_pos]
        test_dates  = dates[train_end_pos:test_end_pos]

        # --- Predict on val blocks → for CMA-ES in backtest.py ---
        if len(val_idx) > 0:
            val_scores = model.predict(X_all.loc[val_idx].values)
            val_df = pd.DataFrame(
                {"score": val_scores, "label": y_all.loc[val_idx].values},
                index=val_idx,
            )
            val_df["step"]      = step_n
            val_df["split"]     = "val"
            val_df["best_iter"] = best_iter
            predictions.append(val_df)

        # --- Predict on test window → true OOS backtest ---
        if len(test_idx) > 0:
            test_scores = model.predict(X_all.loc[test_idx].values)
            test_df = pd.DataFrame(
                {"score": test_scores, "label": y_all.loc[test_idx].values},
                index=test_idx,
            )
            test_df["step"]      = step_n
            test_df["split"]     = "test"
            test_df["best_iter"] = best_iter
            predictions.append(test_df)

        step_n += 1
        print(
            f"  Step {step_n:2d}  "
            f"train [{dates[0].date()} → {train_dates[-1].date()}]  "
            f"val_blocks={val_mask.sum()}rows  "
            f"test [{test_dates[0].date()} → {test_dates[-1].date()}]  "
            f"best_iter={best_iter}"
        )

        train_end_pos += STEP

    if not predictions:
        raise RuntimeError("No predictions produced — check MIN_TRAIN_ROWS vs data length")

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
        f"\nWalk-Forward: MIN_TRAIN={MIN_TRAIN_ROWS}d  "
        f"TEST={TEST_WINDOW}d  STEP={STEP}d  BLOCK={BLOCK_ROWS}d\n"
    )

    oos = run_walk_forward(panel, device)

    out = DATA / "oos_predictions.parquet"
    oos.to_parquet(out)

    # IC on test predictions only (true OOS metric)
    test_oos = oos[oos["split"] == "test"].dropna(subset=["score", "label"])
    ic = test_oos.groupby(test_oos.index.get_level_values("date")).apply(
        lambda x: x["score"].corr(x["label"])
    ).mean()

    print(f"\nOOS predictions saved → {out.name}")
    print(f"  val rows:  {(oos['split']=='val').sum()}")
    print(f"  test rows: {(oos['split']=='test').sum()}")
    print(f"Mean daily IC (test scores vs label): {ic:.4f}")
    print(
        f"Test date range: "
        f"{test_oos.index.get_level_values('date').min().date()} → "
        f"{test_oos.index.get_level_values('date').max().date()}"
    )


if __name__ == "__main__":
    main()
