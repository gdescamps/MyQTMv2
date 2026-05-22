"""
Walk-Forward Expanding XGBoost training for MyQTM-ETF.

For each step in the walk-forward (42-day test block):
  1. Feature selection — 5 folds with varying final embargo (30→10d by 5d):
     each fold trains XGB on even blocks / early-stops on odd blocks,
     collects feature importances. Features ranked by mean/std^1.5 across
     folds; top 110 kept.
  2. Train 10 models — varying final embargo (30→12d by 2d) × different
     seeds. Each model: even blocks → train, odd blocks → early stop.
  3. Average 10 model predictions on the 42-day test block.

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

sys.path.insert(0, str(Path(__file__).parent))

DATA    = Path(__file__).parent / "data"
OUTPUTS = Path(__file__).parent / "outputs" / "smart_money"
DATA.mkdir(parents=True, exist_ok=True)
OUTPUTS.mkdir(parents=True, exist_ok=True)

# Walk-forward parameters
# First test ~2011-01 with rolling 5y train window. Panel starts ~2000-01,
# so 11y × 252 ≈ 2772 trading-day positions before the first test. Early
# steps have few ETFs in the cross-section (only those with smart-money
# already online); the IC gate in backtest.py keeps allocation in calm mode
# until the model's rolling test_ic shows sustained positive alpha (~2018).
MIN_TRAIN_ROWS = 2772
TEST_WINDOW    = 21    # ~1 month per test step
STEP           = 21    # refit every ~1 month
BLOCK_ROWS     = 21    # ~1 month alternating blocks for interlaced train/val
EMBARGO_ROWS   = 10    # 10-day gap between train/val blocks (>= label horizon)
ROLLING_WINDOW = 1250  # ~5 years rolling train window

# Feature selection: 10 folds, final embargo varies 30→12 by 2d
FEAT_SEL_EMBARGOS = list(range(30, 10, -2))  # [30, 28, 26, ..., 12]
FEAT_SEL_POWER    = 1.7
FEAT_SEL_CAP      = 110

# Model ensemble: 20 models, final embargo varies 30→11 by 1d
N_MODELS            = 20
MODEL_EMBARGO_START = 30
MODEL_EMBARGO_STEP  = 1   # → [30, 29, 28, ..., 11]

def _load_feature_cols() -> list[str]:
    """Load selected features from select_features.py output, or fallback to defaults."""
    import json
    sel_path = OUTPUTS / "selected_features.json"
    if sel_path.exists():
        with open(sel_path) as f:
            return json.load(f)
    # Fallback if selected_features.json doesn't exist yet
    return [
        "yield_curve", "yc_x_vix",
        "hy_spread", "hy_spread_velocity_20",
        "dxy_ret_20d", "ret_spx_60d",
        "ret_1d", "ret_5d", "ret_20d", "ret_60d", "ret_120d", "ret_250d",
        "mom_ratio_60v120", "mom_ratio_20v120",
        "mom_accel_20dv60d",
        "vol_20d", "vol_60d", "vol_120d",
        "rsi_14", "rsi_21",
        "price_vs_ma50", "price_vs_ma200",
        "ma_50_slope", "ma_100_slope", "ma_200_slope",
        "ma20_vs_ma50", "ma50_vs_ma100", "ma50_vs_ma200", "ma100_vs_ma200",
        "atr_14", "atr_21",
        "drawdown_60", "drawdown_250", "kurtosis_60",
        "volume_z5", "volume_z10", "volume_z20",
        "shares_outstanding_z5", "shares_outstanding_z20", "shares_outstanding_z60",
        "so_cross_20v60", "so_cross_20v120",
        "so_ret_20d", "so_ret_60d",
    ]

FEATURE_COLS = _load_feature_cols()

LABEL_COL = "label"

XGB_PARAMS = dict(
    tree_method          = "hist",
    max_depth            = 6,
    min_child_weight     = 40,
    subsample            = 0.900,
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
        lambda x: (x - x.mean()) / max(x.std(), 1e-8) if len(x) > 1 else 0.0
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


# ── Block masks (reused by feature selection and model training) ─────────
def _block_masks(row_pos):
    """Return block_parity and not_embargoed masks."""
    block_idx    = row_pos // BLOCK_ROWS
    block_parity = block_idx % 2
    pos_in_block = row_pos % BLOCK_ROWS
    not_embargoed = (pos_in_block >= EMBARGO_ROWS) & (pos_in_block < BLOCK_ROWS - EMBARGO_ROWS)
    return block_parity, not_embargoed


# ── Feature selection (per WF step) ─────────────────────────────────────
def _select_features_for_step(panel, row_pos, y_all,
                              train_start_pos, test_start_pos,
                              all_feature_cols, device):
    """5-fold feature selection with varying final embargo (30→10d by 5d).

    Each fold trains XGB on even blocks / early-stops on odd blocks using
    data up to (test_start - final_embargo). Features ranked by
    mean / std^power across folds; top FEAT_SEL_CAP kept.
    """
    y_z = _zscore_per_date(y_all).astype(np.float32)
    X = panel[all_feature_cols].astype(np.float32).replace([np.inf, -np.inf], np.nan)
    block_parity, not_embargoed = _block_masks(row_pos)

    importances = {}
    for fold_i, final_emb in enumerate(FEAT_SEL_EMBARGOS):
        usable_end = test_start_pos - final_emb
        if usable_end <= train_start_pos + 100:
            continue

        in_usable = (row_pos >= train_start_pos) & (row_pos < usable_end)
        tr_mask  = in_usable & (block_parity == 0) & not_embargoed & y_all.notna()
        val_mask = in_usable & (block_parity == 1) & not_embargoed & y_all.notna()

        tr_idx  = panel.index[tr_mask]
        val_idx = panel.index[val_mask]
        if len(tr_idx) < 50 or len(val_idx) < 10:
            continue

        params = {**XGB_PARAMS, "device": device, "max_depth": 7, "seed": fold_i}
        model = xgb.XGBRegressor(**params)
        model.fit(X.loc[tr_idx].values, y_z.loc[tr_idx].values,
                  eval_set=[(X.loc[val_idx].values, y_z.loc[val_idx].values)],
                  verbose=False)
        importances[f"fold_{fold_i}"] = model.feature_importances_

    if len(importances) < 2:
        return all_feature_cols[:FEAT_SEL_CAP]

    imp_df = pd.DataFrame(importances, index=all_feature_cols)
    imp_df["mean"] = imp_df.mean(axis=1)
    imp_df["std"]  = imp_df.std(axis=1)
    imp_df["stability"] = imp_df["mean"] / (imp_df["std"].replace(0, np.nan) ** FEAT_SEL_POWER)
    imp_df = imp_df.sort_values("stability", ascending=False)
    imp_df = imp_df[imp_df["stability"].notna() & (imp_df["mean"] > 0)]
    return imp_df.index[:FEAT_SEL_CAP].tolist()


# ── Main walk-forward loop ───────────────────────────────────────────────
def run_walk_forward(
    panel: pd.DataFrame,
    device: str,
    xgb_params: dict | None = None,
    verbose: bool = True,
    save_models: bool = True,
    wf_feature_selection: bool = True,
) -> pd.DataFrame:
    params_base = {**XGB_PARAMS, **(xgb_params or {})}
    params_base["device"] = device

    dates = panel.index.get_level_values("date").unique().sort_values()
    n = len(dates)

    # Feature pool
    available_cols = [c for c in FEATURE_COLS if c in panel.columns]
    if wf_feature_selection:
        from select_features import get_all_feature_cols
        all_candidate_cols = get_all_feature_cols(panel)
    else:
        all_candidate_cols = available_cols

    X_all   = panel[available_cols].astype(np.float32).replace([np.inf, -np.inf], np.nan)
    y_all   = panel[LABEL_COL].astype(np.float32)
    y_z     = _zscore_per_date(y_all).astype(np.float32)

    date_to_pos = {d: i for i, d in enumerate(dates)}
    row_dates   = panel.index.get_level_values("date")
    row_pos     = pd.Series([date_to_pos[d] for d in row_dates], index=panel.index)

    block_parity, not_embargoed = _block_masks(row_pos)

    # Model embargo schedule: 30, 28, 26, …, 12
    model_embargos = list(range(MODEL_EMBARGO_START,
                                MODEL_EMBARGO_START - N_MODELS * MODEL_EMBARGO_STEP,
                                -MODEL_EMBARGO_STEP))

    predictions    = []
    ic_log         = []
    importance_log = []
    step_n         = 0

    test_start_pos = MIN_TRAIN_ROWS
    while test_start_pos + TEST_WINDOW <= n:
        test_end_pos    = min(test_start_pos + TEST_WINDOW, n)
        train_start_pos = max(0, test_start_pos - ROLLING_WINDOW)

        test_mask = (row_pos >= test_start_pos) & (row_pos < test_end_pos)
        test_idx  = panel.index[test_mask]

        # ── 1. Feature selection (5 folds, varying final embargo) ────────
        if wf_feature_selection:
            available_cols = _select_features_for_step(
                panel, row_pos, y_all,
                train_start_pos, test_start_pos,
                all_candidate_cols, device,
            )
            X_all = panel[available_cols].astype(np.float32).replace([np.inf, -np.inf], np.nan)
            if verbose:
                print(f"  [feat_sel step {step_n}] {len(available_cols)}/"
                      f"{len(all_candidate_cols)} features selected", flush=True)

        # ── 2. Train N_MODELS (varying final embargo + seed) ─────────────
        trained_models = []
        best_iters     = []

        for m_i, final_emb in enumerate(model_embargos):
            usable_end = test_start_pos - final_emb
            if usable_end <= train_start_pos + 100:
                continue

            in_usable = (row_pos >= train_start_pos) & (row_pos < usable_end)
            tr_mask   = in_usable & (block_parity == 0) & not_embargoed & y_all.notna()
            val_mask  = in_usable & (block_parity == 1) & not_embargoed & y_all.notna()

            tr_idx  = panel.index[tr_mask]
            val_idx = panel.index[val_mask]
            if len(tr_idx) < 100 or len(val_idx) < 10:
                continue

            params = {**params_base, "seed": m_i}
            model  = xgb.XGBRegressor(**params)
            model.fit(
                X_all.loc[tr_idx].values, y_z.loc[tr_idx].values,
                eval_set=[(X_all.loc[val_idx].values, y_z.loc[val_idx].values)],
                verbose=False,
            )
            trained_models.append(model)
            best_iters.append(model.best_iteration)

            if save_models:
                model_dir = OUTPUTS / "models"
                model_dir.mkdir(parents=True, exist_ok=True)
                model.save_model(str(model_dir / f"step_{step_n:02d}_m{m_i}.ubj"))

        if not trained_models:
            test_start_pos += STEP
            continue

        avg_iter = int(np.mean(best_iters))

        # ── 3. Val predictions (conservative cutoff: embargo = max) ──────
        # Only odd blocks (early-stop side, not used for fitting trees) AND only
        # non-embargoed positions are kept — that's the cleanest semi-OOS subset
        # of the training window. Including even blocks would mix in-sample
        # predictions and inflate val_ic vs the true 21d test IC.
        common_usable_end = test_start_pos - MODEL_EMBARGO_START
        in_common = (row_pos >= train_start_pos) & (row_pos < common_usable_end)
        common_val_mask = in_common & y_all.notna() & (block_parity == 1) & not_embargoed
        common_val_idx  = panel.index[common_val_mask]

        val_ic = float("nan")
        if len(common_val_idx) > 0:
            val_scores_stack = [m.predict(X_all.loc[common_val_idx].values)
                                for m in trained_models]
            val_scores = np.mean(val_scores_stack, axis=0)
            val_ic = _daily_ic(val_scores, y_all.loc[common_val_idx].values,
                               common_val_idx)

            val_df = pd.DataFrame({
                "score": val_scores,
                "label": y_all.loc[common_val_idx].values,
            }, index=common_val_idx)
            val_df["step"]      = step_n
            val_df["split"]     = "val"
            val_df["best_iter"] = avg_iter
            val_df["val_ic"]    = val_ic
            val_df["test_ic"]   = np.nan
            predictions.append(val_df)

        # ── 4. Test predictions (average of N models) ────────────────────
        test_ic = float("nan")
        if len(test_idx) > 0:
            test_scores_stack = [m.predict(X_all.loc[test_idx].values)
                                 for m in trained_models]
            test_scores = np.mean(test_scores_stack, axis=0)
            test_ic = _daily_ic(test_scores, y_all.loc[test_idx].values, test_idx)

            test_df = pd.DataFrame({
                "score": test_scores,
                "label": y_all.loc[test_idx].values,
            }, index=test_idx)
            test_df["step"]      = step_n
            test_df["split"]     = "test"
            test_df["best_iter"] = avg_iter
            test_df["val_ic"]    = val_ic
            test_df["test_ic"]   = test_ic
            predictions.append(test_df)

        ic_log.append((step_n, val_ic, test_ic))

        # Feature importance (SHAP from first model)
        test_date = dates[test_start_pos] if test_start_pos < n else dates[-1]
        if len(test_idx) > 0:
            dtest = xgb.DMatrix(X_all.loc[test_idx].values,
                                feature_names=available_cols)
            shap_vals = trained_models[0].get_booster().predict(
                dtest, pred_contribs=True)
            mean_abs_shap = np.abs(shap_vals[:, :-1]).mean(axis=0)
            imp = dict(zip(available_cols, mean_abs_shap))
        else:
            imp = dict(zip(available_cols,
                           trained_models[0].feature_importances_))
        importance_log.append({"step": step_n, "date": test_date, **imp})

        step_n += 1
        if verbose:
            test_dates = dates[test_start_pos:test_end_pos]
            print(
                f"  Step {step_n:2d}  "
                f"val_ic={val_ic:.4f}  test_ic={test_ic:.4f}  "
                f"[{test_dates[0].date()} → {test_dates[-1].date()}]  "
                f"models={len(trained_models)}  avg_iter={avg_iter}"
            )

        test_start_pos += STEP

    if not predictions:
        raise RuntimeError(
            "No predictions produced — check MIN_TRAIN_ROWS vs data length")

    # IC summary
    ic_df = pd.DataFrame(ic_log, columns=["step", "val_ic", "test_ic"])
    if verbose:
        print(f"\n{'='*70}")
        print(f"  {'Mean val IC':35s} {ic_df['val_ic'].mean():+.4f}")
        print(f"  {'Mean test IC (true OOS)':35s} {ic_df['test_ic'].mean():+.4f}")
        print(f"  {'IC stability (val/test corr)':35s} "
              f"{ic_df['val_ic'].corr(ic_df['test_ic']):+.3f}")
        print(f"{'='*70}")

    # Save feature importances over time
    if importance_log:
        imp_df = pd.DataFrame(importance_log).set_index("date")
        imp_df.to_parquet(OUTPUTS / "feature_importances.parquet")
        if verbose:
            print(f"  Saved feature_importances.parquet "
                  f"({len(imp_df)} steps × {len(imp_df.columns)-1} features)")

    return pd.concat(predictions).sort_index()


def main():
    feat_path = DATA / "features.parquet"
    if not feat_path.exists():
        sys.exit(f"ERROR: {feat_path} not found — run feature_engineering.py first")

    print("Loading features...")
    panel = pd.read_parquet(feat_path)
    panel = panel.reset_index()
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.set_index(["date", "etf_id"])
    panel["etf_id_code"] = pd.Categorical(
        panel.index.get_level_values("etf_id")).codes.astype(np.int32)

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
        f"Feature selection: {len(FEAT_SEL_EMBARGOS)} folds (embargo "
        f"{FEAT_SEL_EMBARGOS[0]}→{FEAT_SEL_EMBARGOS[-1]}d), "
        f"mean/std^{FEAT_SEL_POWER}, top {FEAT_SEL_CAP}\n"
        f"Model ensemble: {N_MODELS} models (embargo "
        f"{MODEL_EMBARGO_START}→{MODEL_EMBARGO_START - (N_MODELS-1)*MODEL_EMBARGO_STEP}d "
        f"by {MODEL_EMBARGO_STEP}d, each with different seed)\n"
    )

    oos = run_walk_forward(
        panel, device, wf_feature_selection=True, save_models=True,
    )

    out = DATA / "oos_predictions.parquet"
    oos.to_parquet(out)

    test_oos = oos[oos["split"] == "test"].dropna(subset=["score", "label"])
    mean_test_ic = test_oos.groupby(
        test_oos.index.get_level_values("date")
    ).apply(lambda x: x["score"].corr(x["label"])).mean()
    val_oos = oos[oos["split"] == "val"].dropna(subset=["score", "label"])
    mean_val_ic = val_oos.groupby(
        val_oos.index.get_level_values("date")
    ).apply(lambda x: x["score"].corr(x["label"])).mean()

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
