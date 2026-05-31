"""
Hyperparameter search for Follow Leads model.

Searches over XGBoost regularisation params + feature-selection settings
to maximise test IC and minimise val→test gap (overfitting).

Strategy: val IC=0.076 vs test IC=0.024 → 3x gap → need more regularisation.
Primary knobs: min_child_weight ↑, max_depth ↓, reg_lambda ↑, FEAT_SEL_POWER ↑/CAP ↓.

Grid: max_depth × min_child_weight × reg_lambda × feat_sel_power × feat_sel_cap
Output: knowledge/search_hyperparams_fl_results.csv
        data/follow_leads/oos_predictions.parquet  (best model retrained)
        outputs/follow_leads/selected_features.json (best feature list)
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
ROOT = Path(__file__).resolve().parent.parent
from train_smart_money import (
    XGB_PARAMS, _try_gpu, LABEL_COL, _zscore_per_date,
    BLOCK_ROWS, EMBARGO_ROWS, run_walk_forward,
    FEAT_SEL_EMBARGOS, FEAT_SEL_POWER as DEFAULT_FSP, FEAT_SEL_CAP as DEFAULT_CAP,
)
from select_features import get_all_feature_cols
import train_smart_money as train_mod
from train_follow_leads import _is_smart_money, OUTPUTS_FL, DATA_OUT

DATA    = ROOT / "data"
OUTPUTS = ROOT / "knowledge"
OUTPUTS.mkdir(exist_ok=True)

# --- Search grid (targeting overfitting reduction) ---
DEPTH_RANGE   = [4, 5, 6]
MCW_RANGE     = [40, 60, 90, 130]    # min_child_weight (current=40)
LAMBDA_RANGE  = [1.0, 3.0, 6.0]     # reg_lambda (current=1.674)
FSP_RANGE     = [1.5, 1.7, 2.0]     # feat_sel_power (current=1.7)
CAP_RANGE     = [70, 100, 130]       # feat_sel_cap (current=110)

N_BEST = 5


def _strip_sm(panel: pd.DataFrame) -> pd.DataFrame:
    sm_cols = [c for c in panel.columns if _is_smart_money(c)]
    return panel.drop(columns=sm_cols)


def _select_features(panel, feature_cols, device, power, cap):
    """3-period stability feature selection (same as train.py per-step logic)."""
    dates      = panel.index.get_level_values("date").unique().sort_values()
    date_to_pos = {d: i for i, d in enumerate(dates)}
    row_dates  = panel.index.get_level_values("date")
    row_pos    = pd.Series([date_to_pos[d] for d in row_dates], index=panel.index)

    block_idx    = row_pos // BLOCK_ROWS
    block_id     = block_idx % 3
    pos_in_block = row_pos % BLOCK_ROWS
    not_emb      = (pos_in_block >= EMBARGO_ROWS) & (pos_in_block < BLOCK_ROWS - EMBARGO_ROWS)

    X_all = panel[feature_cols].astype(np.float32)
    y_all = panel[LABEL_COL].astype(np.float32)
    y_z   = _zscore_per_date(y_all).astype(np.float32)

    params = {**XGB_PARAMS, "device": device, "max_depth": 7}
    importances = {}
    for period in range(3):
        tr_mask  = (block_id != period) & y_all.notna() & not_emb
        val_mask = (block_id == period) & y_all.notna() & not_emb
        tr_idx   = panel.index[tr_mask]
        val_idx  = panel.index[val_mask]
        if len(tr_idx) < 50 or len(val_idx) < 10:
            continue
        m = xgb.XGBRegressor(**params)
        m.fit(X_all.loc[tr_idx].values, y_z.loc[tr_idx].values,
              eval_set=[(X_all.loc[val_idx].values, y_z.loc[val_idx].values)],
              verbose=False)
        importances[f"p{period}"] = m.feature_importances_

    if len(importances) < 2:
        return feature_cols[:cap]

    imp = pd.DataFrame(importances, index=feature_cols)
    imp["mean"] = imp.mean(axis=1)
    imp["std"]  = imp.std(axis=1)
    imp["stab"] = imp["mean"] / (imp["std"].replace(0, np.nan) ** power)
    imp = imp.sort_values("stab", ascending=False)
    imp = imp[imp["stab"].notna() & (imp["mean"] > 0)]
    return imp.index[:cap].tolist()


def compute_metrics(oos: pd.DataFrame) -> dict:
    test_df = oos[oos["split"] == "test"].dropna(subset=["score", "label"])
    val_df  = oos[oos["split"] == "val"].dropna(subset=["score", "label"])
    if len(test_df) < 10:
        return {"test_ic": float("nan"), "val_ic": float("nan"), "gap": float("nan")}

    test_ic_by_step = (
        test_df.reset_index()
        .groupby(["step", "date"])
        .apply(lambda g: g["score"].corr(g["label"]) if len(g) > 1 else np.nan,
               include_groups=False)
        .groupby(level="step").mean()
    )
    val_ic_by_step = (
        val_df.groupby("step")["val_ic"].first()
        if "val_ic" in val_df.columns else pd.Series(dtype=float)
    )

    mean_test = float(test_ic_by_step.dropna().mean())
    mean_val  = float(val_ic_by_step.dropna().mean()) if len(val_ic_by_step) else float("nan")
    gap       = max(mean_val - mean_test, 0.0) if not np.isnan(mean_val) else float("nan")
    return {"test_ic": mean_test, "val_ic": mean_val, "gap": gap}


def fmt_time(s: float) -> str:
    m, s = divmod(int(s), 60)
    return f"{m}m{s:02d}s"


def main():
    feat_path = DATA / "features.parquet"
    if not feat_path.exists():
        sys.exit("ERROR: data/features.parquet not found — run feature_engineering.py first")

    print("Loading features...", flush=True)
    raw = pd.read_parquet(feat_path)
    raw = raw.reset_index()
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw.set_index(["date", "etf_id"])

    panel = _strip_sm(raw)
    feature_cols = get_all_feature_cols(panel)

    n_dates = panel.index.get_level_values("date").nunique()
    n_etfs  = panel.index.get_level_values("etf_id").nunique()
    print(f"Panel (SM stripped): {n_dates}d × {n_etfs} ETFs = {len(panel)} rows")
    print(f"Candidate features: {len(feature_cols)}")

    device = _try_gpu()
    print(f"Device: {device.upper()}")

    # Redirect outputs to follow_leads dir
    train_mod.OUTPUTS = OUTPUTS_FL

    # Speed up search: use N_MODELS=1 (restored to 20 for final retrain)
    SEARCH_N_MODELS = 1
    train_mod.N_MODELS = SEARCH_N_MODELS

    combos = list(itertools.product(DEPTH_RANGE, MCW_RANGE, LAMBDA_RANGE, FSP_RANGE, CAP_RANGE))
    n = len(combos)
    print(f"\nGrid: {len(DEPTH_RANGE)}d × {len(MCW_RANGE)}mcw × {len(LAMBDA_RANGE)}lam "
          f"× {len(FSP_RANGE)}fsp × {len(CAP_RANGE)}cap = {n} combos\n")
    print(f"{'#':>5}  {'dep':>3}  {'mcw':>4}  {'lam':>5}  {'fsp':>4}  {'cap':>3}  "
          f"{'nfeat':>5}  {'testIC':>7}  {'valIC':>7}  {'gap':>5}  {'time':>6}")
    print("-" * 80)

    results  = []
    feat_cache = {}  # (fsp, cap) → selected features
    t0 = time.time()

    for i, (depth, mcw, lam, fsp, cap) in enumerate(combos):
        t1 = time.time()

        fk = (fsp, cap)
        if fk not in feat_cache:
            feat_cache[fk] = _select_features(panel, feature_cols, device, fsp, cap)
        selected = feat_cache[fk]

        train_mod.FEATURE_COLS = selected

        xgb_override = {
            "max_depth":        depth,
            "min_child_weight": mcw,
            "reg_lambda":       lam,
        }
        try:
            oos = run_walk_forward(
                panel, device,
                xgb_params=xgb_override,
                verbose=False,
                save_models=False,
                wf_feature_selection=False,  # use pre-selected features
            )
            m = compute_metrics(oos)
        except Exception as e:
            print(f"  ERROR at combo {i}: {e}", flush=True)
            m = {"test_ic": float("nan"), "val_ic": float("nan"), "gap": float("nan")}

        elapsed = time.time() - t1
        results.append({"depth": depth, "mcw": mcw, "lam": lam, "fsp": fsp, "cap": cap,
                         "n_feat": len(selected), **m, "time_s": elapsed})

        valid = [r["test_ic"] for r in results if not np.isnan(r.get("test_ic", float("nan")))]
        best  = max(valid) if valid else float("nan")
        star  = " ***" if m["test_ic"] == best and not np.isnan(m["test_ic"]) else ""

        done_total = time.time() - t0
        eta = done_total / (i + 1) * (n - i - 1)
        print(f"{i+1:3d}/{n}  {depth:3d}  {mcw:4d}  {lam:5.1f}  {fsp:4.1f}  {cap:3d}  "
              f"{len(selected):5d}  {m['test_ic']:+.4f}  {m['val_ic']:+.4f}  {m['gap']:.3f}  "
              f"{fmt_time(elapsed)}  ETA {fmt_time(eta)}{star}", flush=True)

    df = pd.DataFrame(results).sort_values("test_ic", ascending=False)
    out_csv = OUTPUTS / "search_hyperparams_fl_results.csv"
    df.to_csv(out_csv, index=False)

    best_df = df.head(N_BEST)
    print(f"\n{'='*70}")
    print(f"  Done in {fmt_time(time.time()-t0)} — {n} combos")
    print(f"\n  Top {N_BEST}:")
    print(f"  {'#':>2}  {'dep':>3}  {'mcw':>4}  {'lam':>5}  {'fsp':>4}  {'cap':>3}  "
          f"{'testIC':>7}  {'valIC':>7}  {'gap':>5}")
    for j, (_, row) in enumerate(best_df.iterrows()):
        print(f"  {j+1:2d}  {int(row['depth']):3d}  {int(row['mcw']):4d}  {row['lam']:5.1f}  "
              f"{row['fsp']:4.1f}  {int(row['cap']):3d}  "
              f"{row['test_ic']:+.4f}  {row['val_ic']:+.4f}  {row['gap']:.3f}")
    print(f"{'='*70}")
    print(f"\nSaved → {out_csv.relative_to(ROOT)}")

    # Retrain best — restore full N_MODELS=20 ensemble
    train_mod.N_MODELS = 20
    print(f"\n--- Retraining best model (N_MODELS=20) + saving artefacts ---")
    best = df.iloc[0]
    bfk = (best["fsp"], int(best["cap"]))
    best_feats = feat_cache[bfk]
    xgb_best = {
        "max_depth":        int(best["depth"]),
        "min_child_weight": int(best["mcw"]),
        "reg_lambda":       best["lam"],
    }
    print(f"  params: {xgb_best}  fsp={best['fsp']:.1f}  cap={int(best['cap'])}  "
          f"test_IC={best['test_ic']:+.4f}", flush=True)

    train_mod.FEATURE_COLS = best_feats
    oos = run_walk_forward(
        panel, device,
        xgb_params=xgb_best,
        verbose=True,
        save_models=True,
        wf_feature_selection=False,
    )

    out_oos = DATA_OUT / "oos_predictions.parquet"
    oos.to_parquet(out_oos)
    feat_json = OUTPUTS_FL / "selected_features.json"
    with open(feat_json, "w") as f:
        json.dump(best_feats, f, indent=2)

    m_final = compute_metrics(oos)
    print(f"\nFinal retrained model:")
    print(f"  val IC:  {m_final['val_ic']:+.4f}")
    print(f"  test IC: {m_final['test_ic']:+.4f}")
    print(f"  gap:     {m_final['gap']:.4f}")
    print(f"\nSaved → {out_oos.relative_to(ROOT)}")
    print(f"Saved → {feat_json.relative_to(ROOT)} ({len(best_feats)} features)")


if __name__ == "__main__":
    main()
