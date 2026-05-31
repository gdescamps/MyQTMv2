"""
Follow Leads — calm-regime XGBoost (a.k.a. the "no smart-money" twin).

Same K-fold walk-forward methodology as train.py (Smart Money), same
hyperparameters, same label (forward 10d return), same embargo scheme
(BLOCK_ROWS=21, EMBARGO_ROWS=10, FEAT_SEL_EMBARGOS=30→12d, 20-model
ensemble with FEAT_SEL_EMBARGOS=30→11d, seeds 0..19).

The only difference: **smart-money columns are stripped from the panel**
before training. Follow Leads predicts forward 10d return cross-sectionally
using only technical / macro / cross-sectional features. The model is
intended as a sharper calm-mode allocator (top-3 by predicted return) than
the naive top-3 rolling 252d Sharpe heuristic, without any dependency on
smart-money signal.

Output: data/follow_leads/oos_predictions.parquet
        outputs/follow_leads/feature_importances.parquet
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
ROOT = Path(__file__).resolve().parent.parent
import train_smart_money as train  # noqa: E402

DATA       = ROOT / "data"
DATA_OUT   = DATA / "follow_leads"
OUTPUTS_FL = ROOT / "outputs" / "follow_leads"
DATA_OUT.mkdir(parents=True, exist_ok=True)
OUTPUTS_FL.mkdir(parents=True, exist_ok=True)

# Smart-money columns to strip from the panel (everything XGBoost would otherwise
# learn from iShares shares-outstanding flows).
# Prefix filter catches primary SO columns; substring filter catches derived
# interaction features like mom_x_vol_x_so_20d, sharpe_x_so_20d, so_vs_univ,
# so_price_divergence that were leaking SM signal into FL.
SMART_MONEY_PREFIXES = (
    "shares_outstanding_z", "so_cross_", "so_ret_", "so_x_", "so_breadth",
    "so_vs_", "so_price_",
)
SMART_MONEY_SUBSTRINGS = ("_x_so_", "_so_20d", "_so_60d")


def _is_smart_money(col: str) -> bool:
    if any(col.startswith(p) for p in SMART_MONEY_PREFIXES):
        return True
    return any(sub in col for sub in SMART_MONEY_SUBSTRINGS)


def main():
    feat_path = DATA / "features.parquet"
    if not feat_path.exists():
        sys.exit("ERROR: data/features.parquet not found — run feature_engineering.py first")

    print("Loading features...")
    panel = pd.read_parquet(feat_path)
    panel = panel.reset_index()
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.set_index(["date", "etf_id"])
    panel["etf_id_code"] = pd.Categorical(
        panel.index.get_level_values("etf_id")).codes.astype(np.int32)

    sm_cols = [c for c in panel.columns if _is_smart_money(c)]
    panel = panel.drop(columns=sm_cols)
    print(f"Dropped {len(sm_cols)} smart-money columns (kept {panel.shape[1]} cols)")

    # Override label: predict forward Sharpe (ret_10d_fwd / vol_10d_fwd) instead of
    # raw return — aligns with the FL objective of beating the top-3 Sharpe heuristic.
    if "sharpe_10d_fwd" in panel.columns:
        panel["label"] = panel["sharpe_10d_fwd"]
        print("Label overridden → sharpe_10d_fwd (risk-adjusted forward return)")
    print(f"Panel: {panel.index.get_level_values('date').nunique()} dates × "
          f"{panel.index.get_level_values('etf_id').nunique()} ETFs = {len(panel)} rows")

    device = train._try_gpu()
    print(f"Device: {device.upper()}")
    print(
        f"\nFollow Leads — Walk-Forward: MIN_TRAIN={train.MIN_TRAIN_ROWS}d  "
        f"TEST={train.TEST_WINDOW}d  STEP={train.STEP}d  BLOCK={train.BLOCK_ROWS}d  "
        f"EMBARGO={train.EMBARGO_ROWS}d\n"
        f"Feature selection: {len(train.FEAT_SEL_EMBARGOS)} folds (embargo "
        f"{train.FEAT_SEL_EMBARGOS[0]}→{train.FEAT_SEL_EMBARGOS[-1]}d), "
        f"mean/std^{train.FEAT_SEL_POWER}, top {train.FEAT_SEL_CAP}\n"
        f"Model ensemble: {train.N_MODELS} models (embargo "
        f"{train.MODEL_EMBARGO_START}→"
        f"{train.MODEL_EMBARGO_START - (train.N_MODELS-1)*train.MODEL_EMBARGO_STEP}d "
        f"by {train.MODEL_EMBARGO_STEP}d, each with different seed)\n"
    )

    # Redirect train.py side-effect outputs (feature_importances.parquet,
    # per-step model .ubj files) to outputs/follow_leads/ so the Smart Money
    # artefacts under outputs/ stay untouched.
    train.OUTPUTS = OUTPUTS_FL

    oos = train.run_walk_forward(
        panel, device,
        wf_feature_selection=True,
        save_models=True,   # save model files to outputs/follow_leads/models/ for live inference
    )

    out = DATA_OUT / "oos_predictions.parquet"
    oos.to_parquet(out)

    test_oos = oos[oos["split"] == "test"].dropna(subset=["score", "label"])
    mean_test_ic = test_oos.groupby(
        test_oos.index.get_level_values("date")
    ).apply(lambda x: x["score"].corr(x["label"])).mean()
    val_oos = oos[oos["split"] == "val"].dropna(subset=["score", "label"])
    mean_val_ic = val_oos.groupby(
        val_oos.index.get_level_values("date")
    ).apply(lambda x: x["score"].corr(x["label"])).mean()

    print(f"\nSaved → {out.relative_to(ROOT)}")
    print(f"  val rows:       {(oos['split']=='val').sum()}")
    print(f"  test rows:      {(oos['split']=='test').sum()}")
    print(f"  Mean val IC:    {mean_val_ic:+.4f}")
    print(f"  Mean test IC:   {mean_test_ic:+.4f}")


if __name__ == "__main__":
    main()
