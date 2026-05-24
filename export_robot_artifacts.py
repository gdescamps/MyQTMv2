"""
Export last-step model artifacts for the live trading robot.

Creates outputs/robot/ with:
  sm_models/          20 XGBoost .ubj files (last SM step)
  fl_models/          20 XGBoost .ubj files (last FL step)
  sm_features.json    feature list used in last SM step
  fl_features.json    feature list used in last FL step
  ic_gate_state.json  IC EMA values, thresholds, gate on/off
  last_step_info.json metadata (step number, test date range)

Usage:
    python export_robot_artifacts.py
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from backtest import (
    IC_GATE_THRESHOLD, IC_GATE_SPAN, IC_GATE_MIN_STEPS,
    FL_IC_GATE_THRESHOLD, FL_IC_GATE_SPAN, FL_IC_GATE_MIN_STEPS,
)

DATA = Path(__file__).parent / "data"
OUTPUTS = Path(__file__).parent / "outputs"
ROBOT_DIR = OUTPUTS / "robot"


def _compute_ic_per_step(oos: pd.DataFrame) -> pd.Series:
    test = oos[oos["split"] == "test"].dropna(subset=["score", "label"]).reset_index()
    return (
        test.groupby("step")
        .apply(lambda g: g.groupby("date")
               .apply(lambda x: x["score"].corr(x["label"]) if len(x) > 1 else np.nan,
                      include_groups=False).mean(),
               include_groups=False)
    )


def _extract_last_features(importances_path: Path) -> list[str]:
    """Extract feature list from last step's importances."""
    if not importances_path.exists():
        return []
    imp = pd.read_parquet(importances_path)
    last_step = imp["step"].max()
    last_row = imp[imp["step"] == last_step].drop(columns=["step"]).iloc[0]
    return [f for f in last_row.index if last_row[f] > 0]


def export():
    # Clean + create output dir
    if ROBOT_DIR.exists():
        shutil.rmtree(ROBOT_DIR)
    ROBOT_DIR.mkdir(parents=True)

    # --- Load OOS predictions ---
    sm_oos = pd.read_parquet(DATA / "oos_predictions.parquet").reset_index()
    fl_oos = pd.read_parquet(DATA / "follow_leads" / "oos_predictions.parquet").reset_index()

    sm_oos["date"] = pd.to_datetime(sm_oos["date"])
    fl_oos["date"] = pd.to_datetime(fl_oos["date"])

    # --- IC gate state ---
    sm_ic = _compute_ic_per_step(sm_oos)
    fl_ic = _compute_ic_per_step(fl_oos)

    all_steps = sorted(sm_oos["step"].unique())
    sm_ic_ema = sm_ic.reindex(all_steps).shift(1).ewm(span=IC_GATE_SPAN, min_periods=1).mean()
    fl_ic_ema = fl_ic.reindex(all_steps).shift(1).ewm(span=FL_IC_GATE_SPAN, min_periods=1).mean()

    last_step = max(all_steps)
    sm_gate_val = float(sm_ic_ema.get(last_step, 0))
    fl_gate_val = float(fl_ic_ema.get(last_step, 0))
    sm_gate_open = last_step >= IC_GATE_MIN_STEPS and sm_gate_val > IC_GATE_THRESHOLD
    fl_gate_open = last_step >= FL_IC_GATE_MIN_STEPS and fl_gate_val > FL_IC_GATE_THRESHOLD

    ic_state = {
        "smart_money": {
            "ic_ema": sm_gate_val,
            "threshold": IC_GATE_THRESHOLD,
            "span": IC_GATE_SPAN,
            "min_steps": IC_GATE_MIN_STEPS,
            "gate_open": sm_gate_open,
            "last_5_ic": {int(k): float(v) for k, v in sm_ic.tail().items()},
        },
        "follow_leads": {
            "ic_ema": fl_gate_val,
            "threshold": FL_IC_GATE_THRESHOLD,
            "span": FL_IC_GATE_SPAN,
            "min_steps": FL_IC_GATE_MIN_STEPS,
            "gate_open": fl_gate_open,
            "last_5_ic": {int(k): float(v) for k, v in fl_ic.tail().items()},
        },
        "all_sm_ic": {int(k): float(v) for k, v in sm_ic.items()},
        "all_fl_ic": {int(k): float(v) for k, v in fl_ic.items()},
    }
    with open(ROBOT_DIR / "ic_gate_state.json", "w") as f:
        json.dump(ic_state, f, indent=2)
    print(f"IC gate: SM ema={sm_gate_val:.4f} ({'OPEN' if sm_gate_open else 'CLOSED'})  "
          f"FL ema={fl_gate_val:.4f} ({'OPEN' if fl_gate_open else 'CLOSED'})")

    # --- Last step info ---
    last_test = sm_oos[(sm_oos["step"] == last_step) & (sm_oos["split"] == "test")]
    step_info = {
        "last_step": int(last_step),
        "n_steps": len(all_steps),
        "test_start": str(last_test["date"].min().date()),
        "test_end": str(last_test["date"].max().date()),
        "n_models": 20,
    }
    with open(ROBOT_DIR / "last_step_info.json", "w") as f:
        json.dump(step_info, f, indent=2)
    print(f"Last step: {last_step} ({step_info['test_start']} -> {step_info['test_end']})")

    # --- Copy last-step models ---
    for model_type, src_dir in [("sm", "smart_money"), ("fl", "follow_leads")]:
        dst = ROBOT_DIR / f"{model_type}_models"
        dst.mkdir()
        src = OUTPUTS / src_dir / "models"
        copied = 0
        for i in range(20):
            fname = f"step_{last_step}_m{i}.ubj"
            src_file = src / fname
            if src_file.exists():
                shutil.copy2(src_file, dst / f"model_{i}.ubj")
                copied += 1
        print(f"  {model_type.upper()} models: {copied}/20 copied to {dst.relative_to(OUTPUTS.parent)}")

    # --- Copy feature lists (saved by train_smart_money.py / train_follow_leads.py) ---
    sm_feat_src = OUTPUTS / "smart_money" / "last_step_features.json"
    fl_feat_src = OUTPUTS / "follow_leads" / "last_step_features.json"

    for src, dst_name, label in [
        (sm_feat_src, "sm_features.json", "SM"),
        (fl_feat_src, "fl_features.json", "FL"),
    ]:
        if src.exists():
            shutil.copy2(src, ROBOT_DIR / dst_name)
            with open(src) as f:
                n = len(json.load(f))
            print(f"  {label} features: {n}")
        else:
            # Fallback: extract from importances (non-zero only, may differ from model)
            alt = _extract_last_features(
                src.parent / "feature_importances.parquet")
            with open(ROBOT_DIR / dst_name, "w") as f:
                json.dump(alt, f, indent=2)
            print(f"  {label} features: {len(alt)} (from importances, may be incomplete)")

    print(f"\nAll artifacts exported to {ROBOT_DIR.relative_to(Path(__file__).parent)}/")


if __name__ == "__main__":
    export()
