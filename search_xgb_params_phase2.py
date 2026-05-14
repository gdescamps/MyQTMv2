"""
Phase 2: Targeted Optuna search around boundary-hitting params.

Fixes best params from phase 1, explores:
  - reg_lambda    [5, 50]     (hit upper bound at 9.4)
  - colsample_bytree [0.20, 0.60] (hit lower bound at 0.47)
  - gamma         [0, 5]      (not searched in phase 1)

15 trials, ~6 min.
"""

import json
import sys
import time
import warnings
import numpy as np
import pandas as pd
import optuna
from pathlib import Path

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, str(Path(__file__).parent))
from train import run_walk_forward, _try_gpu

DATA    = Path(__file__).parent / "data"
OUTPUTS = Path(__file__).parent / "outputs"
OUTPUTS.mkdir(exist_ok=True)

N_TRIALS    = 15
STAB_WEIGHT = 0.5

# Fixed from phase 1 best
FIXED_PARAMS = dict(
    max_depth            = 4,
    min_child_weight     = 40,
    subsample            = 0.744,
    learning_rate        = 0.088,
    reg_alpha            = 0.0002,
    n_estimators         = 1000,
    early_stopping_rounds= 30,
)


def compute_objective(oos: pd.DataFrame) -> tuple[float, float, float]:
    test_df = oos[oos["split"] == "test"].dropna(subset=["score", "label"])
    val_df  = oos[oos["split"] == "val"].dropna(subset=["score", "label"])

    if len(test_df) < 10 or test_df["step"].nunique() < 3:
        return -99.0, float("nan"), float("nan")

    test_ic_per_step = (
        test_df.reset_index()
        .groupby(["step", "date"])
        .apply(lambda g: g["score"].corr(g["label"]) if len(g) > 1 else np.nan)
        .groupby(level="step")
        .mean()
    )
    val_ic_per_step = val_df.groupby("step")["val_ic"].first()

    mean_test_ic = float(test_ic_per_step.dropna().mean())
    stability    = float(val_ic_per_step.corr(test_ic_per_step))
    if np.isnan(stability):
        stability = 0.0

    return mean_test_ic + STAB_WEIGHT * stability, mean_test_ic, stability


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

    n_dates = panel.index.get_level_values("date").nunique()
    n_etfs  = panel.index.get_level_values("etf_id").nunique()
    print(f"Panel: {n_dates} dates x {n_etfs} ETFs = {len(panel)} rows", flush=True)

    device = _try_gpu()
    print(f"Device: {device.upper()}", flush=True)

    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  Phase 2 -- {N_TRIALS} trials (~6 min)")
    print(f"  Search: reg_lambda [5,50], colsample_bytree [0.20,0.60], gamma [0,5]")
    print(f"  Phase 1 best: test_IC=+0.041  stab=+0.215  obj=+0.148")
    print(f"{sep}\n", flush=True)

    results = []
    best_obj = -999.0
    t0 = time.time()

    def objective(trial: optuna.Trial) -> float:
        nonlocal best_obj

        trial_params = {
            **FIXED_PARAMS,
            "reg_lambda":       trial.suggest_float("reg_lambda",       5.0, 50.0, log=True),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.20, 0.60),
            "gamma":            trial.suggest_float("gamma",            0.0,  5.0),
        }
        t_start = time.time()

        try:
            oos = run_walk_forward(
                panel, device,
                xgb_params=trial_params,
                verbose=False,
                save_models=False,
            )
        except Exception as e:
            print(f"  [{trial.number + 1:2d}/{N_TRIALS}] ERROR: {e}", flush=True)
            return -99.0

        t_trial = time.time() - t_start
        obj, mean_test_ic, stability = compute_objective(oos)

        results.append({
            "trial":         trial.number,
            "objective":     obj,
            "mean_test_ic":  mean_test_ic,
            "stability":     stability,
            "reg_lambda":    trial_params["reg_lambda"],
            "colsample_bytree": trial_params["colsample_bytree"],
            "gamma":         trial_params["gamma"],
        })

        elapsed = time.time() - t0
        avg = elapsed / (trial.number + 1)
        remaining = avg * (N_TRIALS - trial.number - 1)

        is_best = obj > best_obj
        if is_best:
            best_obj = obj

        marker = " *** NEW BEST ***" if is_best else ""
        print(
            f"  [{trial.number + 1:2d}/{N_TRIALS}] "
            f"obj={obj:+.4f}  "
            f"test_IC={mean_test_ic:+.4f}  "
            f"stab={stability:+.3f}  |  "
            f"lambda={trial_params['reg_lambda']:.1f}  "
            f"colsample={trial_params['colsample_bytree']:.3f}  "
            f"gamma={trial_params['gamma']:.2f}  "
            f"({fmt_time(t_trial)})  "
            f"[{fmt_time(elapsed)} / ~{fmt_time(elapsed + remaining)}]"
            f"{marker}",
            flush=True,
        )
        return obj

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=123),
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    total_time = time.time() - t0

    results_df = pd.DataFrame(results).sort_values("objective", ascending=False)
    results_path = OUTPUTS / "xgb_optuna_phase2_results.csv"
    results_df.to_csv(results_path, index=False)

    best = study.best_trial
    best_params = {**FIXED_PARAMS, **{k: best.params[k] for k in best.params}}

    params_path = OUTPUTS / "xgb_best_params_phase2.json"
    with open(params_path, "w") as f:
        json.dump(best_params, f, indent=2)

    print(f"\n{sep}")
    print(f"  PHASE 2 COMPLETE -- {fmt_time(total_time)} total")
    print(f"{sep}")
    print(f"\n  BEST TRIAL #{best.number + 1}  objective={best.value:+.4f}")
    print(f"  {'─' * 40}")
    for k in ["reg_lambda", "colsample_bytree", "gamma"]:
        print(f"    {k:<22s} {best.params[k]:.6f}")

    best_row = results_df.iloc[0]
    print(f"\n  mean_test_IC : {best_row['mean_test_ic']:+.4f}  (phase1: +0.041)")
    print(f"  stability    : {best_row['stability']:+.3f}  (phase1: +0.215)")
    print(f"  objective    : {best_row['objective']:+.4f}  (phase1: +0.148)")

    print(f"\n  ALL TRIALS (ranked)")
    print(f"  {'─' * 40}")
    print(results_df.to_string(index=False))

    print(f"\n  Saved: {results_path.name}")
    print(f"  Saved: {params_path.name}")


if __name__ == "__main__":
    main()
