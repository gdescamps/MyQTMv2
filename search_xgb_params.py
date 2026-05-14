"""
Optuna hyperparameter search for XGBoost IC maximisation.

Objective: mean_test_IC + 0.5 x (val/test IC stability)

~25 trials targeting ~10 minutes on CUDA GB10.

Output:
  outputs/xgb_optuna_results.csv  -- all trials ranked
  outputs/xgb_best_params.json    -- best params (drop-in for XGB_PARAMS)
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
from train import run_walk_forward, _try_gpu, FEATURE_COLS, LABEL_COL

DATA    = Path(__file__).parent / "data"
OUTPUTS = Path(__file__).parent / "outputs"
OUTPUTS.mkdir(exist_ok=True)

N_TRIALS    = 25
STAB_WEIGHT = 0.5


def compute_objective(oos: pd.DataFrame) -> tuple[float, float, float]:
    """Returns (objective, mean_test_ic, stability)."""
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

    objective = mean_test_ic + STAB_WEIGHT * stability
    return objective, mean_test_ic, stability


def make_trial_params(trial: optuna.Trial) -> dict:
    return dict(
        max_depth         = trial.suggest_int  ("max_depth",         3,    7),
        min_child_weight  = trial.suggest_int  ("min_child_weight",  20, 250, log=True),
        subsample         = trial.suggest_float("subsample",         0.50, 0.95),
        colsample_bytree  = trial.suggest_float("colsample_bytree",  0.40, 0.90),
        learning_rate     = trial.suggest_float("learning_rate",     0.01, 0.15, log=True),
        reg_alpha         = trial.suggest_float("reg_alpha",         1e-4,  2.0, log=True),
        reg_lambda        = trial.suggest_float("reg_lambda",        0.10, 10.0, log=True),
        n_estimators      = 1000,
        early_stopping_rounds = 30,
    )


def fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s"


def main():
    feat_path = DATA / "features.parquet"
    if not feat_path.exists():
        sys.exit("ERROR: data/features.parquet not found -- run feature_engineering.py first")

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
    print(f"  Optuna TPE search -- {N_TRIALS} trials (~10 min)")
    print(f"  Objective: mean_test_IC + {STAB_WEIGHT} x stability")
    print(f"  Baseline:  test_IC=+0.033  stability=+0.070  obj=+0.068")
    print(f"{sep}\n", flush=True)

    results = []
    best_obj = -999.0
    t0 = time.time()

    def objective(trial: optuna.Trial) -> float:
        nonlocal best_obj

        trial_params = make_trial_params(trial)
        t_start = time.time()

        try:
            oos = run_walk_forward(
                panel, device,
                xgb_params=trial_params,
                verbose=False,
                save_models=False,
            )
        except Exception as e:
            print(f"  [{trial.number:2d}/{N_TRIALS}] ERROR: {e}", flush=True)
            return -99.0

        t_trial = time.time() - t_start
        obj, mean_test_ic, stability = compute_objective(oos)

        results.append({
            "trial":         trial.number,
            "objective":     obj,
            "mean_test_ic":  mean_test_ic,
            "stability":     stability,
            **trial_params,
        })

        elapsed = time.time() - t0
        avg_per_trial = elapsed / (trial.number + 1)
        remaining = avg_per_trial * (N_TRIALS - trial.number - 1)

        is_best = obj > best_obj
        if is_best:
            best_obj = obj

        marker = " *** NEW BEST ***" if is_best else ""
        print(
            f"  [{trial.number + 1:2d}/{N_TRIALS}] "
            f"obj={obj:+.4f}  "
            f"test_IC={mean_test_ic:+.4f}  "
            f"stab={stability:+.3f}  |  "
            f"d={trial_params['max_depth']}  "
            f"mcw={trial_params['min_child_weight']:3d}  "
            f"lr={trial_params['learning_rate']:.4f}  "
            f"({fmt_time(t_trial)})  "
            f"[{fmt_time(elapsed)} / ~{fmt_time(elapsed + remaining)}]"
            f"{marker}",
            flush=True,
        )
        return obj

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    total_time = time.time() - t0

    # Save results
    results_df = pd.DataFrame(results).sort_values("objective", ascending=False)
    results_path = OUTPUTS / "xgb_optuna_results.csv"
    results_df.to_csv(results_path, index=False)

    best = study.best_trial
    best_params = {k: best.params[k] for k in best.params}

    params_path = OUTPUTS / "xgb_best_params.json"
    with open(params_path, "w") as f:
        json.dump(best_params, f, indent=2)

    # Final report
    print(f"\n{sep}")
    print(f"  SEARCH COMPLETE -- {fmt_time(total_time)} total")
    print(f"{sep}")
    print(f"\n  BEST TRIAL #{best.number + 1}  objective={best.value:+.4f}")
    print(f"  {'─' * 40}")
    for k, v in best_params.items():
        if isinstance(v, float):
            print(f"    {k:<22s} {v:.6f}")
        else:
            print(f"    {k:<22s} {v}")

    best_row = results_df.iloc[0]
    print(f"\n  mean_test_IC : {best_row['mean_test_ic']:+.4f}  (baseline: +0.033)")
    print(f"  stability    : {best_row['stability']:+.3f}  (baseline: +0.070)")
    print(f"  objective    : {best_row['objective']:+.4f}  (baseline: +0.068)")

    # Top 5
    print(f"\n  TOP 5 TRIALS")
    print(f"  {'─' * 40}")
    cols = ["trial", "objective", "mean_test_ic", "stability",
            "max_depth", "min_child_weight", "learning_rate"]
    top5 = results_df[cols].head(5).copy()
    top5["trial"] = top5["trial"] + 1
    print(top5.to_string(index=False))

    print(f"\n  Saved: {results_path.name}")
    print(f"  Saved: {params_path.name}")


if __name__ == "__main__":
    main()
