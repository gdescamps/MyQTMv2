"""
Portfolio backtest + per-step CMA-ES allocation optimisation for MyQTM-ETF.

Walk-forward structure (matches train.py):
  For each step:
    1. CMA-ES optimises allocation params on val predictions (split="val")
       → These are XGBoost OOS scores (model never trained on those blocks)
       → Interlaced monthly blocks → covers all market regimes in training period
    2. Apply best params to test predictions (split="test") → true OOS returns

This eliminates CMA-ES overfitting: the params are found on unbiased val scores
and evaluated on a future test period never seen by CMA-ES.

Allocation model:
  1. budget_régime = equity_max × (1 - sigmoid(p1×vix + p2×hy_z60 + p3))
  2. equity weights = softmax(scores[equity], temperature) × equity_budget
  3. defensive weights = softmax(scores[defensive], temperature) × (1 - equity_budget - cash_min)
  4. Cash = 1 - Σweights

Output:
  data/backtest_results.parquet   (daily portfolio returns, test periods only)
  outputs/best_params.csv         (per-step CMA-ES params)
  outputs/backtest_equity.csv     (equity curve)
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
import cma
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

DATA    = Path(__file__).parent / "data"
OUTPUTS = Path(__file__).parent / "outputs"
OUTPUTS.mkdir(exist_ok=True)

EQUITY_SECTIONS    = {"geo", "sector_us", "thematic"}
DEFENSIVE_SECTIONS = {"bond", "commodity", "crypto"}

PARAM_NAMES = [
    "p1_vix", "p2_hy_z60", "p3_bias",
    "equity_max", "cash_min",
    "temperature", "max_single_weight",
    "min_weight_change", "defensive_min",
]
PARAM_INIT   = [-0.05, -0.20, 0.0, 0.80, 0.05, 1.5, 0.25, 0.03, 0.10]
PARAM_SIGMA0 = 0.15
PARAM_BOUNDS = [
    (-0.20, 0.00),   # p1_vix
    (-0.80, 0.00),   # p2_hy_z60
    (-3.00, 3.00),   # p3_bias
    ( 0.50, 0.95),   # equity_max
    ( 0.00, 0.20),   # cash_min
    ( 0.50, 4.00),   # temperature
    ( 0.05, 0.50),   # max_single_weight
    ( 0.01, 0.10),   # min_weight_change
    ( 0.05, 0.40),   # defensive_min
]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def softmax(scores: np.ndarray, temperature: float) -> np.ndarray:
    s = scores / max(temperature, 1e-6)
    s = s - s.max()
    e = np.exp(s)
    return e / e.sum()


def clip_params(params: list) -> list:
    return [float(np.clip(v, lo, hi)) for v, (lo, hi) in zip(params, PARAM_BOUNDS)]


def run_backtest(
    scores_wide: pd.DataFrame,
    labels_wide: pd.DataFrame,
    is_equity: np.ndarray,
    is_defensive: np.ndarray,
    vix_series: pd.Series,
    hy_z60_series: pd.Series,
    params: list,
    transaction_cost: float = 0.0022,
) -> pd.Series:
    """
    Vectorised portfolio simulation.
    scores_wide / labels_wide : DataFrame (dates × etf_ids)
    is_equity / is_defensive  : bool arrays aligned to columns
    Returns daily portfolio return series.
    """
    params = clip_params(params)
    p1, p2, p3, equity_max, cash_min, temp, max_w, min_change, def_min = params

    dates   = scores_wide.index
    n_etfs  = scores_wide.shape[1]

    scores = scores_wide.values
    labels = labels_wide.values

    vix    = vix_series.reindex(dates, method="ffill").fillna(20.0).values
    hy_z60 = hy_z60_series.reindex(dates, method="ffill").fillna(0.0).values
    danger        = sigmoid(p1 * vix + p2 * hy_z60 + p3)
    equity_budget = equity_max * (1.0 - danger)
    def_budget    = np.where(
        danger > 0.5,
        def_min + (1.0 - equity_budget - cash_min) * (1 - danger),
        (1.0 - equity_budget - cash_min) * (1 - danger),
    ).clip(0, None)

    def _softmax_masked(s: np.ndarray, mask: np.ndarray, T: float) -> np.ndarray:
        out = np.zeros_like(s)
        if not mask.any():
            return out
        s_m = s[:, mask].copy() / max(T, 1e-6)
        for i in range(s_m.shape[0]):
            row   = s_m[i]
            valid = ~np.isnan(row)
            if not valid.any():
                continue
            row_v = row[valid] - row[valid].max()
            e     = np.exp(row_v)
            probs = e / e.sum()
            result = np.zeros(len(row))
            result[valid] = probs
            out[i, np.where(mask)[0]] = result
        return out

    w_eq  = _softmax_masked(scores, is_equity,    temp) * equity_budget[:, None]
    w_def = _softmax_masked(scores, is_defensive, temp) * def_budget[:, None]
    weights = np.minimum(w_eq + w_def, max_w)

    # Anti-churn: hold previous weight if |delta| < min_change
    prev = np.zeros(n_etfs)
    final_weights = np.empty_like(weights)
    for t in range(len(dates)):
        delta   = weights[t] - prev
        applied = np.where(np.abs(delta) > min_change, weights[t], prev)
        final_weights[t] = applied
        prev = applied

    nan_mask = np.isnan(labels)
    safe_lbl = np.where(nan_mask, 0.0, labels)
    port_ret = (final_weights * safe_lbl).sum(axis=1)

    turnover  = np.abs(np.diff(final_weights, axis=0, prepend=np.zeros((1, n_etfs)))).sum(axis=1)
    port_ret -= turnover * transaction_cost

    return pd.Series(port_ret, index=dates, name="port_return")


def sharpe(returns: pd.Series, min_obs: int = 30) -> float:
    r = returns.dropna()
    if len(r) < min_obs or r.std() < 1e-10:
        return -10.0
    return float(r.mean() / r.std() * np.sqrt(252))


def run_cmaes(
    scores_wide: pd.DataFrame,
    labels_wide: pd.DataFrame,
    is_equity: np.ndarray,
    is_defensive: np.ndarray,
    vix_s: pd.Series,
    hy_z60_s: pd.Series,
    maxiter: int = 80,
) -> tuple[list, float]:
    """
    Run CMA-ES on val predictions for one walk-forward step.
    Returns (best_params, best_sharpe).
    """
    def objective(params: list) -> float:
        returns = run_backtest(scores_wide, labels_wide, is_equity, is_defensive,
                               vix_s, hy_z60_s, params)
        return -sharpe(returns)

    es = cma.CMAEvolutionStrategy(
        PARAM_INIT,
        PARAM_SIGMA0,
        {
            "maxiter": maxiter,
            "tolx":    1e-4,
            "tolfun":  1e-4,
            "bounds":  [list(b[0] for b in PARAM_BOUNDS),
                        list(b[1] for b in PARAM_BOUNDS)],
            "verbose": -9,
            "popsize": 16,
        },
    )

    best_params = list(PARAM_INIT)
    best_sharpe = -objective(PARAM_INIT)

    while not es.stop():
        solutions = es.ask()
        fitnesses = [objective(x) for x in solutions]
        es.tell(solutions, fitnesses)
        step_best = -min(fitnesses)
        if step_best > best_sharpe:
            best_sharpe = step_best
            best_params = list(solutions[np.argmin(fitnesses)])

    return best_params, best_sharpe


def _pivot_step(step_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pivot (date, etf_id) long → wide scores and labels."""
    sw = step_data["score"].unstack("etf_id").sort_index()
    lw = step_data["label"].unstack("etf_id").sort_index().reindex(columns=sw.columns)
    return sw, lw


def _section_arrays(etf_list: list, sections: dict) -> tuple[np.ndarray, np.ndarray]:
    is_eq  = np.array([sections.get(e) in EQUITY_SECTIONS    for e in etf_list])
    is_def = np.array([sections.get(e) in DEFENSIVE_SECTIONS for e in etf_list])
    return is_eq, is_def


def _save_equity_png(port_returns: pd.Series, eq_curve: pd.Series,
                     params_df: pd.DataFrame, out_dir: Path) -> None:
    """Save a two-panel equity + drawdown PNG for the backtest."""
    dd = eq_curve / eq_curve.cummax() - 1
    ann_ret = port_returns.mean() * 252
    ann_vol = port_returns.std() * np.sqrt(252)
    sh      = sharpe(port_returns)
    max_dd  = dd.min()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                    gridspec_kw={"height_ratios": [3, 1]},
                                    sharex=True)
    fig.suptitle(
        f"MyQTM-ETF — Walk-Forward OOS  "
        f"(Sharpe={sh:.2f}  Ann={ann_ret:.1%}  Vol={ann_vol:.1%}  MaxDD={max_dd:.1%})",
        fontsize=12, fontweight="bold",
    )

    # Panel 1: equity curve with per-step shading
    ax1.plot(eq_curve.index, eq_curve.values, lw=1.8, color="#1f77b4", label="Portfolio")
    ax1.set_ylabel("Portfolio value (base 1)")
    ax1.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:.1f}x"))
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor("#f8f8f8")

    # Shade alternating steps
    if "step" in params_df.columns and len(params_df) > 1:
        step_dates = {}
        # We can't easily get step boundaries here — shade via equity index split
        n_steps = len(params_df)
        dates   = eq_curve.index
        chunk   = max(len(dates) // n_steps, 1)
        for i in range(n_steps):
            if i % 2 == 0:
                s = dates[i * chunk]
                e = dates[min((i + 1) * chunk - 1, len(dates) - 1)]
                ax1.axvspan(s, e, alpha=0.05, color="orange")

    ax1.legend(fontsize=10)

    # Panel 2: drawdown
    ax2.fill_between(dd.index, dd.values, 0, alpha=0.6, color="#d62728")
    ax2.set_ylabel("Drawdown")
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax2.set_facecolor("#f8f8f8")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_dir / "backtest_equity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    oos_path = DATA / "oos_predictions.parquet"
    if not oos_path.exists():
        sys.exit("ERROR: data/oos_predictions.parquet not found — run train.py first")

    print("Loading OOS predictions...")
    oos = pd.read_parquet(oos_path)
    oos = oos.reset_index()
    oos["date"] = pd.to_datetime(oos["date"])
    oos = oos.set_index(["date", "etf_id"])

    if "split" not in oos.columns:
        sys.exit("ERROR: oos_predictions.parquet has no 'split' column — retrain with updated train.py")

    sys.path.insert(0, str(Path(__file__).parent))
    from etf import UNIVERSE
    sections = {e.bourso: e.section for e in UNIVERSE}

    # Macro regime data
    macro = pd.DataFrame()
    for fname, col in [("vix", "vix_level"), ("hy_spread", "hy_spread_z60")]:
        path = DATA / f"fred_{fname}.parquet"
        if path.exists():
            s = pd.read_parquet(path).iloc[:, 0]
            if col == "hy_spread_z60":
                m, sd = s.rolling(60).mean(), s.rolling(60).std()
                s = (s - m) / sd.replace(0, np.nan)
            macro[col] = s

    vix_s    = macro.get("vix_level",     pd.Series(dtype=float))
    hy_z60_s = macro.get("hy_spread_z60", pd.Series(dtype=float))

    steps = sorted(oos["step"].unique())
    print(f"Walk-forward steps: {len(steps)}  "
          f"val rows: {(oos['split']=='val').sum()}  "
          f"test rows: {(oos['split']=='test').sum()}\n")

    all_test_returns = []
    all_params_rows  = []

    print(f"{'Step':>4}  {'Val period':>24}  {'Val↗':>7}  "
          f"{'Test period':>24}  {'Test↗':>7}")
    print("-" * 75)

    for step in steps:
        val_data  = oos[(oos["step"] == step) & (oos["split"] == "val")]
        test_data = oos[(oos["step"] == step) & (oos["split"] == "test")]

        if len(val_data) < 50 or len(test_data) == 0:
            print(f"{step:4d}  SKIP (val={len(val_data)} rows, test={len(test_data)} rows)")
            continue

        # --- CMA-ES on val predictions ---
        val_sw, val_lw = _pivot_step(val_data)
        val_is_eq, val_is_def = _section_arrays(val_sw.columns.tolist(), sections)
        best_params, val_sharpe = run_cmaes(val_sw, val_lw, val_is_eq, val_is_def, vix_s, hy_z60_s)

        # --- Evaluate on test (true OOS) ---
        test_sw, test_lw = _pivot_step(test_data)
        test_is_eq, test_is_def = _section_arrays(test_sw.columns.tolist(), sections)
        test_returns = run_backtest(test_sw, test_lw, test_is_eq, test_is_def,
                                    vix_s, hy_z60_s, best_params)
        test_sh = sharpe(test_returns)

        val_dates  = val_data.index.get_level_values("date")
        test_dates = test_data.index.get_level_values("date")
        print(
            f"{step:4d}  "
            f"[{val_dates.min().date()} → {val_dates.max().date()}]  {val_sharpe:7.3f}  "
            f"[{test_dates.min().date()} → {test_dates.max().date()}]  {test_sh:7.3f}"
        )

        all_test_returns.append(test_returns)
        all_params_rows.append({"step": step, **dict(zip(PARAM_NAMES, best_params))})

    if not all_test_returns:
        sys.exit("No test returns produced — check OOS predictions")

    # --- Final backtest stats (true OOS: test periods only) ---
    port_returns = pd.concat(all_test_returns).sort_index()
    eq_curve     = (1 + port_returns).cumprod()

    total_ret    = eq_curve.iloc[-1] - 1
    n_years      = max(len(port_returns), 1) / 252
    ann_ret      = (1 + total_ret) ** (1 / n_years) - 1
    ann_vol      = port_returns.std() * np.sqrt(252)
    final_sharpe = sharpe(port_returns)
    max_dd       = (eq_curve / eq_curve.cummax() - 1).min()

    print("\n" + "=" * 75)
    print("=== Final OOS backtest (test periods, CMA-ES per step) ===")
    print(f"  Period:       {port_returns.index[0].date()} → {port_returns.index[-1].date()}")
    print(f"  Total return: {total_ret:.1%}")
    print(f"  Ann. return:  {ann_ret:.1%}")
    print(f"  Ann. vol:     {ann_vol:.1%}")
    print(f"  Sharpe:       {final_sharpe:.3f}")
    print(f"  Max drawdown: {max_dd:.1%}")

    # Save results
    port_returns.to_frame().to_parquet(DATA / "backtest_results.parquet")

    params_df = pd.DataFrame(all_params_rows)
    params_df.to_csv(OUTPUTS / "best_params.csv", index=False)

    eq_df = eq_curve.reset_index()
    eq_df.columns = ["date", "equity"]
    eq_df.to_csv(OUTPUTS / "backtest_equity.csv", index=False)

    # Save per-step CMA-ES details (sharpe on val + test)
    step_summary = []
    for row in all_params_rows:
        step_summary.append(row)
    pd.DataFrame(step_summary).to_csv(OUTPUTS / "cmaes_steps.csv", index=False)

    # Equity curve PNG
    _save_equity_png(port_returns, eq_curve, params_df, OUTPUTS)

    print(f"\nSaved → data/backtest_results.parquet")
    print(f"Saved → outputs/best_params.csv")
    print(f"Saved → outputs/cmaes_steps.csv")
    print(f"Saved → outputs/backtest_equity.csv")
    print(f"Saved → outputs/backtest_equity.png")


if __name__ == "__main__":
    main()
