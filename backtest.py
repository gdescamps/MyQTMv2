"""
Portfolio backtest + CMA-ES allocation optimisation for MyQTM-ETF.

Allocation model:
  1. budget_régime = equity_max × (1 - sigmoid(p1×vix + p2×hy_z60 + p3))
  2. equity weights = softmax(scores[equity], temperature) × equity_budget
  3. defensive weights = softmax(scores[defensive], temperature) × (1 - equity_budget - cash_min)
  4. Cash = 1 - Σweights

CMA-ES optimises 9 parameters to maximise OOS Sharpe ratio.

Output:
  data/backtest_results.parquet   (daily portfolio returns)
  data/best_params.npy            (best CMA-ES parameters)
  outputs/backtest_equity.csv     (equity curve)
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
import cma

DATA    = Path(__file__).parent / "data"
OUTPUTS = Path(__file__).parent / "outputs"
OUTPUTS.mkdir(exist_ok=True)

# ETF sections classification for allocation
EQUITY_SECTIONS    = {"geo", "sector_us", "thematic"}
DEFENSIVE_SECTIONS = {"bond", "commodity", "crypto"}

# CMA-ES parameter bounds and initial values
# [p1_vix, p2_hy_z60, p3_bias, equity_max, cash_min, temperature, max_single_weight,
#  min_weight_change, defensive_min]
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
    ( 0.50, 4.00),   # temperature (softmax concentration)
    ( 0.05, 0.50),   # max_single_weight (cap per ETF)
    ( 0.01, 0.10),   # min_weight_change (anti-churn threshold)
    ( 0.05, 0.40),   # defensive_min when danger > 0.5
]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def softmax(scores: np.ndarray, temperature: float) -> np.ndarray:
    s = scores / max(temperature, 1e-6)
    s = s - s.max()  # numerical stability
    e = np.exp(s)
    return e / e.sum()


def clip_params(params: list) -> list:
    return [float(np.clip(v, lo, hi)) for v, (lo, hi) in zip(params, PARAM_BOUNDS)]


def compute_daily_weights(
    scores_eq: np.ndarray, scores_def: np.ndarray,
    vix: float, hy_z60: float,
    params: list,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (weights_equity, weights_defensive) for one day."""
    p1, p2, p3, equity_max, cash_min, temp, max_w, _, def_min = params

    danger        = sigmoid(p1 * vix + p2 * hy_z60 + p3)
    equity_budget = equity_max * (1.0 - danger)
    def_budget    = max(def_min * (danger > 0.5), 0) + (1.0 - equity_budget - cash_min) * (1 - danger)
    def_budget    = float(np.clip(def_budget, 0, 1 - equity_budget - cash_min))

    w_eq  = softmax(scores_eq,  temp) * equity_budget
    w_def = softmax(scores_def, temp) * def_budget

    # Cap individual weights
    w_eq  = np.minimum(w_eq,  max_w)
    w_def = np.minimum(w_def, max_w)

    return w_eq, w_def


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
    scores_wide / labels_wide: DataFrame (dates × etf_ids)
    is_equity / is_defensive:  bool arrays aligned to columns of wide tables
    Returns daily portfolio return series.
    """
    params = clip_params(params)
    p1, p2, p3, equity_max, cash_min, temp, max_w, min_change, def_min = params

    dates   = scores_wide.index
    n_dates = len(dates)
    n_etfs  = scores_wide.shape[1]

    scores = scores_wide.values   # (T, E)
    labels = labels_wide.values   # (T, E)

    # Danger signal per day
    vix    = vix_series.reindex(dates, method="ffill").fillna(20.0).values
    hy_z60 = hy_z60_series.reindex(dates, method="ffill").fillna(0.0).values
    danger        = sigmoid(p1 * vix + p2 * hy_z60 + p3)              # (T,)
    equity_budget = equity_max * (1.0 - danger)                        # (T,)
    def_budget    = np.where(
        danger > 0.5,
        def_min + (1.0 - equity_budget - cash_min) * (1 - danger),
        (1.0 - equity_budget - cash_min) * (1 - danger),
    ).clip(0, None)                                                     # (T,)

    # Softmax weights per day
    def _softmax_masked(s: np.ndarray, mask: np.ndarray, T: float) -> np.ndarray:
        """s: (T,E), mask: (E,) bool → (T,E) weights, only masked cols active."""
        out = np.zeros_like(s)
        if not mask.any():
            return out
        s_m = s[:, mask] / max(T, 1e-6)
        s_m = s_m - s_m.max(axis=1, keepdims=True)
        e   = np.exp(s_m)
        out[:, mask] = e / e.sum(axis=1, keepdims=True)
        return out

    w_eq  = _softmax_masked(scores, is_equity,    temp) * equity_budget[:, None]
    w_def = _softmax_masked(scores, is_defensive, temp) * def_budget[:, None]
    weights = w_eq + w_def

    # Cap per ETF
    weights = np.minimum(weights, max_w)

    # Anti-churn: keep previous weight if |delta| < min_change
    prev = np.zeros(n_etfs)
    final_weights = np.empty_like(weights)
    for t in range(n_dates):
        delta = weights[t] - prev
        applied = np.where(np.abs(delta) > min_change, weights[t], prev)
        final_weights[t] = applied
        prev = applied

    # Portfolio return = Σ w × label (nan-safe)
    nan_mask  = np.isnan(labels)
    safe_lbl  = np.where(nan_mask, 0.0, labels)
    port_ret  = (final_weights * safe_lbl).sum(axis=1)

    # Transaction costs
    turnover   = np.abs(np.diff(final_weights, axis=0, prepend=np.zeros((1, n_etfs)))).sum(axis=1)
    port_ret  -= turnover * transaction_cost

    return pd.Series(port_ret, index=dates, name="port_return")


def sharpe(returns: pd.Series, min_obs: int = 50) -> float:
    r = returns.dropna()
    if len(r) < min_obs or r.std() < 1e-10:
        return -10.0
    return float(r.mean() / r.std() * np.sqrt(252))


def main():
    oos_path = DATA / "oos_predictions.parquet"
    if not oos_path.exists():
        sys.exit("ERROR: data/oos_predictions.parquet not found — run train.py first")

    print("Loading OOS predictions and features...")
    oos = pd.read_parquet(oos_path)
    oos = oos.reset_index()
    oos["date"] = pd.to_datetime(oos["date"])
    oos = oos.set_index(["date", "etf_id"])

    # ETF section lookup
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

    print(f"OOS rows: {len(oos)}  dates: {oos.index.get_level_values('date').nunique()}")

    # Deduplicate: keep most recent step for each (date, etf_id)
    if oos.index.duplicated().any():
        oos = oos.sort_values("step").groupby(level=["date", "etf_id"]).last()
        print(f"After dedup: {len(oos)} rows")

    # Pre-compute wide tables (dates × etf_ids) for vectorised backtest
    print("Pivoting to wide format...")
    scores_wide = oos["score"].unstack("etf_id").sort_index()
    labels_wide = oos["label"].unstack("etf_id").sort_index()
    # Align columns
    all_etfs = scores_wide.columns.tolist()
    labels_wide = labels_wide.reindex(columns=all_etfs)

    is_equity    = np.array([sections.get(e) in EQUITY_SECTIONS    for e in all_etfs])
    is_defensive = np.array([sections.get(e) in DEFENSIVE_SECTIONS for e in all_etfs])

    # Macro series aligned to wide index
    vix_s    = macro.get("vix_level",    pd.Series(20.0,  index=scores_wide.index))
    hy_z60_s = macro.get("hy_spread_z60", pd.Series(0.0, index=scores_wide.index))

    print(f"Wide: {scores_wide.shape[0]} dates × {scores_wide.shape[1]} ETFs")

    # -----------------------------------------------------------------------
    # CMA-ES optimisation
    # -----------------------------------------------------------------------
    def objective(params: list) -> float:
        returns = run_backtest(scores_wide, labels_wide, is_equity, is_defensive,
                               vix_s, hy_z60_s, params)
        return -sharpe(returns)

    print("\n=== CMA-ES optimisation (9 parameters) ===")
    es = cma.CMAEvolutionStrategy(
        PARAM_INIT,
        PARAM_SIGMA0,
        {
            "maxiter":   150,
            "tolx":      1e-4,
            "tolfun":    1e-4,
            "bounds":    [list(b[0] for b in PARAM_BOUNDS),
                          list(b[1] for b in PARAM_BOUNDS)],
            "verbose":   -9,
            "popsize":   16,
        },
    )

    best_params = PARAM_INIT
    best_sharpe = -objective(PARAM_INIT)
    iteration   = 0

    while not es.stop():
        solutions   = es.ask()
        fitnesses   = [objective(x) for x in solutions]
        es.tell(solutions, fitnesses)
        iteration  += 1

        step_best   = -min(fitnesses)
        if step_best > best_sharpe:
            best_sharpe = step_best
            best_params = solutions[np.argmin(fitnesses)]

        if iteration % 10 == 0:
            print(f"  iter {iteration:3d}  best Sharpe={best_sharpe:.3f}  "
                  f"sigma={es.sigma:.4f}")

    print(f"\nCMA-ES done — best Sharpe: {best_sharpe:.4f}")
    print("Best parameters:")
    for name, val in zip(PARAM_NAMES, best_params):
        print(f"  {name:<22} {val:.4f}")

    # -----------------------------------------------------------------------
    # Final backtest with best params
    # -----------------------------------------------------------------------
    print("\n=== Final backtest ===")
    port_returns = run_backtest(scores_wide, labels_wide, is_equity, is_defensive,
                                vix_s, hy_z60_s, best_params)
    eq_curve     = (1 + port_returns).cumprod()

    total_ret  = eq_curve.iloc[-1] - 1
    ann_ret    = (1 + total_ret) ** (252 / max(len(port_returns), 1)) - 1
    ann_vol    = port_returns.std() * np.sqrt(252)
    final_sharpe = sharpe(port_returns)
    max_dd     = (eq_curve / eq_curve.cummax() - 1).min()

    print(f"  Period:       {port_returns.index[0].date()} → {port_returns.index[-1].date()}")
    print(f"  Total return: {total_ret:.1%}")
    print(f"  Ann. return:  {ann_ret:.1%}")
    print(f"  Ann. vol:     {ann_vol:.1%}")
    print(f"  Sharpe:       {final_sharpe:.3f}")
    print(f"  Max drawdown: {max_dd:.1%}")

    # Save results
    port_returns.to_frame().to_parquet(DATA / "backtest_results.parquet")
    np.save(OUTPUTS / "best_params.npy", np.array(best_params))

    eq_df = eq_curve.reset_index()
    eq_df.columns = ["date", "equity"]
    eq_df.to_csv(OUTPUTS / "backtest_equity.csv", index=False)

    print(f"\nSaved → data/backtest_results.parquet")
    print(f"Saved → outputs/best_params.npy")
    print(f"Saved → outputs/backtest_equity.csv")


if __name__ == "__main__":
    main()
