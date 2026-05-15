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

# Geo/sector concentration: only 1 line allowed, redirect to best of these 3
GEO_SECTOR_REDIRECT = ["CSP1.PA", "CNX1.PA", "WPEA.PA"]  # SP500, Nasdaq, MSCI World

PARAM_NAMES = [
    "p1_vix", "p2_hy_z60", "p3_bias",
    "temperature_A", "temperature_B",
]
PARAM_INIT   = [-0.10, -0.30, -0.5, 1.5, 1.5]
PARAM_SIGMA0 = 0.15
PARAM_BOUNDS = [
    (-0.20, -0.02),  # p1_vix — must be negative (VIX increases danger)
    (-0.80, -0.05),  # p2_hy_z60 — must be negative (HY stress increases danger)
    (-1.50, 1.00),   # p3_bias — tighter range, can't fully disable soupape
    ( 0.05, 4.00),   # temperature_A
    ( 0.05, 4.00),   # temperature_B
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
    scores_wide_A: pd.DataFrame,
    scores_wide_B: pd.DataFrame | None,
    daily_returns_wide: pd.DataFrame,
    is_equity: np.ndarray,
    is_defensive: np.ndarray,
    vix_series: pd.Series,
    hy_z60_series: pd.Series,
    params: list,
    transaction_cost: float = 0.0022,
    prev_weights: np.ndarray | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Vectorised portfolio simulation.
    scores_wide_A/B : DataFrame (dates × etf_ids) — model scores for allocation
    daily_returns_wide : DataFrame (dates × etf_ids) — actual daily returns for P&L
    Returns daily portfolio return series.
    """
    params = clip_params(params)
    p1, p2, p3, temp_A, temp_B = params

    dates   = scores_wide_A.index
    n_etfs  = scores_wide_A.shape[1]
    etf_list = scores_wide_A.columns.tolist()

    scores_A = scores_wide_A.values
    scores_B = scores_wide_B.values if scores_wide_B is not None else scores_A
    daily_ret = daily_returns_wide.values

    vix    = vix_series.reindex(dates, method="ffill").fillna(20.0).values
    hy_z60 = hy_z60_series.reindex(dates, method="ffill").fillna(0.0).values
    danger        = sigmoid(p1 * vix + p2 * hy_z60 + p3)
    equity_budget = 1.0 - danger
    def_budget    = danger

    def _softmax_masked(s: np.ndarray, mask: np.ndarray, T: float) -> np.ndarray:
        out = np.zeros_like(s)
        if not mask.any():
            return out
        s_m = s[:, mask].copy() / max(T, 1e-6)
        for i in range(s_m.shape[0]):
            row = s_m[i]
            valid = ~np.isnan(row)
            if not valid.any():
                continue
            row_v = row[valid] - row[valid].max()
            e = np.exp(row_v)
            probs = e / e.sum()
            result = np.zeros(len(row))
            result[valid] = probs
            out[i, np.where(mask)[0]] = result
        return out

    # Compute target weights (vectorized)
    w_eq_A  = _softmax_masked(scores_A, is_equity,    temp_A) * equity_budget[:, None]
    w_def_A = _softmax_masked(scores_A, is_defensive, temp_A) * def_budget[:, None]
    w_eq_B  = _softmax_masked(scores_B, is_equity,    temp_B) * equity_budget[:, None]
    w_def_B = _softmax_masked(scores_B, is_defensive, temp_B) * def_budget[:, None]
    weights = 0.5 * (w_eq_A + w_def_A) + 0.5 * (w_eq_B + w_def_B)

    # Geo/sector redirect
    from etf import BY_BOURSO
    redirect_indices = [i for i, e in enumerate(etf_list) if e in GEO_SECTOR_REDIRECT]
    geo_sector_other = [i for i, e in enumerate(etf_list)
                        if BY_BOURSO.get(e) and BY_BOURSO[e].section in ("geo", "sector_us")
                        and e not in GEO_SECTOR_REDIRECT]
    if redirect_indices and geo_sector_other:
        for t in range(len(dates)):
            extra = weights[t, geo_sector_other].sum()
            weights[t, geo_sector_other] = 0.0
            if extra > 0:
                avg_s = (scores_A[t, redirect_indices] + scores_B[t, redirect_indices]) / 2
                valid_s = ~np.isnan(avg_s)
                if valid_s.any():
                    weights[t, redirect_indices[np.nanargmax(avg_s)]] += extra

    # Portfolio simulation with real daily returns
    safe_ret = np.where(np.isnan(daily_ret), 0.0, daily_ret)
    turnover = np.abs(np.diff(weights, axis=0, prepend=(
        prev_weights[None, :] if prev_weights is not None else np.zeros((1, n_etfs))
    ))).sum(axis=1)

    # Daily P&L: positions at weight_{t} earn return_{t+1}
    # Use weight from previous day to compute today's return (hold overnight)
    w_prev = np.vstack([prev_weights if prev_weights is not None else np.zeros(n_etfs),
                        weights[:-1]])
    port_ret = (w_prev * safe_ret).sum(axis=1) - turnover * transaction_cost

    ret_series = pd.Series(port_ret, index=dates, name="port_return")
    weights_df = pd.DataFrame(weights, index=dates, columns=etf_list)
    return ret_series, weights_df


def sharpe(returns: pd.Series, min_obs: int = 30) -> float:
    r = returns.dropna()
    if len(r) < min_obs or r.std() < 1e-10:
        return -10.0
    return float(r.mean() / r.std() * np.sqrt(252))


def run_cmaes(
    scores_wide_A: pd.DataFrame,
    scores_wide_B: pd.DataFrame,
    daily_returns_wide: pd.DataFrame,
    is_equity: np.ndarray,
    is_defensive: np.ndarray,
    vix_s: pd.Series,
    hy_z60_s: pd.Series,
    maxiter: int = 50,
) -> tuple[list, float]:
    """
    Run CMA-ES on val predictions for one walk-forward step.
    Returns (best_params, best_sharpe).
    """
    def objective(params: list) -> float:
        returns, _ = run_backtest(scores_wide_A, scores_wide_B, daily_returns_wide,
                                  is_equity, is_defensive, vix_s, hy_z60_s, params)
        r = returns.dropna()
        if len(r) < 30 or r.std() < 1e-10:
            return 10.0
        ann_ret = r.mean() * 252
        ann_vol = r.std() * np.sqrt(252)
        eq = (1 + r).cumprod()
        max_dd = abs((eq / eq.cummax() - 1).min())
        if max_dd < 1e-10:
            max_dd = 1e-10
        return -(ann_ret / (max_dd * ann_vol))

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
    """Pivot (date, etf_id) long → wide scores_A, scores_B."""
    if "score_A" in step_data.columns:
        sw_A = step_data["score_A"].unstack("etf_id").sort_index()
        sw_B = step_data["score_B"].unstack("etf_id").sort_index()
    else:
        sw_A = step_data["score"].unstack("etf_id").sort_index()
        sw_B = sw_A
    return sw_A, sw_B


def _section_arrays(etf_list: list, sections: dict) -> tuple[np.ndarray, np.ndarray]:
    is_eq  = np.array([sections.get(e) in EQUITY_SECTIONS    for e in etf_list])
    is_def = np.array([sections.get(e) in DEFENSIVE_SECTIONS for e in etf_list])
    return is_eq, is_def


def _load_benchmark(ticker: str, dates: pd.DatetimeIndex) -> pd.Series | None:
    """Load a benchmark equity curve normalised to 1 at first available date."""
    for fname in [f"{ticker}.parquet", f"{ticker.replace('.', '_')}.parquet"]:
        path = DATA / fname
        if path.exists():
            df = pd.read_parquet(path)
            col = "close" if "close" in df.columns else "adj_close"
            s = df[col].reindex(dates, method="ffill").dropna()
            if len(s) > 10:
                return s / s.iloc[0]
    return None


def _save_equity_png(port_returns: pd.Series, eq_curve: pd.Series,
                     weights_df: pd.DataFrame,
                     params_df: pd.DataFrame, out_dir: Path) -> None:
    """Save 3 separate PNGs: equity, allocation, P&L."""
    from etf import BY_BOURSO
    ann_ret = port_returns.mean() * 252
    ann_vol = port_returns.std() * np.sqrt(252)
    sh      = sharpe(port_returns)
    max_dd  = (eq_curve / eq_curve.cummax() - 1).min()
    title = (f"MyQTM-ETF — Walk-Forward OOS  "
             f"(Sharpe={sh:.2f}  Ann={ann_ret:.1%}  Vol={ann_vol:.1%}  MaxDD={max_dd:.1%})")

    # --- PNG 1: Equity curves + VIX ---
    fig1, ax1 = plt.subplots(figsize=(16, 7))
    fig1.suptitle(title, fontsize=12, fontweight="bold")
    ax1.plot(eq_curve.index, eq_curve.values, lw=2.0, color="#1f77b4", label="Portfolio")
    for ticker, name, color in [("CSPX_AS", "S&P 500", "#ff7f0e"),
                                 ("QQQ", "Nasdaq", "#2ca02c"),
                                 ("GLD", "Gold", "#d4af37")]:
        bm = _load_benchmark(ticker, eq_curve.index)
        if bm is not None:
            ax1.plot(bm.index, bm.values, lw=1.2, color=color, alpha=0.7, label=name)
    ax1.set_yscale("log")
    ax1.set_ylabel("Equity (log scale, base 1)")
    ax1.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:.1f}x"))
    ax1.grid(True, alpha=0.3, which="both")
    ax1.set_facecolor("#f8f8f8")
    ax1.legend(fontsize=10, loc="upper left")
    vix_path = DATA / "fred_vix.parquet"
    if vix_path.exists():
        vix_raw = pd.read_parquet(vix_path).iloc[:, 0]
        vix_raw = vix_raw.reindex(eq_curve.index, method="ffill").dropna()
        ax1b = ax1.twinx()
        ax1b.fill_between(vix_raw.index, vix_raw.values, alpha=0.10, color="#d62728")
        ax1b.set_ylabel("VIX", color="#d62728", fontsize=9)
        ax1b.tick_params(axis="y", labelcolor="#d62728", labelsize=8)
        ax1b.set_ylim(0, 80)
    plt.tight_layout()
    fig1.savefig(out_dir / "backtest_equity.png", dpi=150, bbox_inches="tight")
    plt.close(fig1)

    # --- PNG 2: Allocation detail per ETF (full names) ---
    if len(weights_df) > 0:
        w = weights_df.reindex(eq_curve.index, method="ffill").fillna(0)
        cash = (1 - w.sum(axis=1)).clip(0, 1)
        # Use full ETF names
        col_names = {e: BY_BOURSO[e].name[:45] if e in BY_BOURSO else e for e in w.columns}
        w_named = w.rename(columns=col_names)
        # Sort by average weight
        avg_w = w_named.mean().sort_values(ascending=False)
        top_etfs = avg_w[avg_w > 0.001].index.tolist()
        plot_data = w_named[top_etfs].copy()
        plot_data["Cash"] = cash

        fig2, ax2 = plt.subplots(figsize=(16, 8))
        fig2.suptitle("Portfolio Allocation Detail", fontsize=12, fontweight="bold")

        # Color map: defensive = yellow/orange/brown, equity = blues/greens
        from etf import UNIVERSE as _UNIVERSE
        color_map = {}
        for etf in _UNIVERSE:
            name = BY_BOURSO[etf.bourso].name[:45] if etf.bourso in BY_BOURSO else etf.bourso
            if etf.theme == "gold":
                color_map[name] = "#FFD700"       # gold = jaune or
            elif etf.theme == "gold_miners":
                color_map[name] = "#DAA520"       # gold miners = jaune fonce
            elif etf.theme == "oil":
                color_map[name] = "#8B4513"       # petrole = marron
            elif etf.theme == "commodity":
                color_map[name] = "#FF8C00"       # matieres = orange
            elif etf.section == "thematic":
                color_map[name] = "#9467bd"       # thematic = violet
            elif etf.section == "geo":
                color_map[name] = "#1f77b4"       # geo = bleu
            elif etf.section == "sector_us":
                color_map[name] = "#2ca02c"       # sector = vert
        color_map["Cash"] = "#e8e8e8"

        colors = [color_map.get(c, "#7f7f7f") for c in plot_data.columns]
        ax2.stackplot(plot_data.index, plot_data.values.T,
                      labels=plot_data.columns, colors=colors, alpha=0.85)
        ax2.set_ylabel("Allocation")
        ax2.set_ylim(0, 1)
        ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax2.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5), ncol=1)
        ax2.set_facecolor("#f8f8f8")
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        fig2.savefig(out_dir / "backtest_allocation.png", dpi=150, bbox_inches="tight")
        plt.close(fig2)

    # --- PNG 3: ETF P&L in $ (from 10,000$ capital) ---
    INITIAL_CAPITAL = 10000.0
    if len(weights_df) > 0 and len(port_returns) > 0:
        w = weights_df.reindex(port_returns.index, method="ffill").fillna(0)
        # Load real daily returns
        from etf import UNIVERSE
        daily_ret_etf = {}
        for etf in UNIVERSE:
            fmp_file = DATA / f"{etf.fmp.replace('.', '_')}.parquet"
            if fmp_file.exists():
                df = pd.read_parquet(fmp_file)
                col = "close" if "close" in df.columns else "adj_close"
                dr = df[col].pct_change(1)
                dr.index = pd.to_datetime(dr.index).tz_localize(None)
                daily_ret_etf[etf.bourso] = dr
        dr_panel = pd.DataFrame(daily_ret_etf).reindex(port_returns.index).fillna(0)

        common_etfs = [e for e in w.columns if e in dr_panel.columns]
        if common_etfs:
            # Per-ETF P&L in $ = sum over days of (weight × daily_return × portfolio_value_that_day)
            eq = (1 + port_returns).cumprod() * INITIAL_CAPITAL
            # Weighted daily $ contribution per ETF
            daily_pnl = w[common_etfs] * dr_panel[common_etfs]
            # Scale by portfolio value each day
            daily_pnl_dollar = daily_pnl.multiply(eq.shift(1).fillna(INITIAL_CAPITAL), axis=0)
            contrib_dollar = daily_pnl_dollar.sum()  # cumulative $ P&L per ETF
            contrib_dollar = contrib_dollar.sort_values()

            bar_colors = ["#d62728" if v < 0 else "#2ca02c" for v in contrib_dollar.values]
            etf_names = [BY_BOURSO[e].name[:50] if e in BY_BOURSO else e for e in contrib_dollar.index]

            total_gain = contrib_dollar[contrib_dollar > 0].sum()
            total_loss = contrib_dollar[contrib_dollar < 0].sum()
            net = total_gain + total_loss

            fig3, ax3 = plt.subplots(figsize=(12, max(8, len(contrib_dollar) * 0.35)))
            fig3.suptitle(
                f"ETF P&L ($10,000 initial) — Gain: +${total_gain:,.0f}  Loss: -${abs(total_loss):,.0f}  Net: ${net:+,.0f}",
                fontsize=11, fontweight="bold")
            ax3.barh(range(len(contrib_dollar)), contrib_dollar.values, color=bar_colors, alpha=0.8, height=0.7)
            ax3.set_yticks(range(len(contrib_dollar)))
            ax3.set_yticklabels(etf_names, fontsize=8)
            ax3.set_xlabel("P&L Contribution ($)")
            ax3.axvline(0, color="black", linewidth=0.5)
            ax3.set_facecolor("#f8f8f8")
            ax3.grid(True, alpha=0.3, axis="x")
            # Format x-axis as dollars
            ax3.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"${x:,.0f}"))
            plt.tight_layout()
            fig3.savefig(out_dir / "backtest_pnl.png", dpi=150, bbox_inches="tight")
            plt.close(fig3)


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

    # Load actual daily returns for all ETFs (from price data)
    from etf import UNIVERSE, BY_BOURSO
    daily_returns_all = {}
    for etf in UNIVERSE:
        fmp_file = DATA / f"{etf.fmp.replace('.', '_')}.parquet"
        if fmp_file.exists():
            df = pd.read_parquet(fmp_file)
            col = "close" if "close" in df.columns else "adj_close"
            dr = df[col].pct_change(1)
            dr.index = pd.to_datetime(dr.index).tz_localize(None)
            daily_returns_all[etf.bourso] = dr
    daily_ret_panel = pd.DataFrame(daily_returns_all)
    print(f"Loaded daily returns for {len(daily_returns_all)} ETFs")

    # Filter steps: only keep test periods starting from START_YEAR
    START_YEAR = 2014
    all_steps = sorted(oos["step"].unique())
    steps = []
    for s in all_steps:
        test_dates = oos[(oos["step"] == s) & (oos["split"] == "test")].index.get_level_values("date")
        if len(test_dates) > 0 and test_dates.min().year >= START_YEAR:
            steps.append(s)
    print(f"Walk-forward steps: {len(steps)}  "
          f"val rows: {(oos['split']=='val').sum()}  "
          f"test rows: {(oos['split']=='test').sum()}\n")

    all_test_returns = []
    all_test_weights = []
    all_params_rows  = []
    carry_weights    = None   # chain positions between steps

    print(f"{'Step':>4}  {'Val period':>24}  {'Val↗':>7}  "
          f"{'Test period':>24}  {'Test↗':>7}")
    print("-" * 75)

    for step in steps:
        val_data  = oos[(oos["step"] == step) & (oos["split"] == "val")]
        test_data = oos[(oos["step"] == step) & (oos["split"] == "test")]

        if len(val_data) < 50 or len(test_data) == 0:
            print(f"{step:4d}  SKIP (val={len(val_data)} rows, test={len(test_data)} rows)")
            continue

        # --- CMA-ES on val predictions (last 5 years only) ---
        val_sw_A, val_sw_B = _pivot_step(val_data)
        # Limit val to last 5 years (~1250 trading days)
        if len(val_sw_A) > 1250:
            val_sw_A = val_sw_A.iloc[-1250:]
            val_sw_B = val_sw_B.iloc[-1250:]
        val_is_eq, val_is_def = _section_arrays(val_sw_A.columns.tolist(), sections)
        val_dr = daily_ret_panel.reindex(index=val_sw_A.index, columns=val_sw_A.columns).fillna(0)
        best_params, val_sharpe = run_cmaes(val_sw_A, val_sw_B, val_dr, val_is_eq, val_is_def, vix_s, hy_z60_s)

        # --- Evaluate on test (true OOS) ---
        test_sw_A, test_sw_B = _pivot_step(test_data)
        test_is_eq, test_is_def = _section_arrays(test_sw_A.columns.tolist(), sections)

        # Remap carry_weights to current ETF columns
        prev_w = None
        if carry_weights is not None:
            prev_w = np.zeros(len(test_sw_A.columns))
            for i, etf in enumerate(test_sw_A.columns):
                if etf in carry_weights:
                    prev_w[i] = carry_weights[etf]

        # Daily returns for test period
        test_dr = daily_ret_panel.reindex(index=test_sw_A.index, columns=test_sw_A.columns).fillna(0)
        test_returns, test_weights = run_backtest(test_sw_A, test_sw_B, test_dr,
                                                   test_is_eq, test_is_def,
                                                   vix_s, hy_z60_s, best_params,
                                                   prev_weights=prev_w)

        # Save final weights for next step
        carry_weights = dict(zip(test_weights.columns, test_weights.iloc[-1].values))
        test_sh = sharpe(test_returns)

        val_dates  = val_data.index.get_level_values("date")
        test_dates = test_data.index.get_level_values("date")
        print(
            f"{step:4d}  "
            f"[{val_dates.min().date()} → {val_dates.max().date()}]  {val_sharpe:7.3f}  "
            f"[{test_dates.min().date()} → {test_dates.max().date()}]  {test_sh:7.3f}",
            flush=True,
        )

        all_test_returns.append(test_returns)
        all_test_weights.append(test_weights)
        all_params_rows.append({"step": step, **dict(zip(PARAM_NAMES, best_params))})

        # Live equity curve update after each step
        tmp_returns = pd.concat(all_test_returns).sort_index()
        tmp_weights = pd.concat(all_test_weights).sort_index()
        tmp_eq = (1 + tmp_returns).cumprod()
        tmp_sharpe = sharpe(tmp_returns)
        tmp_dd = (tmp_eq / tmp_eq.cummax() - 1).min()
        print(f"       cumul: {tmp_eq.iloc[-1]-1:+.1%}  sharpe={tmp_sharpe:.2f}  dd={tmp_dd:.1%}", flush=True)
        params_df = pd.DataFrame(all_params_rows)
        _save_equity_png(tmp_returns, tmp_eq, tmp_weights, params_df, OUTPUTS)

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
    all_weights = pd.concat(all_test_weights).sort_index()
    _save_equity_png(port_returns, eq_curve, all_weights, params_df, OUTPUTS)

    print(f"\nSaved → data/backtest_results.parquet")
    print(f"Saved → outputs/best_params.csv")
    print(f"Saved → outputs/cmaes_steps.csv")
    print(f"Saved → outputs/backtest_equity.csv")
    print(f"Saved → outputs/backtest_equity.png")


if __name__ == "__main__":
    main()
