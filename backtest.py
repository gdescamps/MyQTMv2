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

# Geo/sector redirect disabled — allocation libre
GEO_SECTOR_REDIRECT = []

PARAM_NAMES = [
    "temperature_A", "temperature_B",
]
PARAM_INIT   = [1.0, 1.0]
PARAM_SIGMA0 = 0.3
PARAM_BOUNDS = [
    ( 0.01, 1.00),   # temperature_A — softmax concentration (CMA-ES optimized)
    ( 0.01, 1.00),   # temperature_B — softmax concentration (CMA-ES optimized)
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
    params: list,
    transaction_cost: float = 0.0022,
    prev_weights: np.ndarray | None = None,
    block_parity: pd.Series | None = None,
    sharpe_weights: np.ndarray | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Vectorised portfolio simulation.
    Score > 0 → on, softmax(score/temp) × Sharpe weights.
    """
    params = clip_params(params)
    temp_A, temp_B = params

    dates   = scores_wide_A.index
    n_etfs  = scores_wide_A.shape[1]
    etf_list = scores_wide_A.columns.tolist()

    daily_ret = daily_returns_wide.values

    scores_A = scores_wide_A.values
    scores_B = scores_wide_B.values if scores_wide_B is not None else scores_A

    # Detect if A and B are the same scores (test uses model A only)
    a_only = np.allclose(np.nan_to_num(scores_A), np.nan_to_num(scores_B), atol=1e-10)

    # Use Sharpe weights if provided, else equal weight
    sw = sharpe_weights if sharpe_weights is not None else np.ones(n_etfs)

    def _softmax_sharpe_alloc(scores: np.ndarray, T: float, sw: np.ndarray) -> np.ndarray:
        """Score > 0 → on, softmax(score/T) × Sharpe weights among positives."""
        out = np.zeros_like(scores)
        for i in range(scores.shape[0]):
            row = scores[i]
            valid = ~np.isnan(row)
            on = valid & (row > 0)
            if not on.any():
                continue  # all cash
            s = row[on] / max(T, 1e-6)
            s = s - s.max()
            e = np.exp(s)
            softmax_w = e / e.sum()
            combined = softmax_w * sw[on]
            total = combined.sum()
            if total > 0:
                out[i, on] = combined / total
        return out

    if a_only:
        weights = _softmax_sharpe_alloc(scores_A, temp_A, sw)
    elif block_parity is not None:
        alloc_A = _softmax_sharpe_alloc(scores_A, temp_A, sw)
        alloc_B = _softmax_sharpe_alloc(scores_B, temp_B, sw)
        bp = block_parity.reindex(dates).values
        use_A = (bp == 1)[:, None]
        weights = np.where(use_A, alloc_A, alloc_B)
    else:
        weights = _softmax_sharpe_alloc(scores_A, temp_A, sw)
    # Remaining = cash (implicit: 1 - sum(weights))

    # Weekly rebalancing: refresh weights every 5 trading days, no fees
    REBAL_DAYS = 5
    fixed_weights = np.zeros_like(weights)
    for i in range(len(dates)):
        rebal_idx = (i // REBAL_DAYS) * REBAL_DAYS
        fixed_weights[i] = weights[rebal_idx]

    # Portfolio simulation with real daily returns (no transaction costs)
    safe_ret = np.where(np.isnan(daily_ret), 0.0, daily_ret)
    port_ret = (fixed_weights * safe_ret).sum(axis=1)

    weights = fixed_weights  # for weight tracking

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
    block_parity: pd.Series | None = None,
    maxiter: int = 100,
) -> tuple[list, float]:
    """
    Run CMA-ES on val predictions for one walk-forward step.
    Returns (best_params, best_objective).
    """
    def objective(params: list) -> float:
        returns, _ = run_backtest(scores_wide_A, scores_wide_B, daily_returns_wide, params,
                                  block_parity=block_parity)
        r = returns.dropna()
        if len(r) < 30 or r.std() < 1e-10:
            return 10.0
        ann_ret = r.mean() * 252
        eq = (1 + r).cumprod()
        max_dd = abs((eq / eq.cummax() - 1).min())
        if max_dd < 1e-10:
            max_dd = 1e-10
        # ret^2 / |max_dd| — rewards high returns quadratically
        sign = 1.0 if ann_ret >= 0 else -1.0
        return -(sign * ann_ret ** 2 / max_dd)

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


def _pivot_step(step_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series | None]:
    """Pivot (date, etf_id) long → wide scores_A, scores_B + block_parity per date."""
    has_AB = ("score_A" in step_data.columns and
              step_data["score_A"].notna().any())
    if has_AB:
        sw_A = step_data["score_A"].unstack("etf_id").sort_index()
        sw_B = step_data["score_B"].unstack("etf_id").sort_index()
        if "block_parity" in step_data.columns:
            bp = step_data["block_parity"].groupby("date").first().reindex(sw_A.index)
        else:
            bp = None
    else:
        sw_A = step_data["score"].unstack("etf_id").sort_index()
        sw_B = sw_A
        bp = None
    return sw_A, sw_B, bp


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
                     params_df: pd.DataFrame, out_dir: Path,
                     scores_A: pd.DataFrame | None = None) -> None:
    """Save chart: equity, allocation, temp_A + scores_A."""
    from etf import BY_BOURSO
    ann_ret = port_returns.mean() * 252
    ann_vol = port_returns.std() * np.sqrt(252)
    sh      = sharpe(port_returns)
    max_dd  = (eq_curve / eq_curve.cummax() - 1).min()
    title = (f"MyQTM-ETF — Walk-Forward OOS  "
             f"(Sharpe={sh:.2f}  Ann={ann_ret:.1%}  Vol={ann_vol:.1%}  MaxDD={max_dd:.1%})")

    # --- Unified color map (ticker → color), shared between panels ---
    etf_color_map = {
        "IVV": "#ff7f0e",     # S&P 500 = orange
        "SOXX": "#9467bd",    # Semiconductors = violet
        "GLD": "#d4af37",     # Gold = gold
        "EEM": "#2ca02c",     # Emerging = green
        "TLT": "#17becf",     # Treasury = cyan
        "IEO": "#8b4513",     # Oil & Gas = brown
        "EXX1.DE": "#e377c2", # Euro Banks = pink
        "QQQ": "#2ca02c",     # Nasdaq = green
    }
    # Simplified display names (remove iShares, ETF, Shares, etc.)
    SHORT_NAMES = {
        "IVV": "S&P 500",
        "SOXX": "Semiconductors",
        "GLD": "Gold",
        "EEM": "Emerging Mkts",
        "TLT": "Treasury 20y+",
        "IEO": "Oil & Gas",
        "EXX1.DE": "Euro Banks",
        "QQQ": "Nasdaq 100",
    }
    # Tickers to show in bold on equity chart (besides portfolio)
    BOLD_TICKERS = {"IVV", "GLD", "IEO"}

    from etf import UNIVERSE as _UNIVERSE

    # --- Chart: Equity + Allocation + Temp/Scores ---
    fig1 = plt.figure(figsize=(24, 13))
    gs = fig1.add_gridspec(2, 2, width_ratios=[3, 1], height_ratios=[3, 2],
                           hspace=0.08, wspace=0.02)
    ax1 = fig1.add_subplot(gs[0, 0])
    ax2 = fig1.add_subplot(gs[1, 0], sharex=ax1)
    ax_leg = fig1.add_subplot(gs[:, 1])  # right column spans both rows
    ax_leg.axis("off")
    fig1.suptitle(title, fontsize=13, fontweight="bold")

    # Portfolio: very bold red
    ax1.plot(eq_curve.index, eq_curve.values, lw=3.5, color="#d62728", zorder=10)

    # Compute portfolio Sharpe & max DD
    port_sh = sharpe(port_returns)
    port_dd = (eq_curve / eq_curve.cummax() - 1).min()

    # All ETF curves — collect (final_value, sharpe, max_dd, short_name, color)
    etf_curves = []
    for etf in _UNIVERSE:
        ticker = etf.bourso
        color = etf_color_map.get(ticker, "#7f7f7f")
        bm = _load_benchmark(ticker, eq_curve.index)
        if bm is not None:
            is_bold = ticker in BOLD_TICKERS
            lw = 2.8 if is_bold else 1.0
            alpha = 0.9 if is_bold else 0.5
            zorder = 5 if is_bold else 2
            short = SHORT_NAMES.get(ticker, ticker)
            ax1.plot(bm.index, bm.values, lw=lw, color=color, alpha=alpha, zorder=zorder)
            bm_ret = bm.pct_change().dropna()
            etf_sh = sharpe(bm_ret)
            etf_dd = (bm / bm.cummax() - 1).min()
            etf_curves.append((bm.iloc[-1], etf_sh, etf_dd, short, color))

    # Sorted list: portfolio + ETFs all sorted by final value (descending)
    from matplotlib.lines import Line2D
    all_items = [(eq_curve.iloc[-1], port_sh, port_dd, "Portfolio", "#d62728", True)] + \
                [(v, sh, dd, s, c, False) for v, sh, dd, s, c in etf_curves]
    sorted_items = sorted(all_items, key=lambda x: -x[0])

    # Sort by Sharpe (descending) for right-side list
    sorted_by_sharpe = sorted(sorted_items, key=lambda x: -x[1])

    ax1.set_yscale("log")
    ax1.set_ylabel("Equity (log scale, base 1)")
    ax1.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:.1f}x"))
    ax1.grid(True, alpha=0.3, which="both")
    ax1.set_facecolor("#f8f8f8")
    vix_path = DATA / "fred_vix.parquet"
    if vix_path.exists():
        vix_raw = pd.read_parquet(vix_path).iloc[:, 0]
        vix_raw = vix_raw.reindex(eq_curve.index, method="ffill").dropna()
        ax1b = ax1.twinx()
        ax1b.fill_between(vix_raw.index, vix_raw.values, alpha=0.10, color="#d62728")
        ax1b.set_ylabel("VIX", color="#d62728", fontsize=9)
        ax1b.tick_params(axis="y", labelcolor="#d62728", labelsize=8)
        ax1b.set_ylim(0, 80)

    # Panel 2: Allocation — ETFs sorted by total allocation volume (descending)
    if len(weights_df) > 0:
        w = weights_df.reindex(eq_curve.index, method="ffill").fillna(0)
        cash = (1 - w.sum(axis=1)).clip(0, 1)

        # Sort by cumulative allocation volume (descending), keep only allocated
        avg_w = w.mean().sort_values(ascending=False)
        allocated = avg_w[avg_w > 0.001].index.tolist()
        w_sorted = w[allocated]

        # Rename with short names
        col_names = {e: SHORT_NAMES.get(e, e) for e in w_sorted.columns}
        w_named = w_sorted.rename(columns=col_names)

        plot_data = w_named.copy()
        plot_data["Cash"] = cash

        # Build color list matching short names
        color_map = {SHORT_NAMES.get(t, t): c for t, c in etf_color_map.items()}
        color_map["Cash"] = "#e8e8e8"

        colors = [color_map.get(c, "#7f7f7f") for c in plot_data.columns]
        ax2.stackplot(plot_data.index, plot_data.values.T,
                      labels=plot_data.columns, colors=colors, alpha=0.85)
        ax2.set_ylabel("Allocation")
        ax2.set_ylim(0, 1)
        ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax2.set_facecolor("#f8f8f8")
        ax2.grid(True, alpha=0.3)

    # Right column: ETF list sorted by Sharpe
    n_items = len(sorted_by_sharpe)
    y_start = 0.95
    y_step = min(0.035, 0.9 / max(n_items, 1))
    for i, (val, sh, dd, short, color, is_port) in enumerate(sorted_by_sharpe):
        y = y_start - i * y_step
        fw = "bold" if is_port else "normal"
        fs = 11 if is_port else 10
        ax_leg.text(0.0, y, "■", fontsize=14, color=color, va="center",
                    transform=ax_leg.transAxes)
        ax_leg.text(0.08, y, f"{short}", fontsize=fs, fontweight=fw, va="center",
                    transform=ax_leg.transAxes)
        ax_leg.text(0.55, y, f"{val:.1f}x", fontsize=fs, va="center",
                    transform=ax_leg.transAxes)
        ax_leg.text(0.72, y, f"Sh={sh:.2f}", fontsize=fs, va="center",
                    transform=ax_leg.transAxes, color="#333333")
        ax_leg.text(0.95, y, f"{dd:.0%}", fontsize=fs, va="center", ha="right",
                    transform=ax_leg.transAxes, color="#cc0000" if dd < -0.30 else "#666666")
    import subprocess
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    n_etfs = len(weights_df.columns) if len(weights_df) > 0 else 0
    depth = 7  # current XGB max_depth
    fname = f"backtest_{sha}_{n_etfs}etf_d{depth}.jpg"
    fig1.savefig(out_dir / fname, dpi=100, bbox_inches="tight", format="jpeg", pil_kwargs={"quality": 60, "optimize": True})
    fig1.savefig(out_dir / "backtest_equity.jpg", dpi=100, bbox_inches="tight", format="jpeg", pil_kwargs={"quality": 60, "optimize": True})
    plt.close(fig1)



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
    START_YEAR = 2008
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
    all_test_scores  = []   # scores_A per test step
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
        val_sw_A, val_sw_B, val_bp = _pivot_step(val_data)
        # Limit val to last 5 years (~1250 trading days)
        if len(val_sw_A) > 1250:
            val_sw_A = val_sw_A.iloc[-1250:]
            val_sw_B = val_sw_B.iloc[-1250:]
            if val_bp is not None:
                val_bp = val_bp.reindex(val_sw_A.index)
        # Z-score val scores cross-sectionally (same as test)
        v_mean = val_sw_A.mean(axis=1)
        v_std = val_sw_A.std(axis=1).replace(0, 1.0)
        val_sw_A = val_sw_A.sub(v_mean, axis=0).div(v_std, axis=0)
        v_mean_B = val_sw_B.mean(axis=1)
        v_std_B = val_sw_B.std(axis=1).replace(0, 1.0)
        val_sw_B = val_sw_B.sub(v_mean_B, axis=0).div(v_std_B, axis=0)

        val_dr = daily_ret_panel.reindex(index=val_sw_A.index, columns=val_sw_A.columns).fillna(0)
        best_params, val_sharpe = run_cmaes(val_sw_A, val_sw_B, val_dr, block_parity=val_bp)

        # --- Evaluate on test (true OOS) ---
        test_sw_A, test_sw_B, _ = _pivot_step(test_data)

        # Remap carry_weights to current ETF columns
        prev_w = None
        if carry_weights is not None:
            prev_w = np.zeros(len(test_sw_A.columns))
            for i, etf in enumerate(test_sw_A.columns):
                if etf in carry_weights:
                    prev_w[i] = carry_weights[etf]

        # Z-score scores cross-sectionally before softmax
        row_mean = test_sw_A.mean(axis=1)
        row_std = test_sw_A.std(axis=1).replace(0, 1.0)
        test_sw_A = test_sw_A.sub(row_mean, axis=0).div(row_std, axis=0)
        row_mean_B = test_sw_B.mean(axis=1)
        row_std_B = test_sw_B.std(axis=1).replace(0, 1.0)
        test_sw_B = test_sw_B.sub(row_mean_B, axis=0).div(row_std_B, axis=0)

        # Compute expanding Sharpe per ETF at test date (for allocation weights)
        test_date = test_sw_A.index[0]
        dr_hist = daily_ret_panel[test_sw_A.columns].loc[:test_date]
        exp_mean = dr_hist.expanding(min_periods=60).mean().iloc[-1] * 252
        exp_std = dr_hist.expanding(min_periods=60).std().iloc[-1] * np.sqrt(252)
        etf_sharpe = (exp_mean / exp_std.replace(0, np.nan)).clip(0.0).fillna(0.0).values

        # Daily returns for test period
        test_dr = daily_ret_panel.reindex(index=test_sw_A.index, columns=test_sw_A.columns).fillna(0)
        test_returns, test_weights = run_backtest(test_sw_A, test_sw_B, test_dr,
                                                   best_params, prev_weights=prev_w,
                                                   sharpe_weights=etf_sharpe)

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
        # Save scores_A for chart (first day of test step per ETF)
        all_test_scores.append(test_sw_A.iloc[[0]])

        # Live equity curve update after each step
        tmp_returns = pd.concat(all_test_returns).sort_index()
        tmp_weights = pd.concat(all_test_weights).sort_index()
        tmp_eq = (1 + tmp_returns).cumprod()
        tmp_sharpe = sharpe(tmp_returns)
        tmp_dd = (tmp_eq / tmp_eq.cummax() - 1).min()
        print(f"       cumul: {tmp_eq.iloc[-1]-1:+.1%}  sharpe={tmp_sharpe:.2f}  dd={tmp_dd:.1%}", flush=True)
        params_df = pd.DataFrame(all_params_rows)
        tmp_scores = pd.concat(all_test_scores).sort_index() if all_test_scores else pd.DataFrame()
        _save_equity_png(tmp_returns, tmp_eq, tmp_weights, params_df, OUTPUTS,
                         scores_A=tmp_scores)

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
    all_scores = pd.concat(all_test_scores).sort_index() if all_test_scores else pd.DataFrame()
    _save_equity_png(port_returns, eq_curve, all_weights, params_df, OUTPUTS,
                     scores_A=all_scores)

    print(f"\nSaved → data/backtest_results.parquet")
    print(f"Saved → outputs/best_params.csv")
    print(f"Saved → outputs/cmaes_steps.csv")
    print(f"Saved → outputs/backtest_equity.csv")
    print(f"Saved → outputs/backtest_equity.png")


if __name__ == "__main__":
    main()
