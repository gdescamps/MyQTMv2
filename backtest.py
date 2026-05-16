"""
Portfolio backtest for MyQTM-ETF with fixed allocation params.

Walk-forward structure (matches train.py):
  For each step, apply fixed params to test predictions (split="test") → true OOS returns.

Allocation model:
  Score > 0 → on, softmax(score/temperature) × Sharpe weights among positives.
  Rebalance every N days with AV arbitrage delays (J+0 sell, J+1 buy with cash, J+2 buy with settled).

Output:
  data/backtest_results.parquet   (daily portfolio returns, test periods only)
  outputs/best_params.csv         (per-step params)
  outputs/backtest_equity.csv     (equity curve)
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
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

AV_DELAYS = False  # True = délais arbitrage AV (J+0 sell, J+1 buy cash dispo, J+2 buy settled)
USE_SOFTMAX = True  # True = softmax(score/T) × Sharpe, False = equal weight among score > 0 × Sharpe
SHARPE_POWER = 0.8  # Sharpe weight = sharpe^SHARPE_POWER (0=equal, 1=linear, 2=concentrated)

TEMPERATURE = 1.0     # softmax concentration
REBAL_DAYS = 4       # jours entre rebalances


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def softmax(scores: np.ndarray, temperature: float) -> np.ndarray:
    s = scores / max(temperature, 1e-6)
    s = s - s.max()
    e = np.exp(s)
    return e / e.sum()




def run_backtest(
    scores_wide: pd.DataFrame,
    daily_returns_wide: pd.DataFrame,
    prev_weights: np.ndarray | None = None,
    sharpe_weights: np.ndarray | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Vectorised portfolio simulation.
    Score > 0 → on, softmax(score/temp) × Sharpe weights.
    """
    dates   = scores_wide.index
    n_etfs  = scores_wide.shape[1]
    etf_list = scores_wide.columns.tolist()

    daily_ret = daily_returns_wide.values
    scores = scores_wide.values

    # Use Sharpe weights if provided, else equal weight
    sw = sharpe_weights if sharpe_weights is not None else np.ones(n_etfs)
    rebal_days = REBAL_DAYS

    def _alloc(scores: np.ndarray, sw: np.ndarray) -> np.ndarray:
        """Score > 0 → on, softmax(score/T) × Sharpe weights among positives."""
        out = np.zeros_like(scores)
        for i in range(scores.shape[0]):
            row = scores[i]
            valid = ~np.isnan(row)
            on = valid & (row > 0)
            if not on.any():
                continue
            if USE_SOFTMAX:
                s = row[on] / max(TEMPERATURE, 1e-6)
                s = s - s.max()
                e = np.exp(s)
                w = e / e.sum()
            else:
                w = np.ones(on.sum()) / on.sum()
            combined = w * sw[on]
            total = combined.sum()
            if total > 0:
                out[i, on] = combined / total
        return out

    weights = _alloc(scores, sw)
    # Remaining = cash (implicit: 1 - sum(weights))

    # Lot-based portfolio simulation
    # - Daily prediction, rebalance only if allocation changes > rebal_threshold
    # - Rebalance takes 2 days: day 1 = sell (cash), day 2 = buy new allocation
    # - No transaction fees
    LOT_SIZE = 500.0
    INIT_CAPITAL = 10_000.0

    safe_ret = np.where(np.isnan(daily_ret), 0.0, daily_ret)
    n_days = len(dates)

    # Track portfolio
    positions = np.zeros(n_etfs)
    cash = INIT_CAPITAL if prev_weights is None else 0.0
    if prev_weights is not None:
        total_val = INIT_CAPITAL
        for j in range(n_etfs):
            target_val = prev_weights[j] * total_val
            positions[j] = int(target_val / LOT_SIZE) * LOT_SIZE
        cash = total_val - positions.sum()

    port_returns = np.zeros(n_days)
    actual_weights = np.zeros((n_days, n_etfs))

    # AV arbitrage delays (French life insurance):
    # J+0: decision (order placed), all positions still earn returns
    # J+1: sale executed at J+1 NAV, sold positions earn returns on J+0 and J+1
    # J+2: purchase executed at J+2 NAV, new positions start earning from J+2
    # Untouched positions continue earning returns throughout.
    rebal_state = 0
    pending_target_pos = None
    pending_target_w = None

    def _compute_target_lots(target_w, total_val):
        """Compute target positions in 500€ lots given weights and total value."""
        new_pos = np.zeros(n_etfs)
        remaining = total_val
        ranked = np.argsort(-target_w)
        for j in ranked:
            tv = target_w[j] * total_val
            nl = int(tv / LOT_SIZE)
            alloc = nl * LOT_SIZE
            if alloc > remaining:
                alloc = int(remaining / LOT_SIZE) * LOT_SIZE
            new_pos[j] = alloc
            remaining -= alloc
        for j in ranked:
            if remaining < LOT_SIZE:
                break
            if target_w[j] > 0:
                new_pos[j] += LOT_SIZE
                remaining -= LOT_SIZE
        return new_pos, remaining

    for i in range(n_days):
        total_val = positions.sum() + cash

        if rebal_state == 0:
            # Rebalance every N days
            if i % rebal_days != 0:
                pass  # skip to daily P&L below
            elif total_val >= LOT_SIZE:
                target_w = weights[i]
                # Compute target lots
                target_pos, _ = _compute_target_lots(target_w, total_val)
                # Check if allocation actually changes
                if np.allclose(target_pos, positions):
                    pass  # no change, no delay
                else:
                    if not AV_DELAYS:
                        # Instant rebalance: sell and buy same day
                        positions = target_pos.copy()
                        cash = total_val - positions.sum()
                    else:
                        # J+0: decision — store target weights, positions still earn today
                        pending_target_w = target_w
                        rebal_state = 1  # positions still fully invested today

        elif rebal_state == 1:
            # J+1: sale executed at today's NAV
            # Positions have grown/shrunk with returns since J+0.
            # Recompute target lots based on current total value.
            total_val = positions.sum() + cash
            target_pos, _ = _compute_target_lots(pending_target_w, total_val)
            # Sell positions that are above target
            sale_cash = 0.0
            for j in range(n_etfs):
                if positions[j] > target_pos[j]:
                    sale_cash += positions[j] - target_pos[j]
                    positions[j] = target_pos[j]
            # Buy with already-available cash (cash before sale)
            available_cash = cash
            cash += sale_cash
            if available_cash > 0:
                for j in range(n_etfs):
                    if target_pos[j] > positions[j]:
                        buy_amount = target_pos[j] - positions[j]
                        buyable = min(buy_amount, available_cash)
                        buyable = int(buyable / LOT_SIZE) * LOT_SIZE
                        if buyable > 0:
                            positions[j] += buyable
                            cash -= buyable
                            available_cash -= buyable
            pending_target_pos = target_pos
            rebal_state = 2

        elif rebal_state == 2:
            # J+2: purchase executed — buy remaining with settled cash from sale
            for j in range(n_etfs):
                if pending_target_pos[j] > positions[j]:
                    buy_amount = pending_target_pos[j] - positions[j]
                    if buy_amount <= cash:
                        positions[j] += buy_amount
                        cash -= buy_amount
                    else:
                        affordable = int(cash / LOT_SIZE) * LOT_SIZE
                        positions[j] += affordable
                        cash -= affordable
            rebal_state = 0
            pending_target_pos = None

        # Daily P&L on current positions (untouched positions always earn)
        total_val = positions.sum() + cash
        if total_val > 0:
            actual_weights[i] = positions / total_val
            daily_pnl = (positions * safe_ret[i]).sum()
            positions = positions * (1 + safe_ret[i])
            port_returns[i] = daily_pnl / total_val
        else:
            port_returns[i] = 0.0

    ret_series = pd.Series(port_returns, index=dates, name="port_return")
    weights_df = pd.DataFrame(actual_weights, index=dates, columns=etf_list)
    return ret_series, weights_df


def sharpe(returns: pd.Series, min_obs: int = 30) -> float:
    r = returns.dropna()
    if len(r) < min_obs or r.std() < 1e-10:
        return -10.0
    return float(r.mean() / r.std() * np.sqrt(252))




def _pivot_step(step_data: pd.DataFrame) -> pd.DataFrame:
    """Pivot (date, etf_id) long → wide scores."""
    if "score_A" in step_data.columns and step_data["score_A"].notna().any():
        scores = step_data["score_A"].unstack("etf_id").sort_index()
    else:
        scores = step_data["score"].unstack("etf_id").sort_index()
    return scores


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
        # Existing (keep)
        "IVV": "#ff7f0e",     # S&P 500 = orange
        "SOXX": "#9467bd",    # Semiconductors = violet
        "GLD": "#d4af37",     # Gold = gold
        "EEM": "#2ca02c",     # Emerging = green
        "TLT": "#17becf",     # Treasury 20y+ = cyan
        "IEO": "#8b4513",     # Oil & Gas = brown
        "EXX1.DE": "#e377c2", # Euro Banks = pink
        "EXV1.DE": "#c49bc8", # Euro Tech = light pink
        "QQQ": "#98df8a",     # Nasdaq 100 = light green
        # US Sectors
        "XLK": "#1f77b4",     # US Tech = blue
        "XLE": "#d62728",     # US Energy = red
        "XLI": "#7f7f7f",     # US Industrials = grey
        "XLY": "#bcbd22",     # US Cons. Disc. = olive
        "XLV": "#ff9896",     # US Healthcare = light red
        "XLP": "#aec7e8",     # US Cons. Staples = light blue
        "XLF": "#c7c7c7",     # US Financials = silver
        "XLU": "#dbdb8d",     # US Utilities = khaki
        "XLB": "#c49c94",     # US Materials = tan
        # Countries
        "EWJ": "#393b79",     # Japan = navy
        "EWC": "#e7969c",     # Canada = salmon
        "EWY": "#7b4173",     # South Korea = plum
        "EWZ": "#a55194",     # Brazil = magenta
        "EWW": "#ce6dbd",     # Mexico = orchid
        "FXI": "#de9ed6",     # China = light violet
        "TUR": "#ad494a",     # Turkey = brick red
        # Bonds
        "IEF": "#6b6ecf",     # Treasury 7-10y = indigo
        "HYG": "#b5cf6b",     # High Yield = lime
        "TIP": "#e7ba52",     # TIPS = amber
        # Alternatif
        "RING": "#8c6d31",    # Gold Miners = dark gold
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
        "EXV1.DE": "Euro Tech",
        "QQQ": "Nasdaq 100",
        "XLK": "US Tech",
        "XLE": "US Energy",
        "XLI": "US Industrials",
        "XLY": "US Cons. Disc.",
        "XLV": "US Healthcare",
        "XLP": "US Cons. Staples",
        "XLF": "US Financials",
        "XLU": "US Utilities",
        "XLB": "US Materials",
        "EWJ": "Japan",
        "EWC": "Canada",
        "EWY": "South Korea",
        "EWZ": "Brazil",
        "EWW": "Mexico",
        "FXI": "China",
        "TUR": "Turkey",
        "IEF": "Treasury 7-10y",
        "HYG": "High Yield",
        "TIP": "TIPS Inflation",
        "RING": "Gold Miners",
    }
    # Tickers to show in bold on equity chart (besides portfolio)
    BOLD_TICKERS = {"IVV", "GLD", "IEO"}

    from etf import UNIVERSE as _UNIVERSE

    # --- Chart: Equity + Allocation + Temp/Scores ---
    fig1 = plt.figure(figsize=(24, 14))
    gs = fig1.add_gridspec(2, 2, width_ratios=[3, 1], height_ratios=[3, 2],
                           hspace=0.08, wspace=0.02,
                           top=0.98, bottom=0.04, left=0.05, right=0.98)
    ax1 = fig1.add_subplot(gs[0, 0])
    ax2 = fig1.add_subplot(gs[1, 0], sharex=ax1)
    ax_leg = fig1.add_subplot(gs[:, 1])  # right column spans both rows
    ax_leg.axis("off")
    # Title at top of right column
    # Compute trades per month (changes in allocation)
    if len(weights_df) > 0:
        w = weights_df.reindex(eq_curve.index, method="ffill").fillna(0)
        # A trade = any ETF weight changes from 0 to >0 or >0 to 0
        active = (w > 0.01).astype(int)
        trades = active.diff().abs().sum(axis=1)  # count of on/off switches per day
        n_months = max(len(port_returns) / 21, 1)
        trades_per_month = trades.sum() / n_months
    else:
        trades_per_month = 0

    ax_leg.text(0.0, 1.00, f"Ann       {ann_ret:.1%}", fontsize=18, fontweight="bold", va="top", transform=ax_leg.transAxes)
    ax_leg.text(0.0, 0.95, f"Vol        {ann_vol:.1%}", fontsize=18, va="top", transform=ax_leg.transAxes, color="#333333")
    ax_leg.text(0.0, 0.90, f"Trades  {trades_per_month:.1f}/mois", fontsize=18, va="top", transform=ax_leg.transAxes, color="#333333")
    ax_leg.text(0.0, 0.85, f"MaxDD  {max_dd:.1%}", fontsize=18, va="top", transform=ax_leg.transAxes, color="#cc0000")

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
        ax1b.set_yticks([])
        ax1b.set_ylabel("")
        ax1b.set_ylim(0, 80)

    # Panel 2: Allocation — ETFs sorted by total allocation volume (descending)
    if len(weights_df) > 0:
        w = weights_df.reindex(eq_curve.index, method="ffill").fillna(0)
        cash = (1 - w.sum(axis=1)).clip(0, 1)

        # Fixed order: universe order (deterministic), keep only allocated
        universe_order = [e.bourso for e in _UNIVERSE]
        allocated = [t for t in universe_order if t in w.columns and w[t].mean() > 0.001]
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
    y_start = 0.80
    y_step = min(0.035, 0.9 / max(n_items, 1))
    for i, (val, sh, dd, short, color, is_port) in enumerate(sorted_by_sharpe):
        y = y_start - i * y_step
        fw = "bold" if is_port else "normal"
        fs = 11 if is_port else 10
        ax_leg.text(0.0, y, "■", fontsize=28, color=color, va="center",
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
        proxy_file = DATA / f"{etf.proxy.replace('.', '_')}.parquet"
        if proxy_file.exists():
            df = pd.read_parquet(proxy_file)
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

    print(f"{'Step':>4}  {'Test period':>24}  {'Sharpe':>7}")
    print("-" * 45)

    for step in steps:
        val_data  = oos[(oos["step"] == step) & (oos["split"] == "val")]
        test_data = oos[(oos["step"] == step) & (oos["split"] == "test")]

        if len(val_data) < 50 or len(test_data) == 0:
            print(f"{step:4d}  SKIP (val={len(val_data)} rows, test={len(test_data)} rows)")
            continue


        # Compute expanding Sharpe per ETF (for allocation weights)
        val_scores = _pivot_step(val_data)
        val_end_date = val_scores.index[-1]
        dr_hist = daily_ret_panel[val_scores.columns].loc[:val_end_date]
        exp_mean = dr_hist.expanding(min_periods=60).mean().iloc[-1] * 252
        exp_std = dr_hist.expanding(min_periods=60).std().iloc[-1] * np.sqrt(252)
        etf_sharpe = (exp_mean / exp_std.replace(0, np.nan)).clip(0.0).fillna(0.0).values
        etf_sharpe = etf_sharpe ** SHARPE_POWER

        # --- Evaluate on test (true OOS) ---
        test_scores = _pivot_step(test_data)

        # Remap carry_weights to current ETF columns
        prev_w = None
        if carry_weights is not None:
            prev_w = np.zeros(len(test_scores.columns))
            for i, etf in enumerate(test_scores.columns):
                if etf in carry_weights:
                    prev_w[i] = carry_weights[etf]

        # Z-score scores cross-sectionally before softmax
        row_mean = test_scores.mean(axis=1)
        row_std = test_scores.std(axis=1).replace(0, 1.0)
        test_scores = test_scores.sub(row_mean, axis=0).div(row_std, axis=0)

        # Daily returns for test period
        test_dr = daily_ret_panel.reindex(index=test_scores.index, columns=test_scores.columns).fillna(0)
        test_returns, test_weights = run_backtest(test_scores, test_dr,
                                                   prev_weights=prev_w,
                                                   sharpe_weights=etf_sharpe)

        # Save final weights for next step
        carry_weights = dict(zip(test_weights.columns, test_weights.iloc[-1].values))
        test_sh = sharpe(test_returns)

        test_dates = test_data.index.get_level_values("date")
        print(
            f"{step:4d}  "
            f"[{test_dates.min().date()} → {test_dates.max().date()}]  {test_sh:7.3f}",
            flush=True,
        )

        all_test_returns.append(test_returns)
        all_test_weights.append(test_weights)
        all_params_rows.append({"step": step, "temperature": TEMPERATURE, "rebal_days": REBAL_DAYS})
        # Save scores_A for chart (first day of test step per ETF)
        all_test_scores.append(test_scores.iloc[[0]])

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
    print("=== Final OOS backtest ===")
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

    pd.DataFrame(all_params_rows).to_csv(OUTPUTS / "backtest_steps.csv", index=False)

    # Equity curve PNG
    all_weights = pd.concat(all_test_weights).sort_index()
    all_scores = pd.concat(all_test_scores).sort_index() if all_test_scores else pd.DataFrame()
    _save_equity_png(port_returns, eq_curve, all_weights, params_df, OUTPUTS,
                     scores_A=all_scores)

    print(f"\nSaved → data/backtest_results.parquet")
    print(f"Saved → outputs/best_params.csv")
    print(f"Saved → outputs/backtest_steps.csv")
    print(f"Saved → outputs/backtest_equity.csv")
    print(f"Saved → outputs/backtest_equity.png")


if __name__ == "__main__":
    main()
