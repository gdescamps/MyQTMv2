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
import subprocess
import threading
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

USE_SOFTMAX = True  # True = softmax(score/T) × Sharpe, False = equal weight among score > 0 × Sharpe
SHARPE_POWER = 0.0  # Sharpe weight = sharpe^SHARPE_POWER (0=equal, 1=linear, 2=concentrated)

TEMPERATURE = 1.0     # softmax concentration (fixed)
REBAL_DAYS = 5       # jours entre rebalances
TOP_N_ALLOC   = 3     # allocate to top 3
TOP_N_MONITOR = 3     # monitor top 3 — rebalance only when allocated asset leaves top 3
TOP_N_SCORES  = 3     # (legacy, used as fallback)
VIX_ADAPTIVE = False  # adjust TOP_N and REBAL based on VIX level
VIX_LOW = 15         # below this: calm market
VIX_HIGH = 25        # above this: crisis
TOP_N_LOW = 3        # top N when VIX < VIX_LOW (calm → concentrate on best scores)
TOP_N_HIGH = 3       # top N when VIX > VIX_HIGH (crisis → concentrate)
REBAL_DAYS_LOW = 21  # rebalance frequency when VIX < VIX_LOW (calm → slower)
REBAL_DAYS_HIGH = 2  # rebalance frequency when VIX > VIX_HIGH (crisis → faster)
VIX_SPIKE_MIN = 4.0        # VIX 5-day change above this → start reducing allocation
VIX_SPIKE_MAX = 10.0       # VIX 5-day change above this → max reduction
VIX_SPIKE_REBAL = 1        # rebalance every day during spike
CAP_AT_SPIKE_MIN = 0.8     # allocation cap when slope = VIX_SPIKE_MIN
CAP_AT_SPIKE_MAX = 0.4     # allocation cap when slope >= VIX_SPIKE_MAX
RECOVERY_RATE = 0.20       # restore +20% allocation per step until next steep ascending slope
FLAT_TAX_RATE = 0.30       # PFU 30% on realized gains (paid Jan 1st)
INIT_CAPITAL  = 150_000.0  # portfolio starting capital

# --- Robustness backtest (run in parallel by main) ---
N_RUNS = 50                # number of perturbed backtests
N_DROP = (5, 10)           # random 5-10 ETFs dropped at each rebalance

# --- Correlated-block diversification (idea #2) ---
# ETFs sharing a block are near-duplicates (weekly-return corr > 0.9, confirmed
# by asset class). The top-N pre-filter keeps at most ONE ETF per block, so the
# allocator cannot go all-in on a single bet (e.g. IVV+SUSA+ACWI = 3x S&P 500).
# Any ETF not listed is its own singleton block.
BLOCK_DIVERSIFY = False
CORR_BLOCKS = {
    "IVV": "US",   "QQQ": "US",  "ACWI": "US", "SUSA": "US",   # US large/mega cap
    "IEUR": "EUR", "EZU": "EUR",                               # Europe / eurozone
    "EEM": "EM",   "IEMG": "EM", "EMXC": "EM",                 # broad emerging mkts
    "ILF": "LATAM", "EWZ": "LATAM",                            # Latin America
}


def _topn_keep(row: np.ndarray, n: int, block_ids: list) -> np.ndarray:
    """Indices of the top-n positive scores, at most one ETF per correlated
    block. Falls back to pure score order if fewer than n distinct blocks
    carry a positive score."""
    pos = np.where((~np.isnan(row)) & (row > 0))[0]
    if len(pos) <= n:
        return pos
    order = pos[np.argsort(row[pos])[::-1]]          # descending score
    if not BLOCK_DIVERSIFY:
        return order[:n]
    keep, used = [], set()
    for j in order:                                  # pass 1: max 1 per block
        if len(keep) >= n:
            break
        b = block_ids[j]
        if b in used:
            continue
        keep.append(j)
        used.add(b)
    for j in order:                                  # pass 2: fill if short
        if len(keep) >= n:
            break
        if j not in keep:
            keep.append(j)
    return np.array(keep, dtype=int)


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
    rebal_days_override: int | None = None,
    max_alloc: float = 1.0,
    tax_state: dict | None = None,
    monitor_mask: np.ndarray | None = None,
) -> tuple:
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
    rebal_days = rebal_days_override if rebal_days_override is not None else REBAL_DAYS

    def _alloc(scores: np.ndarray, sw: np.ndarray) -> np.ndarray:
        """Score > 0 → on, softmax(score/T) × Sharpe weights (top-N pre-filtered)."""
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
    # Cap total allocation (VIX spike → partial cash)
    if max_alloc < 1.0:
        weights = weights * max_alloc
    # Remaining = cash (implicit: 1 - sum(weights))

    # Lot-based portfolio simulation with realistic fees
    LOT_SIZE = 10_000.0       # 10k€ tranches
    INIT_CAPITAL = 150_000.0  # 150k€ portfolio

    # Bid/ask spread: ~0.01% for liquid ETFs (iShares core on Euronext)
    SPREAD_COST = 0.0001     # 0.01% per trade

    # Broker fee — Interactive Brokers Euronext Fixed SmartRouting:
    # 0.05% of trade value, minimum 3€ per order.
    def _broker_fee(order_amount):
        """Compute the Interactive Brokers fee for a single order."""
        return max(order_amount * 0.0005, 3.0)

    # iShares ETF TER: already included in NAV (price returns are net of TER)

    # Minimum rebalance threshold: skip if allocation change < 3%
    REBAL_MIN_CHANGE = 0.03

    total_fees = 0.0
    total_taxes = 0.0

    # Tax state: persists between steps via tax_state dict
    if tax_state is not None:
        realized_gains_ytd = tax_state.get("realized_gains_ytd", 0.0)
        cost_basis = tax_state.get("cost_basis", np.zeros(n_etfs)).copy()
        current_year = tax_state.get("current_year", None)
        # Remap cost_basis to current ETF columns
        if "etf_list" in tax_state and tax_state["etf_list"] != etf_list:
            old_cb = tax_state["cost_basis"]
            old_etfs = tax_state["etf_list"]
            cost_basis = np.zeros(n_etfs)
            for i, etf in enumerate(etf_list):
                if etf in old_etfs:
                    cost_basis[i] = old_cb[old_etfs.index(etf)]
    else:
        realized_gains_ytd = 0.0
        cost_basis = np.zeros(n_etfs)
        current_year = None

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
            # Only set cost_basis if not restored from tax_state
            if tax_state is None:
                cost_basis[j] = positions[j]
        cash = total_val - positions.sum()

    port_returns = np.zeros(n_days)
    actual_weights = np.zeros((n_days, n_etfs))
    n_rebalances = 0  # count actual rebalances
    # Initialize prev_topn_set from prev_weights (carry between steps)
    if prev_weights is not None:
        prev_topn_set = set(j for j in range(n_etfs) if prev_weights[j] > 0.001)
    else:
        prev_topn_set = set()

    def _compute_target_lots(target_w, total_val):
        """Compute target positions in LOT_SIZE lots given weights and total value."""
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

        # Flat tax 30% on Jan 1st on realized gains of previous year
        day = dates[i]
        if current_year is None:
            current_year = day.year
        elif day.year != current_year:
            if realized_gains_ytd > 0:
                tax = realized_gains_ytd * FLAT_TAX_RATE
                cash -= tax
                total_taxes += tax
            realized_gains_ytd = 0.0
            current_year = day.year
            total_val = positions.sum() + cash  # recalc after tax

        # Rebalance only when a currently allocated asset leaves the monitor set (top 6)
        if monitor_mask is not None:
            cur_monitor_set = set(j for j in range(n_etfs) if monitor_mask[i, j])
            # Check if any currently held position dropped out of monitor set
            allocated_out = prev_topn_set - cur_monitor_set
            topn_changed = len(allocated_out) > 0 or len(prev_topn_set) == 0
        else:
            cur_topn_set = set(j for j in range(n_etfs) if weights[i, j] > 0)
            topn_changed = (cur_topn_set != prev_topn_set)
        if not topn_changed:
            pass
        elif total_val >= LOT_SIZE:
            target_w = weights[i]
            target_pos, _ = _compute_target_lots(target_w, total_val)
            current_w = positions / total_val if total_val > 0 else np.zeros(n_etfs)
            weight_change = np.abs(target_w - current_w).sum()
            if not np.allclose(target_pos, positions) and weight_change >= REBAL_MIN_CHANGE:
                # Compute fees (broker commission + spread)
                trade_fees = 0.0
                for j in range(n_etfs):
                    diff = target_pos[j] - positions[j]
                    abs_diff = abs(diff)
                    if abs_diff >= LOT_SIZE:
                        order_amount = int(abs_diff / LOT_SIZE) * LOT_SIZE
                        broker_fee = _broker_fee(order_amount)
                        spread_fee = order_amount * SPREAD_COST
                        trade_fees += broker_fee + spread_fee
                total_fees += trade_fees
                cash -= trade_fees

                # Track realized gains on sales
                for j in range(n_etfs):
                    if target_pos[j] < positions[j] and positions[j] > 0:
                        sale_amount = positions[j] - target_pos[j]
                        sale_frac = sale_amount / positions[j]
                        gain = sale_amount - cost_basis[j] * sale_frac
                        realized_gains_ytd += gain
                        cost_basis[j] *= (1 - sale_frac)
                    elif target_pos[j] > positions[j]:
                        buy_amount = target_pos[j] - positions[j]
                        cost_basis[j] += buy_amount

                positions = target_pos.copy()
                cash = total_val - positions.sum() - trade_fees
                n_rebalances += 1
                prev_topn_set = set(j for j in range(n_etfs) if target_pos[j] > 0)

        # Daily P&L on current positions
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
    final_val = positions.sum() + cash
    # Return tax state for chaining between steps
    out_tax_state = {
        "realized_gains_ytd": realized_gains_ytd,
        "cost_basis": cost_basis.copy(),
        "current_year": current_year,
        "etf_list": etf_list,
    }
    return ret_series, weights_df, total_fees, total_taxes, n_rebalances, final_val, out_tax_state


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


def compute_red_orange(port_returns: pd.Series, step_fee_records: list,
                       init_capital: float = 150_000.0) -> tuple:
    """Build the two parallel equity curves from a gross daily-return series
    and per-step IB fees:
      RED    — pays IB fees only (no tax). Each step's fee enters as a
               scale-invariant fractional drag (fee / 150k base) so fees
               stay correct as the portfolio grows.
      ORANGE — same, plus the 30% flat tax withdrawn from capital each
               1st January on the prior year's realised gain (net of fees).
    Returns (eq_red, eq_orange, daily_factor, cumul_fees, total_taxes).
    """
    if len(port_returns) == 0:
        empty = pd.Series(dtype=float)
        return empty, empty, empty, 0.0, 0.0

    fee_drag = pd.Series(1.0, index=port_returns.index)
    for idx, sf in step_fee_records:
        fee_drag.loc[idx.max()] *= (1.0 - sf / init_capital)
    daily_factor = (1.0 + port_returns) * fee_drag
    eq_red = daily_factor.cumprod()                       # IB fees only

    eq_orange = pd.Series(index=port_returns.index, dtype=float)
    orange_start = 1.0
    total_taxes = 0.0
    for year in range(int(port_returns.index[0].year), int(port_returns.index[-1].year) + 1):
        ymask = port_returns.index.year == year
        if ymask.sum() == 0:
            continue
        ycum = daily_factor[ymask].cumprod()
        eq_orange[ymask] = orange_start * ycum
        year_end = orange_start * ycum.iloc[-1]
        net_gain = year_end - orange_start                # already net of IB fees
        tax = FLAT_TAX_RATE * net_gain if net_gain > 0 else 0.0
        total_taxes += tax * init_capital
        orange_start = year_end - tax                     # next year starts after tax

    # Real cumulative IB fees, scaled to the (growing) red portfolio
    cumul_fees = sum(sf * eq_red.loc[idx.max()] for idx, sf in step_fee_records)
    return eq_red, eq_orange, daily_factor, cumul_fees, total_taxes


# Unified colour map (bourso ticker → colour), shared by the chart panels.
# Principals (S&P 500, Nasdaq, Gold Miners) get saturated colours; every ETF
# of the universe gets a distinct colour (no grey fallback).
ETF_COLOR_MAP = {
    "IVV": "#1565c0", "QQQ": "#2ca02c", "RING": "#f4b400",        # principals
    "ACWI": "#000000", "EEM": "#9467bd", "IEMG": "#8c564b", "EMXC": "#e377c2",
    "ILF": "#bcbd22", "EWY": "#7b4173", "EWT": "#393b79", "EWZ": "#a55194",
    "EWW": "#ce6dbd", "EWC": "#e7969c", "EWJ": "#6b6ecf", "TUR": "#ad494a",
    "FXI": "#de9ed6", "ISF.L": "#637939", "IEUR": "#3182bd", "EZU": "#b5cf6b",
    "EPP": "#9c9ede", "SUSA": "#5254a3", "SOXX": "#756bb1", "ROBO": "#c49c94",
    "ICLN": "#66c2a5", "EXX1.DE": "#1ab0a8", "IEO": "#8b4513", "SXRS.DE": "#bd9e39",
}


def _short_name(name: str) -> str:
    """ETF display name minus the iShares / Sector / ETF / Acc / USD tokens."""
    drop = {"ishares", "sector", "etf", "acc", "usd"}
    words = [w for w in name.split() if w.lower() not in drop]
    return " ".join(words) or name


def _save_equity_png(port_returns: pd.Series, eq_curve: pd.Series,
                     weights_df: pd.DataFrame,
                     params_df: pd.DataFrame, out_dir: Path,
                     scores_A: pd.DataFrame | None = None,
                     fin: dict | None = None,
                     fname: str = "backtest_equity.jpg",
                     eq_orange: pd.Series | None = None) -> None:
    """Save chart: equity, allocation, temp_A + scores_A."""
    from etf import BY_BOURSO
    # Compound annual growth rate (CAGR) — same definition as the summary box
    ann_ret = eq_curve.iloc[-1] ** (252.0 / max(len(port_returns), 1)) - 1
    ann_vol = port_returns.std() * np.sqrt(252)
    sh      = sharpe(port_returns)
    max_dd  = (eq_curve / eq_curve.cummax() - 1).min()
    title = (f"MyQTM-ETF — Walk-Forward OOS  "
             f"(Sharpe={sh:.2f}  Ann={ann_ret:.1%}  Vol={ann_vol:.1%}  MaxDD={max_dd:.1%})")

    etf_color_map = ETF_COLOR_MAP
    from etf import UNIVERSE as _UNIVERSE
    # Abbreviated display names (full name minus iShares/Sector/ETF/Acc/USD)
    SHORT_NAMES = {e.bourso: _short_name(e.name) for e in _UNIVERSE}
    # Tickers to show in bold on equity chart (besides portfolio)
    BOLD_TICKERS = {"IVV", "GLD", "IEO", "QQQ", "RING"}

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
        mean_alloc = w.mean()
    else:
        trades_per_month = 0
        mean_alloc = pd.Series(dtype=float)
    # Mean allocation per ETF (short name) — drives the legend bullet size
    alloc_by_short = {SHORT_NAMES.get(t, t): float(mean_alloc.get(t, 0.0))
                      for t in mean_alloc.index}

    # --- Right column header: performance + financial summary ---
    def _eur(x):
        return f"{x:,.0f}".replace(",", " ") + " €"

    # Colour code: gross/performance items in red, everything about fees,
    # tax, initial capital and the net final value in orange.
    RED, ORANGE = "#d62728", "#ff7f0e"
    hdr = [
        ("Ann (brut-frais)", f"{ann_ret:+.1%}", RED, True),
    ]
    if fin is not None:
        hdr.append(("Ann. net", f"{fin['ann_net']:+.1%}", ORANGE, True))
    hdr += [
        ("Vol", f"{ann_vol:.1%}", RED, False),
        ("MaxDD", f"{max_dd:.1%}", RED, False),
        ("Trades", f"{trades_per_month:.1f}/mois", RED, False),
    ]
    if fin is not None:
        hdr += [
            ("__sep__", "", "", False),
            ("Capital initial", _eur(fin['init']), ORANGE, False),
            ("Valeur finale (brut-frais)", _eur(fin['gross']), ORANGE, True),
            ("Frais Interactive Broker", _eur(-fin['fees']), ORANGE, False),
            ("Flat tax 30 % (versée)", _eur(-fin['taxes']), ORANGE, False),
            ("Valeur finale NETTE", _eur(fin['net']), ORANGE, True),
        ]

    y = 1.00
    y_step_hdr = 0.033
    for label, val, color, bold in hdr:
        if label == "__sep__":
            ax_leg.plot([0.0, 0.97], [y + 0.010, y + 0.010], color="#999999",
                        lw=1.0, transform=ax_leg.transAxes)
            y -= y_step_hdr
            continue
        fw = "bold" if bold else "normal"
        ax_leg.text(0.0, y, label, fontsize=12.5, fontweight=fw, va="top",
                    transform=ax_leg.transAxes, color=color)
        ax_leg.text(0.97, y, val, fontsize=12.5, fontweight=fw, va="top", ha="right",
                    transform=ax_leg.transAxes, color=color)
        y -= y_step_hdr
    hdr_bottom = y

    # Portfolio curves: red = IB fees only, orange = fees + flat tax 30%
    ax1.plot(eq_curve.index, eq_curve.values, lw=3.5, color="#d62728", zorder=10,
             label="Frais IB seuls")
    if eq_orange is not None:
        ax1.plot(eq_orange.index, eq_orange.values, lw=1.5, color="#ff7f0e",
                 zorder=11, label="Frais IB + flat tax 30 %")
        ax1.legend(loc="upper left", fontsize=12, framealpha=0.92)

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
            if ticker == "QQQ":
                lw, alpha, zorder = 3.5, 0.9, 9
            elif is_bold:
                lw, alpha, zorder = 2.8, 0.9, 5
            else:
                lw, alpha, zorder = 1.0, 0.5, 2
            short = SHORT_NAMES.get(ticker, ticker)
            ax1.plot(bm.index, bm.values, lw=lw, color=color, alpha=alpha, zorder=zorder)
            bm_ret = bm.pct_change().dropna()
            etf_sh = sharpe(bm_ret)
            etf_dd = (bm / bm.cummax() - 1).min()
            etf_curves.append((bm.iloc[-1], etf_sh, etf_dd, short, color))

    # Sorted list: portfolio + ETFs all sorted by final value (descending)
    from matplotlib.lines import Line2D
    all_items = [(eq_curve.iloc[-1], port_sh, port_dd, "Portefeuille (brut-frais)", "#d62728", True)]
    if eq_orange is not None:
        o_ret = eq_orange.pct_change().dropna()
        all_items.append((eq_orange.iloc[-1], sharpe(o_ret),
                          (eq_orange / eq_orange.cummax() - 1).min(),
                          "Portefeuille (net)", "#ff7f0e", True))
    all_items += [(v, sh, dd, s, c, False) for v, sh, dd, s, c in etf_curves]
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
        vix_ema100 = vix_raw.ewm(span=100).mean()
        vix_ema300 = vix_raw.ewm(span=300).mean()
        ax1b.plot(vix_ema100.index, vix_ema100.values, color="#d62728", alpha=0.7,
                  linewidth=2.0, linestyle=(0, (8, 4)))
        ax1b.plot(vix_ema300.index, vix_ema300.values, color="#1565c0", alpha=0.7,
                  linewidth=2.0, linestyle=(0, (8, 4)))
        ax1b.set_ylim(0, 80)
        ax1b.set_ylabel("VIX", color="#d62728", fontsize=8)
        ax1b.tick_params(axis="y", labelcolor="#d62728", labelsize=7)

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

    # VIX spike vertical lines — top 10 distinct events (10d cooldown), proportional thickness
    vix_spike_path = DATA / "fred_vix.parquet"
    if vix_spike_path.exists():
        vix_for_spikes = pd.read_parquet(vix_spike_path).iloc[:, 0]
        vix_for_spikes = vix_for_spikes.reindex(eq_curve.index, method="ffill").dropna()
        vix_5d = vix_for_spikes.diff(5)
        # Deduplicate: keep only first spike per event (10d cooldown)
        all_spikes = vix_5d[vix_5d > VIX_SPIKE_MIN].sort_values(ascending=False)
        events = []
        used_dates = set()
        for d, val in all_spikes.items():
            if any(abs((d - u).days) <= 20 for u in used_dates):
                continue
            events.append((d, val))
            used_dates.add(d)
        # Top 10 by intensity, then display in chronological order
        events.sort(key=lambda x: -x[1])
        events = events[:10]
        if events:
            max_spike = max(v for _, v in events)
            for d, val in events:
                frac = val / max_spike
                lw = 0.8 + 2.5 * frac
                ax1.axvline(d, color="black", linewidth=lw, alpha=0.5 + 0.4 * frac, zorder=1)
                ax2.axvline(d, color="black", linewidth=lw, alpha=0.5 + 0.4 * frac, zorder=1)

    # Right column: ETF list sorted by Sharpe (starts below the header block)
    n_items = len(sorted_by_sharpe)
    y_start = hdr_bottom - 0.030
    avail = y_start + 0.06
    y_step = min(0.030, avail / max(n_items, 1))
    marker_fs = min(26, max(12, y_step * 850))   # max bullet size
    marker_min = 5.0                              # bullet size at zero allocation
    max_a = max(alloc_by_short.values(), default=0.0)
    for i, (val, sh, dd, short, color, is_port) in enumerate(sorted_by_sharpe):
        y = y_start - i * y_step
        fw = "bold" if is_port else "normal"
        fs = (11 if is_port else 10) if y_step > 0.024 else (10 if is_port else 9)
        # Bullet size ∝ mean allocation (portfolio rows always full size)
        if is_port:
            m_fs = marker_fs
        else:
            frac = (alloc_by_short.get(short, 0.0) / max_a) if max_a > 0 else 0.0
            m_fs = marker_min + (marker_fs - marker_min) * frac
        ax_leg.text(0.0, y, "■", fontsize=m_fs, color=color, va="center",
                    transform=ax_leg.transAxes)
        ax_leg.text(0.08, y, f"{short}", fontsize=fs, fontweight=fw, va="center",
                    transform=ax_leg.transAxes)
        ax_leg.text(0.55, y, f"{val:.1f}x", fontsize=fs, va="center",
                    transform=ax_leg.transAxes)
        ax_leg.text(0.72, y, f"Sh={sh:.2f}", fontsize=fs, va="center",
                    transform=ax_leg.transAxes, color="#333333")
        ax_leg.text(0.95, y, f"{dd:.0%}", fontsize=fs, va="center", ha="right",
                    transform=ax_leg.transAxes, color="#cc0000" if dd < -0.30 else "#666666")
    fig1.savefig(out_dir / fname, dpi=100, bbox_inches="tight", format="jpeg", pil_kwargs={"quality": 60, "optimize": True})
    plt.close(fig1)


def _save_winners_losers_pie(period_label: str, weights_df: pd.DataFrame,
                             daily_ret_panel: pd.DataFrame, out_dir: Path,
                             fname: str) -> None:
    """Two pie charts — losers (left) and winners (right). Each ETF held over
    the period is a slice sized by its allocation volume (Σ daily weight);
    winner/loser split by the sign of its P&L contribution (Σ weight×return).
    Pie radius scales with each side's total volume."""
    from etf import UNIVERSE as _UNIVERSE
    short = {e.bourso: _short_name(e.name) for e in _UNIVERSE}

    dr = daily_ret_panel.reindex(index=weights_df.index).fillna(0.0)
    rows = []   # (ticker, volume, contribution)
    for t in weights_df.columns:
        if t not in dr.columns:
            continue
        w = weights_df[t].fillna(0.0)
        vol = float(w.sum())
        if vol < 1e-6:
            continue
        rows.append((t, vol, float((w * dr[t]).sum())))

    winners = sorted([r for r in rows if r[2] >= 0], key=lambda x: -x[1])
    losers  = sorted([r for r in rows if r[2] < 0],  key=lambda x: -x[1])
    vol_w, vol_l = sum(r[1] for r in winners), sum(r[1] for r in losers)
    con_w, con_l = sum(r[2] for r in winners), sum(r[2] for r in losers)
    vmax = max(vol_w, vol_l, 1e-9)

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(20, 11))
    fig.suptitle(f"Gagnants / Perdants par volume alloué — {period_label}",
                 fontsize=18, fontweight="bold")

    def _pie(ax, items, side_vol, title, tcolor, radius):
        ax.set_title(title, fontsize=14, fontweight="bold", color=tcolor)
        ax.axis("off")
        if not items or side_vol <= 0:
            ax.text(0.5, 0.5, "—", ha="center", va="center", fontsize=22,
                    color="#999999")
            return
        # Group slices below 2.5% of the side into a single "Autres" wedge
        thr = 0.025 * side_vol
        big = [r for r in items if r[1] >= thr]
        small = [r for r in items if r[1] < thr]
        sizes  = [r[1] for r in big]
        colors = [ETF_COLOR_MAP.get(r[0], "#999999") for r in big]
        labels = [f"{short.get(r[0], r[0])}  {r[1] / side_vol:.0%}" for r in big]
        if small:
            sv = sum(r[1] for r in small)
            sizes.append(sv)
            colors.append("#cccccc")
            labels.append(f"Autres ({len(small)})  {sv / side_vol:.0%}")
        ax.pie(sizes, labels=labels, colors=colors, radius=radius,
               startangle=90, counterclock=False,
               wedgeprops=dict(edgecolor="white", linewidth=1.0),
               textprops=dict(fontsize=9), labeldistance=1.06)
        ax.set(aspect="equal")

    _pie(axl, losers, vol_l,
         f"Perdants — {len(losers)} ETF   (contribution {con_l:+.1%})",
         "#d62728", radius=(vol_l / vmax) ** 0.5)
    _pie(axr, winners, vol_w,
         f"Gagnants — {len(winners)} ETF   (contribution {con_w:+.1%})",
         "#1a7a1a", radius=(vol_w / vmax) ** 0.5)

    fig.savefig(out_dir / fname, dpi=100, bbox_inches="tight", format="jpeg",
                pil_kwargs={"quality": 60, "optimize": True})
    plt.close(fig)


def _plot_feature_importance() -> None:
    """SHAP feature-importance evolution chart, with the red backtest equity
    curve overlaid.

    Reads two inputs:
      outputs/feature_importances.parquet  — produced by train.py (the model)
      outputs/backtest_equity.csv          — produced by this backtest run
    Drawing it here (rather than in train.py) keeps the overlaid backtest
    curve in sync with the current backtest whenever the backtest changes.
    """
    path = OUTPUTS / "feature_importances.parquet"
    if not path.exists():
        print("  (feature_importances.parquet absent — run train.py to generate it)")
        return

    df = pd.read_parquet(path)
    if "step" in df.columns:
        df = df.drop(columns=["step"])
    # With walk-forward feature selection a feature is absent from the steps
    # where it was not selected → treat those as zero importance.
    df = df.fillna(0.0)

    # Rank features by CUMULATIVE global SHAP importance (sum over every step),
    # so a feature that enters/leaves the per-step WF selection is still ranked
    # on its total contribution across the whole walk-forward.
    total_imp = df.sum().sort_values(ascending=False)
    TOP_N = min(50, len(total_imp))
    top_features = total_imp.head(TOP_N).index.tolist()

    print(f"\nTop {TOP_N} features by cumulative SHAP importance:")
    for i, feat in enumerate(top_features):
        is_sm = "so_" in feat or "shares" in feat
        tag = " <- SMART MONEY" if is_sm else ""
        print(f"  {i+1:2d}. {feat:<35s}  cumul={total_imp[feat]:.3f}{tag}")

    fig = plt.figure(figsize=(24, 14))
    gs = fig.add_gridspec(2, 2, width_ratios=[3, 1], height_ratios=[3, 1],
                          hspace=0.08, wspace=0.02,
                          top=0.95, bottom=0.05, left=0.05, right=0.98)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax_leg = fig.add_subplot(gs[:, 1])
    ax_leg.axis("off")

    top_df = df[top_features]
    top_smooth = top_df.rolling(5, min_periods=1).mean()

    base_colors = plt.cm.gist_ncar(np.linspace(0.02, 0.95, TOP_N))
    colors = []
    for i, feat in enumerate(top_features):
        if "so_" in feat or "shares" in feat:
            colors.append("#d62728")
        else:
            colors.append(base_colors[i])

    ax1.stackplot(top_smooth.index, top_smooth.values.T,
                  labels=top_features, colors=colors, alpha=0.85)
    ax1.set_ylabel("SHAP Importance (stacked)")
    ax1.set_title(f"Top {TOP_N} Features — SHAP Importance Walk-Forward "
                  f"(ranked by cumulative importance)", fontsize=14)
    ax1.grid(True, alpha=0.3)

    # Overlay the red backtest equity curve (IB fees only) on a twin axis
    bt_path = OUTPUTS / "backtest_equity.csv"
    if bt_path.exists():
        bt = pd.read_csv(bt_path, parse_dates=["date"]).set_index("date")["equity"]
        ax1b = ax1.twinx()
        ax1b.plot(bt.index, bt.values, lw=3.0, color="#d62728", zorder=20,
                  label="Backtest (frais IB)")
        ax1b.set_yscale("log")
        ax1b.set_ylabel("Backtest equity — frais IB (log, base 1)", color="#d62728")
        ax1b.tick_params(axis="y", colors="#d62728")
        ax1b.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:.0f}x"))
        ax1b.legend(loc="upper left", fontsize=11, framealpha=0.92)

    fs = 9 if TOP_N <= 25 else 7
    mk = min(16, max(7, 0.9 / max(TOP_N, 1) * 620))
    y_start = 0.98
    y_step = min(0.035, 0.94 / max(TOP_N, 1))
    for i, feat in enumerate(top_features):
        y = y_start - i * y_step
        is_sm = "so_" in feat or "shares" in feat
        color = "#d62728" if is_sm else colors[i]
        fw = "bold" if is_sm else "normal"
        ax_leg.text(0.0, y, "■", fontsize=mk, color=color, va="center",
                    transform=ax_leg.transAxes)
        ax_leg.text(0.08, y, f"{i+1:2d}. {feat}", fontsize=fs, fontweight=fw,
                    va="center", transform=ax_leg.transAxes)
        ax_leg.text(0.97, y, f"{total_imp[feat]:.2f}", fontsize=fs, va="center",
                    ha="right", transform=ax_leg.transAxes, color="#333333")

    n_active = (df > 0.001).sum(axis=1)
    ax2.plot(n_active.index, n_active.values, lw=2, color="#1f77b4")
    ax2.set_ylabel("Active features (SHAP > 0.001)")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)

    out = OUTPUTS / "feature_importance_evolution.jpg"
    fig.savefig(out, dpi=100, bbox_inches="tight", format="jpeg",
                pil_kwargs={"quality": 70, "optimize": True})
    plt.close(fig)
    print(f"Saved → outputs/feature_importance_evolution.jpg")


def run_equity():
    """Equity backtest → global + per-year equity charts and winner/loser pies."""
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

    # VIX EMA for calm-market detection
    VIX_CALM_THRESHOLD = 22.0
    VIX_CALM_COND_EMA100_SUP_EMA300 = False
    vix_ema100 = vix_s.ewm(span=100).mean() if not vix_s.empty else pd.Series(dtype=float)
    vix_ema300 = vix_s.ewm(span=300).mean() if not vix_s.empty else pd.Series(dtype=float)

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
    current_alloc_cap = 1.0   # persists between steps, reduced on spike, recovers gradually
    recovering        = False # once recovery starts, climb monthly until 100% (ignore new spikes)
    step_fee_records = []     # (date_index, fee) per step — fees scaled later
    cumul_rebals     = 0     # total rebalances
    chain_tax_state  = None  # tax state chained between steps

    print(f"Temperature: {TEMPERATURE:.2f}")

    print(f"\n{'Step':>4}  {'Test period':>24}  {'Sharpe':>7}")
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

        # --- VIX-adaptive TOP_N + REBAL_DAYS + spike detection ---
        if VIX_ADAPTIVE and not vix_s.empty:
            vix_aligned = vix_s.reindex(val_scores.index, method="ffill")
            vix_at_step = vix_aligned.iloc[-1]
            # Detect VIX spike: 5-day change
            vix_5d_change = vix_aligned.diff(5).iloc[-1] if len(vix_aligned) > 5 else 0.0

            if not np.isnan(vix_at_step):
                # Spike detection: progressive cap based on VIX slope
                is_spike = (not np.isnan(vix_5d_change)) and vix_5d_change > VIX_SPIKE_MIN

                # --- TOP_N + REBAL frequency ---
                if is_spike:
                    step_top_n = TOP_N_HIGH
                    step_rebal = VIX_SPIKE_REBAL
                elif vix_at_step < VIX_LOW:
                    step_top_n = TOP_N_LOW
                    step_rebal = REBAL_DAYS_LOW
                elif vix_at_step > VIX_HIGH:
                    step_top_n = TOP_N_HIGH
                    step_rebal = REBAL_DAYS_HIGH
                else:
                    frac = (vix_at_step - VIX_LOW) / (VIX_HIGH - VIX_LOW)
                    step_top_n = int(round(TOP_N_LOW + frac * (TOP_N_HIGH - TOP_N_LOW)))
                    step_rebal = int(round(REBAL_DAYS_LOW + frac * (REBAL_DAYS_HIGH - REBAL_DAYS_LOW)))

                # --- Allocation cap: slope-driven only (independent of VIX level).
                #     Cut on steep ascending slope; once the slope stops being
                #     steep ascending, recover monthly until 100% — and stay in
                #     recovery mode even if new spikes occur (no re-pinning). ---
                if is_spike and not recovering:
                    frac_spike = min((vix_5d_change - VIX_SPIKE_MIN) / (VIX_SPIKE_MAX - VIX_SPIKE_MIN), 1.0)
                    spike_cap = CAP_AT_SPIKE_MIN + frac_spike * (CAP_AT_SPIKE_MAX - CAP_AT_SPIKE_MIN)
                    # Reduce cap immediately (take the lower of current and spike cap)
                    current_alloc_cap = min(current_alloc_cap, spike_cap)
                    # Once the cap hits the floor, start a guaranteed monthly
                    # recovery — even if spikes keep coming (no more re-pinning)
                    if current_alloc_cap <= CAP_AT_SPIKE_MAX + 1e-9:
                        recovering = True
                else:
                    # Slope no longer steep ascending (or already recovering)
                    # → guaranteed monthly recovery toward 100%
                    if current_alloc_cap < 1.0:
                        recovering = True
                    current_alloc_cap = min(1.0, current_alloc_cap + RECOVERY_RATE)
                    if current_alloc_cap >= 1.0:
                        recovering = False
            else:
                step_top_n = TOP_N_SCORES
                step_rebal = REBAL_DAYS
        else:
            step_top_n = TOP_N_SCORES
            step_rebal = REBAL_DAYS

        step_max_alloc = current_alloc_cap

        # --- Evaluate on test (true OOS) ---
        test_scores = _pivot_step(test_data)

        # Remap carry_weights to current ETF columns
        prev_w = None
        if carry_weights is not None:
            prev_w = np.zeros(len(test_scores.columns))
            for i, etf in enumerate(test_scores.columns):
                if etf in carry_weights:
                    prev_w[i] = carry_weights[etf]

        # --- Calm market override: VIX EMA300 < threshold → Sharpe leaders ---
        calm_market = False
        if not vix_ema100.empty and not vix_ema300.empty:
                ema100_aligned = vix_ema100.reindex(val_scores.index, method="ffill")
                ema300_aligned = vix_ema300.reindex(val_scores.index, method="ffill")
                if len(ema100_aligned) > 0 and len(ema300_aligned) > 0:
                    ema100_at_step = ema100_aligned.iloc[-1]
                    ema300_at_step = ema300_aligned.iloc[-1]
                    if not np.isnan(ema100_at_step) and not np.isnan(ema300_at_step):
                        calm_market = ( ema100_at_step < VIX_CALM_THRESHOLD
                                        and (not VIX_CALM_COND_EMA100_SUP_EMA300 or (ema100_at_step < ema300_at_step)) )
        
        if calm_market:
            # Ignore model scores — allocate top 3 by rolling 252d Sharpe
            test_scores = _pivot_step(test_data)
            roll_mean = dr_hist.rolling(252, min_periods=200).mean().iloc[-1] * 252
            roll_std = dr_hist.rolling(252, min_periods=200).std().iloc[-1] * np.sqrt(252)
            roll_sharpe = (roll_mean / roll_std.replace(0, np.nan)).clip(0.0).fillna(0.0)
            # Align to test columns
            sharpe_series = roll_sharpe.reindex(test_scores.columns).fillna(0.0)
            top3_sharpe = sharpe_series.nlargest(TOP_N_ALLOC).index.tolist()
            # Build scores = rolling Sharpe for top 3, NaN for rest
            for col in test_scores.columns:
                col_loc = test_scores.columns.get_loc(col)
                if col in top3_sharpe:
                    test_scores.iloc[:, col_loc] = sharpe_series[col]
                else:
                    test_scores.iloc[:, col_loc] = np.nan
            # Monitor mask: top 3 = monitor (no hysteresis in calm mode)
            n_cols = len(test_scores.columns)
            step_monitor_mask = np.zeros((len(test_scores), n_cols), dtype=bool)
            for col in top3_sharpe:
                step_monitor_mask[:, test_scores.columns.get_loc(col)] = True
        else:
            # Z-score scores cross-sectionally before softmax
            row_mean = test_scores.mean(axis=1)
            row_std = test_scores.std(axis=1).replace(0, 1.0)
            test_scores = test_scores.sub(row_mean, axis=0).div(row_std, axis=0)

            # Build monitor mask (top N_MONITOR) and allocation scores (top N_ALLOC)
            block_ids = [CORR_BLOCKS.get(c, c) for c in test_scores.columns]
            n_cols = len(test_scores.columns)
            step_monitor_mask = np.zeros((len(test_scores), n_cols), dtype=bool)

            for idx in range(len(test_scores)):
                row = test_scores.iloc[idx].values
                keep_monitor = _topn_keep(row, TOP_N_MONITOR, block_ids)
                step_monitor_mask[idx, keep_monitor] = True
                keep_alloc = _topn_keep(row, TOP_N_ALLOC, block_ids)
                pos_idx = np.where((~np.isnan(row)) & (row > 0))[0]
                drop = np.setdiff1d(pos_idx, keep_alloc)
                test_scores.iloc[idx, drop] = np.nan

        # VIX spike → go to cash for 3 days
        VIX_SPIKE_CASH_DAYS = 3
        if not vix_s.empty:
            vix_test = vix_s.reindex(test_scores.index, method="ffill")
            vix_5d_test = vix_test.diff(5)
            spike_dates = vix_5d_test[vix_5d_test > VIX_SPIKE_MIN].index
            cash_dates = set()
            for sd in spike_dates:
                for offset in range(VIX_SPIKE_CASH_DAYS):
                    idx_pos = test_scores.index.get_loc(sd) + offset if sd in test_scores.index else -1
                    if 0 <= idx_pos < len(test_scores):
                        cash_dates.add(test_scores.index[idx_pos])
            for cd in cash_dates:
                test_scores.loc[cd] = np.nan
                step_monitor_mask[test_scores.index.get_loc(cd)] = False

        # Daily returns for test period
        test_dr = daily_ret_panel.reindex(index=test_scores.index, columns=test_scores.columns).fillna(0)
        test_returns, test_weights, step_fees, step_taxes, step_rebals, _, chain_tax_state = run_backtest(
                                                   test_scores, test_dr,
                                                   prev_weights=prev_w,
                                                   sharpe_weights=etf_sharpe,
                                                   rebal_days_override=step_rebal,
                                                   max_alloc=step_max_alloc,
                                                   tax_state=chain_tax_state,
                                                   monitor_mask=step_monitor_mask)

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
        step_fee_records.append((test_returns.index, step_fees))
        cumul_rebals += step_rebals
        all_params_rows.append({"step": step, "temperature": TEMPERATURE, "rebal_days": REBAL_DAYS,
                                "alloc_cap": step_max_alloc})
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
    init_capital = 150_000.0
    n_years      = max(len(port_returns), 1) / 252
    ann_vol      = port_returns.std() * np.sqrt(252)
    final_sharpe = sharpe(port_returns)

    # --- Two parallel backtests: red (IB fees only) + orange (fees + tax) ---
    eq_red, eq_orange, daily_factor, cumul_fees, total_taxes = compute_red_orange(
        port_returns, step_fee_records, init_capital)

    eq_curve     = eq_red                                 # red = reference curve
    total_ret    = eq_red.iloc[-1] - 1
    ann_ret      = (1 + total_ret) ** (1 / n_years) - 1
    max_dd       = (eq_red / eq_red.cummax() - 1).min()

    final_portfolio = eq_red.iloc[-1] * init_capital      # red final (fees only)
    net_final  = eq_orange.iloc[-1] * init_capital        # orange final (fees + tax)
    net_return = net_final / init_capital - 1
    net_ann    = (1 + net_return) ** (1 / n_years) - 1

    print("\n" + "=" * 75)
    print("=== Final OOS backtest ===")
    print(f"  Period:       {port_returns.index[0].date()} → {port_returns.index[-1].date()}")
    print(f"  Init capital: {init_capital:,.0f}€")
    print(f"  Ann. vol:     {ann_vol:.1%}")
    print(f"  Sharpe:       {final_sharpe:.3f}")
    print(f"  Max drawdown: {max_dd:.1%}")
    print(f"  --- RED — IB fees only (no tax) ---")
    print(f"  Total return: {total_ret:.1%}")
    print(f"  Ann. return:  {ann_ret:.1%}")
    print(f"  Rebalances:   {cumul_rebals} ({cumul_rebals/n_years:.0f}/an)")
    print(f"  Trading fees: {cumul_fees:,.0f}€ ({cumul_fees/n_years:,.0f}€/an)")
    print(f"  Final value:  {final_portfolio:,.0f}€")
    print(f"  --- ORANGE — IB fees + flat tax 30% ---")
    print(f"  Flat tax:     {total_taxes:,.0f}€ ({total_taxes/n_years:,.0f}€/an)")
    print(f"  Final value:  {net_final:,.0f}€")
    print(f"  Net return:   {net_return:.1%}")
    print(f"  Net ann.:     {net_ann:.1%}")

    # Save results
    port_returns.to_frame().to_parquet(DATA / "backtest_results.parquet")

    params_df = pd.DataFrame(all_params_rows)
    params_df.to_csv(OUTPUTS / "best_params.csv", index=False)

    eq_df = eq_curve.reset_index()
    eq_df.columns = ["date", "equity"]
    eq_df.to_csv(OUTPUTS / "backtest_equity.csv", index=False)

    pd.DataFrame(all_params_rows).to_csv(OUTPUTS / "backtest_steps.csv", index=False)

    # Equity chart — red (fees) + orange (fees + flat tax)
    all_weights = pd.concat(all_test_weights).sort_index()
    all_scores = pd.concat(all_test_scores).sort_index() if all_test_scores else pd.DataFrame()
    fin = {
        "init": init_capital,
        "gross": final_portfolio,   # red final  (IB fees only)
        "net": net_final,           # orange final (fees + flat tax)
        "fees": cumul_fees,
        "taxes": total_taxes,
        "ann_gross": ann_ret,
        "ann_net": net_ann,
    }
    _save_equity_png(port_returns, eq_red, all_weights, params_df, OUTPUTS,
                     scores_A=all_scores, fin=fin, eq_orange=eq_orange)

    print(f"\nSaved → data/backtest_results.parquet")
    print(f"Saved → outputs/best_params.csv")
    print(f"Saved → outputs/backtest_steps.csv")
    print(f"Saved → outputs/backtest_equity.csv")
    print(f"Saved → outputs/backtest_equity.jpg")

    # --- Winners / losers pie (global, all years chained) ---
    global_label = f"{port_returns.index[0].year}-{port_returns.index[-1].year}"
    _save_winners_losers_pie(global_label, all_weights, daily_ret_panel,
                             OUTPUTS, "backtest_pie.jpg")
    print(f"Saved → outputs/backtest_pie.jpg")

    # --- Per-year equity charts (each year restarts fresh at 150k€) ---
    for yr in range(int(port_returns.index[0].year), int(port_returns.index[-1].year) + 1):
        ymask = port_returns.index.year == yr
        if ymask.sum() < 5:
            continue
        yr_returns = port_returns[ymask]
        yr_red     = daily_factor[ymask].cumprod()         # IB fees included
        yr_weights = all_weights[all_weights.index.year == yr]
        yr_scores  = all_scores[all_scores.index.year == yr] if len(all_scores) else all_scores
        yr_days    = max(len(yr_returns), 1)

        yr_red_final = yr_red.iloc[-1] * init_capital       # fees only
        yr_gain      = yr_red_final - init_capital          # already net of fees
        yr_tax       = FLAT_TAX_RATE * yr_gain if yr_gain > 0 else 0.0
        yr_net       = yr_red_final - yr_tax                # fees + flat tax
        yr_fees      = sum(sf * yr_red.loc[idx.max()] for idx, sf in step_fee_records
                           if idx.max() in yr_red.index)
        # Orange curve: net capital with the flat tax on the gain-to-date
        # marked-to-market (converges to yr_net on the last day).
        yr_orange = yr_red - FLAT_TAX_RATE * (yr_red - 1.0).clip(lower=0)
        fin_yr = {
            "init": init_capital,
            "gross": yr_red_final,
            "net": yr_net,
            "fees": yr_fees,
            "taxes": yr_tax,
            "ann_gross": yr_red.iloc[-1] ** (252.0 / yr_days) - 1,
            "ann_net": (yr_net / init_capital) ** (252.0 / yr_days) - 1,
        }
        _save_equity_png(yr_returns, yr_red, yr_weights, params_df, OUTPUTS,
                         scores_A=yr_scores, fin=fin_yr,
                         fname=f"backtest_equity_{yr}.jpg", eq_orange=yr_orange)
        print(f"Saved → outputs/backtest_equity_{yr}.jpg")
        _save_winners_losers_pie(str(yr), yr_weights, daily_ret_panel,
                                 OUTPUTS, f"backtest_pie_{yr}.jpg")
        print(f"Saved → outputs/backtest_pie_{yr}.jpg")

    # --- Feature-importance evolution, with this backtest's curve overlaid ---
    _plot_feature_importance()


# ===========================================================================
#  Robustness backtest — N_RUNS perturbed backtests dropping random ETFs
# ===========================================================================

def _rob_sharpe(returns: pd.Series) -> float:
    if len(returns) < 10 or returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(252))


def _pivot_step_rob(step_data: pd.DataFrame) -> pd.DataFrame:
    return step_data.reset_index().pivot(index="date", columns="etf_id", values="score")


def _run_metrics(returns: pd.Series, step_fee_records: list) -> tuple:
    """Build red/orange curves (same model as the equity backtest) and scalar
    metrics. Returns (metrics_dict, eq_red, eq_orange)."""
    eq_red, eq_orange, _, _, _ = compute_red_orange(returns, step_fee_records, INIT_CAPITAL)
    n_years = max(len(returns), 1) / 252
    red_total = eq_red.iloc[-1] - 1                        # IB fees only
    net_final = eq_orange.iloc[-1] * INIT_CAPITAL          # fees + flat tax
    net_ret = net_final / INIT_CAPITAL - 1
    m = {
        "total_return_pct": red_total * 100,
        "ann_gross_pct": ((1 + red_total) ** (1 / n_years) - 1) * 100,
        "net_final": net_final,
        "net_return_pct": net_ret * 100,
        "net_ann_pct": ((1 + net_ret) ** (1 / n_years) - 1) * 100,
        "sharpe": _rob_sharpe(returns),
        "max_dd_pct": (eq_red / eq_red.cummax() - 1).min() * 100,
    }
    return m, eq_red, eq_orange


def _run_single(oos, steps, daily_ret_panel, drop_etfs=None, seed=None, vix_s=None):
    """Run a single backtest with full VIX-adaptive logic, optionally dropping N
    random ETFs. Returns (daily_returns, step_fee_records)."""
    rng = np.random.default_rng(seed) if seed is not None else None

    all_test_returns = []
    step_fee_records = []
    current_alloc_cap = 1.0
    recovering = False

    for step in steps:
        val_data = oos[(oos["step"] == step) & (oos["split"] == "val")]
        test_data = oos[(oos["step"] == step) & (oos["split"] == "test")]

        if len(val_data) < 50 or len(test_data) == 0:
            continue

        val_scores = _pivot_step_rob(val_data)
        val_end_date = val_scores.index[-1]
        test_scores = _pivot_step_rob(test_data)

        # Drop random ETFs from scores
        if drop_etfs is not None and rng is not None:
            available = test_scores.columns.tolist()
            if isinstance(drop_etfs, tuple):
                n_drop = rng.integers(drop_etfs[0], drop_etfs[1] + 1)
            else:
                n_drop = drop_etfs
            n_drop = min(n_drop, len(available) - 2)
            dropped = rng.choice(available, size=n_drop, replace=False)
            test_scores[dropped] = np.nan

        # Sharpe weights
        dr_hist = daily_ret_panel[test_scores.columns].loc[:val_end_date]
        exp_mean = dr_hist.expanding(min_periods=60).mean().iloc[-1] * 252
        exp_std = dr_hist.expanding(min_periods=60).std().iloc[-1] * np.sqrt(252)
        etf_sharpe = (exp_mean / exp_std.replace(0, np.nan)).clip(0.0).fillna(0.0).values
        etf_sharpe = etf_sharpe ** SHARPE_POWER

        # VIX-adaptive logic
        step_top_n = TOP_N_SCORES
        step_rebal = REBAL_DAYS
        if VIX_ADAPTIVE and vix_s is not None and not vix_s.empty:
            vix_aligned = vix_s.reindex(val_scores.index, method="ffill")
            vix_at_step = vix_aligned.iloc[-1] if len(vix_aligned) > 0 else np.nan
            vix_5d_change = vix_aligned.diff(5).iloc[-1] if len(vix_aligned) > 5 else 0.0

            if not np.isnan(vix_at_step):
                is_spike = (not np.isnan(vix_5d_change)) and vix_5d_change > VIX_SPIKE_MIN

                # --- TOP_N + REBAL frequency ---
                if is_spike:
                    step_top_n = TOP_N_HIGH
                    step_rebal = VIX_SPIKE_REBAL
                elif vix_at_step < VIX_LOW:
                    step_top_n = TOP_N_LOW
                    step_rebal = REBAL_DAYS_LOW
                elif vix_at_step > VIX_HIGH:
                    step_top_n = TOP_N_HIGH
                    step_rebal = REBAL_DAYS_HIGH
                else:
                    frac = (vix_at_step - VIX_LOW) / (VIX_HIGH - VIX_LOW)
                    step_top_n = int(round(TOP_N_LOW + frac * (TOP_N_HIGH - TOP_N_LOW)))
                    step_rebal = int(round(REBAL_DAYS_LOW + frac * (REBAL_DAYS_HIGH - REBAL_DAYS_LOW)))

                # --- Allocation cap: slope-driven only. Cut on steep ascending
                #     slope; once the floor is hit, recover monthly until 100%
                #     even if spikes keep coming (no re-pinning). ---
                if is_spike and not recovering:
                    frac_spike = min((vix_5d_change - VIX_SPIKE_MIN) / (VIX_SPIKE_MAX - VIX_SPIKE_MIN), 1.0)
                    spike_cap = CAP_AT_SPIKE_MIN + frac_spike * (CAP_AT_SPIKE_MAX - CAP_AT_SPIKE_MIN)
                    current_alloc_cap = min(current_alloc_cap, spike_cap)
                    if current_alloc_cap <= CAP_AT_SPIKE_MAX + 1e-9:
                        recovering = True
                else:
                    if current_alloc_cap < 1.0:
                        recovering = True
                    current_alloc_cap = min(1.0, current_alloc_cap + RECOVERY_RATE)
                    if current_alloc_cap >= 1.0:
                        recovering = False

        step_max_alloc = current_alloc_cap

        # Z-score scores
        row_mean = test_scores.mean(axis=1)
        row_std = test_scores.std(axis=1).replace(0, 1.0)
        test_scores = test_scores.sub(row_mean, axis=0).div(row_std, axis=0)

        # Pre-filter top N — at most 1 ETF per correlated block
        block_ids = [CORR_BLOCKS.get(c, c) for c in test_scores.columns]
        for idx in range(len(test_scores)):
            row = test_scores.iloc[idx].values
            keep = _topn_keep(row, step_top_n, block_ids)
            pos_idx = np.where((~np.isnan(row)) & (row > 0))[0]
            drop_idx = np.setdiff1d(pos_idx, keep)
            test_scores.iloc[idx, drop_idx] = np.nan

        # Daily returns
        test_dr = daily_ret_panel.reindex(index=test_scores.index, columns=test_scores.columns).fillna(0)
        result = run_backtest(test_scores, test_dr, sharpe_weights=etf_sharpe,
                              rebal_days_override=step_rebal, max_alloc=step_max_alloc)
        test_returns = result[0]
        step_fee_records.append((test_returns.index, result[2]))
        all_test_returns.append(test_returns)

    if not all_test_returns:
        return pd.Series(dtype=float), []
    return pd.concat(all_test_returns).sort_index(), step_fee_records


def run_robustness():
    """Robustness backtest → outputs/backtest_robustness.jpg + summary CSV."""
    oos_path = DATA / "oos_predictions.parquet"
    if not oos_path.exists():
        sys.exit("ERROR: data/oos_predictions.parquet not found")

    print("Loading OOS predictions...", flush=True)
    oos = pd.read_parquet(oos_path)
    oos = oos.reset_index()
    oos["date"] = pd.to_datetime(oos["date"])
    oos = oos.set_index(["date", "etf_id"])

    sys.path.insert(0, str(Path(__file__).parent))
    from etf import UNIVERSE
    all_etfs = [e.bourso for e in UNIVERSE]

    # Load daily returns
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

    # Get steps
    START_YEAR = 2008
    all_steps = sorted(oos["step"].unique())
    steps = []
    for s in all_steps:
        test_dates = oos[(oos["step"] == s) & (oos["split"] == "test")].index.get_level_values("date")
        if len(test_dates) > 0 and test_dates.min().year >= START_YEAR:
            steps.append(s)

    # Load VIX for adaptive logic
    vix_path = DATA / "fred_vix.parquet"
    vix_s = pd.read_parquet(vix_path).iloc[:, 0] if vix_path.exists() else pd.Series(dtype=float)

    print(f"Steps: {len(steps)}, ETFs: {len(all_etfs)}, Runs: {N_RUNS}, Drop: {N_DROP}")

    # Run original (no drop) first
    print("Running original (no drop)...", flush=True)
    orig_returns, orig_fees = _run_single(oos, steps, daily_ret_panel, drop_etfs=None, seed=None, vix_s=vix_s)
    orig_m, orig_red, orig_orange = _run_metrics(orig_returns, orig_fees)

    # Run N_RUNS with random drops
    all_run_red = []
    stats = []
    for run in range(N_RUNS):
        print(f"  Run {run+1:2d}/{N_RUNS}...", end="", flush=True)
        run_returns, run_fees = _run_single(oos, steps, daily_ret_panel, drop_etfs=N_DROP, seed=run, vix_s=vix_s)
        m, run_red, _ = _run_metrics(run_returns, run_fees)
        m["run"] = run + 1
        stats.append(m)
        all_run_red.append(run_red)
        print(f"  brut={m['ann_gross_pct']:+.1f}%/an  net={m['net_ann_pct']:+.1f}%/an  "
              f"sharpe={m['sharpe']:.2f}  dd={m['max_dd_pct']:.1f}%", flush=True)

    stats_df = pd.DataFrame(stats)[["run", "total_return_pct", "ann_gross_pct",
                                    "net_return_pct", "net_ann_pct", "net_final",
                                    "sharpe", "max_dd_pct"]]
    stats_df.to_csv(OUTPUTS / "backtest_robustness.csv", index=False)

    # Aggregate stats across the runs: (median, min, max)
    def _agg(col):
        return stats_df[col].median(), stats_df[col].min(), stats_df[col].max()

    ann_g, ann_n = _agg("ann_gross_pct"), _agg("net_ann_pct")
    tot_g, tot_n = _agg("total_return_pct"), _agg("net_return_pct")
    netf, shp, dd = _agg("net_final"), _agg("sharpe"), _agg("max_dd_pct")

    # --- Chart harmonised with backtest_equity.jpg ---
    fig = plt.figure(figsize=(24, 14))
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.02,
                          top=0.96, bottom=0.05, left=0.05, right=0.98)
    ax1 = fig.add_subplot(gs[0, 0])
    ax_leg = fig.add_subplot(gs[0, 1])
    ax_leg.axis("off")

    # All runs in light grey (red curves = IB fees only)
    for eq in all_run_red:
        ax1.plot(eq.index, eq.values, lw=0.8, color="#888888", alpha=0.30, zorder=2)

    # Median of the run curves
    eq_matrix = pd.DataFrame({i: eq for i, eq in enumerate(all_run_red)})
    median_eq = eq_matrix.median(axis=1)
    ax1.plot(median_eq.index, median_eq.values, lw=2.8, color="#1f77b4", zorder=8,
             label=f"Médiane ({N_RUNS} runs)")

    # Original (no drop) — red (IB fees only) + orange (fees + flat tax),
    # same two-curve presentation as the equity chart
    ax1.plot(orig_red.index, orig_red.values, lw=3.5, color="#d62728", zorder=10,
             label="Original — frais IB")
    ax1.plot(orig_orange.index, orig_orange.values, lw=1.5, color="#ff7f0e", zorder=11,
             label="Original — frais IB + flat tax")

    ax1.set_yscale("log")
    ax1.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:.1f}x"))
    ax1.set_ylabel("Equity (log scale, base 1)")
    ax1.set_title(f"MyQTM-ETF — Robustesse  ({N_RUNS} backtests, "
                  f"{N_DROP[0]}-{N_DROP[1]} ETF retirés au hasard)")
    ax1.legend(loc="upper left", fontsize=12)
    ax1.grid(True, alpha=0.3, which="both")
    ax1.set_facecolor("#f8f8f8")

    # VIX shaded background (same as the equity chart)
    if vix_path.exists():
        vix_raw = pd.read_parquet(vix_path).iloc[:, 0]
        vix_raw = vix_raw.reindex(orig_red.index, method="ffill").dropna()
        ax1b = ax1.twinx()
        ax1b.fill_between(vix_raw.index, vix_raw.values, alpha=0.10, color="#d62728")
        vix_ema100 = vix_raw.ewm(span=100).mean()
        vix_ema300 = vix_raw.ewm(span=300).mean()
        ax1b.plot(vix_ema100.index, vix_ema100.values, color="#d62728", alpha=0.7,
                  linewidth=2.0, linestyle=(0, (8, 4)))
        ax1b.plot(vix_ema300.index, vix_ema300.values, color="#1565c0", alpha=0.7,
                  linewidth=2.0, linestyle=(0, (8, 4)))
        ax1b.set_ylim(0, 80)
        ax1b.set_ylabel("VIX", color="#d62728", fontsize=8)
        ax1b.tick_params(axis="y", labelcolor="#d62728", labelsize=7)

    # --- Right column: gross/net return stats across the runs ---
    def _eur(x):
        return f"{x:,.0f}".replace(",", " ") + " €"

    rows = [
        ("title", "ROBUSTESSE", "", ""),
        ("text", f"{N_RUNS} backtests · {N_DROP[0]}-{N_DROP[1]} ETF retirés au hasard", "", ""),
        ("kv", "Capital initial", _eur(INIT_CAPITAL), ""),
        ("sep", "", "", ""),
        ("hdr3", "", "Brut", "Net"),
        ("sub", "Rendement annuel", "", ""),
        ("r3", "Original", f"{orig_m['ann_gross_pct']:+.1f}%", f"{orig_m['net_ann_pct']:+.1f}%"),
        ("r3", "Médian", f"{ann_g[0]:+.1f}%", f"{ann_n[0]:+.1f}%"),
        ("r3", "Min", f"{ann_g[1]:+.1f}%", f"{ann_n[1]:+.1f}%"),
        ("r3", "Max", f"{ann_g[2]:+.1f}%", f"{ann_n[2]:+.1f}%"),
        ("sep", "", "", ""),
        ("sub", "Rendement total", "", ""),
        ("r3", "Original", f"{orig_m['total_return_pct']:+.0f}%", f"{orig_m['net_return_pct']:+.0f}%"),
        ("r3", "Médian", f"{tot_g[0]:+.0f}%", f"{tot_n[0]:+.0f}%"),
        ("r3", "Min", f"{tot_g[1]:+.0f}%", f"{tot_n[1]:+.0f}%"),
        ("r3", "Max", f"{tot_g[2]:+.0f}%", f"{tot_n[2]:+.0f}%"),
        ("sep", "", "", ""),
        ("sub", "Valeur finale nette", "", ""),
        ("kvo", "Original", _eur(orig_m["net_final"]), ""),
        ("kvo", "Médian", _eur(netf[0]), ""),
        ("kvo", "Min", _eur(netf[1]), ""),
        ("kvo", "Max", _eur(netf[2]), ""),
        ("sep", "", "", ""),
        ("sub", "Sharpe (brut)", "", ""),
        ("kvr", "Original", f"{orig_m['sharpe']:.2f}", ""),
        ("kvr", "Médian", f"{shp[0]:.2f}", ""),
        ("kvr", "Min – Max", f"{shp[1]:.2f} – {shp[2]:.2f}", ""),
        ("sep", "", "", ""),
        ("sub", "Max Drawdown", "", ""),
        ("kvr", "Médian", f"{dd[0]:.1f}%", ""),
        ("kvr", "Pire – Meilleur", f"{dd[1]:.1f}% / {dd[2]:.1f}%", ""),
    ]

    # Colour code: brut column red, net column orange, "Médian" rows blue
    # (the median curve on the chart is blue).
    RED, ORANGE, BLUE = "#d62728", "#ff7f0e", "#1f77b4"

    def _lbl(label):
        return (BLUE, "bold") if label.strip() == "Médian" else ("#000000", "normal")

    y, ystep = 0.99, 0.0305
    for kind, a, b, c in rows:
        if kind == "sep":
            ax_leg.plot([0.0, 0.99], [y + 0.012, y + 0.012], color="#bbbbbb",
                        lw=1.0, transform=ax_leg.transAxes)
            y -= ystep * 0.7
        elif kind == "title":
            ax_leg.text(0.0, y, a, fontsize=20, fontweight="bold", va="top",
                        transform=ax_leg.transAxes, color="#d62728")
            y -= ystep * 1.25
        elif kind == "text":
            ax_leg.text(0.0, y, a, fontsize=11, va="top",
                        transform=ax_leg.transAxes, color="#666666")
            y -= ystep
        elif kind == "sub":
            ax_leg.text(0.0, y, a, fontsize=13, fontweight="bold", va="top",
                        transform=ax_leg.transAxes, color="#222222")
            y -= ystep
        elif kind == "hdr3":
            ax_leg.text(0.66, y, b, fontsize=12, fontweight="bold", va="top", ha="right",
                        transform=ax_leg.transAxes, color=RED)
            ax_leg.text(0.99, y, c, fontsize=12, fontweight="bold", va="top", ha="right",
                        transform=ax_leg.transAxes, color=ORANGE)
            y -= ystep
        elif kind == "r3":
            lcol, lfw = _lbl(a)
            ax_leg.text(0.04, y, a, fontsize=12.5, va="top", transform=ax_leg.transAxes,
                        color=lcol, fontweight=lfw)
            ax_leg.text(0.66, y, b, fontsize=12.5, va="top", ha="right",
                        transform=ax_leg.transAxes, color=RED)
            ax_leg.text(0.99, y, c, fontsize=12.5, va="top", ha="right",
                        transform=ax_leg.transAxes, color=ORANGE)
            y -= ystep
        elif kind in ("kv", "kvo", "kvr"):
            lcol, lfw = _lbl(a)
            vcol = ORANGE if kind == "kvo" else (RED if kind == "kvr" else "#333333")
            ax_leg.text(0.04, y, a, fontsize=12.5, va="top", transform=ax_leg.transAxes,
                        color=lcol, fontweight=lfw)
            ax_leg.text(0.99, y, b, fontsize=12.5, va="top", ha="right",
                        transform=ax_leg.transAxes, color=vcol)
            y -= ystep

    out_path = OUTPUTS / "backtest_robustness.jpg"
    fig.savefig(out_path, dpi=100, bbox_inches="tight", format="jpeg",
                pil_kwargs={"quality": 60, "optimize": True})
    plt.close(fig)

    print(f"\n{'='*70}")
    print(f"  Original:  brut {orig_m['ann_gross_pct']:+.1f}%/an  net {orig_m['net_ann_pct']:+.1f}%/an  "
          f"Sharpe {orig_m['sharpe']:.2f}  DD {orig_m['max_dd_pct']:.1f}%")
    print(f"  Médian:    brut {ann_g[0]:+.1f}%/an  net {ann_n[0]:+.1f}%/an  "
          f"Sharpe {shp[0]:.2f}  DD {dd[0]:.1f}%")
    print(f"  Min:       brut {ann_g[1]:+.1f}%/an  net {ann_n[1]:+.1f}%/an  Sharpe {shp[1]:.2f}")
    print(f"  Max:       brut {ann_g[2]:+.1f}%/an  net {ann_n[2]:+.1f}%/an  Sharpe {shp[2]:.2f}")
    print(f"{'='*70}")
    print(f"\nSaved → outputs/backtest_robustness.jpg")
    print(f"Saved → outputs/backtest_robustness.csv")


# ===========================================================================
#  Entry point — runs the equity backtest and, in parallel, the robustness
#  backtest. A single `python backtest.py` produces every artefact:
#    outputs/backtest_equity.jpg        global equity chart
#    outputs/backtest_equity_YYYY.jpg   per-year equity charts
#    outputs/backtest_pie.jpg           global winners/losers pie
#    outputs/backtest_pie_YYYY.jpg      per-year winners/losers pies
#    outputs/backtest_robustness.jpg    robustness chart
# ===========================================================================

def main():
    # Child invocation: run only the robustness backtest.
    if "--robustness" in sys.argv[1:]:
        run_robustness()
        return

    # Equity backtest only (robustness disabled for iteration speed).
    run_equity()
    print("\n■ all backtests done — equity, per-year, pies")


if __name__ == "__main__":
    main()
