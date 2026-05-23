"""
Portfolio backtest for MyQTM-ETF.

Walk-forward structure (matches train.py):
  For each step, apply fixed params to test predictions (split="test") → true OOS returns.

Allocation model:
  Score > 0 → on, softmax(score/temperature) × Sharpe weights among positives.
  Hysteresis rebalance: only when an allocated asset leaves the monitor top-N.

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

sys.path.insert(0, str(Path(__file__).parent))

DATA    = Path(__file__).parent / "data"
OUTPUTS = Path(__file__).parent / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)

USE_SOFTMAX = True  # True = softmax(score/T) × Sharpe, False = equal weight among score > 0 × Sharpe
SHARPE_POWER = 0.0  # Sharpe weight = sharpe^SHARPE_POWER (0=equal, 1=linear, 2=concentrated)

TEMPERATURE = 1.0     # softmax concentration (fixed)
TOP_N_ALLOC   = 3     # allocate to top 3 (model mode)
TOP_N_MONITOR = 3     # monitor top 3 — rebalance only when allocated asset leaves top 3
CALM_TOP_N    = 1     # calm/fallback mode: top 1 by rolling 2y Sharpe
# --- Follow Leads IC gate (same mechanism as Smart Money gate) ---
# Use causal EMA of past FL test_ic values (no leakage: EMA of steps < current).
# FL model activates only when EMA > FL_IC_GATE_THRESHOLD AND step >= FL_IC_GATE_MIN_STEPS.
# Full capital always invested (no cash scaling).
FL_IC_GATE_THRESHOLD = 0.015  # require EMA(past FL test_ic) > 0.015 to use FL model
FL_IC_GATE_SPAN      = 24     # EMA span (same as SM gate, ~2y)
FL_IC_GATE_MIN_STEPS = 84     # min steps before gate can open (same as SM gate)
FL_SHARPE_POWER = 2           # within top-N: weight by expanding Sharpe^FL_SHARPE_POWER (0=FL scores)
VIX_SPIKE_MIN = 6.0        # VIX 5-day change above this → cash-out
VIX_SPIKE_CASH_DAYS = 5    # days to stay in cash after spike detection
VIX_CALM_THRESHOLD = 19.0  # VIX EMA100 below this → calm market
VIX_CALM_COND_EMA100_SUP_EMA300 = False  # if True, also require EMA100 < EMA300

# --- Test-IC gate ------------------------------------------------------------
# Only deploy the model when the EMA12 of past test-IC values (causal: from
# previous WF steps) exceeds IC_GATE_THRESHOLD. While the gate is closed the
# entire step uses calm mode (top-3 rolling 252d Sharpe). With MIN_TRAIN_ROWS
# at 2772, the first test step is ~2011 with a small cross-section; the gate
# naturally keeps the model dormant until ~2018 when most ETFs are active and
# the test_ic EMA12 stabilises above the threshold.
IC_GATE_THRESHOLD = 0.030  # require EMA(past test_ic) > 0.030 to deploy model
IC_GATE_SPAN      = 24     # EMA span (~2y) — long-term test_ic, ignores short bumps
IC_GATE_MIN_STEPS = 84     # need ≥84 past steps (~7y) — gates the model off before 2018

FLAT_TAX_RATE = 0.30       # PFU 30% on realized gains (paid Jan 1st)
INIT_CAPITAL  = 150_000.0  # portfolio starting capital

# --- Calm allocator: "heuristic" = naive top-3 by rolling 252d Sharpe,
#     "follow_leads" = use the Follow Leads XGB predictions (top-3 by score).
#     Set via env var so knowledge/backtest_calm_compare.py can flip it without
#     editing this file.
import os as _os  # local alias to avoid polluting module namespace
CALM_ALLOCATOR = _os.environ.get("CALM_ALLOCATOR", "follow_leads")

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


def run_backtest(
    scores_wide: pd.DataFrame,
    daily_returns_wide: pd.DataFrame,
    prev_weights: np.ndarray | None = None,
    sharpe_weights: np.ndarray | None = None,
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

    # Lot-based portfolio simulation with realistic fees
    LOT_SIZE = 10_000.0       # 10k€ tranches

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
    """Pivot (date, etf_id) long → wide scores, padded to the full UNIVERSE.

    ETFs whose data starts mid-backtest (e.g. RING in 2012-02) are absent from
    OOS predictions during early steps. Padding with NaN keeps the column set
    constant across steps so that per-ETF arrays (sharpe weights, monitor
    masks) line up by position.
    """
    from etf import UNIVERSE as _UNIVERSE
    if "score_A" in step_data.columns and step_data["score_A"].notna().any():
        scores = step_data["score_A"].unstack("etf_id").sort_index()
    else:
        scores = step_data["score"].unstack("etf_id").sort_index()
    full_etfs = [e.bourso for e in _UNIVERSE]
    missing = [c for c in full_etfs if c not in scores.columns]
    for c in missing:
        scores[c] = np.nan
    return scores[full_etfs]


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
                     eq_orange: pd.Series | None = None,
                     model_active_dates: pd.DatetimeIndex | None = None,
                     fl_active_dates: pd.DatetimeIndex | None = None) -> None:
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
        vix_ema100_chart = vix_raw.ewm(span=100).mean()
        vix_ema300_chart = vix_raw.ewm(span=300).mean()
        vix_5d_chart = vix_raw.diff(5)

        # Regime masks
        # 1) Spike: VIX 5d change > threshold → cash for 3 days
        _spike_mask = pd.Series(False, index=vix_raw.index)
        for _sd in vix_5d_chart[vix_5d_chart > VIX_SPIKE_MIN].index:
            _pos = vix_raw.index.get_loc(_sd)
            for _off in range(VIX_SPIKE_CASH_DAYS):
                if _pos + _off < len(vix_raw):
                    _spike_mask.iloc[_pos + _off] = True
        # 2) Model deployed: only the dates the backtest loop actually ran the
        #    model (IC gate open AND per-day model regime). Fallback to the
        #    VIX-only heuristic if no mask was passed in.
        if model_active_dates is not None and len(model_active_dates) > 0:
            _model_mask = pd.Series(False, index=vix_raw.index)
            _intersect = vix_raw.index.intersection(model_active_dates)
            _model_mask.loc[_intersect] = True
        else:
            _calm_cond = vix_ema100_chart < VIX_CALM_THRESHOLD
            if VIX_CALM_COND_EMA100_SUP_EMA300:
                _calm_cond = _calm_cond & (vix_ema100_chart < vix_ema300_chart)
            _model_mask = ~_calm_cond.reindex(vix_raw.index).fillna(False)

        # Color per date:
        #   green  = heuristic top-3 Sharpe (pre-gate-open OR low FL confidence)
        #   red    = Smart Money model deployed (VIX >= 19 + gate open)
        #   orange = Follow Leads model (VIX < 19 + high FL confidence)
        #   grey   = VIX spike cash-out
        _colors = pd.Series("#2ca02c", index=vix_raw.index)    # default: green (heuristic)
        _colors[_model_mask.values] = "#d62728"                # red (Smart Money)
        if fl_active_dates is not None and len(fl_active_dates) > 0:
            _fl_mask = pd.Series(False, index=vix_raw.index)
            _fl_intersect = vix_raw.index.intersection(fl_active_dates)
            _fl_mask.loc[_fl_intersect] = True
            _colors[_fl_mask.values] = "#ff7f0e"               # orange (Follow Leads)
        _colors[_spike_mask.values] = "#6e6e6e"                # grey (spike)

        # Draw VIX line colored by regime (segment by segment)
        prev_c = _colors.iloc[0]
        seg_start = 0
        for _i in range(1, len(vix_raw)):
            if _colors.iloc[_i] != prev_c or _i == len(vix_raw) - 1:
                end = _i + 1 if _i == len(vix_raw) - 1 else _i + 1
                seg = slice(seg_start, end)
                ax1b.fill_between(vix_raw.index[seg], vix_raw.values[seg],
                                  alpha=0.15, color=prev_c, linewidth=0)
                ax1b.plot(vix_raw.index[seg], vix_raw.values[seg],
                          color=prev_c, alpha=0.6, linewidth=0.8)
                seg_start = _i
                prev_c = _colors.iloc[_i]

        ax1b.plot(vix_ema100_chart.index, vix_ema100_chart.values, color="#d62728", alpha=0.7,
                  linewidth=2.0, linestyle=(0, (8, 4)))
        ax1b.plot(vix_ema300_chart.index, vix_ema300_chart.values, color="#1565c0", alpha=0.7,
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




def run_equity():
    """Equity backtest → global equity chart."""
    oos_path = DATA / "oos_predictions.parquet"
    if not oos_path.exists():
        sys.exit(f"ERROR: {oos_path} not found — run train.py first")

    print("Loading OOS predictions...")
    oos = pd.read_parquet(oos_path)
    oos = oos.reset_index()
    oos["date"] = pd.to_datetime(oos["date"])
    oos = oos.set_index(["date", "etf_id"])

    # Load Follow Leads predictions if the calm allocator is configured to use
    # them. Cached as {step → DataFrame[date, etf_id, score]} for O(1) lookup.
    fl_test_by_step = None
    if CALM_ALLOCATOR == "follow_leads":
        fl_path = DATA / "follow_leads" / "oos_predictions.parquet"
        if not fl_path.exists():
            sys.exit(f"ERROR: CALM_ALLOCATOR=follow_leads but {fl_path} missing — "
                     "run train_follow_leads.py first")
        fl_full = pd.read_parquet(fl_path).reset_index()
        fl_full["date"] = pd.to_datetime(fl_full["date"])
        fl_test_full = fl_full[fl_full["split"] == "test"].dropna(subset=["score"])
        fl_test_by_step = {s: g for s, g in fl_test_full.groupby("step")}
        print(f"Follow Leads predictions loaded: {len(fl_test_full)} test rows over "
              f"{len(fl_test_by_step)} steps")

    if "split" not in oos.columns:
        sys.exit("ERROR: oos_predictions.parquet has no 'split' column — retrain with updated train.py")

    # Load VIX for calm-market detection + spike cash-out
    vix_path = DATA / "fred_vix.parquet"
    vix_s = pd.read_parquet(vix_path).iloc[:, 0] if vix_path.exists() else pd.Series(dtype=float)

    # VIX EMA for calm-market detection
    vix_ema100 = vix_s.ewm(span=100).mean() if not vix_s.empty else pd.Series(dtype=float)
    vix_ema300 = vix_s.ewm(span=300).mean() if not vix_s.empty else pd.Series(dtype=float)

    # Load actual daily returns for all ETFs (from price data)
    from etf import UNIVERSE
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

    # --- Pre-compute per-step test_ic and its causal EMA12 (IC gate input) ---
    # Recompute test_ic from predictions (rather than relying on the column
    # saved by train.py — handles legacy files lacking that column).
    test_oos = oos[oos["split"] == "test"].dropna(subset=["score", "label"]).reset_index()
    test_ic_per_step = (
        test_oos.groupby("step")
        .apply(lambda g: g.groupby("date")
               .apply(lambda x: x["score"].corr(x["label"]) if len(x) > 1 else np.nan,
                      include_groups=False).mean(),
               include_groups=False)
        .reindex(all_steps)
    )
    # Causal EMA: at step S, only know test_ic for steps < S.
    test_ic_ema_lagged = (
        test_ic_per_step.shift(1).ewm(span=IC_GATE_SPAN, min_periods=1).mean()
    )
    n_open = sum(
        1 for s in steps
        if s >= IC_GATE_MIN_STEPS
        and pd.notna(test_ic_ema_lagged.get(s))
        and test_ic_ema_lagged.get(s) > IC_GATE_THRESHOLD
    )
    print(f"IC gate: threshold {IC_GATE_THRESHOLD:+.3f}  span {IC_GATE_SPAN}  "
          f"min steps {IC_GATE_MIN_STEPS}")
    print(f"  → model deploys on {n_open}/{len(steps)} steps "
          f"({n_open/max(len(steps),1):.0%}); other steps stay in calm mode.")
    print(f"Calm allocator: {CALM_ALLOCATOR}\n")

    # FL IC gate: causal EMA of past FL test_ic, same mechanism as SM gate.
    fl_test_ic_ema_lagged = pd.Series(dtype=float)
    if fl_test_by_step is not None:
        fl_ic_per_step = (
            fl_test_full.groupby("step")
            .apply(lambda g: g.groupby("date")
                   .apply(lambda x: x["score"].corr(x["label"]) if len(x) > 1 else np.nan,
                          include_groups=False)
                   .mean(), include_groups=False)
            .reindex(all_steps)
        )
        fl_test_ic_ema_lagged = (
            fl_ic_per_step.shift(1).ewm(span=FL_IC_GATE_SPAN, min_periods=1).mean()
        )
        n_fl_open = sum(
            1 for s in steps
            if s >= FL_IC_GATE_MIN_STEPS
            and pd.notna(fl_test_ic_ema_lagged.get(s))
            and fl_test_ic_ema_lagged.get(s) > FL_IC_GATE_THRESHOLD
        )
        print(f"FL IC gate: threshold {FL_IC_GATE_THRESHOLD:+.3f}  span {FL_IC_GATE_SPAN}  "
              f"min steps {FL_IC_GATE_MIN_STEPS}")
        print(f"  → FL model used on {n_fl_open}/{len(steps)} calm steps "
              f"({n_fl_open/max(len(steps),1):.0%})\n")

    all_test_returns = []
    all_test_weights = []
    all_params_rows  = []
    all_test_scores  = []   # scores_A per test step
    model_active_days = []  # list of DatetimeIndex segments where SM model deployed
    fl_active_days    = []  # list of DatetimeIndex segments where Follow Leads deployed
    carry_weights    = None   # chain positions between steps
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
        etf_sharpe_raw = (exp_mean / exp_std.replace(0, np.nan)).clip(0.0).fillna(0.0)
        etf_sharpe = etf_sharpe_raw.values ** SHARPE_POWER

        # --- Evaluate on test (true OOS) ---
        test_scores = _pivot_step(test_data)

        # Remap carry_weights to current ETF columns
        prev_w = None
        if carry_weights is not None:
            prev_w = np.zeros(len(test_scores.columns))
            for i, etf in enumerate(test_scores.columns):
                if etf in carry_weights:
                    prev_w[i] = carry_weights[etf]

        # --- Build MODEL scores (always, used on non-calm days) ---
        model_row_mean = test_scores.mean(axis=1)
        model_row_std  = test_scores.std(axis=1).replace(0, 1.0)
        model_scores   = test_scores.sub(model_row_mean, axis=0).div(model_row_std, axis=0)
        block_ids = [CORR_BLOCKS.get(c, c) for c in model_scores.columns]
        n_cols = len(model_scores.columns)
        model_monitor_mask = np.zeros((len(model_scores), n_cols), dtype=bool)
        for idx in range(len(model_scores)):
            row = model_scores.iloc[idx].values
            keep_monitor = _topn_keep(row, TOP_N_MONITOR, block_ids)
            model_monitor_mask[idx, keep_monitor] = True
            keep_alloc = _topn_keep(row, TOP_N_ALLOC, block_ids)
            pos_idx = np.where((~np.isnan(row)) & (row > 0))[0]
            drop = np.setdiff1d(pos_idx, keep_alloc)
            model_scores.iloc[idx, drop] = np.nan

        # --- Build calm-mode picks (always computed; consumed on calm days) ---
        # Rolling 2y Sharpe (heuristic baseline) — always computed as fallback.
        roll_mean = dr_hist.rolling(504, min_periods=400).mean().iloc[-1] * 252
        roll_std  = dr_hist.rolling(504, min_periods=400).std().iloc[-1] * np.sqrt(252)
        roll_sharpe = (roll_mean / roll_std.replace(0, np.nan)).clip(0.0).fillna(0.0)
        heuristic_sharpe = roll_sharpe.reindex(model_scores.columns).fillna(0.0)

        use_fl_model = False
        if CALM_ALLOCATOR == "follow_leads" and fl_test_by_step is not None:
            fl_step = fl_test_by_step.get(step)
            if fl_step is not None and not fl_step.empty:
                fl_gate_ema = fl_test_ic_ema_lagged.get(step, np.nan)
                use_fl_model = (
                    step >= FL_IC_GATE_MIN_STEPS
                    and pd.notna(fl_gate_ema)
                    and fl_gate_ema > FL_IC_GATE_THRESHOLD
                )

        exp_sharpe_for_floor = etf_sharpe_raw.reindex(model_scores.columns).fillna(0.0)

        if use_fl_model:
            # High confidence → top-N by FL model score.
            fl_mean = fl_step.groupby("etf_id")["score"].mean()
            fl_series = fl_mean.reindex(model_scores.columns).fillna(0.0)
            top_calm_etfs = fl_series.nlargest(TOP_N_ALLOC).index.tolist()
            # Weight within top-N by expanding Sharpe^FL_SHARPE_POWER.
            sharpe_series = (exp_sharpe_for_floor ** FL_SHARPE_POWER)
        else:
            # Low confidence or no FL data → top-N by rolling 2y Sharpe.
            sharpe_series = heuristic_sharpe.copy()
            top_calm_etfs = sharpe_series.nlargest(CALM_TOP_N).index.tolist()
            top_calm_etfs = [e for e in top_calm_etfs if sharpe_series[e] > 0]

        top_calm_set = set(top_calm_etfs)
        sharpe_row = np.array([sharpe_series[c] if c in top_calm_set else np.nan
                               for c in model_scores.columns])
        sharpe_monitor_row = np.array([c in top_calm_set for c in model_scores.columns])

        # --- Per-day calm mask: matches the chart, no step-boundary lag ---
        calm_per_day = pd.Series(False, index=model_scores.index)
        if not vix_ema100.empty and not vix_ema300.empty:
            ema100_t = vix_ema100.reindex(model_scores.index, method="ffill")
            ema300_t = vix_ema300.reindex(model_scores.index, method="ffill")
            cond = (ema100_t < VIX_CALM_THRESHOLD)
            if VIX_CALM_COND_EMA100_SUP_EMA300:
                cond = cond & (ema100_t < ema300_t)
            calm_per_day = cond.fillna(False)

        # --- IC gate: if the EMA12 of past test_ic is below threshold, force
        #     the entire step into calm mode (no model trades) ---
        gate_ema = test_ic_ema_lagged.get(step, np.nan)
        gate_open = (step >= IC_GATE_MIN_STEPS) and pd.notna(gate_ema) \
            and gate_ema > IC_GATE_THRESHOLD
        if not gate_open:
            calm_per_day[:] = True

        # Record dates where the model was actually deployed (gate open AND
        # per-day calm condition false). Drives the equity chart's VIX
        # colouring downstream.
        if gate_open:
            model_days_step = calm_per_day.index[~calm_per_day.values]
            if len(model_days_step) > 0:
                model_active_days.append(model_days_step)
        if use_fl_model:
            fl_days_step = calm_per_day.index[calm_per_day.values]
            if len(fl_days_step) > 0:
                fl_active_days.append(fl_days_step)

        # --- Combine model + Sharpe per day ---
        test_scores = model_scores.copy()
        step_monitor_mask = model_monitor_mask.copy()
        for i, calm in enumerate(calm_per_day.values):
            if calm:
                test_scores.iloc[i] = sharpe_row
                step_monitor_mask[i] = sharpe_monitor_row

        # VIX spike — close-only detection.
        # Detect close J → sell at close J (same day cash).
        cash_dates = set()
        if not vix_s.empty:
            vix_close_test = vix_s.reindex(test_scores.index, method="ffill")
            vix_5d_close = vix_close_test.diff(5)
            spike_dates = vix_5d_close[vix_5d_close > VIX_SPIKE_MIN].index
            for sd in spike_dates:
                if sd not in test_scores.index:
                    continue
                sd_pos = test_scores.index.get_loc(sd)
                for offset in range(1, VIX_SPIKE_CASH_DAYS + 1):
                    cash_pos = sd_pos + offset
                    if 0 <= cash_pos < len(test_scores):
                        cash_dates.add(test_scores.index[cash_pos])
            for cd in cash_dates:
                test_scores.loc[cd] = np.nan
                step_monitor_mask[test_scores.index.get_loc(cd)] = False

        test_dr = daily_ret_panel.reindex(index=test_scores.index, columns=test_scores.columns).fillna(0)
        test_returns, test_weights, step_fees, step_taxes, step_rebals, _, chain_tax_state = run_backtest(
                                                   test_scores, test_dr,
                                                   prev_weights=prev_w,
                                                   sharpe_weights=etf_sharpe,
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
        all_params_rows.append({"step": step, "temperature": TEMPERATURE})
        # Save scores_A for chart (first day of test step per ETF)
        all_test_scores.append(test_scores.iloc[[0]])

        # Live progress after each step
        tmp_returns = pd.concat(all_test_returns).sort_index()
        tmp_eq = (1 + tmp_returns).cumprod()
        tmp_sharpe = sharpe(tmp_returns)
        tmp_dd = (tmp_eq / tmp_eq.cummax() - 1).min()
        gate_str = (f"gate={'MODEL' if gate_open else 'CALM '} "
                    f"(ema={gate_ema:+.3f})" if pd.notna(gate_ema)
                    else f"gate=CALM  (ema=  n/a)")
        calm_src = "FL" if use_fl_model else "heur"
        print(f"       cumul: {tmp_eq.iloc[-1]-1:+.1%}  sharpe={tmp_sharpe:.2f}  dd={tmp_dd:.1%}  "
              f"{gate_str}  calm={calm_src}", flush=True)

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
    params_df.to_csv(OUTPUTS / "backtest_steps.csv", index=False)

    eq_df = eq_curve.reset_index()
    eq_df.columns = ["date", "equity"]
    eq_df.to_csv(OUTPUTS / "backtest_equity.csv", index=False)

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
    full_model_idx = (
        pd.concat([pd.Series(idx) for idx in model_active_days]).values
        if model_active_days else np.array([], dtype="datetime64[ns]")
    )
    full_fl_idx = (
        pd.concat([pd.Series(idx) for idx in fl_active_days]).values
        if fl_active_days else np.array([], dtype="datetime64[ns]")
    )
    full_model_dates = pd.DatetimeIndex(full_model_idx)
    full_fl_dates = pd.DatetimeIndex(full_fl_idx)
    _save_equity_png(port_returns, eq_red, all_weights, params_df, OUTPUTS,
                     scores_A=all_scores, fin=fin, eq_orange=eq_orange,
                     model_active_dates=full_model_dates,
                     fl_active_dates=full_fl_dates)

    print(f"\nSaved → {DATA.relative_to(Path(__file__).parent)}/backtest_results.parquet")
    print(f"Saved → {OUTPUTS.relative_to(Path(__file__).parent)}/best_params.csv")
    print(f"Saved → {OUTPUTS.relative_to(Path(__file__).parent)}/backtest_steps.csv")
    print(f"Saved → {OUTPUTS.relative_to(Path(__file__).parent)}/backtest_equity.csv")
    print(f"Saved → {OUTPUTS.relative_to(Path(__file__).parent)}/backtest_equity.jpg")



# ===========================================================================
#  Robustness backtest — N_RUNS perturbed backtests dropping random ETFs
# ===========================================================================

def _rob_sharpe(returns: pd.Series) -> float:
    if len(returns) < 10 or returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(252))


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


def _run_single(oos, steps, daily_ret_panel, drop_etfs=None, seed=None,
                vix_s=None, vix_ema100=None, vix_ema300=None,
                fl_test_by_step=None,
                test_ic_ema_lagged=None, fl_test_ic_ema_lagged=None):
    """Run a single backtest mirroring the main backtest logic (calm-market
    top-3 Sharpe or Follow Leads, IC gate, hysteresis monitor, VIX spike
    cash-out), optionally dropping N random ETFs per step (applies in all
    modes: SM model, FL model, and Sharpe heuristic fallback).
    Returns (daily_returns, step_fee_records)."""
    rng = np.random.default_rng(seed) if seed is not None else None

    if vix_ema100 is None:
        vix_ema100 = pd.Series(dtype=float)
    if vix_ema300 is None:
        vix_ema300 = pd.Series(dtype=float)
    if vix_s is None:
        vix_s = pd.Series(dtype=float)
    if test_ic_ema_lagged is None:
        test_ic_ema_lagged = pd.Series(dtype=float)
    if fl_test_ic_ema_lagged is None:
        fl_test_ic_ema_lagged = pd.Series(dtype=float)

    all_test_returns = []
    step_fee_records = []
    carry_weights = None
    chain_tax_state = None

    for step in steps:
        val_data = oos[(oos["step"] == step) & (oos["split"] == "val")]
        test_data = oos[(oos["step"] == step) & (oos["split"] == "test")]

        if len(val_data) < 50 or len(test_data) == 0:
            continue

        val_scores = _pivot_step(val_data)
        val_end_date = val_scores.index[-1]
        dr_hist = daily_ret_panel[val_scores.columns].loc[:val_end_date]

        # Expanding Sharpe weights (matches main backtest)
        exp_mean = dr_hist.expanding(min_periods=60).mean().iloc[-1] * 252
        exp_std = dr_hist.expanding(min_periods=60).std().iloc[-1] * np.sqrt(252)
        etf_sharpe = (exp_mean / exp_std.replace(0, np.nan)).clip(0.0).fillna(0.0).values
        etf_sharpe = etf_sharpe ** SHARPE_POWER

        test_scores = _pivot_step(test_data)

        # --- Random ETF drop (applies in both calm and model modes) ---
        if drop_etfs is not None and rng is not None:
            available = test_scores.columns.tolist()
            if isinstance(drop_etfs, tuple):
                n_drop = rng.integers(drop_etfs[0], drop_etfs[1] + 1)
            else:
                n_drop = drop_etfs
            n_drop = min(n_drop, len(available) - 2)
            dropped = rng.choice(available, size=n_drop, replace=False)
            test_scores[dropped] = np.nan
            dropped_set = set(dropped.tolist())
        else:
            dropped_set = set()

        # Remap carry_weights to current ETF columns
        prev_w = None
        if carry_weights is not None:
            prev_w = np.zeros(len(test_scores.columns))
            for i, etf in enumerate(test_scores.columns):
                if etf in carry_weights:
                    prev_w[i] = carry_weights[etf]

        # --- Build MODEL scores (always, used on non-calm days) ---
        model_row_mean = test_scores.mean(axis=1)
        model_row_std  = test_scores.std(axis=1).replace(0, 1.0)
        model_scores   = test_scores.sub(model_row_mean, axis=0).div(model_row_std, axis=0)
        n_cols = len(model_scores.columns)
        block_ids = [CORR_BLOCKS.get(c, c) for c in model_scores.columns]
        model_monitor_mask = np.zeros((len(model_scores), n_cols), dtype=bool)
        for idx in range(len(model_scores)):
            row = model_scores.iloc[idx].values
            keep_monitor = _topn_keep(row, TOP_N_MONITOR, block_ids)
            model_monitor_mask[idx, keep_monitor] = True
            keep_alloc = _topn_keep(row, TOP_N_ALLOC, block_ids)
            pos_idx = np.where((~np.isnan(row)) & (row > 0))[0]
            drop_idx = np.setdiff1d(pos_idx, keep_alloc)
            model_scores.iloc[idx, drop_idx] = np.nan

        # --- Build CALM picks (FL model if available, else heuristic Sharpe) ---
        roll_mean = dr_hist.rolling(504, min_periods=400).mean().iloc[-1] * 252
        roll_std  = dr_hist.rolling(504, min_periods=400).std().iloc[-1] * np.sqrt(252)
        roll_sharpe = (roll_mean / roll_std.replace(0, np.nan)).clip(0.0).fillna(0.0)
        sharpe_series = roll_sharpe.reindex(model_scores.columns).fillna(0.0)
        for etf in dropped_set:
            if etf in sharpe_series.index:
                sharpe_series[etf] = 0.0

        # Expanding Sharpe for floor filter
        exp_mean_r = dr_hist.expanding(min_periods=60).mean().iloc[-1] * 252
        exp_std_r = dr_hist.expanding(min_periods=60).std().iloc[-1] * np.sqrt(252)
        exp_sharpe_for_floor = (exp_mean_r / exp_std_r.replace(0, np.nan)).clip(0.0).fillna(0.0)
        exp_sharpe_for_floor = exp_sharpe_for_floor.reindex(model_scores.columns).fillna(0.0)

        use_fl = False
        if fl_test_by_step is not None:
            fl_step = fl_test_by_step.get(step)
            if fl_step is not None and not fl_step.empty:
                fl_mean = fl_step.groupby("etf_id")["score"].mean()
                fl_series = fl_mean.reindex(model_scores.columns).fillna(0.0)
                # Zero out dropped ETFs
                for etf in fl_series.index:
                    if etf in dropped_set:
                        fl_series[etf] = -999.0
                top_calm_etfs = [e for e in fl_series.nlargest(TOP_N_ALLOC).index
                                 if fl_series[e] > -999.0]
                use_fl = True

        if not use_fl:
            top_calm_etfs = [e for e in sharpe_series.nlargest(CALM_TOP_N).index
                             if sharpe_series[e] > 0]

        top_calm_set = set(top_calm_etfs)
        sharpe_row = np.array([sharpe_series[c] if c in top_calm_set else np.nan
                               for c in model_scores.columns])
        sharpe_monitor_row = np.array([c in top_calm_set for c in model_scores.columns])

        # --- Per-day calm mask (matches main backtest) ---
        calm_per_day = pd.Series(False, index=model_scores.index)
        if not vix_ema100.empty and not vix_ema300.empty:
            ema100_t = vix_ema100.reindex(model_scores.index, method="ffill")
            ema300_t = vix_ema300.reindex(model_scores.index, method="ffill")
            cond = (ema100_t < VIX_CALM_THRESHOLD)
            if VIX_CALM_COND_EMA100_SUP_EMA300:
                cond = cond & (ema100_t < ema300_t)
            calm_per_day = cond.fillna(False)

        # --- IC gate: force calm if SM model not yet reliable ---
        gate_ema = test_ic_ema_lagged.get(step, np.nan) if not test_ic_ema_lagged.empty else np.nan
        gate_open = (step >= IC_GATE_MIN_STEPS) and pd.notna(gate_ema) \
            and gate_ema > IC_GATE_THRESHOLD
        if not gate_open:
            calm_per_day[:] = True

        # --- FL IC gate: use FL model in calm only if its own gate is open ---
        if use_fl:
            fl_gate_ema = fl_test_ic_ema_lagged.get(step, np.nan) if not fl_test_ic_ema_lagged.empty else np.nan
            fl_gate_ok = (step >= FL_IC_GATE_MIN_STEPS) and pd.notna(fl_gate_ema) \
                and fl_gate_ema > FL_IC_GATE_THRESHOLD
            if not fl_gate_ok:
                use_fl = False
                # Fall back to heuristic for this step
                top_calm_etfs = [e for e in sharpe_series.nlargest(CALM_TOP_N).index
                                 if sharpe_series[e] > 0]
                top_calm_set = set(top_calm_etfs)
                sharpe_row = np.array([sharpe_series[c] if c in top_calm_set else np.nan
                                       for c in model_scores.columns])
                sharpe_monitor_row = np.array([c in top_calm_set for c in model_scores.columns])

        # --- Combine per day ---
        test_scores = model_scores.copy()
        step_monitor_mask = model_monitor_mask.copy()
        for i, calm in enumerate(calm_per_day.values):
            if calm:
                test_scores.iloc[i] = sharpe_row
                step_monitor_mask[i] = sharpe_monitor_row

        # VIX spike → cash-out (same logic as main backtest)
        cash_dates = set()
        if not vix_s.empty:
            vix_close_test = vix_s.reindex(test_scores.index, method="ffill")
            vix_5d_close = vix_close_test.diff(5)
            spike_dates = vix_5d_close[vix_5d_close > VIX_SPIKE_MIN].index
            for sd in spike_dates:
                if sd not in test_scores.index:
                    continue
                sd_pos = test_scores.index.get_loc(sd)
                for offset in range(1, VIX_SPIKE_CASH_DAYS + 1):
                    cash_pos = sd_pos + offset
                    if 0 <= cash_pos < len(test_scores):
                        cash_dates.add(test_scores.index[cash_pos])
            for cd in cash_dates:
                test_scores.loc[cd] = np.nan
                step_monitor_mask[test_scores.index.get_loc(cd)] = False

        test_dr = daily_ret_panel.reindex(index=test_scores.index, columns=test_scores.columns).fillna(0)
        result = run_backtest(test_scores, test_dr,
                              prev_weights=prev_w,
                              sharpe_weights=etf_sharpe,
                              tax_state=chain_tax_state,
                              monitor_mask=step_monitor_mask)
        test_returns, test_weights, step_fees = result[0], result[1], result[2]
        chain_tax_state = result[6]
        carry_weights = dict(zip(test_weights.columns, test_weights.iloc[-1].values))
        step_fee_records.append((test_returns.index, step_fees))
        all_test_returns.append(test_returns)

    if not all_test_returns:
        return pd.Series(dtype=float), []
    return pd.concat(all_test_returns).sort_index(), step_fee_records


def run_robustness():
    """Robustness backtest → outputs/{mode}/backtest_robustness.jpg + summary CSV."""
    oos_path = DATA / "oos_predictions.parquet"
    if not oos_path.exists():
        sys.exit(f"ERROR: {oos_path} not found")

    print("Loading OOS predictions...", flush=True)
    oos = pd.read_parquet(oos_path)
    oos = oos.reset_index()
    oos["date"] = pd.to_datetime(oos["date"])
    oos = oos.set_index(["date", "etf_id"])

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

    # Load VIX for adaptive + calm-market logic
    vix_path = DATA / "fred_vix.parquet"
    vix_s = pd.read_parquet(vix_path).iloc[:, 0] if vix_path.exists() else pd.Series(dtype=float)
    vix_ema100 = vix_s.ewm(span=100).mean() if not vix_s.empty else pd.Series(dtype=float)
    vix_ema300 = vix_s.ewm(span=300).mean() if not vix_s.empty else pd.Series(dtype=float)

    # Load Follow Leads predictions for calm mode (if available)
    fl_test_by_step = None
    if CALM_ALLOCATOR == "follow_leads":
        fl_path = DATA / "follow_leads" / "oos_predictions.parquet"
        if fl_path.exists():
            fl_full = pd.read_parquet(fl_path).reset_index()
            fl_full["date"] = pd.to_datetime(fl_full["date"])
            fl_test_full = fl_full[fl_full["split"] == "test"].dropna(subset=["score"])
            fl_test_by_step = {s: g for s, g in fl_test_full.groupby("step")}
            print(f"Follow Leads loaded for robustness: {len(fl_test_by_step)} steps")

    # --- Pre-compute IC gates (same causal EMA as main backtest) ---
    test_oos_rob = oos[oos["split"] == "test"].dropna(subset=["score", "label"]).reset_index()
    test_ic_per_step = (
        test_oos_rob.groupby("step")
        .apply(lambda g: g.groupby("date")
               .apply(lambda x: x["score"].corr(x["label"]) if len(x) > 1 else np.nan,
                      include_groups=False).mean(),
               include_groups=False)
        .reindex(all_steps)
    )
    test_ic_ema_lagged = (
        test_ic_per_step.shift(1).ewm(span=IC_GATE_SPAN, min_periods=1).mean()
    )

    fl_test_ic_ema_lagged = pd.Series(dtype=float)
    if fl_test_by_step is not None:
        fl_test_full_rob = pd.concat(fl_test_by_step.values())
        fl_ic_per_step = (
            fl_test_full_rob.reset_index().groupby("step")
            .apply(lambda g: g.groupby("date")
                   .apply(lambda x: x["score"].corr(x["label"]) if len(x) > 1 else np.nan,
                          include_groups=False).mean(),
                   include_groups=False)
            .reindex(all_steps)
        )
        fl_test_ic_ema_lagged = (
            fl_ic_per_step.shift(1).ewm(span=FL_IC_GATE_SPAN, min_periods=1).mean()
        )

    print(f"Steps: {len(steps)}, ETFs: {len(all_etfs)}, Runs: {N_RUNS}, Drop: {N_DROP}")

    # Run original (no drop) first
    print("Running original (no drop)...", flush=True)
    orig_returns, orig_fees = _run_single(oos, steps, daily_ret_panel,
                                          drop_etfs=None, seed=None, vix_s=vix_s,
                                          vix_ema100=vix_ema100, vix_ema300=vix_ema300,
                                          fl_test_by_step=fl_test_by_step,
                                          test_ic_ema_lagged=test_ic_ema_lagged,
                                          fl_test_ic_ema_lagged=fl_test_ic_ema_lagged)
    orig_m, orig_red, orig_orange = _run_metrics(orig_returns, orig_fees)

    # Run N_RUNS with random drops
    all_run_red = []
    stats = []
    for run in range(N_RUNS):
        print(f"  Run {run+1:2d}/{N_RUNS}...", end="", flush=True)
        run_returns, run_fees = _run_single(oos, steps, daily_ret_panel,
                                            drop_etfs=N_DROP, seed=run, vix_s=vix_s,
                                            vix_ema100=vix_ema100, vix_ema300=vix_ema300,
                                            fl_test_by_step=fl_test_by_step,
                                            test_ic_ema_lagged=test_ic_ema_lagged,
                                            fl_test_ic_ema_lagged=fl_test_ic_ema_lagged)
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
        vix_ema100_chart = vix_raw.ewm(span=100).mean()
        vix_ema300_chart = vix_raw.ewm(span=300).mean()
        vix_5d_chart = vix_raw.diff(5)

        # Regime masks
        # 1) Spike: VIX 5d change > threshold → cash for 3 days
        _spike_mask = pd.Series(False, index=vix_raw.index)
        for _sd in vix_5d_chart[vix_5d_chart > VIX_SPIKE_MIN].index:
            _pos = vix_raw.index.get_loc(_sd)
            for _off in range(VIX_SPIKE_CASH_DAYS):
                if _pos + _off < len(vix_raw):
                    _spike_mask.iloc[_pos + _off] = True
        # 2) Regime coloring: VIX-based proxy (robustness chart doesn't track
        #    per-run model/FL active dates — use VIX regime heuristic).
        _calm_cond = vix_ema100_chart < VIX_CALM_THRESHOLD
        if VIX_CALM_COND_EMA100_SUP_EMA300:
            _calm_cond = _calm_cond & (vix_ema100_chart < vix_ema300_chart)
        _model_mask = ~_calm_cond.reindex(vix_raw.index).fillna(False)

        # Color: green=calm, red=stress, grey=spike
        _colors = pd.Series("#2ca02c", index=vix_raw.index)
        _colors[_model_mask.values] = "#d62728"
        _colors[_spike_mask.values] = "#6e6e6e"                # grey (spike)

        # Draw VIX line colored by regime (segment by segment)
        prev_c = _colors.iloc[0]
        seg_start = 0
        for _i in range(1, len(vix_raw)):
            if _colors.iloc[_i] != prev_c or _i == len(vix_raw) - 1:
                end = _i + 1 if _i == len(vix_raw) - 1 else _i + 1
                seg = slice(seg_start, end)
                ax1b.fill_between(vix_raw.index[seg], vix_raw.values[seg],
                                  alpha=0.15, color=prev_c, linewidth=0)
                ax1b.plot(vix_raw.index[seg], vix_raw.values[seg],
                          color=prev_c, alpha=0.6, linewidth=0.8)
                seg_start = _i
                prev_c = _colors.iloc[_i]

        ax1b.plot(vix_ema100_chart.index, vix_ema100_chart.values, color="#d62728", alpha=0.7,
                  linewidth=2.0, linestyle=(0, (8, 4)))
        ax1b.plot(vix_ema300_chart.index, vix_ema300_chart.values, color="#1565c0", alpha=0.7,
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
    print(f"\nSaved → {OUTPUTS.relative_to(Path(__file__).parent)}/backtest_robustness.jpg")
    print(f"Saved → {OUTPUTS.relative_to(Path(__file__).parent)}/backtest_robustness.csv")


# ===========================================================================
#  Entry point — runs the equity backtest and, in parallel, the robustness
#  backtest. A single `python backtest.py` produces every artefact:
#    outputs/backtest_equity.jpg        global equity chart
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
