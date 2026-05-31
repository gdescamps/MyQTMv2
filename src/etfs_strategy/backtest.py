"""
Portfolio backtest for MyQTM-ETF.

Temporal convention (close-to-close):
  All indicators and model scores are computed at close J.
  Portfolio is rebalanced at close J (in practice ~30 min before close).
  Returns earned: close(J) → close(J+1).

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

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

DATA    = ROOT / "data"
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)

USE_SOFTMAX = True
SHARPE_POWER = 0.0
TEMPERATURE = 1.0
TOP_N_ALLOC   = 3
TOP_N_MONITOR = 3
CALM_TOP_N    = 1

FL_IC_GATE_THRESHOLD = 0.015
FL_IC_GATE_SPAN      = 24
FL_IC_GATE_MIN_STEPS = 84
FL_SHARPE_POWER = 2

VIX_SPIKE_MIN = 3.0
VIX_SPIKE_CASH_DAYS = 5
VIX_CALM_THRESHOLD = 19.0
VIX_CASH_THRESHOLD = 20.0
VIX_CALM_COND_EMA100_SUP_EMA300 = False

IC_GATE_THRESHOLD = 0.030
IC_GATE_SPAN      = 24
IC_GATE_MIN_STEPS = 84

INIT_CAPITAL  = 150_000.0

import os as _os
CALM_ALLOCATOR = _os.environ.get("CALM_ALLOCATOR", "follow_leads")

N_RUNS = 50
N_DROP = (5, 10)

BLOCK_DIVERSIFY = False
CORR_BLOCKS = {
    "IVV": "US",   "QQQ": "US",  "ACWI": "US", "SUSA": "US",
    "IEUR": "EUR", "EZU": "EUR",
    "EEM": "EM",   "IEMG": "EM", "EMXC": "EM",
    "ILF": "LATAM", "EWZ": "LATAM",
}

HEURISTIC_START = pd.Timestamp("2006-01-01")

ETF_COLOR_MAP = {
    "IVV": "#1565c0", "QQQ": "#2ca02c", "RING": "#f4b400",
    "ACWI": "#000000", "EEM": "#9467bd", "IEMG": "#8c564b", "EMXC": "#e377c2",
    "ILF": "#bcbd22", "EWY": "#7b4173", "EWT": "#393b79", "EWZ": "#a55194",
    "EWW": "#ce6dbd", "EWC": "#e7969c", "EWJ": "#6b6ecf", "TUR": "#ad494a",
    "FXI": "#de9ed6", "ISF.L": "#637939", "IEUR": "#3182bd", "EZU": "#b5cf6b",
    "EPP": "#9c9ede", "SUSA": "#5254a3", "SOXX": "#756bb1", "ROBO": "#c49c94",
    "ICLN": "#66c2a5", "EXX1.DE": "#1ab0a8", "IEO": "#8b4513", "SXRS.DE": "#bd9e39",
}


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def sharpe(returns: pd.Series, min_obs: int = 30) -> float:
    r = returns.dropna()
    if len(r) < min_obs or r.std() < 1e-10:
        return -10.0
    return float(r.mean() / r.std() * np.sqrt(252))


def _short_name(name: str) -> str:
    drop = {"ishares", "sector", "etf", "acc", "usd"}
    return " ".join(w for w in name.split() if w.lower() not in drop) or name


def _prepend_heuristic_steps(oos: pd.DataFrame, daily_ret_panel: pd.DataFrame) -> pd.DataFrame:
    """Prepend synthetic walk-forward steps from HEURISTIC_START to first OOS test date."""
    from etf import UNIVERSE as _U
    first_oos_test = (oos.reset_index()
                      .query("split == 'test'")["date"].min())
    if HEURISTIC_START >= first_oos_test:
        return oos

    all_bourso = [e.bourso for e in _U]
    trading_days = daily_ret_panel.dropna(how="all").index.sort_values()
    heur_days = trading_days[(trading_days >= HEURISTIC_START) &
                             (trading_days < first_oos_test)]
    _STEP = 21
    n_steps = (len(heur_days) - _STEP) // _STEP
    syn_rows = []
    for k in range(n_steps):
        step_num = -(n_steps - k)
        val_sl  = slice(k * _STEP, (k + 1) * _STEP)
        test_sl = slice((k + 1) * _STEP, (k + 2) * _STEP)
        for split, sl in [("val", val_sl), ("test", test_sl)]:
            for d in heur_days[sl]:
                for etf in all_bourso:
                    syn_rows.append((d, etf, step_num, split))
    if not syn_rows:
        return oos

    syn_df = pd.DataFrame(syn_rows, columns=["date", "etf_id", "step", "split"])
    syn_df["score"] = 0.0
    syn_df["label"] = 0.0
    syn_df["date"] = pd.to_datetime(syn_df["date"])
    syn_df = syn_df.set_index(["date", "etf_id"])
    for col in oos.columns:
        if col not in syn_df.columns:
            syn_df[col] = np.nan
    print(f"Prepended {n_steps} heuristic-only steps "
          f"({HEURISTIC_START.date()} → {first_oos_test.date()})")
    return pd.concat([syn_df[oos.columns], oos])


def _topn_keep(row: np.ndarray, n: int, block_ids: list) -> np.ndarray:
    """Indices of top-n positive scores, at most one per correlated block."""
    pos = np.where((~np.isnan(row)) & (row > 0))[0]
    if len(pos) <= n:
        return pos
    order = pos[np.argsort(row[pos])[::-1]]
    if not BLOCK_DIVERSIFY:
        return order[:n]
    keep, used = [], set()
    for j in order:
        if len(keep) >= n:
            break
        b = block_ids[j]
        if b in used:
            continue
        keep.append(j)
        used.add(b)
    for j in order:
        if len(keep) >= n:
            break
        if j not in keep:
            keep.append(j)
    return np.array(keep, dtype=int)


def _pivot_step(step_data: pd.DataFrame) -> pd.DataFrame:
    """Pivot (date, etf_id) long → wide scores, padded to the full UNIVERSE."""
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
    for fname in [f"{ticker}.parquet", f"{ticker.replace('.', '_')}.parquet"]:
        path = DATA / fname
        if path.exists():
            df = pd.read_parquet(path)
            col = "close" if "close" in df.columns else "adj_close"
            s = df[col].reindex(dates, method="ffill").dropna()
            if len(s) > 10:
                return s / s.iloc[0]
    return None


def _load_daily_returns() -> pd.DataFrame:
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
    return pd.DataFrame(daily_returns_all)


def _load_vix() -> tuple:
    """Returns (vix_s, vix_ema100, vix_ema300)."""
    vix_path = DATA / "fred_vix.parquet"
    if vix_path.exists():
        vix_s = pd.read_parquet(vix_path).iloc[:, 0]
    else:
        vix_s = pd.Series(dtype=float)
    vix_ema100 = vix_s.ewm(span=100).mean() if not vix_s.empty else pd.Series(dtype=float)
    vix_ema300 = vix_s.ewm(span=300).mean() if not vix_s.empty else pd.Series(dtype=float)
    return vix_s, vix_ema100, vix_ema300


def _load_fl_predictions() -> tuple:
    """Returns (fl_test_by_step, fl_test_full) or (None, None)."""
    if CALM_ALLOCATOR != "follow_leads":
        return None, None
    fl_path = DATA / "follow_leads" / "oos_predictions.parquet"
    if not fl_path.exists():
        sys.exit(f"ERROR: CALM_ALLOCATOR=follow_leads but {fl_path} missing — "
                 "run train_follow_leads.py first")
    fl_full = pd.read_parquet(fl_path).reset_index()
    fl_full["date"] = pd.to_datetime(fl_full["date"])
    fl_test_full = fl_full[fl_full["split"] == "test"].dropna(subset=["score"])
    fl_test_by_step = {s: g for s, g in fl_test_full.groupby("step")}
    return fl_test_by_step, fl_test_full


def _load_oos() -> pd.DataFrame:
    oos_path = DATA / "oos_predictions.parquet"
    if not oos_path.exists():
        sys.exit(f"ERROR: {oos_path} not found — run train.py first")
    oos = pd.read_parquet(oos_path).reset_index()
    oos["date"] = pd.to_datetime(oos["date"])
    return oos.set_index(["date", "etf_id"])


def _get_steps(oos: pd.DataFrame) -> list:
    START_YEAR = HEURISTIC_START.year
    all_steps = sorted(oos["step"].unique())
    steps = []
    for s in all_steps:
        test_dates = oos[(oos["step"] == s) & (oos["split"] == "test")].index.get_level_values("date")
        if len(test_dates) > 0 and test_dates.min().year >= START_YEAR:
            steps.append(s)
    return steps


def _compute_ic_gate(oos: pd.DataFrame, all_steps: list,
                     threshold: float, span: int, min_steps: int) -> pd.Series:
    """Compute causal EMA of per-step test IC, shifted by 1 step."""
    test_oos = oos[oos["split"] == "test"].dropna(subset=["score", "label"]).reset_index()
    test_ic_per_step = (
        test_oos.groupby("step")
        .apply(lambda g: g.groupby("date")
               .apply(lambda x: x["score"].corr(x["label"]) if len(x) > 1 else np.nan,
                      include_groups=False).mean(),
               include_groups=False)
        .reindex(all_steps)
    )
    return test_ic_per_step.shift(1).ewm(span=span, min_periods=1).mean()


def _compute_fl_ic_gate(fl_test_by_step, all_steps: list) -> pd.Series:
    if fl_test_by_step is None:
        return pd.Series(dtype=float)
    fl_test_full = pd.concat(fl_test_by_step.values())
    fl_ic_per_step = (
        fl_test_full.reset_index().groupby("step")
        .apply(lambda g: g.groupby("date")
               .apply(lambda x: x["score"].corr(x["label"]) if len(x) > 1 else np.nan,
                      include_groups=False).mean(),
               include_groups=False)
        .reindex(all_steps)
    )
    return fl_ic_per_step.shift(1).ewm(span=FL_IC_GATE_SPAN, min_periods=1).mean()


# ---------------------------------------------------------------------------
#  Portfolio simulation engine
# ---------------------------------------------------------------------------

def compute_fee_equity(port_returns: pd.Series, step_fee_records: list,
                       init_capital: float = 150_000.0) -> tuple:
    """Build equity curve from gross daily-return series and per-step IB fees."""
    if len(port_returns) == 0:
        return pd.Series(dtype=float), 0.0

    fee_drag = pd.Series(1.0, index=port_returns.index)
    for idx, sf in step_fee_records:
        fee_drag.loc[idx.max()] *= (1.0 - sf / init_capital)
    daily_factor = (1.0 + port_returns) * fee_drag
    eq_curve = daily_factor.cumprod()

    cumul_fees = sum(sf * eq_curve.loc[idx.max()] for idx, sf in step_fee_records)
    return eq_curve, cumul_fees


def run_backtest(
    scores_wide: pd.DataFrame,
    daily_returns_wide: pd.DataFrame,
    prev_weights: np.ndarray | None = None,
    sharpe_weights: np.ndarray | None = None,
    monitor_mask: np.ndarray | None = None,
    regime: np.ndarray | None = None,
    prev_regime: int | None = None,
) -> tuple:
    """Portfolio simulation: decide at close J, execute at close J, earn close(J)→close(J+1).

    daily_returns_wide should already be shifted so that row J contains the
    return from close(J) to close(J+1), i.e. daily_ret_panel.shift(-1).
    """
    dates   = scores_wide.index
    n_etfs  = scores_wide.shape[1]
    etf_list = scores_wide.columns.tolist()

    daily_ret = daily_returns_wide.values
    scores = scores_wide.values
    sw = sharpe_weights if sharpe_weights is not None else np.ones(n_etfs)

    # Allocation: score > 0 → on, softmax(score/T) × Sharpe weights (top-N pre-filtered)
    weights = np.zeros_like(scores)
    for i in range(scores.shape[0]):
        row = scores[i]
        on = (~np.isnan(row)) & (row > 0)
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
            weights[i, on] = combined / total

    # Fractional portfolio simulation
    SPREAD_COST = 0.0001
    REBAL_MIN_CHANGE = 0.03

    # Per-ETF IB fee model from TRADING_MAP
    from etf import TRADING_MAP, ib_commission as _ib_commission
    _fee_models = [TRADING_MAP.get(etf, (None, None, None, None, "ams"))[4]
                   for etf in etf_list]

    def _broker_fee(order_amount, etf_idx=None):
        if etf_idx is not None:
            return _ib_commission(order_amount, _fee_models[etf_idx])
        return max(order_amount * 0.0005, 3.0)

    total_fees = 0.0
    safe_ret = np.where(np.isnan(daily_ret), 0.0, daily_ret)
    n_days = len(dates)

    positions = np.zeros(n_etfs)
    cash = INIT_CAPITAL if prev_weights is None else 0.0
    if prev_weights is not None:
        total_val = INIT_CAPITAL
        for j in range(n_etfs):
            positions[j] = prev_weights[j] * total_val
        cash = total_val - positions.sum()

    port_returns = np.zeros(n_days)
    actual_weights = np.zeros((n_days, n_etfs))
    n_rebalances = 0
    if prev_weights is not None:
        prev_topn_set = set(j for j in range(n_etfs) if prev_weights[j] > 0.001)
    else:
        prev_topn_set = set()

    for i in range(n_days):
        total_val = positions.sum() + cash

        # Rebalance when: allocated asset leaves monitor set, or regime changes
        if regime is not None and i == 0:
            regime_changed = (prev_regime is not None and regime[0] != prev_regime)
        else:
            regime_changed = (regime is not None and i > 0 and regime[i] != regime[i - 1])
        if regime_changed:
            topn_changed = True
        elif monitor_mask is not None:
            cur_monitor_set = set(j for j in range(n_etfs) if monitor_mask[i, j])
            allocated_out = prev_topn_set - cur_monitor_set
            topn_changed = len(allocated_out) > 0 or len(prev_topn_set) == 0
        else:
            cur_topn_set = set(j for j in range(n_etfs) if weights[i, j] > 0)
            topn_changed = (cur_topn_set != prev_topn_set)
        if not topn_changed:
            pass
        elif total_val > 0:
            target_w = weights[i]
            target_pos = target_w * total_val
            current_w = positions / total_val if total_val > 0 else np.zeros(n_etfs)
            weight_change = np.abs(target_w - current_w).sum()
            if not np.allclose(target_pos, positions) and weight_change >= REBAL_MIN_CHANGE:
                trade_fees = 0.0
                for j in range(n_etfs):
                    abs_diff = abs(target_pos[j] - positions[j])
                    if abs_diff > 0:
                        trade_fees += _broker_fee(abs_diff, etf_idx=j) + abs_diff * SPREAD_COST
                total_fees += trade_fees
                positions = target_pos.copy()
                cash = total_val - positions.sum() - trade_fees
                n_rebalances += 1
                prev_topn_set = set(j for j in range(n_etfs) if target_pos[j] > 0)

        # Daily P&L
        total_val = positions.sum() + cash
        if total_val > 0:
            actual_weights[i] = positions / total_val
            daily_pnl = (positions * safe_ret[i]).sum()
            positions = positions * (1 + safe_ret[i])
            port_returns[i] = daily_pnl / total_val

    ret_series = pd.Series(port_returns, index=dates, name="port_return")
    weights_df = pd.DataFrame(actual_weights, index=dates, columns=etf_list)
    final_val = positions.sum() + cash
    last_regime = int(regime[-1]) if regime is not None else None
    return ret_series, weights_df, total_fees, n_rebalances, final_val, last_regime


# ---------------------------------------------------------------------------
#  Tail extension: carry last allocation from last OOS date to last data date
# ---------------------------------------------------------------------------

def _extend_to_last_date(all_test_returns, all_test_weights, carry_weights,
                         carry_regime, daily_ret_panel, vix_s, vix_ema100,
                         step_fee_records,
                         regime_tracker=None,
                         test_ic_ema_lagged=None,
                         fl_test_ic_ema_lagged=None,
                         fl_test_by_step=None,
                         last_step=None):
    """Extend portfolio simulation beyond last WF step to last available data date.

    Loads last-step models from outputs/robot/ and runs inference per day
    with full regime logic (SM/FL/heuristic/cash). Falls back to carry-forward
    if model artifacts are not available.
    """
    if not all_test_returns or carry_weights is None:
        return carry_weights, carry_regime

    last_oos_date = all_test_returns[-1].index[-1]
    last_data_date = daily_ret_panel.dropna(how="all").index[-1]

    if last_data_date <= last_oos_date:
        return carry_weights, carry_regime

    from etf import UNIVERSE as _UNIVERSE
    etf_list = [e.bourso for e in _UNIVERSE]

    tail_dates = daily_ret_panel.index[
        (daily_ret_panel.index > last_oos_date) &
        (daily_ret_panel.index <= last_data_date)
    ]
    if len(tail_dates) == 0:
        return carry_weights, carry_regime

    # --- Per-day regime + scores (QQQ + BAA/VIX cash-out) ---
    tail_scores = pd.DataFrame(np.nan, index=tail_dates, columns=etf_list)
    step_regime = np.ones(len(tail_dates), dtype=int)  # 1 = heuristic

    # Cash-out: VIX spike (diff1 > 3, VIX >= 20) OR credit spread EMA50 > EMA200
    baa_path = ROOT / "data" / "fred_baa_spread.parquet"
    baa_ema50_t, baa_ema200_t = pd.Series(dtype=float), pd.Series(dtype=float)
    if baa_path.exists():
        baa_raw = pd.read_parquet(baa_path).iloc[:, 0]
        baa_reindexed = baa_raw.reindex(tail_dates, method="ffill")
        # Compute EMAs on full history, then reindex to tail dates
        baa_ema50_full = baa_raw.ewm(span=50).mean()
        baa_ema200_full = baa_raw.ewm(span=200).mean()
        baa_ema50_t = baa_ema50_full.reindex(tail_dates, method="ffill")
        baa_ema200_t = baa_ema200_full.reindex(tail_dates, method="ffill")

    spike_active = False
    if not vix_s.empty:
        vix_close = vix_s.reindex(tail_dates, method="ffill")
        vix_diff1 = vix_close.diff(1).fillna(0)
    else:
        vix_diff1 = pd.Series(0, index=tail_dates)
        vix_close = vix_diff1

    for i, d in enumerate(tail_dates):
        vix_spike = vix_diff1.iloc[i] > 5.0
        credit_stress = (not baa_ema50_t.empty and pd.notna(baa_ema50_t.iloc[i])
                and baa_ema50_t.iloc[i] / baa_ema200_t.iloc[i] > 1.07
                and baa_ema50_t.iloc[i] > 2.0)
        if credit_stress or vix_spike:
            spike_active = True
        elif (not baa_ema200_t.empty and pd.notna(baa_ema200_t.iloc[i])
                and baa_ema50_t.iloc[i] / baa_ema200_t.iloc[i] < 1.0):
            spike_active = False

        if spike_active:
            tail_scores.iloc[i] = np.nan
            step_regime[i] = 3  # cash
            continue

        # Heuristic: 100% QQQ
        tail_scores.loc[d, "QQQ"] = 1.0

    # --- Run portfolio simulation ---
    # Shift returns: ret at date J = return from close(J) to close(J+1)
    next_day_ret = daily_ret_panel.shift(-1)
    tail_dr = next_day_ret.reindex(index=tail_dates, columns=etf_list).fillna(0)
    prev_w = np.zeros(len(etf_list))
    for j, etf in enumerate(etf_list):
        if etf in carry_weights:
            prev_w[j] = carry_weights[etf]

    etf_sharpe_w = np.ones(len(etf_list))
    monitor_mask = np.zeros((len(tail_dates), len(etf_list)), dtype=bool)
    for i in range(len(tail_dates)):
        row = tail_scores.iloc[i].values
        pos = np.where((~np.isnan(row)) & (row > 0))[0]
        monitor_mask[i, pos] = True

    result = run_backtest(tail_scores, tail_dr,
                          prev_weights=prev_w,
                          sharpe_weights=etf_sharpe_w,
                          monitor_mask=monitor_mask,
                          regime=step_regime,
                          prev_regime=carry_regime)
    tail_returns, tail_weights = result[0], result[1]
    tail_fees = result[2]

    if len(tail_returns) > 0:
        all_test_returns.append(tail_returns)
        all_test_weights.append(tail_weights)
        step_fee_records.append((tail_returns.index, tail_fees))

        new_carry_weights = dict(zip(tail_weights.columns,
                                     tail_weights.iloc[-1].values))
        new_carry_regime = int(result[5]) if result[5] is not None else carry_regime

        if regime_tracker is not None:
            for i, d in enumerate(tail_dates):
                r = step_regime[i]
                if r == 0:
                    regime_tracker.get("model", []).append(pd.DatetimeIndex([d]))
                elif r == 2:
                    regime_tracker.get("fl", []).append(pd.DatetimeIndex([d]))
                elif r == 3:
                    regime_tracker.get("cash", []).append(pd.DatetimeIndex([d]))

        return new_carry_weights, new_carry_regime

    return carry_weights, carry_regime


def _infer_day(features_panel, date, models, feature_cols, etf_list):
    """Run model inference for a single date. Returns pd.Series or None."""
    try:
        if date not in features_panel.index.get_level_values("date"):
            return None
        day_features = features_panel.loc[date]
        available = [c for c in feature_cols if c in day_features.columns]
        if len(available) != len(feature_cols):
            return None
        X = day_features[available].astype(np.float32).replace(
            [np.inf, -np.inf], np.nan)
        scores_stack = [m.predict(X.values) for m in models]
        mean_scores = np.mean(scores_stack, axis=0)
        idx = day_features.index
        if idx.nlevels > 1:
            idx = idx.droplevel("date") if "date" in idx.names else idx
        return pd.Series(mean_scores, index=idx)
    except Exception:
        return None


# ---------------------------------------------------------------------------
#  Shared step-loop: processes one WF step (used by both equity & robustness)
# ---------------------------------------------------------------------------

def _process_step(oos, step, daily_ret_panel, vix_s, vix_ema100,
                  test_ic_ema_lagged, fl_test_ic_ema_lagged,
                  fl_test_by_step, carry_weights, carry_regime,
                  dropped_set=None):
    """Process a single walk-forward step. Returns dict with all outputs or None to skip."""
    val_data  = oos[(oos["step"] == step) & (oos["split"] == "val")]
    test_data = oos[(oos["step"] == step) & (oos["split"] == "test")]

    if len(val_data) < 50 or len(test_data) == 0:
        return None

    # Pivot test scores
    val_scores = _pivot_step(val_data)
    val_end_date = val_scores.index[-1]
    dr_hist = daily_ret_panel[val_scores.columns].loc[:val_end_date]
    etf_sharpe = np.ones(len(val_scores.columns))  # uniform (heuristic drives allocation)

    test_scores = _pivot_step(test_data)
    etf_list = test_scores.columns.tolist()
    n_cols = len(etf_list)

    # Random ETF drop (robustness only)
    if dropped_set:
        test_scores[list(dropped_set)] = np.nan

    # Remap carry_weights
    prev_w = None
    if carry_weights is not None:
        prev_w = np.zeros(n_cols)
        for i, etf in enumerate(etf_list):
            if etf in carry_weights:
                prev_w[i] = carry_weights[etf]

    # Heuristic: 100% QQQ
    heur_row = np.array([1.0 if c == "QQQ" else np.nan for c in etf_list])
    heur_monitor = np.array([c == "QQQ" for c in etf_list])

    # Per-day regime: heuristic (QQQ) everywhere
    final_scores = test_scores.copy()
    step_monitor_mask = np.zeros((len(test_scores), n_cols), dtype=bool)
    step_regime = np.ones(len(test_scores), dtype=int)  # 1 = heuristic
    for i in range(len(test_scores)):
        final_scores.iloc[i] = heur_row
        step_monitor_mask[i] = heur_monitor

    # Cash-out: VIX spike (diff1 > 3, VIX >= 20) OR credit spread EMA50 > EMA200
    baa_path = ROOT / "data" / "fred_baa_spread.parquet"
    baa_reindexed = pd.Series(dtype=float)
    baa_ema50_s, baa_ema200_s = pd.Series(dtype=float), pd.Series(dtype=float)
    if baa_path.exists():
        baa_raw = pd.read_parquet(baa_path).iloc[:, 0]
        baa_reindexed = baa_raw.reindex(final_scores.index, method="ffill")
        # Compute EMAs on full history, then reindex to score dates
        baa_ema50_full = baa_raw.ewm(span=50).mean()
        baa_ema200_full = baa_raw.ewm(span=200).mean()
        baa_ema50_s = baa_ema50_full.reindex(final_scores.index, method="ffill")
        baa_ema200_s = baa_ema200_full.reindex(final_scores.index, method="ffill")

    if not vix_s.empty:
        vix_close_test = vix_s.reindex(final_scores.index, method="ffill")
        vix_diff1 = vix_close_test.diff(1).fillna(0)
        spike_active = False
        for i in range(len(final_scores)):
            vix_spike = vix_diff1.iloc[i] > 5.0
            credit_stress = (not baa_ema50_s.empty and pd.notna(baa_ema50_s.iloc[i])
                    and baa_ema50_s.iloc[i] / baa_ema200_s.iloc[i] > 1.07
                    and baa_ema50_s.iloc[i] > 2.0)
            if credit_stress or vix_spike:
                spike_active = True
            elif (not baa_ema200_s.empty and pd.notna(baa_ema200_s.iloc[i])
                    and baa_ema50_s.iloc[i] / baa_ema200_s.iloc[i] < 1.0):
                spike_active = False
            if spike_active:
                final_scores.iloc[i] = np.nan
                step_monitor_mask[i] = False
                step_regime[i] = 3

    # Run portfolio simulation
    # Shift returns: ret at date J = return from close(J) to close(J+1)
    next_day_ret = daily_ret_panel.shift(-1)
    test_dr = next_day_ret.reindex(index=final_scores.index,
                                   columns=final_scores.columns).fillna(0)
    result = run_backtest(final_scores, test_dr,
                          prev_weights=prev_w,
                          sharpe_weights=etf_sharpe,
                          monitor_mask=step_monitor_mask,
                          regime=step_regime,
                          prev_regime=carry_regime)
    test_returns, test_weights = result[0], result[1]
    step_fees, step_rebals = result[2], result[3]
    new_carry_regime = result[5]
    new_carry_weights = dict(zip(test_weights.columns, test_weights.iloc[-1].values))

    return {
        "test_returns": test_returns,
        "test_weights": test_weights,
        "step_fees": step_fees,
        "step_rebals": step_rebals,
        "carry_weights": new_carry_weights,
        "carry_regime": new_carry_regime,
        "step_regime": step_regime,
        "gate_open": False,
        "gate_ema": np.nan,
        "use_fl_model": False,
        "final_scores": final_scores,
        "model_scores_index": final_scores.index,
    }


# ---------------------------------------------------------------------------
#  VIX chart overlay (shared by equity & robustness charts)
# ---------------------------------------------------------------------------

def _draw_regime_table(ax_tbl):
    """Draw the VIX regime allocation table on the given axes."""
    ax_tbl.set_xlim(0, 1)
    ax_tbl.set_ylim(0, 1)
    ax_tbl.axis("off")
    regime_rows = [
        ("Condition",          "Allocation",                              "#333333"),
        ("Default",            "Heuristic top-5 Sharpe-weighted 2y",      "#2ca02c"),
        ("Spike Δ5d > 6",     "Cash",                                    "#d62728"),
    ]
    n_rows = len(regime_rows)
    row_h = 1.0 / n_rows
    col_x = [0.02, 0.35]
    for r, (cond, alloc, color) in enumerate(regime_rows):
        ry = 1.0 - (r + 0.5) * row_h
        is_hdr = (r == 0)
        fw = "bold" if is_hdr else "normal"
        fs = 10 if is_hdr else 9.5
        txt_color = "#333333" if is_hdr else "#555555"
        ax_tbl.text(col_x[0], ry, cond, fontsize=fs, fontweight=fw, va="center",
                    transform=ax_tbl.transAxes, color=txt_color)
        if is_hdr:
            ax_tbl.text(col_x[1], ry, alloc, fontsize=fs, fontweight=fw, va="center",
                        transform=ax_tbl.transAxes, color=txt_color)
        else:
            ax_tbl.text(col_x[1], ry, "■ ", fontsize=fs + 2, va="center",
                        transform=ax_tbl.transAxes, color=color)
            ax_tbl.text(col_x[1] + 0.03, ry, alloc, fontsize=fs, fontweight="bold",
                        va="center", transform=ax_tbl.transAxes, color=color)
        if r == 0:
            ax_tbl.plot([0.01, 0.99], [1.0 - row_h, 1.0 - row_h], color="#999999",
                        lw=0.8, transform=ax_tbl.transAxes)
    ax_tbl.plot([0.01, 0.99], [1.0, 1.0], color="#999999", lw=0.8, transform=ax_tbl.transAxes)
    ax_tbl.plot([0.01, 0.99], [0.0, 0.0], color="#999999", lw=0.8, transform=ax_tbl.transAxes)


def _draw_vix_overlay(ax, eq_index, vix_s,
                      model_active_dates=None, fl_active_dates=None,
                      cash_dates=None):
    """Draw VIX colored overlay on a twin axis."""
    vix_raw = vix_s.reindex(eq_index, method="ffill").dropna()
    if vix_raw.empty:
        return
    ax1b = ax.twinx()
    vix_ema100_chart = vix_raw.ewm(span=100).mean()
    vix_ema300_chart = vix_raw.ewm(span=300).mean()
    vix_5d_chart = vix_raw.diff(1)

    # Spike mask
    _spike_mask = pd.Series(False, index=vix_raw.index)
    for _sd in vix_5d_chart[vix_5d_chart > VIX_SPIKE_MIN].index:
        _pos = vix_raw.index.get_loc(_sd)
        for _off in range(VIX_SPIKE_CASH_DAYS):
            if _pos + _off < len(vix_raw):
                _spike_mask.iloc[_pos + _off] = True

    # Build color per date
    _colors = pd.Series("#2ca02c", index=vix_raw.index)  # green default

    if model_active_dates is not None and len(model_active_dates) > 0:
        _model_mask = pd.Series(False, index=vix_raw.index)
        _intersect = vix_raw.index.intersection(model_active_dates)
        _model_mask.loc[_intersect] = True
        _colors[_model_mask.values] = "#ff7f0e"  # orange (SM)
    else:
        _calm_cond = vix_ema100_chart < VIX_CALM_THRESHOLD
        if VIX_CALM_COND_EMA100_SUP_EMA300:
            _calm_cond = _calm_cond & (vix_ema100_chart < vix_ema300_chart)
        _model_mask = ~_calm_cond.reindex(vix_raw.index).fillna(False)
        _colors[_model_mask.values] = "#ff7f0e"

    if fl_active_dates is not None and len(fl_active_dates) > 0:
        _fl_mask = pd.Series(False, index=vix_raw.index)
        _fl_intersect = vix_raw.index.intersection(fl_active_dates)
        _fl_mask.loc[_fl_intersect] = True
        _colors[_fl_mask.values] = "#1f77b4"  # blue (FL)

    if cash_dates is not None and len(cash_dates) > 0:
        _cash_mask = pd.Series(False, index=vix_raw.index)
        _cash_intersect = vix_raw.index.intersection(cash_dates)
        _cash_mask.loc[_cash_intersect] = True
        _colors[_cash_mask.values] = "#d62728"  # red (cash)

    _colors[_spike_mask.values] = "#d62728"  # red (spike)

    # Draw colored segments
    prev_c = _colors.iloc[0]
    seg_start = 0
    for _i in range(1, len(vix_raw)):
        if _colors.iloc[_i] != prev_c or _i == len(vix_raw) - 1:
            seg = slice(seg_start, _i + 1)
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


# ---------------------------------------------------------------------------
#  Equity chart
# ---------------------------------------------------------------------------

def _save_equity_png(port_returns, eq_curve, weights_df, out_dir,
                     fin=None, fname="backtest_equity.jpg",
                     model_active_dates=None, fl_active_dates=None,
                     cash_dates=None):
    ann_ret = eq_curve.iloc[-1] ** (252.0 / max(len(port_returns), 1)) - 1
    ann_vol = port_returns.std() * np.sqrt(252)
    sh      = sharpe(port_returns)
    max_dd  = (eq_curve / eq_curve.cummax() - 1).min()

    from etf import UNIVERSE as _UNIVERSE
    SHORT_NAMES = {e.bourso: _short_name(e.name) for e in _UNIVERSE}
    BOLD_TICKERS = {"IVV", "GLD", "IEO", "QQQ", "RING"}

    fig1 = plt.figure(figsize=(24, 22))
    gs = fig1.add_gridspec(5, 2, width_ratios=[3, 1],
                           height_ratios=[3, 0.35, 1.2, 1.2, 1.2],
                           hspace=0.06, wspace=0.02,
                           top=0.98, bottom=0.04, left=0.05, right=0.98)
    ax1 = fig1.add_subplot(gs[0, 0])
    ax_tbl = fig1.add_subplot(gs[1, 0])     # regime table
    ax3 = fig1.add_subplot(gs[2, 0], sharex=ax1)  # VIX + MAs
    ax4 = fig1.add_subplot(gs[3, 0], sharex=ax1)  # Credit spread
    ax2 = fig1.add_subplot(gs[4, 0], sharex=ax1)  # Allocation
    ax_leg = fig1.add_subplot(gs[:, 1])
    ax_leg.axis("off")

    # Trades per month
    if len(weights_df) > 0:
        w = weights_df.reindex(eq_curve.index, method="ffill").fillna(0)
        active = (w > 0.01).astype(int)
        trades = active.diff().abs().sum(axis=1)
        n_months = max(len(port_returns) / 21, 1)
        trades_per_month = trades.sum() / n_months
        mean_alloc = w.mean()
    else:
        trades_per_month = 0
        mean_alloc = pd.Series(dtype=float)
    alloc_by_short = {SHORT_NAMES.get(t, t): float(mean_alloc.get(t, 0.0))
                      for t in mean_alloc.index}

    # Right column header
    def _eur(x):
        return f"{x:,.0f}".replace(",", " ") + " €"

    RED = "#d62728"
    hdr = [
        ("Ann. return", f"{ann_ret:+.1%}", RED, True),
        ("Vol", f"{ann_vol:.1%}", RED, False),
        ("MaxDD", f"{max_dd:.1%}", RED, False),
        ("Trades", f"{trades_per_month:.1f}/mo", RED, False),
    ]
    if fin is not None:
        hdr += [
            ("__sep__", "", "", False),
            ("Initial capital", _eur(fin['init']), RED, False),
            ("IB trading fees", _eur(-fin['fees']), RED, False),
            ("Final value", _eur(fin['final']), RED, True),
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

    _draw_regime_table(ax_tbl)

    # Portfolio curve
    ax1.plot(eq_curve.index, eq_curve.values, lw=3.5, color="#d62728", zorder=10)
    port_sh = sharpe(port_returns)
    port_dd = (eq_curve / eq_curve.cummax() - 1).min()

    # ETF curves
    etf_curves = []
    for etf in _UNIVERSE:
        ticker = etf.bourso
        color = ETF_COLOR_MAP.get(ticker, "#7f7f7f")
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

    # Sorted list by Sharpe
    all_items = [(eq_curve.iloc[-1], port_sh, port_dd, "Portfolio", "#d62728", True)]
    all_items += [(v, sh, dd, s, c, False) for v, sh, dd, s, c in etf_curves]
    sorted_by_sharpe = sorted(all_items, key=lambda x: -x[1])

    ax1.set_yscale("log")
    ax1.set_ylabel("Equity (log scale, base 1)")
    ax1.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:.1f}x"))
    ax1.grid(True, alpha=0.3, which="both")
    ax1.set_facecolor("#f8f8f8")

    # Panel 2: Allocation
    if len(weights_df) > 0:
        w = weights_df.reindex(eq_curve.index, method="ffill").fillna(0)
        cash_w = (1 - w.sum(axis=1)).clip(0, 1)
        universe_order = [e.bourso for e in _UNIVERSE]
        allocated = [t for t in universe_order if t in w.columns and w[t].mean() > 0.001]
        w_sorted = w[allocated]
        col_names = {e: SHORT_NAMES.get(e, e) for e in w_sorted.columns}
        w_named = w_sorted.rename(columns=col_names)
        plot_data = w_named.copy()
        plot_data["Cash"] = cash_w
        color_map = {SHORT_NAMES.get(t, t): c for t, c in ETF_COLOR_MAP.items()}
        color_map["Cash"] = "#e8e8e8"
        colors = [color_map.get(c, "#7f7f7f") for c in plot_data.columns]
        ax2.stackplot(plot_data.index, plot_data.values.T,
                      labels=plot_data.columns, colors=colors, alpha=0.85)
        ax2.set_ylabel("Allocation")
        ax2.set_ylim(0, 1)
        ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax2.set_facecolor("#f8f8f8")
        ax2.grid(True, alpha=0.3)

    # Panel 3: VIX + MAs + spike cash zones
    vix_path_chart = DATA / "fred_vix.parquet"
    if vix_path_chart.exists():
        vix_raw = pd.read_parquet(vix_path_chart).iloc[:, 0]
        vix_chart = vix_raw.reindex(eq_curve.index, method="ffill").dropna()
        if not vix_chart.empty:
            vix_ema10_chart = vix_chart.ewm(span=10).mean()
            vix_ema20_chart = vix_chart.ewm(span=20).mean()
            vix_ema50_chart = vix_chart.ewm(span=50).mean()
            vix_ema200_chart = vix_chart.ewm(span=200).mean()
            ax3.plot(vix_chart.index, vix_chart.values, color="#000000",
                     alpha=0.7, lw=1.0, label="VIX")
            ax3.plot(vix_ema10_chart.index, vix_ema10_chart.values, color="#d62728",
                     lw=1.0, label="EMA10")
            ax3.plot(vix_ema20_chart.index, vix_ema20_chart.values, color="#2ca02c",
                     lw=1.2, label="EMA20")
            ax3.plot(vix_ema50_chart.index, vix_ema50_chart.values, color="#ff7f0e",
                     lw=1.2, label="EMA50")
            ax3.plot(vix_ema200_chart.index, vix_ema200_chart.values, color="#1565c0",
                     lw=1.2, label="EMA200")
            # Shade SM active periods (orange)
            if model_active_dates is not None and len(model_active_dates) > 0:
                _sm_mask = pd.Series(False, index=vix_chart.index)
                _sm_inter = vix_chart.index.intersection(model_active_dates)
                _sm_mask.loc[_sm_inter] = True
                ax3.fill_between(vix_chart.index, 0, vix_chart.values,
                                 where=_sm_mask.values, color="#ff7f0e",
                                 alpha=0.15, label="SM active")
            # Shade FL active periods (blue)
            if fl_active_dates is not None and len(fl_active_dates) > 0:
                _fl_mask = pd.Series(False, index=vix_chart.index)
                _fl_inter = vix_chart.index.intersection(fl_active_dates)
                _fl_mask.loc[_fl_inter] = True
                ax3.fill_between(vix_chart.index, 0, vix_chart.values,
                                 where=_fl_mask.values, color="#1f77b4",
                                 alpha=0.15, label="FL active")
            # Shade spike cash periods (red)
            if cash_dates is not None and len(cash_dates) > 0:
                _cash_mask = pd.Series(False, index=vix_chart.index)
                _cash_inter = vix_chart.index.intersection(cash_dates)
                _cash_mask.loc[_cash_inter] = True
                ax3.fill_between(vix_chart.index, 0, vix_chart.values,
                                 where=_cash_mask.values, color="#d62728",
                                 alpha=0.15, label="Cash")
            ax3.set_ylabel("VIX")
            ax3.set_ylim(0, 85)  # VIX historical max ~82
            ax3.legend(loc="upper left", fontsize=9)
            ax3.grid(True, alpha=0.3)
            ax3.set_facecolor("#f8f8f8")

    # Panel 4: Credit spread (BAA-10Y)
    baa_path = DATA / "fred_baa_spread.parquet"
    if baa_path.exists():
        baa_raw = pd.read_parquet(baa_path).iloc[:, 0]
        baa_chart = baa_raw.reindex(eq_curve.index, method="ffill").dropna()
        if not baa_chart.empty:
            baa_ema10 = baa_chart.ewm(span=10).mean()
            baa_ema50 = baa_chart.ewm(span=50).mean()
            baa_ema200 = baa_chart.ewm(span=200).mean()
            ax4.plot(baa_chart.index, baa_chart.values, color="#9467bd",
                     lw=1.0, alpha=0.5, label="BAA-10Y")
            ax4.plot(baa_ema10.index, baa_ema10.values, color="#2ca02c",
                     lw=1.2, label="EMA10")
            ax4.plot(baa_ema50.index, baa_ema50.values, color="#ff7f0e",
                     lw=1.5, label="EMA50")
            ax4.plot(baa_ema200.index, baa_ema200.values, color="#1565c0",
                     lw=1.5, label="EMA200")
            ax4.set_ylim(0, 6.5)
            ax4.set_ylabel("Credit spread (BAA-10Y)")
            ax4.legend(loc="upper left", fontsize=9)
            ax4.grid(True, alpha=0.3)
            ax4.set_facecolor("#f8f8f8")

    # Right column: ETF list sorted by Sharpe
    n_items = len(sorted_by_sharpe)
    y_start = hdr_bottom - 0.030
    avail = y_start + 0.06
    y_step = min(0.030, avail / max(n_items, 1))
    marker_fs = min(26, max(12, y_step * 850))
    marker_min = 5.0
    max_a = max(alloc_by_short.values(), default=0.0)
    for i, (val, sh, dd, short, color, is_port) in enumerate(sorted_by_sharpe):
        y = y_start - i * y_step
        fw = "bold" if is_port else "normal"
        fs = (11 if is_port else 10) if y_step > 0.024 else (10 if is_port else 9)
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

    fig1.savefig(out_dir / fname, dpi=100, bbox_inches="tight", format="jpeg",
                 pil_kwargs={"quality": 60, "optimize": True})
    plt.close(fig1)


# ---------------------------------------------------------------------------
#  Equity backtest
# ---------------------------------------------------------------------------

def run_equity():
    """Equity backtest → global equity chart."""
    print("Loading OOS predictions...")
    oos = _load_oos()

    fl_test_by_step, fl_test_full = _load_fl_predictions()
    if fl_test_by_step is not None:
        print(f"Follow Leads predictions loaded: "
              f"{sum(len(g) for g in fl_test_by_step.values())} test rows over "
              f"{len(fl_test_by_step)} steps")

    if "split" not in oos.columns:
        sys.exit("ERROR: oos_predictions.parquet has no 'split' column — retrain with updated train.py")

    vix_s, vix_ema100, vix_ema300 = _load_vix()
    daily_ret_panel = _load_daily_returns()
    print(f"Loaded daily returns for {len(daily_ret_panel.columns)} ETFs")

    oos = _prepend_heuristic_steps(oos, daily_ret_panel)
    all_steps = sorted(oos["step"].unique())
    steps = _get_steps(oos)
    print(f"Walk-forward steps: {len(steps)}  "
          f"val rows: {(oos['split']=='val').sum()}  "
          f"test rows: {(oos['split']=='test').sum()}\n")

    # IC gates
    test_ic_ema_lagged = _compute_ic_gate(oos, all_steps,
                                           IC_GATE_THRESHOLD, IC_GATE_SPAN, IC_GATE_MIN_STEPS)
    fl_test_ic_ema_lagged = _compute_fl_ic_gate(fl_test_by_step, all_steps)

    n_open = sum(1 for s in steps
                 if s >= IC_GATE_MIN_STEPS
                 and pd.notna(test_ic_ema_lagged.get(s))
                 and test_ic_ema_lagged.get(s) > IC_GATE_THRESHOLD)
    print(f"IC gate: threshold {IC_GATE_THRESHOLD:+.3f}  span {IC_GATE_SPAN}  "
          f"min steps {IC_GATE_MIN_STEPS}")
    print(f"  → model deploys on {n_open}/{len(steps)} steps "
          f"({n_open/max(len(steps),1):.0%}); other steps stay in calm mode.")
    print(f"Calm allocator: {CALM_ALLOCATOR}")

    if fl_test_by_step is not None:
        n_fl_open = sum(1 for s in steps
                        if s >= FL_IC_GATE_MIN_STEPS
                        and pd.notna(fl_test_ic_ema_lagged.get(s))
                        and fl_test_ic_ema_lagged.get(s) > FL_IC_GATE_THRESHOLD)
        print(f"FL IC gate: threshold {FL_IC_GATE_THRESHOLD:+.3f}  span {FL_IC_GATE_SPAN}  "
              f"min steps {FL_IC_GATE_MIN_STEPS}")
        print(f"  → FL model used on {n_fl_open}/{len(steps)} calm steps "
              f"({n_fl_open/max(len(steps),1):.0%})")

    all_test_returns = []
    all_test_weights = []
    all_params_rows  = []
    model_active_days = []
    fl_active_days    = []
    cash_days_list    = []
    carry_weights    = None
    carry_regime     = None
    step_fee_records = []
    cumul_rebals     = 0

    print(f"\nTemperature: {TEMPERATURE:.2f}")
    print(f"\n{'Step':>4}  {'Test period':>24}  {'Sharpe':>7}")
    print("-" * 45)

    for step in steps:
        result = _process_step(oos, step, daily_ret_panel, vix_s, vix_ema100,
                               test_ic_ema_lagged, fl_test_ic_ema_lagged,
                               fl_test_by_step, carry_weights, carry_regime)
        if result is None:
            val_data = oos[(oos["step"] == step) & (oos["split"] == "val")]
            test_data = oos[(oos["step"] == step) & (oos["split"] == "test")]
            print(f"{step:4d}  SKIP (val={len(val_data)} rows, test={len(test_data)} rows)")
            continue

        carry_weights = result["carry_weights"]
        carry_regime = result["carry_regime"]
        test_returns = result["test_returns"]
        test_sh = sharpe(test_returns)

        test_data = oos[(oos["step"] == step) & (oos["split"] == "test")]
        test_dates = test_data.index.get_level_values("date")
        print(f"{step:4d}  "
              f"[{test_dates.min().date()} → {test_dates.max().date()}]  {test_sh:7.3f}",
              flush=True)

        all_test_returns.append(test_returns)
        all_test_weights.append(result["test_weights"])
        step_fee_records.append((test_returns.index, result["step_fees"]))
        cumul_rebals += result["step_rebals"]
        all_params_rows.append({"step": step, "temperature": TEMPERATURE})

        # Track active days for chart colouring
        for i in range(len(result["step_regime"])):
            idx = result["model_scores_index"]
            if result["step_regime"][i] == 0:
                model_active_days.append(pd.DatetimeIndex([idx[i]]))
            elif result["step_regime"][i] == 2:
                fl_active_days.append(pd.DatetimeIndex([idx[i]]))
            elif result["step_regime"][i] == 3:
                cash_days_list.append(pd.DatetimeIndex([idx[i]]))

        # Live progress
        tmp_returns = pd.concat(all_test_returns).sort_index()
        tmp_eq = (1 + tmp_returns).cumprod()
        tmp_sharpe = sharpe(tmp_returns)
        tmp_dd = (tmp_eq / tmp_eq.cummax() - 1).min()
        gate_ema = result["gate_ema"]
        gate_str = (f"gate={'MODEL' if result['gate_open'] else 'CALM '} "
                    f"(ema={gate_ema:+.3f})" if pd.notna(gate_ema)
                    else f"gate=CALM  (ema=  n/a)")
        turb_src = "FL" if result["use_fl_model"] else "heur"
        print(f"       cumul: {tmp_eq.iloc[-1]-1:+.1%}  sharpe={tmp_sharpe:.2f}  dd={tmp_dd:.1%}  "
              f"{gate_str}  turb_fb={turb_src}", flush=True)

    if not all_test_returns:
        sys.exit("No test returns produced — check OOS predictions")

    # Extend to last available data date (carry last allocation forward)
    regime_tracker = {"heuristic": [], "model": model_active_days,
                      "fl": fl_active_days, "cash": cash_days_list}
    carry_weights, carry_regime = _extend_to_last_date(
        all_test_returns, all_test_weights, carry_weights, carry_regime,
        daily_ret_panel, vix_s, vix_ema100, step_fee_records,
        regime_tracker=regime_tracker)
    last_oos = all_test_returns[-2].index[-1] if len(all_test_returns) > 1 else None
    tail_end = all_test_returns[-1].index[-1]
    if last_oos and tail_end > last_oos:
        print(f"\n  Tail extension: {last_oos.date()} -> {tail_end.date()} "
              f"({len(all_test_returns[-1])} days, carry-forward)")

    # Final stats
    port_returns = pd.concat(all_test_returns).sort_index()
    # TEMP zoom disabled — full backtest
    init_capital = INIT_CAPITAL

    # Compute equity with original dates (fee records reference these)
    eq_curve, cumul_fees = compute_fee_equity(port_returns, step_fee_records, init_capital)

    # Shift dates to realization: equity[D] = value at close(D), aligns with benchmarks
    _all_days = daily_ret_panel.dropna(how='all').index.sort_values()
    def _next_bday(d):
        pos = _all_days.searchsorted(d, side='right')
        return _all_days[pos] if pos < len(_all_days) else d + pd.Timedelta(days=1)
    _shifted = pd.DatetimeIndex([_next_bday(d) for d in port_returns.index])
    port_returns.index = _shifted
    eq_curve.index = _shifted

    n_years      = max(len(port_returns), 1) / 252
    ann_vol      = port_returns.std() * np.sqrt(252)
    final_sharpe = sharpe(port_returns)

    total_ret    = eq_curve.iloc[-1] - 1
    ann_ret      = (1 + total_ret) ** (1 / n_years) - 1
    max_dd       = (eq_curve / eq_curve.cummax() - 1).min()
    final_portfolio = eq_curve.iloc[-1] * init_capital

    print("\n" + "=" * 75)
    print("=== Final OOS backtest ===")
    print(f"  Period:       {port_returns.index[0].date()} → {port_returns.index[-1].date()}")
    print(f"  Init capital: {init_capital:,.0f}€")
    print(f"  Ann. vol:     {ann_vol:.1%}")
    print(f"  Sharpe:       {final_sharpe:.3f}")
    print(f"  Max drawdown: {max_dd:.1%}")
    print(f"  Total return: {total_ret:.1%}")
    print(f"  Ann. return:  {ann_ret:.1%}")
    print(f"  Rebalances:   {cumul_rebals} ({cumul_rebals/n_years:.0f}/an)")
    print(f"  Trading fees: {cumul_fees:,.0f}€ ({cumul_fees/n_years:,.0f}€/an)")
    print(f"  Final value:  {final_portfolio:,.0f}€")

    # Save results
    port_returns.to_frame().to_parquet(DATA / "backtest_results.parquet")
    pd.DataFrame(all_params_rows).to_csv(OUTPUTS / "best_params.csv", index=False)
    pd.DataFrame(all_params_rows).to_csv(OUTPUTS / "backtest_steps.csv", index=False)
    eq_df = eq_curve.reset_index()
    eq_df.columns = ["date", "equity"]
    eq_df.to_csv(OUTPUTS / "backtest_equity.csv", index=False)

    # Equity chart
    all_weights = pd.concat(all_test_weights).sort_index()
    all_weights.index = pd.DatetimeIndex([_next_bday(d) for d in all_weights.index])
    all_weights.to_parquet(OUTPUTS / "backtest_weights.parquet")
    fin = {"init": init_capital, "fees": cumul_fees, "final": final_portfolio}

    full_model_dates = pd.DatetimeIndex(
        pd.concat([pd.Series(idx) for idx in model_active_days]).values
        if model_active_days else np.array([], dtype="datetime64[ns]"))
    full_fl_dates = pd.DatetimeIndex(
        pd.concat([pd.Series(idx) for idx in fl_active_days]).values
        if fl_active_days else np.array([], dtype="datetime64[ns]"))
    full_cash_dates = pd.DatetimeIndex(
        pd.concat([pd.Series(idx) for idx in cash_days_list]).values
        if cash_days_list else np.array([], dtype="datetime64[ns]"))

    _save_equity_png(port_returns, eq_curve, all_weights, OUTPUTS,
                     fin=fin,
                     model_active_dates=full_model_dates,
                     fl_active_dates=full_fl_dates,
                     cash_dates=full_cash_dates)

    # Save regime dates for dashboard
    import json as _json
    regime_dates = {
        "model": [str(d.date()) for d in full_model_dates],
        "fl": [str(d.date()) for d in full_fl_dates],
        "cash": [str(d.date()) for d in full_cash_dates],
    }
    with open(OUTPUTS / "backtest_regime_dates.json", "w") as _f:
        _json.dump(regime_dates, _f)

    print(f"\nSaved → {DATA.relative_to(ROOT)}/backtest_results.parquet")
    print(f"Saved → {OUTPUTS.relative_to(ROOT)}/best_params.csv")
    print(f"Saved → {OUTPUTS.relative_to(ROOT)}/backtest_steps.csv")
    print(f"Saved → {OUTPUTS.relative_to(ROOT)}/backtest_equity.csv")
    print(f"Saved → {OUTPUTS.relative_to(ROOT)}/backtest_equity.jpg")


# ---------------------------------------------------------------------------
#  Robustness backtest
# ---------------------------------------------------------------------------

def _rob_sharpe(returns: pd.Series) -> float:
    if len(returns) < 10 or returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(252))


def _run_metrics(returns: pd.Series, step_fee_records: list) -> tuple:
    eq_curve, _ = compute_fee_equity(returns, step_fee_records, INIT_CAPITAL)
    n_years = max(len(returns), 1) / 252
    total_ret = eq_curve.iloc[-1] - 1
    return {
        "total_return_pct": total_ret * 100,
        "ann_pct": ((1 + total_ret) ** (1 / n_years) - 1) * 100,
        "final_val": eq_curve.iloc[-1] * INIT_CAPITAL,
        "sharpe": _rob_sharpe(returns),
        "max_dd_pct": (eq_curve / eq_curve.cummax() - 1).min() * 100,
    }, eq_curve


def _run_single(oos, steps, daily_ret_panel, drop_etfs=None, seed=None,
                vix_s=None, vix_ema100=None,
                fl_test_by_step=None,
                test_ic_ema_lagged=None, fl_test_ic_ema_lagged=None,
                track_regimes=False):
    """Run a single backtest (with optional random ETF drops).
    Returns (daily_returns, step_fee_records) or
    (daily_returns, step_fee_records, regime_dates_dict) if track_regimes=True."""
    rng = np.random.default_rng(seed) if seed is not None else None
    if vix_s is None:
        vix_s = pd.Series(dtype=float)
    if vix_ema100 is None:
        vix_ema100 = pd.Series(dtype=float)
    if test_ic_ema_lagged is None:
        test_ic_ema_lagged = pd.Series(dtype=float)
    if fl_test_ic_ema_lagged is None:
        fl_test_ic_ema_lagged = pd.Series(dtype=float)

    all_test_returns = []
    step_fee_records = []
    carry_weights = None
    carry_regime = None
    model_days = [] if track_regimes else None
    fl_days = [] if track_regimes else None
    cash_days_list = [] if track_regimes else None

    for step in steps:
        # Random ETF drop
        dropped_set = set()
        if drop_etfs is not None and rng is not None:
            from etf import UNIVERSE as _U
            available = [e.bourso for e in _U]
            n_drop = rng.integers(drop_etfs[0], drop_etfs[1] + 1) if isinstance(drop_etfs, tuple) else drop_etfs
            n_drop = min(n_drop, len(available) - 2)
            dropped = rng.choice(available, size=n_drop, replace=False)
            dropped_set = set(dropped.tolist())

        result = _process_step(oos, step, daily_ret_panel, vix_s, vix_ema100,
                               test_ic_ema_lagged, fl_test_ic_ema_lagged,
                               fl_test_by_step, carry_weights, carry_regime,
                               dropped_set=dropped_set or None)
        if result is None:
            continue

        carry_weights = result["carry_weights"]
        carry_regime = result["carry_regime"]
        step_fee_records.append((result["test_returns"].index, result["step_fees"]))
        all_test_returns.append(result["test_returns"])

        if track_regimes:
            idx = result["model_scores_index"]
            for i in range(len(result["step_regime"])):
                if result["step_regime"][i] == 0:
                    model_days.append(pd.DatetimeIndex([idx[i]]))
                elif result["step_regime"][i] == 2:
                    fl_days.append(pd.DatetimeIndex([idx[i]]))
                elif result["step_regime"][i] == 3:
                    cash_days_list.append(pd.DatetimeIndex([idx[i]]))

    if not all_test_returns:
        if track_regimes:
            return pd.Series(dtype=float), [], {}
        return pd.Series(dtype=float), []

    # Extend to last available data date
    _extend_to_last_date(
        all_test_returns, [], carry_weights, carry_regime,
        daily_ret_panel, vix_s, vix_ema100, step_fee_records)

    ret = pd.concat(all_test_returns).sort_index()
    if track_regimes:
        def _concat_idx(lst):
            return pd.DatetimeIndex(pd.concat([pd.Series(x) for x in lst]).values) if lst else pd.DatetimeIndex([])
        regime_dates = {
            "model": _concat_idx(model_days),
            "fl": _concat_idx(fl_days),
            "cash": _concat_idx(cash_days_list),
        }
        return ret, step_fee_records, regime_dates
    return ret, step_fee_records


_rob_shared = {}

def _run_one_robustness(run_idx):
    d = _rob_shared
    run_returns, run_fees = _run_single(
        d["oos"], d["steps"], d["daily_ret_panel"],
        drop_etfs=N_DROP, seed=run_idx, vix_s=d["vix_s"],
        vix_ema100=d["vix_ema100"],
        fl_test_by_step=d["fl_test_by_step"],
        test_ic_ema_lagged=d["test_ic_ema_lagged"],
        fl_test_ic_ema_lagged=d["fl_test_ic_ema_lagged"])
    m, run_eq = _run_metrics(run_returns, run_fees)
    m["run"] = run_idx + 1
    return m, run_eq


def run_robustness():
    """Robustness backtest → outputs/backtest_robustness.jpg + summary CSV."""
    print("Loading OOS predictions...", flush=True)
    oos = _load_oos()

    daily_ret_panel = _load_daily_returns()
    oos = _prepend_heuristic_steps(oos, daily_ret_panel)
    steps = _get_steps(oos)
    all_steps = sorted(oos["step"].unique())

    vix_s, vix_ema100, vix_ema300 = _load_vix()

    fl_test_by_step, _ = _load_fl_predictions()
    if fl_test_by_step is not None:
        print(f"Follow Leads loaded for robustness: {len(fl_test_by_step)} steps")

    # IC gates
    test_ic_ema_lagged = _compute_ic_gate(oos, all_steps,
                                           IC_GATE_THRESHOLD, IC_GATE_SPAN, IC_GATE_MIN_STEPS)
    fl_test_ic_ema_lagged = _compute_fl_ic_gate(fl_test_by_step, all_steps)

    from etf import UNIVERSE
    print(f"Steps: {len(steps)}, ETFs: {len(UNIVERSE)}, Runs: {N_RUNS}, Drop: {N_DROP}")

    # Run original (no drop)
    print("Running original (no drop)...", flush=True)
    orig_returns, orig_fees, orig_regime_dates = _run_single(
        oos, steps, daily_ret_panel,
        drop_etfs=None, seed=None, vix_s=vix_s,
        vix_ema100=vix_ema100,
        fl_test_by_step=fl_test_by_step,
        test_ic_ema_lagged=test_ic_ema_lagged,
        fl_test_ic_ema_lagged=fl_test_ic_ema_lagged,
        track_regimes=True)
    orig_m, orig_eq = _run_metrics(orig_returns, orig_fees)

    # Parallel perturbed runs
    from multiprocessing import Pool, cpu_count
    global _rob_shared
    _rob_shared = dict(oos=oos, steps=steps, daily_ret_panel=daily_ret_panel,
                       vix_s=vix_s, vix_ema100=vix_ema100,
                       fl_test_by_step=fl_test_by_step,
                       test_ic_ema_lagged=test_ic_ema_lagged,
                       fl_test_ic_ema_lagged=fl_test_ic_ema_lagged)

    n_workers = min(cpu_count(), N_RUNS, 16)
    print(f"Running {N_RUNS} perturbed backtests on {n_workers} cores...", flush=True)
    with Pool(n_workers) as pool:
        results = pool.map(_run_one_robustness, range(N_RUNS))

    stats = []
    all_run_eq = []
    for m, run_eq in results:
        stats.append(m)
        all_run_eq.append(run_eq)
        print(f"  Run {m['run']:2d}/{N_RUNS}  ann={m['ann_pct']:+.1f}%/an  "
              f"sharpe={m['sharpe']:.2f}  dd={m['max_dd_pct']:.1f}%", flush=True)

    stats_df = pd.DataFrame(stats)[["run", "total_return_pct", "ann_pct",
                                    "final_val", "sharpe", "max_dd_pct"]]
    stats_df.to_csv(OUTPUTS / "backtest_robustness.csv", index=False)

    def _agg(col):
        return stats_df[col].median(), stats_df[col].min(), stats_df[col].max()

    ann_a = _agg("ann_pct")
    tot_a = _agg("total_return_pct")
    finv, shp, dd = _agg("final_val"), _agg("sharpe"), _agg("max_dd_pct")

    # --- Chart ---
    fig = plt.figure(figsize=(24, 15))
    gs = fig.add_gridspec(2, 2, width_ratios=[3, 1], height_ratios=[3, 0.45],
                          hspace=0.06, wspace=0.02,
                          top=0.96, bottom=0.05, left=0.05, right=0.98)
    ax1 = fig.add_subplot(gs[0, 0])
    ax_tbl = fig.add_subplot(gs[1, 0])
    ax_leg = fig.add_subplot(gs[:, 1])
    ax_leg.axis("off")

    for eq in all_run_eq:
        ax1.plot(eq.index, eq.values, lw=0.8, color="#888888", alpha=0.30, zorder=2)

    eq_matrix = pd.DataFrame({i: eq for i, eq in enumerate(all_run_eq)})
    median_eq = eq_matrix.median(axis=1)
    ax1.plot(median_eq.index, median_eq.values, lw=2.8, color="#1f77b4", zorder=8,
             label=f"Median ({N_RUNS} runs)")
    ax1.plot(orig_eq.index, orig_eq.values, lw=3.5, color="#d62728", zorder=10,
             label="Original")

    ax1.set_yscale("log")
    ax1.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:.1f}x"))
    ax1.set_ylabel("Equity (log scale, base 1)")
    ax1.set_title(f"MyQTM-ETF — Robustness  ({N_RUNS} backtests, "
                  f"{N_DROP[0]}-{N_DROP[1]} random ETFs dropped)")
    ax1.legend(loc="upper left", fontsize=12)
    ax1.grid(True, alpha=0.3, which="both")
    ax1.set_facecolor("#f8f8f8")

    # VIX overlay (uses regime dates from original run)
    if not vix_s.empty:
        _draw_vix_overlay(ax1, orig_eq.index, vix_s,
                          model_active_dates=orig_regime_dates.get("model"),
                          fl_active_dates=orig_regime_dates.get("fl"),
                          cash_dates=orig_regime_dates.get("cash"))

    _draw_regime_table(ax_tbl)

    # Right column stats
    def _eur(x):
        return f"{x:,.0f}".replace(",", " ") + " €"

    RED, BLUE = "#d62728", "#1f77b4"
    rows = [
        ("title", "ROBUSTNESS", ""),
        ("text", f"{N_RUNS} backtests · {N_DROP[0]}-{N_DROP[1]} random ETFs dropped", ""),
        ("kv", "Initial capital", _eur(INIT_CAPITAL)),
        ("sep", "", ""),
        ("sub", "Annual return", ""),
        ("kvr", "Original", f"{orig_m['ann_pct']:+.1f}%"),
        ("kvr", "Median", f"{ann_a[0]:+.1f}%"),
        ("kvr", "Min", f"{ann_a[1]:+.1f}%"),
        ("kvr", "Max", f"{ann_a[2]:+.1f}%"),
        ("sep", "", ""),
        ("sub", "Total return", ""),
        ("kvr", "Original", f"{orig_m['total_return_pct']:+.0f}%"),
        ("kvr", "Median", f"{tot_a[0]:+.0f}%"),
        ("kvr", "Min", f"{tot_a[1]:+.0f}%"),
        ("kvr", "Max", f"{tot_a[2]:+.0f}%"),
        ("sep", "", ""),
        ("sub", "Final value", ""),
        ("kvr", "Original", _eur(orig_m["final_val"])),
        ("kvr", "Median", _eur(finv[0])),
        ("kvr", "Min", _eur(finv[1])),
        ("kvr", "Max", _eur(finv[2])),
        ("sep", "", ""),
        ("sub", "Sharpe", ""),
        ("kvr", "Original", f"{orig_m['sharpe']:.2f}"),
        ("kvr", "Median", f"{shp[0]:.2f}"),
        ("kvr", "Min – Max", f"{shp[1]:.2f} – {shp[2]:.2f}"),
        ("sep", "", ""),
        ("sub", "Max Drawdown", ""),
        ("kvr", "Median", f"{dd[0]:.1f}%"),
        ("kvr", "Worst – Best", f"{dd[1]:.1f}% / {dd[2]:.1f}%"),
    ]

    def _lbl(label):
        return (BLUE, "bold") if label.strip() == "Median" else ("#000000", "normal")

    y, ystep = 0.99, 0.0305
    for kind, a, b in rows:
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
        elif kind in ("kv", "kvr"):
            lcol, lfw = _lbl(a)
            vcol = RED if kind == "kvr" else "#333333"
            ax_leg.text(0.04, y, a, fontsize=12.5, va="top", transform=ax_leg.transAxes,
                        color=lcol, fontweight=lfw)
            ax_leg.text(0.99, y, b, fontsize=12.5, va="top", ha="right",
                        transform=ax_leg.transAxes, color=vcol)
            y -= ystep

    fig.savefig(OUTPUTS / "backtest_robustness.jpg", dpi=100, bbox_inches="tight",
                format="jpeg", pil_kwargs={"quality": 60, "optimize": True})
    plt.close(fig)

    print(f"\n{'='*70}")
    print(f"  Original:  ann {orig_m['ann_pct']:+.1f}%/an  "
          f"Sharpe {orig_m['sharpe']:.2f}  DD {orig_m['max_dd_pct']:.1f}%")
    print(f"  Médian:    ann {ann_a[0]:+.1f}%/an  "
          f"Sharpe {shp[0]:.2f}  DD {dd[0]:.1f}%")
    print(f"  Min:       ann {ann_a[1]:+.1f}%/an  Sharpe {shp[1]:.2f}")
    print(f"  Max:       ann {ann_a[2]:+.1f}%/an  Sharpe {shp[2]:.2f}")
    print(f"{'='*70}")
    print(f"\nSaved → {OUTPUTS.relative_to(ROOT)}/backtest_robustness.jpg")
    print(f"Saved → {OUTPUTS.relative_to(ROOT)}/backtest_robustness.csv")


# ---------------------------------------------------------------------------
#  Entry point
# ---------------------------------------------------------------------------

def _auto_refresh_data():
    """Download latest OHLCV + VIX + regenerate features before backtesting."""
    from robot import refresh_ohlcv, refresh_vix
    print("Auto-refreshing data to latest available...")
    refresh_ohlcv()
    refresh_vix()
    # Regenerate features so tail extension can use model inference
    import subprocess
    feat_path = ROOT / "data" / "features.parquet"
    if feat_path.exists():
        last_feat = pd.read_parquet(feat_path).index.get_level_values("date").max()
        last_ohlcv = _load_daily_returns().index[-1]
        if last_ohlcv > last_feat:
            print(f"Features outdated ({last_feat.date()} < {last_ohlcv.date()}), regenerating...")
            feat_eng = Path(__file__).parent / "feature_engineering.py"
            subprocess.run([sys.executable, str(feat_eng)],
                           cwd=str(ROOT), check=True, timeout=600)
    print()


def main():
    # Data refresh disabled — working on backtest only
    # To re-enable: uncomment the two lines below
    # if "--no-refresh" not in sys.argv[1:]:
    #     _auto_refresh_data()
    if "--robustness" in sys.argv[1:]:
        run_robustness()
        return
    run_equity()
    print("\n■ all backtests done — equity, per-year, pies")


if __name__ == "__main__":
    main()
