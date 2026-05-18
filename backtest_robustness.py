"""
Robustness backtest: 20 simulations dropping 5 random ETFs at each rebalance.

For each of 20 runs:
  - At each rebalance step, randomly exclude 5 ETFs from the scores
  - Run the same allocation logic (softmax, Sharpe weights, etc.)
  - Record equity curve

Output:
  outputs/backtest_robustness.jpg  — 20 equity curves + median + original
  myfiles/backtest_robustness.csv  — summary stats per run
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
from backtest import (
    run_backtest, compute_red_orange, TEMPERATURE, REBAL_DAYS, USE_SOFTMAX,
    SHARPE_POWER, DATA, OUTPUTS,
    VIX_ADAPTIVE, VIX_LOW, VIX_HIGH, TOP_N_LOW, TOP_N_HIGH,
    REBAL_DAYS_LOW, REBAL_DAYS_HIGH, TOP_N_SCORES,
    VIX_SPIKE_MIN, VIX_SPIKE_MAX, VIX_SPIKE_REBAL,
    CAP_AT_SPIKE_MIN, CAP_AT_SPIKE_MAX, RECOVERY_RATE,
)

N_RUNS = 50
N_DROP = (5, 10)  # random between 5 and 10 ETFs dropped per rebalance
INIT_CAPITAL = 150_000.0  # same starting capital as backtest.py


def sharpe(returns: pd.Series) -> float:
    if len(returns) < 10 or returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(252))


def _pivot_step(step_data: pd.DataFrame) -> pd.DataFrame:
    return step_data.reset_index().pivot(index="date", columns="etf_id", values="score")


def run_robustness():
    oos_path = DATA / "oos_predictions.parquet"
    if not oos_path.exists():
        sys.exit("ERROR: data/oos_predictions.parquet not found")

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
    stats_df.to_csv(Path(__file__).parent / "myfiles" / "backtest_robustness.csv", index=False)

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
    ax1.plot(orig_orange.index, orig_orange.values, lw=3.5, color="#ff7f0e", zorder=11,
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
    vix_path = DATA / "fred_vix.parquet"
    if vix_path.exists():
        vix_raw = pd.read_parquet(vix_path).iloc[:, 0]
        vix_raw = vix_raw.reindex(orig_red.index, method="ffill").dropna()
        ax1b = ax1.twinx()
        ax1b.fill_between(vix_raw.index, vix_raw.values, alpha=0.10, color="#d62728")
        ax1b.set_yticks([])
        ax1b.set_ylabel("")
        ax1b.set_ylim(0, 80)

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
    print(f"Saved → myfiles/backtest_robustness.csv")


def _run_metrics(returns: pd.Series, step_fee_records: list) -> tuple:
    """Build red/orange curves (same model as backtest.py) and scalar metrics.
    Returns (metrics_dict, eq_red, eq_orange)."""
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
        "sharpe": sharpe(returns),
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

        val_scores = _pivot_step(val_data)
        val_end_date = val_scores.index[-1]
        test_scores = _pivot_step(test_data)

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

        # Pre-filter top N
        for idx in range(len(test_scores)):
            row = test_scores.iloc[idx].values
            valid = ~np.isnan(row)
            pos = valid & (row > 0)
            if pos.sum() > step_top_n:
                pos_idx = np.where(pos)[0]
                keep = pos_idx[np.argsort(row[pos_idx])[::-1][:step_top_n]]
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


if __name__ == "__main__":
    run_robustness()
