"""
Robustness backtest: 20 simulations dropping 5 random ETFs at each rebalance.

For each of 20 runs:
  - At each rebalance step, randomly exclude 5 ETFs from the scores
  - Run the same allocation logic (softmax, Sharpe weights, etc.)
  - Record equity curve

Output:
  outputs/backtest_robustness.png  — 20 equity curves + median + original
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
    run_backtest, TEMPERATURE, REBAL_DAYS, USE_SOFTMAX, SHARPE_POWER,
    DATA, OUTPUTS,
    VIX_ADAPTIVE, VIX_LOW, VIX_HIGH, TOP_N_LOW, TOP_N_HIGH,
    REBAL_DAYS_LOW, REBAL_DAYS_HIGH, TOP_N_SCORES,
    VIX_SPIKE_MIN, VIX_SPIKE_MAX, VIX_SPIKE_REBAL,
    CAP_AT_SPIKE_MIN, CAP_AT_SPIKE_MAX, RECOVERY_RATE,
)

N_RUNS = 50
N_DROP = (5, 10)  # random between 5 and 10 ETFs dropped per rebalance


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
    orig_returns = _run_single(oos, steps, daily_ret_panel, drop_etfs=None, seed=None, vix_s=vix_s)

    # Run N_RUNS with random drops
    all_run_returns = []
    for run in range(N_RUNS):
        print(f"  Run {run+1:2d}/{N_RUNS}...", end="", flush=True)
        run_returns = _run_single(oos, steps, daily_ret_panel, drop_etfs=N_DROP, seed=run, vix_s=vix_s)
        all_run_returns.append(run_returns)
        eq = (1 + run_returns).cumprod()
        print(f"  ret={eq.iloc[-1]-1:+.1%}  sharpe={sharpe(run_returns):.2f}  "
              f"dd={(eq/eq.cummax()-1).min():.1%}", flush=True)

    # Build equity curves
    orig_eq = (1 + orig_returns).cumprod()
    run_eqs = [((1 + r).cumprod()) for r in all_run_returns]

    # Stats
    stats = []
    for i, r in enumerate(all_run_returns):
        eq = (1 + r).cumprod()
        stats.append({
            "run": i + 1,
            "total_return_pct": (eq.iloc[-1] - 1) * 100,
            "sharpe": sharpe(r),
            "max_dd_pct": (eq / eq.cummax() - 1).min() * 100,
        })
    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(Path(__file__).parent / "myfiles" / "backtest_robustness.csv", index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(14, 7))

    # 20 runs in light gray
    for eq in run_eqs:
        ax.plot(eq.index, eq.values, lw=0.7, color="#888888", alpha=0.4)

    # Median curve
    eq_matrix = pd.DataFrame({i: eq for i, eq in enumerate(run_eqs)})
    median_eq = eq_matrix.median(axis=1)
    ax.plot(median_eq.index, median_eq.values, lw=2.0, color="#1f77b4", label="Median (20 runs)")

    # Original
    ax.plot(orig_eq.index, orig_eq.values, lw=2.5, color="#d62728", label="Original (no drop)")

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:.1f}x"))
    ax.set_ylabel("Equity (log scale)")
    ax.set_title(f"Robustness: {N_RUNS} runs, dropping {N_DROP} random ETFs at each rebalance")
    ax.legend(loc="upper left", fontsize=11)
    ax.grid(True, alpha=0.3, which="both")
    ax.set_facecolor("#f8f8f8")

    # Add stats box
    med_sharpe = stats_df["sharpe"].median()
    med_dd = stats_df["max_dd_pct"].median()
    med_ret = stats_df["total_return_pct"].median()
    orig_sharpe = sharpe(orig_returns)
    orig_dd = (orig_eq / orig_eq.cummax() - 1).min() * 100

    txt = (f"Original:  Sharpe={orig_sharpe:.2f}  DD={orig_dd:.1f}%\n"
           f"Median:    Sharpe={med_sharpe:.2f}  DD={med_dd:.1f}%\n"
           f"Range:     Sharpe=[{stats_df['sharpe'].min():.2f}, {stats_df['sharpe'].max():.2f}]  "
           f"DD=[{stats_df['max_dd_pct'].min():.1f}%, {stats_df['max_dd_pct'].max():.1f}%]")
    ax.text(0.02, 0.02, txt, transform=ax.transAxes, fontsize=10,
            verticalalignment="bottom", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    fig.tight_layout()
    out_path = OUTPUTS / "backtest_robustness.png"
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)

    print(f"\n{'='*60}")
    print(f"  Original:  return={orig_eq.iloc[-1]-1:+.1%}  Sharpe={orig_sharpe:.2f}  DD={orig_dd:.1f}%")
    print(f"  Median:    return={med_ret:+.1f}%  Sharpe={med_sharpe:.2f}  DD={med_dd:.1f}%")
    print(f"  Min:       return={stats_df['total_return_pct'].min():+.1f}%  "
          f"Sharpe={stats_df['sharpe'].min():.2f}  DD={stats_df['max_dd_pct'].min():.1f}%")
    print(f"  Max:       return={stats_df['total_return_pct'].max():+.1f}%  "
          f"Sharpe={stats_df['sharpe'].max():.2f}  DD={stats_df['max_dd_pct'].max():.1f}%")
    print(f"{'='*60}")
    print(f"\nSaved → outputs/backtest_robustness.png")
    print(f"Saved → myfiles/backtest_robustness.csv")


def _run_single(oos, steps, daily_ret_panel, drop_etfs=None, seed=None, vix_s=None):
    """Run a single backtest with full VIX-adaptive logic, optionally dropping N random ETFs."""
    rng = np.random.default_rng(seed) if seed is not None else None

    all_test_returns = []
    current_alloc_cap = 1.0

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
                if not np.isnan(vix_5d_change) and vix_5d_change > VIX_SPIKE_MIN:
                    step_top_n = TOP_N_HIGH
                    step_rebal = VIX_SPIKE_REBAL
                    frac_spike = min((vix_5d_change - VIX_SPIKE_MIN) / (VIX_SPIKE_MAX - VIX_SPIKE_MIN), 1.0)
                    spike_cap = CAP_AT_SPIKE_MIN + frac_spike * (CAP_AT_SPIKE_MAX - CAP_AT_SPIKE_MIN)
                    current_alloc_cap = min(current_alloc_cap, spike_cap)
                elif vix_at_step < VIX_LOW:
                    step_top_n = TOP_N_LOW
                    step_rebal = REBAL_DAYS_LOW
                    current_alloc_cap = min(1.0, current_alloc_cap + RECOVERY_RATE)
                elif vix_at_step > VIX_HIGH:
                    step_top_n = TOP_N_HIGH
                    step_rebal = REBAL_DAYS_HIGH
                else:
                    frac = (vix_at_step - VIX_LOW) / (VIX_HIGH - VIX_LOW)
                    step_top_n = int(round(TOP_N_LOW + frac * (TOP_N_HIGH - TOP_N_LOW)))
                    step_rebal = int(round(REBAL_DAYS_LOW + frac * (REBAL_DAYS_HIGH - REBAL_DAYS_LOW)))
                    current_alloc_cap = min(1.0, current_alloc_cap + RECOVERY_RATE * 0.5)

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
        test_returns, _ = run_backtest(test_scores, test_dr, sharpe_weights=etf_sharpe,
                                       rebal_days_override=step_rebal, max_alloc=step_max_alloc)
        all_test_returns.append(test_returns)

    if not all_test_returns:
        return pd.Series(dtype=float)
    return pd.concat(all_test_returns).sort_index()


if __name__ == "__main__":
    run_robustness()
