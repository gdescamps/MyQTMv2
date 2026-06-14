"""
Compare Point-in-Time vs Revised data for the risk-off strategy.

Downloads fresh data, compares with stored point-in-time data,
runs walk-forward on both, and plots 2 equity curves.

Usage: python src/risk_off_strategy/compare_pit.py [QQQ|SPY]
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.risk_off_strategy.data import build_features, build_realtime_target
from src.risk_off_strategy.backtest import walk_forward, simulate_with_fees

DATA_DIR = ROOT / "data"

# Same config as run.py
START = "2000-01-01"
DD_EXIT = -0.10
DD_REENTER = -0.05
MIN_TRAIN = 504
STEP = 21
TEMPERATURE = 3.0
PROB_CASH = 0.70
PROB_FULL = 0.75
LOOKAHEAD = 6


def load_pair(ticker="QQQ", start=START, end="2026-12-31", spread_lag=2):
    """Load both point-in-time and revised data. Returns (pit, revised) tuples."""

    def _load(suffix=""):
        s = suffix
        price = pd.read_parquet(DATA_DIR / f"{ticker}{s}.parquet")["close"]
        vix = pd.read_parquet(DATA_DIR / f"vix_ohlc{s}.parquet")["close"]
        tlt = pd.read_parquet(DATA_DIR / f"TLT{s}.parquet")["close"]
        spread = pd.read_parquet(DATA_DIR / f"fred_baa_spread{s}.parquet")["baa_spread"]
        if spread_lag > 0:
            spread = spread.shift(spread_lag)
        price = price.loc[start:end].dropna()
        vix = vix.loc[start:end].dropna()
        spread = spread.loc[start:end].dropna()
        tlt = tlt.loc[start:end].dropna()
        common = price.index.intersection(vix.index).intersection(
            spread.index).intersection(tlt.index)
        return price.loc[common], vix.loc[common], spread.loc[common], tlt.loc[common]

    pit = _load("")
    revised = _load("_revised")
    return pit, revised


def report_diffs(pit, revised, ticker="QQQ"):
    """Print data differences between point-in-time and revised."""
    labels = [f"{ticker} price", "VIX", "BAA spread", "TLT"]
    total_diffs = 0
    for i, label in enumerate(labels):
        s_pit, s_rev = pit[i], revised[i]
        common = s_pit.index.intersection(s_rev.index)
        if len(common) == 0:
            print(f"  {label}: no overlapping dates")
            continue
        pit_vals = s_pit.loc[common]
        rev_vals = s_rev.loc[common]
        diff_mask = ~np.isclose(pit_vals.values, rev_vals.values, rtol=1e-6)
        n_diff = diff_mask.sum()
        total_diffs += n_diff
        if n_diff > 0:
            pct = n_diff / len(common) * 100
            max_delta = np.abs(pit_vals.values[diff_mask] - rev_vals.values[diff_mask]).max()
            print(f"  {label}: {n_diff}/{len(common)} rows differ ({pct:.2f}%), "
                  f"max delta={max_delta:.6f}")
            # Show first few diffs
            diff_dates = common[diff_mask]
            for d in diff_dates[:5]:
                print(f"    {d.date()}: PIT={pit_vals.loc[d]:.4f}  REV={rev_vals.loc[d]:.4f}  "
                      f"delta={rev_vals.loc[d] - pit_vals.loc[d]:+.4f}")
            if len(diff_dates) > 5:
                print(f"    ... ({len(diff_dates) - 5} more)")
        else:
            print(f"  {label}: identical ({len(common)} rows)")
    return total_diffs


def run_wf(price, vix, spread, tlt, ticker, cache_suffix=""):
    """Run walk-forward and return (dates, price_ret, proba)."""
    prefix = ticker.lower()
    df = build_features(price, vix, spread, tlt, prefix=prefix)
    target, _ = build_realtime_target(price.values, DD_EXIT, DD_REENTER, lookahead=LOOKAHEAD)
    df["target"] = target
    df = df.dropna()

    feature_cols = [c for c in df.columns if c != "target"]
    X = df[feature_cols].values
    y = df["target"].values

    wf_pred, wf_proba, _ = walk_forward(
        X, y, feature_cols,
        min_train=MIN_TRAIN, step=STEP, temperature=TEMPERATURE,
        use_cache=(cache_suffix == ""),  # only use cache for PIT
    )

    pred_mask = wf_pred >= 0
    wf_dates = df.index[pred_mask]
    wf_prob = wf_proba[pred_mask]
    price_ret = price.pct_change().fillna(0).loc[df.index].values[pred_mask]

    return wf_dates, price_ret, wf_prob


def plot_comparison(dates_pit, ret_pit, prob_pit,
                    dates_rev, ret_rev, prob_rev,
                    ticker="QQQ", save_path=None):
    """Plot PIT vs Revised equity curves."""
    lev = 1.0

    eq_pit, al_pit, _ = simulate_with_fees(ret_pit, prob_pit, lev, PROB_CASH, PROB_FULL)
    eq_rev, al_rev, _ = simulate_with_fees(ret_rev, prob_rev, lev, PROB_CASH, PROB_FULL)

    # Buy & hold (from PIT, should be same)
    bh = np.cumprod(1 + ret_pit)

    years = (dates_pit[-1] - dates_pit[0]).days / 365.25
    cagr_pit = eq_pit[-1] ** (1 / years) - 1
    cagr_rev = eq_rev[-1] ** (1 / years) - 1
    dd_pit = ((eq_pit - np.maximum.accumulate(eq_pit)) / np.maximum.accumulate(eq_pit)).min()
    dd_rev = ((eq_rev - np.maximum.accumulate(eq_rev)) / np.maximum.accumulate(eq_rev)).min()

    fig, axes = plt.subplots(3, 1, figsize=(16, 10), height_ratios=[3, 1, 1],
                             sharex=True, gridspec_kw={"hspace": 0.08})

    # Equity curves
    ax1 = axes[0]
    ax1.semilogy(dates_pit, bh, label=f"{ticker} B&H", color="tab:blue",
                 linewidth=2, alpha=0.5)
    ax1.semilogy(dates_pit, eq_pit,
                 label=f"Point-in-Time ({cagr_pit*100:.1f}%, DD {dd_pit*100:.1f}%)",
                 color="tab:orange", linewidth=2)
    ax1.semilogy(dates_rev, eq_rev,
                 label=f"Revised ({cagr_rev*100:.1f}%, DD {dd_rev*100:.1f}%)",
                 color="tab:red", linewidth=2, linestyle="--")
    ax1.set_ylabel("Equity (log)")
    ax1.set_title(f"{ticker} — Point-in-Time vs Revised data")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Allocation comparison
    ax2 = axes[1]
    ax2.fill_between(dates_pit, 0, al_pit * 100, alpha=0.4, color="tab:orange",
                     label="PIT alloc")
    ax2.plot(dates_rev, al_rev * 100, color="tab:red", linewidth=0.8,
             linestyle="--", label="Revised alloc", alpha=0.8)
    ax2.set_ylabel("Alloc %")
    ax2.set_ylim(-5, 115)
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Probability difference
    ax3 = axes[2]
    common_len = min(len(prob_pit), len(prob_rev))
    prob_diff = prob_rev[:common_len] - prob_pit[:common_len]
    ax3.fill_between(dates_pit[:common_len], 0, prob_diff,
                     where=prob_diff >= 0, color="green", alpha=0.4)
    ax3.fill_between(dates_pit[:common_len], 0, prob_diff,
                     where=prob_diff < 0, color="red", alpha=0.4)
    ax3.axhline(0, color="gray", linewidth=0.5)
    ax3.set_ylabel("P(revised) - P(pit)")
    ax3.set_xlabel("Date")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def main():
    ticker = sys.argv[1].upper() if len(sys.argv) > 1 else "QQQ"

    # Check that _revised files exist
    revised_check = DATA_DIR / f"{ticker}_revised.parquet"
    if not revised_check.exists():
        print(f"No revised data found ({revised_check})")
        print("Run 'python src/risk_off_strategy/run.py' first to generate _revised files.")
        return

    end = pd.read_parquet(DATA_DIR / f"{ticker}.parquet").index[-1].strftime("%Y-%m-%d")
    print(f"\n{'='*60}")
    print(f"  {ticker} Point-in-Time vs Revised comparison")
    print(f"{'='*60}\n")

    pit, revised = load_pair(ticker, end=end)

    print("Data differences:")
    n_diffs = report_diffs(pit, revised, ticker)
    if n_diffs == 0:
        print("\n  => No differences found. PIT and revised data are identical.")
        print("     Run this again after a few days of incremental downloads to see diffs.\n")
        return

    print(f"\nRunning walk-forward on PIT data...")
    dates_pit, ret_pit, prob_pit = run_wf(*pit, ticker, cache_suffix="")

    print(f"\nRunning walk-forward on Revised data...")
    dates_rev, ret_rev, prob_rev = run_wf(*revised, ticker, cache_suffix="_revised")

    OUT = ROOT / "outputs" / f"{ticker.lower()}_strategy"
    OUT.mkdir(parents=True, exist_ok=True)
    save_path = str(OUT / "pit_vs_revised.png")

    plot_comparison(dates_pit, ret_pit, prob_pit,
                    dates_rev, ret_rev, prob_rev,
                    ticker=ticker, save_path=save_path)


if __name__ == "__main__":
    main()
