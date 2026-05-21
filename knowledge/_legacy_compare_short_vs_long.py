"""
Diagnostic: short vs long modes, focused on 2019-2026 overlap.

Goal: explain why the long-mode model fails despite having smart-money — and
in particular what happens at the covid VIX spike (March 2020). The only
structural difference between short and long is the constituent ETF list:
short has 27 ETFs (geo + US sectors + thematic + commodity); long has 17
(mostly geo + SOXX + IEO + RING).

Outputs: outputs/short_vs_long_diagnostic.jpg
  Panel 1: Test IC per WF step, EMA12 (short vs long, same x-axis)
  Panel 2: Equity, log scale, indexed at 100 on 2019-09-30 (overlap start)
  Panel 3: Allocation overlap each step — Jaccard(top3_short, top3_long)
  Panel 4: VIX EMA100 + spike events (context)

Also prints a textual diff: top-weighted ETFs short vs long around covid
(2020-02 → 2020-06) and 2022 oil shock (2022-02 → 2022-07).
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from etf import UNIVERSE_SHORT, UNIVERSE_LONG  # noqa: E402

SHORT_OOS = ROOT / "data" / "short" / "oos_predictions.parquet"
LONG_OOS  = ROOT / "data" / "long"  / "oos_predictions.parquet"
SHORT_EQ  = ROOT / "outputs" / "short" / "backtest_equity.csv"
LONG_EQ   = ROOT / "outputs" / "long"  / "backtest_equity.csv"
VIX_PATH  = ROOT / "data" / "fred_vix.parquet"
OUT       = ROOT / "outputs" / "short_vs_long_diagnostic.jpg"


def step_test_ic(oos_path: Path) -> pd.DataFrame:
    oos = pd.read_parquet(oos_path).reset_index()
    oos["date"] = pd.to_datetime(oos["date"])
    t = oos[oos["split"] == "test"].dropna(subset=["score", "label"])

    def _step_ic(g):
        return g.groupby("date").apply(
            lambda x: x["score"].corr(x["label"]) if len(x) > 1 else np.nan,
            include_groups=False,
        ).mean()

    ic = t.groupby("step").apply(_step_ic, include_groups=False).rename("ic")
    start = t.groupby("step")["date"].min().rename("start")
    df = pd.concat([start, ic], axis=1).dropna().sort_values("start")
    df["ic_ema"] = df["ic"].ewm(span=12).mean()
    return df


def per_step_top_etfs(oos_path: Path, k: int = 3) -> pd.DataFrame:
    """For each WF step, return the top-k ETFs by mean test score
    (cross-section over the test window). This is a coarse proxy for what
    the allocator would pick (before per-day calm overrides)."""
    oos = pd.read_parquet(oos_path).reset_index()
    oos["date"] = pd.to_datetime(oos["date"])
    t = oos[oos["split"] == "test"].dropna(subset=["score"])
    g = t.groupby(["step", "etf_id"])["score"].mean().reset_index()
    starts = t.groupby("step")["date"].min().rename("start")

    rows = []
    for step, sub in g.groupby("step"):
        top = sub.sort_values("score", ascending=False).head(k)["etf_id"].tolist()
        rows.append({"step": step, "start": starts.loc[step], "top": top})
    return pd.DataFrame(rows).sort_values("start").reset_index(drop=True)


def main():
    print("Loading short OOS predictions...")
    short_ic = step_test_ic(SHORT_OOS)
    print(f"  {len(short_ic)} steps, mean test IC = {short_ic['ic'].mean():+.4f}")

    print("Loading long OOS predictions...")
    long_ic = step_test_ic(LONG_OOS)
    print(f"  {len(long_ic)} steps, mean test IC = {long_ic['ic'].mean():+.4f}")

    print("Loading equity curves...")
    short_eq = pd.read_csv(SHORT_EQ, parse_dates=["date"]).set_index("date")["equity"]
    long_eq  = pd.read_csv(LONG_EQ,  parse_dates=["date"]).set_index("date")["equity"]

    # Overlap: short starts ~2019-09; long starts 2011-01. Re-index both to
    # value 1.0 at the overlap start so the divergence is visible at a glance.
    overlap_start = short_eq.index.min()
    short_eq_idx = short_eq / short_eq.loc[overlap_start]
    long_eq_idx  = long_eq.loc[overlap_start:] / long_eq.loc[overlap_start]
    print(f"  overlap start = {overlap_start.date()}")
    print(f"  short final ×{short_eq_idx.iloc[-1]:.2f}   long final ×{long_eq_idx.iloc[-1]:.2f}")

    print("Loading VIX...")
    vix = pd.read_parquet(VIX_PATH).iloc[:, 0]
    vix.index = pd.to_datetime(vix.index).tz_localize(None) if vix.index.tz is not None else pd.to_datetime(vix.index)
    vix_ema100 = vix.ewm(span=100).mean()

    print("\nPer-step top-3 ETF differences (long vs short)...")
    short_top = per_step_top_etfs(SHORT_OOS, k=3)
    long_top  = per_step_top_etfs(LONG_OOS,  k=3)

    # Jaccard overlap per step (aligned on test_start)
    long_top["start_norm"] = long_top["start"]
    short_top["start_norm"] = short_top["start"]
    merged = pd.merge_asof(
        short_top.sort_values("start_norm"),
        long_top.sort_values("start_norm"),
        on="start_norm", suffixes=("_s", "_l"),
        direction="nearest", tolerance=pd.Timedelta("15D"),
    ).dropna(subset=["top_l"])

    def jacc(a, b):
        sa, sb = set(a), set(b)
        return len(sa & sb) / max(len(sa | sb), 1)
    merged["jaccard"] = merged.apply(lambda r: jacc(r["top_s"], r["top_l"]), axis=1)
    print(f"  {len(merged)} overlapping steps")
    print(f"  mean Jaccard(top3_short, top3_long) = {merged['jaccard'].mean():.3f}")
    print(f"  steps with zero overlap            = {(merged['jaccard']==0).sum()}/{len(merged)}")

    # Spotlight windows: covid (2020-02 → 2020-06) and oil shock (2022-02 → 2022-07)
    for label, lo, hi in [
        ("covid 2020-Q1/Q2",  "2020-01-01", "2020-07-01"),
        ("oil shock 2022",    "2022-01-01", "2022-07-01"),
        ("2024 calm",         "2024-01-01", "2024-07-01"),
    ]:
        sub = merged[(merged["start_norm"] >= lo) & (merged["start_norm"] < hi)]
        print(f"\n  === {label} ({len(sub)} steps) ===")
        for _, r in sub.iterrows():
            common = sorted(set(r["top_s"]) & set(r["top_l"]))
            only_s = sorted(set(r["top_s"]) - set(r["top_l"]))
            only_l = sorted(set(r["top_l"]) - set(r["top_s"]))
            print(f"    {pd.Timestamp(r['start_norm']).date()}  "
                  f"common={common or '∅'}  "
                  f"short-only={only_s or '∅'}  "
                  f"long-only={only_l or '∅'}")

    # ── PLOT ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(4, 1, figsize=(18, 16), sharex=True)

    # (1) Test IC EMA per step
    ax = axes[0]
    ax.plot(short_ic["start"], short_ic["ic"],     color="#1f77b4", alpha=0.20, lw=0.7)
    ax.plot(short_ic["start"], short_ic["ic_ema"], color="#1f77b4", lw=2.2,
            label=f"Short (mean {short_ic['ic'].mean():+.4f})")
    ax.plot(long_ic["start"],  long_ic["ic"],     color="#d62728", alpha=0.20, lw=0.7)
    ax.plot(long_ic["start"],  long_ic["ic_ema"], color="#d62728", lw=2.2,
            label=f"Long (mean {long_ic['ic'].mean():+.4f})")
    ax.axhline(0, color="black", lw=0.5, ls="--")
    ax.set_ylabel("Test IC per WF step (EMA12)")
    ax.set_title("Short vs Long — Test IC per walk-forward step (true OOS)", fontsize=13)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    # (2) Equity, log, indexed at 1.0 on overlap start
    ax = axes[1]
    ax.semilogy(short_eq_idx.index, short_eq_idx, color="#1f77b4", lw=1.8,
                label=f"Short (×{short_eq_idx.iloc[-1]:.2f})")
    ax.semilogy(long_eq_idx.index, long_eq_idx, color="#d62728", lw=1.8,
                label=f"Long  (×{long_eq_idx.iloc[-1]:.2f})")
    ax.set_ylabel("Equity (log, indexed=1.0 at overlap start)")
    ax.set_title(f"Equity (log) — indexed to 1.0 on {overlap_start.date()}", fontsize=12)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3, which="both")

    # (3) Allocation overlap (Jaccard of top-3 picks)
    ax = axes[2]
    ax.fill_between(merged["start_norm"], 0, merged["jaccard"],
                    color="#2ca02c", alpha=0.35, step="post")
    ax.plot(merged["start_norm"], merged["jaccard"].rolling(6, min_periods=1).mean(),
            color="#2ca02c", lw=2.0,
            label=f"6-step rolling mean ({merged['jaccard'].mean():.2f} overall)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Jaccard(top-3 short, top-3 long)")
    ax.set_title("Top-3 allocation overlap per WF step (1.0 = same picks; 0.0 = nothing in common)",
                 fontsize=12)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    # (4) VIX EMA100 + calm threshold context
    ax = axes[3]
    ax.plot(vix.loc[overlap_start:], color="#888", lw=0.7, alpha=0.5, label="VIX")
    ax.plot(vix_ema100.loc[overlap_start:], color="black", lw=1.8, label="VIX EMA100")
    ax.axhline(19, color="green", ls="--", lw=1.0, alpha=0.7, label="calm threshold")
    ax.set_ylabel("VIX")
    ax.set_xlabel("Test window start")
    ax.set_title("VIX context", fontsize=12)
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    for ax in axes:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, dpi=110, bbox_inches="tight")
    print(f"\nSaved → {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
