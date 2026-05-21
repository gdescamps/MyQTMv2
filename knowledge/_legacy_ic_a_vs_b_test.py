"""
Disambiguate A (the 17 ETFs themselves carry weak smart-money signal) vs
B (the 10 missing ETFs are needed for the model to extract a usable
cross-sectional signal from those 17).

Method:
  1. Load short OOS predictions (model trained on 27 ETFs, cross-section over 27).
  2. Restrict those predictions to the 17 ETFs of UNIVERSE_LONG, same dates.
  3. Compute per-step test IC on this restricted subset.
  4. Compare with the long model's per-step test IC.

Interpretation:
  IC(short restricted to 17) ≈ IC(long)  ⇒ A: the 17 ETFs are intrinsically
                                            hard to rank, the smart-money
                                            signal they carry doesn't
                                            generalize.
  IC(short restricted to 17) ≫ IC(long)  ⇒ B: the 10 extra ETFs (and the
                                            cross-sectional features built
                                            over 27 vs 17) contain context
                                            the long model misses. The 17
                                            *are* rankable, but only with
                                            a richer universe.

Output: outputs/ic_a_vs_b_test.jpg + console summary.
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
from etf import UNIVERSE_LONG  # noqa: E402

LONG_ETFS = {e.bourso for e in UNIVERSE_LONG}
SHORT_OOS = ROOT / "data" / "short" / "oos_predictions.parquet"
LONG_OOS  = ROOT / "data" / "long"  / "oos_predictions.parquet"
OUT       = ROOT / "outputs" / "ic_a_vs_b_test.jpg"


def per_step_ic(oos: pd.DataFrame, restrict_etfs: set | None = None) -> pd.DataFrame:
    t = oos[oos["split"] == "test"].dropna(subset=["score", "label"]).copy()
    if restrict_etfs is not None:
        t = t[t["etf_id"].isin(restrict_etfs)]

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


def main():
    print(f"Long universe = {len(LONG_ETFS)} ETFs: {sorted(LONG_ETFS)}")

    short = pd.read_parquet(SHORT_OOS).reset_index()
    short["date"] = pd.to_datetime(short["date"])
    long_ = pd.read_parquet(LONG_OOS).reset_index()
    long_["date"] = pd.to_datetime(long_["date"])

    short_full  = per_step_ic(short)
    short_res17 = per_step_ic(short, restrict_etfs=LONG_ETFS)
    long_native = per_step_ic(long_)

    # Long has many more steps (2011-2026); restrict to overlap with short (2019-2026)
    overlap_start = short_full["start"].min()
    long_overlap = long_native[long_native["start"] >= overlap_start].copy()
    long_overlap["ic_ema"] = long_overlap["ic"].ewm(span=12).mean()

    print(f"\nOverlap period: {overlap_start.date()} → ...")
    print(f"{'series':<30} {'steps':>6} {'mean IC':>10} {'median IC':>11} {'std':>8}")
    print("-" * 70)
    for name, df in [
        ("Short model — full 27 ETFs",       short_full),
        ("Short model — restricted to 17",   short_res17),
        ("Long model  — native 17 ETFs",     long_overlap),
    ]:
        print(f"{name:<30} {len(df):>6} {df['ic'].mean():>+10.4f} "
              f"{df['ic'].median():>+11.4f} {df['ic'].std():>8.4f}")

    # Per-step paired comparison (same WF step → same date window)
    merged = pd.merge_asof(
        short_res17.sort_values("start"),
        long_overlap.sort_values("start"),
        on="start", suffixes=("_short17", "_long"),
        direction="nearest", tolerance=pd.Timedelta("15D"),
    ).dropna(subset=["ic_long"])
    merged["diff"] = merged["ic_short17"] - merged["ic_long"]
    print(f"\nPaired diff (short@17 − long@17), n={len(merged)} steps:")
    print(f"  mean   = {merged['diff'].mean():+.4f}")
    print(f"  median = {merged['diff'].median():+.4f}")
    print(f"  pct steps where short@17 > long@17 = {(merged['diff'] > 0).mean():.1%}")

    # Also: covid window specifically
    covid = merged[(merged["start"] >= "2020-01-01") & (merged["start"] < "2020-07-01")]
    if len(covid):
        print(f"\nCovid window 2020-Q1/Q2 (n={len(covid)}):")
        print(f"  IC short@17 = {covid['ic_short17'].mean():+.4f}")
        print(f"  IC long@17  = {covid['ic_long'].mean():+.4f}")
        print(f"  diff        = {covid['diff'].mean():+.4f}")

    # ─── Verdict ─────────────────────────────────────────────────────────
    mean_diff = merged["diff"].mean()
    mean_short_res = short_res17["ic"].mean()
    mean_long = long_overlap["ic"].mean()
    if mean_diff > 0.02 and mean_short_res > mean_long + 0.02:
        verdict = ("\nVERDICT: B — context matters. The 17 ETFs *are* rankable, "
                   "but only when the model trains on the broader 27-ETF universe.")
    elif abs(mean_diff) < 0.02:
        verdict = ("\nVERDICT: A — the 17 ETFs are intrinsically weak. Even the "
                   "short model, with its richer cross-section, can't rank them well.")
    else:
        verdict = (f"\nVERDICT: ambiguous (mean diff {mean_diff:+.4f}). Inspect "
                   "the per-step IC plot.")
    print(verdict)

    # ─── PLOT ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(18, 10), sharex=True)

    ax = axes[0]
    ax.plot(short_full["start"],  short_full["ic_ema"],
            color="#1f77b4", lw=2.2, label=f"Short / 27 ETFs  (mean {short_full['ic'].mean():+.4f})")
    ax.plot(short_res17["start"], short_res17["ic_ema"],
            color="#9467bd", lw=2.2, label=f"Short restricted to 17  (mean {short_res17['ic'].mean():+.4f})")
    ax.plot(long_overlap["start"], long_overlap["ic_ema"],
            color="#d62728", lw=2.2, label=f"Long / 17 ETFs  (mean {long_overlap['ic'].mean():+.4f})")
    ax.axhline(0, color="black", lw=0.5, ls="--")
    ax.set_ylabel("Test IC (EMA12)")
    ax.set_title("A vs B test — Test IC per WF step (raw light, EMA bold)", fontsize=13)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    # Light raw points
    for df, color in [(short_full, "#1f77b4"), (short_res17, "#9467bd"), (long_overlap, "#d62728")]:
        ax.plot(df["start"], df["ic"], color=color, alpha=0.15, lw=0.5)

    ax = axes[1]
    ax.fill_between(merged["start"], 0, merged["diff"],
                    where=(merged["diff"] > 0), color="#2ca02c", alpha=0.4,
                    label="short@17 > long  (B-evidence)")
    ax.fill_between(merged["start"], 0, merged["diff"],
                    where=(merged["diff"] <= 0), color="#d62728", alpha=0.4,
                    label="short@17 ≤ long  (A-evidence)")
    ax.plot(merged["start"], merged["diff"].rolling(6, min_periods=1).mean(),
            color="black", lw=1.6, label="6-step rolling mean")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("IC(short@17) − IC(long native)")
    ax.set_xlabel("Test window start")
    ax.set_title(f"Paired per-step IC delta — positive = the 10 extra ETFs help rank the 17 "
                 f"(mean diff {merged['diff'].mean():+.4f})", fontsize=12)
    ax.legend(loc="upper left")
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
