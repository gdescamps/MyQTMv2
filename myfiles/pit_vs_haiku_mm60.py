"""Un seul graphique : NDX daily (log) + zones de tendance, et sur un axe secondaire la MM60 des notes
de sentiment Haiku (fiches) et PIT-4B-FT-201912 (articles bruts), en z-score.
    python -m myfiles.pit_vs_haiku_mm60 [--ma 60]
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator  # noqa: E402

from myfiles.ndx_regimes import PRIMARY, SECONDARY, segments  # noqa: E402
from myfiles.ndx_sentiment_chart import GREEN, RED, START, load_ndx  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ma", type=int, default=60)
    n = ap.parse_args().ma
    h = pd.read_parquet("data/text/cnbc/sentiment/sentiment.parquet").set_index("date")["score"]
    p = pd.read_parquet("data/text/cnbc/pit/pit_201912.parquet").set_index("date")["score"]
    d = pd.concat([h.rename("haiku"), p.rename("pit")], axis=1).dropna().sort_index()
    z = (d - d.mean()) / d.std()
    mm = z.rolling(n, min_periods=n // 2).mean()
    rho = d.haiku.rolling(n).mean().corr(d.pit.rolling(n).mean(), method="spearman")

    full = load_ndx()
    c = full[full.index >= START]
    x0 = c.index[0]
    fig, ax = plt.subplots(figsize=(16, 8))
    for a0, b0, sgn, r in [s for s in segments(full, PRIMARY) if s[1] > x0]:
        ax.axvspan(max(a0, x0), b0, color=GREEN if sgn > 0 else RED, alpha=0.13, lw=0)
        mid = max(a0, x0) + (b0 - max(a0, x0)) / 2
        ax.text(mid, 0.985, f"{'↑' if sgn > 0 else '↓'} {r:+.0%}", transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=10, color=GREEN if sgn > 0 else RED, weight="bold")
    prim = [s for s in segments(full, PRIMARY) if s[1] > x0]
    for a0, b0, sgn, _ in [s for s in segments(full, SECONDARY) if s[1] > x0]:
        host = [q for q in prim if q[0] <= a0 + (b0 - a0) / 2 <= q[1]]
        if host and host[0][2] != sgn:
            ax.axvspan(max(a0, x0), b0, ymin=0, ymax=0.04, color=GREEN if sgn > 0 else RED, alpha=0.55, lw=0)
    ax.plot(c.index, c.values, color="#1f2937", lw=1.3, label="Nasdaq-100 (log)")
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(FixedLocator([10000, 12000, 14000, 16000, 18000, 20000, 22500, 25000,
                                             27500, 30000, 32500]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}".replace(",", " ")))
    ax.set_ylabel("NDX (échelle log)")
    ax.grid(alpha=0.25)

    a2 = ax.twinx()
    a2.axhline(0, color="#6b7280", lw=0.8, ls="--")
    a2.plot(mm.index, mm.haiku, color="#4f46e5", lw=2.2, label=f"Haiku 5.5 — fiches (MM{n}, z)")
    a2.plot(mm.index, mm.pit, color="#f97316", lw=2.2, label=f"PIT-4B 2019 — articles bruts (MM{n}, z)")
    lim = mm.abs().max().max() * 1.1
    a2.set_ylim(-lim * 1.9, lim * 1.15)   # courbes dans la moitié haute pour ne pas écraser le prix
    a2.set_ylabel(f"Sentiment fondamental MM{n} (z-score)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = a2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=10)
    ax.set_xlim(x0, c.index[-1])
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))
    ax.set_title(f"Nasdaq-100 et sentiment fondamental MM{n} : Haiku (coupure 06/2026) vs PIT (coupure 12/2019) — "
                 f"Spearman MM{n} = {rho:+.2f}", fontsize=11)
    out = f"myfiles/pit_vs_haiku_mm{n}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"→ {out}")


if __name__ == "__main__":
    main()
