"""NDX daily (log) + zones de tendance, puis sentiment fondamental brut et plusieurs moyennes mobiles.

Panneaux (axe des dates partagé, zones de tendance en fond partout) :
1. NDX clôture, échelle log ;
2. note brute de chaque fiche (−5…+5), en barres ;
3. moyennes mobiles de la note (MAS fiches notées).
    python myfiles/ndx_sentiment_multi.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from ndx_regimes import PRIMARY, SECONDARY, segments  # noqa: E402
from ndx_sentiment_chart import GREEN, RED, SENT, START, load_ndx  # noqa: E402

OUT = "myfiles/ndx_sentiment_multi.png"
MAS = [(5, "#a5b4fc"), (20, "#6366f1"), (60, "#312e81"), (120, "#f59e0b")]


def shade(ax, prim, sec, x0, bars=False):
    for a0, b0, sgn, _ in prim:
        ax.axvspan(max(a0, x0), b0, color=GREEN if sgn > 0 else RED, alpha=0.14, lw=0)
    if bars:
        for a0, b0, sgn, _ in sec:
            host = [p for p in prim if p[0] <= a0 + (b0 - a0) / 2 <= p[1]]
            if host and host[0][2] != sgn:
                ax.axvspan(max(a0, x0), b0, ymin=0, ymax=0.05, color=GREEN if sgn > 0 else RED, alpha=0.55, lw=0)


def main():
    full = load_ndx()
    c = full[full.index >= START]
    x0 = c.index[0]
    prim = [s for s in segments(full, PRIMARY) if s[1] > x0]
    sec = [s for s in segments(full, SECONDARY) if s[1] > x0]
    s = pd.read_parquet(SENT).set_index("date").sort_index()["score"]
    s = s[s.index >= START]

    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(16, 13), sharex=True,
                                     gridspec_kw={"height_ratios": [2.2, 1.2, 1.6], "hspace": 0.06})
    # 1. prix
    shade(a1, prim, sec, x0, bars=True)
    for a0, b0, sgn, r in prim:
        mid = max(a0, x0) + (b0 - max(a0, x0)) / 2
        a1.text(mid, 0.97, f"{'↑' if sgn > 0 else '↓'} {r:+.0%}", transform=a1.get_xaxis_transform(),
                ha="center", va="top", fontsize=10, color=GREEN if sgn > 0 else RED, weight="bold")
    a1.plot(c.index, c.values, color="#1f2937", lw=1.2)
    a1.set_yscale("log")
    a1.yaxis.set_major_locator(FixedLocator([10000, 12000, 14000, 16000, 18000, 20000, 22500, 25000,
                                             27500, 30000, 32500]))
    a1.yaxis.set_minor_locator(NullLocator())
    a1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}".replace(",", " ")))
    a1.set_ylabel("NDX (log)")
    a1.set_title("Nasdaq-100 daily et sentiment fondamental des fiches CNBC (Haiku 5.5, faits tangibles, "
                 "−5…+5) — fond : tendance primaire −20 %/+25 %", fontsize=11)

    # 2. brut
    shade(a2, prim, sec, x0)
    a2.bar(s.index, s.values, width=1.0, color=[GREEN if v > 0 else RED if v < 0 else "#9ca3af" for v in s.values],
           lw=0)
    a2.axhline(0, color="#6b7280", lw=0.7)
    a2.set_ylim(-5, 5)
    a2.set_yticks(range(-4, 5, 2))
    a2.set_ylabel("note brute\n(1 barre = 1 fiche)")

    # 3. moyennes mobiles
    shade(a3, prim, sec, x0)
    a3.axhline(0, color="#6b7280", lw=0.7, ls="--")
    for n, col in MAS:
        a3.plot(s.index, s.rolling(n, min_periods=max(3, n // 2)).mean(), color=col,
                lw=1.0 if n == 5 else 1.8, label=f"MM {n} fiches")
    a3.set_ylabel("moyennes mobiles")
    a3.legend(loc="lower left", ncol=len(MAS), fontsize=9)

    for ax in (a1, a2, a3):
        ax.grid(alpha=0.25)
    a3.set_xlim(x0, c.index[-1])
    a3.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
    a3.xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))
    fig.savefig(OUT, dpi=120, bbox_inches="tight")
    print(f"→ {OUT}")


if __name__ == "__main__":
    main()
