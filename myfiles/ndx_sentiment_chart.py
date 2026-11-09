"""NDX daily (échelle log) depuis 2022 + zones de tendance + moyenne mobile du sentiment fondamental.

- Prix : ^NDX daily (yfinance).
- Zones : myfiles/ndx_regimes.py — fond rouge/vert soutenu = tendance primaire (−20 % / +25 %),
  barre basse = mouvements secondaires ≥ 10 % de sens contraire à l'intérieur.
- Sentiment : data/text/cnbc/sentiment/sentiment.parquet (cnbc_fiche_sentiment.py), note −5…+5
  par fiche (faits économiques tangibles seulement) ; points = note du jour, courbe = MM.
    python myfiles/ndx_sentiment_chart.py [--ma 20]
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import yfinance as yf  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from ndx_regimes import PRIMARY, SECONDARY, segments  # noqa: E402

START = "2022-01-01"
SENT = "data/text/cnbc/sentiment/sentiment.parquet"
OUT = "myfiles/ndx_sentiment_chart.png"
RED, GREEN = "#d64545", "#2e9e5b"


def load_ndx():
    d = yf.download("^NDX", start="2021-06-01", progress=False, auto_adjust=False)
    c = d["Close"].squeeze()
    c.index = pd.to_datetime(c.index)
    return c.dropna()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ma", type=int, default=20, help="fenêtre de la moyenne mobile (séances notées)")
    a = ap.parse_args()

    full = load_ndx()
    c = full[full.index >= START]
    prim = [s for s in segments(full, PRIMARY) if s[1] > pd.Timestamp(START)]
    sec = [s for s in segments(full, SECONDARY) if s[1] > pd.Timestamp(START)]

    fig, ax = plt.subplots(figsize=(16, 7.5))
    for a0, b0, sgn, r in prim:
        ax.axvspan(max(a0, c.index[0]), b0, color=GREEN if sgn > 0 else RED, alpha=0.16, lw=0)
        mid = max(a0, c.index[0]) + (b0 - max(a0, c.index[0])) / 2
        ax.text(mid, 0.985, f"{'↑' if sgn > 0 else '↓'} {r:+.0%}", transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=10, color=GREEN if sgn > 0 else RED, weight="bold")
    # contre-mouvements secondaires (≥ 10 %) à l'intérieur d'une zone primaire de sens opposé
    for a0, b0, sgn, r in sec:
        host = [p for p in prim if p[0] <= a0 + (b0 - a0) / 2 <= p[1]]
        if host and host[0][2] != sgn:
            ax.axvspan(max(a0, c.index[0]), b0, ymin=0, ymax=0.06, color=GREEN if sgn > 0 else RED,
                       alpha=0.55, lw=0)
    ax.plot(c.index, c.values, color="#1f2937", lw=1.3, label="Nasdaq-100 (clôture, log)")
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(FixedLocator([10000, 12000, 14000, 16000, 18000, 20000, 22500, 25000,
                                             27500, 30000, 32500]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}".replace(",", " ")))
    ax.set_ylabel("NDX (échelle log)")
    ax.set_xlim(c.index[0], c.index[-1] + pd.Timedelta(days=10).to_pytimedelta())
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.grid(alpha=0.25, which="both")

    if os.path.exists(SENT):
        s = pd.read_parquet(SENT).set_index("date").sort_index()["score"]
        s = s[s.index >= START]
        mm = s.rolling(a.ma, min_periods=max(5, a.ma // 2)).mean()
        ax2 = ax.twinx()
        ax2.axhline(0, color="#6b7280", lw=0.8, ls="--")
        ax2.scatter(s.index, s.values, s=5, color="#4f46e5", alpha=0.15, lw=0, label="note du jour (−5…+5)")
        ax2.plot(mm.index, mm.values, color="#4f46e5", lw=2, label=f"sentiment fondamental, MM {a.ma} fiches")
        lim = max(1.0, mm.abs().max() * 1.25)
        ax2.set_ylim(-lim * 2.2, lim * 1.15)   # courbe dans la moitié haute, sous les étiquettes
        ax2.set_ylabel("Sentiment fondamental (MM)", color="#4f46e5")
        ax2.tick_params(axis="y", colors="#4f46e5")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=9)
        print(f"{len(s)} notes de sentiment, MM {a.ma}")
    else:
        ax.legend(loc="lower right")
        print(f"(pas de {SENT} : graphique sans sentiment)")

    ax.set_title(f"Nasdaq-100 daily {c.index[0]:%m/%Y} → {c.index[-1]:%d/%m/%Y} — zones de tendance "
                 "(fond : primaire −20 %/+25 % ; barre basse : contre-mouvements ≥ 10 %)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT, dpi=130)
    print(f"→ {OUT}")


if __name__ == "__main__":
    main()
