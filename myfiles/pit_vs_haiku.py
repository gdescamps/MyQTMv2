"""Comparaison sentiment PIT-4B-FT-201912 (articles CNBC bruts, coupure 2019) vs Haiku 5.5 (fiches).

- corrélations de Spearman jour par jour et sur MM20/MM60, par année ;
- avance/retard : corrélation de la MM20 avec les rendements NDX passés et futurs (−20…+60 séances) ;
- graphique : NDX log + zones, puis MM20 et MM60 des deux notes standardisées (z-score).
    python -m myfiles.pit_vs_haiku
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from myfiles.ndx_regimes import PRIMARY, segments  # noqa: E402
from myfiles.ndx_sentiment_chart import GREEN, RED, START, load_ndx  # noqa: E402

OUT = "myfiles/pit_vs_haiku.png"


def z(x):
    return (x - x.mean()) / x.std()


def main():
    h = pd.read_parquet("data/text/cnbc/sentiment/sentiment.parquet").set_index("date")["score"].rename("haiku")
    p = pd.read_parquet("data/text/cnbc/pit/pit_201912.parquet").set_index("date")
    d = pd.concat([h, p.score.rename("pit"), p.p_up, p.p_dn], axis=1).dropna().sort_index()
    print(f"{len(d)} séances communes ; PIT p_up moy {d.p_up.mean():.3f}, p_dn moy {d.p_dn.mean():.3f}, "
          f"score moy {d.pit.mean():+.3f} ± {d.pit.std():.3f}")
    print("\nSpearman Haiku vs PIT :")
    for n in (1, 5, 20, 60):
        a, b = d.haiku.rolling(n).mean(), d.pit.rolling(n).mean()
        print(f"  MM{n:<3d} tout : {a.corr(b, method='spearman'):+.2f}   "
              + "  ".join(f"{y}: {a[a.index.year == y].corr(b[b.index.year == y], method='spearman'):+.2f}"
                          for y in range(2022, 2027)))
    # avance / retard vs NDX
    ndx = load_ndx()
    lr = np.log(ndx)
    print("\nSpearman MM20 vs rendement NDX (passé : J−k→J ; futur : J+1→J+1+k, note dispo J+1) :")
    for name in ("haiku", "pit"):
        mm = d[name].rolling(20, min_periods=10).mean().reindex(ndx.index).ffill()
        out = []
        for k in (20, 60):
            past = lr - lr.shift(k)
            fut = lr.shift(-(k + 1)) - lr.shift(-1)
            m = mm.index >= START
            out.append(f"passé {k}j {mm[m].corr(past[m], method='spearman'):+.2f}  "
                       f"futur {k}j {mm[m].corr(fut[m], method='spearman'):+.2f}")
        print(f"  {name:5s} : " + " | ".join(out))

    c = ndx[ndx.index >= START]
    prim = [s for s in segments(ndx, PRIMARY) if s[1] > c.index[0]]
    fig, axs = plt.subplots(3, 1, figsize=(16, 11), sharex=True, gridspec_kw={"height_ratios": [1.6, 1, 1], "hspace": 0.06})
    for ax in axs:
        for a0, b0, sgn, _ in prim:
            ax.axvspan(max(a0, c.index[0]), b0, color=GREEN if sgn > 0 else RED, alpha=0.12, lw=0)
        ax.grid(alpha=0.25)
    axs[0].plot(c, color="#1f2937", lw=1.1)
    axs[0].set_yscale("log")
    axs[0].set_ylabel("NDX (log)")
    axs[0].set_title("Sentiment fondamental : Haiku 5.5 (fiches, coupure 06/2026) vs PIT-4B-FT-201912 "
                     "(articles CNBC bruts, coupure 12/2019) — z-scores", fontsize=11)
    for ax, n in ((axs[1], 20), (axs[2], 60)):
        ax.plot(d.index, z(d.haiku).rolling(n, min_periods=n // 2).mean(), color="#4f46e5", lw=1.6, label="Haiku")
        ax.plot(d.index, z(d.pit).rolling(n, min_periods=n // 2).mean(), color="#f97316", lw=1.6, label="PIT 2019")
        ax.axhline(0, color="#6b7280", lw=0.7)
        ax.set_ylabel(f"MM{n} (z)")
        ax.legend(loc="lower left", fontsize=9, ncol=2)
    axs[2].set_xlim(c.index[0], c.index[-1])
    axs[2].xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
    axs[2].xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))
    fig.savefig(OUT, dpi=110, bbox_inches="tight")
    print(f"\n→ {OUT}")


if __name__ == "__main__":
    main()
