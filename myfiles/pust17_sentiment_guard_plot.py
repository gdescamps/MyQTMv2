"""Graphique : base déployée PUST+LQQ x1.7 vs base + garde-fou MM20 sentiment ≤ −1 (plafond PUST seul).

Panneaux : NDX/QQQ log + zones de tendance + jours coupés ; equity nette des deux variantes (base 100) ;
drawdown ; exposition effective cible ; MM20 du sentiment avec le seuil.
    python -m myfiles.pust17_sentiment_guard_plot
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from myfiles.ndx_regimes import PRIMARY, segments  # noqa: E402
from myfiles.pust17_sentiment_guard import EVAL_START  # noqa: E402
from myfiles.pust17_tweet_overlay import metrics, simulate  # noqa: E402
from src.risk_off_strategy.data import load_funding, load_macro, load_ohlc  # noqa: E402
from src.risk_off_strategy.strategy import E_MAX, compute_allocation  # noqa: E402

N, THR, CAP = 20, -1.0, 1.0
OUT = "myfiles/pust17_sentiment_guard.png"
RED, GREEN, BASE_C, GUARD_C = "#d64545", "#2e9e5b", "#6b7280", "#f97316"


def main():
    ohlc = load_ohlc("QQQ", "2000-01-01", None)
    price = ohlc["close"]
    idx = price.index
    nfci, cpi = load_macro(idx)
    fund = np.asarray(load_funding(idx), float)
    ret = price.pct_change().fillna(0).values
    alloc = compute_allocation(price.values, nfci, cpi, high=ohlc["high"].values, low=ohlc["low"].values,
                               open_=ohlc["open"].values)
    s = pd.read_parquet("data/text/cnbc/sentiment/sentiment.parquet").set_index("date").sort_index()["score"]
    mm = s.rolling(N, min_periods=N // 2).mean()
    mm = mm.reindex(idx.union(mm.index)).ffill().reindex(idx).shift(1)
    on = (mm.fillna(0) <= THR).values
    cap = np.where(on, CAP, np.nan)

    eq_b, f_b = simulate(ret, alloc, fund)
    eq_g, f_g = simulate(ret, alloc, fund, cap)
    m = idx >= pd.Timestamp(EVAL_START)
    d = idx[m]
    eb, eg = pd.Series(eq_b[m] / eq_b[m][0] * 100, d), pd.Series(eq_g[m] / eq_g[m][0] * 100, d)
    E_b = np.clip(2 * alloc, 0, E_MAX)
    E_g = np.where(on, np.minimum(E_b, CAP), E_b)
    mb, mg = metrics(eq_b, f_b, idx, EVAL_START), metrics(eq_g, f_g, idx, EVAL_START)
    lab = lambda n, x: (f"{n} — {x['%/an']:.1f} %/an, maxDD {x['maxDD %']:.1f} %, "  # noqa: E731
                        f"Sharpe {x['Sharpe']:.2f}, Calmar {x['Calmar']:.2f}, frais {x['frais %/an']:.2f} %/an")

    fig, axs = plt.subplots(5, 1, figsize=(16, 16), sharex=True,
                            gridspec_kw={"height_ratios": [1.6, 2.0, 1.0, 1.0, 1.0], "hspace": 0.07})
    a0, a1, a2, a3, a4 = axs
    prim = [x for x in segments(price, PRIMARY) if x[1] > d[0]]
    for ax in axs:
        for p0, p1, sgn, _ in prim:
            ax.axvspan(max(p0, d[0]), p1, color=GREEN if sgn > 0 else RED, alpha=0.10, lw=0)
        ax.grid(alpha=0.25)
    # jours coupés par le garde-fou
    onm = pd.Series(on[m], d)
    blk = (onm != onm.shift()).cumsum()[onm]
    for _, x in onm[onm].groupby(blk):
        for ax in axs:
            ax.axvspan(x.index[0], x.index[-1] + pd.Timedelta(days=1), color=GUARD_C, alpha=0.13, lw=0)

    a0.plot(d, price[m], color="#1f2937", lw=1.1)
    a0.set_yscale("log")
    a0.set_ylabel("QQQ (log)")
    a0.set_title(f"PUST+LQQ x1.7 net de frais, {d[0]:%d/%m/%Y} → {d[-1]:%d/%m/%Y} — base déployée vs base + "
                 f"garde-fou sentiment MM{N} ≤ {THR:g} → expo ≤ {CAP:g} (PUST seul)\n"
                 "fond vert/rouge : tendance primaire NDX ; bandes orange : jours où le garde-fou est actif",
                 fontsize=11)

    a1.plot(eb, color=BASE_C, lw=1.6, label=lab("Base (NFCI+CPI)", mb))
    a1.plot(eg, color=GUARD_C, lw=1.8, label=lab(f"Base + MM{N}", mg))
    a1.set_yscale("log")
    a1.set_ylabel("Equity nette (base 100, log)")
    a1.legend(loc="upper left", fontsize=9)
    a1.text(d[-1], eb.iloc[-1], f" {eb.iloc[-1]:.0f}", color=BASE_C, va="center", fontsize=9)
    a1.text(d[-1], eg.iloc[-1], f" {eg.iloc[-1]:.0f}", color=GUARD_C, va="center", fontsize=9)

    for e, c in ((eb, BASE_C), (eg, GUARD_C)):
        a2.fill_between(d, (e / e.cummax() - 1) * 100, 0, color=c, alpha=0.25, lw=0)
        a2.plot(d, (e / e.cummax() - 1) * 100, color=c, lw=1)
    a2.set_ylabel("Drawdown %")

    a3.step(d, E_b[m], where="post", color=BASE_C, lw=1.2, label="cible base")
    a3.step(d, E_g[m], where="post", color=GUARD_C, lw=1.2, label=f"cible base + MM{N}")
    a3.set_ylabel("Exposition cible")
    a3.set_ylim(-0.05, 1.8)
    a3.legend(loc="lower left", fontsize=9, ncol=2)

    a4.plot(d, mm[m], color="#6366f1", lw=1.4, label=f"MM{N} sentiment (dispo J+1)")
    a4.axhline(THR, color=GUARD_C, ls="--", lw=1, label=f"seuil {THR:g}")
    a4.axhline(0, color="#6b7280", lw=0.7)
    a4.set_ylabel("Sentiment")
    a4.legend(loc="lower left", fontsize=9, ncol=2)

    a4.set_xlim(d[0], d[-1])
    a4.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
    a4.xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))
    fig.savefig(OUT, dpi=110, bbox_inches="tight")
    print(f"→ {OUT}\n{lab('Base', mb)}\n{lab('Base+MM20', mg)}")


if __name__ == "__main__":
    main()
