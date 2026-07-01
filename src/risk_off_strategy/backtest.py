"""
Backtest + graphiques de la strategie deployee (trend250 + vol-managed + garde-fous).

x1, close-to-close, sans frais, exec_lag=1. Historique depuis 2000.
  plot_backtest(..., last_days=None) : chart 5 panneaux (equity+SMA250, vol,
  allocation, NFCI, inflation), bandes rouges = bears lents.
  last_days=252 / 21 -> meme mise en page, fenetree sur la derniere annee / mois
  (indicateurs calcules sur tout l'historique pour le warmup, puis fenetres ;
  equity rebasee au debut de la fenetre, metriques recalculees sur la fenetre).
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from src.risk_off_strategy.strategy import (
    realized_vol, SMA_LONG, VOL_TARGET, NFCI_OFF, CPI_OFF, ANN,
)


def _sma(a, w):
    return pd.Series(a).rolling(w, min_periods=1).mean().values


def swing_slow(price, dates, thr=0.25):
    """Bears lents = swings pic->creux >= thr, duree >= 100j, profondeur <= -15%."""
    p = np.asarray(price, float)
    eps, hi, lo, mode = [], 0, 0, "up"
    for i in range(1, len(p)):
        if mode == "up":
            if p[i] > p[hi]:
                hi = i
            elif p[i] <= p[hi] * (1 - thr):
                lo, mode = i, "down"
        else:
            if p[i] < p[lo]:
                lo = i
            elif p[i] >= p[lo] * (1 + thr):
                eps.append((hi, lo)); hi, mode = i, "up"
    if mode == "down":
        eps.append((hi, lo))
    return [(a, b) for a, b in eps
            if (dates[b] - dates[a]).days >= 100 and p[b] / p[a] - 1 <= -0.15]


def _window_metrics(ret, pos, w0):
    """Equity (full) rebasee a la fenetre + metriques sur la fenetre [w0:]."""
    r = ret * pos
    eq = np.cumprod(1 + r)
    e = eq[w0:] / eq[w0]
    yrs = len(e) / ANN
    peak = np.maximum.accumulate(e)
    dd = ((e - peak) / peak).min()
    cagr = e[-1] ** (1 / yrs) - 1
    rr = r[w0:]
    sh = rr.mean() / rr.std() * np.sqrt(ANN) if rr.std() > 0 else 0.0
    return e, cagr, dd, sh


def plot_backtest(price, alloc, nfci=None, cpi=None, save_path=None, ticker="QQQ",
                  last_days=None):
    dates = price.index
    p = price.values
    ret = price.pct_change().fillna(0).values
    rv = realized_vol(ret)
    s = _sma(p, SMA_LONG)
    n = len(p)
    alloc = np.asarray(alloc, float)

    w0 = 0 if last_days is None else max(0, n - last_days)
    d = dates[w0:]

    # positions (exec_lag=1)
    pos_m = np.concatenate([[0.0], alloc[:-1]])
    pos_bh = np.ones(n)
    alloc_ref = np.where(p > _sma(p, 150), 1.0, np.clip(0.12 / rv, 0, 1))
    pos_ref = np.concatenate([[0.0], alloc_ref[:-1]])

    em, cagr_m, dd_m, sh_m = _window_metrics(ret, pos_m, w0)
    ebh, cagr_bh, dd_bh, sh_bh = _window_metrics(ret, pos_bh, w0)
    eref, _, dd_ref, sh_ref = _window_metrics(ret, pos_ref, w0)
    sma_reb = (s / s[w0])[w0:]

    npan = 3 + (nfci is not None) + (cpi is not None)
    ratios = [2.6, 1.1, 1.1] + [1.1] * (npan - 3)
    fig, axes = plt.subplots(npan, 1, figsize=(15, 2.4 * npan + 3), sharex=True,
                             gridspec_kw={"height_ratios": ratios})
    ax = iter(axes)

    a1 = next(ax)
    a1.semilogy(d, ebh, color="black", lw=1.0,
                label=f"B&H (CAGR {cagr_bh*100:.1f}%, DD {dd_bh*100:.0f}%, Sh {sh_bh:.2f})")
    a1.semilogy(d, sma_reb, color="green", lw=0.9, alpha=0.8, label=f"SMA{SMA_LONG} (rebase)")
    a1.semilogy(d, eref, color="grey", lw=1.0, ls="--",
                label=f"trend150+VM ref (Sh {sh_ref:.2f}, DD {dd_ref*100:.0f}%)")
    a1.semilogy(d, em, color="crimson", lw=1.5,
                label=f"strategie (CAGR {cagr_m*100:.1f}%, DD {dd_m*100:.0f}%, Sh {sh_m:.2f})")
    a1.set_ylabel("Equity (log, base 1)")
    a1.legend(loc="upper left", fontsize=9); a1.grid(True, which="both", alpha=0.2)
    span = (f"depuis {d[0].date()}" if last_days is None
            else f"{last_days}j : {d[0].date()} -> {d[-1].date()}")
    a1.set_title(f"{ticker} — strategie deployee (trend{SMA_LONG} + vol-managed + garde-fous macro)  {span}")

    a2 = next(ax)
    a2.plot(d, rv[w0:] * 100, color="purple", lw=0.7, label="vol realisee 20j (ann.)")
    a2.axhline(VOL_TARGET * 100, color="grey", ls="--", lw=0.9, label=f"V = {VOL_TARGET*100:.0f}%")
    a2.set_ylabel("vol (%)", fontsize=9)
    a2.set_ylim(0, max(rv[w0:].max() * 105, 30))
    a2.legend(loc="upper left", fontsize=8); a2.grid(True, alpha=0.2)

    a3 = next(ax)
    a3.fill_between(d, alloc[w0:], color="steelblue", alpha=0.35, step="mid")
    a3.plot(d, alloc[w0:], color="steelblue", lw=0.5)
    a3.set_ylabel("allocation", fontsize=9); a3.set_ylim(-0.05, 1.05); a3.grid(True, alpha=0.2)

    if nfci is not None:
        an = next(ax)
        an.plot(d, np.asarray(nfci, float)[w0:], color="firebrick", lw=0.8, label="NFCI")
        an.axhline(0, color="grey", ls=":", lw=0.7, alpha=0.7)
        an.axhline(NFCI_OFF, color="red", ls="--", lw=0.9, alpha=0.8, label=f"OFF > {NFCI_OFF}")
        an.set_ylabel("NFCI", fontsize=9); an.legend(loc="upper left", fontsize=8); an.grid(True, alpha=0.2)
    if cpi is not None:
        ac = next(ax)
        ac.plot(d, np.asarray(cpi, float)[w0:], color="darkgreen", lw=0.8, label="IPC YoY (%)")
        ac.axhline(2, color="grey", ls=":", lw=0.7, alpha=0.7)
        ac.axhline(CPI_OFF, color="red", ls="--", lw=0.9, alpha=0.8, label=f"OFF > {CPI_OFF:.0f}%")
        ac.set_ylabel("inflation (%)", fontsize=9); ac.legend(loc="upper left", fontsize=8); ac.grid(True, alpha=0.2)

    if last_days is None:
        for axx in axes:
            for a, b in swing_slow(p, dates):
                axx.axvspan(dates[a], dates[b], color="red", alpha=0.10, lw=0)
        axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    else:
        axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator())
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d" if last_days <= 21 else "%Y-%m"))
        for lbl in axes[-1].get_xticklabels():
            lbl.set_rotation(30); lbl.set_ha("right")
    axes[-1].set_xlabel("Date")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.close(fig)

    # metriques plein-echantillon (pour le log de run.py) quand chart complet
    if last_days is None:
        _, cagr_f, dd_f, sh_f = _window_metrics(ret, pos_m, 0)
        yrs = n / ANN
        peak = np.maximum.accumulate(np.cumprod(1 + ret * pos_m))
        return {"cagr": cagr_f, "maxdd": dd_f, "sharpe": sh_f,
                "calmar": cagr_f / abs(dd_f) if dd_f < 0 else np.inf}
    return None
