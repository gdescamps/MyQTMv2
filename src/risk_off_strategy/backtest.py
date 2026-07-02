"""
Backtest + graphiques de la strategie deployee (trend250 + vol-managed + garde-fous).

x1, close-to-close, sans frais, exec_lag=1. Historique depuis 2000.
  plot_backtest(..., last_days=None) : chart 5 panneaux (equity+SMA250, vol,
  allocation, NFCI, inflation), bandes rouges = top-5 crises (bears les + profonds).
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
    ABOVE_CAP, BELOW_SCALE, GAP_CUTOFF, GAP2_START, GAP2_SPAN, DECAY2_FLOOR,
)


def _formula_text():
    """Formule d'allocation (auto-synchronisee sur les constantes de strategy.py)."""
    return (
        f"alloc x1 = 0  si  NFCI > {NFCI_OFF} ou IPC YoY > {CPI_OFF:.0f}%      sinon :   "
        f"close > SMA{SMA_LONG}  ->  min({ABOVE_CAP:.2f} / vol, 1) . decay_up      "
        f"close <= SMA{SMA_LONG}  ->  {BELOW_SCALE:.1f} . min({VOL_TARGET:.2f} / vol, 1) . decay_down\n"
        f"vol = Yang-Zhang(OHLC, 10j) annualisee        gap = close / SMA{SMA_LONG} - 1        "
        f"decay_up = clip(1 - max(0, gap - {GAP2_START:.2f}) / {GAP2_SPAN:.2f}, {DECAY2_FLOOR:.1f}, 1)        "
        f"decay_down = clip(1 + gap / {GAP_CUTOFF:.2f}, 0, 1)\n"
        f"x2.0 = 2 . alloc x1 (via LQQ)        exec_lag = 1 : decision au close du soir, execution le lendemain matin"
    )


def _sma(a, w):
    return pd.Series(a).rolling(w, min_periods=1).mean().values


def top_crises(price, dates, k=5, drop=0.15, rebound=0.20, merge_gap=200):
    """Les k pires crises = drawdowns pic-local -> creux, fusionnes en episodes.

    Detecte les swings de baisse (>= `drop` depuis un pic local, clotures par un
    rebond >= `rebound` depuis le creux), fusionne les jambes proches (< `merge_gap`
    jours) en une seule crise -- sinon le dot-com, en dents de scie, monopolise le
    classement -- puis garde les k plus profondes. Attrape les krachs lents
    (dot-com, 2008) comme rapides (Covid). Renvoie une liste de (pic, creux).
    """
    p = np.asarray(price, float)
    swings, peak, trough, start, in_ep = [], 0, 0, 0, False
    for i in range(1, len(p)):
        if not in_ep:
            if p[i] >= p[peak]:
                peak = i
            elif p[i] <= p[peak] * (1 - drop):
                in_ep, start, trough = True, peak, i
        else:
            if p[i] < p[trough]:
                trough = i
            elif p[i] >= p[trough] * (1 + rebound):
                swings.append((start, trough)); peak, in_ep = i, False
    if in_ep:
        swings.append((start, trough))
    if not swings:
        return []
    # fusion des jambes contigues (les multiples jambes du dot-com -> une crise)
    merged = [list(swings[0])]
    for a, b in swings[1:]:
        if (dates[a] - dates[merged[-1][1]]).days < merge_gap:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    crises = [(a, a + int(np.argmin(p[a:b + 1]))) for a, b in merged]  # (pic, plus-bas)
    crises.sort(key=lambda ab: p[ab[1]] / p[ab[0]])
    return crises[:k]


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


def plot_backtest(price, alloc, nfci=None, cpi=None, pe=None, save_path=None, ticker="QQQ",
                  last_days=None, vol=None, above_cap=None):
    dates = price.index
    p = price.values
    ret = price.pct_change().fillna(0).values
    rv = np.asarray(vol, float) if vol is not None else realized_vol(ret)
    s = _sma(p, SMA_LONG)
    n = len(p)
    alloc = np.asarray(alloc, float)

    w0 = 0 if last_days is None else max(0, n - last_days)
    d = dates[w0:]

    # positions (exec_lag=1)
    pos_m = np.concatenate([[0.0], alloc[:-1]])
    pos_bh = np.ones(n)
    em, cagr_m, dd_m, sh_m = _window_metrics(ret, pos_m, w0)
    ebh, cagr_bh, dd_bh, sh_bh = _window_metrics(ret, pos_bh, w0)

    # variante LEVIER x2.0 : on detient LQQ (Nasdaq x2) dimensionne par l'alloc x1
    # (risk-off precoce inclus) -> multiplicateur PLAT 2 sur l'expo. Rendement
    # quotidien = 2 * alloc * ret : le decay de levier emerge du compounding
    # quotidien. Rebalance quotidien, SANS frais ni cout de financement -> optimiste
    # pour le levier (LQQ reel : ~0.6%/an de frais + portage du financement x2).
    def _lever(L):
        pos = np.concatenate([[0.0], alloc[:-1]])
        e, cg, dd, sh = _window_metrics(ret * L, pos, w0)
        return alloc * L, e, cg, dd, sh

    lev20, e20, cagr20, dd20, sh20 = _lever(2.0)

    # SMA250 rebasee sur le PRIX au debut de fenetre (meme base que le B&H rebasee),
    # sinon elle est mal positionnee dans les vues fenetrees (1y/1m).
    sma_reb = (s / p[w0])[w0:]

    span = (f"depuis {d[0].date()}" if last_days is None
            else f"{last_days}j : {d[0].date()} -> {d[-1].date()}")

    # Table des resultats en haut + panneaux data en dessous (via gridspec)
    npan = 3 + (nfci is not None) + (cpi is not None) + (pe is not None)
    ratios = [2.6, 1.1, 1.1] + [1.1] * (npan - 3)
    fig = plt.figure(figsize=(15, 2.4 * npan + 5))
    gs = fig.add_gridspec(npan + 1, 1, height_ratios=[1.9] + ratios)
    ax_tbl = fig.add_subplot(gs[0])
    a1 = fig.add_subplot(gs[1])
    axes = [a1] + [fig.add_subplot(gs[i], sharex=a1) for i in range(2, npan + 1)]
    ax = iter(axes[1:])

    # ── panneau resultats (grand tableau) ──
    # CAGR/Calmar n'ont de sens qu'annualises sur >= 1 an ; sur une fenetre courte
    # (1 mois) l'annualisation "e**(252/21)" explose -> on les masque.
    annualize = last_days is None or last_days >= ANN

    def _row(name, e, cagr, dd, sh):
        cagr_s = f"{cagr*100:+.1f}%" if annualize else "-"
        cal_s = (f"{cagr/abs(dd):.2f}" if dd < 0 else "-") if annualize else "-"
        return [name, cagr_s, f"{(e[-1]-1)*100:+.0f}%",
                f"{dd*100:.0f}%", f"{sh:.2f}", cal_s]
    table_rows = [
        _row("B&H", ebh, cagr_bh, dd_bh, sh_bh),
        _row("strategie x1", em, cagr_m, dd_m, sh_m),
        _row("strategie x2.0 (LQQ)", e20, cagr20, dd20, sh20),
    ]
    row_colors = ["black", "crimson", "purple"]
    ax_tbl.axis("off")
    ax_tbl.set_title(f"{ticker} — resultats des strategies  ({span})", fontsize=14, weight="bold", pad=12)
    tbl = ax_tbl.table(cellText=table_rows,
                       colLabels=["strategie", "CAGR", "rendement total", "maxDD", "Sharpe", "Calmar"],
                       loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(13); tbl.scale(1, 2.4)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if r == 0:
            cell.set_text_props(weight="bold"); cell.set_facecolor("#e8e8e8")
        elif c == 0:
            cell.set_text_props(weight="bold", color=row_colors[r - 1])

    # ── P1 equity : SMA250, B&H, x1, x2.0 ──
    a1.semilogy(d, ebh, color="black", lw=1.0, label="B&H")
    a1.semilogy(d, sma_reb, color="green", lw=0.9, alpha=0.8, label=f"SMA{SMA_LONG}")
    a1.semilogy(d, em, color="crimson", lw=1.5, label="strategie x1")
    a1.semilogy(d, e20, color="purple", lw=1.2, label="strategie x2.0 (LQQ)")
    a1.set_ylabel("Equity (log, base 1)")
    a1.legend(loc="upper left", fontsize=9, ncol=2); a1.grid(True, which="both", alpha=0.2)

    a2 = next(ax)
    vol_lbl = "vol Yang-Zhang 10j (ann.)" if vol is not None else "vol realisee 20j (ann.)"
    a2.plot(d, rv[w0:] * 100, color="purple", lw=0.7, label=vol_lbl)
    a2.axhline(VOL_TARGET * 100, color="grey", ls="--", lw=0.9, label=f"V sous MA = {VOL_TARGET*100:.0f}%")
    if above_cap is not None:
        a2.axhline(above_cap * 100, color="red", ls="--", lw=0.9, label=f"cap sur MA = {above_cap*100:.0f}%")
    a2.set_ylabel("vol (%)", fontsize=9)
    a2.set_ylim(0, max(rv[w0:].max() * 105, 30))
    a2.legend(loc="upper left", fontsize=8); a2.grid(True, alpha=0.2)

    a3 = next(ax)
    a3.fill_between(d, alloc[w0:], color="steelblue", alpha=0.35, step="mid", label="allocation x1")
    a3.plot(d, lev20[w0:], color="purple", lw=0.7, alpha=0.9, label="expo x2.0 (LQQ)")
    a3.axhline(1.0, color="grey", ls=":", lw=0.6)
    a3.axhline(2.0, color="purple", ls=":", lw=0.5, alpha=0.5)
    a3.set_ylabel("allocation / expo", fontsize=9); a3.set_ylim(-0.05, 2.15); a3.grid(True, alpha=0.2)
    a3.legend(loc="upper left", fontsize=7, ncol=2)

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
    if pe is not None:
        ap = next(ax)
        pe_w = np.asarray(pe, float)[w0:]
        ap.semilogy(d, pe_w, color="teal", lw=0.8, label="PE cap-weighted top-5 NDX")
        for lvl in (20, 40):
            ap.axhline(lvl, color="grey", ls=":", lw=0.7, alpha=0.6)
        ap.set_ylabel("PE (log)", fontsize=9); ap.legend(loc="upper left", fontsize=8)
        ap.grid(True, which="both", alpha=0.2)

    # top-5 crises = les 5 drawdowns les plus profonds (pic-local -> creux) sur
    # tout l'historique. Memes bandes rouges verticales sur toutes les vues,
    # tronquees a la fenetre affichee (sinon une crise hors fenetre etirerait
    # l'axe des vues 1y/1m).
    xmin, xmax = d[0], d[-1]
    for a, b in top_crises(p, dates):
        ca, cb = dates[a], dates[b]
        if cb < xmin or ca > xmax:
            continue
        for axx in axes:
            axx.axvspan(max(ca, xmin), min(cb, xmax), color="red", alpha=0.10, lw=0)

    if last_days is None:
        axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    else:
        axes[-1].set_xlim(xmin, xmax)
        axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator())
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d" if last_days <= 21 else "%Y-%m"))
        for lbl in axes[-1].get_xticklabels():
            lbl.set_rotation(30); lbl.set_ha("right")
    axes[-1].set_xlabel("Date")

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.text(0.5, 0.008, _formula_text(), ha="center", va="bottom", fontsize=8,
             family="monospace", linespacing=1.6,
             bbox=dict(boxstyle="round", fc="#f5f5f5", ec="#cccccc", alpha=0.95))
    if save_path:
        fig.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.close(fig)

    # metriques plein-echantillon (pour le log de run.py) quand chart complet
    if last_days is None:
        _, cagr_f, dd_f, sh_f = _window_metrics(ret, pos_m, 0)
        return {"cagr": cagr_f, "maxdd": dd_f, "sharpe": sh_f,
                "calmar": cagr_f / abs(dd_f) if dd_f < 0 else np.inf}
    return None
