"""
Backtest + graphiques de la strategie deployee (trend250 + vol-managed + garde-fous).

exec_lag=1, historique depuis 2000. Le tableau + la courbe equity n'affichent que
DEUX lignes : le B&H (index) et la strategie DEPLOYEE (PUST + LQQ, exposition
plafonnee a E_MAX) NET DE FRAIS via simulate_net (TER PUST/LQQ, financement du
levier, vente 0.5% au seuil live). La courbe brute de la meme strategie reste en
pointille pour visualiser le cout des frais ; funding=None (taux court FRED
indispo) -> repli sur le brut.
  plot_backtest(..., last_days=None) : chart 6 panneaux (equity+SMA250, vol,
  allocation, NFCI, inflation, CAPE/ECY S&P Shiller), bandes rouges = top-5 crises.
  last_days=252 / 21 -> meme mise en page, fenetree sur la derniere annee / mois
  (indicateurs calcules sur tout l'historique pour le warmup, puis fenetres ;
  equity rebasee au debut de la fenetre, metriques recalculees sur la fenetre).
"""
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from src.risk_off_strategy.strategy import (
    realized_vol, simulate_net, SMA_LONG, VOL_TARGET, NFCI_OFF, CPI_OFF, ANN,
    ABOVE_CAP, BELOW_SCALE, GAP_CUTOFF, GAP2_START, GAP2_SPAN, DECAY2_FLOOR,
    SLOPE_K, TER_PUST, TER_LQQ, SWAP_SPREAD, SELL_FEE, BUY_THR_ALLOC, SELL_THR_ALLOC,
    E_MAX,
)


def _formula_text():
    """Formule d'allocation (auto-synchronisee sur les constantes de strategy.py)."""
    return (
        f"alloc x1 :   NFCI > {NFCI_OFF}  ou  IPC YoY > {CPI_OFF:.0f}%   ->   0        (garde-fous macro ; sinon :)\n"
        f"   close > SMA{SMA_LONG} et SMA{SMA_LONG} montante ({SLOPE_K}j)   ->   min({ABOVE_CAP:.2f} / vol, 1) . decay_up   [risk-off precoce]\n"
        f"   sinon (sous la MA ou MA plate/baissiere)   ->   {BELOW_SCALE:.1f} . min({VOL_TARGET:.2f} / vol, 1) . decay_down\n"
        f"vol = Yang-Zhang(OHLC, 10j) ann.        gap = close / SMA{SMA_LONG} - 1\n"
        f"decay_up = clip(1 - max(0, gap - {GAP2_START:.2f}) / {GAP2_SPAN:.2f}, {DECAY2_FLOOR:.1f}, 1)        "
        f"decay_down = clip(1 + gap / {GAP_CUTOFF:.2f}, 0, 1)\n"
        f"deployee x{E_MAX} = min(2 . alloc x1, {E_MAX}) via PUST + LQQ (LQQ = part > 100%)     "
        f"exec_lag = 1 : close du soir  ->  execution J+1\n"
        f"NET de frais : TER PUST {TER_PUST*100:.2f}% / LQQ {TER_LQQ*100:.2f}% + financement LQQ (taux court +{SWAP_SPREAD*100:.1f}%) ; "
        f"bande asym. .levier : achat si +{BUY_THR_ALLOC:.2f} (libre), vente {SELL_FEE*100:.1f}% si -{SELL_THR_ALLOC:.2f}"
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


def _metrics_from_equity(eq, w0):
    """Rebase une equity pleine a la fenetre [w0:] + metriques (net de frais)."""
    e = eq[w0:] / eq[w0]
    yrs = len(e) / ANN
    peak = np.maximum.accumulate(e)
    dd = ((e - peak) / peak).min()
    cagr = e[-1] ** (1 / yrs) - 1
    r = e[1:] / e[:-1] - 1
    sh = r.mean() / r.std() * np.sqrt(ANN) if len(r) and r.std() > 0 else 0.0
    return e, cagr, dd, sh


def plot_backtest(price, alloc, nfci=None, cpi=None, cape=None, ecy=None, ndx_ey=None,
                  save_path=None, ticker="QQQ", last_days=None, vol=None, above_cap=None,
                  funding=None):
    dates = price.index
    p = price.values
    ret = price.pct_change().fillna(0).values
    rv = np.asarray(vol, float) if vol is not None else realized_vol(ret)
    s = _sma(p, SMA_LONG)
    n = len(p)
    alloc = np.asarray(alloc, float)

    w0 = 0 if last_days is None else max(0, n - last_days)
    d = dates[w0:]

    # B&H (index) = reference
    pos_bh = np.ones(n)
    ebh, cagr_bh, dd_bh, sh_bh = _window_metrics(ret, pos_bh, w0)

    # ── strategie DEPLOYEE : PUST + LQQ, exposition E = min(2.alloc, E_MAX) ──
    # BRUT : rendement quotidien = E * ret, rebalance quotidien, sans frais ni
    # financement (le decay de levier emerge du compounding) -> reference optimiste,
    # tracee en pointille pour visualiser le cout des frais.
    expo_d = np.minimum(alloc * 2.0, E_MAX)    # expo cible (signal du soir)
    pos_d = np.concatenate([[0.0], expo_d[:-1]])   # positions (exec_lag=1)
    edg, cagrdg, dddg, shdg = _window_metrics(ret, pos_d, w0)
    # NET DE FRAIS (execution discretisee Bourso : TER PUST/LQQ, financement du
    # levier au taux court, bande asymetrique, vente 0.5%). funding=None -> pas de net.
    netd = simulate_net(price, alloc, leverage=2, funding=funding, e_max=E_MAX)
    if netd is not None:
        eqdn, feesd, revd = netd
        edn, cagrdn, dddn, shdn = _metrics_from_equity(eqdn, w0)
    lbl_d = f"x{E_MAX} PUST+LQQ"

    # SMA250 rebasee sur le PRIX au debut de fenetre (meme base que le B&H rebasee),
    # sinon elle est mal positionnee dans les vues fenetrees (1y/1m).
    sma_reb = (s / p[w0])[w0:]

    span = (f"depuis {d[0].date()}" if last_days is None
            else f"{last_days}j : {d[0].date()} -> {d[-1].date()}")

    # Titre (suptitle) + formule + tableau en haut, panneaux data en dessous
    npan = 3 + (nfci is not None) + (cpi is not None) + (cape is not None)
    ratios = [2.6, 1.1, 1.1] + [1.1] * (npan - 3)
    fig = plt.figure(figsize=(15, 2.4 * npan + 5.5))
    fig.suptitle(f"{ticker} — resultats des strategies  ({span})",
                 fontsize=15, weight="bold", y=0.985)
    fig.text(0.5, 0.962, _formula_text(), ha="center", va="top", fontsize=10.5,
             family="monospace", linespacing=1.45,
             bbox=dict(boxstyle="round", fc="#f5f5f5", ec="#bbbbbb", alpha=0.95))
    gs = fig.add_gridspec(npan + 1, 1, height_ratios=[1.0] + ratios, hspace=0.18)
    ax_tbl = fig.add_subplot(gs[0])
    a1 = fig.add_subplot(gs[1])
    axes = [a1] + [fig.add_subplot(gs[i], sharex=a1) for i in range(2, npan + 1)]
    ax = iter(axes[1:])

    # ── panneau resultats (grand tableau) ──
    # CAGR/Calmar n'ont de sens qu'annualises sur >= 1 an ; sur une fenetre courte
    # (1 mois) l'annualisation "e**(252/21)" explose -> on les masque.
    annualize = last_days is None or last_days >= ANN

    def _row(name, e, cagr, dd, sh, fees=None, rev=None):
        cagr_s = f"{cagr*100:+.1f}%" if annualize else "-"
        cal_s = (f"{cagr/abs(dd):.2f}" if dd < 0 else "-") if annualize else "-"
        fees_s = "-" if fees is None else f"{fees:.2f}%"
        rev_s = "-" if rev is None else f"{rev:.0f}"
        return [name, cagr_s, f"{(e[-1]-1)*100:+.0f}%",
                f"{dd*100:.0f}%", f"{sh:.2f}", cal_s, fees_s, rev_s]

    # deux lignes seulement : B&H (reference) + strategie deployee NET de frais
    # (repli sur le brut si le taux court FRED est indisponible).
    table_rows = [_row("B&H (index)", ebh, cagr_bh, dd_bh, sh_bh)]
    row_colors = ["black"]
    if netd is not None:
        table_rows.append(_row(f"{lbl_d} (net)", edn, cagrdn, dddn, shdn, feesd, revd))
    else:
        table_rows.append(_row(f"{lbl_d} (brut)", edg, cagrdg, dddg, shdg))
    row_colors.append("darkorange")
    ax_tbl.axis("off")
    tbl = ax_tbl.table(cellText=table_rows,
                       colLabels=["strategie", "CAGR", "rendement total", "maxDD", "Sharpe",
                                  "Calmar", "frais/an", "revis/an"],
                       loc="lower center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(12); tbl.scale(1, 2.4)
    tbl.auto_set_column_width([0])
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if r == 0:
            cell.set_text_props(weight="bold"); cell.set_facecolor("#e8e8e8")
        elif c == 0:
            cell.set_text_props(weight="bold", color=row_colors[r - 1])

    # ── P1 equity : B&H, SMA250, strategie deployee NET (plein) + brut (pointille = cout des frais) ──
    a1.semilogy(d, ebh, color="black", lw=1.0, label="B&H (index)")
    a1.semilogy(d, sma_reb, color="green", lw=0.9, alpha=0.8, label=f"SMA{SMA_LONG}")
    if netd is not None:
        a1.semilogy(d, edg, color="darkorange", lw=0.8, ls=":", alpha=0.5, label=f"{lbl_d} brut")
        a1.semilogy(d, edn, color="darkorange", lw=1.6, label=f"strategie {lbl_d} (net, deployee)")
    else:
        a1.semilogy(d, edg, color="darkorange", lw=1.6, label=f"strategie {lbl_d} (brut, deployee)")
    a1.set_ylabel("Equity (log, base 1)")
    a1.legend(loc="upper left", fontsize=8, ncol=2); a1.grid(True, which="both", alpha=0.2)

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
    a3.fill_between(d, alloc[w0:], color="steelblue", alpha=0.35, step="mid", label="allocation x1 (signal)")
    a3.plot(d, expo_d[w0:], color="darkorange", lw=0.9, alpha=0.95,
            label=f"expo deployee = min(2 . alloc, {E_MAX}) via PUST + LQQ")
    a3.axhline(1.0, color="grey", ls=":", lw=0.6)
    a3.axhline(E_MAX, color="darkorange", ls=":", lw=0.6, alpha=0.7)
    a3.set_ylabel("allocation / expo", fontsize=9); a3.set_ylim(-0.05, E_MAX + 0.2); a3.grid(True, alpha=0.2)
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
    if cape is not None:
        ap = next(ax)
        cape_w = np.asarray(cape, float)[w0:]
        ap.semilogy(d, cape_w, color="teal", lw=0.9, label="CAPE S&P (Shiller P/E10)")
        for lvl in (16, 30, 44):
            ap.axhline(lvl, color="grey", ls=":", lw=0.7, alpha=0.6)
        ap.set_ylabel("CAPE (log)", fontsize=9, color="teal")
        ap.tick_params(axis="y", labelcolor="teal")
        ap.grid(True, which="both", alpha=0.2)
        ap.legend(loc="upper left", fontsize=8)
        if ecy is not None:
            ae = ap.twinx()
            ecy_w = np.asarray(ecy, float)[w0:] * 100
            ae.plot(d, ecy_w, color="darkorange", lw=0.9, label="Excess CAPE Yield (%)")
            ae.axhline(0, color="darkorange", ls="--", lw=0.7, alpha=0.5)
            ae.set_ylabel("ECY (%)", fontsize=9, color="darkorange")
            ae.tick_params(axis="y", labelcolor="darkorange")
            # leaders Nasdaq AUJOURD'HUI (snapshot) : lignes de reference sur la meme
            # echelle "rendement - taux reel" que l'ECY S&P, pour situer le top-5 du jour.
            if ndx_ey:
                t5 = ndx_ey.get("top5", {})
                et, ef = t5.get("excess_trailing"), t5.get("excess_forward")
                if et is not None:
                    ae.axhline(et * 100, color="steelblue", ls=":", lw=1.1, alpha=0.85,
                               label=f"NDX top-5 excess trailing auj. ({et*100:+.1f}%)")
                if ef is not None:
                    ae.axhline(ef * 100, color="green", ls=":", lw=1.1, alpha=0.85,
                               label=f"NDX top-5 excess forward auj. ({ef*100:+.1f}%)")
            ae.legend(loc="lower left", fontsize=7)

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

    with warnings.catch_warnings():
        # le twinx du panneau CAPE/ECY n'est pas compatible tight_layout (rendu OK)
        warnings.simplefilter("ignore", UserWarning)
        fig.tight_layout(rect=[0, 0, 1, 0.85])
    if save_path:
        fig.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.close(fig)

    # metriques plein-echantillon (pour le log de run.py) quand chart complet :
    # strategie deployee nette si dispo (ce que le compte encaisse reellement), sinon brute.
    if last_days is None:
        if netd is not None:
            _, cf, df_, shf = _metrics_from_equity(eqdn, 0)
            return {"cagr": cf, "maxdd": df_, "sharpe": shf,
                    "calmar": cf / abs(df_) if df_ < 0 else np.inf,
                    "fees_yr": feesd, "revis_yr": revd, "net": True}
        _, cagr_f, dd_f, sh_f = _window_metrics(ret, pos_d, 0)
        return {"cagr": cagr_f, "maxdd": dd_f, "sharpe": sh_f,
                "calmar": cagr_f / abs(dd_f) if dd_f < 0 else np.inf, "net": False}
    return None
