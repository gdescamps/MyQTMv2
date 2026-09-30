"""
LQQ+cash x1.7 vs PUST+LQQ x1.7 avec les frais d'achat 0.5% (prompts.txt 2026-09-28).

Question : maintenant que les achats coutent 0.5% (constate le 28/09), la realisation
"tout LQQ + cash" (poids w = E/2 en LQQ, reste en cash) est-elle toujours moins bonne
que la realisation drag-minimale PUST + LQQ deployee ? On decompose : frais de
transaction, portage (financement DFF + spread swap + TER), derive d'exposition.

Meme signal (compute_allocation, E = min(2*alloc, 1.7)), meme bande asymetrique en
espace exposition (achat 0.50 / vente 1.00), exec_lag=1, memes hypotheses de portage.

    python -m myfiles.lqq_cash_vs_pust_lqq_buyfee     # depuis la racine, venv actif
"""
import numpy as np
import pandas as pd

from src.risk_off_strategy.data import load_ohlc, load_macro, load_funding
from src.risk_off_strategy.strategy import (
    ANN, SELL_FEE, BUY_FEE, TER_PUST, TER_LQQ, SWAP_SPREAD, BUY_THR_E, SELL_THR_E, E_MAX,
    compute_allocation,
)

ohlc = load_ohlc("QQQ")
price = ohlc["close"]
o, h, l, c = (ohlc[x].values for x in ["open", "high", "low", "close"])
nfci, cpi = load_macro(price.index)
funding = load_funding(price.index)
alloc = compute_allocation(c, nfci, cpi, high=h, low=l, open_=o)
N = len(c)
MID = N // 2
I16 = int(np.searchsorted(price.index, pd.Timestamp("2016-01-01")))
RET = pd.Series(c).pct_change().fillna(0).values
DRAG_LQQ = (funding + SWAP_SPREAD + TER_LQQ) / ANN     # portage quotidien du LQQ
E = np.minimum(2.0 * alloc, E_MAX)


def sim(E, sl, mode, buy_fee=BUY_FEE, sell_fee=SELL_FEE, carry=True, daily=False):
    """mode='L' : LQQ + cash (w = E/2) ; mode='P' : PUST + LQQ drag-minimal.
    carry=False : TER et financement a zero (isole le portage).
    daily=True  : rebalancement quotidien sans frais (isole la derive)."""
    e = E[sl]; ret = RET[sl]; dl = DRAG_LQQ[sl]
    rp = ret - (TER_PUST / ANN if carry else 0.0)
    rl = 2 * ret - (dl if carry else 0.0)
    tgt = np.concatenate([[0.0], e[:-1]])
    vp = vl = 0.0; vc = 1.0
    eq = np.empty(len(e)); fees = 0.0; trades = 0
    w_l = np.empty(len(e)); w_p = np.empty(len(e))
    for t in range(len(e)):
        vp *= 1 + rp[t]; vl *= 1 + rl[t]
        V = vp + vl + vc
        ee = (vp + 2 * vl) / V
        et = tgt[t]
        force = (et == 0.0 and ee > 1e-9) or (ee == 0.0 and et > 1e-9)
        thr = BUY_THR_E if et > ee else SELL_THR_E
        if daily or force or abs(et - ee) >= thr:
            if mode == "L":
                wl, wp = et / 2.0, 0.0
            else:
                wl = max(0.0, et - 1.0); wp = et if et <= 1 else 2.0 - et
            tp, tl = wp * V, wl * V
            fee = 0.0 if daily else (sell_fee * (max(0.0, vp - tp) + max(0.0, vl - tl))
                                     + buy_fee * (max(0.0, tp - vp) + max(0.0, tl - vl)))
            fees += fee / V; V -= fee
            vp, vl, vc = wp * V, wl * V, max(0.0, 1.0 - wp - wl) * V
            trades += 1
        eq[t] = V; w_l[t] = vl / V; w_p[t] = vp / V
    carry_yr = float(np.mean(w_l * dl + w_p * TER_PUST / ANN)) * ANN * 100   # % / an
    return eq, fees, trades, carry_yr


def metrics(eq, fees, trades, carry_yr):
    yrs = len(eq) / ANN
    pk = np.maximum.accumulate(eq)
    dd = float(((eq - pk) / pk).min())
    cagr = eq[-1] ** (1 / yrs) - 1
    r = eq[1:] / eq[:-1] - 1
    return dict(cagr=cagr * 100, dd=dd * 100, sh=r.mean() / r.std() * np.sqrt(ANN),
                cal=cagr / abs(dd), fees=fees / yrs * 100, tr=trades / yrs, carry=carry_yr)


WINDOWS = (("2000-26", slice(0, N)), ("H1", slice(0, MID)), ("H2", slice(MID, N)), ("2016+", slice(I16, N)))
LBL = {"L": "LQQ + cash", "P": "PUST + LQQ"}


def table(title, **kw):
    print(f"\n{title}")
    print(f"{'fenetre':<8} {'realisation':<11} | {'CAGR':>6} {'maxDD':>6} {'Shrp':>5} {'Calm':>5} | "
          f"{'frais/an':>8} {'portage/an':>10} {'trd/an':>6}")
    for tag, sl in WINDOWS:
        for mode in "LP":
            m = metrics(*sim(E, sl, mode, **kw))
            print(f"{tag:<8} {LBL[mode]:<11} | {m['cagr']:6.2f} {m['dd']:6.1f} {m['sh']:5.2f} {m['cal']:5.2f} | "
                  f"{m['fees']:8.2f} {m['carry']:10.2f} {m['tr']:6.1f}")


if __name__ == "__main__":
    print(f"QQQ {price.index[0].date()} -> {price.index[-1].date()} ; E = min(2*alloc, {E_MAX}) ; "
          f"bande achat {BUY_THR_E} / vente {SELL_THR_E} (expo) ; DFF moyen {funding.mean()*100:.2f}%")
    table("A) NET : achat 0.5% + vente 0.5% + portage (= modele courant)")
    table("B) achats gratuits + vente 0.5% + portage (= modele avant le 28/09)", buy_fee=0.0)
    table("C) sans frais de transaction, avec portage (isole la derive + le portage)", buy_fee=0.0, sell_fee=0.0)
    table("D) sans frais, sans portage (pure derive d'exposition entre rebalancements)",
          buy_fee=0.0, sell_fee=0.0, carry=False)
    table("E) rebalancement quotidien sans frais, avec portage (pur cout de portage, derive nulle)",
          daily=True)
    # sensibilite au taux court : portage LQQ = DFF + 0.40% + 0.60%
    print("\nF) NET, sensibilite au taux court (DFF remplace par une constante) — fenetre complete")
    print(f"{'DFF':>5} | {'LQQ+cash CAGR':>13} {'Calm':>5} | {'PUST+LQQ CAGR':>13} {'Calm':>5} | {'dCAGR':>6}")
    save = DRAG_LQQ.copy()
    for dff in (0.0, 0.02, 0.04, 0.05):
        DRAG_LQQ[:] = (dff + SWAP_SPREAD + TER_LQQ) / ANN
        mL = metrics(*sim(E, slice(0, N), "L")); mP = metrics(*sim(E, slice(0, N), "P"))
        print(f"{dff*100:5.1f} | {mL['cagr']:13.2f} {mL['cal']:5.2f} | {mP['cagr']:13.2f} {mP['cal']:5.2f} | "
              f"{mP['cagr']-mL['cagr']:6.2f}")
    DRAG_LQQ[:] = save
