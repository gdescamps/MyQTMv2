"""
Sous-investissement du a la bande d'achat 0.25 (prompts.txt) : en live (LQQ, x2) le
poids reel peut rester a ~82% alors que la cible est a 93% (delta 11% < 25% -> aucun
achat, alors que les achats sont GRATUITS). Le capital "dort" en cash PEA.

Simulation en ESPACE POIDS, identique au live real_bourso : poids w de l'instrument
(PUST x1 ou LQQ x2), cash 1-w non remunere ; vente 0.5% du notionnel vendu. Variantes
de la regle d'ACHAT (la regle de vente reste 0.50 / alloc=0 force) :

  ref        : achat si cible - w >= 0.25                    (actuel)
  buy=X      : seuil d'achat abaisse a X
  full>=F    : en plus, achat vers la cible des que cible >= F (et w < cible - 0.02)
  persist K  : en plus, achat si cible - w >= 0.05 depuis K seances consecutives
  weekly     : en plus, complement hebdo (1 seance / 5) si cible - w >= 0.05

    python -m myfiles.topup_band_test          # depuis la racine, venv actif
"""
import numpy as np
import pandas as pd

import myfiles.rsi_overlay_test as T
from src.risk_off_strategy.strategy import ANN, SELL_FEE, TER_PUST, TER_LQQ, SWAP_SPREAD

A = T.base
N = len(A)
MID = N // 2
RET = pd.Series(T.c).pct_change().fillna(0).values
FUND = T.funding
R_INS = {1: RET - TER_PUST / ANN, 2: 2 * RET - (FUND + SWAP_SPREAD + TER_LQQ) / ANN}


def sim(lev, sl, buy=0.25, sell=0.50, full=None, persist=None, weekly=False, small=0.05):
    a = A[sl]
    r = R_INS[lev][sl]
    n = len(a)
    tgt = np.concatenate([[0.0], a[:-1]])      # exec_lag=1
    vi, vc = 0.0, 1.0
    eq, under = np.empty(n), np.empty(n)
    fees = 0.0
    trades = 0
    streak = 0
    for t in range(n):
        vi *= 1 + r[t]
        V = vi + vc
        w = vi / V if V > 0 else 0.0
        g = tgt[t]
        d = g - w
        streak = streak + 1 if d >= small else 0
        force = (g == 0.0 and w > 1e-9) or (w == 0.0 and g > 1e-9)
        go = force or d >= buy or -d >= sell
        if full is not None and g >= full and d >= 0.02:
            go = True
        if persist is not None and streak >= persist:
            go = True
        if weekly and t % 5 == 0 and d >= small:
            go = True
        if go and abs(d) > 1e-9:
            fee = SELL_FEE * max(0.0, vi - g * V)
            fees += fee / V
            V -= fee
            vi, vc = g * V, (1 - g) * V
            trades += 1
            streak = 0
        eq[t] = vi + vc
        under[t] = max(0.0, g - vi / eq[t])
    yrs = n / ANN
    pk = np.maximum.accumulate(eq)
    dd = float(((eq - pk) / pk).min())
    cagr = eq[-1] ** (1 / yrs) - 1
    rr = eq[1:] / eq[:-1] - 1
    return dict(cagr=cagr, dd=dd, sh=rr.mean() / rr.std() * np.sqrt(ANN), cal=cagr / abs(dd),
                fees=fees / yrs * 100, tr=trades / yrs, under=under.mean() * 100)


VARIANTS = [("ref (achat 0.25)", {})]
VARIANTS += [(f"buy={b}", dict(buy=b)) for b in (0.05, 0.10, 0.15, 0.20)]
VARIANTS += [(f"full>={f}", dict(full=f)) for f in (0.80, 0.85, 0.90, 0.95)]
VARIANTS += [(f"persist {k}j", dict(persist=k)) for k in (5, 10, 20, 40)]
VARIANTS += [("weekly", dict(weekly=True))]

if __name__ == "__main__":
    for lev, lab in ((1, "PUST x1"), (2, "LQQ x2 (live)")):
        print(f"\n=== {lab} — backtest net QQQ 2000-2026, espace poids (comme real_bourso) ===")
        print(f"{'variante':<18} | {'CAGR':>6} {'maxDD':>6} {'Shrp':>5} {'Calm':>5} {'frais':>6} "
              f"{'trd/an':>6} {'sous-inv':>8} | {'CalH1':>5} {'CalH2':>5}")
        for name, kw in VARIANTS:
            m = sim(lev, slice(0, N), **kw)
            h1, h2 = sim(lev, slice(0, MID), **kw), sim(lev, slice(MID, N), **kw)
            print(f"{name:<18} | {m['cagr']*100:5.1f}% {m['dd']*100:5.1f}% {m['sh']:5.2f} {m['cal']:5.2f} "
                  f"{m['fees']:5.2f}% {m['tr']:6.1f} {m['under']:7.1f}% | {h1['cal']:5.2f} {h2['cal']:5.2f}")
