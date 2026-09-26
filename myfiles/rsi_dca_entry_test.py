"""
Entree progressive (DCA) pilotee par le RSI(14) lors d'un APPORT MASSIF de capital.

Question (prompts.txt) : aujourd'hui un apport est deploye le lendemain directement a
l'allocation cible, meme si le RSI est haut. Peut-on reduire le risque d'une tres
mauvaise PREMIERE ANNEE en deployant par tranches hebdomadaires, plus grosses quand
le RSI est bas, jusqu'a atteindre la cible ?

Protocole : on simule un apport de 100% du capital (depart en cash) a CHAQUE date de
l'historique QQQ (pas de 5 seances), puis la strategie deployee (compute_allocation,
x1 PUST, TER 0.30%, vente 0.5%, bande 0.25/0.50) pendant 1 an (et 3 ans pour le cout
d'attente). On compare la distribution des metriques de premiere annee.

Rampe (budget B = part du capital autorisee a etre investie, 0 au depart) :
  - chaque semaine (toutes les 5 seances depuis l'apport) : si RSI < GATE,
        B += clip(F_BASE + F_SLOPE * (GATE - RSI), 0, 1)
  - expo cible pendant la rampe = min(alloc_strategie, B) ; achats executes sans bande
  - les VENTES suivent la regle normale (cible strategie ; bande 0.50 ou alloc=0 force)
  - fin de rampe : B >= alloc cible (alloc > 0)      [mode "cible", demande utilisateur]
                   ou B >= 1                         [mode "plein"]
                   ou MAX_WEEKS semaines ecoulees     [filet : on ne reste pas dehors]
  ensuite execution normale (identique a simulate_net).

Controles : "immediat" (implementation actuelle) et DCA calendaire 1/N sans RSI.

    python -m myfiles.rsi_dca_entry_test          # depuis la racine, venv actif
"""
import itertools

import numpy as np
import pandas as pd

import myfiles.rsi_overlay_test as T
from src.risk_off_strategy.strategy import (
    ANN, BUY_THR_ALLOC, SELL_THR_ALLOC, SELL_FEE, TER_PUST,
)

A = T.base                                     # allocation deployee (x1)
RSI = T.rsi
RET = pd.Series(T.c).pct_change().fillna(0).values
R_PUST = RET - TER_PUST / ANN
N = len(A)
H1Y, H3Y = ANN, 3 * ANN
STARTS = np.arange(260, N - H1Y - 1, 5)        # SMA250 chaude ; 1 an de futur requis


def run(t0, horizon, tranche=None, mode="cible", max_weeks=26):
    """Equity (horizon+1 points) d'un apport a la cloture t0.
    tranche(rsi, week) -> increment de budget ; None = entree immediate (actuelle)."""
    vp, vc = 0.0, 1.0
    B = np.inf if tranche is None else 0.0
    eq = np.empty(horizon + 1)
    eq[0] = 1.0
    for i in range(1, horizon + 1):
        d = t0 + i - 1                          # decision a la cloture d, executee a d+1
        if not np.isinf(B):
            week = (i - 1) // 5
            if (i - 1) % 5 == 0:
                B += tranche(RSI[d], week)
            done_tgt = B >= 1.0 if mode == "plein" else (B >= A[d] and A[d] > 0)
            if done_tgt or week >= max_weeks:
                B = np.inf
        V = vp + vc
        e = vp / V
        tgt = A[d]
        if np.isinf(B):                         # regime normal (simulate_net x1)
            force = (tgt == 0.0 and e > 1e-9) or (e == 0.0 and tgt > 1e-9)
            thr = BUY_THR_ALLOC if tgt > e else SELL_THR_ALLOC
            go = abs(tgt - e) >= thr or force
        else:                                   # rampe : achats libres jusqu'a min(A, B)
            ramp_tgt = min(tgt, B)
            if ramp_tgt > e + 1e-9:
                tgt, go = ramp_tgt, True
            else:
                go = (e - tgt >= SELL_THR_ALLOC) or (tgt == 0.0 and e > 1e-9)
        if go:
            fee = SELL_FEE * max(0.0, vp - tgt * V)
            V -= fee
            vp, vc = tgt * V, (1 - tgt) * V
        vp *= 1 + R_PUST[d + 1]
        eq[i] = vp + vc
    return eq


def window_metrics(eq):
    r = eq[1:] / eq[:-1] - 1
    pk = np.maximum.accumulate(eq)
    dd = float(((eq - pk) / pk).min())
    tot = eq[-1] - 1
    sh = r.mean() / r.std() * np.sqrt(ANN) if r.std() > 0 else 0.0
    return tot, dd, sh, tot / abs(dd) if dd < -1e-9 else np.nan


def evaluate(**kw):
    rows = []
    for t0 in STARTS:
        h3 = min(H3Y, N - 1 - t0)
        eq = run(t0, h3, **kw)                  # la 1ere annee = prefixe du run 3 ans
        tot, dd, sh, cal = window_metrics(eq[:H1Y + 1])
        tot3 = eq[-1] ** (ANN / h3) - 1
        rows.append((tot, dd, sh, cal, tot3))
    return pd.DataFrame(rows, columns=["ret", "dd", "sh", "cal", "cagr3"], index=STARTS)


def summary(df, ref=None):
    s = dict(ret=df.ret.mean() * 100, p5=df.ret.quantile(0.05) * 100,
             pbad=(df.ret < -0.10).mean() * 100, dd=df.dd.mean() * 100,
             dd5=df.dd.quantile(0.05) * 100, sh=df.sh.mean(), cal=df.cal.median(),
             c3=df.cagr3.mean() * 100)
    if ref is not None:
        yr = pd.Series(T.price.index[df.index].year, index=df.index)
        dyr = (df.ret - ref.ret).groupby(yr).mean()        # delta moyen par annee d'apport
        ddyr = (df.dd - ref.dd).groupby(yr).mean()
        s["yr_dd"] = f"{(ddyr > 0).sum()}/{len(ddyr)}"
        s["yr_ret"] = f"{(dyr > 0).sum()}/{len(dyr)}"
    return s


HDR = (f"{'variante':<40} | {'ret1a':>6} {'P5':>6} {'P<-10%':>6} {'DD':>6} {'DD P5':>6} "
       f"{'Shrp':>5} {'Cal~':>5} | {'CAGR3a':>6} | {'DD>ref':>6} {'ret>ref':>7}")


def line(name, s):
    print(f"{name:<40} | {s['ret']:5.1f}% {s['p5']:5.1f}% {s['pbad']:5.1f}% {s['dd']:5.1f}% "
          f"{s['dd5']:5.1f}% {s['sh']:5.2f} {s['cal']:5.2f} | {s['cagr3']:5.1f}% | "
          f"{s.get('yr_dd', '-'):>6} {s.get('yr_ret', '-'):>7}"
          if "cagr3" in s else "")


class RsiTranche:
    """Increment hebdo de budget : base + slope*(gate - RSI) si RSI < gate (picklable)."""
    def __init__(self, gate, base, slope):
        self.gate, self.base, self.slope = gate, base, slope

    def __call__(self, rsi, week):
        return float(np.clip(self.base + self.slope * (self.gate - rsi), 0, 1)) if rsi < self.gate else 0.0


class Fixed:
    def __init__(self, n):
        self.n = n

    def __call__(self, rsi, week):
        return 1.0 / self.n


def _job(args):
    name, kw = args
    return name, evaluate(**kw)


if __name__ == "__main__":
    from multiprocessing import Pool
    jobs = [("immediat (actuel)", {})]
    jobs += [(f"DCA calendaire 1/{n} par semaine", dict(tranche=Fixed(n), mode="plein", max_weeks=99))
             for n in (4, 8, 13)]
    for gate, base, slope, mode, mw in itertools.product(
            (50, 55, 60), (0.1, 0.2, 0.33), (0.0, 0.02, 0.04), ("cible", "plein"), (13, 26)):
        jobs.append((f"RSI<{gate} b{base} s{slope} {mode} max{mw}s",
                     dict(tranche=RsiTranche(gate, base, slope), mode=mode, max_weeks=mw)))
    with Pool() as pool:
        res = pool.map(_job, jobs)
    ref = res[0][1]

    print(f"{len(STARTS)} apports simules ({T.price.index[STARTS[0]].date()} -> "
          f"{T.price.index[STARTS[-1]].date()}), 1 an chacun ; DD>ref / ret>ref = nb "
          f"d'annees d'apport ou la variante fait mieux en moyenne")
    print(HDR.replace("CAGR3a", "CAGR3a")); print("-" * len(HDR))
    table = []
    for name, df in res:
        s = summary(df, None if df is ref else ref)
        s["cagr3"] = s.pop("c3")
        table.append((name, s))
    line(*table[0])
    for name, s in table[1:4]:
        line(name, s)
    print("-" * len(HDR) + "\n  RSI : top 15 par DD P5 (pire 5% des drawdowns de 1ere annee)")
    for name, s in sorted(table[4:], key=lambda x: -x[1]["dd5"])[:15]:
        line(name, s)
    pd.to_pickle(dict(res), "myfiles/rsi_dca_entry_results.pkl")
