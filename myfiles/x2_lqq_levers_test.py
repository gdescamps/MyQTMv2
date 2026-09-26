"""
Amelioration Sharpe / Calmar de la strategie x2 LQQ (prompts.txt 2026-09-26).

Les parametres de la formule (cap2, e_max, bras below, floor2, bande, anti-chop) ont
deja ete balayes (x2_lqq_optimize / x2_improvement_analysis / x2_calmar_*). On teste
ici des leviers ORTHOGONAUX qui n'agissent que sur la PART DE LEVIER (E > 1), le
bras x1 restant identique :

  L1 rampe MA  : la part >1 n'est deployee que si le prix est a g_ramp au-dessus de la
                 SMA250 (lineaire de 0 a g_ramp)          -> tue le levier en zone de chop
  L2 vol up    : part >1 coupee/reduite quand la vol YZ monte (rv > rv il y a k jours)
  L3 funding   : part >1 reduite quand le taux court est haut (clip((F0-DFF)/F0, 0, 1))
  L4 breaker   : DD de la strategie x2 brute < -X -> E plafonnee a 1 jusqu'a DD > -Y
  L5 w_max     : plafond de poids LQQ en live (cash tampon)
  L6 reference : cap2 / floor2 dedies (deja connus, relus en net)
  R  realisation PUST+LQQ (drag-minimale) au lieu de LQQ+cash

Mesure NETTE (TER, financement DFF+spread, vente 0.5%, bande 0.25/0.50, exec_lag=1)
dans les DEUX realisations : LQQ+cash (espace poids, = live) et PUST+LQQ.
Robustesse : moitiés H1/H2 + periode 2016-2026 (existence reelle du LQQ).

    python -m myfiles.x2_lqq_levers_test          # depuis la racine, venv actif
"""
import itertools
from multiprocessing import Pool

import numpy as np
import pandas as pd

import myfiles.rsi_overlay_test as T
from src.risk_off_strategy.strategy import (
    ANN, SELL_FEE, TER_PUST, TER_LQQ, SWAP_SPREAD, BUY_THR_ALLOC, SELL_THR_ALLOC,
    ABOVE_CAP, DECAY2_FLOOR, GAP2_START, GAP2_SPAN,
)

N = T.N
MID = N // 2
I16 = int(np.searchsorted(T.price.index, pd.Timestamp("2016-01-01")))
RET = pd.Series(T.c).pct_change().fillna(0).values
R_PUST = RET - TER_PUST / ANN
R_LQQ = 2 * RET - (T.funding + SWAP_SPREAD + TER_LQQ) / ANN
BUY_E, SELL_E = BUY_THR_ALLOC * 2, SELL_THR_ALLOC * 2      # bande en espace exposition x2
E_BASE = 2.0 * T.base                                       # x2 actuel = 2 * alloc x1


# ── Simulateurs nets (E = exposition cible dans [0, 2]) ──────────────────────
def sim_lqq_cash(E, sl, w_max=1.0):
    """Live actuel : poids w = E/2 en LQQ, reste en cash (0%). Bande en poids."""
    e, r = E[sl], R_LQQ[sl]
    n = len(e)
    tgt = np.minimum(np.concatenate([[0.0], e[:-1]]) / 2.0, w_max)
    vi, vc = 0.0, 1.0
    eq = np.empty(n); fees = 0.0; trades = 0
    for t in range(n):
        vi *= 1 + r[t]
        V = vi + vc
        w = vi / V
        g = tgt[t]; d = g - w
        force = (g == 0.0 and w > 1e-9) or (w == 0.0 and g > 1e-9)
        if force or d >= BUY_THR_ALLOC or -d >= SELL_THR_ALLOC:
            fee = SELL_FEE * max(0.0, vi - g * V)
            fees += fee / V; V -= fee
            vi, vc = g * V, (1 - g) * V
            trades += 1
        eq[t] = vi + vc
    return eq, fees, trades


def sim_pust_lqq(E, sl):
    """Realisation drag-minimale (simulate_net) : E<=1 PUST+cash ; E>1 PUST=2-E, LQQ=E-1."""
    e, rp, rl = E[sl], R_PUST[sl], R_LQQ[sl]
    n = len(e)
    tgt = np.concatenate([[0.0], e[:-1]])
    vp = vl = 0.0; vc = 1.0
    eq = np.empty(n); fees = 0.0; trades = 0
    for t in range(n):
        vp *= 1 + rp[t]; vl *= 1 + rl[t]
        V = vp + vl + vc
        ee = (vp + 2 * vl) / V
        et = tgt[t]
        force = (et == 0.0 and ee > 1e-9) or (ee == 0.0 and et > 1e-9)
        thr = BUY_E if et > ee else SELL_E
        if force or abs(et - ee) >= thr:
            wl = max(0.0, et - 1.0); wp = et if et <= 1 else 2.0 - et
            fee = SELL_FEE * (max(0.0, vp - wp * V) + max(0.0, vl - wl * V))
            fees += fee / V; V -= fee
            vp, vl, vc = wp * V, wl * V, max(0.0, 1.0 - et) * V
            trades += 1
        eq[t] = V
    return eq, fees, trades


def metrics(eq, fees=0.0, trades=0):
    yrs = len(eq) / ANN
    pk = np.maximum.accumulate(eq)
    dd = float(((eq - pk) / pk).min())
    cagr = eq[-1] ** (1 / yrs) - 1
    r = eq[1:] / eq[:-1] - 1
    return dict(cagr=cagr, dd=dd, sh=r.mean() / r.std() * np.sqrt(ANN), cal=cagr / abs(dd),
                fees=fees / yrs * 100, tr=trades / yrs)


WINDOWS = (("F", slice(0, N)), ("H1", slice(0, MID)), ("H2", slice(MID, N)), ("16", slice(I16, N)))


def evaluate(E, w_max=1.0):
    out = {}
    for tag, sl in WINDOWS:
        out[("L", tag)] = metrics(*sim_lqq_cash(E, sl, w_max))
        out[("P", tag)] = metrics(*sim_pust_lqq(E, sl))
    return out


def score(m, ref):
    """Gain robuste : pire delta de Calmar sur (LQQ+cash, PUST+LQQ) x (H1, H2, 2016+)."""
    return min(m[(rz, t)]["cal"] - ref[(rz, t)]["cal"] for rz in "LP" for t in ("H1", "H2", "16"))


HDR = (f"{'variante':<36} | {'LQQ+cash CAGR':>13} {'maxDD':>6} {'Shrp':>5} {'Calm':>5} {'H1':>5} {'H2':>5} "
       f"{'2016+':>5} {'trd':>4} | {'PUST+LQQ Calm':>13} {'H1':>5} {'H2':>5} {'2016+':>5} {'Shrp':>5}")


def line(name, m, sc=None):
    a = m[("L", "F")]; p = m[("P", "F")]
    tag = f" [{sc:+.2f}]" if sc is not None else ""
    print(f"{(name + tag):<36} | {a['cagr']*100:12.1f}% {a['dd']*100:5.1f}% {a['sh']:5.2f} {a['cal']:5.2f} "
          f"{m[('L','H1')]['cal']:5.2f} {m[('L','H2')]['cal']:5.2f} {m[('L','16')]['cal']:5.2f} {a['tr']:4.1f} | "
          f"{p['cal']:13.2f} {m[('P','H1')]['cal']:5.2f} {m[('P','H2')]['cal']:5.2f} {m[('P','16')]['cal']:5.2f} "
          f"{p['sh']:5.2f}")


# ── Leviers (agissent sur la part >1 : E = min(E,1) + (E-1)+ * f) ────────────
def lever_part(E, f):
    return np.minimum(E, 1.0) + np.maximum(E - 1.0, 0.0) * np.clip(f, 0, 1)


def l1_ramp(g_ramp):
    return lever_part(E_BASE, T.gap / g_ramp)


def l2_volup(k, soft=None):
    ratio = np.ones(N); ratio[k:] = T.rv[k:] / T.rv[:-k]
    f = np.where(ratio <= 1.0, 1.0, 0.0) if soft is None else 1.0 - (ratio - 1.0) / soft
    return lever_part(E_BASE, f)


def l3_funding(f0):
    return lever_part(E_BASE, (f0 - T.funding) / f0)


def l4_breaker(x, y, E=None):
    """DD de la x2 brute continue (causal : DD connu au close t, applique a t+1 via exec_lag)."""
    E = E_BASE if E is None else E
    pos = np.concatenate([[0.0], E[:-1]])
    eq = np.cumprod(1 + pos * RET)
    dd = eq / np.maximum.accumulate(eq) - 1
    off = np.zeros(N, bool); state = False
    for t in range(N):
        if state and dd[t] > -y:
            state = False
        elif not state and dd[t] < -x:
            state = True
        off[t] = state
    return np.where(off, np.minimum(E, 1.0), E)


def l6_cap_floor(cap2, floor2):
    decay_up = np.clip(1.0 - np.maximum(0.0, T.gap - GAP2_START) / GAP2_SPAN, floor2, 1.0)
    above = np.clip(cap2 / T.rv, 0, 2) * decay_up
    return np.where(T.macro_off, 0.0, np.where(T.regime, above, 2 * T.below_arm))


def _job(args):
    name, E, w_max = args
    return name, evaluate(E, w_max)


if __name__ == "__main__":
    jobs = [("x2 actuelle (ref)", E_BASE, 1.0)]
    jobs += [(f"L1 rampe MA g={g:.2f}", l1_ramp(g), 1.0) for g in (0.02, 0.04, 0.06, 0.08, 0.12)]
    jobs += [(f"L2 vol up k={k} dur", l2_volup(k), 1.0) for k in (5, 10, 20)]
    jobs += [(f"L2 vol up k={k} soft{s}", l2_volup(k, s), 1.0) for k in (5, 10, 20) for s in (0.3, 0.6)]
    jobs += [(f"L3 funding F0={f0:.2f}", l3_funding(f0), 1.0) for f0 in (0.04, 0.06, 0.08)]
    jobs += [(f"L4 breaker X={x:.2f} Y={y:.2f}", l4_breaker(x, y), 1.0)
             for x in (0.10, 0.15, 0.20, 0.25) for y in (0.03, 0.05, 0.10) if y < x]
    jobs += [(f"L5 w_max={w:.2f}", E_BASE, w) for w in (0.80, 0.90)]
    jobs += [(f"L6 cap2={c2:.2f} floor2={fl:.1f}", l6_cap_floor(c2, fl), 1.0)
             for c2 in (0.24, 0.27, 0.30) for fl in (0.3, 0.4, 0.6) if not (c2 == 0.30 and fl == 0.4)]
    with Pool() as pool:
        res = pool.map(_job, jobs)
    ref = res[0][1]
    print(f"QQQ {T.price.index[0].date()} -> {T.price.index[-1].date()} ; H2 depuis "
          f"{T.price.index[MID].date()} ; score = pire delta Calmar sur 2 realisations x (H1, H2, 2016+)")
    print(HDR); print("-" * len(HDR))
    line(res[0][0], ref)
    fam = None
    for name, m in res[1:]:
        if name[:2] != fam:
            fam = name[:2]; print("-" * len(HDR))
        line(name, m, score(m, ref))
    pd.to_pickle(dict(res), "myfiles/x2_lqq_levers_results.pkl")
