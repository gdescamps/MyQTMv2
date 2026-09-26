"""
Tour 2 de x2_lqq_levers_test : sweep fin des deux leviers survivants (w_max live ;
L2 levier reduit quand la vol YZ monte), plafond d'exposition E_max (les deux
realisations), combos, et detail des 5 pires drawdowns (le gain vient-il d'un seul
episode ?).

    python -m myfiles.x2_lqq_levers_round2          # depuis la racine, venv actif
"""
import itertools
from multiprocessing import Pool

import numpy as np
import pandas as pd

import myfiles.x2_lqq_levers_test as X
import myfiles.rsi_overlay_test as T


def _job(args):
    name, E, w_max = args
    return name, X.evaluate(E, w_max)


def worst_dd(eq, k=5, sep=126):
    """(date creux, DD%) des k pires drawdowns disjoints (>= sep seances entre creux)."""
    pk = np.maximum.accumulate(eq)
    dd = eq / pk - 1
    out = []
    used = np.zeros(len(eq), bool)
    for _ in range(k):
        cand = np.where(used, 0.0, dd)
        i = int(cand.argmin())
        if cand[i] >= -1e-9:
            break
        out.append((T.price.index[i].strftime("%Y-%m"), dd[i] * 100))
        used[max(0, i - sep):i + sep] = True
    return out


if __name__ == "__main__":
    EB = X.E_BASE
    jobs = [("x2 actuelle (ref)", EB, 1.0)]
    jobs += [(f"w_max={w:.2f}", EB, w) for w in (0.85, 0.90, 0.95)]
    jobs += [(f"E_max={e:.1f}", np.minimum(EB, e), 1.0) for e in (1.6, 1.7, 1.8, 1.9)]
    jobs += [(f"L2 k={k} soft{s}", X.l2_volup(k, s), 1.0)
             for k in (10, 15, 20) for s in (0.2, 0.3, 0.4, 0.5)]
    c27 = X.l6_cap_floor(0.27, 0.6)
    jobs += [("L2 k=10 s0.3 + w_max0.90", X.l2_volup(10, 0.3), 0.90),
             ("L2 k=20 s0.3 + w_max0.90", X.l2_volup(20, 0.3), 0.90),
             ("L2 k=15 s0.3 + w_max0.90", X.l2_volup(15, 0.3), 0.90),
             ("cap2=0.27 fl0.6 + w_max0.90", c27, 0.90),
             ("L2 k=20 s0.3 + E_max1.8", np.minimum(X.l2_volup(20, 0.3), 1.8), 1.0),
             ("L2 k=10 s0.3 + E_max1.8", np.minimum(X.l2_volup(10, 0.3), 1.8), 1.0),
             ("L2 k=20 s0.3 + cap2=0.27 fl0.6", X.lever_part(c27, 1.0 - (np.r_[np.ones(20), T.rv[20:] / T.rv[:-20]] - 1) / 0.3), 1.0),
             ]
    with Pool() as pool:
        res = pool.map(_job, jobs)
    ref = res[0][1]
    print(X.HDR); print("-" * len(X.HDR))
    X.line(res[0][0], ref)
    for name, m in res[1:]:
        X.line(name, m, X.score(m, ref))

    print("\n5 pires drawdowns (creux, %) — realisation LQQ+cash (live) puis PUST+LQQ")
    for name, E, w in [jobs[0], ("w_max=0.90", EB, 0.90), ("L2 k=20 s0.3", X.l2_volup(20, 0.3), 1.0),
                       ("L2 k=10 s0.3", X.l2_volup(10, 0.3), 1.0),
                       ("L2 k=20 s0.3 + w_max0.90", X.l2_volup(20, 0.3), 0.90)]:
        eqL, _, _ = X.sim_lqq_cash(E, slice(0, X.N), w)
        eqP, _, _ = X.sim_pust_lqq(E, slice(0, X.N))
        fmt = lambda lst: "  ".join(f"{d} {v:5.1f}" for d, v in lst)
        print(f"  {name:<26} L | {fmt(worst_dd(eqL))}")
        print(f"  {'':<26} P | {fmt(worst_dd(eqP))}")

    # Part du temps ou le levier est reduit par L2 (k=20, soft 0.3) et effet moyen
    for k, s in ((10, 0.3), (20, 0.3)):
        E2 = X.l2_volup(k, s)
        lev = EB > 1
        red = (E2 < EB - 1e-9)
        print(f"\nL2 k={k} soft{s} : part >1 reduite {red[lev].mean()*100:.0f}% des jours en levier ; "
              f"exposition moyenne en levier {EB[lev].mean():.2f} -> {E2[lev].mean():.2f}")
