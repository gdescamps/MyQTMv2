"""
Features RSI(14) "dernier pic" -> modulation de l'allocation (prompts.txt) :

  peak_lvl  = niveau du dernier pic de RSI (max d'un episode RSI > PEAK_THR)
  peak_age  = jours de bourse ecoules depuis ce pic

Causalite : un episode n'est connu qu'une fois TERMINE (RSI repasse sous PEAK_THR).
Au jour t on n'utilise que les episodes clos <= t ; l'age est compte depuis la date
du max (connue a la cloture de l'episode) -> aucun look-ahead.

Modulation (bras "above" seulement, ou les deux bras) :
    intensite = clip((peak_lvl - PEAK_THR) / (85 - PEAK_THR), 0, 1)   # pic haut = fort
    recence   = exp(-peak_age / TAU)                                   # pic recent = 1
    facteur   = 1 - CUT * intensite * recence                          # pic recent -> baisse
              * (1 + BOOST si peak_age > T_OLD)                        # pic ancien -> favorise

    python -m myfiles.rsi_peak_test          # depuis la racine, venv actif
"""
import itertools

import numpy as np

import myfiles.rsi_overlay_test as T


def rsi_peak_features(rsi, thr):
    """(peak_lvl, peak_age) causaux ; NaN / inf avant le premier episode clos."""
    n = len(rsi)
    lvl = np.full(n, np.nan)
    age = np.full(n, np.inf)
    cur_lvl, cur_idx = np.nan, None          # dernier pic CONFIRME
    ep_max, ep_idx = -np.inf, None           # episode en cours
    for t in range(n):
        if rsi[t] > thr:
            if rsi[t] > ep_max:
                ep_max, ep_idx = rsi[t], t
        elif ep_idx is not None:             # l'episode se termine -> pic confirme
            cur_lvl, cur_idx = ep_max, ep_idx
            ep_max, ep_idx = -np.inf, None
        if cur_idx is not None:
            lvl[t], age[t] = cur_lvl, t - cur_idx
    return lvl, age


def factor(lvl, age, thr, tau, cut, boost, t_old):
    inten = np.clip((np.nan_to_num(lvl, nan=thr) - thr) / (85 - thr), 0, 1)
    rec = np.exp(-age / tau)
    f = 1 - cut * inten * rec
    return f * np.where(age > t_old, 1 + boost, 1.0)


def alloc_with(f, both):
    a_arm = T.above_arm * f
    b_arm = T.below_arm * f if both else T.below_arm
    return np.where(T.macro_off, 0.0, np.clip(np.where(T.regime, a_arm, b_arm), 0, 1))


if __name__ == "__main__":
    ref = T.evaluate(T.base)
    rows = []
    for thr in (65, 70, 75):
        lvl, age = rsi_peak_features(T.rsi, thr)
        for tau, cut, boost, t_old, both in itertools.product(
                (10, 20, 40, 60, 120), (0.3, 0.5, 0.7), (0.0, 0.15, 0.3), (60, 120), (False, True)):
            if boost == 0 and t_old == 120:
                continue
            m = T.evaluate(alloc_with(factor(lvl, age, thr, tau, cut, boost, t_old), both))
            rows.append(((thr, tau, cut, boost, t_old, both), T.score(m, ref), m))

    print(f"Dernier pic RSI>70 : ", end="")
    lvl, age = rsi_peak_features(T.rsi, 70)
    print(f"niveau {lvl[-1]:.1f}, il y a {age[-1]:.0f} seances")
    print(T.HDR); print("-" * len(T.HDR)); T.line("actuelle (ref)", ref)
    for key, sc, m in sorted(rows, key=lambda r: -r[1])[:20]:
        thr, tau, cut, boost, t_old, both = key
        T.line(f"p>{thr} tau{tau} c{cut} b{boost}@{t_old}{' 2b' if both else ''} [{sc:+.2f}]", m)

    # Robustesse : score moyen / part > 0 par valeur de chaque parametre (plateau ?)
    print("\nScore robuste (pire delta Calmar H1/H2 x1/x2) par parametre : moyenne | %>0")
    names = ["thr", "tau", "cut", "boost", "t_old", "both"]
    for i, nm in enumerate(names):
        vals = sorted({r[0][i] for r in rows})
        cells = []
        for v in vals:
            sc = np.array([r[1] for r in rows if r[0][i] == v])
            cells.append(f"{v}: {sc.mean():+.3f} | {(sc > 0).mean()*100:3.0f}%")
        print(f"  {nm:<6} " + "   ".join(cells))
