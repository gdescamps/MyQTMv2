"""Controle du drawdown sur la propre courbe de capital (CPPI/TIPP Black & Perold 1992, Grossman & Zhou 1993)
applique a la strategie deployee x1.7 PUST+LQQ (frais 0.5 %/cote, sans saturation, bande 0.50/1.00).

Plafond dynamique d'exposition, calcule sur le drawdown connu a la cloture de la veille (pas de look-ahead) :
  lin  : cap = max(cap_min, E_MAX · clip(1 − dd/DDmax, 0, 1))        (Grossman-Zhou linearise)
  step : cap = cap_lvl des que dd >= X, relache quand dd < X/2 (hysteresis) — "deleverage en drawdown"
Source du drawdown :
  own    : courbe reelle de la strategie plafonnee (CPPI pur, path-dependent -> risque de blocage en cash)
  shadow : courbe de la strategie NON plafonnee (calculee a part) -> se retablit seule, pas de blocage
Pic : 'all' = plus haut historique, 252/126 = plus haut glissant (deblocage par oubli du pic).
Execution : 'band' = le cap ne modifie que la cible (la bande 0.50/1.00 decide) ;
            'hard' = vente forcee jusqu'au cap des que expo effective − cap >= 0.20.
Validation : parametres choisis sur une moitie (Calmar), lus sur l'autre — H1 2000-2012 / H2 2013-2026.
    python -W ignore -m myfiles.failed_xp.pust17_dd_control      (log : myfiles/failed_xp/pust17_dd_control.log)

Resultats (2026-11-09, ref = 16.4 %/an, maxDD -29.5 %, Calmar 0.55, H1/H2 0.23/0.88) — ECARTE :
  - 300 variantes : AUCUNE ne bat la ref en Calmar (meilleure : step 25 %/cap 1.0/shadow/126/band = 0.53).
  - On peut acheter du maxDD (lin 0.30 own 252 : -24.9 %) mais au prix de -3.3 pts/an (Calmar 0.52) ;
    en vente forcee ('hard') -23 % de maxDD pour 10.8 %/an.
  - Validation croisee : les 16 familles perdent hors echantillon (gain Calmar -0.03 a -0.57).
  Pourquoi : le maxDD de la ref est un plateau de 6 episodes a -25/-30 % (2008, 2010, 2016, 2018, 2020,
  2025) dont la 1re jambe tombe a pleine expo 1.7 (-14 a -17 % en 10 seances) avant toute reaction ; le
  plafond agit APRES la perte, vend pres du creux (0.5 % de frais) et re-leve trop tard sur les rebonds en V.
  Le filtre tendance+vol fait deja ce travail, plus tot. Le seul levier sur ces 1res jambes reste E_MAX.
"""
import itertools

import numpy as np
import pandas as pd

import src.risk_off_strategy.strategy as st
from src.risk_off_strategy.data import load_funding, load_macro, load_ohlc
from src.risk_off_strategy.strategy import E_MAX, compute_allocation, simulate_net

HALVES = {"H1": ("2000-01-01", "2012-12-31"), "H2": ("2013-01-01", None), "tout": ("2000-01-01", None)}
HARD_TOL = 0.20


def drawdown(eq, window):
    s = pd.Series(eq)
    peak = s.cummax() if window == "all" else s.rolling(int(window), min_periods=1).max()
    return (1 - s / peak).values


def cap_from_dd(dd, kind, p1, p2, state):
    """kind='lin' : p1=DDmax, p2=cap_min ; kind='step' : p1=X, p2=cap_lvl (state = deleverage actif)."""
    if kind == "lin":
        return max(p2, E_MAX * min(max(1 - dd / p1, 0.0), 1.0)), state
    if state:
        state = dd >= p1 / 2
    else:
        state = dd >= p1
    return (p2 if state else E_MAX), state


def sim_capped(price, alloc, funding, kind, p1, p2, source, window, mode, dd_shadow=None):
    """Copie de strategy.simulate_net (leverage=2, e_max=E_MAX, exec_lag=1, sat_gap=None) + plafond dynamique."""
    p = np.asarray(price, float)
    ret = pd.Series(p).pct_change().fillna(0).values
    n = len(p)
    fund = np.asarray(funding, float)
    sell_thr, buy_thr = st.SELL_THR_ALLOC * 2, st.BUY_THR_ALLOC * 2
    r_pust = ret - st.TER_PUST / st.ANN
    r_lqq = 2 * ret - (fund + st.SWAP_SPREAD + st.TER_LQQ) / st.ANN
    E = np.clip(2 * np.asarray(alloc, float), 0.0, E_MAX)
    Etgt = np.concatenate([[0.0], E[:-1]])
    vp = vl = 0.0
    vc = 1.0
    eq = np.empty(n)
    expo = np.empty(n)
    fees_frac, revis = 0.0, 0
    peak_hist = []
    state = False
    caps = np.empty(n)
    for t in range(n):
        vp *= (1 + r_pust[t]); vl *= (1 + r_lqq[t])
        V = vp + vl + vc
        e_eff = (vp + 2 * vl) / V if V > 0 else 0.0
        # drawdown connu a la cloture t-1
        if t == 0:
            dd = 0.0
        elif source == "shadow":
            dd = dd_shadow[t - 1]
        else:
            w = peak_hist if window == "all" else peak_hist[-int(window):]
            dd = 1 - eq[t - 1] / max(w)
        cap, state = cap_from_dd(dd, kind, p1, p2, state)
        caps[t] = cap
        et = min(Etgt[t], cap)
        force = (et == 0.0 and e_eff > 1e-9) or (e_eff == 0.0 and et > 1e-9)
        hard = mode == "hard" and e_eff - cap >= HARD_TOL
        thr = buy_thr if et > e_eff else sell_thr
        if abs(et - e_eff) >= thr or force or hard:
            wl = max(0.0, et - 1.0)
            wp = et if et <= 1.0 else 2.0 - et
            tp, tl = wp * V, wl * V
            fee = (st.SELL_FEE * (max(0.0, vp - tp) + max(0.0, vl - tl))
                   + st.BUY_FEE * (max(0.0, tp - vp) + max(0.0, tl - vl)))
            fees_frac += fee / V
            V -= fee
            vp, vl, vc = wp * V, wl * V, max(0.0, 1.0 - et) * V
            revis += 1
        eq[t] = V
        expo[t] = (vp + 2 * vl) / V if V > 0 else 0.0
        if window == "all":
            peak_hist = [max(peak_hist[0], V)] if peak_hist else [V]
        else:
            peak_hist.append(V)
    yrs = n / st.ANN
    return eq, fees_frac / yrs * 100.0, revis / yrs, expo, caps


def win(eq, idx, a, b):
    m = (idx >= pd.Timestamp(a)) & (idx <= (pd.Timestamp(b) if b else idx[-1]))
    e = eq[m] / eq[m][0]
    yrs = m.sum() / 252
    cagr = e[-1] ** (1 / yrs) - 1
    dd = (e / np.maximum.accumulate(e) - 1).min()
    return 100 * cagr, 100 * dd, cagr / abs(dd)


def main():
    o = load_ohlc("QQQ", "2000-01-01", None)
    p, idx = o["close"], o.index
    nfci, cpi = load_macro(idx)
    fund = load_funding(idx)
    alloc = compute_allocation(p.values, nfci, cpi, high=o["high"].values, low=o["low"].values,
                               open_=o["open"].values)
    eq_ref, fees_ref, rev_ref, expo_ref = simulate_net(p, alloc, leverage=2, funding=fund, e_max=E_MAX,
                                                       return_expo=True)
    rows = []

    def record(name, params, eq, fees, rev, expo, caps):
        lowcap = caps < E_MAX - 1e-9
        r = {"regle": name, **params, "% jours plafonnes": round(100 * lowcap.mean(), 1),
             "expo moy": round(expo.mean(), 2), "frais %/an": round(fees, 2), "revis/an": round(rev, 1)}
        for h, (x, y) in HALVES.items():
            c, d, cal = win(eq, idx, x, y)
            r.update({f"{h} %/an": round(c, 1), f"{h} maxDD": round(d, 1), f"{h} Calmar": round(cal, 2)})
        rows.append(r)

    # controle : la copie du simulateur sans plafond doit redonner simulate_net
    eq0, f0, r0, x0, c0 = sim_capped(p, alloc, fund, "lin", 1e9, E_MAX, "own", "all", "band")
    assert np.allclose(eq0, eq_ref), "la copie du simulateur diverge de simulate_net"
    record("ref", {"p1": "-", "p2": "-", "source": "-", "pic": "-", "exec": "-"}, eq0, f0, r0, x0, c0)

    dd_sh = {w: drawdown(eq_ref, w) for w in ("all", 252, 126)}
    grid = []
    for p1, p2 in itertools.product([0.15, 0.20, 0.25, 0.30, 0.40], [0.0, 0.5, 1.0]):
        grid.append(("lin", p1, p2))
    for p1, p2 in itertools.product([0.10, 0.15, 0.20, 0.25], [0.0, 0.5, 1.0]):
        grid.append(("step", p1, p2))
    for (kind, p1, p2), source, window, mode in itertools.product(grid, ("own", "shadow"), ("all", 252, 126),
                                                                    ("band", "hard")):
        res = sim_capped(p, alloc, fund, kind, p1, p2, source, window, mode, dd_shadow=dd_sh[window])
        record(kind, {"p1": p1, "p2": p2, "source": source, "pic": window, "exec": mode}, *res)

    df = pd.DataFrame(rows)
    df.to_csv("myfiles/failed_xp/pust17_dd_control.csv", index=False)
    pd.set_option("display.width", 280)
    pd.set_option("display.max_rows", 300)
    cols = ["regle", "p1", "p2", "source", "pic", "exec", "% jours plafonnes", "expo moy", "H1 %/an", "H1 maxDD",
            "H1 Calmar", "H2 %/an", "H2 maxDD", "H2 Calmar", "tout %/an", "tout maxDD", "tout Calmar",
            "frais %/an", "revis/an"]
    ref = df.iloc[0]
    print("=== Reference ===")
    print(df.iloc[[0]][cols].to_string(index=False))
    print("\n=== Top 20 Calmar tout l'historique ===")
    print(df.sort_values("tout Calmar", ascending=False)[cols].head(20).to_string(index=False))
    print("\n=== Top 15 maxDD tout l'historique (le moins profond) avec CAGR >= ref − 2 pts ===")
    ok = df[df["tout %/an"] >= ref["tout %/an"] - 2]
    print(ok.sort_values("tout maxDD", ascending=False)[cols].head(15).to_string(index=False))

    print("\n=== Validation croisee par famille (regle x source x exec) : reglage sur une moitie, lu sur l'autre ===")
    out = []
    for (kind, source, mode), g in df[df.regle != "ref"].groupby(["regle", "source", "exec"]):
        for fit, test in (("H1", "H2"), ("H2", "H1")):
            b = g.loc[g[f"{fit} Calmar"].idxmax()]
            out.append({"regle": kind, "source": source, "exec": mode, "regle sur": fit,
                        "params": f"{b.p1}/{b.p2}/{b.pic}",
                        "Calmar test": b[f"{test} Calmar"], "ref": ref[f"{test} Calmar"],
                        "maxDD test": b[f"{test} maxDD"], "ref maxDD": ref[f"{test} maxDD"],
                        "%/an test": b[f"{test} %/an"], "ref %/an": ref[f"{test} %/an"]})
    o2 = pd.DataFrame(out)
    o2["gain Calmar"] = (o2["Calmar test"] - o2["ref"]).round(2)
    print(o2.to_string(index=False))


if __name__ == "__main__":
    main()
