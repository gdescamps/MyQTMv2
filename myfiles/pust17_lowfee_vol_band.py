"""Frais réduits (0.1 %/côté = 0.2 % A/R) : couper plus souvent ? Grille bande de vente / d'achat × sévérité vol.

Leviers (stratégie déployée x1.7 PUST+LQQ, nette, NFCI+CPI conservés) :
  SELL_E : seuil de VENTE en exposition (actuel 1.00 = « on ne dé-lève que par 50 % d'alloc »)
  BUY_E  : seuil d'ACHAT en exposition (actuel 0.50)
  ABOVE_CAP : vol au-dessus de laquelle on coupe déjà sur la MA (actuel 0.15 ; plus bas = plus sévère)
  VOL_TARGET: cible de vol du bras sous la MA (actuel 0.08 ; plus bas = plus sévère)
Validation : métriques sur H1 2000-2012 / H2 2013-2026 ; un réglage n'est retenu que s'il bat l'actuel
sur LES DEUX moitiés (Calmar), + épisodes de crise.
    python -m myfiles.pust17_lowfee_vol_band
"""
import itertools

import numpy as np
import pandas as pd

import src.risk_off_strategy.strategy as st
from src.risk_off_strategy.data import load_funding, load_macro, load_ohlc
from src.risk_off_strategy.strategy import E_MAX, compute_allocation, simulate_net

st.BUY_FEE = st.SELL_FEE = 0.001
SELLS, BUYS = (1.00, 0.80, 0.60, 0.50, 0.40, 0.30, 0.20), (0.50, 0.30, 0.20)
CAPS, VTS = (0.15, 0.12, 0.10), (0.08, 0.06, 0.05)
HALVES = {"H1": ("2000-01-01", "2012-12-31"), "H2": ("2013-01-01", None), "tout": ("2000-01-01", None)}
EPIS = {"2000-02": ("2000-03-01", "2002-10-31"), "2008": ("2007-10-01", "2009-03-31"), "2018T4": ("2018-10-01", "2018-12-31"),
        "2020": ("2020-02-15", "2020-04-30"), "2022": ("2021-11-15", "2022-12-31"), "2025": ("2025-02-15", "2025-05-31")}


def win(eq, idx, a, b):
    m = (idx >= pd.Timestamp(a)) & (idx <= (pd.Timestamp(b) if b else idx[-1]))
    e = eq[m] / eq[m][0]
    cagr = e[-1] ** (252 / m.sum()) - 1
    dd = (e / np.maximum.accumulate(e) - 1).min()
    return 100 * cagr, 100 * dd, cagr / abs(dd)


def main():
    o = load_ohlc("QQQ", "2000-01-01", None)
    p, idx = o["close"], o.index
    nfci, cpi = load_macro(idx)
    fund = load_funding(idx)
    kw = dict(high=o["high"].values, low=o["low"].values, open_=o["open"].values)
    rows = []
    for cap, vt in itertools.product(CAPS, VTS):
        st.VOL_TARGET = vt
        a = compute_allocation(p.values, nfci, cpi, above_cap=cap, **kw)
        for s, b in itertools.product(SELLS, BUYS):
            eq, fees, rev = simulate_net(p, a, leverage=2, funding=fund, e_max=E_MAX, sell_thr=s, buy_thr=b)
            r = {"above_cap": cap, "vol_target": vt, "vente E": s, "achat E": b, "frais %/an": round(fees, 2),
                 "revis/an": round(rev, 1)}
            for h, (x, y) in HALVES.items():
                c, d, k = win(eq, idx, x, y)
                r.update({f"{h} %/an": round(c, 1), f"{h} maxDD": round(d, 1), f"{h} Calmar": round(k, 2)})
            for k, (x, y) in EPIS.items():
                m = (idx >= x) & (idx <= y)
                r[k] = round(100 * (eq[m][-1] / eq[m][0] - 1))
            rows.append(r)
    st.VOL_TARGET = 0.08
    df = pd.DataFrame(rows)
    df.to_csv("myfiles/pust17_lowfee_vol_band.csv", index=False)
    ref = df[(df.above_cap == 0.15) & (df.vol_target == 0.08) & (df["vente E"] == 1.0) & (df["achat E"] == 0.5)].iloc[0]
    pd.set_option("display.width", 260)
    cols = ["above_cap", "vol_target", "vente E", "achat E", "H1 Calmar", "H2 Calmar", "tout %/an", "tout maxDD",
            "tout Calmar", "frais %/an", "revis/an"] + list(EPIS)
    print("=== Actuel (cap 0.15, V 0.08, vente 1.00, achat 0.50) ===")
    print(ref[cols].to_frame().T.to_string(index=False))
    print("\n=== Effet de la seule bande de vente (réglages vol actuels, achat 0.50) ===")
    print(df[(df.above_cap == 0.15) & (df.vol_target == 0.08) & (df["achat E"] == 0.5)][cols].to_string(index=False))
    both = df[(df["H1 Calmar"] > ref["H1 Calmar"]) & (df["H2 Calmar"] > ref["H2 Calmar"])]
    print(f"\n=== Réglages qui battent l'actuel sur LES DEUX moitiés : {len(both)} / {len(df)} ===")
    print(both.sort_values("tout Calmar", ascending=False)[cols].head(20).to_string(index=False))
    print("\n=== Top 10 tout l'historique (pour info, risque de surajustement) ===")
    print(df.sort_values("tout Calmar", ascending=False)[cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
