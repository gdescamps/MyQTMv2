"""Re-calibrage de la bande de non-action pour la stratégie DÉPLOYÉE x1.7 PUST+LQQ (seuils en ALLOC x1, comme
BUY_THR_ALLOC / SELL_THR_ALLOC ; en expo = ×2). Actuel 0.25 / 0.50, calibré à l'époque avec achats gratuits.
Frais 0.5 %/côté (tarif actuel) et 0.1 %/côté. Validation H1 2000-2012 / H2 2013-2026 (Calmar + Sharpe).
    python -m myfiles.pust17_band_sweep
"""
import itertools

import numpy as np
import pandas as pd

import src.risk_off_strategy.strategy as st
from myfiles.pust17_emax_sharpe import stats
from src.risk_off_strategy.data import load_funding, load_macro, load_ohlc
from src.risk_off_strategy.strategy import E_MAX, compute_allocation, simulate_net

BUYS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40)
SELLS = (0.10, 0.20, 0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70)


def main():
    o = load_ohlc("QQQ", "2000-01-01", None)
    p, idx = o["close"], o.index
    nf, cp = load_macro(idx)
    fund = load_funding(idx)
    al = compute_allocation(p.values, nf, cp, high=o.high.values, low=o.low.values, open_=o.open.values)
    rows = []
    for fee in (0.005, 0.001):
        st.BUY_FEE = st.SELL_FEE = fee
        for b, s in itertools.product(BUYS, SELLS):
            eq, fz, rv = simulate_net(p, al, leverage=2, funding=fund, e_max=E_MAX, buy_thr=2 * b, sell_thr=2 * s)
            row = {"frais": fee, "achat": b, "vente": s, "frais %/an": round(fz, 2), "revis/an": round(rv, 1)}
            for h, (x, y) in {"tout": ("2000-01-01", None), "H1": ("2000-01-01", "2012-12-31"), "H2": ("2013-01-01", None)}.items():
                sh, cg, dd, k = stats(eq, idx, x, y)
                row.update({f"{h} Sharpe": round(sh, 2), f"{h} %/an": round(cg, 1), f"{h} maxDD": round(dd, 1), f"{h} Calmar": round(k, 2)})
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv("myfiles/pust17_band_sweep.csv", index=False)
    pd.set_option("display.width", 250)
    for fee in (0.005, 0.001):
        g = df[df.frais == fee].copy()
        ref = g[(g.achat == 0.25) & (g.vente == 0.50)].iloc[0]
        print(f"\n===== frais {fee*100:.1f}%  — actuelle 0.25/0.50 : H1 Calmar {ref['H1 Calmar']} H2 {ref['H2 Calmar']} "
              f"tout {ref['tout %/an']}%/an {ref['tout maxDD']} Sharpe {ref['tout Sharpe']} =====")
        for m in ("tout Calmar", "tout Sharpe", "H1 Calmar", "H2 Calmar"):
            print(f"\n-- grille {m} (lignes = achat, colonnes = vente) --")
            print(g.pivot(index="achat", columns="vente", values=m).to_string())
        g["pire moitié Calmar"] = g[["H1 Calmar", "H2 Calmar"]].min(axis=1)
        both = g[(g["H1 Calmar"] > ref["H1 Calmar"]) & (g["H2 Calmar"] > ref["H2 Calmar"])]
        print(f"\nBattent l'actuelle sur LES DEUX moitiés : {len(both)}")
        cols = ["achat", "vente", "H1 Calmar", "H2 Calmar", "tout %/an", "tout maxDD", "tout Sharpe", "tout Calmar", "frais %/an", "revis/an"]
        print(both.sort_values("tout Calmar", ascending=False)[cols].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
