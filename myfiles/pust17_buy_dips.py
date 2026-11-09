"""Stratégie INVERSE : acheter les creux de choc, sortir quand la hausse est « épuisée » (RSI haut).
En position = exposition x1.7 PUST+LQQ (alloc 1, comme la stratégie au max) ; hors position = cash.
Net de frais 0.1 %/côté (simulate_net, exec J+1). Comparée à la stratégie actuelle et au B&H x1.7.
Familles :
  rsi14   : entrée RSI14 < L, sortie RSI14 > H
  rsi2    : entrée RSI2 < L, sortie RSI2 > H ; variante « +tendance » : entrée seulement si close > SMA200
  creux   : entrée si drawdown depuis le plus haut 252 j ≤ −D, sortie RSI14 > H
  creux+macro : idem, cash forcé si NFCI > 0.5 ou CPI > 7 (garde-fous de la stratégie actuelle)
Validation H1 2000-2012 / H2 2013-2026 + validation croisée par famille + épisodes.
    python -m myfiles.pust17_buy_dips
"""
import itertools

import numpy as np
import pandas as pd

import src.risk_off_strategy.strategy as st
from myfiles.pust17_vol_estimators import EPIS, HALVES, win
from src.risk_off_strategy.data import load_funding, load_macro, load_ohlc
from src.risk_off_strategy.strategy import E_MAX, compute_allocation, rsi_wilder, simulate_net

st.BUY_FEE = st.SELL_FEE = 0.001


def state_machine(enter, exit_):
    pos = np.zeros(len(enter))
    on = False
    for i in range(len(enter)):
        if not on and enter[i]:
            on = True
        elif on and exit_[i]:
            on = False
        pos[i] = 1.0 if on else 0.0
    return pos


def main():
    o = load_ohlc("QQQ", "2000-01-01", None)
    p, idx = o["close"], o.index
    nfci, cpi = load_macro(idx)
    fund = load_funding(idx)
    c = p.values
    r14 = np.asarray(rsi_wilder(p, 14), float)
    r2 = np.asarray(rsi_wilder(p, 2), float)
    dd = (p / p.rolling(252, min_periods=20).max() - 1).values
    sma200 = p.rolling(200).mean().values
    macro_off = (np.nan_to_num(nfci, nan=-9) > st.NFCI_OFF) | (np.nan_to_num(cpi, nan=-9) > st.CPI_OFF)
    rows = []

    def run(name, fam, prm, alloc):
        eq, fees, rev = simulate_net(p, alloc, leverage=2, funding=fund, e_max=E_MAX)
        row = {"stratégie": name, "famille": fam, "param": prm, "% temps investi": round(100 * (alloc > 0).mean()),
               "frais %/an": round(fees, 2), "revis/an": round(rev, 1)}
        for hn, (a, b) in HALVES.items():
            cg, d_, k = win(eq, idx, a, b)
            row.update({f"{hn} %/an": round(cg, 1), f"{hn} maxDD": round(d_, 1), f"{hn} Calmar": round(k, 2)})
        for k_, (a, b) in EPIS.items():
            m = (idx >= a) & (idx <= b)
            row[k_] = round(100 * (eq[m][-1] / eq[m][0] - 1))
        rows.append(row)

    run("ACTUELLE (tendance+vol+macro)", "ref", "", compute_allocation(c, nfci, cpi, high=o.high.values, low=o.low.values, open_=o.open.values))
    run("B&H x1.7", "ref", "", np.ones(len(c)))
    for L, H in itertools.product((25, 30, 35, 40), (60, 65, 70, 75, 80)):
        run(f"RSI14 <{L} → >{H}", "rsi14", f"{L}/{H}", state_machine(r14 < L, r14 > H))
    for L, H in itertools.product((5, 10, 15), (60, 70, 80, 90)):
        run(f"RSI2 <{L} → >{H}", "rsi2", f"{L}/{H}", state_machine(r2 < L, r2 > H))
        run(f"RSI2 <{L} → >{H} +tendance", "rsi2+tendance", f"{L}/{H}", state_machine((r2 < L) & (c > sma200), r2 > H))
    for D, H in itertools.product((0.10, 0.15, 0.20, 0.25), (65, 70, 75, 80)):
        pos = state_machine(dd <= -D, r14 > H)
        run(f"creux −{int(100 * D)}% → RSI14 >{H}", "creux", f"{D}/{H}", pos)
        run(f"creux −{int(100 * D)}% → RSI14 >{H} +macro", "creux+macro", f"{D}/{H}", np.where(macro_off, 0.0, pos))
    df = pd.DataFrame(rows)
    df.to_csv("myfiles/pust17_buy_dips.csv", index=False)
    pd.set_option("display.width", 300)
    cols = ["stratégie", "% temps investi", "H1 Calmar", "H2 Calmar", "tout %/an", "tout maxDD", "tout Calmar", "revis/an"] + list(EPIS)
    print(df.iloc[:2][cols].to_string(index=False))
    print("\n=== Meilleure de chaque famille (Calmar tout l'historique — optimiste) ===")
    for fam in df.famille.unique()[1:]:
        g = df[df.famille == fam]
        print(g.loc[[g["tout Calmar"].idxmax()]][cols].to_string(index=False, header=False))
    ref = df.iloc[0]
    print("\n=== Validation croisée (réglé sur une moitié, lu sur l'autre) vs ACTUELLE ===")
    for fam in df.famille.unique()[1:]:
        g = df[df.famille == fam]
        for fit, test in (("H1", "H2"), ("H2", "H1")):
            b = g.loc[g[f"{fit} Calmar"].idxmax()]
            print(f"  {fam:14s} réglé {fit} → {b['stratégie']:34s} {test}: Calmar {b[f'{test} Calmar']:.2f} (actuelle {ref[f'{test} Calmar']:.2f})"
                  f"  %/an {b[f'{test} %/an']:.1f} (actuelle {ref[f'{test} %/an']:.1f})  maxDD {b[f'{test} maxDD']:.1f} (actuelle {ref[f'{test} maxDD']:.1f})"
                  f"  investi {b['% temps investi']}%")
    both = df[(df["H1 Calmar"] > ref["H1 Calmar"]) & (df["H2 Calmar"] > ref["H2 Calmar"]) & (df.famille != "ref")]
    print(f"\nBattent l'actuelle sur LES DEUX moitiés : {len(both)} / {len(df) - 2}")
    if len(both):
        print(both[cols].to_string(index=False))


if __name__ == "__main__":
    main()
