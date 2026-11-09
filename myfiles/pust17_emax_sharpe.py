"""Sharpe / CAGR / maxDD / Calmar de la stratégie actuelle selon le plafond d'exposition E_MAX (1.0 → 1.7),
leverage=2 (PUST+LQQ drag-minimal), net de frais (0.5 %/côté tarif actuel et 0.1 %/côté), 2000 → fin + moitiés.
    python -m myfiles.pust17_emax_sharpe
"""
import numpy as np
import pandas as pd

import src.risk_off_strategy.strategy as st
from myfiles.pust17_vol_estimators import HALVES
from src.risk_off_strategy.data import load_funding, load_macro, load_ohlc
from src.risk_off_strategy.strategy import compute_allocation, simulate_net


def stats(eq, idx, a, b):
    m = (idx >= pd.Timestamp(a)) & (idx <= (pd.Timestamp(b) if b else idx[-1]))
    e = eq[m] / eq[m][0]
    r = np.diff(e) / e[:-1]
    cagr = e[-1] ** (252 / m.sum()) - 1
    dd = (e / np.maximum.accumulate(e) - 1).min()
    return r.mean() / r.std() * np.sqrt(252), 100 * cagr, 100 * dd, cagr / abs(dd)


def main():
    o = load_ohlc("QQQ", "2000-01-01", None)
    p, idx = o["close"], o.index
    nfci, cpi = load_macro(idx)
    fund = load_funding(idx)
    al = compute_allocation(p.values, nfci, cpi, high=o.high.values, low=o.low.values, open_=o.open.values)
    rows = []
    for fee in (0.005, 0.001):
        st.BUY_FEE = st.SELL_FEE = fee
        for em in np.arange(1.0, 1.71, 0.1):
            eq, fz, rv = simulate_net(p, al, leverage=2, funding=fund, e_max=round(em, 1))
            row = {"frais/côté": f"{100 * fee:.1f}%", "E_MAX": round(em, 1)}
            for h, (a, b) in HALVES.items():
                sh, cg, dd, k = stats(eq, idx, a, b)
                row.update({f"{h} Sharpe": round(sh, 2), f"{h} %/an": round(cg, 1), f"{h} maxDD": round(dd, 1), f"{h} Calmar": round(k, 2)})
            row.update({"frais %/an": round(fz, 2), "revis/an": round(rv, 1)})
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv("myfiles/pust17_emax_sharpe.csv", index=False)
    pd.set_option("display.width", 250)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
