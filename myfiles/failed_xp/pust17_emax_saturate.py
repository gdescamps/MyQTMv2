"""Bilan par plafond d'exposition E_MAX 1.0 -> 1.7, AVEC la saturation au plafond (SAT_GAP_E=0.10,
plafond = e_max) vs sans. Frais 0.5 %/côté (tarif actuel).
    python -m myfiles.failed_xp.pust17_emax_saturate
"""
import numpy as np
import pandas as pd

import src.risk_off_strategy.strategy as st
from src.risk_off_strategy.data import load_funding, load_macro, load_ohlc


def metr(eq, idx, a, b=None):
    m = (idx >= a) & ((idx <= b) if b else True)
    e = eq[m] / eq[m][0]
    r = pd.Series(e).pct_change().dropna()
    yrs = m.sum() / 252
    cg = (e[-1] ** (1 / yrs) - 1) * 100
    dd = (e / np.maximum.accumulate(e) - 1).min() * 100
    return cg, dd, r.mean() / r.std() * np.sqrt(252), cg / -dd


def main():
    st.BUY_FEE = st.SELL_FEE = 0.005
    o = load_ohlc("QQQ", "2000-01-01", None)
    p, idx = o.close, o.index
    nf, cp = load_macro(idx)
    fund = np.nan_to_num(np.asarray(load_funding(idx), float))
    al = st.compute_allocation(p.values, nf, cp, high=o.high.values, low=o.low.values, open_=o.open.values)
    rows = []
    for em in (1.7, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1, 1.0):
        for sat in (st.SAT_GAP_E, None):
            eq, fz, rv = st.simulate_net(p, al, leverage=2, funding=fund, e_max=em, sat_gap=sat)
            r = {"E_MAX": em, "saturation": "oui" if sat else "non", "frais%/an": round(fz, 2), "ordres/an": round(rv, 1)}
            for h, (a, b) in {"tout": ("2000-01-01", None), "H1": ("2000-01-01", "2012-12-31"),
                              "2013+": ("2013-01-01", None), "5ans": ("2021-11-08", None)}.items():
                cg, dd, sh, k = metr(eq, idx, a, b)
                r.update({f"{h} %/an": round(cg, 1), f"{h} DD": round(dd, 1), f"{h} Sh": round(sh, 2), f"{h} Cal": round(k, 2)})
            rows.append(r)
    for h, (a, b) in {"tout": ("2000-01-01", None), "H1": ("2000-01-01", "2012-12-31"),
                      "2013+": ("2013-01-01", None), "5ans": ("2021-11-08", None)}.items():
        cg, dd, sh, k = metr(p.values, idx, a, b)
        print(f"B&H {h}: {cg:.1f}%/an DD {dd:.1f} Sharpe {sh:.2f} Calmar {k:.2f}")
    df = pd.DataFrame(rows)
    df.to_csv("myfiles/failed_xp/pust17_emax_saturate.csv", index=False)
    pd.set_option("display.width", 300)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
