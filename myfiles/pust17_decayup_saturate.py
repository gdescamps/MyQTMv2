"""decay_up est-il toujours inutile AVEC la saturation au plafond ?
Re-applique l'ancien trim de sur-extension au bras « au-dessus » :
    decay_up = clip(1 − max(0, gap − 0.15)/0.20, FLOOR, 1)   (FLOOR=0.4 = valeur retirée en 2026-11)
Grille : E_MAX 1.7/1.6/1.5 × saturation oui/non × decay_up non/oui (+ planchers 0.6, 0.8). Frais 0.5 %/côté.
    python -m myfiles.pust17_decayup_saturate
"""
import numpy as np
import pandas as pd

import src.risk_off_strategy.strategy as st
from myfiles.pust17_emax_saturate import metr
from src.risk_off_strategy.data import load_funding, load_macro, load_ohlc

GAP2_START, GAP2_SPAN = 0.15, 0.20


def alloc_decay_up(o, nf, cp, floor):
    p = o.close.values
    rv = st.yang_zhang_vol(o.open.values, o.high.values, o.low.values, p)
    c = st.allocation_components(p, rv)
    above = c["above"]
    if floor is not None:
        above = above * np.clip(1 - np.maximum(0, c["gap"] - GAP2_START) / GAP2_SPAN, floor, 1)
    a = np.clip(np.where(c["arm_above"], above, c["below"]), 0, 1)
    a = np.where(np.nan_to_num(np.asarray(nf, float), nan=-9) > st.NFCI_OFF, 0.0, a)
    return np.where(np.nan_to_num(np.asarray(cp, float), nan=-9) > st.CPI_OFF, 0.0, a)


def main():
    st.BUY_FEE = st.SELL_FEE = 0.005
    o = load_ohlc("QQQ", "2000-01-01", None)
    p, idx = o.close, o.index
    nf, cp = load_macro(idx)
    fund = np.nan_to_num(np.asarray(load_funding(idx), float))
    ref = st.compute_allocation(p.values, nf, cp, high=o.high.values, low=o.low.values, open_=o.open.values)
    als = {f: alloc_decay_up(o, nf, cp, f) for f in (None, 0.4, 0.6, 0.8)}
    assert np.allclose(als[None], ref)
    hz = {"tout": ("2000-01-01", None), "H1": ("2000-01-01", "2012-12-31"), "H2": ("2013-01-01", None),
          "10ans": (str(idx[-2520].date()), None), "5ans": (str(idx[-1260].date()), None)}
    rows = []
    for em in (1.7, 1.6, 1.5):
        for sat in (st.SAT_GAP_E, None):
            for f, al in als.items():
                eq, fz, rv = st.simulate_net(p, al, leverage=2, funding=fund, e_max=em, sat_gap=sat)
                r = {"E_MAX": em, "satur.": "oui" if sat else "non", "decay_up": "non" if f is None else f"plancher {f}",
                     "frais": round(fz, 2), "ordres": round(rv, 1)}
                for h, (a, b) in hz.items():
                    cg, dd, sh, k = metr(eq, idx, a, b)
                    r.update({f"{h} %/an": round(cg, 1), f"{h} DD": round(dd, 1), f"{h} Cal": round(k, 2)})
                    if h == "tout":
                        r["tout Sh"] = round(sh, 2)
                rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv("myfiles/pust17_decayup_saturate.csv", index=False)
    pd.set_option("display.width", 320)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
