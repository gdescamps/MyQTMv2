"""E_MAX 1.5 + saturation : balayage des seuils de réallocation (espace exposition).
Actuel : achat 0.50 / vente 1.00. Frais 0.5 %/côté.
    python -m myfiles.pust15_sat_band_sweep
"""
import numpy as np
import pandas as pd

import src.risk_off_strategy.strategy as st
from myfiles.pust17_emax_saturate import metr
from src.risk_off_strategy.data import load_funding, load_macro, load_ohlc


def main():
    import sys
    st.BUY_FEE = st.SELL_FEE = float(sys.argv[1]) if len(sys.argv) > 1 else 0.005
    o = load_ohlc("QQQ", "2000-01-01", None)
    p, idx = o.close, o.index
    nf, cp = load_macro(idx)
    fund = np.nan_to_num(np.asarray(load_funding(idx), float))
    al = st.compute_allocation(p.values, nf, cp, high=o.high.values, low=o.low.values, open_=o.open.values)
    hz = {"tout": ("2000-01-01", None), "H1": ("2000-01-01", "2012-12-31"), "H2": ("2013-01-01", None),
          "10ans": (str(idx[-2520].date()), None), "5ans": (str(idx[-1260].date()), None)}
    rows = []
    for b in (0.10, 0.20, 0.30, 0.40, 0.50):
        for s in (0.30, 0.40, 0.50, 0.60, 0.80, 1.00):
            eq, fz, rv = st.simulate_net(p, al, leverage=2, funding=fund, e_max=1.5, sat_gap=st.SAT_GAP_E, buy_thr=b, sell_thr=s)
            r = {"achat": b, "vente": s, "frais": round(fz, 2), "ordres": round(rv, 1)}
            for h, (x, y) in hz.items():
                cg, dd, sh, k = metr(eq, idx, x, y)
                r.update({f"{h} %/an": round(cg, 1), f"{h} DD": round(dd, 1), f"{h} Sh": round(sh, 2), f"{h} Cal": round(k, 2)})
            rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(f"myfiles/pust15_sat_band_sweep_{st.BUY_FEE*1000:.0f}bp10.csv", index=False)
    pd.set_option("display.width", 320)
    cols = ["achat", "vente", "frais", "ordres", "tout %/an", "tout DD", "tout Sh", "tout Cal", "H1 Cal", "H2 Cal",
            "10ans %/an", "10ans Cal", "5ans %/an", "5ans DD"]
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
