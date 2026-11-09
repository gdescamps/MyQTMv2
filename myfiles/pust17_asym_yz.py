"""YZ 10 j × facteur d'asymétrie (vol pondérée baisses/hausses ÷ même vol sans asymétrie) — suite de pust17_asym_vol.
Le facteur > 1 quand les baisses dominent les 10 dernières séances, < 1 après une série de hausses.
    python -m myfiles.pust17_asym_yz
"""
import numpy as np
import pandas as pd

import src.risk_off_strategy.strategy as st
from myfiles.pust17_vol_estimators import EPIS, HALVES, win
from src.risk_off_strategy.data import load_funding, load_macro
from src.risk_off_strategy.strategy import E_MAX, compute_allocation, simulate_net

st.BUY_FEE = st.SELL_FEE = 0.001
YZ = st.yang_zhang_vol


def main():
    df = pd.read_parquet("data/QQQ.parquet").loc["2000-01-01":].dropna().sort_index()
    idx, p = df.index, df["close"]
    o, h, lw, c = (df[k] for k in ("open", "high", "low", "close"))
    nfci, cpi = load_macro(idx)
    fund = load_funding(idx)
    kw = dict(high=h.values, low=lw.values, open_=o.values)
    yz = YZ(o.values, h.values, lw.values, c.values)
    d = (np.log(o / c.shift(1)) ** 2 + 0.5 * np.log(h / lw) ** 2 - (2 * np.log(2) - 1) * np.log(c / o) ** 2).fillna(0)
    down = (c.pct_change().fillna(0) < 0).values
    base = d.rolling(10, min_periods=5).mean()
    rows = []
    for k in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
        fac = np.sqrt(pd.Series(d.values * np.where(down, 1 + k, 1 - k), idx).rolling(10, min_periods=5).mean() / base)
        v = yz * fac.fillna(1).values
        st.yang_zhang_vol = lambda *a, _v=v, **kk: _v
        al = compute_allocation(p.values, nfci, cpi, **kw)
        st.yang_zhang_vol = YZ
        eq, fees, rev = simulate_net(p, al, leverage=2, funding=fund, e_max=E_MAX)
        row = {"candidat": f"YZ asym k={k}" + (" (= actuel)" if k == 0 else ""), "frais %/an": round(fees, 2), "revis/an": round(rev, 1),
               "facteur min/max": f"{np.nanmin(fac):.2f}/{np.nanmax(fac):.2f}"}
        for hn, (a, b) in HALVES.items():
            cg, dd, kk = win(eq, idx, a, b)
            row.update({f"{hn} %/an": round(cg, 1), f"{hn} maxDD": round(dd, 1), f"{hn} Calmar": round(kk, 2)})
        for k_, (a, b) in EPIS.items():
            m = (idx >= a) & (idx <= b)
            row[k_] = round(100 * (eq[m][-1] / eq[m][0] - 1))
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv("myfiles/pust17_asym_yz.csv", index=False)
    pd.set_option("display.width", 280)
    cols = ["candidat", "H1 %/an", "H1 maxDD", "H1 Calmar", "H2 %/an", "H2 maxDD", "H2 Calmar", "tout %/an", "tout maxDD",
            "tout Calmar", "frais %/an", "revis/an"] + list(EPIS)
    print(out[cols].to_string(index=False))


if __name__ == "__main__":
    main()
