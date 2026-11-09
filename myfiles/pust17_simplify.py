"""Peut-on SIMPLIFIER la formule x1.7 PUST+LQQ, vu que la bande (achat +0.50 / vente −1.00 en expo) en lisse
une grande partie ? Net de frais au tarif actuel (0.5 %/côté) ET 0.1 %/côté, NFCI+CPI conservés partout.

1) ABLATIONS (on retire un morceau à la fois) : decay_up, decay_down, filtre de pente, cap de vol au-dessus,
   bras bas (→ cash sous la MA), bras bas sans vol-target.
2) FORMULES SIMPLES À PALIERS (expo ∈ {0, 1, 1.7}), avec la bande actuelle ou sans bande (ordre à chaque
   changement de palier) :
     palier(a, b) : au-dessus de la MA ET pente montante → 1.7 si vol < a, 1.0 si vol < b, sinon 0 ;
                    sous la MA → 0.
Validation H1 2000-2012 / H2 2013-2026 + épisodes.
    python -m myfiles.pust17_simplify
"""
import numpy as np
import pandas as pd

import src.risk_off_strategy.strategy as st
from myfiles.pust17_vol_estimators import EPIS, HALVES, win
from src.risk_off_strategy.data import load_funding, load_macro, load_ohlc
from src.risk_off_strategy.strategy import E_MAX, allocation_components, compute_allocation, simulate_net


def main():
    o = load_ohlc("QQQ", "2000-01-01", None)
    p, idx = o["close"], o.index
    c = p.values
    nfci, cpi = load_macro(idx)
    fund = load_funding(idx)
    rv = st.yang_zhang_vol(o.open.values, o.high.values, o.low.values, c)
    k = allocation_components(c, rv)
    macro_off = (np.nan_to_num(np.asarray(nfci, float), nan=-9) > st.NFCI_OFF) | \
                (np.nan_to_num(np.asarray(cpi, float), nan=-9) > st.CPI_OFF)
    above_ma = c > k["s"]

    def fin(arm, up, dn):
        return np.where(macro_off, 0.0, np.clip(np.where(arm, up, dn), 0, 1))

    cand = {}
    cand["ACTUELLE"] = ("ref", compute_allocation(c, nfci, cpi, high=o.high.values, low=o.low.values, open_=o.open.values), None)
    cand["sans decay_up"] = ("ablation", fin(k["arm_above"], k["cap_vol"], k["below"]), None)
    cand["sans decay_down"] = ("ablation", fin(k["arm_above"], k["above"], st.BELOW_SCALE * k["vt"] * (c > k["s"] * (1 - st.GAP_CUTOFF))), None)
    cand["sans filtre de pente"] = ("ablation", fin(above_ma, k["above"], k["below"]), None)
    cand["sans cap de vol (haut = decay_up)"] = ("ablation", fin(k["arm_above"], k["decay_up"], k["below"]), None)
    cand["bras bas = cash"] = ("ablation", fin(k["arm_above"], k["above"], 0.0), None)
    cand["bras bas = 0.5 . decay_down (sans vol)"] = ("ablation", fin(k["arm_above"], k["above"], 0.5 * k["decay_down"]), None)
    cand["haut = cap_vol seul, bas = cash"] = ("ablation", fin(k["arm_above"], k["cap_vol"], 0.0), None)
    cand["tendance pure (1.7 au-dessus, 0 sinon)"] = ("palier", fin(k["arm_above"], 1.0, 0.0), None)
    for a in (0.15, 0.20, 0.25):
        for b in (0.25, 0.30, 0.40, None):
            if b is not None and b <= a:
                continue
            e = np.where(rv < a, E_MAX, np.where(rv < (b or 9), 1.0, 0.0))
            al = fin(k["arm_above"], e / 2, 0.0)
            nm = f"palier vol<{a:.2f}→1.7" + (f", <{b:.2f}→1" if b else "") + ", sinon 0"
            cand[nm + " | bande"] = ("palier", al, None)
            cand[nm + " | sans bande"] = ("palier", al, 0.01)

    rows = []
    for fee in (0.005, 0.001):
        st.BUY_FEE = st.SELL_FEE = fee
        for name, (fam, al, thr) in cand.items():
            kw = {} if thr is None else dict(buy_thr=thr, sell_thr=thr)
            eq, fz, rev = simulate_net(p, al, leverage=2, funding=fund, e_max=E_MAX, **kw)
            row = {"frais": f"{100 * fee:.1f}%", "variante": name, "famille": fam,
                   "frais %/an": round(fz, 2), "revis/an": round(rev, 1)}
            for hn, (a, b) in HALVES.items():
                cg, d_, kk = win(eq, idx, a, b)
                row.update({f"{hn} %/an": round(cg, 1), f"{hn} maxDD": round(d_, 1), f"{hn} Calmar": round(kk, 2)})
            r = np.diff(eq) / eq[:-1]
            row["Sharpe"] = round(r.mean() / r.std() * np.sqrt(252), 2)
            for k_, (a, b) in EPIS.items():
                m = (idx >= a) & (idx <= b)
                row[k_] = round(100 * (eq[m][-1] / eq[m][0] - 1))
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv("myfiles/pust17_simplify.csv", index=False)
    pd.set_option("display.width", 320)
    pd.set_option("display.max_rows", 200)
    cols = ["variante", "H1 Calmar", "H2 Calmar", "tout %/an", "tout maxDD", "tout Calmar", "Sharpe", "frais %/an",
            "revis/an"] + list(EPIS)
    for fee in df.frais.unique():
        g = df[df.frais == fee]
        print(f"\n=== frais {fee} / côté ===")
        print(g[cols].to_string(index=False))


if __name__ == "__main__":
    main()
