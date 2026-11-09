"""« Complément au plafond » (x1.7 PUST+LQQ) : quand la cible est au plafond E_MAX (marché calme, au-dessus de
la MA montante), on autorise l'achat jusqu'à 1.7 dès que l'écart cible − réelle ≥ f, au lieu de 0.50.
Corrige le blocage vers 1.2-1.5 (ex. nov. 2020 → mars 2021). Reste de la bande inchangé (achat 0.50 / vente 1.00).
Variante « surplomb » : quand la bande normale déclenche un achat, viser min(cible + 0.2, 1.7).
Frais 0.1 %/côté (tarif abaissé) et 0.5 % (rappel). Validation H1/H2 + épisodes.
    python -m myfiles.pust17_topup_cap
"""
import numpy as np
import pandas as pd

import src.risk_off_strategy.strategy as st
from myfiles.pust17_emax_sharpe import stats
from myfiles.pust17_vol_estimators import EPIS
from src.risk_off_strategy.data import load_funding, load_macro, load_ohlc
from src.risk_off_strategy.strategy import E_MAX, compute_allocation, simulate_net


def sim(price, E, fund, f=None, cap_lvl=E_MAX, overshoot=0.0):
    p = np.asarray(price, float)
    ret = pd.Series(p).pct_change().fillna(0).values
    n = len(p)
    r_pust = ret - st.TER_PUST / st.ANN
    r_lqq = 2 * ret - (fund + st.SWAP_SPREAD + st.TER_LQQ) / st.ANN
    Et = np.concatenate([[0.0], E[:-1]])
    vp = vl = 0.0
    vc = 1.0
    eq, ex = np.empty(n), np.empty(n)
    fees = 0.0
    revis = top = 0
    for t in range(n):
        vp *= 1 + r_pust[t]
        vl *= 1 + r_lqq[t]
        V = vp + vl + vc
        e = (vp + 2 * vl) / V
        et = Et[t]
        force = (et == 0 and e > 1e-9) or (e == 0 and et > 1e-9)
        trade = force or (et - e >= st.BUY_THR_E) or (e - et >= st.SELL_THR_E)
        if trade and et > e and overshoot:
            et = min(et + overshoot, E_MAX)
        if not trade and f is not None and et >= cap_lvl - 1e-9 and et - e >= f:
            trade = True
            top += 1
        if trade:
            wl = max(0.0, et - 1.0)
            wp = et if et <= 1.0 else 2.0 - et
            tp, tl = wp * V, wl * V
            fee = (st.SELL_FEE * (max(0.0, vp - tp) + max(0.0, vl - tl))
                   + st.BUY_FEE * (max(0.0, tp - vp) + max(0.0, tl - vl)))
            fees += fee / V
            V -= fee
            vp, vl, vc = wp * V, wl * V, max(0.0, 1.0 - et) * V
            revis += 1
        eq[t] = V
        ex[t] = (vp + 2 * vl) / V
    yrs = n / st.ANN
    return eq, 100 * fees / yrs, revis / yrs, top / yrs, ex


def main():
    o = load_ohlc("QQQ", "2000-01-01", None)
    p, idx = o["close"], o.index
    nf, cp = load_macro(idx)
    fund = np.nan_to_num(np.asarray(load_funding(idx), float))
    al = compute_allocation(p.values, nf, cp, high=o.high.values, low=o.low.values, open_=o.open.values)
    E = np.minimum(2 * al, E_MAX)
    rows = []
    for fee in (0.001, 0.005):
        st.BUY_FEE = st.SELL_FEE = fee
        ref = simulate_net(p, al, leverage=2, funding=fund, e_max=E_MAX)[0]
        cfgs = [("ACTUELLE", {})]
        cfgs += [(f"complément au plafond, écart ≥ {f:.2f}", dict(f=f)) for f in (0.05, 0.10, 0.20, 0.30)]
        cfgs += [(f"complément si cible ≥ 1.5, écart ≥ {f:.2f}", dict(f=f, cap_lvl=1.5)) for f in (0.10, 0.20)]
        cfgs += [("surplomb +0.2 au rachat", dict(overshoot=0.2)),
                 ("surplomb +0.2 + complément 0.10", dict(overshoot=0.2, f=0.10))]
        for name, kw in cfgs:
            eq, fz, rv, tp, ex = sim(p.values, E, fund, **kw)
            if name == "ACTUELLE":
                assert abs(eq[-1] / ref[-1] - 1) < 1e-9
            row = {"frais": f"{fee * 100:.1f}%", "variante": name, "frais %/an": round(fz, 2), "ordres/an": round(rv, 1),
                   "dont compléments/an": round(tp, 1), "expo moy. quand cible=1.7": round(ex[1:][E[:-1] >= E_MAX - 1e-9].mean(), 2)}
            for h, (x, y) in {"tout": ("2000-01-01", None), "H1": ("2000-01-01", "2012-12-31"), "H2": ("2013-01-01", None),
                              "5 ans": ("2021-11-01", None)}.items():
                sh, cg, dd, k = stats(eq, idx, x, y)
                row.update({f"{h} %/an": round(cg, 1), f"{h} maxDD": round(dd, 1), f"{h} Calmar": round(k, 2), f"{h} Sharpe": round(sh, 2)})
            m = (idx >= "2020-10-01") & (idx <= "2021-03-31")
            row["T4 2020-T1 2021"] = round(100 * (eq[m][-1] / eq[m][0] - 1))
            for k_, (x, y) in EPIS.items():
                m = (idx >= x) & (idx <= y)
                row[k_] = round(100 * (eq[m][-1] / eq[m][0] - 1))
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv("myfiles/pust17_topup_cap.csv", index=False)
    pd.set_option("display.width", 340)
    cols = ["variante", "H1 Calmar", "H2 Calmar", "tout %/an", "tout maxDD", "tout Calmar", "tout Sharpe", "5 ans %/an",
            "5 ans maxDD", "frais %/an", "ordres/an", "dont compléments/an", "expo moy. quand cible=1.7", "T4 2020-T1 2021"] + list(EPIS)
    for fee in df.frais.unique():
        print(f"\n=== frais {fee} ===")
        print(df[df.frais == fee][cols].to_string(index=False))


if __name__ == "__main__":
    main()
