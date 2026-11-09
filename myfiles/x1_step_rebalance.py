"""Formule actuelle en x1 (PUST + cash, pas de levier), frais 0.1 %/côté : réviser l'allocation par PAS de 10 %.
  pas q    : cible arrondie au multiple de q le plus proche ; on ne trade QUE quand ce palier change
             (sinon la position dérive avec le marché) → on remet la position exactement sur le palier.
  bande sym: on rebalance quand |cible − position réelle| ≥ q (simulate_net, buy_thr = sell_thr = q).
Références : brut quotidien sans frais, bande actuelle (0.25 / 0.50).
    python -m myfiles.x1_step_rebalance
"""
import numpy as np
import pandas as pd

import src.risk_off_strategy.strategy as st
from myfiles.pust17_emax_sharpe import stats
from src.risk_off_strategy.data import load_macro, load_ohlc
from src.risk_off_strategy.strategy import compute_allocation, simulate_net

FEE = 0.001
PERIODS = {"2000-2026": ("2000-01-01", None), "2000-2012": ("2000-01-01", "2012-12-31"),
           "2013-2026": ("2013-01-01", None), "5 ans": ("2021-11-01", None)}


def sim_steps(price, alloc, q, fee=FEE):
    """x1 PUST + cash ; trade seulement quand le palier arrondi change (exec J+1)."""
    ret = pd.Series(np.asarray(price, float)).pct_change().fillna(0).values
    tgt = np.round(np.asarray(alloc, float) / q) * q
    tgt = np.concatenate([[0.0], tgt[:-1]])
    vp, vc = 0.0, 1.0
    last = 0.0
    eq = np.empty(len(ret))
    fees = n = 0
    for t in range(len(ret)):
        vp *= 1 + ret[t] - st.TER_PUST / st.ANN
        V = vp + vc
        if tgt[t] != last:
            f = fee * abs(tgt[t] * V - vp)
            fees += f / V
            V -= f
            vp, vc = tgt[t] * V, (1 - tgt[t]) * V
            last = tgt[t]
            n += 1
        eq[t] = V
    yrs = len(ret) / st.ANN
    return eq, 100 * fees / yrs, n / yrs


def main():
    o = load_ohlc("QQQ", "2000-01-01", None)
    p, idx = o["close"], o.index
    nf, cp = load_macro(idx)
    al = compute_allocation(p.values, nf, cp, high=o.high.values, low=o.low.values, open_=o.open.values)
    ret = p.pct_change().fillna(0).values
    st.BUY_FEE = st.SELL_FEE = FEE
    res = {"brut quotidien (sans frais ni bande)": (np.cumprod(1 + np.concatenate([[0], al[:-1]]) * ret), 0, 252),
           "bande actuelle 0.25 / 0.50": simulate_net(p, al, leverage=1)}
    for q in (0.05, 0.10, 0.20):
        res[f"pas de {int(q * 100)} % (trade au changement de palier)"] = sim_steps(p, al, q)
        res[f"bande symétrique {int(q * 100)} %"] = simulate_net(p, al, leverage=1, buy_thr=q, sell_thr=q)
    rows = []
    for name, (eq, fz, rv) in res.items():
        row = {"variante": name, "frais %/an": round(fz, 2), "ordres/an": round(rv, 1)}
        for pn, (a, b) in PERIODS.items():
            sh, cg, dd, k = stats(eq, idx, a, b)
            row.update({f"{pn} %/an": round(cg, 1), f"{pn} maxDD": round(dd, 1), f"{pn} Sharpe": round(sh, 2)})
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv("myfiles/x1_step_rebalance.csv", index=False)
    pd.set_option("display.width", 300)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
