"""Seuillage SÉLECTIF (x1.7 PUST+LQQ) : autoriser la réallocation quand l'écart vient UNIQUEMENT de decay_up.

Cible E = min(2.alloc, 1.7) ; cible SANS decay_up En (même formule, decay_up = 1). À chaque jour (exec J+1) :
  bande normale : achat si E − e ≥ 0.50, vente si e − E ≥ 1.00 (actuel), ou passage cash.
  sinon, règle sélective — l'écart est dû à decay_up si, SANS decay_up, la bande normale ne déclencherait pas
  (En dans la bande autour de e) mais E s'écarte de e d'au moins f (seuil fin) :
      V « decay_up seul »     : |E − e| ≥ f  ET  En dans la bande normale
      V « decay_up actif »    : |E − e| ≥ f  ET  decay_up < 1 (aujourd'hui) ET bras haut
  f ∈ {0.10, 0.20, 0.30} en expo. Variantes « vente seule » (on n'autorise que la baisse due à decay_up).
Frais 0.5 % et 0.1 %/côté. Validation H1 2000-2012 / H2 2013-2026 + épisodes.
    python -m myfiles.pust17_selective_band
"""
import numpy as np
import pandas as pd

import src.risk_off_strategy.strategy as st
from myfiles.pust17_emax_sharpe import stats
from myfiles.pust17_vol_estimators import EPIS
from src.risk_off_strategy.data import load_funding, load_macro, load_ohlc
from src.risk_off_strategy.strategy import E_MAX, allocation_components, compute_allocation, simulate_net


def sim(price, E, En, du_on, fund, mode, f, sell_only):
    p = np.asarray(price, float)
    ret = pd.Series(p).pct_change().fillna(0).values
    n = len(p)
    r_pust = ret - st.TER_PUST / st.ANN
    r_lqq = 2 * ret - (fund + st.SWAP_SPREAD + st.TER_LQQ) / st.ANN
    lag = lambda x: np.concatenate([[x[0] * 0], x[:-1]])  # noqa: E731
    Et, Ent, dut = lag(E), lag(En), lag(du_on.astype(float)) > 0
    vp = vl = 0.0
    vc = 1.0
    eq = np.empty(n)
    fees = 0.0
    revis = sel = 0
    for t in range(n):
        vp *= 1 + r_pust[t]
        vl *= 1 + r_lqq[t]
        V = vp + vl + vc
        e = (vp + 2 * vl) / V
        et = Et[t]
        force = (et == 0 and e > 1e-9) or (e == 0 and et > 1e-9)
        normal = force or (et - e >= st.BUY_THR_E) or (e - et >= st.SELL_THR_E)
        trade = normal
        if not normal and mode and abs(et - e) >= f and (not sell_only or et < e):
            if mode == "seul":
                en = Ent[t]
                trade = not ((en - e >= st.BUY_THR_E) or (e - en >= st.SELL_THR_E))
            else:
                trade = bool(dut[t])
            sel += trade
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
    yrs = n / st.ANN
    return eq, 100 * fees / yrs, revis / yrs, sel / yrs


def main():
    o = load_ohlc("QQQ", "2000-01-01", None)
    p, idx = o["close"], o.index
    c = p.values
    nf, cp = load_macro(idx)
    fund = np.nan_to_num(np.asarray(load_funding(idx), float))
    al = compute_allocation(c, nf, cp, high=o.high.values, low=o.low.values, open_=o.open.values)
    rv = st.yang_zhang_vol(o.open.values, o.high.values, o.low.values, c)
    k = allocation_components(c, rv)
    macro = (np.nan_to_num(np.asarray(nf, float), nan=-9) > st.NFCI_OFF) | (np.nan_to_num(np.asarray(cp, float), nan=-9) > st.CPI_OFF)
    al_nodu = np.where(macro, 0, np.clip(np.where(k["arm_above"], k["cap_vol"], k["below"]), 0, 1))
    E, En = np.minimum(2 * al, E_MAX), np.minimum(2 * al_nodu, E_MAX)
    du_on = k["arm_above"] & (k["decay_up"] < 1)
    print(f"jours decay_up actif : {100 * du_on.mean():.1f} %")
    rows = []
    for fee in (0.005, 0.001):
        st.BUY_FEE = st.SELL_FEE = fee
        e_ref = simulate_net(p, al, leverage=2, funding=fund, e_max=E_MAX)[0]
        cfgs = [("ACTUELLE (bande normale)", None, 0, False), ("sans decay_up (bande normale)", "nodu", 0, False)]
        for mode in ("seul", "actif"):
            for f in (0.10, 0.20, 0.30):
                for so in (False, True):
                    cfgs.append((f"decay_up {mode} f={f}" + (" vente seule" if so else ""), mode, f, so))
        for name, mode, f, so in cfgs:
            if mode == "nodu":
                eq, fz, rvs, sl = sim(c, En, En, du_on, fund, None, 0, False)
            else:
                eq, fz, rvs, sl = sim(c, E, En, du_on, fund, mode, f, so)
            if name.startswith("ACTUELLE"):
                assert abs(eq[-1] / e_ref[-1] - 1) < 1e-9
            row = {"frais": f"{fee * 100:.1f}%", "variante": name, "frais %/an": round(fz, 2), "revis/an": round(rvs, 1),
                   "dont sélectifs/an": round(sl, 1)}
            for h, (x, y) in {"tout": ("2000-01-01", None), "H1": ("2000-01-01", "2012-12-31"), "H2": ("2013-01-01", None)}.items():
                sh, cg, dd, kk = stats(eq, idx, x, y)
                row.update({f"{h} %/an": round(cg, 1), f"{h} maxDD": round(dd, 1), f"{h} Calmar": round(kk, 2), f"{h} Sharpe": round(sh, 2)})
            for kk_, (x, y) in EPIS.items():
                m = (idx >= x) & (idx <= y)
                row[kk_] = round(100 * (eq[m][-1] / eq[m][0] - 1))
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv("myfiles/pust17_selective_band.csv", index=False)
    pd.set_option("display.width", 330)
    cols = ["variante", "H1 Calmar", "H2 Calmar", "tout %/an", "tout maxDD", "tout Calmar", "tout Sharpe", "frais %/an",
            "revis/an", "dont sélectifs/an"] + list(EPIS)
    for fee in df.frais.unique():
        print(f"\n=== frais {fee} ===")
        print(df[df.frais == fee][cols].to_string(index=False))


if __name__ == "__main__":
    main()
