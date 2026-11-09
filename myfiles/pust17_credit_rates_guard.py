"""Garde-fous candidats crédit / choc de taux sur la stratégie déployée x1.7 PUST+LQQ (net, frais 0.1 %/côté).

Familles (signal décalé de publication : FRED quotidien J+1, via data.load_credit_rates) :
  credit_lvl : BAA10Y > T                       (crise de crédit / récession)
  credit_mom : BAA10Y − BAA10Y(63 séances) > T  (élargissement rapide)
  rate_shock : Δ6 mois du taux 2 ans > T        (choc de taux type 2022)
Action : cash (alloc=0, comme NFCI/CPI) ou cap1 (expo ≤ 1 → PUST seul, alloc ≤ 0.5).
Macro existante : base = NFCI+CPI ; aussi sans NFCI, sans CPI, sans macro (remplacement ?).
Validation : seuil choisi sur une moitié (Calmar), lu sur l'autre — H1 2000-2012 / H2 2013-2026.
    python -m myfiles.pust17_credit_rates_guard
"""
import numpy as np
import pandas as pd

import src.risk_off_strategy.strategy as st
from src.risk_off_strategy.data import load_credit_rates, load_funding, load_macro, load_ohlc
from src.risk_off_strategy.strategy import E_MAX, compute_allocation, simulate_net

st.BUY_FEE = st.SELL_FEE = 0.001
HALVES = {"H1": ("2000-01-01", "2012-12-31"), "H2": ("2013-01-01", None), "tout": ("2000-01-01", None)}
FAM = {"credit_lvl": [2.5, 2.75, 3.0, 3.25, 3.5, 4.0],
       "credit_mom": [0.3, 0.5, 0.75, 1.0],
       "rate_shock": [1.0, 1.25, 1.5, 1.75, 2.0, 2.5]}


def win(eq, idx, a, b):
    m = (idx >= pd.Timestamp(a)) & (idx <= (pd.Timestamp(b) if b else idx[-1]))
    e = eq[m] / eq[m][0]
    yrs = m.sum() / 252
    cagr = e[-1] ** (1 / yrs) - 1
    dd = (e / np.maximum.accumulate(e) - 1).min()
    return 100 * cagr, 100 * dd, cagr / abs(dd)


def main():
    o = load_ohlc("QQQ", "2000-01-01", None)
    p, idx = o["close"], o.index
    nfci, cpi = load_macro(idx)
    fund = load_funding(idx)
    baa, y2c = load_credit_rates(idx)
    baa_s = pd.Series(baa, idx)
    sig = {"credit_lvl": baa_s.values, "credit_mom": (baa_s - baa_s.shift(63)).values, "rate_shock": y2c}
    kw = dict(high=o["high"].values, low=o["low"].values, open_=o["open"].values)
    macros = {"NFCI+CPI": (nfci, cpi), "NFCI seul": (nfci, None), "CPI seul": (None, cpi), "aucune": (None, None)}
    rows = []

    def run(macro, fam, thr, act):
        n_, c_ = macros[macro]
        a = compute_allocation(p.values, n_, c_, **kw)
        if fam:
            on = np.nan_to_num(sig[fam], nan=-9) > thr
            a = np.where(on, 0.0 if act == "cash" else np.minimum(a, 0.5), a)
            pct_on = 100 * on.mean()
        else:
            pct_on = 0.0
        eq, fees, rev = simulate_net(p, a, leverage=2, funding=fund, e_max=E_MAX)
        r = {"macro": macro, "famille": fam or "-", "seuil": thr, "action": act, "% jours actifs": round(pct_on, 1),
             "frais %/an": round(fees, 2)}
        for h, (x, y) in HALVES.items():
            c, d, cal = win(eq, idx, x, y)
            r.update({f"{h} %/an": round(c, 1), f"{h} maxDD": round(d, 1), f"{h} Calmar": round(cal, 2)})
        rows.append(r)

    for macro in macros:
        run(macro, None, None, "-")
        for fam, ths in FAM.items():
            for t in ths:
                for act in ("cash", "cap1"):
                    run(macro, fam, t, act)
    df = pd.DataFrame(rows)
    df.to_csv("myfiles/pust17_credit_rates_guard.csv", index=False)
    pd.set_option("display.width", 250)
    cols = ["macro", "famille", "seuil", "action", "% jours actifs", "H1 %/an", "H1 maxDD", "H1 Calmar",
            "H2 %/an", "H2 maxDD", "H2 Calmar", "tout %/an", "tout maxDD", "tout Calmar", "frais %/an"]
    print("=== Références (sans nouveau garde-fou) ===")
    print(df[df.famille == "-"][cols].to_string(index=False))
    print("\n=== Validation croisée : seuil choisi sur une moitié (Calmar max), lu sur l'autre ===")
    out = []
    for macro in macros:
        ref = df[(df.macro == macro) & (df.famille == "-")].iloc[0]
        for fam in FAM:
            for act in ("cash", "cap1"):
                g = df[(df.macro == macro) & (df.famille == fam) & (df.action == act)]
                for fit, test in (("H1", "H2"), ("H2", "H1")):
                    b = g.loc[g[f"{fit} Calmar"].idxmax()]
                    out.append({"macro": macro, "famille": fam, "action": act, "réglé sur": fit, "seuil": b.seuil,
                                f"Calmar test": b[f"{test} Calmar"], "Calmar réf test": ref[f"{test} Calmar"],
                                "maxDD test": b[f"{test} maxDD"], "maxDD réf test": ref[f"{test} maxDD"],
                                "%/an test": b[f"{test} %/an"], "%/an réf test": ref[f"{test} %/an"]})
    o2 = pd.DataFrame(out)
    o2["gain Calmar"] = (o2["Calmar test"] - o2["Calmar réf test"]).round(2)
    print(o2.to_string(index=False))
    o2.to_csv("myfiles/pust17_credit_rates_guard_cv.csv", index=False)
    print("\n=== Meilleurs sur tout l'historique (macro NFCI+CPI) ===")
    print(df[df.macro == "NFCI+CPI"].sort_values("tout Calmar", ascending=False)[cols].head(12).to_string(index=False))


if __name__ == "__main__":
    main()
