"""Faire mieux que la vol Yang-Zhang 10 j dans la stratégie x1.7 PUST+LQQ ? (frais 0.1 %/côté, NFCI+CPI conservés)

La vol entre dans compute_allocation via st.yang_zhang_vol (cap vol au-dessus de la MA + cible de vol
en dessous). On la remplace par des candidats, tous CAUSAUX (connus au close J, exécutés J+1) :
  implicite : VXN (Nasdaq-100, 2001+) / VIX, mis à l'échelle de la vol réalisée par le ratio médian
              YZ/IV des 250 séances PRÉCÉDENTES (prime de variance retirée sans look-ahead) ;
              seul, max(YZ, IV), moyenne ; avant 2001 → YZ.
  HAR       : vol multi-horizon sqrt(a·v1 + b·v5 + c·v22) sur la variance journalière Garman-Klass + gap.
  YZ court  : fenêtre 5 j.
  baissière : max(YZ, semi-vol des baisses 10 j ×√2).
  volume    : YZ × f quand volume / médiane 50 j (passée) > T  (accélérateur), ou YZ × clip(ratio,1,2)^0.5.
Validation : H1 2000-2012 / H2 2013-2026, un réglage ne compte que s'il bat la YZ sur LES DEUX (Calmar) ;
validation croisée par famille (réglé sur une moitié, lu sur l'autre) ; épisodes de crise.
    python -m myfiles.pust17_vol_estimators
"""
import os

import numpy as np
import pandas as pd
import yfinance as yf

import src.risk_off_strategy.strategy as st
from src.risk_off_strategy.data import load_funding, load_macro
from src.risk_off_strategy.strategy import E_MAX, compute_allocation, simulate_net

st.BUY_FEE = st.SELL_FEE = 0.001
IV_FILE = "data/vix/implied_vol.parquet"
YZ = st.yang_zhang_vol
HALVES = {"H1": ("2000-01-01", "2012-12-31"), "H2": ("2013-01-01", None), "tout": ("2000-01-01", None)}
EPIS = {"2000-02": ("2000-03-01", "2002-10-31"), "2008": ("2007-10-01", "2009-03-31"), "2011": ("2011-07-01", "2011-10-31"),
        "2015-16": ("2015-08-01", "2016-02-29"), "2018T4": ("2018-10-01", "2018-12-31"), "2020": ("2020-02-15", "2020-04-30"),
        "2022": ("2021-11-15", "2022-12-31"), "2025": ("2025-02-15", "2025-05-31")}


def load_iv():
    if not os.path.exists(IV_FILE):
        os.makedirs(os.path.dirname(IV_FILE), exist_ok=True)
        d = {c: yf.download(t, start="1999-01-01", progress=False, auto_adjust=False)["Close"].squeeze()
             for c, t in (("vxn", "^VXN"), ("vix", "^VIX"))}
        pd.DataFrame(d).to_parquet(IV_FILE)
    return pd.read_parquet(IV_FILE)


def win(eq, idx, a, b):
    m = (idx >= pd.Timestamp(a)) & (idx <= (pd.Timestamp(b) if b else idx[-1]))
    e = eq[m] / eq[m][0]
    cagr = e[-1] ** (252 / m.sum()) - 1
    dd = (e / np.maximum.accumulate(e) - 1).min()
    return 100 * cagr, 100 * dd, cagr / abs(dd)


def main():
    d = pd.read_parquet("data/QQQ.parquet").loc["2000-01-01":].dropna().sort_index()
    idx, p = d.index, d["close"]
    o, h, lw, c, vol = (d[k] for k in ("open", "high", "low", "close", "volume"))
    nfci, cpi = load_macro(idx)
    fund = load_funding(idx)
    kw = dict(high=h.values, low=lw.values, open_=o.values)
    yz = pd.Series(YZ(o.values, h.values, lw.values, c.values), idx)
    # variance journalière Garman-Klass + gap overnight
    gk = (np.log(o / c.shift(1)) ** 2 + 0.5 * np.log(h / lw) ** 2 - (2 * np.log(2) - 1) * np.log(c / o) ** 2).fillna(0)
    v1, v5, v22 = gk * 252, gk.rolling(5).mean() * 252, gk.rolling(22, min_periods=5).mean() * 252
    r = np.log(c).diff()
    semi = np.sqrt((np.minimum(r, 0) ** 2).rolling(10, min_periods=5).mean() * 2 * 252)
    vr = vol / vol.rolling(50, min_periods=20).median().shift(1)
    iv = load_iv().reindex(idx).ffill(limit=3) / 100
    print(f"VXN {iv.vxn.first_valid_index().date()} →, VIX {iv.vix.first_valid_index().date()} →, "
          f"ratio médian YZ/VXN {np.nanmedian(yz / iv.vxn):.2f}")

    def scaled(s):
        k = (yz / s).rolling(250, min_periods=60).median().shift(1)
        out = s * k
        return out.where(out.notna(), yz)

    cand = {"YZ 10 j (actuel)": ("base", None, yz)}
    for nm in ("vxn", "vix"):
        s = scaled(iv[nm])
        cand[f"{nm.upper()} seul"] = (f"{nm} seul", 0, s)
        cand[f"max(YZ,{nm.upper()})"] = (f"{nm} max", 0, np.maximum(yz, s))
        cand[f"moy(YZ,{nm.upper()})"] = (f"{nm} moy", 0, 0.5 * (yz + s))
    for wts in ((0.2, 0.4, 0.4), (0.33, 0.33, 0.34), (0.1, 0.5, 0.4), (0.4, 0.4, 0.2)):
        cand[f"HAR {wts}"] = ("HAR", str(wts), np.sqrt(wts[0] * v1 + wts[1] * v5 + wts[2] * v22).clip(lower=1e-6))
    cand["YZ 5 j"] = ("YZ court", 5, pd.Series(YZ(o.values, h.values, lw.values, c.values, w=5), idx))
    cand["YZ 7 j"] = ("YZ court", 7, pd.Series(YZ(o.values, h.values, lw.values, c.values, w=7), idx))
    cand["max(YZ, semi-vol)"] = ("baissière", 0, np.maximum(yz, semi.fillna(0)))
    for T in (1.5, 2.0, 2.5):
        for f in (1.2, 1.5):
            cand[f"volume >{T}× → YZ×{f}"] = ("volume accél.", f"{T}/{f}", yz.where(~(vr > T), yz * f))
    for ex in (0.3, 0.5, 1.0):
        cand[f"YZ×clip(vol ratio)^{ex}"] = ("volume continu", ex, yz * vr.clip(1, 2).fillna(1) ** ex)

    rows = []
    for name, (fam, prm, v) in cand.items():
        vv = np.asarray(v, float)
        st.yang_zhang_vol = lambda *a, _v=vv, **k: _v
        al = compute_allocation(p.values, nfci, cpi, **kw)
        st.yang_zhang_vol = YZ
        eq, fees, rev = simulate_net(p, al, leverage=2, funding=fund, e_max=E_MAX)
        row = {"candidat": name, "famille": fam, "param": prm, "frais %/an": round(fees, 2), "revis/an": round(rev, 1)}
        for hn, (a, b) in HALVES.items():
            cg, dd, k = win(eq, idx, a, b)
            row.update({f"{hn} %/an": round(cg, 1), f"{hn} maxDD": round(dd, 1), f"{hn} Calmar": round(k, 2)})
        for k_, (a, b) in EPIS.items():
            m = (idx >= a) & (idx <= b)
            row[k_] = round(100 * (eq[m][-1] / eq[m][0] - 1))
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv("myfiles/pust17_vol_estimators.csv", index=False)
    pd.set_option("display.width", 280)
    cols = ["candidat", "H1 Calmar", "H2 Calmar", "tout %/an", "tout maxDD", "tout Calmar", "frais %/an", "revis/an"] + list(EPIS)
    print(df[cols].to_string(index=False))
    ref = df.iloc[0]
    both = df[(df["H1 Calmar"] > ref["H1 Calmar"]) & (df["H2 Calmar"] > ref["H2 Calmar"])]
    print(f"\n=== Battent la YZ 10 j sur LES DEUX moitiés : {len(both)} / {len(df) - 1} ===")
    print(both[cols].to_string(index=False) if len(both) else "  aucun")
    print("\n=== Validation croisée par famille (meilleur réglage sur une moitié, lu sur l'autre) ===")
    for fam in df.famille.unique()[1:]:
        g = df[df.famille == fam]
        for fit, test in (("H1", "H2"), ("H2", "H1")):
            b = g.loc[g[f"{fit} Calmar"].idxmax()]
            print(f"  {fam:15s} réglé {fit} → {b.candidat:28s} {test} Calmar {b[f'{test} Calmar']:.2f} "
                  f"(YZ {ref[f'{test} Calmar']:.2f})  maxDD {b[f'{test} maxDD']:.1f} (YZ {ref[f'{test} maxDD']:.1f})")


if __name__ == "__main__":
    main()
