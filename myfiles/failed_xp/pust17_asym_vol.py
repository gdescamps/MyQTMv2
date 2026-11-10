"""Vol ASYMÉTRIQUE « inventée » : les baisses pèsent plus, une série de hausses fait fondre la vol plus vite.

Variance journalière d_t = gap overnight² + Garman-Klass intraday (même matière que la Yang-Zhang).
  pondérée  : vol² = moyenne 10 j de d_t × w_t,  w_t = 1+k si séance baissière (close/close), 1−k si haussière
              (k=0 → même estimateur sans asymétrie = témoin) ; k ∈ {0.2, 0.4, 0.6, 0.8}
  mémoire   : var_t = λ_t·var_{t−1} + (1−λ_t)·d_t, λ = λ_baisse les jours baissiers (oubli lent),
              λ_hausse les jours haussiers (oubli rapide)
Injectée à la place de la YZ 10 j dans compute_allocation (x1.7 PUST+LQQ, frais 0.1 %/côté, NFCI+CPI).
Validation : H1 2000-2012 / H2 2013-2026 (battre la YZ sur les deux), validation croisée, épisodes.
    python -m myfiles.failed_xp.pust17_asym_vol
"""
import numpy as np
import pandas as pd

import src.risk_off_strategy.strategy as st
from myfiles.pust17_vol_estimators import EPIS, HALVES, win
from src.risk_off_strategy.data import load_funding, load_macro
from src.risk_off_strategy.strategy import E_MAX, compute_allocation, simulate_net

st.BUY_FEE = st.SELL_FEE = 0.001
YZ = st.yang_zhang_vol


def ewma_asym(d, down, lam_dn, lam_up, init):
    out = np.empty(len(d))
    v = init
    for i, (x, dn) in enumerate(zip(d, down)):
        lam = lam_dn if dn else lam_up
        v = lam * v + (1 - lam) * x
        out[i] = v
    return out


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
    cand = {"YZ 10 j (actuel)": ("base", None, yz)}
    for k in (0.0, 0.2, 0.4, 0.6, 0.8):
        w = np.where(down, 1 + k, 1 - k)
        v = np.sqrt(pd.Series(d.values * w, idx).rolling(10, min_periods=5).mean() * 252).bfill().clip(lower=1e-6)
        cand[f"pondérée k={k}" + (" (témoin)" if k == 0 else "")] = ("pondérée" if k else "témoin", k, v.values)
    for ld, lu in ((0.90, 0.90), (0.92, 0.80), (0.90, 0.75), (0.85, 0.70), (0.94, 0.85), (0.90, 0.60)):
        v = np.sqrt(np.maximum(ewma_asym(d.values, down, ld, lu, d.values[:20].mean()) * 252, 1e-12))
        fam = "mémoire" if ld != lu else "témoin"
        cand[f"mémoire λ↓{ld} λ↑{lu}" + (" (témoin)" if ld == lu else "")] = (fam, f"{ld}/{lu}", v)
    rows = []
    for name, (fam, prm, v) in cand.items():
        st.yang_zhang_vol = lambda *a, _v=np.asarray(v, float), **k: _v
        al = compute_allocation(p.values, nfci, cpi, **kw)
        st.yang_zhang_vol = YZ
        eq, fees, rev = simulate_net(p, al, leverage=2, funding=fund, e_max=E_MAX)
        row = {"candidat": name, "famille": fam, "param": prm, "frais %/an": round(fees, 2), "revis/an": round(rev, 1)}
        for hn, (a, b) in HALVES.items():
            cg, dd, kk = win(eq, idx, a, b)
            row.update({f"{hn} %/an": round(cg, 1), f"{hn} maxDD": round(dd, 1), f"{hn} Calmar": round(kk, 2)})
        for k_, (a, b) in EPIS.items():
            m = (idx >= a) & (idx <= b)
            row[k_] = round(100 * (eq[m][-1] / eq[m][0] - 1))
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv("myfiles/failed_xp/pust17_asym_vol.csv", index=False)
    pd.set_option("display.width", 280)
    cols = ["candidat", "H1 Calmar", "H2 Calmar", "tout %/an", "tout maxDD", "tout Calmar", "frais %/an", "revis/an"] + list(EPIS)
    print(out[cols].to_string(index=False))
    ref = out.iloc[0]
    both = out[(out["H1 Calmar"] > ref["H1 Calmar"]) & (out["H2 Calmar"] > ref["H2 Calmar"])]
    print(f"\n=== Battent la YZ 10 j sur LES DEUX moitiés : {len(both)} / {len(out) - 1} ===")
    print(both[cols].to_string(index=False) if len(both) else "  aucun")
    print("\n=== Validation croisée ===")
    for fam in ("pondérée", "mémoire"):
        g = out[out.famille == fam]
        for fit, test in (("H1", "H2"), ("H2", "H1")):
            b = g.loc[g[f"{fit} Calmar"].idxmax()]
            print(f"  {fam:9s} réglé {fit} → {b.candidat:26s} {test} Calmar {b[f'{test} Calmar']:.2f} (YZ {ref[f'{test} Calmar']:.2f})"
                  f"  maxDD {b[f'{test} maxDD']:.1f} (YZ {ref[f'{test} maxDD']:.1f})  %/an {b[f'{test} %/an']:.1f} (YZ {ref[f'{test} %/an']:.1f})")


if __name__ == "__main__":
    main()
