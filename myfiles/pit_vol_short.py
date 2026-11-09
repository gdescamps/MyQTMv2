"""Vol à COURT terme prédite par les embeddings PIT (walk-forward par millésime, comme pit_vintage_regress).

Cibles (article du soir J ; positions exécutées J+1) :
  vol1   : |rendement J+1| annualisé              (le lendemain — très bruité)
  vol5   : vol réalisée des rendements J+1 … J+5
  yz5    : vol Yang-Zhang 10 j de la stratégie, lue à J+5 (ce que la stratégie « verra »)
X : embedding du jour, MM5, MM20. Références prix : vol 5 j passée, YZ 10 j actuelle.
Puis injection dans la stratégie (lin / max avec YZ), frais 0.1 %/côté, jours sans news = YZ.
    python -m myfiles.pit_vol_short
"""
import glob
import os

import numpy as np
import pandas as pd

import src.risk_off_strategy.strategy as st
from myfiles.pit_vintage_regress import PURGE, fit_predict
from src.risk_off_strategy.data import load_funding, load_macro, load_ohlc
from src.risk_off_strategy.strategy import E_MAX, compute_allocation, simulate_net

st.BUY_FEE = st.SELL_FEE = 0.001
DIR = "data/text/cnbc/pit_vintage"
YZ = st.yang_zhang_vol


def win(eq, m):
    e = eq[m] / eq[m][0]
    cagr = e[-1] ** (252 / m.sum()) - 1
    dd = (e / np.maximum.accumulate(e) - 1).min()
    return 100 * cagr, 100 * dd, cagr / abs(dd)


def main():
    o = load_ohlc("QQQ", "2000-01-01", None)
    p, idx = o["close"], o.index
    r = np.log(p).diff()
    yz = pd.Series(YZ(o["open"].values, o["high"].values, o["low"].values, p.values), idx)
    a252 = np.sqrt(252) * 100
    fwd = pd.DataFrame({"vol1": r.shift(-1).abs() * a252,
                        "vol5": r.rolling(5).std().shift(-5) * a252,
                        "yz5": yz.shift(-5) * 100})
    ref = pd.DataFrame({"vol5_passée": r.rolling(5).std() * a252, "yz10_actuelle": yz * 100})
    oos = []
    for f in sorted(glob.glob(f"{DIR}/*.npz")):
        z = np.load(f)
        d = pd.DatetimeIndex(z["dates"])
        E = pd.DataFrame(z["emb"].astype(np.float32), index=d)
        ty = list(z["test_years"])
        for xn, X in {"jour": E, "MM5": E.rolling(5, min_periods=3).mean(),
                      "MM20": E.rolling(20, min_periods=10).mean()}.items():
            X = X.dropna()
            tg = fwd.reindex(X.index, method="ffill")
            te = X.index.year.isin(ty)
            tr_end = X.index[X.index < pd.Timestamp(f"{min(ty)}-01-01")]
            tr = X.index <= tr_end[-PURGE - 1]
            for tgt in fwd.columns:
                y = tg[tgt].values
                ok = tr & ~np.isnan(y)
                _, _, pte, _, _ = fit_predict(X.values[ok], y[ok], X.values[te])
                oos.append(pd.DataFrame({"date": X.index[te], "X": xn, "cible": tgt, "pred": pte}))
        print(f"{os.path.basename(f)[:6]} fait (test {ty})", flush=True)
    P = pd.concat(oos)
    P.to_parquet(f"{DIR}/oos_vol_short.parquet")
    print("\n=== Corrélations hors échantillon (Spearman) ===")
    for (xn, tgt), g in P.groupby(["cible", "X"]) if False else P.groupby(["X", "cible"]):
        g = g.set_index("date").sort_index()
        y = fwd[tgt].reindex(g.index, method="ffill")
        rf = ref.reindex(g.index, method="ffill")
        R = pd.concat([g.pred, y, rf], axis=1).dropna().rank()
        A = np.c_[np.ones(len(R)), R.iloc[:, 2:].values]
        res = lambda c: c - A @ np.linalg.lstsq(A, c, rcond=None)[0]  # noqa: E731
        part = np.corrcoef(res(R.iloc[:, 0].values), res(R.iloc[:, 1].values))[0, 1]
        print(f"  X={xn:5s} cible={tgt:5s} n={len(g)}  ρ(news)={g.pred.corr(y, method='spearman'):+.2f}  "
              f"réf vol5 passée {rf.iloc[:, 0].corr(y, method='spearman'):+.2f}, "
              f"YZ10 actuelle {rf.iloc[:, 1].corr(y, method='spearman'):+.2f}  | apport net {part:+.2f}")
    # injection dans la stratégie
    nfci, cpi = load_macro(idx)
    fund = load_funding(idx)
    kw = dict(high=o["high"].values, low=o["low"].values, open_=o["open"].values)
    rows = []

    def run(name, vol, years):
        st.yang_zhang_vol = lambda *a, _v=vol, **k: _v
        al = compute_allocation(p.values, nfci, cpi, **kw)
        st.yang_zhang_vol = YZ
        eq, fees, rev = simulate_net(p, al, leverage=2, funding=fund, e_max=E_MAX)
        c, dd, k = win(eq, idx >= pd.Timestamp("2014-01-01"))
        row = {"variante": name, "%/an": round(c, 1), "maxDD": round(dd, 1), "Calmar": round(k, 2),
               "frais %/an": round(fees, 2)}
        for y in years:
            my = idx.year == y
            row[str(y)] = round(100 * (eq[my][-1] / eq[my][0] - 1), 1)
        rows.append(row)

    yv = yz.values
    years = sorted(set(P.date.dt.year))
    run("base YZ", yv, years)
    for (xn, tgt), g in P[P.cible != "vol1"].groupby(["X", "cible"]):
        news = (g.set_index("date").pred / 100).reindex(idx).ffill(limit=3).clip(lower=0.03).values
        has = ~np.isnan(news)
        nv = np.nan_to_num(news)
        for w in (0.5, 0.0):
            run(f"{xn} {tgt} lin w={w}", np.where(has, w * yv + (1 - w) * nv, yv), years)
        run(f"{xn} {tgt} max", np.where(has, np.maximum(yv, nv), yv), years)
    df = pd.DataFrame(rows)
    df.to_csv("myfiles/pit_vol_short.csv", index=False)
    pd.set_option("display.width", 250)
    print(f"\n=== Stratégie x1.7 (2014 → fin, news sur {years}, sinon YZ) ===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
