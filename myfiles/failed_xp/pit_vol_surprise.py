"""Surprise de vol prédite par les news PIT → accélérateur de coupe / de retour (stratégie x1.7 PUST+LQQ).

Cible : surp = log(vol réalisée J+1…J+3 / YZ10 actuelle)  — « demain sera plus (ou moins) agité que
ce que la vol des prix mesure déjà ». X : embedding du jour, MM3. Walk-forward par millésime PIT.
Action (seuils = quantiles des prédictions sur l'APPRENTISSAGE, pas de look-ahead) :
  coupe  : pred > q90 train → YZ × f (f = 1.3, 1.5)        → la stratégie coupe plus tôt
  retour : pred < q10 train → YZ × g (g = 0.8, 0.7)        → elle revient plus tôt après un choc
  les deux
Jours sans news → YZ seule. Frais 0.1 %/côté.
    python -m myfiles.failed_xp.pit_vol_surprise
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
EPIS = {"2015-08": ("2015-08-10", "2015-10-31"), "2016-01": ("2015-12-15", "2016-03-31"),
        "2018T4": ("2018-10-01", "2019-02-28"), "2019-05": ("2019-05-01", "2019-08-31"), "2020": ("2020-02-15", "2020-06-30")}


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
    v3 = np.sqrt((r.shift(-1) ** 2 + r.shift(-2) ** 2 + r.shift(-3) ** 2) / 3 * 252)
    surp = np.log(v3.clip(lower=0.02) / yz)
    oos = []
    for f in sorted(glob.glob(f"{DIR}/*.npz")):
        z = np.load(f)
        E = pd.DataFrame(z["emb"].astype(np.float32), index=pd.DatetimeIndex(z["dates"]))
        ty = list(z["test_years"])
        for xn, X in {"jour": E, "MM3": E.rolling(3, min_periods=2).mean()}.items():
            X = X.dropna()
            y = surp.reindex(X.index, method="ffill").values
            te = X.index.year.isin(ty)
            tr_end = X.index[X.index < pd.Timestamp(f"{min(ty)}-01-01")]
            ok = (X.index <= tr_end[-PURGE - 1]) & ~np.isnan(y)
            _, ptr, pte, best, cvs = fit_predict(X.values[ok], y[ok], X.values[te])
            oos.append(pd.DataFrame({"date": X.index[te], "X": xn, "pred": pte,
                                     "q90": np.quantile(ptr, 0.9), "q10": np.quantile(ptr, 0.1)}))
        print(f"{os.path.basename(f)[:6]} fait (test {ty})", flush=True)
    P = pd.concat(oos)
    P.to_parquet(f"{DIR}/oos_vol_surprise.parquet")
    ref = pd.DataFrame({"YZ10": yz, "ret5": np.log(p).diff(5)})
    print("\n=== Surprise de vol : hors échantillon ===")
    for xn, g in P.groupby("X"):
        g = g.set_index("date").sort_index()
        y = surp.reindex(g.index, method="ffill")
        R = pd.concat([g.pred, y, ref.reindex(g.index, method="ffill")], axis=1).dropna().rank()
        A = np.c_[np.ones(len(R)), R.iloc[:, 2:].values]
        res = lambda c: c - A @ np.linalg.lstsq(A, c, rcond=None)[0]  # noqa: E731
        part = np.corrcoef(res(R.iloc[:, 0].values), res(R.iloc[:, 1].values))[0, 1]
        hi, lo = g.pred > g.q90, g.pred < g.q10
        print(f"  X={xn:4s} n={len(g)} ρ={g.pred.corr(y, method='spearman'):+.2f} apport net {part:+.2f} | "
              f"surprise réalisée moyenne : jours ALERTE ({100 * hi.mean():.0f}%) {y[hi].mean():+.2f}, "
              f"jours CALME ({100 * lo.mean():.0f}%) {y[lo].mean():+.2f}, autres {y[~hi & ~lo].mean():+.2f}")
    nfci, cpi = load_macro(idx)
    fund = load_funding(idx)
    kw = dict(high=o["high"].values, low=o["low"].values, open_=o["open"].values)
    years = sorted(set(P.date.dt.year))
    rows = []

    def run(name, vol, on=None):
        st.yang_zhang_vol = lambda *a, _v=vol, **k: _v
        al = compute_allocation(p.values, nfci, cpi, **kw)
        st.yang_zhang_vol = YZ
        eq, fees, rev = simulate_net(p, al, leverage=2, funding=fund, e_max=E_MAX)
        c, dd, k = win(eq, idx >= pd.Timestamp("2014-01-01"))
        row = {"variante": name, "%/an": round(c, 1), "maxDD": round(dd, 1), "Calmar": round(k, 2),
               "frais %/an": round(fees, 2), "% jours modifiés": round(100 * on[idx.year >= 2014].mean(), 1) if on is not None else 0}
        for y in years:
            my = idx.year == y
            row[str(y)] = round(100 * (eq[my][-1] / eq[my][0] - 1), 1)
        for k_, (a, b) in EPIS.items():
            m = (idx >= a) & (idx <= b)
            row[k_] = round(100 * (eq[m][-1] / eq[m][0] - 1), 1)
        rows.append(row)

    yv = yz.values
    run("base YZ", yv)
    for xn, g in P.groupby("X"):
        g = g.set_index("date").sort_index()
        hi = (g.pred > g.q90).reindex(idx).ffill(limit=3).fillna(False).values.astype(bool)
        lo = (g.pred < g.q10).reindex(idx).ffill(limit=3).fillna(False).values.astype(bool)
        for fct in (1.3, 1.5):
            run(f"{xn} coupe ×{fct}", np.where(hi, yv * fct, yv), hi)
        for gct in (0.8, 0.7):
            run(f"{xn} retour ×{gct}", np.where(lo, yv * gct, yv), lo)
        run(f"{xn} coupe ×1.3 + retour ×0.8", np.where(hi, yv * 1.3, np.where(lo, yv * 0.8, yv)), hi | lo)
    df = pd.DataFrame(rows)
    df.to_csv("myfiles/failed_xp/pit_vol_surprise.csv", index=False)
    pd.set_option("display.width", 260)
    print(f"\n=== Stratégie x1.7 (2014 → fin ; news sur {years}, sinon YZ) ===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
