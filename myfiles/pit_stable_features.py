"""Filtrage des features d'embedding PIT par STABILITÉ de leur contribution (idée : ne garder que ce qui
contribue pareil sur des intervalles différents → retire le bruit et les raccourcis « d'époque »).

Par millésime (walk-forward, tout reste DANS l'apprentissage, aucune fuite vers l'année testée) :
1. StandardScaler + PCA(K=64) ajustés sur tout l'apprentissage (non supervisé) → base commune ;
2. ridge (alpha fixé) refait sur chaque intervalle dans cette base → coefficients par intervalle ;
3. filtre :
     moities : même signe sur les 2 moitiés ET rapport des |coef| ≤ 3            (idée initiale)
     annees  : même signe sur ≥ 80 % des années d'apprentissage                  (plus robuste)
4. ridge final sur les composantes retenues (alpha par validation temporelle) → prédiction test.
Référence : même pipeline sans filtre (K=64, alpha par validation).
Cibles : vol20 (X = MM20), surp = log(vol J+1..J+3 / YZ10) (X = jour, MM3).
    python -m myfiles.pit_stable_features
"""
import glob
import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

import src.risk_off_strategy.strategy as st
from myfiles.pit_vintage_regress import PURGE
from src.risk_off_strategy.data import load_ohlc

DIR = "data/text/cnbc/pit_vintage"
K, ALPHAS, A_BLOCK = 64, (10.0, 100.0, 1000.0, 10000.0), 100.0


def ridge_cv(Z, y):
    cut = int(len(Z) * 0.75)
    best, bs = ALPHAS[0], -np.inf
    for a in ALPHAS:
        m = Ridge(alpha=a).fit(Z[:cut - PURGE], y[:cut - PURGE])
        s = pd.Series(m.predict(Z[cut:])).corr(pd.Series(y[cut:]), method="spearman")
        if s > bs:
            best, bs = a, s
    return Ridge(alpha=best).fit(Z, y)


def stable_mask(Z, y, years, mode):
    if mode == "moities":
        h = np.arange(len(Z)) < len(Z) // 2
        c1 = Ridge(alpha=A_BLOCK).fit(Z[h], y[h]).coef_
        c2 = Ridge(alpha=A_BLOCK).fit(Z[~h], y[~h]).coef_
        ratio = np.maximum(abs(c1), abs(c2)) / np.maximum(np.minimum(abs(c1), abs(c2)), 1e-12)
        return (np.sign(c1) == np.sign(c2)) & (ratio <= 3)
    C = np.array([Ridge(alpha=A_BLOCK).fit(Z[years == yr], y[years == yr]).coef_
                  for yr in np.unique(years) if (years == yr).sum() > 100])
    s = np.sign(C)
    return np.maximum((s > 0).mean(0), (s < 0).mean(0)) >= 0.8


def main():
    o = load_ohlc("QQQ", "2000-01-01", None)
    p, idx = o["close"], o.index
    r = np.log(p).diff()
    yz = pd.Series(st.yang_zhang_vol(o["open"].values, o["high"].values, o["low"].values, p.values), idx)
    v3 = np.sqrt((r.shift(-1) ** 2 + r.shift(-2) ** 2 + r.shift(-3) ** 2) / 3 * 252)
    T = pd.DataFrame({"vol20": r.rolling(20).std().shift(-21) * np.sqrt(252) * 100,
                      "surp": np.log(v3.clip(lower=0.02) / yz)})
    ref = pd.DataFrame({"vol20_passée": r.rolling(20).std(), "YZ10": yz, "ret20": np.log(p).diff(20)})
    CASES = [("vol20", "MM20"), ("surp", "jour"), ("surp", "MM3")]
    rows, kept = [], []
    for f in sorted(glob.glob(f"{DIR}/*.npz")):
        z = np.load(f)
        v = os.path.basename(f)[:6]
        E = pd.DataFrame(z["emb"].astype(np.float32), index=pd.DatetimeIndex(z["dates"]))
        ty = list(z["test_years"])
        Xs = {"jour": E, "MM3": E.rolling(3, min_periods=2).mean(), "MM20": E.rolling(20, min_periods=10).mean()}
        for tgt, xn in CASES:
            X = Xs[xn].dropna()
            y = T[tgt].reindex(X.index, method="ffill").values
            te = X.index.year.isin(ty)
            tr_end = X.index[X.index < pd.Timestamp(f"{min(ty)}-01-01")]
            ok = (X.index <= tr_end[-PURGE - 1]) & ~np.isnan(y)
            sc = StandardScaler().fit(X.values[ok])
            pca = PCA(K, random_state=0).fit(sc.transform(X.values[ok]))
            Ztr, Zte = pca.transform(sc.transform(X.values[ok])), pca.transform(sc.transform(X.values[te]))
            ytr, yrs = y[ok], X.index[ok].year.values
            for mode in ("aucun", "moities", "annees"):
                msk = np.ones(K, bool) if mode == "aucun" else stable_mask(Ztr, ytr, yrs, mode)
                if msk.sum() == 0:
                    msk[:1] = True
                m = ridge_cv(Ztr[:, msk], ytr)
                rows.append(pd.DataFrame({"date": X.index[te], "cible": tgt, "X": xn, "filtre": mode,
                                          "pred": m.predict(Zte[:, msk])}))
                kept.append({"millésime": v, "cible": tgt, "X": xn, "filtre": mode, "gardées": int(msk.sum())})
        print(f"{v} fait (test {ty})", flush=True)
    P = pd.concat(rows)
    P.to_parquet(f"{DIR}/oos_stable_features.parquet")
    kd = pd.DataFrame(kept)
    print("\n=== Composantes gardées (sur 64), moyenne par millésime ===")
    print(kd.groupby(["cible", "X", "filtre"]).gardées.mean().round(1).unstack().to_string())
    print("\n=== Hors échantillon : Spearman global, apport net (prix retirés), et par année ===")
    out = []
    for (tgt, xn, mode), g in P.groupby(["cible", "X", "filtre"]):
        g = g.set_index("date").sort_index()
        y = T[tgt].reindex(g.index, method="ffill")
        R = pd.concat([g.pred, y, ref.reindex(g.index, method="ffill")], axis=1).dropna().rank()
        A = np.c_[np.ones(len(R)), R.iloc[:, 2:].values]
        res = lambda c: c - A @ np.linalg.lstsq(A, c, rcond=None)[0]  # noqa: E731
        row = {"cible": tgt, "X": xn, "filtre": mode, "ρ": round(g.pred.corr(y, method="spearman"), 3),
               "apport net": round(np.corrcoef(res(R.iloc[:, 0].values), res(R.iloc[:, 1].values))[0, 1], 3)}
        for yr, gy in g.groupby(g.index.year):
            row[str(yr)] = round(gy.pred.corr(y.loc[gy.index], method="spearman"), 2)
        out.append(row)
    df = pd.DataFrame(out)
    df.to_csv("myfiles/pit_stable_features.csv", index=False)
    pd.set_option("display.width", 250)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
