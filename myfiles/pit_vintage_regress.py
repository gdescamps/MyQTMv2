"""Régression walk-forward des embeddings PIT (bon millésime par année) sur l'état FUTUR du marché.

Pour chaque millésime V (fichier data/text/cnbc/pit_vintage/V.npz, cf. pit_vintage_embed.py) :
- X = embedding du jour, ou sa moyenne mobile sur 20 articles (« contexte des 20 dernières séances ») ;
- y = état futur de QQQ, connu seulement après : l'article de J est utilisable au soir de J, la cible
  porte sur J+1 … J+h (pas de recouvrement avec l'info disponible) :
    dd20  : drawdown max sur les 20 séances suivantes (%)       ← cible principale (risque)
    vol20 : volatilité réalisée des 20 séances suivantes (%/an)
    ret60 : rendement des 60 séances suivantes (%)
- apprentissage sur les dates < 1er janvier de la 1re année de test − 60 séances (purge : la cible
  des dernières dates d'apprentissage ne déborde pas dans le test), standardisation + PCA(k) + ridge
  (k, alpha choisis par validation temporelle DANS l'apprentissage) ; prédiction sur les années de test.
Les prédictions hors échantillon de tous les millésimes sont concaténées (2014 → 2026).

Évaluation :
1. Spearman hors échantillon prédiction vs cible, contre des références prix seuls (vol 20 j passée,
   rendement 20 j passé) et p_up (note oui/non PIT) ;
2. apport marginal : corrélation partielle de la prédiction avec la cible une fois retirées les références prix ;
3. « ce que l'embedding a appris » : pour chaque millésime, titres des articles les plus extrêmes le long
   de la direction ridge (risque élevé / faible) ;
4. garde-fou PUST x1.7 : risque prédit (MM20) > quantile q de sa distribution d'apprentissage → expo ≤ 1.
    python -m myfiles.pit_vintage_regress
"""
import glob
import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DIR = "data/text/cnbc/pit_vintage"
CNBC = "data/text/cnbc/cnbc_close.parquet"
PURGE = 60
KS, ALPHAS = (8, 16, 32, 64), (10.0, 100.0, 1000.0, 10000.0)


def targets(close):
    lr = np.log(close)
    r = lr.diff()
    fwd = {}
    n = len(close)
    c = close.values
    dd = np.full(n, np.nan)
    for i in range(n - 21):
        w = c[i + 1:i + 22] / c[i + 1]          # entrée au close J+1 (exécution du signal du soir J)
        dd[i] = 100 * (w / np.maximum.accumulate(w) - 1).min()
    fwd["dd20"] = pd.Series(dd, close.index)
    # vol réalisée des rendements J+2 … J+21 (positions prises au close J+1)
    fwd["vol20"] = r.rolling(20).std().shift(-21) * np.sqrt(252) * 100
    fwd["ret60"] = 100 * (lr.shift(-61) - lr.shift(-1))
    ref = {"vol20_passée": r.rolling(20).std() * np.sqrt(252) * 100, "ret20_passé": 100 * (lr - lr.shift(20))}
    return pd.DataFrame(fwd), pd.DataFrame(ref)


def fit_predict(Xtr, ytr, Xte):
    """k et alpha par validation temporelle simple (dernier 25 % de l'apprentissage, purgé)."""
    n = len(Xtr)
    cut = int(n * 0.75)
    best, bs = None, -np.inf
    for k in KS:
        for a in ALPHAS:
            m = make_pipeline(StandardScaler(), PCA(min(k, cut - 1), random_state=0), Ridge(alpha=a)).fit(Xtr[:cut - PURGE], ytr[:cut - PURGE])
            s = pd.Series(m.predict(Xtr[cut:])).corr(pd.Series(ytr[cut:]), method="spearman")
            if s > bs:
                best, bs = (k, a), s
    m = make_pipeline(StandardScaler(), PCA(best[0], random_state=0), Ridge(alpha=best[1])).fit(Xtr, ytr)
    return m, m.predict(Xtr), m.predict(Xte), best, bs


def direction(m):
    sc, pca, rg = m.steps[0][1], m.steps[1][1], m.steps[2][1]
    return (pca.components_.T @ rg.coef_) / sc.scale_


def main():
    import myfiles.pust17_tweet_overlay as ov
    from src.risk_off_strategy.data import load_funding, load_macro, load_ohlc
    from src.risk_off_strategy.strategy import compute_allocation
    ov.BUY_FEE = ov.SELL_FEE = 0.001
    o = load_ohlc("QQQ", "2000-01-01", None)
    close = o["close"]
    fwd, ref = targets(close)
    titles = pd.read_parquet(CNBC)
    titles["date"] = pd.to_datetime(titles["date"])
    titles = titles.set_index("date")["title"]
    files = sorted(glob.glob(f"{DIR}/*.npz"))
    print(f"millésimes disponibles : {[os.path.basename(f)[:6] for f in files]}")
    oos = []
    for f in files:
        z = np.load(f)
        v = os.path.basename(f)[:6]
        d = pd.DatetimeIndex(z["dates"])
        E = pd.DataFrame(z["emb"].astype(np.float32), index=d)
        ty = list(z["test_years"])
        variants = {"jour": E, "MM20": E.rolling(20, min_periods=10).mean()}
        for name, X in variants.items():
            X = X.dropna()
            # cible alignée sur la séance de l'article (article du soir J → cible à partir de J+1)
            tg = fwd.reindex(X.index, method="ffill")
            te = X.index.year.isin(ty)
            tr_end = X.index[X.index < pd.Timestamp(f"{min(ty)}-01-01")]
            tr = X.index <= tr_end[-PURGE - 1] if len(tr_end) > PURGE else np.zeros(len(X), bool)
            for tgt in ("dd20", "vol20", "ret60"):
                y = tg[tgt].values
                ok_tr, ok_te = tr & ~np.isnan(y), te
                if ok_tr.sum() < 300 or ok_te.sum() == 0:
                    continue
                m, ptr, pte, best, cvs = fit_predict(X.values[ok_tr], y[ok_tr], X.values[ok_te])
                oos.append(pd.DataFrame({"date": X.index[ok_te], "vintage": v, "X": name, "cible": tgt, "pred": pte,
                                         # seuils de risque issus de l'APPRENTISSAGE seulement (pas de look-ahead)
                                         "q20_train": np.quantile(ptr, 0.2), "q10_train": np.quantile(ptr, 0.1)}))
                if name == "jour" and tgt == "dd20":
                    w = direction(m)
                    proj = pd.Series(X.values @ w, X.index)
                    lo, hi = proj.nsmallest(4).index, proj.nlargest(4).index
                    print(f"\n[{v}] k={best[0]} α={best[1]:g} (val {cvs:+.2f}) — direction « dd20 » apprise sur ≤{tr_end[-PURGE - 1].date()}")
                    print("  risque ÉLEVÉ (drawdown à venir) :")
                    for t in lo:
                        print(f"    {t.date()}  {titles.get(t, '')[:100]}")
                    print("  risque FAIBLE :")
                    for t in hi:
                        print(f"    {t.date()}  {titles.get(t, '')[:100]}")
    if not oos:
        print("pas encore de prédictions hors échantillon")
        return
    P = pd.concat(oos)
    P.to_parquet(f"{DIR}/oos_predictions.parquet")
    # 1-2. évaluation hors échantillon
    print("\n=== Hors échantillon (concaténé sur les millésimes disponibles) ===")
    pu = pd.concat([pd.Series(np.load(f)["p_up"], pd.DatetimeIndex(np.load(f)["dates"]))[
        pd.DatetimeIndex(np.load(f)["dates"]).year.isin(list(np.load(f)["test_years"]))] for f in files])
    for (xn, tgt), g in P.groupby(["X", "cible"]):
        g = g.set_index("date").sort_index()
        y = fwd[tgt].reindex(g.index, method="ffill")
        rf = ref.reindex(g.index, method="ffill")
        s = g.pred.corr(y, method="spearman")
        sv = rf["vol20_passée"].corr(y, method="spearman")
        sr = rf["ret20_passé"].corr(y, method="spearman")
        sp = pu.reindex(g.index).rolling(20, min_periods=10).mean().corr(y, method="spearman") if xn == "MM20" else \
            pu.reindex(g.index).corr(y, method="spearman")
        # corrélation partielle (rangs) en retirant vol et rendement passés
        R = pd.concat([g.pred, y, rf], axis=1).dropna().rank()
        A = np.c_[np.ones(len(R)), R.iloc[:, 2:].values]
        res = lambda col: col - A @ np.linalg.lstsq(A, col, rcond=None)[0]  # noqa: E731
        part = np.corrcoef(res(R.iloc[:, 0].values), res(R.iloc[:, 1].values))[0, 1]
        print(f"  X={xn:5s} cible={tgt:6s} n={len(g):5d}  ρ(pred)={s:+.2f}   réf : vol passée {sv:+.2f}, "
              f"ret passé {sr:+.2f}, p_up {sp:+.2f}   | partielle (hors prix) {part:+.2f}")
    # 4. garde-fou PUST x1.7 sur la prédiction dd20 MM20
    g = P[(P.X == "MM20") & (P.cible == "dd20")].set_index("date").sort_index()
    if len(g) > 250:
        idx = close.index
        nf, cp = load_macro(idx)
        fund = np.asarray(load_funding(idx), float)
        alloc = compute_allocation(close.values, nf, cp, high=o["high"].values, low=o["low"].values, open_=o["open"].values)
        ret = close.pct_change().fillna(0).values
        start = g.index[0]
        print(f"\n=== Garde-fou PUST x1.7 (frais 0.1 %/côté), {start.date()} → fin ; risque prédit < seuil du train → plafond ===")
        eq0, f0 = ov.simulate(ret, alloc, fund, None)
        print("  base : " + "  ".join(f"{k} {v:.2f}" for k, v in ov.metrics(eq0, f0, idx, start).items()))
        for qn in ("q20_train", "q10_train"):   # dd20 prédit plus négatif que le quantile bas du train = risque
            on = (g.pred < g[qn]).reindex(idx).shift(1).fillna(False).values.astype(bool)
            for capv in (1.0, 0.0):
                eq, fe = ov.simulate(ret, alloc, fund, np.where(on, capv, np.nan))
                m = ov.metrics(eq, fe, idx, start)
                print(f"  risque < {qn[:3]} train, cap {capv:g} : " + "  ".join(f"{k} {v:.2f}" for k, v in m.items())
                      + f"  jours actifs {100 * on[idx >= start].mean():.0f}%")


if __name__ == "__main__":
    main()
