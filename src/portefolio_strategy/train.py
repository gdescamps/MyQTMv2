"""Entraînement XGBoost en walk-forward — portage de ref/MyQTM/src/train.py sans fuite temporelle.

ref/MyQTM entraînait deux modèles A/B sur des blocs de 90 j entrelacés couvrant toute la période, chacun
prédisant les blocs de l'autre : un bloc prédit est encadré par des blocs d'apprentissage postérieurs, et
le label (tendance ex post) déborde d'un bloc sur l'autre. Ici :
- à chaque date de coupe T (tous les RETRAIN_MONTHS mois), on n'utilise que les lignes < T dont le label
  est définitif avec les cours < T (trends.trend_labels → final_until) ;
- la mécanique de ref/MyQTM est conservée À L'INTÉRIEUR de cette fenêtre : blocs A/B entrelacés de 90 j,
  6 modèles pour la stabilité de l'importance (moyenne / écart-type^puissance), sélection des meilleures
  features, modèle A arrêté sur le F1 macro des blocs B et inversement ;
- les dates T ≤ date < T suivant sont prédites par la moyenne des modèles A et B → prédictions 100 % hors
  échantillon, écrites dans data/portefolio_strategy/predictions.parquet.

    python -m src.portefolio_strategy.train
"""
import json
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score

from . import config as C
from . import features as F
from .trends import trend_labels

PREDICTIONS = C.DATA / "predictions.parquet"


def cutoffs(first, last):
    out, t = [], pd.Timestamp(first)
    while t <= last:
        out.append(t)
        t += pd.DateOffset(months=C.RETRAIN_MONTHS)
    return out


def labels_before(df, cutoff):
    """Labels définitifs connus avec les cours < cutoff. → Series alignée sur df (NaN = pas de label)."""
    y = pd.Series(np.nan, index=df.index)
    for _, g in df[df["date"] < cutoff].groupby("symbol", sort=False):
        lab, final_until = trend_labels(g.set_index("date")["c"], **C.TREND)
        ok = g["date"].to_numpy() < np.datetime64(final_until)
        y.loc[g.index[ok]] = lab.to_numpy()[ok]
    return y


def interval_index(dates, start):
    return ((dates - pd.Timestamp(start)).dt.days // C.INTERVAL_DAYS).to_numpy()


def _f1(predt, dmat):
    return "f1", float(f1_score(dmat.get_label(), np.argmax(predt, axis=1), average="macro"))


def _params():
    p = dict(C.XGB_PARAMS, disable_default_eval_metric=1)
    try:
        import torch
        if torch.cuda.is_available():
            p["device"] = "cuda"
    except ImportError:
        pass
    return p


def fit(X, y, Xv, yv, rounds=C.XGB_ROUNDS):
    dtr, dva = xgb.DMatrix(X, label=y), xgb.DMatrix(Xv, label=yv)
    m = xgb.train(_params(), dtr, num_boost_round=rounds, evals=[(dva, "eval")], custom_metric=_f1,
                  maximize=True, early_stopping_rounds=C.XGB_PATIENCE, verbose_eval=False)
    return m, m.best_iteration + 1, m.best_score


def select_features(X, y, idx, cols):
    """6 modèles (3 sous-blocs A, 3 sous-blocs B) → importance stable = moyenne / écart-type^puissance."""
    a, b = idx % 2 == 0, idx % 2 == 1
    imp = {}
    for k in range(6):
        part = idx % 6 == k
        if part.sum() < 200 or len(np.unique(y[part])) < 3:
            continue
        other = b if k % 2 == 0 else a
        m, _, _ = fit(X[part], y[part], X[other], y[other], rounds=200)
        s = m.get_score(importance_type="weight")
        imp[k] = [s.get(f"f{j}", 0) for j in range(len(cols))]
    d = pd.DataFrame(imp, index=cols)
    d = d / d.max().replace(0, 1)
    d = d[(d != 0).all(axis=1)]
    d["mean"], d["std"] = d.mean(axis=1), d.std(axis=1)
    d["score"] = d["mean"] / (d["std"] ** C.MEAN_STD_POWER + 1e-9)
    return d.sort_values("score", ascending=False)


def train_cutoff(df, cols, y, cutoff, nxt):
    tr = (df["member"] & (df["date"] >= C.START_DATE) & (df["date"] < cutoff) & y.notna()).to_numpy()
    te = (df["member"] & (df["date"] >= cutoff) & (df["date"] < nxt)).to_numpy()
    if not te.any():
        return None
    X, yy = df.loc[tr, cols].to_numpy(np.float32), y[tr].to_numpy(int)
    idx = interval_index(df.loc[tr, "date"], C.START_DATE)
    a, b = idx % 2 == 0, idx % 2 == 1
    rank = select_features(X, yy, idx, cols)
    best = None
    for top in C.TOP_FEATURES:
        sel = [cols.index(c) for c in rank.index[:top]]
        ma, na, f1a = fit(X[a][:, sel], yy[a], X[b][:, sel], yy[b])
        mb, nb, f1b = fit(X[b][:, sel], yy[b], X[a][:, sel], yy[a])
        f1 = (f1a + f1b) / 2
        if best is None or f1 > best["f1"]:
            best = dict(f1=f1, f1a=f1a, f1b=f1b, sel=sel, ma=ma, mb=mb, na=na, nb=nb, top=len(sel))
    Xt = xgb.DMatrix(df.loc[te, cols].to_numpy(np.float32)[:, best["sel"]])
    p = (best["ma"].predict(Xt, iteration_range=(0, best["na"])) + best["mb"].predict(Xt, iteration_range=(0, best["nb"]))) / 2
    out = df.loc[te, ["date", "symbol", "o", "c"]].copy()
    out["ybear"], out["yrange"], out["ybull"] = p[:, 0], p[:, 1], p[:, 2]
    info = dict(cutoff=f"{cutoff:%Y-%m-%d}", train_rows=int(tr.sum()), test_rows=int(te.sum()),
                train_until=f"{df.loc[tr, 'date'].max():%Y-%m-%d}", f1_cv=round(best["f1"], 4),
                top_features=best["top"], rounds=[best["na"], best["nb"]],
                class_share=[round(float((yy == k).mean()), 3) for k in range(3)])
    return out, info, rank


def main(df=None):
    df = pd.read_parquet(F.FEATURES) if df is None else df
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    cols = F.feature_columns(df)
    last = df["date"].max()
    cuts = cutoffs(C.FIRST_OOS_DATE, last)
    preds, infos, ranks, t0 = [], [], [], time.time()
    for i, cut in enumerate(cuts):
        nxt = cuts[i + 1] if i + 1 < len(cuts) else last + pd.Timedelta(days=1)
        r = train_cutoff(df, cols, labels_before(df, cut), cut, nxt)
        if r is None:
            continue
        preds.append(r[0])
        infos.append(r[1])
        ranks.append(r[2]["score"].rename(r[1]["cutoff"]))
        print(f"  {r[1]}  [{time.time() - t0:.0f} s]", flush=True)
    p = pd.concat(preds, ignore_index=True)
    # Diagnostic seulement (jamais utilisé pour décider) : F1 hors échantillon contre les labels définitifs.
    y_full = labels_before(df, last + pd.Timedelta(days=1))
    p = p.merge(pd.DataFrame({"date": df["date"], "symbol": df["symbol"], "y": y_full}), on=["date", "symbol"])
    p["cls"] = p[["ybear", "yrange", "ybull"]].to_numpy().argmax(axis=1)
    ok = p["y"].notna()
    f1 = f1_score(p.loc[ok, "y"].astype(int), p.loc[ok, "cls"], average="macro")
    acc = float((p.loc[ok, "y"].astype(int) == p.loc[ok, "cls"]).mean())
    p.to_parquet(PREDICTIONS, index=False)
    C.OUT.mkdir(parents=True, exist_ok=True)
    imp = pd.concat(ranks, axis=1)
    imp["mean_rank_score"] = imp.mean(axis=1)
    imp.sort_values("mean_rank_score", ascending=False).to_csv(C.OUT / "feature_importance.csv")
    sent_cov = float(df.loc[df["member"] & (df["date"] >= C.START_DATE), "ts_news_sentiment_company_0"].notna().mean())
    rep = dict(oos_macro_f1=round(float(f1), 4), oos_accuracy=round(acc, 4), n_predictions=len(p),
               first=f"{p['date'].min():%Y-%m-%d}", last=f"{p['date'].max():%Y-%m-%d}", n_features=len(cols),
               sentiment_coverage=round(sent_cov, 4), cutoffs=infos)
    (C.OUT / "train_report.json").write_text(json.dumps(rep, indent=2))
    print(f"prédictions hors échantillon : {len(p)} lignes {rep['first']} → {rep['last']} ; "
          f"F1 macro {f1:.3f}, exactitude {acc:.3f} ; couverture sentiment {sent_cov:.1%}")
    return p


if __name__ == "__main__":
    main()
