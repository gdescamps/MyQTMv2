"""Modèle de classement en walk-forward : quels titres de l'univers feront mieux que les autres ?

Pour chaque horizon h de config.HORIZONS, un XGBoost prédit le rang du jour du rendement futur (labels.py).
À chaque date de coupe T (tous les RETRAIN_MONTHS mois) :
- apprentissage sur les lignes de membres dont le label est connu avec les cours < T (fenêtre de h jours purgée) ;
- prédiction des dates T ≤ date < T suivant → scores 100 % hors échantillon.
Hyper-paramètres fixes et nombre d'arbres fixe : rien n'est réglé sur la période prédite.

Sorties : data/portefolio_strategy/predictions_<variante>.parquet (date, symbol, o, c, s_<h>, fwd_<h>) et
outputs/portefolio_strategy/<variante>/train_report.json, feature_importance.csv.

    python -m src.portefolio_strategy.train [variante]
"""
import json
import time

import numpy as np
import pandas as pd
import xgboost as xgb

from . import config as C
from . import features as F
from .labels import add_labels, label_known


def predictions_path(variant):
    return C.DATA / f"predictions_{variant}.parquet"


def out_dir(variant):
    return C.OUT / variant


def cutoffs(first, last):
    out, t = [], pd.Timestamp(first)
    while t <= last:
        out.append(t)
        t += pd.DateOffset(months=C.RETRAIN_MONTHS)
    return out


def split(df, h, cutoff, nxt):
    """→ (masque d'apprentissage, masque de prédiction) pour la date de coupe."""
    cutoff_pos = int((np.sort(df["date"].unique()) < np.datetime64(cutoff)).sum())
    tr = (df["member"] & df[f"y_{h}"].notna() & label_known(df["pos"], h, cutoff_pos)).to_numpy()
    te = (df["member"] & (df["date"] >= cutoff) & (df["date"] < nxt)).to_numpy()
    return tr, te


def fit_predict(Xtr, ytr, Xte, cols):
    m = xgb.train(C.XGB_PARAMS, xgb.DMatrix(Xtr, label=ytr, feature_names=cols), num_boost_round=C.XGB_ROUNDS)
    return m.predict(xgb.DMatrix(Xte, feature_names=cols)), m.get_score(importance_type="total_gain")


def daily_ic(p, h):
    """Corrélation de rang, jour par jour, entre le score et le rendement futur réalisé."""
    d = p.dropna(subset=[f"s_{h}", f"fwd_{h}"])
    return d.groupby("date")[[f"s_{h}", f"fwd_{h}"]].apply(lambda g: g[f"s_{h}"].corr(g[f"fwd_{h}"], method="spearman"))


def diagnostics(p, h):
    ic = daily_ic(p, h).dropna()
    d = p.dropna(subset=[f"s_{h}", f"fwd_{h}"]).copy()
    d["q"] = d.groupby("date")[f"s_{h}"].rank(pct=True)
    ex = d[f"fwd_{h}"] - d.groupby("date")[f"fwd_{h}"].transform("mean")
    return {"ic_mean": round(float(ic.mean()), 4),
            "ic_t": round(float(ic.mean() / ic.std() * np.sqrt(len(ic) / h)), 2) if len(ic) > h else None,
            "ic_positive_days_pct": round(float((ic > 0).mean() * 100), 1),
            "top_quintile_excess_pct": round(float(100 * ex[d["q"] > 0.8].mean()), 3),
            "bottom_quintile_excess_pct": round(float(100 * ex[d["q"] <= 0.2].mean()), 3), "days": int(len(ic))}


def main(df=None, variant="pit"):
    df = pd.read_parquet(F.FEATURES) if df is None else df
    cols = F.feature_columns(df, variant)             # avant d'ajouter les labels : ils ne sont jamais des features
    df = add_labels(df.sort_values(["symbol", "date"]).reset_index(drop=True), C.HORIZONS)
    assert not [c for c in cols if c.startswith(("fwd_", "y_")) or c == "pos"]
    out = out_dir(variant)
    print(f"variante « {variant} » ({F.VARIANTS[variant][1]}) : {len(cols)} features", flush=True)
    last = df["date"].max()
    cuts = cutoffs(C.FIRST_OOS_DATE, last)
    X = df[cols].to_numpy(np.float32)
    preds, infos, imps, t0 = {}, [], {h: [] for h in C.HORIZONS}, time.time()
    for i, cut in enumerate(cuts):
        nxt = cuts[i + 1] if i + 1 < len(cuts) else last + pd.Timedelta(days=1)
        info = {"cutoff": f"{cut:%Y-%m-%d}"}
        for h in C.HORIZONS:
            tr, te = split(df, h, cut, nxt)
            if not te.any():
                continue
            s, imp = fit_predict(X[tr], df.loc[tr, f"y_{h}"].to_numpy(np.float32), X[te], cols)
            preds.setdefault(h, []).append(pd.Series(s, index=df.index[te]))
            imps[h].append(pd.Series(imp, name=info["cutoff"]))
            info[f"h{h}"] = {"train_rows": int(tr.sum()), "train_until": f"{df.loc[tr, 'date'].max():%Y-%m-%d}",
                             "test_rows": int(te.sum())}
        if len(info) > 1:
            infos.append(info)
            print(f"  {info['cutoff']} : {info[f'h{C.HORIZONS[0]}']['train_rows']} lignes d'apprentissage "
                  f"[{time.time() - t0:.0f} s]", flush=True)
    idx = pd.concat(preds[C.HORIZONS[0]]).index
    p = df.loc[idx, ["date", "symbol", "o", "c"] + [f"fwd_{h}" for h in C.HORIZONS]].copy()
    for h in C.HORIZONS:
        p[f"s_{h}"] = pd.concat(preds[h]).reindex(idx).to_numpy()
    p = p.sort_values(["date", "symbol"]).reset_index(drop=True)
    p.to_parquet(predictions_path(variant), index=False)
    out.mkdir(parents=True, exist_ok=True)
    imp = pd.DataFrame({f"h{h}": pd.concat(imps[h], axis=1).fillna(0).pipe(lambda d: d / d.sum()).mean(axis=1)
                        for h in C.HORIZONS}).fillna(0)
    imp["family"] = [F.family_of(c) or "base" for c in imp.index]
    imp.sort_values(f"h{C.HORIZONS[1]}", ascending=False).round(5).to_csv(out / "feature_importance.csv")
    diag = {f"h{h}": diagnostics(p, h) for h in C.HORIZONS}
    rep = dict(variant=variant, description=F.VARIANTS[variant][1], n_features=len(cols), n_predictions=len(p),
               first=f"{p['date'].min():%Y-%m-%d}", last=f"{p['date'].max():%Y-%m-%d}", diagnostics=diag,
               importance_by_family={f"h{h}": imp.groupby("family")[f"h{h}"].sum().round(4).to_dict() for h in C.HORIZONS},
               cutoffs=infos)
    (out / "train_report.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    print(f"scores hors échantillon : {len(p)} lignes {rep['first']} → {rep['last']}")
    for h in C.HORIZONS:
        d = diag[f"h{h}"]
        print(f"  horizon {h:2d} j : IC moyen {d['ic_mean']:+.4f} (t {d['ic_t']}), meilleur quintile "
              f"{d['top_quintile_excess_pct']:+.2f} % / pire {d['bottom_quintile_excess_pct']:+.2f} % vs moyenne du jour")
    return p


if __name__ == "__main__":
    import sys
    main(variant=sys.argv[1] if len(sys.argv) > 1 else "pit")
