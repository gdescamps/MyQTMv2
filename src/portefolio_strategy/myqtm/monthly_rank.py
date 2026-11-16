"""Indice concentré à rotation mensuelle : classer les titres du S&P 500 et garder les 20 ou 30 premiers.

Décision à la clôture de la dernière séance du mois m ; achat à l'ouverture de la séance suivante ; portefeuille
gardé jusqu'à l'ouverture qui suit la dernière séance du mois m+1. Deux cibles, un modèle chacune :
- « next »  : rendement du mois suivant (ouverture → ouverture), rang parmi les membres ;
- « mom6n » : rendement sur 6 mois tel qu'on le mesurera à la fin du mois suivant (clôture m−5 → clôture m+1),
  rang parmi les membres : les « futurs leaders 6 mois ». Cinq mois sur six sont déjà connus à la décision.
Features : celles du S&P 500 sans LLM ni news (prix, macro, bilan, surprises, calendrier, fondamentaux,
valorisation, groupe GICS — fundamentals.py), reconstruites depuis fin 2015 pour avoir de l'historique.
XGBoost en régression sur le rang (réglages portefolio_strategy.config, 300 arbres, 3 graines moyennées) ;
walk-forward : modèle réappris tous les 3 mois sur tous les mois dont la cible est connue (≤ m−2), première
décision hors échantillon fin décembre 2018.

Sorties (data/portefolio_strategy/myqtm_sp) : monthly_features.parquet, monthly_preds.parquet, monthly_opens.parquet
(ouvertures quotidiennes), monthly_bench.parquet (univers équipondéré et indice pondéré capitalisation).

    python -m src.portefolio_strategy.myqtm.monthly_rank [--build] [--train]     # défaut : les deux
"""
import sys

import numpy as np
import pandas as pd
import xgboost as xgb

from . import config as C
from . import features as F
from . import sp as S
from .. import config as PC

DATA = S.DATA
FEATS, PREDS = DATA / "monthly_features.parquet", DATA / "monthly_preds.parquet"
OPENS, BENCH = DATA / "monthly_opens.parquet", DATA / "monthly_bench.parquet"
FIRST_DECISION = "2018-12-01"
RETRAIN = 3
SEEDS = (42, 7, 123)
TARGETS = ("next", "mom6n")
LABELS = ["entry", "exit", "y_next", "y_mom6n", "mom6", "cap"]


def build():
    S.use_sp()
    F.FEATURES, F.INDEX = DATA / "features_long.parquet", DATA / "index_long.parquet"
    F.ROWS_START, C.UNIVERSE_FIRST_YEAR = "2015-06-01", 2015
    df = F.main()
    df = df.drop(columns=[c for c in df.columns if F.is_pit_col(c) or F.is_vendor_col(c)])
    df = S.augment(df).sort_values(["symbol", "date"]).reset_index(drop=True)
    F.FEATURES.unlink(missing_ok=True)                      # fichier quotidien long : intermédiaire seulement
    o = df.pivot(index="date", columns="symbol", values="o").sort_index()
    c = df.pivot(index="date", columns="symbol", values="c").sort_index().ffill()
    mem = df.pivot(index="date", columns="symbol", values="member").reindex(o.index).fillna(False).astype(bool)
    of = o.ffill()
    days = o.index
    ro = (o.shift(-1) / o - 1).clip(-0.5, 1.0).where(mem).mean(axis=1)                # univers équipondéré t → t+1
    idx = pd.read_parquet(F.INDEX)["level"].reindex(days).ffill()
    pd.DataFrame({"ro": ro, "index_level": idx}).to_parquet(BENCH)
    of.astype("float32").to_parquet(OPENS)
    last = pd.Series(days, index=days).groupby(days.to_period("M")).last()
    me = last.iloc[:-1].to_numpy() if days[-1] != (days[-1] + pd.offsets.BMonthEnd(0)) else last.to_numpy()
    pos = days.get_indexer(me)
    rows = []
    for k in range(6, len(me)):
        e = pos[k] + 1
        x = pos[k + 1] + 1 if k + 1 < len(me) and pos[k + 1] + 1 < len(days) else None
        d = pd.DataFrame({"date": me[k], "symbol": o.columns, "entry": days[e] if e < len(days) else pd.NaT,
                          "exit": days[x] if x is not None else pd.NaT,
                          "y_next": (of.iloc[x] / of.iloc[e] - 1).to_numpy() if x is not None else np.nan,
                          "y_mom6n": (c.iloc[pos[k + 1]] / c.iloc[pos[k - 5]] - 1).to_numpy() if x is not None else np.nan,
                          "mom6": (c.iloc[pos[k]] / c.iloc[pos[k - 6]] - 1).to_numpy()})
        rows.append(d)
    lab = pd.concat(rows, ignore_index=True)
    m = df[df["date"].isin(me)].merge(lab, on=["date", "symbol"], how="inner")
    m["cap"] = np.exp(m["fv_logcap"])
    m.to_parquet(FEATS, index=False)
    mm = m[m["member"]]
    print(f"mensuel : {len(m)} lignes ({len(mm)} membres), {m['date'].nunique()} fins de mois "
          f"{m['date'].min():%Y-%m-%d} → {m['date'].max():%Y-%m-%d}, {len(feature_columns(m))} features", flush=True)


def feature_columns(df):
    return [c for c in F.feature_columns(df) if c not in LABELS]


def train():
    m = pd.read_parquet(FEATS).sort_values(["date", "symbol"]).reset_index(drop=True)
    cols = feature_columns(m)
    months = np.sort(m["date"].unique())
    k_of = pd.Series(np.arange(len(months)), index=months)
    mk = m["date"].map(k_of).to_numpy()
    mem = m["member"].to_numpy()
    X = m[cols].to_numpy(np.float32)
    out = m.loc[mem, ["date", "symbol"] + LABELS].copy()
    imp = {}
    for tgt in TARGETS:
        y = m[f"y_{tgt}"]
        rank = (y.where(m["member"]).groupby(m["date"]).rank(pct=True) - 0.5).to_numpy()
        pred = np.full(len(m), np.nan)
        models, fitted_at = None, None
        for k in range(len(months)):
            if months[k] < np.datetime64(FIRST_DECISION):
                continue
            if models is None or k - fitted_at >= RETRAIN:
                tr = mem & (mk <= k - 2) & ~np.isnan(rank)
                dtr = xgb.DMatrix(X[tr], label=rank[tr])
                models = [xgb.train(dict(PC.XGB_PARAMS, seed=s), dtr, num_boost_round=PC.XGB_ROUNDS) for s in SEEDS]
                fitted_at = k
                for s, b in zip(SEEDS, models):
                    g = b.get_score(importance_type="total_gain")
                    imp[(tgt, k, s)] = {cols[int(f[1:])]: v for f, v in g.items()}
            te = mem & (mk == k)
            pred[te] = np.mean([b.predict(xgb.DMatrix(X[te])) for b in models], axis=0)
        out[f"pred_{tgt}"] = pred[mem]
        print(f"cible {tgt} : {int(np.isfinite(pred).sum())} prédictions, {out.loc[out[f'pred_{tgt}'].notna(), 'date'].nunique()} mois",
              flush=True)
    out = out[out[[f"pred_{t}" for t in TARGETS]].notna().all(axis=1)]
    out.to_parquet(PREDS, index=False)
    im = pd.DataFrame(imp).fillna(0)
    im = (im / im.sum()).T.groupby(level=0).mean().T
    im.sort_values(TARGETS[0], ascending=False).to_csv(PC.OUT / "myqtm_sp" / "monthly_rank_importance.csv")
    for t in TARGETS:
        print(f"importance ({t}) :", im[t].sort_values(ascending=False).head(12).round(3).to_dict(), flush=True)


if __name__ == "__main__":
    if "--build" in sys.argv or "--train" not in sys.argv:
        build()
    if "--train" in sys.argv or "--build" not in sys.argv:
        train()
