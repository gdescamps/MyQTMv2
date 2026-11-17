"""Indice concentré à rotation mensuelle : classer les titres du S&P 500 et garder les 20 ou 30 premiers.

Décision à la clôture de la dernière séance du mois m ; achat à l'ouverture de la séance suivante ; portefeuille
gardé jusqu'à l'ouverture qui suit la dernière séance du mois m+1. Trois cibles (rang parmi les membres), un
modèle chacune :
- « mom6n »  : rendement sur 6 mois tel qu'on le mesurera à la fin du mois suivant (clôture m−5 → clôture m+1) :
  les « futurs leaders 6 mois ». Cinq mois sur six sont déjà connus à la décision ;
- « sharpe » : Sharpe réalisé du titre pendant le mois de détention (moyenne / écart-type des rendements
  quotidiens) : viser le rendement par unité de risque plutôt que le rendement ;
- « vol »    : volatilité réalisée pendant le mois de détention (pour choisir les titres les moins risqués).
(La cible « rendement du mois suivant », sans pouvoir prédictif, a été retirée ; y_next reste pour l'évaluation.)
Features : celles du S&P 500 sans LLM ni news (prix, macro, bilan, surprises, calendrier, fondamentaux,
valorisation, groupe GICS — fundamentals.py), reconstruites depuis fin 2015 pour avoir de l'historique.
XGBoost en régression sur le rang (réglages portefolio_strategy.config, 300 arbres, 3 graines moyennées) ;
walk-forward : modèle réappris tous les 3 mois sur tous les mois dont la cible est connue (≤ m−2), première
décision hors échantillon fin décembre 2018.

Variante hebdomadaire : les mêmes modèles (horizon d'un mois) notent les membres à chaque fin de semaine
(weekly_features / weekly_preds), pour comparer une mise à jour hebdomadaire du portefeuille à la mensuelle.

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
FEATS_W, PREDS_W = DATA / "weekly_features.parquet", DATA / "weekly_preds.parquet"
OPENS, BENCH = DATA / "monthly_opens.parquet", DATA / "monthly_bench.parquet"
FIRST_DECISION = "2018-12-01"
RETRAIN = 3
SEEDS = (42, 7, 123)
TARGETS = ("mom6n", "sharpe", "vol")
LABELS = ["entry", "exit", "y_next", "y_mom6n", "y_sharpe", "y_vol", "mom6", "cap", "vol63_d"]


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
        if x is not None:                                   # rendements quotidiens du mois de détention
            r = of.iloc[e:x + 1].pct_change(fill_method=None).iloc[1:]
            sd = r.std().replace(0, np.nan)
            y_sh, y_vol = (r.mean() / sd).to_numpy(), sd.to_numpy()
        else:
            y_sh = y_vol = np.nan
        d = pd.DataFrame({"date": me[k], "symbol": o.columns, "entry": days[e] if e < len(days) else pd.NaT,
                          "y_sharpe": y_sh, "y_vol": y_vol,
                          "exit": days[x] if x is not None else pd.NaT,
                          "y_next": (of.iloc[x] / of.iloc[e] - 1).to_numpy() if x is not None else np.nan,
                          "y_mom6n": (c.iloc[pos[k + 1]] / c.iloc[pos[k - 5]] - 1).to_numpy() if x is not None else np.nan,
                          "mom6": (c.iloc[pos[k]] / c.iloc[pos[k - 6]] - 1).to_numpy()})
        rows.append(d)
    lab = pd.concat(rows, ignore_index=True)
    m = df[df["date"].isin(me)].merge(lab, on=["date", "symbol"], how="inner")
    m["cap"], m["vol63_d"] = np.exp(m["fv_logcap"]), m["vol63"]      # copies hors features, pour les pondérations
    m.to_parquet(FEATS, index=False)
    # lignes de fin de semaine (membres) : mêmes modèles mensuels, portefeuille mis à jour chaque semaine
    wk = pd.Series(days, index=days).groupby(days.to_period("W")).last().to_numpy()
    wk = wk[wk >= np.datetime64(FIRST_DECISION)][:-1]       # la dernière semaine peut être incomplète
    wpos = days.get_indexer(wk)
    wrows = []
    for j in range(len(wk)):
        e = wpos[j] + 1
        x = wpos[j + 1] + 1 if j + 1 < len(wk) and wpos[j + 1] + 1 < len(days) else None
        wrows.append(pd.DataFrame({"date": wk[j], "symbol": o.columns, "entry": days[e], "exit": days[x] if x is not None else pd.NaT,
                                   "y_next": (of.iloc[x] / of.iloc[e] - 1).to_numpy() if x is not None else np.nan,
                                   "mom6": (c.iloc[wpos[j]] / c.iloc[wpos[j] - 126] - 1).to_numpy()}))
    w = df[df["date"].isin(wk) & df["member"]].merge(pd.concat(wrows, ignore_index=True), on=["date", "symbol"])
    w["cap"], w["vol63_d"] = np.exp(w["fv_logcap"]), w["vol63"]
    for col in ("y_mom6n", "y_sharpe", "y_vol"):
        w[col] = np.nan
    w.to_parquet(FEATS_W, index=False)
    mm = m[m["member"]]
    print(f"hebdomadaire : {len(w)} lignes, {w['date'].nunique()} fins de semaine", flush=True)
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
    w = pd.read_parquet(FEATS_W).sort_values(["date", "symbol"]).reset_index(drop=True)
    Xw = w[cols].to_numpy(np.float32)
    wk = np.searchsorted(months, w["date"].to_numpy(), side="right") - 1      # dernier mois clos à la date
    outw = w[["date", "symbol"] + LABELS].copy()
    imp = {}
    for tgt in TARGETS:
        y = m[f"y_{tgt}"]
        rank = (y.where(m["member"]).groupby(m["date"]).rank(pct=True) - 0.5).to_numpy()
        pred, predw = np.full(len(m), np.nan), np.full(len(w), np.nan)
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
            tw = wk == k                                    # semaines du mois suivant : même modèle, features du jour
            if tw.any():
                predw[tw] = np.mean([b.predict(xgb.DMatrix(Xw[tw])) for b in models], axis=0)
        out[f"pred_{tgt}"] = pred[mem]
        outw[f"pred_{tgt}"] = predw
        print(f"cible {tgt} : {int(np.isfinite(pred).sum())} prédictions, {out.loc[out[f'pred_{tgt}'].notna(), 'date'].nunique()} mois",
              flush=True)
    out = out[out[[f"pred_{t}" for t in TARGETS]].notna().all(axis=1)]
    out.to_parquet(PREDS, index=False)
    outw[outw[[f"pred_{t}" for t in TARGETS]].notna().all(axis=1)].to_parquet(PREDS_W, index=False)
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
