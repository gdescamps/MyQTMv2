"""Classement à horizon fixe sur le S&P 500 : rendement relatif à 21 / 63 séances (sans tendance ni seuils).

Usage naturel des features fondamentales et sectorielles (fundamentals.py), qui bougent lentement : pour chaque
(titre, jour), cible = rang du jour, parmi les membres, du rendement de l'ouverture J+1 à l'ouverture J+1+h, relatif
à l'univers équipondéré. XGBoost en régression sur ce rang (réglages de portefolio_strategy.config, 300 arbres,
moyenne de 3 graines). Walk-forward trimestriel dès 2024 : apprentissage sur les lignes depuis 2020 dont la cible
est connue avant la coupe, une séance sur STEP (les cibles de jours voisins se recouvrent presque entièrement).

Sorties : data/portefolio_strategy/myqtm_sp/fwd_pred_h{h}.parquet (date, symbol, pred, y = rendement relatif),
outputs/portefolio_strategy/myqtm_sp/fwd_rank_importance_h{h}.csv.

    python -m src.portefolio_strategy.myqtm.fwd_rank
"""
import numpy as np
import pandas as pd
import xgboost as xgb

from . import config as C
from . import features as F
from . import train as T
from .. import config as PC
from .sp import use_sp

HORIZONS = (21, 63)
STEP = 5
SEEDS = (42, 7, 123)
FIRST = "2024-01-01"


def targets(df, h):
    """→ (rendement relatif h séances, date de fin) alignés sur df (trié par titre, date)."""
    g = df.groupby("symbol")["o"]
    o1, oh = g.shift(-1), g.shift(-(1 + h))
    ro = (g.shift(-1) / df["o"] - 1).clip(-0.5, 1.0).where(df["member"]).groupby(df["date"]).mean().sort_index()
    lo = (1 + ro.fillna(0)).cumprod().shift(1, fill_value=1.0)              # niveau à l'ouverture de chaque séance
    bench = (lo.shift(-(1 + h)) / lo.shift(-1) - 1)
    end = pd.Series(lo.index, index=lo.index).shift(-(1 + h))
    y = oh / o1 - 1 - df["date"].map(bench)
    return y.where(y.abs() < 2), df["date"].map(end)


def run(df, cols, h):
    y, end = targets(df, h)
    rank = y.where(df["member"]).groupby(df["date"]).rank(pct=True) - 0.5
    days = pd.Series(np.arange(df["date"].nunique()), index=np.sort(df["date"].unique()))
    sampled = (df["date"].map(days) % STEP == 0).to_numpy()
    last = df["date"].max()
    cuts = T.cutoffs(FIRST, last)
    preds, imp = [], {}
    for i, cut in enumerate(cuts):
        nxt = cuts[i + 1] if i + 1 < len(cuts) else last + pd.Timedelta(days=1)
        tr = (df["member"] & (df["date"] >= C.START_DATE) & (end < cut) & rank.notna()).to_numpy() & sampled
        te = (df["member"] & (df["date"] >= cut) & (df["date"] < nxt)).to_numpy()
        if not te.any():
            continue
        dtr = xgb.DMatrix(df.loc[tr, cols].to_numpy(np.float32), label=rank[tr].to_numpy(np.float32))
        dte = xgb.DMatrix(df.loc[te, cols].to_numpy(np.float32))
        p = 0.0
        for s in SEEDS:
            m = xgb.train(dict(PC.XGB_PARAMS, seed=s), dtr, num_boost_round=PC.XGB_ROUNDS)
            p = p + m.predict(dte) / len(SEEDS)
            imp[(f"{cut:%Y-%m-%d}", s)] = {cols[int(k[1:])]: v for k, v in m.get_score(importance_type="total_gain").items()}
        out = df.loc[te, ["date", "symbol"]].copy()
        out["pred"], out["y"] = p, y[te].to_numpy()
        preds.append(out)
        print(f"  h={h} coupe {cut:%Y-%m-%d} : {int(tr.sum())} lignes d'apprentissage, {int(te.sum())} à prédire",
              flush=True)
    im = pd.DataFrame(imp).fillna(0)
    return pd.concat(preds, ignore_index=True), (im / im.sum()).mean(axis=1).sort_values(ascending=False)


def main():
    use_sp()
    df = pd.read_parquet(F.FEATURES).sort_values(["symbol", "date"]).reset_index(drop=True)
    cols = F.feature_columns(df)
    for h in HORIZONS:
        p, im = run(df, cols, h)
        p.to_parquet(C.DATA / f"fwd_pred_h{h}.parquet", index=False)
        im.rename("part_du_gain").to_csv(C.OUT / f"fwd_rank_importance_h{h}.csv")
        print(f"h={h} : {len(p)} prédictions ; 10 premières features : {im.head(10).round(3).to_dict()}", flush=True)


if __name__ == "__main__":
    main()
