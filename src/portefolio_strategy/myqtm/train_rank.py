"""Variante « classement par jour sur le gain du trade » du walk-forward de train.py.

Deux changements par rapport au classifieur de tendance (train.py) :
1. cible = gain réel du trade qu'on ouvrirait sur la ligne : entrée à l'ouverture J+1, sortie à l'ouverture qui
   suit la première clôture J' > J où EMA 3 < EMA 10 (trade.ema_exit), frais C.FEE par sens. Les fins de tendance
   deviennent d'elles-mêmes de mauvais trades (entrée tardive, sortie proche) : plus de sous-échantillonnage.
   Le gain n'est connu qu'une fois le trade fermé : à la coupe T, seules les lignes dont l'ouverture de sortie
   est < T entrent dans l'apprentissage (pas de fuite).
2. objectif = classement des titres de chaque jour (XGBoost rank:ndcg, groupe = date) au lieu d'une
   classification ligne par ligne : seul l'ordre entre titres du même jour compte, avec le poids sur le haut
   de la liste. Pertinence = note par quantile du gain dans la journée (GRADES).

La mécanique de train.py est conservée : coupes trimestrielles, fenêtre depuis C.START_DATE, blocs A/B de 90 j
entrelacés, 6 modèles pour le classement stable des features (moyenne / écart-type^puissance), choix entre
C.TOP_FEATURES par la métrique de validation croisée A↔B (NDCG@3 au lieu du F1), prédiction = moyenne A et B.

Sortie : data/portefolio_strategy/myqtm/predictions_rank_from2024.parquet, colonnes compatibles avec les scripts
d'évaluation (ybull = score de classement ; ybear = yrange = 0, sans signification ici).

    python -m src.portefolio_strategy.myqtm.train_rank --first=2024-01-01 [--save-models]
        [--metric=ndcg@10] [--objective=rank:pairwise] [--patience=100] [--risk-adj] [--trend] [--tag=nom]
"""
import json
import sys
import time

import numpy as np
import pandas as pd
import xgboost as xgb

from . import config as C
from . import features as F
from . import train as T
from .trade import ema_exit

# Note de pertinence par quantile du gain dans la journée : ≥ q → note.
GRADES = ((0.95, 4), (0.90, 3), (0.80, 2), (0.60, 1))
RANK_PARAMS = dict(objective="rank:ndcg", eval_metric="ndcg@3", lambdarank_pair_method="topk",
                   lambdarank_num_pair_per_sample=8, ndcg_exp_gain=True)
PATIENCE = C.XGB_PATIENCE
RISK_ADJ = False               # --risk-adj : note sur gain / volatilité 20 j du titre (évite les « billets de loterie »)
TREND = False                  # --trend : pertinence = étiquette de tendance (trend_grades) au lieu du gain
TAG = ""                       # suffixe des fichiers de sortie (--tag=)


def trade_gains(df):
    """df trié par (symbol, date) avec o, c → (gain, date d'ouverture de sortie) alignés sur df. NaN si le trade
    n'est pas fermé dans les données."""
    gain = np.full(len(df), np.nan)
    exit_open = np.full(len(df), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    for _, g in df.groupby("symbol", sort=False):
        idx = g.index.to_numpy()
        o, d = g["o"].to_numpy(float), g["date"].to_numpy()
        ex = ema_exit(g["c"]).to_numpy(bool)
        n, nxt = len(g), -1
        nxt_true = np.full(n, -1)                 # première ligne j > i où la règle de sortie se déclenche
        for i in range(n - 1, -1, -1):
            nxt_true[i] = nxt
            if ex[i]:
                nxt = i
        ok = (nxt_true >= 0) & (nxt_true + 1 < n)
        i = np.flatnonzero(ok)
        j = nxt_true[ok] + 1
        gain[idx[i]] = o[j] / o[i + 1] * (1 - C.FEE) ** 2 - 1
        exit_open[idx[i]] = d[j]
    return pd.Series(gain, index=df.index), pd.Series(exit_open, index=df.index)


def grades(dates, gain):
    q = pd.Series(gain).groupby(np.asarray(dates)).rank(pct=True).to_numpy()
    g = np.zeros(len(q), dtype=np.float32)
    for thr, v in sorted(GRADES):
        g[q >= thr] = v
    return g


def _params():
    p = dict(T._params(), **RANK_PARAMS)
    for k in ("num_class", "disable_default_eval_metric"):
        p.pop(k, None)
    return p


def _dm(X, y, qid):
    return xgb.DMatrix(X, label=y, qid=qid)


def fit(X, y, q, Xv, yv, qv, rounds=C.XGB_ROUNDS):
    m = xgb.train(_params(), _dm(X, y, q), num_boost_round=rounds, evals=[(_dm(Xv, yv, qv), "eval")],
                  early_stopping_rounds=PATIENCE, verbose_eval=False)
    return m, m.best_iteration + 1, m.best_score


def select_features(X, y, q, idx, cols):
    a, b = idx % 2 == 0, idx % 2 == 1
    imp = {}
    for k in range(6):
        part = idx % 6 == k
        if part.sum() < 200:
            continue
        other = b if k % 2 == 0 else a
        m, _, _ = fit(X[part], y[part], q[part], X[other], y[other], q[other], rounds=200)
        s = m.get_score(importance_type="weight")
        imp[k] = [s.get(f"f{j}", 0) for j in range(len(cols))]
    d = pd.DataFrame(imp, index=cols)
    d = d / d.max().replace(0, 1)
    d = d[(d != 0).all(axis=1)]
    d["mean"], d["std"] = d.mean(axis=1), d.std(axis=1)
    d["score"] = d["mean"] / (d["std"] ** C.MEAN_STD_POWER + 1e-9)
    return d.sort_values("score", ascending=False)


def trend_grades(df, cutoff):
    """Point 2 seul (--trend) : pertinence tirée de l'étiquette de tendance de train.py, définitive avant la coupe :
    début de tendance haussière (phase < C.TAIL_RAMP_START) → 3, reste de la tendance haussière → 1, range et
    baisse → 0, NaN sans étiquette."""
    y, _, phase = T.labels_before(df, cutoff, with_phase=True)
    g = np.where(y == 2, np.where(phase.to_numpy() < C.TAIL_RAMP_START, 3.0, 1.0), 0.0)
    return pd.Series(np.where(y.notna(), g, np.nan), index=df.index)


def train_cutoff(df, cols, gain, exit_open, cutoff, nxt, save=None, lab=None):
    """lab : notes de pertinence alignées sur df (NaN = pas d'étiquette) à la place des quantiles de gain."""
    if lab is not None:
        gain, exit_open = lab, pd.Series(pd.Timestamp.min, index=df.index)
    tr = (df["member"] & (df["date"] >= C.START_DATE) & (exit_open < cutoff) & gain.notna()).to_numpy()
    te = (df["member"] & (df["date"] >= cutoff) & (df["date"] < nxt)).to_numpy()
    if not te.any():
        return None
    sub = df.loc[tr, ["date"]].assign(g=gain[tr].to_numpy()).sort_values("date", kind="stable")
    rows = sub.index.to_numpy()
    X = df.loc[rows, cols].to_numpy(np.float32)
    y = sub["g"].to_numpy(np.float32) if lab is not None else grades(sub["date"], sub["g"])
    q = ((sub["date"] - pd.Timestamp(C.START_DATE)).dt.days).to_numpy()
    idx = T.interval_index(sub["date"], C.START_DATE)
    a, b = idx % 2 == 0, idx % 2 == 1
    rank = select_features(X, y, q, idx, cols)
    best = None
    for top in C.TOP_FEATURES:
        sel = [cols.index(c) for c in rank.index[:top]]
        ma, na, sa = fit(X[a][:, sel], y[a], q[a], X[b][:, sel], y[b], q[b])
        mb, nb, sb = fit(X[b][:, sel], y[b], q[b], X[a][:, sel], y[a], q[a])
        s = (sa + sb) / 2
        if best is None or s > best["ndcg"]:
            best = dict(ndcg=s, sel=sel, ma=ma, mb=mb, na=na, nb=nb, top=len(sel))
    Xt = xgb.DMatrix(df.loc[te, cols].to_numpy(np.float32)[:, best["sel"]])
    p = (best["ma"].predict(Xt, iteration_range=(0, best["na"]))
         + best["mb"].predict(Xt, iteration_range=(0, best["nb"]))) / 2
    out = df.loc[te, ["date", "symbol", "o", "c"]].copy()
    out["ybull"], out["ybear"], out["yrange"] = p, 0.0, 0.0
    if save is not None:
        d = save / f"{cutoff:%Y-%m-%d}"
        d.mkdir(parents=True, exist_ok=True)
        best["ma"].save_model(str(d / "A.json"))
        best["mb"].save_model(str(d / "B.json"))
        (d / "meta.json").write_text(json.dumps(dict(features=[cols[j] for j in best["sel"]],
                                                     na=best["na"], nb=best["nb"])))
    info = dict(cutoff=f"{cutoff:%Y-%m-%d}", train_rows=int(tr.sum()), test_rows=int(te.sum()),
                train_until=f"{df.loc[tr, 'date'].max():%Y-%m-%d}", ndcg3_cv=round(float(best["ndcg"]), 4),
                top_features=best["top"], rounds=[best["na"], best["nb"]],
                target_mean=round(float(gain[tr].mean()), 5))
    return out, info, rank


def predictions_path(first):
    return C.DATA / f"predictions_rank{TAG}{T._first_sfx(first)}.parquet"


def out_dir(first):
    return C.OUT / f"rank{TAG}" / (T._first_sfx(first).lstrip("_") or "base")


def main(first=None, save_models=False):
    df = pd.read_parquet(F.FEATURES).sort_values(["symbol", "date"]).reset_index(drop=True)
    cols = F.feature_columns(df, "all")
    gain, exit_open = trade_gains(df)
    target = gain
    if RISK_ADJ:
        vol = df.groupby("symbol")["c"].transform(lambda c: c.pct_change().rolling(20).std())
        target = gain / vol.clip(lower=0.005)
    out = out_dir(first)
    out.mkdir(parents=True, exist_ok=True)
    print(f"classement par jour sur le gain du trade : {len(cols)} features, {int(gain.notna().sum())} trades "
          f"étiquetés", flush=True)
    last = df["date"].max()
    cuts = T.cutoffs(first or C.FIRST_OOS_DATE, last)
    preds, infos, ranks, t0 = [], [], [], time.time()
    for i, cut in enumerate(cuts):
        nxt = cuts[i + 1] if i + 1 < len(cuts) else last + pd.Timedelta(days=1)
        r = train_cutoff(df, cols, target, exit_open, cut, nxt, out / "models" if save_models else None,
                         lab=trend_grades(df, cut) if TREND else None)
        if r is None:
            continue
        preds.append(r[0])
        infos.append(r[1])
        ranks.append(r[2]["score"].rename(r[1]["cutoff"]))
        print(f"  {r[1]}  [{time.time() - t0:.0f} s]", flush=True)
    p = pd.concat(preds, ignore_index=True)
    p.to_parquet(predictions_path(first), index=False)
    imp = pd.concat(ranks, axis=1)
    imp["mean_rank_score"] = imp.mean(axis=1)
    imp.sort_values("mean_rank_score", ascending=False).to_csv(out / "feature_importance.csv")
    (out / "train_report.json").write_text(json.dumps(dict(n_predictions=len(p), n_features=len(cols),
                                                           grades=GRADES, params=RANK_PARAMS, patience=PATIENCE,
                                                           risk_adj=RISK_ADJ, trend=TREND, cutoffs=infos),
                                                      indent=2))
    print(f"prédictions hors échantillon : {len(p)} lignes → {predictions_path(first)}", flush=True)
    return p


if __name__ == "__main__":
    for a in sys.argv[1:]:
        k, _, v = a.partition("=")
        if k == "--metric":
            RANK_PARAMS["eval_metric"] = v
        elif k == "--objective":
            RANK_PARAMS["objective"] = v
        elif k == "--patience":
            PATIENCE = int(v)
        elif k == "--risk-adj":
            RISK_ADJ = True
        elif k == "--trend":
            TREND = True
        elif k == "--tag":
            TAG = "_" + v
    main(first=next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--first=")), None),
         save_models="--save-models" in sys.argv)
