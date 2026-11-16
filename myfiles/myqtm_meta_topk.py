"""Point 4 : second modèle (méta-étiquetage) sur les signaux du top k du jour.

Modèle de base = classifieur ts0 de référence, moyenne de 3 graines (--tag, défaut x4m100l20p100). Pour chaque coupe
trimestrielle c (2024-01 → 2026-10) :
- apprentissage : signaux du top k (entrée quand le titre entre dans les k premiers, sortie EMA 3 < EMA 10) construits
  sur les scores croisés A↔B du modèle de base de la coupe c (val_all*, train.py --save-val : toutes les lignes
  2020 → c, journées complètes, hors échantillon par bloc), trades fermés avant c ; cible = trade gagnant (frais
  compris) ;
- test : signaux du top k des prédictions hors échantillon du trimestre [c, c suivant) ;
- petit XGBoost (profondeur 2) sur ~20 features au jour du signal : position dans le classement du jour, écart au
  n° 1, calendrier et surprise de résultats, momentum relatif secteur / marché, RSI, VIX, news structurées.
Seuil causal : on garde un signal si son score méta ≥ quantile q des scores des signaux d'apprentissage de la coupe.

Comparaison sur les mêmes signaux : tous / filtre « résultats ≤ 20 j » / méta q = 0,3, 0,5, 0,7 / méta + filtre.
Portefeuille = lots dynamiques (myqtm_topk_dynamic.simulate), du 2024-01-02 à la fin.

    PYTHONPATH=.:myfiles python myfiles/myqtm_meta_topk.py [--tag=x4m100l20p100] [--max-k=4] [--big=5]
                                                    # --big=x : cible = gain > x % (défaut : trade gagnant)
"""
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from myqtm_topk_dynamic import simulate
from myqtm_topk_slots import signals, stats
from src.portefolio_strategy.myqtm import config as C
from src.portefolio_strategy.myqtm import features as F
from src.portefolio_strategy.myqtm.trade import ema_exit

FIRST = "2024-01-01"
SEEDS = ("ts0", "ts0s7", "ts0s123")
CTX = ["earn_days_to_next", "earn_days_since", "ts_es_surprise_0", "ts_es_surprise_1", "rel_mkt_r5", "rel_mkt_r21",
       "rel_sect_r5", "rel_sect_r21", "sect_r21", "rsi_14", "momentum_1w", "vixp_ema_50", "ev_co_product_pos_5",
       "ev_co_contract_pos_5", "ev_co_earnings_pos_5", "ev_co_analyst_pos_5", "ev_co_guidance_pos_5"]
BASE = ["rk", "gap1", "zday"]
QS = (0.3, 0.5, 0.7)
PARAMS = dict(objective="binary:logistic", max_depth=2, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
              min_child_weight=10, reg_lambda=4, nthread=C.WORKERS, seed=42, verbosity=0)
ROUNDS = 200


def base_feats(p):
    """Position du titre dans le classement du jour, écart de score au n° 1, score centré-réduit du jour."""
    g = p.groupby("date")["ybull"]
    p["rk"] = g.rank(ascending=False, method="first")
    p["gap1"] = g.transform("max") - p["ybull"]
    p["zday"] = (p["ybull"] - g.transform("mean")) / g.transform("std")
    return p


def cands(p, k, px, opens):
    """p (date, symbol, ybull) → signaux du top k avec gain réalisé et features au jour du signal."""
    p = base_feats(p.merge(px[["date", "symbol", "x_ema", "next_date"]], on=["date", "symbol"]))
    p = p.sort_values(["symbol", "date"]).reset_index(drop=True)
    s = signals(p, k)
    s = s[s["exit"] <= opens.index[-1]]
    s["gain"] = [opens.at[x, y] / opens.at[e, y] * (1 - C.FEE) ** 2 - 1
                 for e, x, y in zip(s["entry"], s["exit"], s["symbol"])]
    s = s.merge(p[["date", "symbol"] + BASE], left_on=["signal", "symbol"], right_on=["date", "symbol"])
    return s.drop(columns="date").merge(px[["date", "symbol"] + CTX], left_on=["signal", "symbol"],
                                        right_on=["date", "symbol"]).drop(columns="date")


def main():
    tag = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--tag=")), "x4m100l20p100")
    big = next((float(a.split("=", 1)[1]) for a in sys.argv if a.startswith("--big=")), 0.0)
    kmax = next((int(a.split("=", 1)[1]) for a in sys.argv if a.startswith("--max-k=")), 4)
    px = pd.read_parquet(F.FEATURES, columns=["date", "symbol", "o", "c"] + CTX).sort_values(["symbol", "date"])
    px["x_ema"] = px.groupby("symbol")["c"].transform(ema_exit).astype(bool)
    opens = px.pivot(index="date", columns="symbol", values="o").sort_index()
    nd = pd.Series(opens.index[1:].append(pd.DatetimeIndex([pd.NaT])), index=opens.index)
    px["next_date"] = px["date"].map(nd)
    oos = pd.read_parquet(C.DATA / f"predictions_ts0avg_{tag}_from2024.parquet", columns=["date", "symbol", "ybull"])
    vals = [pd.read_parquet(C.DATA / f"val_all_{s}_{tag}_from2024.parquet",
                            columns=["cutoff", "date", "symbol", "ybull"]) for s in SEEDS]
    val = pd.concat(vals).groupby(["cutoff", "date", "symbol"], as_index=False)["ybull"].mean()
    cuts = sorted(val["cutoff"].unique())
    feats = BASE + CTX
    rows, aucs = [], []
    for k in range(1, kmax + 1):
        test_all = cands(oos.copy(), k, px, opens)
        parts = []
        for i, c in enumerate(cuts):
            nxt = cuts[i + 1] if i + 1 < len(cuts) else pd.Timestamp.max
            tr = cands(val[val["cutoff"] == c][["date", "symbol", "ybull"]].copy(), k, px, opens)
            tr = tr[tr["exit"] < c]
            te = test_all[(test_all["signal"] >= c) & (test_all["signal"] < nxt)].copy()
            if te.empty:
                continue
            m = xgb.train(PARAMS, xgb.DMatrix(tr[feats].to_numpy(np.float32), label=(tr["gain"] > big / 100).astype(int)),
                          num_boost_round=ROUNDS)
            str_ = m.predict(xgb.DMatrix(tr[feats].to_numpy(np.float32)))
            te["meta"] = m.predict(xgb.DMatrix(te[feats].to_numpy(np.float32)))
            for q in QS:
                te[f"thr{q}"] = np.quantile(str_, q)
            te["n_train"] = len(tr)
            parts.append(te)
        te = pd.concat(parts, ignore_index=True)
        win = te["gain"] > big / 100
        for per, msk in (("2024-25", te["signal"] < "2026-01-01"), ("2026", te["signal"] >= "2026-01-01")):
            aucs.append(dict(top=k, période=per, signaux=int(msk.sum()), part_gagnants=win[msk].mean(),
                             auc_meta=roc_auc_score(win[msk], te.loc[msk, "meta"]),
                             auc_resultats_20j=roc_auc_score(win[msk], -te.loc[msk, "earn_days_to_next"])))
        earn = te["earn_days_to_next"] <= 20
        variants = {"tous": te["meta"] >= -1, "résultats ≤ 20 j": earn}
        for q in QS:
            variants[f"méta q{q}"] = te["meta"] >= te[f"thr{q}"]
            variants[f"méta q{q} + résultats"] = (te["meta"] >= te[f"thr{q}"]) & earn
        for name, msk in variants.items():
            c = te[msk]
            if len(c) < 3:
                continue
            v, _, _ = simulate(c, opens.loc[FIRST:])
            rows.append(dict(top=k, filtre=name, trades=len(c), gain_moy_pct=100 * c["gain"].mean(),
                             gain_med_pct=100 * c["gain"].median(), gagnants_pct=100 * (c["gain"] > 0).mean(),
                             apprentissage_moy=int(te.loc[msk, "n_train"].mean()), **stats(v)))
        print(f"top {k} : {len(te)} signaux hors échantillon, {int(te['n_train'].mean())} signaux d'apprentissage "
              f"en moyenne par coupe", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(f"myfiles/myqtm_meta_topk_{tag}" + (f"_big{big:g}" if big else "") + ".csv", index=False)
    pd.set_option("display.width", 250, "display.float_format", "{:.2f}".format)
    print(pd.DataFrame(aucs).to_string(index=False))
    for v in ("trades", "gagnants_pct", "gain_moy_pct", "rendement_an_pct", "max_drawdown_pct", "sharpe"):
        print(f"\n{v}")
        print(out.pivot(index="filtre", columns="top", values=v).reindex(list(variants)).to_string())


if __name__ == "__main__":
    main()
