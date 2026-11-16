"""Features à l'origine de chaque signal du top 1 sélectif (sous-échantillonnage sévère graine 42, top 0,7).

Pour chaque trade, jour du signal = séance avant l'entrée ; modèles A et B de la coupe trimestrielle qui a
produit la prédiction (outputs/.../ts0/from2024/models, train.py --save-models) ; contributions SHAP exactes
de XGBoost (pred_contribs) à la classe « haussier », en log-odds, moyenne de A et B. Contribution > 0 = la
feature a poussé vers « haussier ». Sorties : détail par trade (5 premières features), moyenne par feature et
par famille, comparaison gros gains (≥ 15 %) / autres.

    PYTHONPATH=.:myfiles python myfiles/myqtm_trade_shap.py
"""
import json
import re

import numpy as np
import pandas as pd
import xgboost as xgb

import myqtm_compare_seed_trades as M
from src.portefolio_strategy.myqtm import features as F
from src.portefolio_strategy.myqtm import train as T

MODELS = T.out_dir(first="2024-01-01", ts=0.0) / "models"
BIG = 15.0
FAMILIES = [("news structurées entreprise", r"^ev_co_"), ("news structurées concurrents", r"^ev_cp_"),
            ("news structurées macro", r"^ev_mac_"), ("contexte secteur / résultats", r"sect_r|rel_mkt|earn_days"),
            ("sentiment news (LLM)", r"news_sentiment|sentiment_vs"), ("nombre de news", r"vs_|news_days"),
            ("résultats (surprise)", r"es_surprise"), ("macro", r"ei_|vixp"), ("bilan", r"km_"),
            ("RSI", r"rsi"), ("momentum", r"momentum"), ("moyennes mobiles", r"ema"),
            ("volume", r"volume"), ("variation de cours", r"d_close|w_close")]


def family(f):
    return next((n for n, p in FAMILIES if re.search(p, f)), "autre")


def main():
    allf = pd.read_parquet(F.FEATURES)
    opens = allf.pivot(index="date", columns="symbol", values="o").sort_index().ffill()
    nd = pd.Series(opens.index[1:].append(pd.DatetimeIndex([pd.NaT])), index=opens.index)
    tr = M.trades(opens, nd, ts=0.0, seed=42).reset_index()
    days = opens.index
    tr["signal"] = [days[days.get_loc(e) - 1] for e in tr["entry"]]
    cuts = sorted(p.name for p in MODELS.iterdir())
    tr["cutoff"] = [max(c for c in cuts if c <= f"{d:%Y-%m-%d}") for d in tr["signal"]]
    rows = allf.set_index(["date", "symbol"])
    out = []
    for cut, g in tr.groupby("cutoff"):
        meta = json.loads((MODELS / cut / "meta.json").read_text())
        feats = meta["features"]
        X = rows.loc[list(zip(g["signal"], g["symbol"])), feats].to_numpy(np.float32)
        dm = xgb.DMatrix(X)
        ca = xgb.Booster(model_file=str(MODELS / cut / "A.json")).predict(dm, pred_contribs=True,
                                                                         iteration_range=(0, meta["na"]))
        cb = xgb.Booster(model_file=str(MODELS / cut / "B.json")).predict(dm, pred_contribs=True,
                                                                         iteration_range=(0, meta["nb"]))
        c = (ca[:, 2, :-1] + cb[:, 2, :-1]) / 2        # classe 2 = haussier ; dernière colonne = biais
        for k, r in enumerate(g.itertuples()):
            for j, f in enumerate(feats):
                out.append((r.symbol, r.signal, r.ret_pct, f, c[k, j], X[k, j]))
    s = pd.DataFrame(out, columns=["symbol", "signal", "ret_pct", "feature", "shap", "valeur"])
    s["famille"] = s["feature"].map(family)
    s.to_parquet("myfiles/myqtm_trade_shap.parquet", index=False)
    pd.set_option("display.width", 220, "display.float_format", "{:.3f}".format, "display.max_colwidth", 200)
    n = s.groupby(["symbol", "signal"]).ngroups
    print(f"{n} trades (graine 42, top 0,7) ; SHAP en log-odds de la classe haussier, moyenne modèles A/B\n")
    fam = s.groupby(["symbol", "signal", "famille"])["shap"].sum().reset_index()
    big = s.groupby(["symbol", "signal"])["ret_pct"].first() >= BIG
    fam["gros_gain"] = [big[(a, b)] for a, b in zip(fam["symbol"], fam["signal"])]
    print("=== contribution moyenne par famille (somme des features de la famille) ===")
    t = fam.groupby(["famille", "gros_gain"])["shap"].mean().unstack()
    t.columns = ["autres trades", "gros gains"]
    t["tous"] = fam.groupby("famille")["shap"].mean()
    print(t.sort_values("tous", ascending=False).to_string())
    print("\n=== 15 features les plus contributrices en moyenne ===")
    print(s.groupby("feature")["shap"].agg(["mean", lambda x: (x > 0).mean()]).rename(
        columns={"<lambda_0>": "part_positive"}).sort_values("mean", ascending=False).head(15).to_string())
    print("\n=== détail des plus gros gains : 5 features qui ont le plus poussé vers « haussier » ===")
    tops = s.drop_duplicates(["symbol", "signal"]).sort_values("ret_pct", ascending=False).head(10)
    for r in tops.itertuples():
        x = s[(s["symbol"] == r.symbol) & (s["signal"] == r.signal)].nlargest(5, "shap")
        lst = ", ".join(f"{f} {v:+.2f}" for f, v in zip(x["feature"], x["shap"]))
        ff = fam[(fam["symbol"] == r.symbol) & (fam["signal"] == r.signal)].set_index("famille")["shap"]
        print(f"{r.symbol:5s} {r.signal:%Y-%m-%d} {r.ret_pct:+6.1f} % | {lst}")
        print("      familles : " + ", ".join(f"{k} {v:+.2f}" for k, v in ff.sort_values(ascending=False).head(4).items()))


if __name__ == "__main__":
    main()
