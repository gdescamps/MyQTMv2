"""Variantes de filtrage des features (blocs A/B) : comparaison sur 2024 → fin, lots dynamiques (tous les trades).

Variantes (toutes en walk-forward, prédictions 100 % hors échantillon ; celles entraînées dès 2021 sont
identiques sur 2024+ à celles entraînées dès 2024) :
- sans A/B          : un modèle par coupe, toutes les features, 110 arbres (pas de filtrage) ;
- A/B 10            : 10 features ;            - A/B 20/30 (sévère) ;      - A/B 40 ;
- A/B 60/80 (normal);  - A/B 100/120 ;          - A/B toutes stables (features présentes dans les 6 modèles).
Pour chaque variante : F1 hors échantillon (début de tendance / tous les jours) et portefeuille lots dynamiques
1/n pour top 1 … 15 (myqtm_topk_dynamic.simulate), plus l'indice de l'univers.

    PYTHONPATH=.:myfiles python myfiles/myqtm_filter_variants.py
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import myqtm_close_threshold as B
from myqtm_topk_dynamic import simulate
from myqtm_topk_slots import TOPS, signals, stats
from src.portefolio_strategy.myqtm import features as F
from src.portefolio_strategy.myqtm import train as T
from src.portefolio_strategy.myqtm.trade import ema_exit

FIRST = "2024-01-01"
# nom → (arguments de train.predictions_path / out_dir, nombre de valeurs de features essayées)
VARIANTS = {
    "sans A/B": (dict(ab=False, first=FIRST), 0),
    "A/B 10": (dict(tops=[10], first=FIRST), 1),
    "A/B 20/30 (sévère)": (dict(severe=True), 2),
    "A/B 40": (dict(tops=[40], first=FIRST), 1),
    "A/B 60/80 (normal)": (dict(first=FIRST), 2),
    "A/B 100/120": (dict(tops=[100, 120], first=FIRST), 2),
    "A/B toutes stables": (dict(tops=[999], first=FIRST), 1),
}


def training(kw, n_tops):
    """Apprentissages derrière les prédictions 2024+ : coupes (réentraînements trimestriels), modèles XGBoost
    entraînés (A/B : 6 pour l'importance des features + A et B par nombre de features essayé), modèles qui
    prédisent (A/B : A + B ; sans A/B : 1), lignes d'apprentissage, features retenues, arbres par modèle."""
    r = json.loads((T.out_dir(**kw) / "train_report.json").read_text())
    cuts = [c for c in r["cutoffs"] if c["cutoff"] >= FIRST]
    per_cut = 6 + 2 * n_tops if n_tops else 1
    return dict(coupes=len(cuts), modeles_entraines=len(cuts) * per_cut,
                modeles_predicteurs=len(cuts) * (2 if n_tops else 1),
                lignes_appr_min=min(c["train_rows"] for c in cuts), lignes_appr_max=max(c["train_rows"] for c in cuts),
                features_med=float(np.median([c["top_features"] for c in cuts])),
                arbres_moy=float(np.mean([x for c in cuts for x in c["rounds"]])))


def f1s(path):
    p = pd.read_parquet(path)
    p = p[(p["date"] >= FIRST) & p["y"].notna()]
    ok = p["f1_keep"].astype(bool)
    return (f1_score(p.loc[ok, "y"].astype(int), p.loc[ok, "cls"], average="macro"),
            f1_score(p["y"].astype(int), p["cls"], average="macro"))


def main():
    allf = pd.read_parquet(F.FEATURES, columns=["date", "symbol", "o"])
    opens_all = allf.pivot(index="date", columns="symbol", values="o").sort_index()
    nd = pd.Series(opens_all.index[1:].append(pd.DatetimeIndex([pd.NaT])), index=opens_all.index)
    rows, summary = [], []
    for name, (kw, n_tops) in VARIANTS.items():
        path = T.predictions_path(**kw)
        if not path.exists():
            print(f"absent : {path}")
            continue
        T.PREDICTIONS = path
        p = B.load()
        p = p[p["date"] >= FIRST].reset_index(drop=True)
        p["x_ema"] = p.groupby("symbol")["c"].transform(ema_exit).astype(bool)
        p["next_date"] = p["date"].map(nd)
        start, end = p["date"].min(), p["date"].max()
        opens = opens_all.loc[start:end]
        f1h, f1a = f1s(path)
        for k in TOPS:
            cand = signals(p, k)
            cand = cand[cand["exit"] <= end]
            v, n, rot = simulate(cand, opens)
            rows.append(dict(variante=name, top=k, trades=len(cand), pos_moy=n.mean(), **stats(v)))
        r = pd.DataFrame([x for x in rows if x["variante"] == name])
        summary.append(dict(variante=name, **training(kw, n_tops), trades_top1=r.loc[r.top == 1, "trades"].iat[0],
                            trades_top1_15=int(r["trades"].sum()), f1_debut=f1h, f1_tous=f1a,
                            top1_rdt=r.loc[r.top == 1, "rendement_an_pct"].iat[0],
                            top1_sharpe=r.loc[r.top == 1, "sharpe"].iat[0],
                            top2_5_rdt_moy=r.loc[r.top.between(2, 5), "rendement_an_pct"].mean(),
                            top6_15_rdt_moy=r.loc[r.top >= 6, "rendement_an_pct"].mean(),
                            sharpe_moy=r["sharpe"].mean(), tops_battant_indice=0))
    idx = stats(pd.read_parquet(F.INDEX)["level"].loc[start:end])
    out, summ = pd.DataFrame(rows), pd.DataFrame(summary)
    summ["tops_battant_indice"] = summ["variante"].map(
        out[out["rendement_an_pct"] > idx["rendement_an_pct"]].groupby("variante").size()).fillna(0).astype(int)
    out.to_csv("myfiles/myqtm_filter_variants.csv", index=False)
    summ.to_csv("myfiles/myqtm_filter_variants_summary.csv", index=False)
    pd.set_option("display.width", 220, "display.float_format", "{:.2f}".format)
    print(f"lots dynamiques 1/n, tous les trades, {start:%Y-%m-%d} → {end:%Y-%m-%d}")
    print(out.pivot(index="top", columns="variante", values="rendement_an_pct")[list(summ["variante"])].to_string())
    print("\nsharpe :")
    print(out.pivot(index="top", columns="variante", values="sharpe")[list(summ["variante"])].to_string())
    print("\n" + summ.to_string(index=False))
    print(f"\nindice de l'univers : {idx}")


if __name__ == "__main__":
    main()
