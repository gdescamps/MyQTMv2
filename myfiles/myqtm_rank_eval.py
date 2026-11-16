"""Évaluation par jour (protocole « choisir sur 2024-25, vérifier sur 2026 ») de fichiers de prédictions.

Pour chaque jour et chaque modèle : gain du trade (entrée ouverture J+1, sortie règle EMA 3 < EMA 10, frais
compris, train_rank.trade_gains) du n° 1 du jour, moyenne des n° 1-3 et 1-5, moyenne de tous les titres
(référence) et corrélation de rang (IC) score ↔ gain sur la journée. Beaucoup plus de points (~700 jours) que
le Sharpe d'un portefeuille de quelques dizaines de trades.

    PYTHONPATH=.:myfiles python myfiles/myqtm_rank_eval.py [--rank]   # --rank : + variantes de train_rank
"""
import sys

import pandas as pd

from src.portefolio_strategy.myqtm import config as C
from src.portefolio_strategy.myqtm import features as F
from src.portefolio_strategy.myqtm import train_rank as R

MODELS = {"ts0 graine 42": C.DATA / "predictions_ts0_from2024.parquet",
          "ts0 graine 7": C.DATA / "predictions_ts0s7_from2024.parquet",
          "ts0 graine 123": C.DATA / "predictions_ts0s123_from2024.parquet",
          "ts0 moyenne 3 graines": C.DATA / "predictions_ts0avg_from2024.parquet",
          **{"ts0 " + p.name.split("_")[2]: p for p in sorted(C.DATA.glob("predictions_ts0_x*_from2024.parquet"))},
          **{"ts0 moyenne 3 graines " + p.name.split("_")[2]: p
             for p in sorted(C.DATA.glob("predictions_ts0avg_x*_from2024.parquet"))}}
RANK = {p.name[len("predictions_"):-len("_from2024.parquet")]: p
        for p in sorted(C.DATA.glob("predictions_rank*_from2024.parquet"))}
PERIODS = {"2024-25 (choix)": ("2024-01-01", "2026-01-01"), "2026 (vérification)": ("2026-01-01", "2027-01-01"),
           "2024-26": ("2024-01-01", "2027-01-01")}


def daily(p):
    p = p.dropna(subset=["g"]).copy()
    p["rk"] = p.groupby("date")["ybull"].rank(ascending=False, method="first")
    g = p.groupby("date")
    return pd.DataFrame({"n1": p[p["rk"] == 1].set_index("date")["g"],
                         "top3": p[p["rk"] <= 3].groupby("date")["g"].mean(),
                         "top5": p[p["rk"] <= 5].groupby("date")["g"].mean(),
                         "tous": g["g"].mean(),
                         "ic": g[["ybull", "g"]].apply(lambda x: x["ybull"].corr(x["g"], method="spearman"))})


def main():
    df = pd.read_parquet(F.FEATURES, columns=["date", "symbol", "o", "c"]).sort_values(["symbol", "date"])
    df = df.reset_index(drop=True)
    df["g"], _ = R.trade_gains(df)
    rows = []
    for name, path in {**MODELS, **(RANK if "--rank" in sys.argv else {})}.items():
        if not path.exists():
            continue
        p = pd.read_parquet(path, columns=["date", "symbol", "ybull"]).merge(df[["date", "symbol", "g"]],
                                                                             on=["date", "symbol"])
        d = daily(p)
        for per, (a, b) in PERIODS.items():
            x = d[(d.index >= a) & (d.index < b)]
            rows.append({"modèle": name, "période": per, "jours": len(x),
                         "n°1 gain moy %": 100 * x["n1"].mean(), "n°1 gain méd %": 100 * x["n1"].median(),
                         "n°1 > 0 %": 100 * (x["n1"] > 0).mean(),
                         "n°1 moy écrêté ±25 %": 100 * x["n1"].clip(-0.25, 0.25).mean(), "top3 moy %": 100 * x["top3"].mean(),
                         "top5 moy %": 100 * x["top5"].mean(), "tous moy %": 100 * x["tous"].mean(),
                         "IC moyen": x["ic"].mean(), "IC t-stat": x["ic"].mean() / x["ic"].std() * len(x) ** 0.5})
    out = pd.DataFrame(rows)
    out.to_csv("myfiles/myqtm_rank_eval.csv", index=False)
    pd.set_option("display.width", 250, "display.float_format", "{:.3f}".format)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
