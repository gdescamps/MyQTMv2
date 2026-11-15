"""Précision du classifieur myqtm sur le début des tendances, selon le seuil de confiance.

Pour chaque palier H (5, 10, 15, 20, 25 % premiers jours de chaque tendance haussière / baissière), le jeu
d'évaluation = jours de range + jours de tendance dont la phase < H (les fins de tendance sont exclues).
Pour chaque seuil t sur P(classe) : précision = part des jours prédits (P ≥ t) qui sont bien un début de
tendance de ce sens ; rappel = part des débuts de tendance de ce sens détectés ; base = précision du hasard.

    PYTHONPATH=. python myfiles/myqtm_confidence_head.py
"""
import numpy as np
import pandas as pd

from src.portefolio_strategy.myqtm import config as C
from src.portefolio_strategy.myqtm import features as F
from src.portefolio_strategy.myqtm import train as T
from src.portefolio_strategy.myqtm.trends import trend_labels

HEADS = (0.05, 0.10, 0.15, 0.20, 0.25)
THRESHOLDS = (0.33, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def labels_with_phase(df):
    """Labels définitifs (cours complets) + phase dans la tendance, alignés sur df."""
    y = pd.Series(np.nan, index=df.index)
    ph = pd.Series(np.nan, index=df.index)
    for _, g in df.groupby("symbol", sort=False):
        lab, phase, final_until = trend_labels(g.set_index("date")["c"], with_phase=True, **C.TREND)
        ok = g["date"].to_numpy() < np.datetime64(final_until)
        y.loc[g.index[ok]] = lab.to_numpy()[ok]
        ph.loc[g.index[ok]] = phase.to_numpy()[ok]
    return y, ph


def main():
    df = pd.read_parquet(F.FEATURES, columns=["date", "symbol", "c"]).sort_values(["symbol", "date"])
    df = df.reset_index(drop=True)
    df["y"], df["phase"] = labels_with_phase(df)
    p = pd.read_parquet(T.PREDICTIONS)[["date", "symbol", "ybear", "ybull"]].merge(df, on=["date", "symbol"])
    p = p[p["y"].notna()]
    res = []
    for col, cls, name in (("ybull", 2, "haussier"), ("ybear", 0, "baissier")):
        for h in HEADS:
            ev = p[(p["y"] == 1) | (p["phase"] < h)]           # range + début de tendance
            pos = ev["y"] == cls
            for t in THRESHOLDS:
                s = ev[col] >= t
                res.append(dict(classe=name, palier=f"{int(h * 100)} %", seuil=t, base=pos.mean(),
                                jours_predits=int(s.sum()), precision=pos[s].mean() if s.any() else np.nan,
                                rappel=(s & pos).sum() / pos.sum(), lift=(pos[s].mean() / pos.mean()) if s.any() else np.nan))
    r = pd.DataFrame(res)
    r.to_csv("myfiles/myqtm_confidence_head.csv", index=False)
    pd.set_option("display.width", 200)
    for name in ("haussier", "baissier"):
        q = r[r["classe"] == name]
        print(f"\n=== {name.upper()} — précision (rappel) par palier de début de tendance et seuil sur P({name}) ===")
        base = q.groupby("palier", sort=False)["base"].first()
        print("base (hasard) :", "  ".join(f"{k} {v:.1%}" for k, v in base.items()))
        tab = q.assign(cell=q.apply(lambda x: f"{x.precision:.1%} ({x.rappel:.0%})", axis=1)).pivot(
            index="seuil", columns="palier", values="cell")[[f"{int(h * 100)} %" for h in HEADS]]
        print(tab.to_string())
        lift = q.pivot(index="seuil", columns="palier", values="lift")[[f"{int(h * 100)} %" for h in HEADS]]
        print("lift (précision / base) :\n" + lift.round(2).to_string())


if __name__ == "__main__":
    main()
