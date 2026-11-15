"""La confiance du classifieur myqtm augmente-t-elle la précision ?

Sur les prédictions hors échantillon (data/portefolio_strategy/myqtm/predictions.parquet), labels définitifs :
1. précision de « haussier » / « baissier » selon un seuil sur P(classe), tous les jours et début de tendance ;
2. signal d'entrée = premier jour où P(classe) franchit le seuil (comme le moteur de trading) : part des
   signaux tombant dans une tendance de ce sens, dans son début (20 % premiers jours), et rendement à venir
   (ouverture J+1 → ouverture J+1+h) relatif à la moyenne des membres du jour ;
3. exactitude par décile de marge (P max − P 2e).

    python myfiles/myqtm_confidence.py
"""
import numpy as np
import pandas as pd

from src.portefolio_strategy.myqtm import config as C
from src.portefolio_strategy.myqtm import features as F
from src.portefolio_strategy.myqtm import train as T

THRESHOLDS = (0.33, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
HORIZONS = (5, 21)


def load():
    df = pd.read_parquet(F.FEATURES, columns=["date", "symbol", "member", "o", "c"])
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    y, keep = T.labels_before(df, df["date"].max() + pd.Timedelta(days=1))
    # phase dans la tendance (pour « début de tendance » dans le sens du signal)
    df["y"], df["keep"] = y, keep
    g = df.groupby("symbol")["o"]
    for h in HORIZONS:
        df[f"f{h}"] = g.shift(-1 - h) / g.shift(-1) - 1
    p = pd.read_parquet(T.PREDICTIONS)[["date", "symbol", "ybear", "yrange", "ybull"]]
    p = p.merge(df, on=["date", "symbol"])
    for h in HORIZONS:                                   # rendement relatif à la moyenne des membres du jour
        p[f"r{h}"] = p[f"f{h}"] - p.groupby("date")[f"f{h}"].transform("mean")
    return p.sort_values(["symbol", "date"]).reset_index(drop=True)


def threshold_table(p, col, cls):
    lab = p["y"].notna()
    head = lab & p["keep"]
    rows = []
    for t in THRESHOLDS:
        s = p[col] >= t
        prev = s.groupby(p["symbol"]).shift(1, fill_value=False)
        entry = s & ~prev
        r = dict(seuil=t, jours=int((s & lab).sum()),
                 precision=(p.loc[s & lab, "y"] == cls).mean(),
                 precision_debut=(p.loc[s & head, "y"] == cls).mean(),
                 entrees=int((entry & lab).sum()),
                 entree_dans_tendance=(p.loc[entry & lab, "y"] == cls).mean(),
                 entree_debut_tendance=((p.loc[entry & lab, "y"] == cls) & p.loc[entry & lab, "keep"]).mean())
        for h in HORIZONS:
            sign = 1 if cls == 2 else -1
            r[f"rel_{h}j_%"] = 100 * sign * p.loc[entry, f"r{h}"].mean()
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    p = load()
    lab = p["y"].notna()
    print(f"{len(p)} prédictions, {lab.sum()} labellisées ; parts des classes : "
          f"baissier {(p.loc[lab, 'y'] == 0).mean():.3f}, range {(p.loc[lab, 'y'] == 1).mean():.3f}, "
          f"haussier {(p.loc[lab, 'y'] == 2).mean():.3f}")
    head = lab & p["keep"]
    print(f"début de tendance + range : baissier {(p.loc[head, 'y'] == 0).mean():.3f}, "
          f"haussier {(p.loc[head, 'y'] == 2).mean():.3f}\n")
    pd.set_option("display.width", 200, "display.float_format", "{:.3f}".format)
    for col, cls, name in (("ybull", 2, "HAUSSIER"), ("ybear", 0, "BAISSIER")):
        print(f"=== {name} : seuil sur P({name.lower()}) ===")
        print(threshold_table(p, col, cls).to_string(index=False), "\n")
    prob = p[["ybear", "yrange", "ybull"]].to_numpy()
    srt = np.sort(prob, axis=1)
    p["marge"], p["cls"] = srt[:, 2] - srt[:, 1], prob.argmax(axis=1)
    q = p[lab].copy()
    q["decile"] = pd.qcut(q["marge"], 10, labels=False) + 1
    out = q.groupby("decile").apply(lambda g: pd.Series({
        "marge_min": g["marge"].min(), "exactitude": (g["cls"] == g["y"]).mean(),
        "exactitude_debut": (g.loc[g["keep"], "cls"] == g.loc[g["keep"], "y"]).mean(),
        "part_tendance_predite": (g["cls"] != 1).mean()}), include_groups=False)
    print("=== exactitude par décile de marge (P max − P 2e) ===")
    print(out.to_string())
    by_year = p[lab & (p["ybull"] >= 0.6)].groupby(p["date"].dt.year).apply(
        lambda g: (g["y"] == 2).mean(), include_groups=False)
    print("\nprécision haussier (P ≥ 0,6) par année :", by_year.round(3).to_dict())


if __name__ == "__main__":
    main()
