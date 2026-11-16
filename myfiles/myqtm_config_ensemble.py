"""Ensemble de tous les classifieurs ts0 entraînés (284 features) : réglages XGBoost × graines de sous-échantillonnage.

Le meilleur réglage (4 / 100 / 0,02) est un pic isolé parmi ses voisins → on moyenne aussi les réglages. Score de
chaque modèle converti en rang centile du jour (calibrations différentes), puis moyenne → ybull ; ybear = yrange = 0.
Sortie : data/portefolio_strategy/myqtm/predictions_ts0ens_from2024.parquet.

    PYTHONPATH=.:myfiles python myfiles/myqtm_config_ensemble.py
"""
import re

import pandas as pd

from src.portefolio_strategy.myqtm import config as C

PAT = re.compile(r"^predictions_ts0(s\d+)?(_x\w+)?_from2024\.parquet$")
KEY = ["date", "symbol", "o", "c"]


def main():
    files = sorted(p for p in C.DATA.glob("predictions_ts0*_from2024.parquet") if PAT.match(p.name))
    ranks = []
    for f in files:
        p = pd.read_parquet(f, columns=KEY + ["ybull"]).set_index(KEY)["ybull"]
        ranks.append(p.groupby(level="date").rank(pct=True))
        print(" ", f.name)
    out = pd.concat(ranks, axis=1).mean(axis=1).rename("ybull").reset_index()
    out["ybear"], out["yrange"] = 0.0, 0.0
    path = C.DATA / "predictions_ts0ens_from2024.parquet"
    out.to_parquet(path, index=False)
    print(f"{len(files)} modèles, {len(out)} lignes → {path}")


if __name__ == "__main__":
    main()
