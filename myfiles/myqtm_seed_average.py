"""Point 6 : moyenne des prédictions des graines de sous-échantillonnage sévère (ts0, graines 42 / 7 / 123).

Chaque graine = un tirage différent des fins de tendance retirées de l'apprentissage → 3 × (A + B) = 6 modèles ;
moyenne de ybear / yrange / ybull ligne à ligne → data/portefolio_strategy/myqtm/predictions_ts0avg_from2024.parquet.

    PYTHONPATH=.:myfiles python myfiles/myqtm_seed_average.py [--tag=x4m100l20p100]   # réglage XGBoost (train --xgb)
"""
import sys

import pandas as pd

from src.portefolio_strategy.myqtm import config as C

SEEDS = {42: "predictions_ts0_from2024.parquet", 7: "predictions_ts0s7_from2024.parquet",
         123: "predictions_ts0s123_from2024.parquet"}
KEY, PROBS = ["date", "symbol", "o", "c"], ["ybear", "yrange", "ybull"]


def main():
    tag = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--tag=")), "")
    files = {s: f.replace("_from2024", f"_{tag}_from2024") if tag else f for s, f in SEEDS.items()}
    out = C.DATA / f"predictions_ts0avg{'_' + tag if tag else ''}_from2024.parquet"
    ps = [pd.read_parquet(C.DATA / f, columns=KEY + PROBS).set_index(KEY) for f in files.values()]
    assert all(p.index.equals(ps[0].index) for p in ps), "lignes différentes entre graines"
    avg = sum(ps) / len(ps)
    avg.reset_index().to_parquet(out, index=False)
    top = [p["ybull"].groupby(level="date").idxmax().map(lambda t: t[1]) for p in ps + [avg]]
    print(f"{len(avg)} lignes → {out}")
    for s, t in zip(SEEDS, top[:3]):
        print(f"  n° 1 du jour identique à la moyenne : graine {s} {100 * (t == top[3]).mean():.0f} %")


if __name__ == "__main__":
    main()
