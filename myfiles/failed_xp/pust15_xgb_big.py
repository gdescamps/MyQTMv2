"""XGBoost plus gros (10 -> ~9000 paramètres) : apprentissage années PAIRES, test années IMPAIRES.
Même objectif que pust15_xgb_tiny (utilité moyenne-variance de a = σ(f) sur le rendement QQQ des h jours suivants,
exécution J+1, purge des h derniers jours de chaque année paire), mêmes entrées que la formule.
Taille = arbres × (seuils + feuilles) ≈ n × (2^(d+1) − 1) (borne haute). eta 0.1, subsample 0.8.
    python -m myfiles.failed_xp.pust15_xgb_big
"""
import numpy as np
import pandas as pd
import xgboost as xgb

import myfiles.failed_xp.pust15_learned_evenodd as eo
from myfiles.failed_xp.pust15_xgb_tiny import X, fwd_ret, make_obj
from src.risk_off_strategy.data import load_macro  # noqa: F401

IDX, EVEN, C = eo.IDX, eo.EVEN, eo.C


def fit(h, gamma, n, d, eta=0.1, seed=0):
    r = fwd_ret(h)
    yr = IDX.year.values
    purge = np.zeros(len(C), bool)
    for y in np.unique(yr[EVEN]):
        purge[np.where(yr == y)[0][-h:]] = True
    m = EVEN & ~np.isnan(r) & ~purge & ~X.isna().any(axis=1).values
    params = dict(max_depth=d, eta=eta, reg_lambda=1.0, min_child_weight=5.0, base_score=0.0, subsample=0.8,
                  tree_method="hist", seed=seed)
    bst = xgb.train(params, xgb.DMatrix(X[m], label=r[m]), num_boost_round=n, obj=make_obj(r[m], gamma))
    f = bst.predict(xgb.DMatrix(X), output_margin=True)
    df = bst.trees_to_dataframe()
    k = len(df) + (df.Feature == "Leaf").sum() * 0     # nœuds = seuils + feuilles
    return 1 / (1 + np.exp(-f)), int(k)


def main():
    rows = [eo.row("FORMULE experte", 8, eo.formula(eo.FORM_DEFAULT))]
    for h in (5, 21, 63):
        for gamma in (2.0, 5.0, 10.0):
            for d in (2, 3, 4):
                for n in (10, 30, 100, 300):
                    al, k = fit(h, gamma, n, d)
                    r = eo.row(f"XGB h={h} γ={gamma:g} prof{d} ×{n}", k, al)
                    r.update(h=h, gamma=gamma, depth=d, trees=n, alloc_moy=round(al.mean(), 2), alloc_std=round(al.std(), 2))
                    rows.append(r)
            print(f"done h={h} γ={gamma}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv("myfiles/failed_xp/pust15_xgb_big.csv", index=False)


if __name__ == "__main__":
    main()
