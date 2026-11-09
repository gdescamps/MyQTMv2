"""XGBoost minuscule (≤ 10 paramètres) à la place de la formule experte.
Paramètres d'un modèle = Σ arbres (seuils + feuilles) : stump = 1 seuil + 2 feuilles = 3 ; arbre prof. 2 = 3 + 4 = 7.
Entrées (close t) = variables de la formule : g = prix/SMA250−1, sl = pente SMA250 40j, vol YZ10, NFCI, IPC YoY.
Objectif XGBoost personnalisé = utilité moyenne-variance de l'allocation a = σ(f) sur le rendement QQQ des
h jours suivants (t+1 .. t+h, cohérent avec l'exécution J+1) :  max Σ a·r − γ/2·a²·r²
-> le modèle apprend DIRECTEMENT l'allocation (pas de seuil de probabilité arbitraire).
Apprentissage = années PAIRES (purge des h derniers jours de chaque année paire : leur cible déborde sur l'impaire),
validation = années IMPAIRES. Backtest net déployé : E_MAX 1.5 + saturation, bande 0.50/1.00, frais 0.5 %.
    python -m myfiles.pust15_xgb_tiny
"""
import numpy as np
import pandas as pd
import xgboost as xgb

import src.risk_off_strategy.strategy as st
from myfiles.pust15_learned_evenodd import C, CP, EVEN, FUND, G, IDX, NF, ODD, P, SL, VOL, run, stitched, row as row_eo
from myfiles.pust17_emax_saturate import metr  # noqa: F401

X = pd.DataFrame({"g": G, "sl": SL, "vol": VOL,
                  "nfci": pd.Series(NF).ffill().values, "cpi": pd.Series(CP).ffill().values})


def fwd_ret(h):
    r = np.full(len(C), np.nan)
    r[:-h] = C[h:] / C[:-h] - 1          # alloc[t] (close t) gagne t+1.. : exec J+1 -> r de t+1 à t+h ≈ C[t+h]/C[t]
    return r


def make_obj(r, gamma, scale=100.0):
    def obj(pred, dtrain):
        a = 1 / (1 + np.exp(-pred))
        da = a * (1 - a)
        grad = -(r - gamma * a * r * r) * da * scale
        hess = (gamma * r * r * da * da + 1e-6) * scale
        return grad, hess
    return obj


def fit(h, gamma, n_trees, depth, lr=1.0):
    r = fwd_ret(h)
    yr = IDX.year.values
    purge = np.zeros(len(C), bool)
    for y in np.unique(yr[EVEN]):
        last = np.where(yr == y)[0][-h:]
        purge[last] = True
    m = EVEN & ~np.isnan(r) & ~purge & ~X.isna().any(axis=1).values
    d = xgb.DMatrix(X[m], label=r[m])
    params = dict(max_depth=depth, eta=lr, reg_lambda=0.0, min_child_weight=5.0, base_score=0.0, tree_method="exact")
    bst = xgb.train(params, d, num_boost_round=n_trees, obj=make_obj(r[m], gamma))
    f = bst.predict(xgb.DMatrix(X), output_margin=True)
    a = 1 / (1 + np.exp(-f))
    return np.clip(a, 0, 1), bst


def main():
    import myfiles.pust15_learned_evenodd as eo
    rows = [row_eo("FORMULE experte (8 param.)", 8, eo.formula(eo.FORM_DEFAULT))]
    dumps = {}
    for h in (5, 21, 63):
        for gamma in (2.0, 5.0, 10.0):
            for n_trees, depth in ((1, 1), (2, 1), (3, 1), (1, 2)):
                k = n_trees * (2 ** depth - 1 + 2 ** depth)
                al, bst = fit(h, gamma, n_trees, depth)
                lab = f"XGB h={h}j γ={gamma:g} {n_trees}×prof{depth}"
                r = row_eo(lab, k, al)
                r["alloc moy."] = round(al.mean(), 2)
                rows.append(r)
                dumps[lab] = bst.get_dump()
    df = pd.DataFrame(rows)
    df.to_csv("myfiles/pust15_xgb_tiny.csv", index=False)
    with open("myfiles/pust15_xgb_tiny_trees.txt", "w") as f:
        for k_, v in dumps.items():
            f.write(f"### {k_}\n" + "\n".join(v) + "\n")
    pd.set_option("display.width", 340)
    pd.set_option("display.max_rows", 200)
    cols = ["modèle", "param.", "alloc moy.", "PAIRES %/an", "PAIRES DD", "PAIRES Cal", "IMPAIRES %/an", "IMPAIRES DD",
            "IMPAIRES Cal", "tout %/an", "tout DD", "tout Cal", "10ans %/an", "10ans Cal", "5ans %/an", "ordres/an"]
    print(df[[c for c in cols if c in df]].to_string(index=False))


if __name__ == "__main__":
    main()
