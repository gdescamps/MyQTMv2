"""Arrêt anticipé au Sharpe (train.py --stop=sharpe) : nombre d'arbres et seuils choisis en tradant.

Le modèle A (appris sur les blocs A) grandit par paquets de STEP arbres. À chaque palier il prédit TOUTES les
lignes des blocs B (journées complètes, y compris les fins de tendance écartées par le sous-échantillonnage) ;
CMA-ES cherche les seuils du moteur ref/MyQTM (trade.simulate : long_open, increase ; sortie EMA 3 < EMA 10)
qui maximisent le Sharpe des blocs B. Chaque bloc de 90 j est simulé seul (pas de position portée d'un bloc à
l'autre) et le Sharpe est calculé sur les rendements quotidiens mis bout à bout. On garde le palier au meilleur
Sharpe ; arrêt après PATIENCE paliers sans amélioration. Symétrique pour le modèle B sur les blocs A.
FIXED (train.py --stop=sharpe-fixed) : pas de CMA-ES, le Sharpe de chaque palier est celui des seuils par défaut
(C.DEFAULT_PARAMS), ceux avec lesquels on trade ensuite.
"""
from multiprocessing import Pool

import cma
import numpy as np
import xgboost as xgb

from . import config as C
from . import trade as T

STEP = 50
PATIENCE = 3
CMA_ITERS = 15
CMA_POP = 10
KEYS = ("long_open", "increase")
FIXED = False                  # True (train.py --stop=sharpe-fixed) : Sharpe aux seuils par défaut, sans CMA-ES
_B = {}                        # données de trading, héritées par les processus du Pool (fork)


def blocks_data(rows, probs):
    """rows : (date, symbol, o, ema_exit, blk) ; probs (n, 3) → une trade_data par bloc."""
    p = rows.assign(ybear=probs[:, 0], yrange=probs[:, 1], ybull=probs[:, 2])
    return [T.build_trade_data(g) for _, g in p.groupby("blk", sort=True)]


def params_of(x):
    prm = dict(C.DEFAULT_PARAMS)
    prm.update(zip(KEYS, (float(v) for v in np.clip(x, 0.01, 0.999))))
    return prm


def sharpe(blocks, prm):
    rets = []
    for td in blocks:
        if len(td) < 5:
            continue
        v = T.simulate(td, lambda d: prm)["values"]
        rets.append(np.diff(v) / v[:-1])
    r = np.concatenate(rets) if rets else np.zeros(1)
    return float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0.0


def _obj(x):
    return -sharpe(_B["blocks"], params_of(x))


def best_thresholds(blocks, seed=42):
    """CMA-ES sur les seuils → (Sharpe, seuils) ; seuils par défaut s'ils font mieux."""
    _B["blocks"] = blocks
    x0 = [C.DEFAULT_PARAMS[k] for k in KEYS]
    best_x, best = x0, -_obj(x0)
    es = cma.CMAEvolutionStrategy(x0, C.CMA_STD, {"bounds": [0.01, 0.999], "popsize": CMA_POP, "seed": seed,
                                                  "verbose": -9, "maxiter": CMA_ITERS})
    with Pool(C.WORKERS) as pool:
        while not es.stop():
            xs = es.ask()
            fs = pool.map(_obj, [list(x) for x in xs])
            es.tell(xs, fs)
            k = int(np.argmin(fs))
            if -fs[k] > best:
                best, best_x = -fs[k], list(xs[k])
    return best, params_of(best_x)


def fit_sharpe(params, X, y, Xt, rows, w=None):
    """→ (booster, nombre d'arbres retenu, Sharpe, seuils). Xt / rows : lignes de trading de l'autre moitié."""
    dtr, dte = xgb.DMatrix(X, label=y, weight=w), xgb.DMatrix(Xt)
    booster, n, since = None, 0, 0
    best = (-np.inf, STEP, dict(C.DEFAULT_PARAMS))
    while n < C.XGB_ROUNDS:
        booster = xgb.train(params, dtr, num_boost_round=STEP, xgb_model=booster)
        n += STEP
        blocks = blocks_data(rows, booster.predict(dte))
        if FIXED:                                           # seuils par défaut, une seule simulation
            prm = dict(C.DEFAULT_PARAMS)
            sh = sharpe(blocks, prm)
        else:
            sh, prm = best_thresholds(blocks)
        if sh > best[0]:
            best, since = (sh, n, prm), 0
        else:
            since += 1
            if since >= PATIENCE:
                break
    return booster, best[1], best[0], best[2]
