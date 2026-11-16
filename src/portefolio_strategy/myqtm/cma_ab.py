"""Seuils d'entrée réglés par CMA-ES sur la ligne de probabilités croisées A↔B de chaque coupe, testés après.

À chaque coupe trimestrielle T du walk-forward (train.py --save-val) :
- la fenêtre d'apprentissage 2020 → T est découpée en blocs A / B de 90 j entrelacés ; le modèle A (appris sur A,
  arrêté sur B) prédit les blocs B, le modèle B prédit les blocs A → une ligne continue de probabilités
  quasiment hors échantillon sur toute la fenêtre (val_all*.parquet), journées complètes ;
- CMA-ES règle les seuils du moteur ref/MyQTM (trade.simulate : long_open, increase ; sortie EMA 3 < EMA 10) sur
  CETTE ligne seulement : objectif Sharpe max (sharpe_stop.best_thresholds, défaut) ou perf_score de ref/MyQTM
  avec retraits aléatoires de titres (--objective=perf, benchmark.optimise) ;
- le vrai test : ces seuils sont appliqués au trimestre [T, T suivant), prédit par la moyenne des modèles A et B
  (predictions*.parquet), une seule simulation continue de 2024 à la fin.
Comparaison : mêmes prédictions avec les seuils par défaut (C.DEFAULT_PARAMS), et l'indice de l'univers.

    python -m src.portefolio_strategy.myqtm.cma_ab [--pred=ts0] [--avg] [--sfx=_ss] [--objective=perf]
        # --avg : moyenne des graines 42 / 7 / 123 ; --sfx=_ss : modèles arrêtés au Sharpe (train.py --stop=sharpe)
        # --no-cma : pas de réglage, seuils par défaut seulement (ex. --sfx=_ssf, train.py --stop=sharpe-fixed)
"""
import json
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from . import benchmark as BM
from . import config as C
from . import features as F
from . import sharpe_stop as SS
from . import trade as T

SEEDS = ("", "s7", "s123")


def load(pred, avg, sfx=""):
    """→ (ligne croisée de chaque coupe, prédictions hors échantillon) avec o, c. sfx : réglage (ex. _ss)."""
    names = [f"{pred}{s}{sfx}" for s in SEEDS] if avg else [pred + sfx]
    px = pd.read_parquet(F.FEATURES, columns=["date", "symbol", "o", "c"])
    probs = ["ybear", "yrange", "ybull"]
    val = pd.concat([pd.read_parquet(C.DATA / f"val_all_{n}_from2024.parquet") for n in names])
    val = val.groupby(["cutoff", "date", "symbol"], as_index=False)[probs].mean().merge(px, on=["date", "symbol"])
    oos = pd.concat([pd.read_parquet(C.DATA / f"predictions_{n}_from2024.parquet", columns=["date", "symbol"] + probs)
                     for n in names])
    oos = oos.groupby(["date", "symbol"], as_index=False)[probs].mean().merge(px, on=["date", "symbol"])
    return val, oos


def summary(res, capital=C.INITIAL_CAPITAL):
    m = T.metrics(res["dates"], res["values"], capital)
    pos = pd.DataFrame(res["history"] + res["open_positions"])
    return dict(cagr_pct=round(m["cagr_pct"], 2), max_dd_pct=round(m["max_drawdown_pct"], 2),
                sharpe=round(m["sharpe"], 3), trades=len(pos),
                gagnants_pct=round(float((pos["gain"] > 0).mean() * 100), 1) if len(pos) else None,
                gain_moy_pct=round(float(pos["gain"].mean()), 2) if len(pos) else None,
                positions_moy=round(float(res["counts"].mean()), 2),
                cash_moy_pct=round(float(100 * (res["cash"] / res["values"]).mean()), 1),
                annuel=m["annual_roi_pct"])


def main():
    pred = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--pred=")), "ts0")
    avg = "--avg" in sys.argv
    sfx = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--sfx=")), "")
    objective = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--objective=")), "sharpe")
    tag = pred + sfx + ("_avg" if avg else "") + ("" if objective == "sharpe" else "_perf")
    val, oos = load(pred, avg, sfx)
    cuts = sorted(val["cutoff"].unique())
    q = [0.5, 0.9, 0.99]
    print(f"{tag} : P(haussier) quantiles {q} — ligne croisée {np.round(val['ybull'].quantile(q).values, 3)}, "
          f"hors échantillon {np.round(oos['ybull'].quantile(q).values, 3)}", flush=True)
    params, t0 = {}, time.time()
    for c in (cuts if "--no-cma" not in sys.argv else cuts[:0]):   # --no-cma : seuils par défaut seulement
        td = T.build_trade_data(val[val["cutoff"] == c])
        if objective == "sharpe":                           # Sharpe max de la ligne croisée (simulation continue)
            score, prm = SS.best_thresholds([td])
        else:                                               # objectif perf_score de ref/MyQTM
            BM._TD["data"] = td
            with Pool(C.WORKERS) as pool:                   # après BM._TD : les processus héritent des données
                prm, score = BM.optimise(None, pool)
        ins = summary(T.simulate(td, lambda d, p=prm: p))
        key = f"{pd.Timestamp(c):%Y-%m-%d}"
        params[key] = dict(params=prm, score=score, ligne_croisee=ins)
        print(f"  coupe {key} : { {k: round(v, 3) for k, v in prm.items()} }  sur la ligne croisée : CAGR "
              f"{ins['cagr_pct']} %, Sharpe {ins['sharpe']}, {ins['trades']} trades  [{time.time() - t0:.0f} s]",
              flush=True)
    keys = sorted(params)

    def params_for(d):
        if not keys:
            return dict(C.DEFAULT_PARAMS)
        return params[max([k for k in keys if k <= d] or keys[:1])]["params"]

    td = T.build_trade_data(oos)
    res, dft = T.simulate(td, params_for), T.simulate(td, lambda d: dict(C.DEFAULT_PARAMS))
    bench = BM.index_curve(res["dates"])
    out = dict(pred=tag, periode=[f"{res['dates'][0]:%Y-%m-%d}", f"{res['dates'][-1]:%Y-%m-%d}"],
               cma_ab=summary(res), defaut=summary(dft),
               indice={k: v for k, v in T.metrics(res["dates"], bench.values).items()
                       if k in ("cagr_pct", "max_drawdown_pct", "sharpe", "annual_roi_pct")},
               seuils_par_coupe=params)
    (C.OUT / "cma_ab").mkdir(parents=True, exist_ok=True)
    (C.OUT / "cma_ab" / f"{tag}.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=float))
    print(f"\nhors échantillon {out['periode'][0]} → {out['periode'][1]}")
    for name in ("cma_ab", "defaut"):
        print(f"  {name:7} : {out[name]}")
    print(f"  indice  : {out['indice']}")


if __name__ == "__main__":
    main()
