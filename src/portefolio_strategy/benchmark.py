"""Backtest hors échantillon — portage de ref/MyQTM/src/search_params.py + benchmark.py.

ref/MyQTM optimisait les seuils par CMA-ES sur la période même du benchmark. Ici les seuils sont choisis
en walk-forward : pour l'année N, CMA-ES n'optimise que sur les prédictions hors échantillon antérieures
au 1er janvier N (seuils par défaut tant qu'il y a moins de PARAM_MIN_MONTHS mois d'historique).

Sorties (outputs/portefolio_strategy) : backtest.png, metrics.json, equity.csv, positions.csv,
params_by_year.json.

    python -m src.portefolio_strategy.benchmark
"""
import json
from multiprocessing import Pool

import cma
import matplotlib
import numpy as np
import pandas as pd

from . import config as C
from . import features as F
from . import trade as T
from .train import PREDICTIONS

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

KEYS = ("long_open", "long_close", "short_open", "short_close", "increase")
_TD = {}


def _objective(args):
    x, end, seed = args
    prm = dict(zip(KEYS, np.clip(x, 0.01, 0.999)))
    scores = []
    for k in range(1 + C.CMA_DROPOUT_ROUNDS):
        res = T.simulate(_TD["data"], lambda d: prm, remove_stocks=0 if k == 0 else C.CMA_DROPOUT,
                         seed=seed + k, end=end)
        scores.append(T.perf_score(res, prm))
    return -float(np.mean(scores))


def optimise(end, pool, seed=42):
    """CMA-ES sur les dates < end. → (params, score) ; seuils par défaut si rien de mieux."""
    x0 = [C.DEFAULT_PARAMS[k] for k in KEYS]
    base = -_objective((x0, end, seed))
    es = cma.CMAEvolutionStrategy(x0, C.CMA_STD, {"bounds": [0.01, 0.999], "popsize": C.CMA_POP, "seed": seed,
                                                  "verbose": -9, "maxiter": C.CMA_ITERS})
    while not es.stop():
        xs = es.ask()
        es.tell(xs, pool.map(_objective, [(list(x), end, seed) for x in xs]))
    best = -es.result.fbest
    if best > base:
        return dict(zip(KEYS, (float(v) for v in np.clip(es.result.xbest, 0.01, 0.999)))), best
    return dict(C.DEFAULT_PARAMS), base


def walk_forward_params(trade_data):
    dates = list(trade_data)
    first = pd.Timestamp(dates[0])
    out = {}
    with Pool(C.WORKERS) as pool:
        for year in range(first.year, pd.Timestamp(dates[-1]).year + 1):
            start = pd.Timestamp(f"{year}-01-01")
            months = (start - first).days / 30.44
            if months < C.PARAM_MIN_MONTHS:
                out[year] = {"params": dict(C.DEFAULT_PARAMS), "source": "défaut (historique insuffisant)"}
            else:
                prm, sc = optimise(f"{year}-01-01", pool)
                out[year] = {"params": prm, "score": sc, "source": f"CMA-ES sur {dates[0]} → {year - 1}-12-31"}
            print(f"  seuils {year} : {out[year]['source']} "
                  f"{ {k: round(v, 3) for k, v in out[year]['params'].items()} }", flush=True)
    return out


def index_curve(dates, capital=C.INITIAL_CAPITAL):
    idx = pd.read_parquet(F.INDEX)["level"]
    s = idx.reindex(idx.index.union(dates)).ffill().reindex(dates)
    return s / s.iloc[0] * capital


def plot(res, bench, mc, m, mb, path, title):
    d = res["dates"]
    fig, ax = plt.subplots(3, 1, figsize=(13, 11), sharex=True, gridspec_kw={"height_ratios": [3, 1.2, 1]})
    if len(mc):
        ax[0].fill_between(d, np.percentile(mc, 5, axis=0), np.percentile(mc, 95, axis=0), color="tab:orange",
                           alpha=0.2, label=f"Monte-Carlo 5–95 % ({len(mc)} tirages, −{C.CMA_DROPOUT} titres/jour)")
    ax[0].plot(d, res["values"], color="tab:orange", lw=1.8, label="Stratégie (hors échantillon, net de frais)")
    ax[0].plot(d, bench.values, color="tab:blue", lw=1.3, label="Indice de l'univers (Nasdaq dynamique, pondéré capi)")
    ax[0].set_yscale("log")
    ax[0].set_ylabel("Valeur ($, log)")
    ax[0].legend(loc="upper left")
    ax[0].set_title(title)
    txt = (f"Stratégie : {m['return_pct']:+.0f} %  CAGR {m['cagr_pct']:.1f} %  maxDD {m['max_drawdown_pct']:.1f} %  "
           f"Sharpe {m['sharpe']:.2f}\nIndice    : {mb['return_pct']:+.0f} %  CAGR {mb['cagr_pct']:.1f} %  "
           f"maxDD {mb['max_drawdown_pct']:.1f} %  Sharpe {mb['sharpe']:.2f}")
    ax[0].text(0.99, 0.03, txt, transform=ax[0].transAxes, ha="right", va="bottom", family="monospace",
               bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    for v, c, lab in ((res["values"], "tab:orange", "Stratégie"), (bench.values, "tab:blue", "Indice")):
        peak = np.maximum.accumulate(v)
        ax[1].plot(d, 100 * (v - peak) / peak, color=c, lw=1, label=lab)
    ax[1].set_ylabel("Drawdown (%)")
    ax[1].legend(loc="lower left")
    ax[2].plot(d, res["counts"], color="tab:gray", lw=1)
    ax[2].set_ylabel("Positions ouvertes")
    ax2 = ax[2].twinx()
    ax2.plot(d, 100 * res["cash"] / res["values"], color="tab:green", lw=0.8, alpha=0.7)
    ax2.set_ylabel("Liquidités (%)", color="tab:green")
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    preds = pd.read_parquet(PREDICTIONS)
    td = T.build_trade_data(preds)
    _TD["data"] = td
    by_year = walk_forward_params(td)

    def params_for(d):
        return by_year[int(d[:4])]["params"]

    res = T.simulate(td, params_for)
    mc = np.array([T.simulate(td, params_for, remove_stocks=C.CMA_DROPOUT, seed=100 + i)["values"]
                   for i in range(C.MC_RUNS)])
    bench = index_curve(res["dates"])
    m, mb = T.metrics(res["dates"], res["values"], C.INITIAL_CAPITAL), T.metrics(res["dates"], bench.values)
    allpos = res["history"] + res["open_positions"]
    pos = pd.DataFrame(allpos)
    stats = {}
    if len(pos):
        for typ, g in pos.groupby("type"):
            stats[typ] = {"n": int(len(g)), "win_rate_pct": round(float((g["gain"] > 0).mean() * 100), 1),
                          "mean_gain_pct": round(float(g["gain"].mean()), 2)}
    train_rep = json.loads((C.OUT / "train_report.json").read_text()) if (C.OUT / "train_report.json").exists() else {}
    sent_cov = train_rep.get("sentiment_coverage", 0)
    out = {"period": [f"{res['dates'][0]:%Y-%m-%d}", f"{res['dates'][-1]:%Y-%m-%d}"],
           "sentiment_coverage": sent_cov, "fee_per_side": C.FEE,
           "strategy": m, "index": mb, "positions": stats,
           "monte_carlo_final_value": {"p5": float(np.percentile(mc[:, -1], 5)), "median": float(np.median(mc[:, -1])),
                                       "p95": float(np.percentile(mc[:, -1], 95))},
           "mean_positions": float(res["counts"].mean()), "mean_cash_pct": float(100 * (res["cash"] / res["values"]).mean())}
    C.OUT.mkdir(parents=True, exist_ok=True)
    (C.OUT / "metrics.json").write_text(json.dumps(out, indent=2))
    (C.OUT / "params_by_year.json").write_text(json.dumps(by_year, indent=2, ensure_ascii=False))
    pd.DataFrame({"date": res["dates"], "strategy": res["values"], "index": bench.values, "cash": res["cash"],
                  "positions": res["counts"]}).to_csv(C.OUT / "equity.csv", index=False)
    if len(pos):
        pos.sort_values("start").to_csv(C.OUT / "positions.csv", index=False)
    note = "sans sentiment (corpus non traité)" if sent_cov < 0.05 else f"sentiment PIT sur {sent_cov:.0%} des lignes"
    plot(res, bench, mc, m, mb, C.OUT / "backtest.png",
         f"portefolio_strategy — XGBoost walk-forward sur l'univers Nasdaq dynamique, {note}")
    print(f"période {out['period'][0]} → {out['period'][1]}")
    for name, x in (("stratégie", m), ("indice", mb)):
        print(f"  {name:9} : {x['return_pct']:+8.1f} %  CAGR {x['cagr_pct']:6.1f} %  maxDD {x['max_drawdown_pct']:6.1f} %  "
              f"Sharpe {x['sharpe']:.2f}  par an {x['annual_roi_pct']}")
    print(f"  positions : {stats} ; moyenne {out['mean_positions']:.1f} ouvertes, liquidités {out['mean_cash_pct']:.0f} %")
    return out


if __name__ == "__main__":
    main()
