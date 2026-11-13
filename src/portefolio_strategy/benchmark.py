"""Backtest hors échantillon du portefeuille long, et recherche du compromis de fréquence de rééquilibrage.

1. Grille : horizon du modèle × fréquence de rééquilibrage × nombre de titres × buffer × pondération
   (poids égaux ou capitalisation ; config.GRID_*), chaque
   configuration simulée nette de frais (config.FEE par sens sur le montant échangé) et brute.
2. Choix de la configuration en walk-forward (remplace les seuils CMA-ES) : pour l'année N, la configuration
   au meilleur ratio d'information net contre l'indice sur les dates antérieures au 1er janvier N
   (configuration par défaut tant qu'il y a moins de PARAM_MIN_MONTHS mois d'historique). La courbe
   « stratégie » enchaîne ces choix : elle n'utilise jamais une configuration choisie sur sa propre période.
3. Références : indice de l'univers pondéré par la capitalisation, et univers équipondéré (même moteur, mêmes
   frais) pour séparer le choix des titres de l'effet « poids égaux ».

Sorties (outputs/portefolio_strategy/<variante>) : backtest.png, metrics.json, grid.csv, frequency.csv,
equity.csv, holdings.csv, config_by_year.json.

    python -m src.portefolio_strategy.benchmark [variante]
"""
import itertools
import json

import matplotlib
import numpy as np
import pandas as pd

from . import config as C
from . import features as F
from . import portfolio as PF
from .train import out_dir, predictions_path

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

KEYS = ("horizon", "every", "n_top", "buffer", "weighting")


def load(variant):
    """→ (scores {horizon: dates × titres}, ouvertures dates × titres, capitalisations dates × titres)."""
    p = pd.read_parquet(predictions_path(variant))
    scores = {h: p.pivot(index="date", columns="symbol", values=f"s_{h}").sort_index() for h in C.HORIZONS}
    o = pd.read_parquet(F.FEATURES, columns=["date", "symbol", "o", "log_cap"])
    dates, cols = scores[C.HORIZONS[0]].index, scores[C.HORIZONS[0]].columns
    o = o[o["date"] >= dates[0]]
    opens = o.pivot(index="date", columns="symbol", values="o").reindex(dates)[cols]
    caps = np.exp(o.pivot(index="date", columns="symbol", values="log_cap").reindex(dates)[cols])
    return scores, opens, caps


def index_curve(dates):
    idx = pd.read_parquet(F.INDEX)["level"]
    s = idx.reindex(idx.index.union(dates)).ffill().reindex(dates)
    return s / s.iloc[0]


def sim(scores, opens, caps, key, fee=C.FEE):
    h, every, n, buf, wt = key
    return PF.simulate_offsets(scores[h], opens, n, every, buf, fee, size=caps if wt == "cap" else None)


def run_grid(scores, opens, caps, bench, fee=C.FEE):
    """→ (tableau des configurations, {clé: courbe nette moyenne des décalages})."""
    rows, curves = [], {}
    for key in itertools.product(C.HORIZONS, C.GRID_EVERY, C.GRID_TOP, C.GRID_BUFFER, C.GRID_WEIGHTING):
        v, to = sim(scores, opens, caps, key, fee)
        g, _ = sim(scores, opens, caps, key, 0.0)
        m, a = PF.metrics(v), PF.active(v, bench)
        curves[key] = v
        rows.append({**dict(zip(KEYS, key)), "cagr_pct": m["cagr_pct"],
                     "excess_cagr_pct": a["excess_cagr_pct"], "info_ratio": a["info_ratio"],
                     "max_drawdown_pct": m["max_drawdown_pct"], "sharpe": m["sharpe"], "turnover_per_year": to,
                     "fees_pct_per_year": 100 * fee * to, "gross_cagr_pct": PF.metrics(g)["cagr_pct"]})
    return pd.DataFrame(rows), curves


def info_ratio(values, bench, end):
    a = (values.pct_change() - bench.pct_change())[values.index < end].dropna()
    return float(a.mean() / a.std() * np.sqrt(252)) if len(a) > 20 and a.std() > 0 else -np.inf


def walk_forward(curves, bench, fee=C.FEE):
    """Choix annuel de la configuration sur le passé seulement. → (courbe enchaînée, {année: choix})."""
    dates = bench.index
    default = tuple(C.DEFAULT_CONFIG[k] for k in KEYS)
    by_year, ret, prev = {}, [], None
    for year in range(dates[0].year, dates[-1].year + 1):
        start = pd.Timestamp(f"{year}-01-01")
        if (start - dates[0]).days / 30.44 < C.PARAM_MIN_MONTHS:
            key, src = default, "défaut (historique insuffisant)"
        else:
            key = max(curves, key=lambda k: info_ratio(curves[k], bench, start))
            src = f"meilleur ratio d'information net sur {dates[0]:%Y-%m-%d} → {year - 1}-12-31"
        r = curves[key].pct_change().fillna(0.0)
        r = r[(r.index >= start) & (r.index < start + pd.DateOffset(years=1))].copy()
        if prev is not None and key != prev and len(r):
            r.iloc[0] -= 2 * fee                       # changement de configuration : portefeuille entièrement renouvelé
        by_year[year] = {**dict(zip(KEYS, key)), "source": src}
        ret.append(r)
        prev = key
    return (1 + pd.concat(ret)).cumprod(), by_year


def frequency_table(grid):
    """Compromis de fréquence : par nombre de lignes et période de rééquilibrage, sur les configurations de la grille."""
    g = grid.groupby(["n_top", "every"] if len(C.GRID_WEIGHTING) == 1 else ["weighting", "n_top", "every"])
    return pd.DataFrame({"excès net médian (pts/an)": g["excess_cagr_pct"].median(),
                         "excès net max (pts/an)": g["excess_cagr_pct"].max(),
                         "excès brut médian (pts/an)": g.apply(lambda d: (d["gross_cagr_pct"] - d["cagr_pct"]
                                                                          + d["excess_cagr_pct"]).median()),
                         "frais médians (%/an)": g["fees_pct_per_year"].median(),
                         "rotation médiane (×/an)": g["turnover_per_year"].median(),
                         "configurations battant l'indice": g.apply(lambda d: f"{(d['excess_cagr_pct'] > 0).sum()}/{len(d)}")
                         }).round(2)


def plot(strategy, best, bench, ew, cw, grid, m, path, title):
    d = strategy.index
    fig, ax = plt.subplots(3, 1, figsize=(13, 12), gridspec_kw={"height_ratios": [3, 1.2, 1.8]})
    ax[0].plot(d, strategy, color="tab:orange", lw=1.9, label="Stratégie (configuration choisie en walk-forward, nette de frais)")
    ax[0].plot(d, best, color="tab:orange", lw=0.9, ls=":", label="Meilleure configuration de la grille (a posteriori, optimiste)")
    ax[0].plot(d, bench, color="tab:blue", lw=1.4, label="Indice de l'univers (pondéré capitalisation)")
    ax[0].plot(d, ew, color="tab:gray", lw=1.1, label="Univers équipondéré (mêmes frais)")
    ax[0].plot(d, cw, color="tab:cyan", lw=0.9, ls="--", label="Univers pondéré capitalisation, même moteur (contrôle)")
    ax[0].set_yscale("log")
    ax[0].set_ylabel("Valeur (base 1, log)")
    ax[0].legend(loc="upper left")
    ax[0].set_title(title)
    txt = "\n".join(f"{n:11}: CAGR {x['cagr_pct']:5.1f} %  maxDD {x['max_drawdown_pct']:6.1f} %  Sharpe {x['sharpe']:.2f}"
                    for n, x in (("Stratégie", m["strategy"]), ("Indice", m["index"]), ("Équipondéré", m["equal_weight"])))
    ax[0].text(0.99, 0.03, txt, transform=ax[0].transAxes, ha="right", va="bottom", family="monospace",
               bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    for v, c in ((strategy, "tab:orange"), (bench, "tab:blue"), (ew, "tab:gray")):
        ax[1].plot(d, 100 * (v / v.cummax() - 1), color=c, lw=1)
    ax[1].set_ylabel("Drawdown (%)")
    ax[1].sharex(ax[0])
    med = grid.groupby(["weighting", "n_top", "every"])[["excess_cagr_pct", "gross_cagr_pct", "cagr_pct"]].median()
    for wt, ls in zip(C.GRID_WEIGHTING, ("-", ":")):
        for n, c in zip(C.GRID_TOP, ("tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink")):
            x = med.loc[(wt, n)]
            name = f"{n} lignes" + ("" if len(C.GRID_WEIGHTING) == 1 else f", {'capi' if wt == 'cap' else 'égaux'}")
            ax[2].plot(range(len(x)), x["excess_cagr_pct"], color=c, ls=ls, marker="o", ms=4, label=f"{name}, net")
            ax[2].plot(range(len(x)), x["excess_cagr_pct"] + x["gross_cagr_pct"] - x["cagr_pct"], color=c, ls="--",
                       lw=0.8, alpha=0.6, label=f"{name}, brut")
    ax[2].axhline(0, color="k", lw=0.8)
    ax[2].set_xticks(range(len(C.GRID_EVERY)), [f"{e} j" for e in C.GRID_EVERY])
    ax[2].set_xlabel("Fréquence de rééquilibrage (jours de bourse)")
    ax[2].set_ylabel("Écart de CAGR avec l'indice (pts/an)\nmédiane des horizons et buffers")
    ax[2].legend(ncol=3, fontsize=8)
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(variant="pit"):
    out = out_dir(variant)
    out.mkdir(parents=True, exist_ok=True)
    scores, opens, caps = load(variant)
    dates = opens.index
    bench = index_curve(dates)
    elig = scores[C.HORIZONS[0]].notna()
    ones = elig.astype(float).where(elig)
    ew = PF.simulate(ones, opens, None, 21)["values"]
    cw = PF.simulate(ones, opens, None, 21, size=caps)["values"]
    grid, curves = run_grid(scores, opens, caps, bench)
    strategy, by_year = walk_forward(curves, bench)
    best_key = max(curves, key=lambda k: PF.active(curves[k], bench)["excess_cagr_pct"])
    last = by_year[max(by_year)]
    sel = tuple(last[k] for k in KEYS)
    default = tuple(C.DEFAULT_CONFIG[k] for k in KEYS)
    fees = {f"{100 * f:.2f} %": round(PF.active(sim(scores, opens, caps, sel, f)[0], bench)["excess_cagr_pct"], 2)
            for f in C.FEE_SENSITIVITY}
    run = PF.simulate(scores[sel[0]], opens, sel[2], sel[1], sel[3], record=True, size=caps if sel[4] == "cap" else None)
    m = {"variant": variant, "description": F.VARIANTS[variant][1], "period": [f"{dates[0]:%Y-%m-%d}", f"{dates[-1]:%Y-%m-%d}"],
         "fee_per_side": C.FEE, "strategy": {**PF.metrics(strategy), **PF.active(strategy, bench)},
         "index": PF.metrics(bench), "equal_weight": {**PF.metrics(ew), **PF.active(ew, bench)},
         "cap_weight_engine": {**PF.metrics(cw), **PF.active(cw, bench)},
         "by_weighting": {wt: {"beating_index": int((g["excess_cagr_pct"] > 0).sum()), "configurations": len(g),
                               "median_excess_cagr_pct": round(float(g["excess_cagr_pct"].median()), 2)}
                          for wt, g in grid.groupby("weighting")},
         "default_config": {**C.DEFAULT_CONFIG, **PF.metrics(curves[default]), **PF.active(curves[default], bench)},
         "best_in_sample": {**dict(zip(KEYS, best_key)), **PF.metrics(curves[best_key]), **PF.active(curves[best_key], bench)},
         "grid": {"configurations": len(grid), "beating_index": int((grid["excess_cagr_pct"] > 0).sum()),
                  "median_excess_cagr_pct": float(grid["excess_cagr_pct"].median())},
         "selected_config": {k: last[k] for k in KEYS}, "selected_config_excess_cagr_by_fee": fees}
    freq = frequency_table(grid)
    (out / "metrics.json").write_text(json.dumps(m, indent=2, ensure_ascii=False))
    (out / "config_by_year.json").write_text(json.dumps(by_year, indent=2, ensure_ascii=False))
    grid.sort_values("excess_cagr_pct", ascending=False).round(3).to_csv(out / "grid.csv", index=False)
    freq.to_csv(out / "frequency.csv")
    pd.DataFrame({"strategy": strategy, "index": bench, "equal_weight": ew, "cap_weight_engine": cw, "default_config": curves[default],
                  "best_in_sample": curves[best_key]}).mul(C.INITIAL_CAPITAL).round(2).to_csv(out / "equity.csv")
    if run["weights"]:
        w = pd.concat(run["weights"], axis=1).T.rename_axis("date").stack().rename("weight").reset_index()
        w.to_csv(out / "holdings.csv", index=False)
    plot(strategy, curves[best_key], bench, ew, cw, grid, m, out / "backtest.png",
         f"portefolio_strategy — sélection longue dans l'univers Nasdaq dynamique ({F.VARIANTS[variant][1]})")
    print(f"variante « {variant} », période {m['period'][0]} → {m['period'][1]}, frais {100 * C.FEE:.2f} % par sens")
    for name, x in (("stratégie (walk-forward)", m["strategy"]), ("config. par défaut", m["default_config"]),
                    ("meilleure a posteriori", m["best_in_sample"]), ("univers équipondéré", m["equal_weight"]),
                    ("univers capi (contrôle)", m["cap_weight_engine"]),
                    ("indice", m["index"])):
        extra = f"  écart {x['excess_cagr_pct']:+5.1f} pts  RI {x['info_ratio']:+.2f}  années gagnantes {x['years_beating_index']}" \
            if "excess_cagr_pct" in x else ""
        print(f"  {name:24}: CAGR {x['cagr_pct']:5.1f} %  maxDD {x['max_drawdown_pct']:6.1f} %  Sharpe {x['sharpe']:.2f}{extra}")
    print(f"  grille : {m['grid']['beating_index']}/{m['grid']['configurations']} configurations battent l'indice, "
          f"écart médian {m['grid']['median_excess_cagr_pct']:+.1f} pts/an ; par pondération {m['by_weighting']}")
    print(f"  choix par année : { {y: tuple(v[k] for k in KEYS) for y, v in by_year.items()} }")
    print(freq.to_string())
    print(f"  configuration retenue {m['selected_config']} : écart de CAGR selon les frais par sens {fees}")
    return m


if __name__ == "__main__":
    import sys
    main(variant=sys.argv[1] if len(sys.argv) > 1 else "pit")
