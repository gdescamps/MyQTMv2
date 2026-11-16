"""Évaluation du modèle de dérive post-annonce (src/portefolio_strategy/myqtm/earn_drift.py), 2024 → fin.

Par publication prédite hors échantillon (entrée à l'ouverture T+1) : rendement relatif à l'univers équipondéré à
5 / 20 / 60 séances, pour plusieurs sélections :
- toutes les publications ;
- règle classique : surprise > 0 et réaction relative > 0 ;
- modèle : prédiction ≥ seuil causal 80 % / 90 % ; prédiction ≤ seuil 20 % (côté « à éviter ») ;
t de Student sur les moyennes hebdomadaires (les publications d'une même semaine ne sont pas indépendantes).
Corrélation de rang prédiction ↔ rendement relatif 20 séances, par trimestre.
Portefeuilles : entrée ouverture T+1, sortie ouverture T+1+20, parts égales plafonnées à 1 / N_MIN (le reste en
cash), frais 0,05 % par sens ; alpha et bêta contre l'univers équipondéré (myqtm_alpha_eval.rel_stats).
Apprentissage : mêmes statistiques par sélection simple sur 2015-2023, pour situer l'effet historique.

    PYTHONPATH=.:myfiles python myfiles/myqtm_earn_drift_eval.py
"""
import numpy as np
import pandas as pd

from myqtm_alpha_eval import rel_stats
from src.portefolio_strategy.myqtm import config as C
from src.portefolio_strategy.myqtm import earn_drift as E

N_MIN = 10
SPLIT = pd.Timestamp("2026-01-01")


def simulate(cand, opens, n_min=N_MIN, fee=C.FEE):
    """Parts égales 1 / max(n, n_min) entre les positions ouvertes ; rééquilibrage à chaque entrée ou sortie."""
    dates, syms = opens.index, list(opens.columns)
    px = opens.to_numpy(float)
    held = np.zeros(px.shape, dtype=bool)
    pos, col = {d: i for i, d in enumerate(dates)}, {s: i for i, s in enumerate(syms)}
    for r in cand.itertuples():
        if r.entry in pos and r.exit in pos:
            held[pos[r.entry]:pos[r.exit], col[r.symbol]] = True
    n = held.sum(axis=1)
    v, w = np.ones(len(dates)), np.zeros(len(syms))
    for t in range(len(dates) - 1):
        target = held[t] / max(n[t], n_min)
        if not np.array_equal(target > 0, w > 0):
            v[t] *= 1 - fee * np.abs(target - w).sum()
            w = target
        g = np.nan_to_num(np.where(w > 0, px[t + 1] / px[t], 1.0), nan=1.0)
        gross = (w * g).sum() + (1 - w.sum())
        v[t + 1] = v[t] * gross
        w = w * g / gross
    return pd.Series(v, index=dates), n


def weekly_t(x, dates):
    w = x.groupby(dates.dt.to_period("W")).mean().dropna()
    return w.mean() / w.std() * np.sqrt(len(w)) if len(w) > 2 and w.std() > 0 else np.nan


def table(e, sels, label):
    rows = []
    for name, m in sels.items():
        x = e[m]
        r = dict(période=label, sélection=name, publications=len(x))
        for h in E.HORIZONS:
            y = x[f"y{h}"].dropna()
            y = y[y.abs() < 1.5]
            r[f"rel_{h}j_moy_pct"] = 100 * y.mean()
            r[f"rel_{h}j_t"] = weekly_t(y, x.loc[y.index, "tdate"])
        y = x["y20"].dropna()
        r["rel_20j_méd_pct"], r["gagnants_20j_pct"] = 100 * y.median(), 100 * (y > 0).mean()
        rows.append(r)
    return rows


def main():
    e = pd.read_parquet(E.EVENTS)
    opens = pd.read_parquet(E.OPENS)
    bench = pd.read_parquet(E.BENCH)
    pd.set_option("display.width", 250, "display.float_format", "{:.2f}".format, "display.max_rows", 200)
    simple = lambda d: {"toutes": d["react"].notna(),                                        # noqa: E731
                        "surprise > 0 et réaction > 0": (d["surprise"] > 0) & (d["react"] > 0),
                        "surprise < 0 et réaction < 0": (d["surprise"] < 0) & (d["react"] < 0),
                        "réaction > +5 %": d["react"] > 0.05, "réaction < −5 %": d["react"] < -0.05}
    tr = e[e["tdate"] < E.FIRST]
    rows = table(tr, simple(tr), "2015-2023 (apprentissage)")
    o = e[e["pred"].notna()].copy()
    for label, d in (("2024-25", o[o["tdate"] < SPLIT]), ("2026", o[o["tdate"] >= SPLIT]), ("2024-26", o)):
        sels = {**simple(d), "modèle ≥ seuil 80 %": d["pred"] >= d["thr80"], "modèle ≥ seuil 90 %": d["pred"] >= d["thr90"],
                "modèle ≤ seuil 20 %": d["pred"] <= d["thr20"]}
        rows += table(d, sels, label)
    out = pd.DataFrame(rows)
    out.to_csv("myfiles/myqtm_earn_drift_eval.csv", index=False)
    print(out.to_string(index=False))
    # corrélation de rang par trimestre
    ic = o.dropna(subset=["y20"]).groupby(o["tdate"].dt.to_period("Q")).apply(
        lambda d: pd.Series({"ic_modèle": d["pred"].corr(d["y20"], method="spearman"),
                             "ic_réaction": d["react"].corr(d["y20"], method="spearman"),
                             "ic_surprise": d["surprise"].corr(d["y20"], method="spearman"), "n": len(d)}))
    print("\ncorrélation de rang avec le rendement relatif à 20 séances, par trimestre")
    print(ic.round(3).to_string())
    print("moyenne :", ic.iloc[:, :3].mean().round(3).to_dict(),
          " t :", (ic.iloc[:, :3].mean() / ic.iloc[:, :3].std() * np.sqrt(len(ic))).round(2).to_dict())
    # portefeuilles
    ref = bench["ro"].loc[opens.index[0]:]
    o = o.dropna(subset=["entry"]).rename(columns={"end20": "exit"})
    o = o[o["exit"] > o["entry"]]
    ports = {"toutes": o, "surprise > 0 et réaction > 0": o[(o["surprise"] > 0) & (o["react"] > 0)],
             "modèle ≥ seuil 80 %": o[o["pred"] >= o["thr80"]], "modèle ≥ seuil 90 %": o[o["pred"] >= o["thr90"]]}
    prow = []
    ew = (1 + ref.fillna(0)).cumprod().shift(1, fill_value=1.0)
    prow.append(dict(portefeuille="univers équipondéré", trades=np.nan, pos_moy=np.nan,
                     **{**rel_stats(ew, ref), "alpha_t": 0.0, "ratio_info": 0.0}))
    for name, c in ports.items():
        v, n = simulate(c, opens)
        prow.append(dict(portefeuille=name, trades=len(c), pos_moy=n.mean(), **rel_stats(v, ref)))
    pf = pd.DataFrame(prow)
    pf.to_csv("myfiles/myqtm_earn_drift_portfolios.csv", index=False)
    print("\nportefeuilles (20 séances, parts ≤ 1/10)")
    print(pf.to_string(index=False))


if __name__ == "__main__":
    main()
