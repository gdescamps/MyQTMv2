"""Évaluation de l'indice concentré à rotation mensuelle (src/portefolio_strategy/myqtm/monthly_rank.py).

Chaque fin de mois, on garde les N premiers titres d'un classement, achetés à l'ouverture suivante et gardés un mois ;
frais 0,05 % par sens sur la rotation ; valeur quotidienne (drawdown réel). Classements comparés :
- modèle « rendement du mois suivant » et modèle « futurs leaders 6 mois » (top 20 et top 30, parts égales ; top 30
  aussi pondéré par la capitalisation) ;
- références : momentum 6 mois (top 30), les 30 plus grosses capitalisations (parts égales et pondérées), l'univers
  équipondéré et l'indice pondéré par la capitalisation.
Par portefeuille : rendement, bêta, alpha (contre l'univers équipondéré) et son t, drawdown, Sharpe, rotation.
Qualité du classement : corrélation de rang mensuelle avec le rendement du mois suivant, part des N titres choisis
qui finissent dans les N meilleurs du mois (hasard = N / 500), rendement relatif moyen des titres choisis.
Périodes : 2019-2023 (hors échantillon aussi : aucune donnée LLM ici), 2024 → fin, et l'ensemble.

    PYTHONPATH=.:myfiles python myfiles/myqtm_monthly_eval.py
"""
import numpy as np
import pandas as pd

from myqtm_alpha_eval import rel_stats
from src.portefolio_strategy.myqtm import config as C
from src.portefolio_strategy.myqtm import monthly_rank as M

PERIODS = {"2019-2023": ("2018-12-01", "2023-12-01"), "2024-26": ("2023-12-01", "2030-01-01"),
           "2019-2026": ("2018-12-01", "2030-01-01")}


def simulate(picks, opens, fee=C.FEE):
    """picks : [(entrée, sortie, Series titre → poids)] contigus → (valeur quotidienne, rotation moyenne par mois)."""
    v, prev, parts, turn = 1.0, pd.Series(dtype=float), [], []
    for entry, exit_, w in picks:
        both = w.index.union(prev.index)
        tr = (w.reindex(both, fill_value=0) - prev.reindex(both, fill_value=0)).abs().sum()
        v *= 1 - fee * tr
        turn.append(tr / 2)
        px = opens.loc[entry:exit_, w.index]
        rel = px / px.iloc[0]
        path = (rel * w).sum(axis=1)
        parts.append(v * path.iloc[:-1])
        prev = w * rel.iloc[-1] / path.iloc[-1]
        v *= path.iloc[-1]
    parts.append(pd.Series([v], index=[picks[-1][1]]))
    return pd.concat(parts), float(np.mean(turn[1:])) if len(turn) > 1 else np.nan


def main():
    p = pd.read_parquet(M.PREDS)
    opens = pd.read_parquet(M.OPENS)
    bench = pd.read_parquet(M.BENCH)
    p = p[p["exit"].notna()]
    rankings = {"modèle mois suivant": "pred_next", "modèle futurs leaders 6 mois": "pred_mom6n",
                "momentum 6 mois": "mom6", "plus grosses capitalisations": "cap"}
    pd.set_option("display.width", 250, "display.float_format", "{:.2f}".format, "display.max_rows", 200)
    rows, qual = [], []
    for per, (a, b) in PERIODS.items():
        d = p[(p["date"] >= a) & (p["date"] < b)]
        months = [g for _, g in d.groupby("date")]
        ref = bench["ro"].loc[months[0]["entry"].iloc[0]:months[-1]["exit"].iloc[0]]
        ew = (1 + ref.fillna(0)).cumprod().shift(1, fill_value=1.0)
        rows.append(dict(période=per, portefeuille="univers équipondéré", titres=500, rotation_mois_pct=np.nan,
                         **{**rel_stats(ew, ref), "alpha_t": 0.0}))
        idx = bench["index_level"].loc[ref.index[0]:ref.index[-1]]
        rows.append(dict(période=per, portefeuille="indice pondéré capitalisation", titres=500, rotation_mois_pct=np.nan,
                         **rel_stats(idx, ref)))
        specs = [(name, col, n, False) for name, col in rankings.items() for n in (20, 30)]
        specs += [("modèle mois suivant", "pred_next", 30, True), ("modèle futurs leaders 6 mois", "pred_mom6n", 30, True),
                  ("plus grosses capitalisations", "cap", 30, True)]
        for name, col, n, capw in specs:
            picks = []
            for g in months:
                top = g.nlargest(n, col)
                w = top["cap"] / top["cap"].sum() if capw else pd.Series(1.0 / len(top), index=top.index)
                picks.append((g["entry"].iloc[0], g["exit"].iloc[0], pd.Series(w.to_numpy(), index=top["symbol"].to_numpy())))
            v, turn = simulate(picks, opens)
            rows.append(dict(période=per, portefeuille=f"{name}, top {n}" + (" pondéré capi" if capw else ""), titres=n,
                             rotation_mois_pct=100 * turn, **rel_stats(v, ref)))
        for name, col in rankings.items():
            ic = d.groupby("date")[[col, "y_next"]].apply(lambda g: g[col].corr(g["y_next"], method="spearman"))
            r = dict(période=per, classement=name, mois=len(ic), ic_moyen=ic.mean(), ic_t=ic.mean() / ic.std() * np.sqrt(len(ic)))
            for n in (20, 30):
                hit, rel = [], []
                for g in months:
                    top = g.nlargest(n, col)
                    best = set(g.nlargest(n, "y_next")["symbol"])
                    hit.append(top["symbol"].isin(best).mean())
                    rel.append(top["y_next"].mean() - g["y_next"].mean())
                rel = pd.Series(rel)
                r[f"top{n}_dans_les_{n}_meilleurs_pct"] = 100 * np.mean(hit)
                r[f"top{n}_rel_mois_pct"], r[f"top{n}_rel_t"] = 100 * rel.mean(), rel.mean() / rel.std() * np.sqrt(len(rel))
                r[f"top{n}_mois_gagnants_pct"] = 100 * (rel > 0).mean()
            qual.append(r)
    out, q = pd.DataFrame(rows), pd.DataFrame(qual)
    out.to_csv("myfiles/myqtm_monthly_eval.csv", index=False)
    q.to_csv("myfiles/myqtm_monthly_quality.csv", index=False)
    cols = ["période", "portefeuille", "cagr_pct", "beta", "alpha_an_pct", "alpha_t", "max_dd_pct", "sharpe", "rotation_mois_pct"]
    print(out[cols].to_string(index=False))
    print("\nqualité du classement (cible : rendement du mois suivant ; hasard : 4 % pour 20 titres, 6 % pour 30)")
    print(q.to_string(index=False))


if __name__ == "__main__":
    main()
