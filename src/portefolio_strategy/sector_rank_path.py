"""Les secteurs qui entrent dans le top 5 de la semaine montent-ils régulièrement dans le classement avant ?

Classement hebdomadaire : rang (1 = meilleur) du rendement de la semaine qui suit chaque décision (secteurs du
S&P 500 reconstitué, paniers pondérés capitalisation, comme sector_oracle.py). « Entrée » = rang <= 5 la semaine w
alors que le rang était > 5 la semaine w−1. Témoin : les secteurs également hors du top 5 en w−1 qui n'y entrent pas.
On compare :
- les rangs hebdomadaires des semaines w−6 … w−1, puis w … w+3 (le secteur reste-t-il dans le top 5 ?) ;
- les rangs quotidiens des 15 jours de bourse avant la décision : rang du rendement du jour et rang du rendement
  cumulé sur 5 jours (connus à la clôture de chaque jour) ;
- ce qui est observable à la décision et pourrait annoncer l'entrée : progression du rang d'une semaine sur
  l'autre, nombre de semaines consécutives de progression, rang sur les 5 et 2 derniers jours.

Sorties : outputs/portefolio_strategy/sp500/sector_rank_path.md, sector_rank_path.png.

    python -m src.portefolio_strategy.sector_rank_path
"""
import matplotlib
import numpy as np
import pandas as pd

from . import config as C
from . import rotation_compare as RC
from . import sector_oracle as SO
from . import sp500 as SP

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = C.OUT / "sp500"
TOP = 5
WEEKS_BEFORE, WEEKS_AFTER, DAYS_BEFORE = 6, 3, 15


def daily_sector_returns(d, key, caps, start, sectors):
    x = d[(d["date"] >= start)].sort_values(["symbol", "date"]).copy()
    x["r1"] = x.groupby("symbol")["c"].pct_change(fill_method=None)
    x = x[x["member"]].assign(g=lambda t: t["symbol"].map(key)).dropna(subset=["g", "r1"])
    x = x[x["g"].isin(sectors)]
    x["month"] = x["date"].dt.to_period("M") - 1
    x = x.merge(caps, on=["month", "symbol"], how="left")
    x["w"] = x["cap"].fillna(x.groupby("date")["cap"].transform("median"))
    x["wr"] = x["w"] * x["r1"]
    g = x.groupby(["date", "g"])
    return (g["wr"].sum() / g["w"].sum()).unstack("g").sort_index()


def analyse(R, Rd, label):
    rank = R.rank(axis=1, ascending=False)                    # rang de la semaine qui suit chaque décision
    K = R.shape[1]
    r5 = (1 + Rd).rolling(5).apply(np.prod, raw=True) - 1
    r2 = (1 + Rd).rolling(2).apply(np.prod, raw=True) - 1
    drank, d5rank, d2rank = (z.rank(axis=1, ascending=False) for z in (Rd, r5, r2))
    dpos = pd.Series(np.arange(len(Rd)), index=Rd.index)
    rows = []
    for i in range(WEEKS_BEFORE + 1, len(R) - WEEKS_AFTER):
        J = R.index[i]
        if J not in dpos.index:
            continue
        p = dpos[J]
        for s in R.columns:
            if not rank.iloc[i - 1][s] > TOP or np.isnan(rank.iloc[i][s]):
                continue
            hist = [rank.iloc[i - k][s] for k in range(1, WEEKS_BEFORE + 1)]
            streak = 0
            for k in range(1, WEEKS_BEFORE):
                if hist[k - 1] < hist[k]:
                    streak += 1
                else:
                    break
            rows.append({"date": J, "secteur": s, "entre": rank.iloc[i][s] <= TOP, "excès": R.iloc[i][s] - R.iloc[i].mean(),
                         **{f"w-{k}": hist[k - 1] for k in range(1, WEEKS_BEFORE + 1)},
                         **{f"w+{k}": rank.iloc[i + k][s] for k in range(0, WEEKS_AFTER + 1)},
                         **{f"d-{l}": drank.iloc[p - l][s] for l in range(DAYS_BEFORE)},
                         **{f"c5-{l}": d5rank.iloc[p - l][s] for l in range(DAYS_BEFORE)},
                         "progression": hist[1] - hist[0], "série": streak, "rang 5 j": d5rank.iloc[p][s], "rang 2 j": d2rank.iloc[p][s]})
    ev = pd.DataFrame(rows)
    ent, out = ev[ev["entre"]], ev[~ev["entre"]]
    wcols = [f"w-{k}" for k in range(WEEKS_BEFORE, 0, -1)] + [f"w+{k}" for k in range(0, WEEKS_AFTER + 1)]
    path = pd.DataFrame({"entre dans le top 5": ent[wcols].mean(), "reste hors du top 5": out[wcols].mean(),
                         "écart (entrants − témoins)": ent[wcols].mean() - out[wcols].mean()}).T.round(2)
    path = path.rename(columns={"w+0": "w (semaine d'entrée)"})
    dcols = [f"d-{l}" for l in range(DAYS_BEFORE - 1, -1, -1)]
    ccols = [f"c5-{l}" for l in range(DAYS_BEFORE - 1, -1, -1)]
    daily = pd.DataFrame({"rang du jour — entrants": ent[dcols].mean().to_numpy(), "rang du jour — témoins": out[dcols].mean().to_numpy(),
                          "rang 5 j — entrants": ent[ccols].mean().to_numpy(), "rang 5 j — témoins": out[ccols].mean().to_numpy()},
                         index=[f"J−{l}" if l else "J" for l in range(DAYS_BEFORE - 1, -1, -1)]).T.round(2)
    base = ev["entre"].mean()

    def cond(col, bins, labels):
        c = pd.cut(ev[col], bins, labels=labels)
        t = ev.groupby(c, observed=True).agg(n=("entre", "size"), p=("entre", "mean"), ex=("excès", "mean"))
        t["p"] *= 100
        t["ex"] *= 100
        return t.round(2).rename(columns={"p": "entre dans le top 5 (%)", "ex": "écart semaine suivante (%)"})

    q = [0, round(K / 4), round(K / 2), round(3 * K / 4), K]               # quarts du classement
    qb, ql = [x + 0.5 for x in q], [f"{q[i] + 1}–{q[i + 1]}" for i in range(4)]
    qb[0] = 0
    tables = {"progression du rang de w−2 à w−1": cond("progression", [-99, -2.5, 0.5, 2.5, 99], ["recul ≥ 3", "recul ≤ 2 ou stable", "gain 1 à 2", "gain ≥ 3"]),
              "semaines consécutives de progression avant w": cond("série", [-1, 0, 1, 2, 99], ["0", "1", "2", "3 ou plus"]),
              "rang sur les 5 derniers jours (à J)": cond("rang 5 j", qb, ql),
              "rang sur les 2 derniers jours (à J)": cond("rang 2 j", qb, ql)}
    stay = (ent["w+1"] <= TOP).mean()
    md = [f"## {label} ({K} groupes ; {len(ent)} entrées dans le top {TOP}, {len(out)} témoins)", "",
          f"Hors du top {TOP} en w−1, un groupe y entre la semaine w dans **{100 * base:.1f} %** des cas "
          f"(il y en a {TOP} sur {K}, soit {100 * TOP / K:.1f} % au hasard). Après son entrée, il reste dans le top {TOP} "
          f"la semaine suivante dans {100 * stay:.1f} % des cas.", "",
          "### Rang hebdomadaire moyen (1 = meilleur)", "", path.to_markdown(), "",
          "### Rang quotidien moyen dans les jours qui précèdent la décision", "", daily.to_markdown(), ""]
    for name, t in tables.items():
        md += [f"### Selon la {name}" if name.startswith("progression") else f"### Selon le nombre de {name}" if name.startswith("semaines")
               else f"### Selon le {name}", "", t.to_markdown(), ""]
    return md, path, daily, K


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = RC.datasets()
    d, maps, _ = data["S&P 500 (reconstitué)"]
    caps = SO.monthly_caps(SP.CAPS)
    start = f"{pd.read_parquet(SP.UNIVERSE)['start'].min() + pd.Timedelta(days=40):%Y-%m-%d}"
    md = ["# Changements de rang avant l'entrée d'un secteur dans le top 5 de la semaine", "",
          f"S&P 500 reconstitué depuis {start[:7]}, décision tous les 5 jours de bourse, secteurs pondérés par la "
          "capitalisation. Rang 1 = meilleur rendement. Témoins = groupes hors du top 5 en w−1 qui n'y entrent pas en w.", ""]
    figs = []
    for level in ("secteur", "groupe d'industries"):
        R, _, _ = SO.sector_returns(d, maps[level], caps, start, 5)
        R = R.dropna(how="all")
        Rd = daily_sector_returns(d, maps[level], caps, start, list(R.columns))
        part, path, daily, K = analyse(R, Rd, f"Par {level}")
        md += part
        figs.append((level, path, daily, K))
    (OUT / "sector_rank_path.md").write_text("\n".join(md))
    fig, ax = plt.subplots(2, 2, figsize=(16, 10))
    for row, (level, path, daily, K) in enumerate(figs):
        x = list(range(-WEEKS_BEFORE, WEEKS_AFTER + 1))
        ax[row, 0].plot(x, path.iloc[0], marker="o", color="tab:green", label="entre dans le top 5 en w")
        ax[row, 0].plot(x, path.iloc[1], marker="o", color="tab:gray", label="reste hors du top 5")
        ax[row, 0].axvline(-0.5, color="k", ls="--", lw=0.8)
        ax[row, 0].axhline((K + 1) / 2, color="k", lw=0.5)
        ax[row, 0].invert_yaxis()
        ax[row, 0].set_xticks(x, [f"w{k:+d}" if k else "w" for k in x])
        ax[row, 0].set_ylabel("Rang moyen (1 = meilleur)")
        ax[row, 0].set_title(f"Rang hebdomadaire, par {level} ({K} groupes)")
        ax[row, 0].legend()
        xd = list(range(-DAYS_BEFORE + 1, 1))
        for i, (lab, c, ls) in enumerate((("rang du jour — entrants", "tab:green", ":"), ("rang du jour — témoins", "tab:gray", ":"),
                                          ("rang 5 j — entrants", "tab:green", "-"), ("rang 5 j — témoins", "tab:gray", "-"))):
            ax[row, 1].plot(xd, daily.loc[lab], color=c, ls=ls, marker=".", label=lab)
        ax[row, 1].axhline((K + 1) / 2, color="k", lw=0.5)
        ax[row, 1].invert_yaxis()
        ax[row, 1].set_xlabel("Jours de bourse avant la décision (0 = soir de J)")
        ax[row, 1].set_ylabel("Rang moyen (1 = meilleur)")
        ax[row, 1].set_title(f"Rang quotidien avant la décision, par {level}")
        ax[row, 1].legend(fontsize=8)
    for a in ax.flat:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "sector_rank_path.png", dpi=110)
    plt.close(fig)
    print("\n".join(md))


if __name__ == "__main__":
    main()
