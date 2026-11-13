"""Règles expertes (sans apprentissage) pour choisir N_TOP lignes à poids égaux dans l'univers Nasdaq dynamique.

Point de départ : le classement du Nasdaq lui-même, c'est-à-dire la capitalisation. Puis des règles
techniques classiques (moyennes mobiles, momentum), seules ou combinées à ce classement. Aucun paramètre n'est
ajusté sur les données : périodes usuelles (50 / 200 jours, 6 et 12 mois). Même moteur, mêmes frais et même
période que le backtest des modèles (portfolio.simulate) ; tout est connu à la clôture de J, exécuté à
l'ouverture de J+1.

Règle avec filtre : s'il reste moins de N_TOP titres éligibles, chaque ligne garde 1/N_TOP et le reste est en
liquidités.

Sorties : outputs/portefolio_strategy/rules/rules.md, rules.csv, rules.png, equity.csv.

    python -m src.portefolio_strategy.rules
"""
import matplotlib
import numpy as np
import pandas as pd

from . import benchmark as B
from . import config as C
from . import features as F
from . import portfolio as PF

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = C.OUT / "rules"
N_TOP = 20
PERIODS = {"semaine": 5, "mois": 21}
COLS = ["log_cap", "gap_sma_50", "gap_sma_200", "mom_12_1", "ret_126", "ret_21", "vol_60", "dist_high_252"]


def rule_scores(x):
    """x : {variable: dates × titres, NaN hors univers} → {nom de la règle: (score, description)}.
    Le score le plus élevé est retenu en premier ; NaN = titre écarté par le filtre."""
    cap, g50, g200 = x["log_cap"], x["gap_sma_50"], x["gap_sma_200"]
    above200 = g200 > 0                      # cours au-dessus de sa moyenne mobile 200 jours
    golden = g200 > g50                      # moyenne 50 j au-dessus de la moyenne 200 j (c/m200 > c/m50)
    top40 = cap.rank(axis=1, ascending=False) <= 2 * N_TOP
    return {
        "capitalisation": (cap, "les 20 plus grosses capitalisations (classement du Nasdaq)"),
        "capi + cours > MM200": (cap.where(above200), "les plus grosses parmi celles au-dessus de leur moyenne 200 j"),
        "capi + MM50 > MM200": (cap.where(golden), "les plus grosses parmi celles dont la moyenne 50 j dépasse la 200 j"),
        "momentum 12-1 mois": (x["mom_12_1"], "meilleure hausse sur 12 mois hors dernier mois"),
        "momentum 6 mois": (x["ret_126"], "meilleure hausse sur 6 mois"),
        "écart à la MM200": (g200, "cours le plus au-dessus de sa moyenne 200 j"),
        "momentum 12-1 + cours > MM200": (x["mom_12_1"].where(above200), "momentum 12-1 parmi les titres au-dessus de leur MM200"),
        "momentum 6 mois / volatilité": (x["ret_126"] / x["vol_60"], "hausse sur 6 mois rapportée à la volatilité 60 j"),
        "proche du plus haut 52 sem.": (x["dist_high_252"], "cours le plus proche de son plus haut sur un an"),
        "40 plus grosses → momentum 6 mois": (x["ret_126"].where(top40), "momentum 6 mois parmi les 40 plus grosses capitalisations"),
        "40 plus grosses → cours > MM200, momentum": (x["ret_126"].where(top40 & above200),
                                                     "idem, en écartant les titres sous leur MM200"),
        "repli 1 mois (contrarian)": (-x["ret_21"], "plus forte baisse sur 1 mois (retour à la moyenne)"),
    }


def load():
    df = pd.read_parquet(F.FEATURES, columns=["date", "symbol", "member", "o"] + COLS)
    df = df[df["date"] >= C.FIRST_OOS_DATE]
    dates, syms = np.sort(df["date"].unique()), sorted(df["symbol"].unique())
    m = df[df["member"]]
    x = {c: m.pivot(index="date", columns="symbol", values=c).reindex(index=dates, columns=syms) for c in COLS}
    opens = df.pivot(index="date", columns="symbol", values="o").reindex(index=dates, columns=syms)
    return x, opens, B.index_curve(pd.DatetimeIndex(dates))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    x, opens, bench = load()
    member = x["log_cap"].notna()
    ew = PF.simulate(member.astype(float).where(member), opens, None, 21)["values"]
    mb = PF.metrics(bench)
    rows, curves = [], {}
    for name, (score, desc) in rule_scores(x).items():
        row = {"règle": name}
        for label, every in PERIODS.items():
            v, to = PF.simulate_offsets(score, opens, N_TOP, every, cash_if_short=True)
            m, a = PF.metrics(v), PF.active(v, bench)
            curves[(name, label)] = v
            row.update({f"CAGR net % ({label})": round(m["cagr_pct"], 1), f"écart indice ({label})": round(a["excess_cagr_pct"], 1),
                        f"maxDD % ({label})": round(m["max_drawdown_pct"], 1), f"Sharpe ({label})": round(m["sharpe"], 2),
                        f"rotation ×/an ({label})": round(to, 1)})
        row["années > indice (mois)"] = PF.active(curves[(name, "mois")], bench)["years_beating_index"]
        row["description"] = desc
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "rules.csv", index=False)
    pd.DataFrame({**{f"{n} ({p})": v for (n, p), v in curves.items()}, "univers équipondéré": ew, "indice": bench}).to_csv(OUT / "equity.csv")
    me = PF.metrics(ew)
    md = [f"# Règles expertes : {N_TOP} lignes à poids égaux dans l'univers Nasdaq dynamique", "",
          f"Période {bench.index[0]:%Y-%m-%d} → {bench.index[-1]:%Y-%m-%d}, frais {100 * C.FEE:.1f} % par sens, "
          "renouvellement chaque semaine (5 j) ou chaque mois (21 j). Aucun paramètre ajusté.", "",
          f"Références : indice (pondéré capitalisation) CAGR {mb['cagr_pct']:.1f} %, maxDD {mb['max_drawdown_pct']:.1f} %, "
          f"Sharpe {mb['sharpe']:.2f} ; univers équipondéré CAGR {me['cagr_pct']:.1f} %, maxDD {me['max_drawdown_pct']:.1f} %, "
          f"Sharpe {me['sharpe']:.2f}.", "", table.drop(columns="description").to_markdown(index=False), "",
          "\n".join(f"- **{r['règle']}** : {r['description']}" for r in rows), ""]
    (OUT / "rules.md").write_text("\n".join(md))
    order = table.sort_values("CAGR net % (mois)", ascending=False)["règle"].tolist()
    fig, ax = plt.subplots(1, 2, figsize=(19, 7.5))
    cmap = plt.get_cmap("tab20")
    for i, name in enumerate(order):
        v = curves[(name, "mois")]
        ax[0].plot(v.index, v, lw=1.2, color=cmap(i), label=f"{name} ({PF.metrics(v)['cagr_pct']:.1f} %)")
    ax[0].plot(bench.index, bench, lw=2.4, color="k", label=f"Indice ({mb['cagr_pct']:.1f} %)")
    ax[0].plot(ew.index, ew, lw=2, color="gray", ls="--", label=f"Univers équipondéré ({me['cagr_pct']:.1f} %)")
    ax[0].set_yscale("log")
    ax[0].set_ylabel("Valeur (base 1, log)")
    ax[0].set_title(f"{N_TOP} lignes à poids égaux, renouvelées chaque mois, nettes de frais (CAGR entre parenthèses)")
    ax[0].legend(fontsize=8)
    y = np.arange(len(order))
    t = table.set_index("règle").loc[order]
    ax[1].barh(y - 0.2, t["écart indice (mois)"], 0.4, color="tab:blue", label="chaque mois")
    ax[1].barh(y + 0.2, t["écart indice (semaine)"], 0.4, color="tab:orange", label="chaque semaine")
    ax[1].axvline(0, color="k", lw=0.8)
    ax[1].set_yticks(y, order, fontsize=8)
    ax[1].invert_yaxis()
    ax[1].set_xlabel("Écart de CAGR avec l'indice (pts/an), net de frais")
    ax[1].legend()
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "rules.png", dpi=110)
    plt.close(fig)
    print("\n".join(md))
    return table


if __name__ == "__main__":
    main()
