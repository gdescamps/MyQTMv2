"""Oracle : borne supérieure de ce qu'une sous-sélection de l'univers Nasdaq peut rapporter.

L'oracle connaît le rendement futur réalisé (ouverture J+1 → ouverture J+1+h) et retient à chaque
rééquilibrage les n titres qui feront le mieux pendant la période suivante. Même moteur, mêmes frais et même
période que le backtest (portfolio.simulate) : ce n'est PAS une stratégie, seulement le gain maximum possible.

Oracle bruité : score = ρ · rang du rendement futur + √(1 − ρ²) · bruit. ρ est la corrélation visée entre le
score et le rendement futur (l'« IC » du modèle) : il montre combien de prévision il faut pour battre l'indice.

Poids égaux, nombre de lignes de 5 à 20 (N_TOPS).

Sorties : outputs/portefolio_strategy/oracle/oracle.md, oracle.csv, oracle.png, noisy_oracle.csv.

    python -m src.portefolio_strategy.oracle
"""
import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import norm

from . import benchmark as B
from . import config as C
from . import features as F
from . import portfolio as PF
from .labels import add_labels

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = C.OUT / "oracle"
PERIODS = {"chaque semaine (5 j)": 5, "chaque mois (21 j)": 21}
N_TOPS = (5, 10, 15, 20)       # nombre de lignes, à poids égaux
RHOS = (0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
SEEDS = 10


def load():
    """→ (rendements futurs {h: dates × titres, NaN hors univers}, ouvertures, capitalisations, indice)."""
    df = pd.read_parquet(F.FEATURES, columns=["date", "symbol", "member", "o", "log_cap"])
    lab = add_labels(df, list(PERIODS.values()))
    lab = lab[lab["date"] >= C.FIRST_OOS_DATE]
    dates = np.sort(lab["date"].unique())
    piv = lambda col, d=lab: d.pivot(index="date", columns="symbol", values=col).reindex(dates)  # noqa: E731
    fwd = {h: piv(f"fwd_{h}", lab[lab["member"]]).reindex(columns=sorted(lab["symbol"].unique())) for h in PERIODS.values()}
    opens, caps = piv("o")[fwd[5].columns], np.exp(piv("log_cap"))[fwd[5].columns]
    return fwd, opens, caps, B.index_curve(pd.DatetimeIndex(dates))


def noisy(fwd, rho, seed):
    """Score de corrélation de rang ≈ rho avec le rendement futur, jour par jour."""
    z = pd.DataFrame(norm.ppf(fwd.rank(axis=1, pct=True).clip(0.001, 0.999)), index=fwd.index, columns=fwd.columns)
    e = np.random.default_rng(seed).standard_normal(z.shape)
    return rho * z + np.sqrt(1 - rho ** 2) * e


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fwd, opens, _, bench = load()
    mb = PF.metrics(bench)
    years = (bench.index[-1] - bench.index[0]).days / 365.25
    rows, curves = [], {}
    for label, h in PERIODS.items():
        for n in N_TOPS:
            r = PF.simulate(fwd[h], opens, n, h)
            m, g = PF.metrics(r["values"]), PF.metrics(PF.simulate(fwd[h], opens, n, h, fee=0.0)["values"])
            curves[(label, n)] = r["values"]
            rows.append({"renouvellement": label, "lignes": n, "CAGR net %": round(m["cagr_pct"]),
                         "CAGR brut %": round(g["cagr_pct"]), "écart indice (pts/an)": round(m["cagr_pct"] - mb["cagr_pct"]),
                         "valeur finale (base 1)": f"{r['values'].iloc[-1]:.3g}", "maxDD %": round(m["max_drawdown_pct"], 1),
                         "Sharpe": round(m["sharpe"], 1), "rotation (×/an)": round(float(r["turnover"].sum() / years))})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "oracle.csv", index=False)
    nrows = []
    for label, h in PERIODS.items():
        for rho in RHOS:
            scores = [noisy(fwd[h], rho, s) for s in range(1 if rho == 1.0 else SEEDS)]
            for n in N_TOPS:
                ex = [PF.metrics(PF.simulate(s, opens, n, h)["values"])["cagr_pct"] - mb["cagr_pct"] for s in scores]
                nrows.append({"renouvellement": label, "corrélation": rho, "lignes": n, "moyen": round(float(np.mean(ex)), 1),
                              "min": round(float(np.min(ex)), 1), "max": round(float(np.max(ex)), 1),
                              "tirages > indice": f"{sum(e > 0 for e in ex)}/{len(ex)}"})
    ntable = pd.DataFrame(nrows)
    ntable.to_csv(OUT / "noisy_oracle.csv", index=False)

    def pivot(label):
        d = ntable[ntable["renouvellement"] == label]
        cell = d["moyen"].map("{:+.1f}".format) + " (" + d["min"].map("{:+.0f}".format) + " à " + d["max"].map("{:+.0f}".format) + ")"
        return d.assign(cell=cell).pivot(index="corrélation", columns="lignes", values="cell").rename(
            columns=lambda n: f"{n} lignes")

    md = ["# Oracle : gain maximum d'une sous-sélection de l'univers Nasdaq (poids égaux)", "",
          f"Période {bench.index[0]:%Y-%m-%d} → {bench.index[-1]:%Y-%m-%d}, frais {100 * C.FEE:.1f} % par sens. L'oracle "
          "retient à chaque rééquilibrage les n membres de l'univers qui auront le meilleur rendement jusqu'au "
          "rééquilibrage suivant (achat à l'ouverture du lendemain), à poids égaux. Borne supérieure, pas une stratégie.", "",
          f"Indice de l'univers : CAGR {mb['cagr_pct']:.1f} %, valeur finale {bench.iloc[-1]:.2f}, maxDD "
          f"{mb['max_drawdown_pct']:.1f} %.", "", table.to_markdown(index=False), ""]
    for label in PERIODS:
        md += [f"## Oracle bruité, renouvelé {label} : écart de CAGR avec l'indice (pts/an)", "",
               "Le score ne connaît le rendement futur qu'en partie ; la corrélation de rang visée est celle qu'on mesure "
               f"sur le modèle (« IC »). Moyenne (minimum à maximum) sur {SEEDS} tirages du bruit, nets de frais.", "",
               pivot(label).to_markdown(), ""]
    (OUT / "oracle.md").write_text("\n".join(md))
    colors = dict(zip(N_TOPS, ("tab:red", "tab:orange", "tab:green", "tab:purple")))
    fig, ax = plt.subplots(1, 3, figsize=(21, 6.5))
    for (label, n), v in curves.items():
        ax[0].plot(v.index, v, lw=1.3, color=colors[n], ls="-" if PERIODS[label] == 21 else ":",
                   label=f"{n} lignes, {label} (CAGR {PF.metrics(v)['cagr_pct']:.0f} %)")
    ax[0].plot(bench.index, bench, lw=2, color="tab:blue", label=f"Indice de l'univers (CAGR {mb['cagr_pct']:.0f} %)")
    ax[0].set_yscale("log")
    ax[0].set_ylabel("Valeur (base 1, log)")
    ax[0].set_title("Oracle parfait, poids égaux, net de frais")
    ax[0].legend(fontsize=7)
    for a, (label, h) in zip(ax[1:], PERIODS.items()):
        for n in N_TOPS:
            d = ntable[(ntable["renouvellement"] == label) & (ntable["lignes"] == n)]
            x = range(len(d))
            a.plot(x, d["moyen"], marker="o", ms=4, color=colors[n], label=f"{n} lignes")
            a.fill_between(x, d["min"], d["max"], color=colors[n], alpha=0.12)
        a.axhline(0, color="k", lw=0.8)
        a.set_yscale("symlog", linthresh=20)
        a.set_xticks(range(len(RHOS)), [str(r) for r in RHOS])
        a.set_xlabel(f"Corrélation de rang entre le score et le rendement futur à {h} j (1 = oracle parfait)")
        a.set_ylabel("Écart de CAGR avec l'indice (pts/an)")
        a.set_title(f"Oracle bruité, renouvelé {label}")
        a.legend()
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "oracle.png", dpi=110)
    plt.close(fig)
    print("\n".join(md))
    return table, ntable


if __name__ == "__main__":
    main()
