"""Oracle : borne supérieure de ce qu'une sous-sélection de l'univers Nasdaq peut rapporter.

L'oracle connaît le rendement futur réalisé (ouverture J+1 → ouverture J+1+h) et retient à chaque
rééquilibrage les n titres qui feront le mieux pendant la période suivante. Même moteur, mêmes frais et même
période que le backtest (portfolio.simulate) : ce n'est PAS une stratégie, seulement le gain maximum possible.

Oracle bruité : score = ρ · rang du rendement futur + √(1 − ρ²) · bruit. ρ est la corrélation visée entre le
score et le rendement futur (l'« IC » du modèle) : il montre combien de prévision il faut pour battre l'indice.

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
N_TOP = 10
RHOS = (0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
SEEDS = 5


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
    fwd, opens, caps, bench = load()
    mb = PF.metrics(bench)
    rows, curves = [], {}
    for label, h in PERIODS.items():
        for wt, size in (("poids égaux", None), ("capitalisation", caps)):
            for fee in (C.FEE, 0.0):
                r = PF.simulate(fwd[h], opens, N_TOP, h, fee=fee, size=size)
                m = PF.metrics(r["values"])
                years = (bench.index[-1] - bench.index[0]).days / 365.25
                if fee:
                    curves[f"Oracle {N_TOP} titres, {label}, {wt}"] = r["values"]
                rows.append({"renouvellement": label, "pondération": wt, "frais par sens": f"{100 * fee:.1f} %",
                             "CAGR %": round(m["cagr_pct"], 1), "valeur finale (base 1)": round(float(r["values"].iloc[-1]), 1),
                             "écart indice (pts/an)": round(m["cagr_pct"] - mb["cagr_pct"], 1),
                             "maxDD %": round(m["max_drawdown_pct"], 1), "Sharpe": round(m["sharpe"], 2),
                             "rotation (×/an)": round(float(r["turnover"].sum() / years), 1)})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "oracle.csv", index=False)
    nrows = []
    for rho in RHOS:
        for wt, size in (("poids égaux", None), ("capitalisation", caps)):
            ex = [PF.metrics(PF.simulate(noisy(fwd[21], rho, s), opens, N_TOP, 21, size=size)["values"])["cagr_pct"]
                  - mb["cagr_pct"] for s in range(1 if rho == 1.0 else SEEDS)]
            nrows.append({"corrélation score / rendement futur": rho, "pondération": wt,
                          "écart indice moyen (pts/an)": round(float(np.mean(ex)), 1),
                          "min": round(float(np.min(ex)), 1), "max": round(float(np.max(ex)), 1)})
    ntable = pd.DataFrame(nrows)
    ntable.to_csv(OUT / "noisy_oracle.csv", index=False)
    md = ["# Oracle : gain maximum d'une sous-sélection de l'univers Nasdaq", "",
          f"Période {bench.index[0]:%Y-%m-%d} → {bench.index[-1]:%Y-%m-%d}. L'oracle retient à chaque rééquilibrage les "
          f"{N_TOP} membres de l'univers qui auront le meilleur rendement jusqu'au rééquilibrage suivant (achat à "
          "l'ouverture du lendemain). Borne supérieure, pas une stratégie.", "",
          f"Indice de l'univers : CAGR {mb['cagr_pct']:.1f} %, valeur finale {bench.iloc[-1]:.2f}, maxDD "
          f"{mb['max_drawdown_pct']:.1f} %.", "", table.to_markdown(index=False), "",
          f"## Oracle bruité ({N_TOP} titres, renouvelés chaque mois, frais {100 * C.FEE:.1f} % par sens)", "",
          "Le score ne connaît le rendement futur qu'en partie ; la corrélation de rang visée est celle qu'on mesure "
          f"sur le modèle (« IC »). Moyenne, minimum et maximum sur {SEEDS} tirages du bruit.", "",
          ntable.to_markdown(index=False), ""]
    (OUT / "oracle.md").write_text("\n".join(md))
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    for (name, v), c in zip(curves.items(), ("tab:red", "tab:orange", "tab:purple", "tab:green")):
        ax[0].plot(v.index, v, lw=1.4, color=c, label=f"{name} (CAGR {PF.metrics(v)['cagr_pct']:.0f} %)")
    ax[0].plot(bench.index, bench, lw=2, color="tab:blue", label=f"Indice de l'univers (CAGR {mb['cagr_pct']:.0f} %)")
    ax[0].set_yscale("log")
    ax[0].set_ylabel("Valeur (base 1, log)")
    ax[0].set_title(f"Oracle : les {N_TOP} meilleurs titres de la période suivante, nets de frais")
    ax[0].legend(fontsize=8)
    for wt, c in (("poids égaux", "tab:orange"), ("capitalisation", "tab:green")):
        d = ntable[ntable["pondération"] == wt]
        x = range(len(d))
        ax[1].plot(x, d["écart indice moyen (pts/an)"], marker="o", color=c, label=wt)
        ax[1].fill_between(x, d["min"], d["max"], color=c, alpha=0.2)
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_yscale("symlog", linthresh=10)
    ax[1].set_xticks(range(len(RHOS)), [str(r) for r in RHOS])
    ax[1].set_xlabel("Corrélation de rang entre le score et le rendement futur à 21 j (1 = oracle parfait)")
    ax[1].set_ylabel("Écart de CAGR avec l'indice (pts/an)")
    ax[1].set_title(f"Oracle bruité : {N_TOP} titres renouvelés chaque mois, nets de frais")
    ax[1].legend()
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "oracle.png", dpi=110)
    plt.close(fig)
    print("\n".join(md))
    return table, ntable


if __name__ == "__main__":
    main()
