"""Oracle « secteurs » : chaque semaine (ou chaque mois), détenir les k secteurs qui feront le mieux la période
suivante, en raisonnant uniquement par secteur (chaque secteur = un panier de ses membres, comme un ETF sectoriel).

- rendement d'un secteur = moyenne de ses membres pondérée par la capitalisation de la fin du mois précédent
  (variante « poids égaux » dans le secteur en annexe) ; décision le soir de J, exécution à l'ouverture de J+1 ;
- portefeuille = les k secteurs retenus à poids égaux ; frais 0,1 % par sens sur la part renouvelée ;
- références : tous les secteurs pondérés par la capitalisation (≈ l'indice), et tous les secteurs à poids égaux ;
- oracle parfait (top k de la période suivante), oracle bruité (score corrélé ρ au rendement futur), et deux
  règles réalistes : garder le top k de la période écoulée (momentum) ou le flop k (retour à la moyenne).

Sorties : outputs/portefolio_strategy/sp500/sector_oracle.md, sector_oracle.png.

    python -m src.portefolio_strategy.sector_oracle
"""
import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import norm

from . import config as C
from . import rotation_compare as RC
from . import sp500 as SP
from . import universe as U

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = C.OUT / "sp500"
KS = (1, 2, 3, 4, 5)
RHOS = (0.0, 0.1, 0.2, 0.3, 0.5, 1.0)
SEEDS = 50
PERIODS = {"semaine": 5, "mois": 21}


def monthly_caps(path):
    c = pd.read_parquet(path, columns=["month", "symbol", "cap"])
    c["month"] = pd.PeriodIndex(c["month"], freq="M")
    return c


def sector_returns(d, key, caps, start, h):
    """→ (rendements des secteurs pondérés capi, idem poids égaux, poids capi des secteurs), dates × secteurs,
    décisions tous les h jours de bourse."""
    x = d[d["member"] & (d["date"] >= start)]
    dates = np.sort(x["date"].unique())[::h]
    x = x[x["date"].isin(dates)].assign(g=lambda t: t["symbol"].map(key)).dropna(subset=["g", f"fwd_{h}"])
    keep = x.groupby(["date", "g"]).size().groupby("g").median()
    x = x[x["g"].isin(keep[keep >= RC.MIN_MEMBERS].index)].copy()
    x["month"] = (x["date"].dt.to_period("M") - 1)
    x = x.merge(caps, on=["month", "symbol"], how="left")
    x["w"] = x["cap"].fillna(x.groupby("date")["cap"].transform("median"))
    x["wr"] = x["w"] * x[f"fwd_{h}"]
    g = x.groupby(["date", "g"])
    Rc = (g["wr"].sum() / g["w"].sum()).unstack("g")
    Re = g[f"fwd_{h}"].mean().unstack("g")
    W = g["w"].sum().unstack("g")
    return Rc, Re, W.div(W.sum(axis=1), axis=0)


def run(R, pick, fee=C.FEE, weights=None):
    """pick(i) → secteurs détenus pendant la période i (poids égaux, ou `weights` dates × secteurs)."""
    v, prev = [1.0], None
    for i in range(len(R)):
        r = R.iloc[i]
        if weights is not None:
            w = weights.iloc[i].reindex(r.index).fillna(0)
            ret, churn = float((w * r.fillna(0)).sum()), 0.0
        else:
            gs = list(pick(i))
            ret = float(r[gs].mean())
            churn = 1.0 if prev is None else len(set(gs) - prev) / len(gs)
            prev = set(gs)
        v.append(v[-1] * (1 + ret - 2 * fee * churn))
    return np.array(v)


def stats(v, dates, bench):
    years = max((dates[-1] - dates[0]).days / 365.25, 1e-9)
    r = np.diff(v) / v[:-1]
    per_year = len(r) / years
    peak = np.maximum.accumulate(v)
    c = 100 * ((v[-1] / v[0]) ** (1 / years) - 1)
    return {"CAGR %": c, "écart indice (pts/an)": c - bench, "maxDD %": 100 * ((v - peak) / peak).min(),
            "Sharpe": r.mean() / r.std() * np.sqrt(per_year) if r.std() > 0 else 0.0}


def analyse(R, W):
    R = R.dropna(how="all")
    W = W.reindex(R.index)
    step = (R.index[1] - R.index[0]).days
    dates = list(R.index) + [R.index[-1] + pd.Timedelta(days=step)]
    idx = run(R, None, weights=W)
    ew = run(R, lambda i: list(R.iloc[i].dropna().index))
    bench = stats(idx, dates, 0)["CAGR %"]
    rows = {"tous les secteurs, pondérés capi (≈ indice)": stats(idx, dates, bench),
            "tous les secteurs, poids égaux": stats(ew, dates, bench)}
    for k in KS:
        rows[f"oracle top {k}"] = stats(run(R, lambda i: R.iloc[i].nlargest(k).index), dates, bench)
    for k in (3, 5):
        rows[f"momentum : top {k} de la période écoulée"] = stats(
            run(R.iloc[1:], lambda i: R.iloc[i].nlargest(k).index), dates[1:], bench)
        rows[f"retour à la moyenne : flop {k} de la période écoulée"] = stats(
            run(R.iloc[1:], lambda i: R.iloc[i].nsmallest(k).index), dates[1:], bench)
    z = pd.DataFrame(norm.ppf((R.rank(axis=1) - 0.5).div(R.notna().sum(axis=1), axis=0)), index=R.index, columns=R.columns)
    noisy = {}
    for rho in RHOS:
        ex = []
        for s in range(1 if rho == 1.0 else SEEDS):
            sc = (rho * z + np.sqrt(1 - rho ** 2) * np.random.default_rng(s).standard_normal(z.shape)).where(R.notna())
            ex.append(stats(run(R, lambda i: sc.iloc[i].nlargest(5).index), dates, bench)["écart indice (pts/an)"])
        noisy[rho] = f"{np.mean(ex):+.1f} ({np.percentile(ex, 10):+.0f} à {np.percentile(ex, 90):+.0f})"
    spread = (R.apply(lambda r: r.nlargest(5).mean() - r.mean(), axis=1)).mean() * 100
    return pd.DataFrame(rows).T.round(2), noisy, spread, R.shape[1]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = RC.datasets()
    caps = {"Nasdaq-100 (reconstitué)": monthly_caps(U.CAPS), "S&P 500 (reconstitué)": monthly_caps(SP.CAPS)}
    sp_start = f"{pd.read_parquet(SP.UNIVERSE)['start'].min() + pd.Timedelta(days=40):%Y-%m-%d}"
    cases = [("S&P 500 (reconstitué)", "secteur", sp_start), ("S&P 500 (reconstitué)", "secteur", RC.ROT_START),
             ("Nasdaq-100 (reconstitué)", "secteur", RC.ROT_START), ("S&P 500 (reconstitué)", "groupe d'industries", RC.ROT_START)]
    md = ["# Oracle « secteurs » : détenir les k meilleurs secteurs de la période suivante", "",
          "Chaque secteur est un panier de ses membres pondéré par la capitalisation (comme un ETF sectoriel). Le "
          "portefeuille détient les k secteurs retenus à poids égaux ; frais 0,1 % par sens sur la part renouvelée. "
          "« Écart indice » = CAGR − CAGR de l'ensemble des secteurs pondérés par la capitalisation.", ""]
    curves = {}
    for name, level, start in cases:
        d, maps, _ = data[name]
        for per, h in PERIODS.items():
            Rc, Re, W = sector_returns(d, maps[level], caps[name], start, h)
            t, noisy, spread, K = analyse(Rc, W)
            md += [f"## {name}, par {level} ({K} groupes), décision chaque {per}, depuis {start[:7]}", "",
                   f"Écart moyen entre le top 5 et la moyenne des secteurs : **{spread:.2f} pts par {per}**.", "",
                   t.to_markdown(), "", f"Top 5 avec un score corrélé ρ au rendement futur (écart indice, pts/an ; "
                   f"moyenne et 10e–90e centiles sur {SEEDS} tirages) :", "",
                   pd.Series(noisy, name="écart indice").rename_axis("ρ").to_frame().T.to_markdown(index=False), ""]
            if per == "semaine" and level == "secteur" and start == RC.ROT_START:
                curves[name] = t
            if level == "secteur" and start == sp_start:
                te, _, _, _ = analyse(Re, W)
                md += [f"Variante poids égaux à l'intérieur des secteurs (même période) :", "",
                       te.loc[[i for i in te.index if i.startswith(("oracle", "tous"))]].to_markdown(), ""]
    (OUT / "sector_oracle.md").write_text("\n".join(md))
    fig, ax = plt.subplots(figsize=(11, 6))
    for (name, t), c in zip(curves.items(), ("tab:blue", "tab:red")):
        ax.plot(KS, [t.loc[f"oracle top {k}", "écart indice (pts/an)"] for k in KS], marker="o", color=c, label=name)
    ax.set_yscale("log")
    ax.set_xticks(KS)
    ax.set_xlabel("Nombre de secteurs détenus (les k meilleurs de la semaine suivante)")
    ax.set_ylabel("Écart de CAGR avec l'indice (pts/an, échelle log)")
    ax.set_title(f"Oracle « secteurs », décision chaque semaine, depuis {RC.ROT_START[:7]}, net de frais")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "sector_oracle.png", dpi=110)
    plt.close(fig)
    print("\n".join(md))


if __name__ == "__main__":
    main()
