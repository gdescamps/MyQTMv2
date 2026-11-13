"""Statistiques sur le top N_TOP de l'oracle hebdomadaire : qui y est, à quelle fréquence, et pourquoi.

Chaque jour, le « top 20 » est l'ensemble des N_TOP membres de l'univers au meilleur rendement sur les
HORIZON jours suivants (ouverture J+1 → J+1+HORIZON). Au hasard, un titre y serait N_TOP / (nombre de membres)
du temps, soit environ 20 %. On mesure la fréquence de présence par titre, par année, par quintile de
capitalisation et de volatilité, la présence dans le « flop 20 » (les pires), et la persistance d'une
semaine à la suivante.

Sorties : outputs/portefolio_strategy/oracle/top20_frequency.csv, top20_stats.md, top20_stats.png.

    python -m src.portefolio_strategy.oracle_stats
"""
import matplotlib
import numpy as np
import pandas as pd

from . import config as C
from . import features as F
from . import universe as U
from .labels import add_labels

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = C.OUT / "oracle"
HORIZON = 5
N_TOP = 20
MIN_WEEKS = 26                 # présence minimale dans l'univers pour figurer au classement par titre


def load():
    df = pd.read_parquet(F.FEATURES, columns=["date", "symbol", "member", "o", "log_cap", "vol_60", "sector_id"])
    d = add_labels(df, [HORIZON])
    d = d[d["member"] & (d["date"] >= C.FIRST_OOS_DATE) & d[f"fwd_{HORIZON}"].notna()].copy()
    g = d.groupby("date")[f"fwd_{HORIZON}"]
    d["top"] = g.rank(ascending=False, method="first") <= N_TOP
    d["flop"] = g.rank(ascending=True, method="first") <= N_TOP
    d["cap_q"] = d.groupby("date")["log_cap"].transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False)) + 1
    d["vol_q"] = d.groupby("date")["vol_60"].transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False)) + 1
    return d


def by_symbol(d):
    names = U.load_universe().drop_duplicates("symbol").set_index("symbol")["name"]
    g = d.groupby("symbol")
    t = pd.DataFrame({"nom": names.reindex(g.size().index), "semaines dans l'univers": (g.size() / HORIZON).round(0).astype(int),
                      "présence top 20 %": (100 * g["top"].mean()).round(1), "présence flop 20 %": (100 * g["flop"].mean()).round(1),
                      "volatilité 60 j % (moy.)": (100 * g["vol_60"].mean()).round(0),
                      "rang capi (moy.)": d.assign(r=d.groupby("date")["log_cap"].rank(ascending=False)).groupby("symbol")["r"].mean().round(0),
                      "rendement hebdo moyen %": (100 * g[f"fwd_{HORIZON}"].mean()).round(2)})
    t["top − flop (pts)"] = (t["présence top 20 %"] - t["présence flop 20 %"]).round(1)
    return t.sort_values("présence top 20 %", ascending=False)


def persistence(d):
    """Parmi les titres du top 20 d'une semaine, part encore dans le top 20 la semaine suivante (fenêtres disjointes)."""
    p = d.pivot(index="date", columns="symbol", values="top")
    now, nxt = p.iloc[:-HORIZON], p.shift(-HORIZON).iloc[:-HORIZON]
    both = now.notna() & nxt.notna()
    a, b = now.where(both).astype(float), nxt.where(both).astype(float)
    return float((a * b).sum().sum() / a.sum().sum()), float(((1 - a) * b).sum().sum() / (1 - a).sum().sum())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = load()
    base = 100 * d["top"].mean()
    t = by_symbol(d)
    t.to_csv(OUT / "top20_frequency.csv")
    big = t[t["semaines dans l'univers"] >= MIN_WEEKS]
    stay, enter = persistence(d)
    q = lambda col: (100 * d.groupby(col)[["top", "flop"]].mean()).round(1).rename(  # noqa: E731
        columns={"top": "présence top 20 %", "flop": "présence flop 20 %"})
    capq, volq = q("cap_q").rename_axis("quintile de capitalisation (1 = petites)"), q("vol_q").rename_axis("quintile de volatilité (1 = calmes)")
    year = d.assign(year=d["date"].dt.year).groupby(["year", "symbol"])["top"].mean().unstack()
    yy = float(np.nanmean([year.loc[y].corr(year.loc[y + 1]) for y in year.index[:-1]]))
    freq = big["présence top 20 %"]
    r_vol = float(big["présence top 20 %"].corr(big["volatilité 60 j % (moy.)"], method="spearman"))
    r_flop = float(big["présence top 20 %"].corr(big["présence flop 20 %"], method="spearman"))
    r_net = float(big["top − flop (pts)"].corr(big["rendement hebdo moyen %"], method="spearman"))
    cols = ["nom", "semaines dans l'univers", "présence top 20 %", "présence flop 20 %", "top − flop (pts)",
            "volatilité 60 j % (moy.)", "rang capi (moy.)"]
    md = [f"# Top {N_TOP} de l'oracle hebdomadaire : fréquence de présence", "",
          f"Période {d['date'].min():%Y-%m-%d} → {d['date'].max():%Y-%m-%d}, {d['date'].nunique()} jours, "
          f"{d['symbol'].nunique()} titres passés par l'univers. Au hasard, un titre serait dans le top {N_TOP} "
          f"**{base:.1f} %** du temps.", "",
          f"- Fréquence par titre (au moins {MIN_WEEKS} semaines dans l'univers, {len(big)} titres) : de {freq.min():.1f} % à "
          f"{freq.max():.1f} %, médiane {freq.median():.1f} %, 80 % des titres entre {freq.quantile(.1):.1f} % et {freq.quantile(.9):.1f} %.",
          f"- Corrélation de rang entre présence dans le top {N_TOP} et volatilité : {r_vol:+.2f} ; entre présence dans le top "
          f"et dans le flop : {r_flop:+.2f}.",
          f"- Persistance : un titre du top {N_TOP} y est encore la semaine suivante {100 * stay:.1f} % du temps ; un titre "
          f"hors du top y entre {100 * enter:.1f} % du temps.",
          f"- Stabilité d'une année sur l'autre de la fréquence par titre : corrélation moyenne {yy:+.2f}.",
          f"- « Top − flop » (présence nette) contre rendement hebdomadaire moyen du titre : corrélation de rang {r_net:+.2f}.", "",
          "## Les 15 titres les plus souvent dans le top 20", "", big.head(15)[cols].to_markdown(), "",
          "## Les 15 titres les moins souvent dans le top 20", "", big.tail(15)[cols].to_markdown(), "",
          "## Les 15 meilleures présences nettes (top − flop)", "",
          big.sort_values("top − flop (pts)", ascending=False).head(15)[cols].to_markdown(), "",
          "## Par quintile de capitalisation et de volatilité", "", capq.to_markdown(), "", volq.to_markdown(), ""]
    (OUT / "top20_stats.md").write_text("\n".join(md))
    fig, ax = plt.subplots(1, 3, figsize=(20, 6))
    ax[0].hist(freq, bins=25, color="tab:orange", alpha=0.8)
    ax[0].axvline(base, color="k", ls="--", label=f"hasard ({base:.1f} %)")
    ax[0].set_xlabel(f"Présence dans le top {N_TOP} de l'oracle (% des jours)")
    ax[0].set_ylabel("Nombre de titres")
    ax[0].set_title("Répartition de la fréquence de présence par titre")
    ax[0].legend()
    sc = ax[1].scatter(big["présence flop 20 %"], big["présence top 20 %"], c=big["volatilité 60 j % (moy.)"], cmap="viridis", s=28)
    for s_, r in pd.concat([big.head(8), big.sort_values("top − flop (pts)").head(3), big.sort_values("top − flop (pts)").tail(5)]).iterrows():
        ax[1].annotate(s_, (r["présence flop 20 %"], r["présence top 20 %"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    lim = [0, max(big["présence flop 20 %"].max(), freq.max()) + 2]
    ax[1].plot(lim, lim, color="k", lw=0.8, ls="--")
    ax[1].set_xlabel(f"Présence dans le flop {N_TOP} (%)")
    ax[1].set_ylabel(f"Présence dans le top {N_TOP} (%)")
    ax[1].set_title("Top contre flop : couleur = volatilité moyenne 60 j (%)")
    fig.colorbar(sc, ax=ax[1])
    w = 0.38
    ax[2].bar(np.arange(5) - w / 2, volq["présence top 20 %"], w, color="tab:orange", label="top 20, par quintile de volatilité")
    ax[2].bar(np.arange(5) + w / 2, capq["présence top 20 %"], w, color="tab:blue", label="top 20, par quintile de capitalisation")
    ax[2].axhline(base, color="k", ls="--", lw=0.8)
    ax[2].set_xticks(range(5), ["Q1 (bas)", "Q2", "Q3", "Q4", "Q5 (haut)"])
    ax[2].set_ylabel(f"Présence dans le top {N_TOP} (%)")
    ax[2].set_title("Présence selon la volatilité et la capitalisation")
    ax[2].legend()
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "top20_stats.png", dpi=110)
    plt.close(fig)
    print("\n".join(md))
    return t


if __name__ == "__main__":
    main()
