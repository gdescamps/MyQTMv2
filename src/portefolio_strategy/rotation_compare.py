"""Nasdaq-100 reconstitué contre S&P 500 reconstitué : oracle et rotations entre secteurs.

Pour chaque univers (même moteur, mêmes frais, même fenêtre) :
1. oracle « titres » : les 20 meilleurs titres de la semaine / du mois suivant, à poids égaux (borne supérieure) ;
2. composition du top 20 de l'oracle hebdomadaire : combien de secteurs, part du secteur le plus représenté
   (comparée à 20 titres tirés au hasard), fréquence à laquelle ce secteur dominant change ;
3. rotations entre groupes (secteurs GICS, puis groupes d'industries ; au moins MIN_MEMBERS membres) : écart
   entre groupes, changement du groupe meneur d'une semaine à l'autre, persistance, et oracle « secteur » qui
   détient chaque semaine tous les titres du meilleur groupe (frais quand le groupe change), parfait puis bruité ;
4. RSI relatif des groupes avant les rotations (même étude que sector_rsi.py).

Sorties : outputs/portefolio_strategy/sp500/rotation_compare.md, rotation_compare.png.

    python -m src.portefolio_strategy.rotation_compare
"""
import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import norm

from . import config as C
from . import features as F
from . import portfolio as PF
from . import sector_rsi as SR
from . import sp500 as SP
from .labels import add_labels

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = C.OUT / "sp500"
N_TOP = 20
MIN_MEMBERS = SR.MIN_MEMBERS
ORACLE_START = C.FIRST_OOS_DATE         # comme l'oracle Nasdaq
ROT_START = "2019-07-01"                 # fenêtre commune des rotations (début de l'univers Nasdaq)
RHOS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0)
SEEDS = 50


def datasets():
    """→ {nom: (panel avec labels, {niveau: titre → groupe}, niveau de l'indice pondéré capitalisation)}."""
    nd = pd.read_parquet(F.FEATURES, columns=["date", "symbol", "member", "o", "rsi_14", "ret_21"])
    sp = pd.read_parquet(SP.PANEL)
    n = sp[sp["member"]].groupby("date").size()           # jours où seuls quelques fichiers ont une ligne : écartés
    sp = sp[sp["date"].isin(n[n >= 0.8 * n.median()].index)]
    caps = pd.read_parquet(SP.CAPS).drop_duplicates("symbol", keep="last").set_index("symbol")
    return {"Nasdaq-100 (reconstitué)": (add_labels(nd, [5, 21]), SR.group_maps(), pd.read_parquet(F.INDEX)["level"]),
            "S&P 500 (reconstitué)": (add_labels(sp, [5, 21]), {"secteur": caps["sector"].replace("", np.nan),
                                                                 "groupe d'industries": caps["group"].replace("", np.nan)},
                                      pd.read_parquet(SP.INDEX)["level"])}


def cagr(v, dates):
    return 100 * ((v[-1] / v[0]) ** (365.25 / max((dates[-1] - dates[0]).days, 1)) - 1)


def stock_oracle(d, level, start):
    x = d[d["date"] >= start]
    rows = {}
    opens = x.pivot(index="date", columns="symbol", values="o")
    lv = level.reindex(opens.index).ffill()
    mb = PF.metrics(lv / lv.iloc[0])
    rows["indice (capitalisation)"] = {"CAGR %": round(mb["cagr_pct"], 1), "maxDD %": round(mb["max_drawdown_pct"], 1), "Sharpe": round(mb["sharpe"], 2)}
    for h, lab in ((5, "chaque semaine"), (21, "chaque mois")):
        fwd = x[x["member"]].pivot(index="date", columns="symbol", values=f"fwd_{h}").reindex(index=opens.index, columns=opens.columns)
        m = PF.metrics(PF.simulate(fwd, opens, N_TOP, h)["values"])
        rows[f"oracle 20 titres, {lab}"] = {"CAGR %": round(m["cagr_pct"]), "maxDD %": round(m["max_drawdown_pct"], 1), "Sharpe": round(m["sharpe"], 1)}
    return rows


def weekly(d, key, start):
    """Membres aux dates de décision hebdomadaires (une sur 5), avec leur groupe."""
    x = d[d["member"] & (d["date"] >= start)]
    dates = np.sort(x["date"].unique())[::5]
    x = x[x["date"].isin(dates)].assign(g=lambda t: t["symbol"].map(key)).dropna(subset=["g", "fwd_5"])
    return x


def top20_composition(x, seed=0):
    rng = np.random.default_rng(seed)
    rows, prev = [], None
    for dt, g in x.groupby("date"):
        top = g.nlargest(N_TOP, "fwd_5")["g"].value_counts()
        rnd = np.mean([g["g"].sample(N_TOP, random_state=int(rng.integers(1e9))).value_counts().iloc[0] for _ in range(20)])
        lead = top.index[0]
        rows.append({"secteurs distincts": len(top), "part du 1er secteur %": 100 * top.iloc[0] / N_TOP,
                     "au hasard %": 100 * rnd / N_TOP, "change": prev is not None and lead != prev})
        prev = lead
    r = pd.DataFrame(rows)
    return {"secteurs distincts dans le top 20": round(r["secteurs distincts"].mean(), 1),
            "part du secteur le plus représenté (%)": round(r["part du 1er secteur %"].mean(), 1),
            "même chose pour 20 titres au hasard (%)": round(r["au hasard %"].mean(), 1),
            "le secteur dominant change d'une semaine à l'autre (%)": round(100 * r["change"].iloc[1:].mean(), 1)}


def group_returns(x):
    keep = x.groupby(["date", "g"]).size().groupby("g").median()
    x = x[x["g"].isin(keep[keep >= MIN_MEMBERS].index)]
    R = x.groupby(["date", "g"])["fwd_5"].mean().unstack("g")
    R21 = x.groupby(["date", "g"])["fwd_21"].mean().unstack("g")
    ew = x.groupby("date")["fwd_5"].mean()
    return R, R21, ew


def chain(R, choose, fee=C.FEE):
    """choose(i) → liste de groupes détenus la semaine i (poids égaux par groupe) ; frais sur la part renouvelée."""
    v, prev = [1.0], set()
    for i in range(len(R)):
        gs = choose(i)
        r = float(R.iloc[i][gs].mean())
        churn = len(set(gs) - prev) / len(gs) if prev else 1.0
        v.append(v[-1] * (1 + r - 2 * fee * churn))
        prev = set(gs)
    return np.array(v)


def rotation_stats(R, R21, ew, level):
    R = R.dropna(how="all")
    dates = list(R.index) + [R.index[-1] + pd.Timedelta(days=7)]
    K = R.shape[1]
    rel = R.sub(R.mean(axis=1), axis=0)
    lead = R.idxmax(axis=1)
    rk = R.rank(axis=1)
    pers = np.nanmean([rk.iloc[i].corr(rk.iloc[i + 1]) for i in range(len(rk) - 1)])
    rk21 = R21.dropna(how="all").iloc[::4].rank(axis=1)
    pers21 = np.nanmean([rk21.iloc[i].corr(rk21.iloc[i + 1]) for i in range(len(rk21) - 1)])
    ewv = np.r_[1.0, np.cumprod(1 + ew.reindex(R.index).fillna(0).to_numpy())]
    lv = level.reindex(level.index.union(R.index)).ffill().reindex(R.index)
    best = chain(R, lambda i: [R.iloc[i].idxmax()])
    top3 = chain(R, lambda i: list(R.iloc[i].nlargest(3).index))
    return {"groupes (≥ 4 membres)": K,
            "écart-type hebdo entre groupes (pts %)": round(100 * rel.std(axis=1).mean(), 2),
            "écart meilleur − pire groupe par semaine (pts %)": round(100 * (R.max(axis=1) - R.min(axis=1)).mean(), 2),
            "le groupe meneur change d'une semaine à l'autre (%)": round(100 * (lead != lead.shift()).iloc[1:].mean(), 1),
            "… au hasard, il changerait (%)": round(100 * (1 - 1 / K), 1),
            "persistance du classement des groupes, semaine → semaine": round(float(pers), 3),
            "persistance du classement des groupes, mois → mois": round(float(pers21), 3),
            "univers équipondéré, CAGR %": round(cagr(ewv, dates), 1),
            "indice (capitalisation), CAGR %": round(cagr(lv.to_numpy(), list(R.index)), 1),
            "oracle 1 groupe par semaine, CAGR %": round(cagr(best, dates)),
            "oracle 3 groupes par semaine, CAGR %": round(cagr(top3, dates))}


def noisy_sector(R, ew):
    R = R.dropna(how="all")
    dates = list(R.index) + [R.index[-1] + pd.Timedelta(days=7)]
    ewc = cagr(np.r_[1.0, np.cumprod(1 + ew.reindex(R.index).fillna(0).to_numpy())], dates)
    z = pd.DataFrame(norm.ppf(R.rank(axis=1, pct=True).clip(0.01, 0.99) - 0.5 / R.shape[1]), index=R.index, columns=R.columns)
    out = {}
    for rho in RHOS:
        ex = []
        for s in range(1 if rho == 1.0 else SEEDS):
            sc = rho * z + np.sqrt(1 - rho ** 2) * np.random.default_rng(s).standard_normal(z.shape)
            sc = sc.where(R.notna())
            ex.append(cagr(chain(R, lambda i: [sc.iloc[i].idxmax()]), dates) - ewc)
        out[rho] = (float(np.mean(ex)), float(np.percentile(ex, 10)), float(np.percentile(ex, 90)))
    return out


def rsi_tables(d, key, start):
    m = d[d["member"] & (d["date"] >= start)]
    P, rsi_abs, _ = SR.panels(m, key)
    ev = SR.events(P, rsi_abs)
    return SR.path_table(ev), SR.laggard_test(ev)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = datasets()
    sp_start = pd.read_parquet(SP.UNIVERSE)["start"].min() + pd.Timedelta(days=40)
    md = ["# Nasdaq-100 contre S&P 500 (reconstitués) : oracle et rotations entre secteurs", "",
          "Deux univers reconstitués point-in-time avec les données EODHD (voir universe.py et sp500.py). Frais 0,1 % par "
          "sens, poids égaux, décision le soir de J et exécution à l'ouverture de J+1.", "",
          f"## 1. Oracle « titres » (20 lignes), depuis {ORACLE_START}", ""]
    t = {name: stock_oracle(d, lv, ORACLE_START) for name, (d, _, lv) in data.items()}
    md += [pd.DataFrame({k: {f"{r} — {c}": v[r][c] for r in v for c in v[r]} for k, v in t.items()}).to_markdown(), ""]
    noisy, rot, comp = {}, {}, {}
    windows = [(name, ROT_START) for name in data] + [("S&P 500 (reconstitué)", f"{sp_start:%Y-%m-%d}")]
    for name, start in windows:
        d, maps, lv = data[name]
        col = f"{name}, depuis {start[:7]}"
        comp[col] = top20_composition(weekly(d, maps["secteur"], start))
        for level, key in maps.items():
            R, R21, ew = group_returns(weekly(d, key, start))
            rot[(level, col)] = rotation_stats(R, R21, ew, lv)
            if start == ROT_START:
                noisy[(level, name)] = noisy_sector(R, ew)
    md += [f"## 2. Composition du top 20 de l'oracle hebdomadaire (par secteur GICS)", "", pd.DataFrame(comp).to_markdown(), ""]
    for level in ("secteur", "groupe d'industries"):
        md += [f"## 3. Rotations entre groupes — par {level}", "",
               pd.DataFrame({c: v for (lv_, c), v in rot.items() if lv_ == level}).to_markdown(), "",
               f"### Oracle « {level} » bruité, depuis {ROT_START} : écart de CAGR avec l'univers équipondéré (pts/an)", "",
               "Le score du groupe est corrélé à son rendement de la semaine suivante avec la corrélation indiquée ; "
               f"moyenne et 10e–90e centiles sur {SEEDS} tirages.", "",
               pd.DataFrame({name: {rho: f"{a:+.1f} ({b:+.0f} à {c:+.0f})" for rho, (a, b, c) in noisy[(level, name)].items()}
                             for (lv_, name) in noisy if lv_ == level}).rename_axis("corrélation").to_markdown(), ""]
    for name, (d, maps, _) in data.items():
        for level in maps:
            path, lag = rsi_tables(d, maps[level], ROT_START)
            md += [f"## 4. RSI relatif avant les rotations — {name}, par {level}, depuis {ROT_START}", "",
                   path.to_markdown(), "", "Parmi les groupes en retard sur le mois écoulé :", "", lag.to_markdown(), ""]
    (OUT / "rotation_compare.md").write_text("\n".join(md))
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    for (level, name), c in zip(noisy, ("tab:blue", "tab:cyan", "tab:red", "tab:orange")):
        v = noisy[(level, name)]
        x = range(len(RHOS))
        ax[0].plot(x, [v[r][0] for r in RHOS], marker="o", color=c, label=f"{name}, {level}")
        ax[0].fill_between(x, [v[r][1] for r in RHOS], [v[r][2] for r in RHOS], color=c, alpha=0.12)
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set_yscale("symlog", linthresh=10)
    ax[0].set_xticks(range(len(RHOS)), [str(r) for r in RHOS])
    ax[0].set_xlabel("Corrélation entre le score du groupe et son rendement de la semaine suivante")
    ax[0].set_ylabel("Écart de CAGR avec l'univers équipondéré (pts/an)")
    ax[0].set_title(f"Oracle « 1 groupe par semaine », bruité, depuis {ROT_START}")
    ax[0].legend(fontsize=8)
    keys = ["écart-type hebdo entre groupes (pts %)", "écart meilleur − pire groupe par semaine (pts %)"]
    cols = [c for (lv_, c) in rot if lv_ == "secteur" and ROT_START[:7] in c] + [c for (lv_, c) in rot if lv_ == "groupe d'industries" and ROT_START[:7] in c]
    lv_of = ["secteur"] * 2 + ["groupe d'industries"] * 2
    xs = np.arange(len(keys))
    for j, (c, lvl) in enumerate(zip(cols, lv_of)):
        ax[1].bar(xs + (j - 1.5) * 0.2, [rot[(lvl, c)][k] for k in keys], 0.2, label=f"{c.split(',')[0]}, {lvl}")
    ax[1].set_xticks(xs, ["écart-type entre groupes", "meilleur − pire groupe"])
    ax[1].set_ylabel("Points de % par semaine (moyenne)")
    ax[1].set_title(f"Écart de rendement entre groupes, depuis {ROT_START}")
    ax[1].legend(fontsize=8)
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "rotation_compare.png", dpi=110)
    plt.close(fig)
    print("\n".join(md))


if __name__ == "__main__":
    main()
