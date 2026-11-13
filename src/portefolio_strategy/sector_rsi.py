"""Rotations sectorielles : RSI moyen des secteurs AVANT qu'ils prennent la tête de l'univers Nasdaq.

Deux niveaux de regroupement (GICS, au moins MIN_MEMBERS membres) : secteurs, et groupes d'industries (sépare
semi-conducteurs, logiciels, médias, pharmacie…). Chaque semaine (décision le soir de J, tous les STEP jours de
bourse), pour chaque groupe :
- RSI moyen (RSI 14 j des membres, moyenne simple) et RSI relatif = RSI du groupe − RSI moyen de l'univers ;
- rendement de la semaine suivante (ouverture J+1 → J+6) moins celui de l'univers (poids égaux).

Événements :
- « meneur » : le groupe au meilleur rendement relatif de la semaine ;
- « rotation entrante » : meneur alors qu'il était dans la moitié basse des groupes sur le mois précédent ;
- « rotation sortante » : le pire groupe de la semaine alors qu'il était dans la moitié haute sur le mois précédent.
On trace le RSI relatif moyen des 3 mois qui précèdent chaque événement, comparé à l'ensemble des semaines.
Puis la question prédictive : un groupe à RSI relatif bas fait-il mieux la semaine (ou le mois) suivante ?

Sorties : outputs/portefolio_strategy/sectors/sector_rsi.md, sector_rsi_<niveau>.png, events_<niveau>.csv.

    python -m src.portefolio_strategy.sector_rsi
"""
import matplotlib
import numpy as np
import pandas as pd

from . import config as C
from . import eod
from . import features as F
from . import universe as U
from .labels import add_labels

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = C.OUT / "sectors"
START = "2019-07-01"           # RSI connu dès le début des features ; analyse descriptive, sans modèle
STEP = 5
MIN_MEMBERS = 4
LAGS = (63, 42, 21, 10, 5, 0)  # jours de bourse avant la décision
COLS = {"rsi": "rsi_14", "r21": "ret_21", "f5": "fwd_5", "f21": "fwd_21"}


def group_maps():
    u = U.load_universe()
    caps = pd.read_parquet(U.CAPS)
    sector = caps.drop_duplicates("symbol", keep="last").set_index("symbol")["sector"]
    grp = pd.Series({s: ((eod.load_fund(s) or {}).get("General") or {}).get("GicGroup") for s in u["symbol"].unique()})
    return {"secteur": sector, "groupe d'industries": grp}


def panels(m, key):
    """→ {variable: dates × groupes} en écart à l'univers, RSI absolu, nombre de membres."""
    m = m.assign(g=m["symbol"].map(key)).dropna(subset=["g"])
    agg = {k: (v, "mean") for k, v in COLS.items()}
    s = m.groupby(["date", "g"]).agg(n=("symbol", "size"), **agg)
    u = m.groupby("date").agg(**agg)
    keep = s.groupby("g")["n"].median()
    s = s[s.index.get_level_values("g").isin(keep[keep >= MIN_MEMBERS].index)]
    rel = s[list(COLS)] - u.reindex(s.index.get_level_values("date")).to_numpy()
    P = {k: rel[k].unstack("g").sort_index() for k in COLS}
    return P, s["rsi"].unstack("g").sort_index(), s["n"].unstack("g").sort_index()


def events(P, rsi_abs):
    dates = P["rsi"].index
    rows = []
    for i in range(max(LAGS), len(dates), STEP):
        f5, r21 = P["f5"].iloc[i].dropna(), P["r21"].iloc[i]
        if len(f5) < 5:
            continue
        lag = r21.rank(pct=True)
        best, worst = f5.idxmax(), f5.idxmin()
        for g in f5.index:
            rows.append({"date": dates[i], "groupe": g, "f5": f5[g], "f21": P["f21"].iloc[i][g], "r21": r21[g],
                         "meneur": g == best, "dernier": g == worst,
                         "rotation_entrante": g == best and lag[g] <= 0.5, "rotation_sortante": g == worst and lag[g] > 0.5,
                         "rsi_abs": rsi_abs.iloc[i][g], "rsi_rel_rank": P["rsi"].iloc[i].rank(pct=True)[g],
                         **{f"rsi_rel_{k}": P["rsi"].iloc[i - k][g] for k in LAGS},
                         "rsi_rel_après": P["rsi"].iloc[min(i + 5, len(dates) - 1)][g]})
    return pd.DataFrame(rows)


def path_table(ev):
    cols = [f"rsi_rel_{k}" for k in LAGS] + ["rsi_rel_après"]
    sets = {"rotation entrante": ev["rotation_entrante"], "meneur (tous)": ev["meneur"],
            "rotation sortante": ev["rotation_sortante"], "dernier (tous)": ev["dernier"], "toutes les semaines": ev["f5"].notna()}
    rows = {}
    for name, mask in sets.items():
        x = ev.loc[mask, cols]
        rows[name] = {"n": int(mask.sum()), **{c: f"{x[c].mean():+.1f} ± {1.96 * x[c].std() / np.sqrt(len(x)):.1f}" for c in cols},
                      "RSI absolu à J": round(float(ev.loc[mask, "rsi_abs"].mean()), 1)}
    t = pd.DataFrame(rows).T
    return t.rename(columns={f"rsi_rel_{k}": ("J" if k == 0 else f"J−{k}") for k in LAGS} | {"rsi_rel_après": "J+5"})


def predictive(ev):
    """Tercile de RSI relatif à J (parmi les groupes du jour) → rendement relatif suivant et probabilité de mener."""
    e = ev.copy()
    e["tercile"] = pd.cut(e["rsi_rel_rank"], [0, 1 / 3, 2 / 3, 1.0001], labels=["RSI bas", "RSI moyen", "RSI haut"])
    t = e.groupby("tercile", observed=True).agg(n=("f5", "size"), f5=("f5", "mean"), f21=("f21", "mean"),
                                                 meneur=("meneur", "mean"), rot=("rotation_entrante", "mean"),
                                                 dernier=("dernier", "mean"))
    t[["f5", "f21"]] *= 100
    t[["meneur", "rot", "dernier"]] *= 100
    t = t.round(2).rename(columns={"f5": "écart semaine suivante (%)", "f21": "écart mois suivant (%)",
                                   "meneur": "meneur (%)", "rot": "rotation entrante (%)", "dernier": "dernier (%)"})
    ic = {}
    for var, label in (("rsi_rel_0", "RSI relatif"), ("r21", "rendement relatif 1 mois")):
        for h, col in ((5, "f5"), (21, "f21")):
            s = e.dropna(subset=[col]).groupby("date")[[var, col]].apply(lambda g: g[var].corr(g[col], method="spearman")).dropna()
            ic[(label, f"{h} j")] = {"corrélation moyenne": round(float(s.mean()), 3),
                                     "t": round(float(s.mean() / s.std() * np.sqrt(len(s) * STEP / h)), 2)}
    return t, pd.DataFrame(ic).T


def laggard_test(ev):
    """Test sans biais de sélection : parmi les groupes EN RETARD sur le mois écoulé (les seuls qui peuvent faire une
    rotation entrante), un RSI relatif plus bas annonce-t-il plus souvent la tête de la semaine suivante ?"""
    e = ev[ev.groupby("date")["r21"].rank(pct=True) <= 0.5].copy()
    e["moitié"] = pd.qcut(e.groupby("date")["rsi_rel_0"].rank(pct=True), 2, labels=["RSI relatif le plus bas", "RSI relatif le plus haut"])
    t = e.groupby("moitié", observed=True).agg(n=("f5", "size"), meneur=("meneur", "mean"), f5=("f5", "mean"), f21=("f21", "mean"))
    t[["meneur", "f5", "f21"]] *= 100
    return t.round(2).rename(columns={"meneur": "meneur la semaine suivante (%)", "f5": "écart semaine suivante (%)",
                                      "f21": "écart mois suivant (%)"})


def by_group(ev, n):
    g = ev.groupby("groupe")
    t = pd.DataFrame({"membres (médiane)": n.median().round(0), "semaines meneur": g["meneur"].sum(),
                      "rotations entrantes": g["rotation_entrante"].sum(), "RSI moyen (toutes semaines)": g["rsi_abs"].mean().round(1),
                      "RSI avant rotation entrante": ev[ev["rotation_entrante"]].groupby("groupe")["rsi_abs"].mean().round(1),
                      "RSI relatif avant rotation": ev[ev["rotation_entrante"]].groupby("groupe")["rsi_rel_0"].mean().round(1)})
    return t.sort_values("membres (médiane)", ascending=False)


def plot(ev, pred, level, path):
    fig, ax = plt.subplots(1, 2, figsize=(17, 6))
    x = [-k for k in LAGS] + [5]
    cols = [f"rsi_rel_{k}" for k in LAGS] + ["rsi_rel_après"]
    for name, mask, c in (("rotation entrante", ev["rotation_entrante"], "tab:green"), ("meneur (tous)", ev["meneur"], "tab:olive"),
                          ("rotation sortante", ev["rotation_sortante"], "tab:red"), ("toutes les semaines", ev["f5"].notna(), "k")):
        v = ev.loc[mask, cols]
        mu, se = v.mean().to_numpy(), 1.96 * v.std().to_numpy() / np.sqrt(len(v))
        ax[0].plot(x, mu, marker="o", color=c, label=f"{name} (n = {int(mask.sum())})")
        ax[0].fill_between(x, mu - se, mu + se, color=c, alpha=0.12)
    ax[0].axvline(0, color="gray", ls="--", lw=0.8)
    ax[0].axhline(0, color="gray", lw=0.6)
    ax[0].set_xlabel("Jours de bourse par rapport à la décision (0 = soir de J ; la semaine mesurée va de J+1 à J+5)")
    ax[0].set_ylabel("RSI moyen du groupe − RSI moyen de l'univers")
    ax[0].set_title(f"RSI relatif avant les rotations ({level}), intervalle 95 %")
    ax[0].legend()
    w = 0.38
    xs = np.arange(len(pred))
    ax[1].bar(xs - w / 2, pred["écart semaine suivante (%)"], w, color="tab:orange", label="semaine suivante")
    ax[1].bar(xs + w / 2, pred["écart mois suivant (%)"], w, color="tab:blue", label="mois suivant")
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_xticks(xs, pred.index)
    ax[1].set_ylabel("Rendement du groupe − univers (%)")
    ax[1].set_title("Rendement relatif suivant, selon le RSI relatif du groupe à J")
    ax[1].legend()
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = add_labels(pd.read_parquet(F.FEATURES, columns=["date", "symbol", "member", "o", "rsi_14", "ret_21"]), [5, 21])
    m = d[d["member"] & (d["date"] >= START)]
    md = ["# Rotations sectorielles : RSI moyen des groupes avant la rotation", "",
          f"Période {START} → {m['date'].max():%Y-%m-%d}, une décision tous les {STEP} jours de bourse. RSI relatif = RSI 14 j "
          "moyen des membres du groupe − RSI moyen de l'univers. « Rotation entrante » = le groupe mène la semaine suivante "
          "alors qu'il était dans la moitié basse sur le mois écoulé ; « sortante » = il est dernier alors qu'il était dans "
          "la moitié haute. Valeurs : moyenne ± intervalle à 95 %.", ""]
    for level, key in group_maps().items():
        P, rsi_abs, n = panels(m, key)
        ev = events(P, rsi_abs)
        ev.to_csv(OUT / f"events_{'secteurs' if level == 'secteur' else 'groupes'}.csv", index=False)
        pred, ic = predictive(ev)
        plot(ev, pred, level, OUT / f"sector_rsi_{'secteurs' if level == 'secteur' else 'groupes'}.png")
        md += [f"## Par {level} ({P['rsi'].shape[1]} groupes d'au moins {MIN_MEMBERS} membres, {ev['date'].nunique()} semaines)", "",
               "### RSI relatif avant l'événement", "", path_table(ev).to_markdown(), "",
               "### Rendement suivant selon le RSI relatif à J", "", pred.to_markdown(), "", ic.to_markdown(), "",
               "### Parmi les groupes en retard sur le mois écoulé (test sans biais de sélection)", "",
               laggard_test(ev).to_markdown(), "",
               "### Détail par groupe", "", by_group(ev, n).to_markdown(), ""]
    (OUT / "sector_rsi.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
