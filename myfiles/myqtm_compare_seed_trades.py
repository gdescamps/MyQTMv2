"""Compare les trades du top 1 sélectif (rang trimestriel ≥ 1 − TOP) entre graines du sous-échantillonnage sévère
(--tail-sample=0, graines 42 / 7 / 123) et le modèle normal : trades communs / propres, gains, plus gros
contributeurs. Trade identifié par (titre, date d'entrée).

    PYTHONPATH=.:myfiles python myfiles/myqtm_compare_seed_trades.py [--top=0.7]
"""
import sys

import pandas as pd

import myqtm_close_threshold as B
import myqtm_top1_selective as S
from myqtm_topk_slots import signals
from src.portefolio_strategy.myqtm import config as C
from src.portefolio_strategy.myqtm import features as F
from src.portefolio_strategy.myqtm import train as T
from src.portefolio_strategy.myqtm.trade import ema_exit

TOP = float(next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--top=")), 0.7))
RUNS = {"normal": dict(), "graine 42": dict(ts=0.0, seed=42), "graine 7": dict(ts=0.0, seed=7),
        "graine 123": dict(ts=0.0, seed=123)}


def trades(opens, nd, ts=None, seed=42):
    C.TAIL_SAMPLE_SEED = seed
    T.PREDICTIONS = T.predictions_path(first=S.FIRST, ts=ts)
    p = B.load()
    p = p.merge(pd.read_parquet(T.PREDICTIONS, columns=["date", "symbol", "ybear"]), on=["date", "symbol"])
    p = p[p["date"] >= S.FIRST].sort_values(["symbol", "date"]).reset_index(drop=True)
    p["x_ema"] = p.groupby("symbol")["c"].transform(ema_exit).astype(bool)
    p["next_date"] = p["date"].map(nd)
    end = p["date"].max()
    c = signals(p, 1)
    c = c[c["exit"] <= end].merge(S.day_scores(p)[["rang_trim"]], left_on="signal", right_index=True)
    c = c[c["rang_trim"] >= 1 - TOP - 1e-9].copy()
    c["ret_pct"] = [100 * (opens.at[x, s] / opens.at[e, s] - 1) for e, x, s in zip(c["entry"], c["exit"], c["symbol"])]
    c["jours"] = [opens.index.get_loc(x) - opens.index.get_loc(e) for e, x in zip(c["entry"], c["exit"])]
    return c.set_index(["symbol", "entry"])


def main():
    allf = pd.read_parquet(F.FEATURES, columns=["date", "symbol", "o"])
    opens = allf.pivot(index="date", columns="symbol", values="o").sort_index().ffill()
    nd = pd.Series(opens.index[1:].append(pd.DatetimeIndex([pd.NaT])), index=opens.index)
    tr = {k: trades(opens, nd, **v) for k, v in RUNS.items()}
    names = list(tr)
    pd.set_option("display.width", 220, "display.float_format", "{:.2f}".format)
    print(f"top {TOP} (rang trimestriel ≥ {1 - TOP:.1f}), {S.FIRST} → fin\n")
    print("=== résumé par modèle ===")
    print(pd.DataFrame({k: dict(trades=len(v), gain_moy_pct=v["ret_pct"].mean(), gain_med_pct=v["ret_pct"].median(),
                                gagnants_pct=100 * (v["ret_pct"] > 0).mean(), somme_gains_pct=v["ret_pct"].sum(),
                                jours_moy=v["jours"].mean()) for k, v in tr.items()}).T.to_string())
    print("\n=== trades communs (part des trades de la ligne retrouvés dans la colonne) ===")
    print(pd.DataFrame({a: {b: len(tr[a].index.intersection(tr[b].index)) / len(tr[a]) * 100 for b in names}
                        for a in names}).T.round(0).to_string())
    seeds = [k for k in names if k != "normal"]
    common = tr[seeds[0]].index
    for k in seeds[1:]:
        common = common.intersection(tr[k].index)
    print(f"\n=== communs aux 3 graines : {len(common)} trades ===")
    rows = []
    for k in seeds:
        v = tr[k]
        own = v.index.difference(common)
        rows.append(dict(modele=k, communs=len(common), gain_moy_communs=v.loc[common, "ret_pct"].mean(),
                         somme_communs=v.loc[common, "ret_pct"].sum(), propres=len(own),
                         gain_moy_propres=v.loc[own, "ret_pct"].mean(), somme_propres=v.loc[own, "ret_pct"].sum()))
    print(pd.DataFrame(rows).to_string(index=False))
    in_norm = common.intersection(tr["normal"].index)
    print(f"dont présents aussi dans le modèle normal : {len(in_norm)}")
    allidx = sorted(set().union(*[set(v.index) for v in tr.values()]))
    tab = pd.DataFrame(index=pd.MultiIndex.from_tuples(allidx, names=["symbol", "entry"]))
    for k, v in tr.items():
        tab[k] = v["ret_pct"]
        tab.loc[v.index, "exit"] = v["exit"]
    tab["ret_pct"] = tab[names].bfill(axis=1).iloc[:, 0]
    tab["n_graines"] = tab[seeds].notna().sum(axis=1)
    tab["normal_aussi"] = tab["normal"].notna()
    out = tab.reset_index()[["symbol", "entry", "exit", "ret_pct", "n_graines", "normal_aussi"]]
    out.to_csv(f"myfiles/myqtm_compare_seed_trades_top{round(TOP * 10)}.csv", index=False)
    print("\n=== 15 plus gros gains (tous modèles) ===")
    print(out.sort_values("ret_pct", ascending=False).head(15).to_string(index=False))
    print("\n=== 10 plus grosses pertes ===")
    print(out.sort_values("ret_pct").head(10).to_string(index=False))
    print("\n=== gain moyen selon le nombre de graines qui prennent le trade ===")
    print(out.groupby("n_graines")["ret_pct"].agg(["count", "mean", "median"]).to_string())


if __name__ == "__main__":
    main()
