"""Étape 2 : calendrier de résultats comme filtre du top 1 sélectif (modèle de référence ts0, 164 features).

Top k (rang trimestriel ≥ 1 − k), lots dynamiques, sortie EMA 3 < EMA 10 ; filtre sur earn_days_to_next au jour
du signal (jours calendaires avant la prochaine réaction aux résultats, plafonné à 30 = aucune connue) :
aucun filtre / résultats ≤ 20 j / < 30 j / aucun résultat dans les 30 j.

    PYTHONPATH=.:myfiles python myfiles/myqtm_earn_filter.py
"""
import pandas as pd

import myqtm_close_threshold as B
import myqtm_top1_selective as S
from myqtm_topk_dynamic import simulate
from myqtm_topk_slots import signals, stats
from src.portefolio_strategy.myqtm import features as F
from src.portefolio_strategy.myqtm import train as T
from src.portefolio_strategy.myqtm.trade import ema_exit

TOPS = (1.0, 0.7, 0.5, 0.4, 0.3, 0.2, 0.1)
FILTERS = {"aucun filtre": lambda d: d >= 0, "résultats ≤ 20 j": lambda d: d <= 20,
           "résultats < 30 j": lambda d: d < 30, "pas de résultats dans 30 j": lambda d: d >= 30}


def main():
    T.PREDICTIONS = T.predictions_path(first=S.FIRST, ts=0.0)
    p = B.load()
    p = p.merge(pd.read_parquet(T.PREDICTIONS, columns=["date", "symbol", "ybear"]), on=["date", "symbol"])
    p = p[p["date"] >= S.FIRST].sort_values(["symbol", "date"]).reset_index(drop=True)
    p["x_ema"] = p.groupby("symbol")["c"].transform(ema_exit).astype(bool)
    allf = pd.read_parquet(F.FEATURES, columns=["date", "symbol", "o", "earn_days_to_next"])
    opens = allf.pivot(index="date", columns="symbol", values="o").sort_index()
    nd = pd.Series(opens.index[1:].append(pd.DatetimeIndex([pd.NaT])), index=opens.index)
    p["next_date"] = p["date"].map(nd)
    end = p["date"].max()
    opens = opens.loc[p["date"].min():end]
    cand = signals(p, 1)
    cand = cand[cand["exit"] <= end].merge(S.day_scores(p)[["rang_trim"]], left_on="signal", right_index=True)
    cand = cand.merge(allf[["date", "symbol", "earn_days_to_next"]], left_on=["signal", "symbol"],
                      right_on=["date", "symbol"])
    rows = []
    for name, f in FILTERS.items():
        for k in TOPS:
            c = cand[(cand["rang_trim"] >= 1 - k - 1e-9) & f(cand["earn_days_to_next"])]
            if c.empty:
                continue
            v, _, _ = simulate(c, opens)
            ret = [opens.at[x, s] / opens.at[e, s] - 1 for e, x, s in zip(c["entry"], c["exit"], c["symbol"])]
            rows.append(dict(filtre=name, top=k, trades=len(c), gain_moy_pct=100 * pd.Series(ret).mean(),
                             gain_med_pct=100 * pd.Series(ret).median(), **stats(v)))
    out = pd.DataFrame(rows)
    out.to_csv("myfiles/myqtm_earn_filter.csv", index=False)
    pd.set_option("display.width", 220, "display.float_format", "{:.2f}".format)
    for v in ("trades", "gain_med_pct", "rendement_an_pct", "max_drawdown_pct", "sharpe"):
        print(f"\n{v}")
        print(out.pivot(index="top", columns="filtre", values=v)[list(FILTERS)].sort_index(ascending=False).to_string())


if __name__ == "__main__":
    main()
