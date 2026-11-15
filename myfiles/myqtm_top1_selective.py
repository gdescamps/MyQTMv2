"""Top 1 du jour rendu plus sélectif : le signal d'entrée (titre n° 1 du jour en P(haussier), qui n'y était pas
la veille) n'est pris que si un score dépasse un seuil. Portefeuille lots dynamiques 1/n (tous les trades retenus),
sortie EMA 3 < EMA 10, frais 0,05 % par sens, 2024 → fin (myqtm_topk_dynamic.simulate).

Scores (tous connus à la clôture du jour du signal) :
- p_bull       : P(haussier) du titre ;
- bull_m_bear  : P(haussier) − P(baissier) ;
- ecart_2e     : P(haussier) du n° 1 − P(haussier) du n° 2 du jour ;
- rang_trim    : rang (0 → 1) de la P(haussier) du n° 1 du jour parmi les n° 1 des jours précédents du même
                 trimestre (même modèle → même calibration ; causal) ; pas d'entrée les 10 premiers jours.

    PYTHONPATH=.:myfiles python myfiles/myqtm_top1_selective.py [--top=100,120] [--no-ab]
"""
import sys

import numpy as np
import pandas as pd

import myqtm_close_threshold as B
from myqtm_topk_dynamic import simulate
from myqtm_topk_slots import signals, stats
from src.portefolio_strategy.myqtm import features as F
from src.portefolio_strategy.myqtm import train as T
from src.portefolio_strategy.myqtm.trade import ema_exit

FIRST = "2024-01-01"
MIN_DAYS = 10
GRID = {
    "p_bull": (0.0, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9),
    "bull_m_bear": (0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8),
    "ecart_2e": (0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1),
    "rang_trim": (0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9),
}


def day_scores(p):
    """→ par date : symbole n° 1, ses scores."""
    s = p.sort_values(["date", "ybull"], ascending=[True, False])
    g = s.groupby("date")
    d = g.head(1).set_index("date")[["symbol", "ybull", "ybear"]].copy()
    d["ecart_2e"] = d["ybull"] - g["ybull"].nth(1).groupby(s.loc[g["ybull"].nth(1).index, "date"]).first()
    d["p_bull"], d["bull_m_bear"] = d["ybull"], d["ybull"] - d["ybear"]
    q = d.index.to_period("Q")
    rk = []
    for _, x in d.groupby(q)["ybull"]:
        v = x.to_numpy()
        r = [np.nan if i < MIN_DAYS else (v[:i] < v[i]).mean() for i in range(len(v))]
        rk.append(pd.Series(r, index=x.index))
    d["rang_trim"] = pd.concat(rk)
    return d


def main():
    tops = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--top=")), None)
    tops = [int(t) for t in tops.split(",")] if tops else None
    noab = "--no-ab" in sys.argv
    T.PREDICTIONS = T.predictions_path(ab=not noab, first=FIRST, tops=tops)
    p = B.load()
    p = p.merge(pd.read_parquet(T.PREDICTIONS, columns=["date", "symbol", "ybear"]), on=["date", "symbol"])
    p = p[p["date"] >= FIRST].sort_values(["symbol", "date"]).reset_index(drop=True)
    p["x_ema"] = p.groupby("symbol")["c"].transform(ema_exit).astype(bool)
    allf = pd.read_parquet(F.FEATURES, columns=["date", "symbol", "o"])
    opens = allf.pivot(index="date", columns="symbol", values="o").sort_index()
    nd = pd.Series(opens.index[1:].append(pd.DatetimeIndex([pd.NaT])), index=opens.index)
    p["next_date"] = p["date"].map(nd)
    start, end = p["date"].min(), p["date"].max()
    opens = opens.loc[start:end]
    cand = signals(p, 1)
    cand = cand[cand["exit"] <= end].merge(day_scores(p).drop(columns=["symbol", "ybull", "ybear"]),
                                           left_on="signal", right_index=True)
    rows = []
    for score, ths in GRID.items():
        for t in ths:
            c = cand[cand[score] >= t]
            if c.empty:
                continue
            v, n, rot = simulate(c, opens)
            ret = c.apply(lambda r: opens.loc[r.exit, r.symbol] / opens.loc[r.entry, r.symbol] - 1, axis=1)
            rows.append(dict(score=score, seuil=t, trades=len(c), gain_moy_trade_pct=100 * ret.mean(),
                             gagnants_pct=100 * (ret > 0).mean(), pos_moy=n.mean(),
                             jours_investis_pct=100 * (n > 0).mean(), **stats(v)))
    idx = pd.read_parquet(F.INDEX)["level"].loc[start:end]
    rows.append(dict(score="indice de l'univers", **stats(idx)))
    out = pd.DataFrame(rows)
    name = "myfiles/myqtm_top1_selective" + ("_noab" if noab else "") + T._top_sfx(tops)
    out.to_csv(name + ".csv", index=False)
    pd.set_option("display.width", 220, "display.float_format", "{:.2f}".format)
    print(f"{T.PREDICTIONS.name} — top 1 sélectif, lots dynamiques, {start:%Y-%m-%d} → {end:%Y-%m-%d}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
