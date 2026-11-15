"""Qu'est-ce qui déclenche les hausses captées par le top 1 sélectif (top 0,7, modèle normal + 3 graines) ?

Pour chaque trade (myfiles/myqtm_compare_seed_trades_top7.csv) : jour du signal = séance avant l'entrée ;
- résultats trimestriels : publication EODHD (report_date, avant/après séance) dans les 10 séances avant le signal
  ou le jour du signal, surprise (%) ;
- news : nombre d'articles sur le titre le jour du signal et les 4 séances avant, vs sa moyenne de l'année ;
- hausse déjà faite : rendement du titre sur les 5 séances avant l'entrée.
Comparaison gros gains (≥ 15 %) / autres, et digests Haiku des jours précédant les plus gros gains.

    PYTHONPATH=.:myfiles python myfiles/myqtm_trade_triggers.py
"""
import glob

import numpy as np
import pandas as pd

from src.portefolio_strategy import digest as D
from src.portefolio_strategy import news as N
from src.portefolio_strategy.myqtm import features as F

BIG = 15.0
WIN = 10


def main():
    t = pd.read_csv("myfiles/myqtm_compare_seed_trades_top7.csv", parse_dates=["entry"])
    allf = pd.read_parquet(F.FEATURES, columns=["date", "symbol", "o", "c"])
    opens = allf.pivot(index="date", columns="symbol", values="o").sort_index().ffill()
    days = opens.index
    pos = {d: i for i, d in enumerate(days)}
    t["i_entry"] = t["entry"].map(pos)
    t["signal"] = days[t["i_entry"] - 1]
    t["pre5_pct"] = [100 * (opens.at[days[i], s] / opens.at[days[i - 5], s] - 1) for i, s in zip(t["i_entry"], t["symbol"])]
    # résultats
    e = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob("data/eodhd/earnings/202[3-6]-*.parquet"))])
    e = e[e["code"].str.endswith(".US")].assign(symbol=lambda x: x["code"].str[:-3])
    e["report_date"] = pd.to_datetime(e["report_date"])
    e["percent"] = pd.to_numeric(e["percent"], errors="coerce")
    e = e[e["symbol"].isin(t["symbol"].unique())].drop_duplicates(["symbol", "report_date"])
    # séance où le marché réagit : avant ouverture → le jour même ; après clôture → séance suivante
    e["react"] = [days[min(days.searchsorted(d) + (b == "AfterMarket"), len(days) - 1)]
                  for d, b in zip(e["report_date"], e["before_after_market"])]
    rows = []
    for r in t.itertuples():
        x = e[(e["symbol"] == r.symbol)]
        lag = [(pos[r.signal] - pos[d]) for d in x["react"] if d in pos]
        k = [j for j, g in enumerate(lag) if 0 <= g <= WIN]
        if k:
            j = k[np.argmin([lag[i] for i in k])]
            rows.append(dict(resultats=True, seances_depuis_resultats=lag[j], surprise_pct=x["percent"].iat[j]))
        else:
            rows.append(dict(resultats=False, seances_depuis_resultats=np.nan, surprise_pct=np.nan))
    t = pd.concat([t, pd.DataFrame(rows, index=t.index)], axis=1)
    # news
    n = N.load_news([2023, 2024, 2025, 2026])
    cnt = n.groupby(["symbol", "day"]).size().rename("n").reset_index()
    cnt["day"] = pd.to_datetime(cnt["day"])
    m = cnt.pivot(index="day", columns="symbol", values="n").reindex(days).fillna(0)
    t["news_5j"] = [m[s].iloc[pos[d] - 4:pos[d] + 1].sum() if s in m else 0 for s, d in zip(t["symbol"], t["signal"])]
    t["news_moy_5j"] = [5 * m[s].loc[str(d.year)].mean() if s in m else np.nan for s, d in zip(t["symbol"], t["signal"])]
    t["news_ratio"] = t["news_5j"] / t["news_moy_5j"].replace(0, np.nan)
    t["gros_gain"] = t["ret_pct"] >= BIG
    t.to_csv("myfiles/myqtm_trade_triggers.csv", index=False)
    pd.set_option("display.width", 220, "display.float_format", "{:.2f}".format, "display.max_colwidth", 400)
    g = t.groupby("gros_gain").agg(trades=("ret_pct", "size"), gain_moy=("ret_pct", "mean"),
                                   part_resultats_10s=("resultats", "mean"),
                                   seances_depuis_res_med=("seances_depuis_resultats", "median"),
                                   surprise_med=("surprise_pct", "median"), hausse_5j_avant_med=("pre5_pct", "median"),
                                   news_ratio_med=("news_ratio", "median"))
    print(f"{len(t)} trades distincts (top 0,7, normal + 3 graines) ; gros gain = ≥ {BIG:.0f} %\n")
    print(g.to_string())
    print("\n=== gain selon la présence de résultats dans les 10 séances avant le signal ===")
    print(t.groupby("resultats")["ret_pct"].agg(["count", "mean", "median", lambda s: (s > 0).mean()]).to_string())
    big = t.sort_values("ret_pct", ascending=False).drop_duplicates(["symbol"]).head(12)
    print("\n=== plus gros gains (un par titre) ===")
    print(big[["symbol", "signal", "ret_pct", "pre5_pct", "resultats", "seances_depuis_resultats", "surprise_pct",
               "news_5j", "news_ratio"]].to_string(index=False))
    dg = D.load_digests()
    print("\n=== digests Haiku (entreprise) des jours précédant les plus gros gains ===")
    for r in big.head(8).itertuples():
        x = dg[(dg["symbol"] == r.symbol) & (dg["day"] <= r.signal) & (dg["day"] > r.signal - pd.Timedelta(days=10))]
        print(f"\n--- {r.symbol} signal {r.signal:%Y-%m-%d}, gain {r.ret_pct:+.1f} % ---")
        for y in x.sort_values("day").itertuples():
            print(f"  {y.day:%Y-%m-%d} ({y.n_articles} art.) : {str(y.company)[:350]}")


if __name__ == "__main__":
    main()
