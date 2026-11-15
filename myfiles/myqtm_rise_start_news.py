"""Les news de résultats / d'analystes expliquent-elles le DÉBUT de la montée des trades ?

Trades : top 1 du jour du modèle de référence (sous-échantillonnage sévère, graine 42, 164 features), 2024 → fin.
Début de la montée = plus bas de clôture des 20 séances précédant le signal. Fenêtre « début » = du plus bas
−2 séances au plus bas +3 séances ; fenêtre « avant signal » = du plus bas au signal.
Événements repérés par mots-clés dans les TITRES des news du titre (sans look-ahead) : relèvement / dégradation
d'analyste, résultats au-dessus / en dessous des attentes, prévisions relevées, contrat / partenariat ;
+ réaction aux résultats (calendrier EODHD) dans la fenêtre. Comparaison gros gains (≥ 15 %) / autres et
gain selon la présence d'un événement.

    PYTHONPATH=.:myfiles python myfiles/myqtm_rise_start_news.py
"""
import glob
import re

import numpy as np
import pandas as pd

import myqtm_compare_seed_trades as M
from src.portefolio_strategy import news as N
from src.portefolio_strategy.myqtm import features as F

LOOKBACK, BIG = 20, 15.0
RULES = {
    "relevement_analyste": r"\bupgrade|\b(?:raises?|raised|lifts?|boosts?|hikes?|ups)\b.{0,40}\b(?:price target|PT|target)\b"
                           r"|\binitiat\w+ .{0,30}\b(?:buy|outperform|overweight)\b|\btop pick\b",
    "degradation_analyste": r"\bdowngrade|\b(?:cuts?|lowers?|slashes?|trims?)\b.{0,40}\b(?:price target|PT|target)\b",
    "resultats_au_dessus": r"\b(?:beats?|tops?|surpass\w*|exceeds?|crush\w*)\b.{0,40}\b(?:estimates?|expectations|consensus|views?)\b"
                           r"|\brecord (?:revenue|quarter|results)\b",
    "resultats_en_dessous": r"\b(?:miss(?:es|ed)?|falls? short)\b.{0,40}\b(?:estimates?|expectations|consensus|views?)\b",
    "prevision_relevee": r"\b(?:raises?|raised|lifts?|boosts?|hikes?)\b.{0,30}\b(?:guidance|outlook|forecast)\b",
    "contrat": r"\b(?:contract|partnership|partners with|deal with|agreement with|multi-?year deal|gigawatt)\b",
}


def main():
    allf = pd.read_parquet(F.FEATURES, columns=["date", "symbol", "o", "c"])
    opens = allf.pivot(index="date", columns="symbol", values="o").sort_index().ffill()
    close = allf.pivot(index="date", columns="symbol", values="c").sort_index().ffill()
    nd = pd.Series(opens.index[1:].append(pd.DatetimeIndex([pd.NaT])), index=opens.index)
    tr = M.trades(opens, nd, ts=0.0, seed=42)
    tr = tr.reset_index()
    tr = tr.copy()
    days = opens.index
    pos = {d: i for i, d in enumerate(days)}
    starts, rise = [], []
    for s, sig in zip(tr["symbol"], tr["signal"]):
        i = pos[sig]
        w = close[s].iloc[i - LOOKBACK:i + 1]
        lo = w.idxmin()
        starts.append(lo)
        rise.append(100 * (w.iloc[-1] / w.min() - 1))
    tr["debut_montee"], tr["montee_avant_signal_pct"] = starts, rise
    tr["seances_debut_signal"] = [pos[g] - pos[lo] for g, lo in zip(tr["signal"], tr["debut_montee"])]
    n = N.load_news([2023, 2024, 2025, 2026])
    n["day"] = pd.to_datetime(n["day"])
    n = n[n["symbol"].isin(tr["symbol"].unique())]
    for k, p in RULES.items():
        n[k] = n["title"].str.contains(p, flags=re.I, regex=True)
    e = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob("data/eodhd/earnings/202[3-6]-*.parquet"))])
    e = e[e["code"].str.endswith(".US")].assign(symbol=lambda x: x["code"].str[:-3])
    e["react"] = pd.to_datetime(e["report_date"]) + pd.to_timedelta(
        (e["before_after_market"] == "AfterMarket").astype(int), unit="D")
    react = e.groupby("symbol")["react"].apply(lambda x: np.sort(x.unique()))
    out, examples = [], []
    for r in tr.itertuples():
        i0 = pos[r.debut_montee]
        a, b = days[max(i0 - 2, 0)], days[min(i0 + 3, len(days) - 1)]
        x = n[(n["symbol"] == r.symbol) & n["day"].between(a, b)]
        y = n[(n["symbol"] == r.symbol) & n["day"].between(r.debut_montee, r.signal)]
        rr = react.get(r.symbol, np.array([], dtype="datetime64[ns]"))
        res_debut = bool(((rr >= np.datetime64(a)) & (rr <= np.datetime64(b))).any())
        res_avant = bool(((rr >= np.datetime64(r.debut_montee)) & (rr <= np.datetime64(r.signal))).any())
        row = dict(symbol=r.symbol, signal=r.signal, ret_pct=r.ret_pct, montee_avant_signal_pct=r.montee_avant_signal_pct,
                   seances_debut_signal=r.seances_debut_signal, publication_resultats_debut=res_debut,
                   publication_resultats_avant_signal=res_avant, news_debut=len(x))
        for k in RULES:
            row[f"{k}_debut"] = bool(x[k].any())
            row[f"{k}_avant_signal"] = bool(y[k].any())
        row["evenement_debut"] = res_debut or any(row[f"{k}_debut"] for k in RULES)
        row["evenement_avant_signal"] = res_avant or any(row[f"{k}_avant_signal"] for k in RULES)
        out.append(row)
        if r.ret_pct >= BIG:
            examples.append((r.symbol, r.signal, r.ret_pct, r.debut_montee, x.loc[x[list(RULES)].any(axis=1), "title"].tolist()[:3]))
    o = pd.DataFrame(out)
    o["gros_gain"] = o["ret_pct"] >= BIG
    o.to_csv("myfiles/myqtm_rise_start_news.csv", index=False)
    pd.set_option("display.width", 220, "display.float_format", "{:.2f}".format, "display.max_colwidth", 160)
    print(f"{len(o)} trades top 1 ; montée médiane avant le signal {o['montee_avant_signal_pct'].median():.1f} % "
          f"en {o['seances_debut_signal'].median():.0f} séances\n")
    cols = ["publication_resultats_debut"] + [f"{k}_debut" for k in RULES] + ["evenement_debut", "evenement_avant_signal"]
    print("=== part des trades avec l'événement, gros gains / autres ===")
    print(o.groupby("gros_gain")[cols].mean().T.rename(columns={False: "autres", True: "gros gains"}).to_string())
    print("\n=== gain du trade selon un événement au début de la montée ===")
    for c in ["publication_resultats_debut", "relevement_analyste_debut", "evenement_debut", "evenement_avant_signal"]:
        print(c, o.groupby(c)["ret_pct"].agg(["count", "mean", "median"]).round(2).to_dict("index"))
    print("\n=== gros gains : titres repérés au début de la montée ===")
    for s, sig, ret, lo, t in examples:
        print(f"{s:5s} signal {sig:%Y-%m-%d} {ret:+5.1f} %  début {lo:%Y-%m-%d} : {t if t else '—'}")


if __name__ == "__main__":
    main()
