"""Garde-fou « sentiment fondamental » (MM des notes de fiches CNBC) sur PUST+LQQ x1.7.

Note de chaque fiche : data/text/cnbc/sentiment/sentiment.parquet (cnbc_fiche_sentiment.py,
Haiku 5.5, faits économiques tangibles seulement, −5…+5). MM sur N fiches notées.
Disponibilité : la fiche J (article CNBC de clôture, modifié après le close) n'est utilisée qu'au
close J+1 → décalage d'1 séance, puis exec_lag=1 de simulate (exécution J+2). Ffill sur les trous.

Garde-fou : MM ≤ SEUIL → plafond d'expo CAP (0 = cash, 1 = PUST seul) forcé hors bande.
Variantes : base déployée (NFCI+CPI) / sans le garde-fou CPI / base ou sans-CPI + sentiment.
Évaluation : dès que la MM la plus longue existe (2022-04) → fin. ATTENTION : 1 seul marché baissier
(2022), n d'épisodes ≈ 3, et Haiku connaît toute la période (cutoff 06/2026) → in-sample pur.
"""
import itertools

import numpy as np
import pandas as pd

from myfiles.pust17_tweet_overlay import metrics, simulate
from src.risk_off_strategy.data import load_funding, load_macro, load_ohlc
from src.risk_off_strategy.strategy import compute_allocation

MAS = [20, 60, 120]
SEUILS = [-0.5, -0.75, -1.0, -1.25, -1.5]
CAPS = [1.0, 0.0]
EVAL_START = "2022-04-01"


def main():
    ohlc = load_ohlc("QQQ", "2000-01-01", None)
    price = ohlc["close"]
    idx = price.index
    o, h, l = ohlc["open"].values, ohlc["high"].values, ohlc["low"].values
    nfci, cpi = load_macro(idx)
    fund = np.asarray(load_funding(idx), float)
    ret = price.pct_change().fillna(0).values
    allocs = {
        "base (NFCI+CPI)": compute_allocation(price.values, nfci, cpi, high=h, low=l, open_=o),
        "sans CPI": compute_allocation(price.values, nfci, None, high=h, low=l, open_=o),
        "sans macro": compute_allocation(price.values, None, None, high=h, low=l, open_=o),
    }
    s = pd.read_parquet("data/text/cnbc/sentiment/sentiment.parquet").set_index("date").sort_index()["score"]

    rows = []

    def add(name, al, cap=None, ndays=0):
        eq, fees = simulate(ret, al, fund, cap)
        m = metrics(eq, fees, idx, EVAL_START)
        sub = {f"{y} %": metrics(eq, fees, idx, f"{y}-01-01" if y > 2022 else EVAL_START, f"{y}-12-31")["%/an"]
               for y in (2022, 2025, 2026)}
        rows.append({"variante": name, **m, **sub, "jours coupés": ndays})

    for k, al in allocs.items():
        add(k, al)
    cpi_on = np.nan_to_num(np.asarray(cpi, float), nan=0) > 7.0
    for n, thr, capv in itertools.product(MAS, SEUILS, CAPS):
        mm = s.rolling(n, min_periods=n // 2).mean()
        mm = mm.reindex(idx.union(mm.index)).ffill().reindex(idx).shift(1).values  # dispo au close J+1
        on = np.nan_to_num(mm, nan=0) <= thr
        cap = np.where(on, capv, np.nan)
        ev = idx >= pd.Timestamp(EVAL_START)
        for k in ("base (NFCI+CPI)", "sans CPI"):
            add(f"{k} + MM{n}≤{thr} cap={capv:g}", allocs[k], cap, int((on & ev).sum()))
    res = pd.DataFrame(rows).round(2)
    pd.set_option("display.width", 220)
    print(f"Évaluation {EVAL_START} → {idx[-1].date()} ; CPI>7 % actif {int((cpi_on & (idx >= EVAL_START)).sum())} séances\n")
    print(res.iloc[:3].to_string(index=False))
    for k in ("base (NFCI+CPI)", "sans CPI"):
        r = res[res.variante.str.startswith(k + " +")]
        print(f"\n{k} + sentiment — top 10 Calmar :")
        print(r.sort_values("Calmar", ascending=False).head(10).to_string(index=False))
    res.to_csv("myfiles/pust17_sentiment_guard.csv", index=False)


if __name__ == "__main__":
    main()
