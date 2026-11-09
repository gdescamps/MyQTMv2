"""Balayage fin du seuil du garde-fou sentiment MM20 sur PUST+LQQ x1.7 (base NFCI+CPI).

Seuil de −0.25 à −2.0 (pas 0.125), plafond PUST seul (cap=1) et cash (cap=0).
Métriques sur EVAL_START → fin + rendement par épisode (tarifs 2025, printemps 2026, automne 2026).
    python -m myfiles.pust17_sentiment_thr_sweep [--fee 0.001]   # frais par côté (défaut 0.5 %)
"""
import argparse

import numpy as np
import pandas as pd

from myfiles.pust17_sentiment_guard import EVAL_START
import myfiles.pust17_tweet_overlay as ov
from myfiles.pust17_tweet_overlay import metrics, simulate
from src.risk_off_strategy.data import load_funding, load_macro, load_ohlc
from src.risk_off_strategy.strategy import compute_allocation

N = 20
THRS = np.round(np.arange(-0.25, -2.01, -0.125), 3)
EPIS = {"tarifs 25": ("2025-02-19", "2025-06-30"), "print. 26": ("2026-02-15", "2026-06-30"),
        "aut. 26": ("2026-09-01", "2026-11-07")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fee", type=float, default=None, help="frais par côté (achat = vente), ex. 0.001")
    a = ap.parse_args()
    if a.fee is not None:
        ov.BUY_FEE = ov.SELL_FEE = a.fee
    tag = "" if a.fee is None else f"_fee{a.fee * 1000:g}pm"
    print(f"Frais : achat {ov.BUY_FEE:.2%} + vente {ov.SELL_FEE:.2%}")
    ohlc = load_ohlc("QQQ", "2000-01-01", None)
    price = ohlc["close"]
    idx = price.index
    nfci, cpi = load_macro(idx)
    fund = np.asarray(load_funding(idx), float)
    ret = price.pct_change().fillna(0).values
    alloc = compute_allocation(price.values, nfci, cpi, high=ohlc["high"].values, low=ohlc["low"].values,
                               open_=ohlc["open"].values)
    s = pd.read_parquet("data/text/cnbc/sentiment/sentiment.parquet").set_index("date").sort_index()["score"]
    mm = s.rolling(N, min_periods=N // 2).mean()
    mm = mm.reindex(idx.union(mm.index)).ffill().reindex(idx).shift(1).fillna(0).values
    ev = idx >= pd.Timestamp(EVAL_START)
    E = np.clip(2 * alloc, 0, 1.7)

    def row(name, cap, on, capv):
        eq, fees = simulate(ret, alloc, fund, cap)
        m = metrics(eq, fees, idx, EVAL_START)
        ep = {k: 100 * (eq[(idx >= a) & (idx <= b)][-1] / eq[(idx >= a) & (idx <= b)][0] - 1) for k, (a, b) in EPIS.items()}
        act = on & ev
        binding = act & (E > capv + 1e-9)    # jours où le plafond change réellement l'expo
        blk = np.diff(np.concatenate([[0], act.astype(int)])) == 1
        return {"seuil": name, **m, **{f"{k} %": v for k, v in ep.items()},
                "jours actifs": int(act.sum()), "jours contraignants": int(binding.sum()), "activations": int(blk.sum())}

    for capv in (1.0, 0.0):
        rows = [row("base", None, np.zeros(len(idx), bool), capv)]
        for t in THRS:
            on = mm <= t
            rows.append(row(f"{t:g}", np.where(on, capv, np.nan), on, capv))
        res = pd.DataFrame(rows).round(2)
        pd.set_option("display.width", 250)
        print(f"\n=== MM{N} ≤ seuil → plafond {capv:g} ({'PUST seul' if capv else 'cash'}) — {EVAL_START} → {idx[-1].date()}")
        print(res.to_string(index=False))
        res.to_csv(f"myfiles/pust17_sentiment_thr_sweep_cap{int(capv)}{tag}.csv", index=False)


if __name__ == "__main__":
    main()
