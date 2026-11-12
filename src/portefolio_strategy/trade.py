"""Moteur de trading — portage de ref/MyQTM/src/trade.py + benchmark.compute_bench.

Même logique : le soir de J, on choisit les positions à ouvrir (titre qui vient de passer en classe
haussière/baissière avec une probabilité >= seuil d'ouverture) et à fermer (probabilité < seuil de
fermeture) ; elles sont exécutées à l'ouverture de J+1. Positions longues et courtes, taille fixée par
`increase` comme dans ref/MyQTM.

Écarts avec ref/MyQTM :
- un seul jeu de seuils par période (plus de couple A/B : il n'y a plus de modèles entrelacés) ;
- univers dynamique : un titre absent des données du jour (sorti de l'univers, radié) est valorisé à
  son dernier cours et sa position est fermée ;
- frais par sens (config.FEE).
"""
import random

import numpy as np
import pandas as pd

from . import config as C

LONG, SHORT = 2, 0


def build_trade_data(preds):
    """preds : (date, symbol, o, ybear, yrange, ybull) → {date: {titre: {...}}} avec index_long / index_short
    (0 le jour où le titre entre dans la classe, puis nombre de jours depuis)."""
    p = preds.sort_values(["symbol", "date"]).copy()
    p["cls"] = p[["ybear", "yrange", "ybull"]].to_numpy().argmax(axis=1)
    data = {}
    for sym, g in p.groupby("symbol", sort=False):
        pc, il, ish = None, -1, -1
        for d, o, yb, yu, c in zip(g["date"], g["o"], g["ybear"], g["ybull"], g["cls"]):
            if pc is not None and c == LONG and pc != LONG:
                il = 0
            elif pc is not None and c == SHORT and pc != SHORT:
                ish = 0
            data.setdefault(f"{d:%Y-%m-%d}", {})[sym] = {"class": int(c), "ybear": float(yb), "ybull": float(yu),
                                                         "open": float(o), "index_long": il, "index_short": ish}
            pc = c
            il += il >= 0
            ish += ish >= 0
    return dict(sorted(data.items()))


def _gain(pos, price):
    g = (price - pos["open_price"]) / pos["open_price"]
    return g if pos["type"] == "long" else -g


def simulate(trade_data, params_for, capital=C.INITIAL_CAPITAL, max_positions=C.MAX_POSITIONS,
             remove_stocks=0, seed=0, fee=C.FEE, start=None, end=None):
    """params_for(date_str) → dict(long_open, long_close, short_open, short_close, increase)."""
    rng = random.Random(seed)
    dates = [d for d in trade_data if (start is None or d >= start) and (end is None or d < end)]
    last_price, positions, history = {}, [], []
    to_open, to_close = [], []
    values, cash_series, counts = [], [], []
    prev = None
    for d in dates:
        item = trade_data[d]
        for t, v in item.items():
            last_price[t] = v["open"]
        prm = params_for(d)
        # 1. exécution à l'ouverture des décisions de la veille
        for pos in to_close:
            pos["close_price"], pos["end"] = last_price[pos["ticker"]], d
            g = _gain(pos, pos["close_price"])
            pos["gain"] = 100 * g
            capital += pos["size"] * (1 + g) * (1 - fee)
            history.append(pos)
        to_close = []
        total = capital + sum(p["size"] * (1 + _gain(p, last_price[p["ticker"]])) for p in positions)
        unit = total / max_positions
        factor = min(max(0.5 * (1.0 - prm["increase"]) * max_positions, 1), max_positions)
        for typ, ticker, prob in to_open:
            if capital > 100.0 and ticker in item:
                size = min(capital, factor * unit)
                capital -= size
                positions.append({"type": typ, "ticker": ticker, "size": size * (1 - fee), "start": d,
                                  "open_price": item[ticker]["open"], "yprob": prob})
        to_open = []
        total = capital + sum(p["size"] * (1 + _gain(p, last_price[p["ticker"]])) for p in positions)
        values.append(total)
        cash_series.append(capital)
        counts.append(len(positions))
        if prev is None:
            prev = d
            continue
        # 2. décisions du soir (exécutées demain)
        allowed = list(item)
        if remove_stocks > 0:
            rng.shuffle(allowed)
            allowed = allowed[:-remove_stocks]
        held = {p["ticker"] for p in positions}
        allowed = set(allowed) | held
        prev_item = trade_data[prev]
        for typ, key, idx, thr in (("long", "ybull", "index_long", prm["long_open"]),
                                   ("short", "ybear", "index_short", prm["short_open"])):
            for ticker, v in sorted(item.items(), key=lambda kv: kv[1][key], reverse=True):
                if ticker not in allowed or ticker in held or ticker not in prev_item or v[idx] != 0:
                    continue
                if v[key] >= thr:
                    to_open.append((typ, ticker, v[key]))
                else:
                    break
        keep = []
        for pos in positions:
            v = item.get(pos["ticker"])
            if v is None or (pos["type"] == "long" and v["ybull"] < prm["long_close"]) or \
               (pos["type"] == "short" and v["ybear"] < prm["short_close"]):
                to_close.append(pos)
            else:
                keep.append(pos)
        positions = keep
        prev = d
    for pos in positions + to_close:                      # positions encore ouvertes : valorisées au dernier cours
        pos["end"], pos["close_price"] = dates[-1], last_price[pos["ticker"]]
        pos["gain"] = 100 * _gain(pos, pos["close_price"])
        pos["open"] = True
    return {"dates": pd.to_datetime(dates), "values": np.array(values), "cash": np.array(cash_series),
            "counts": np.array(counts), "history": history, "open_positions": positions + to_close}


def metrics(dates, values, capital=None):
    v = np.asarray(values, dtype=float)
    capital = v[0] if capital is None else capital
    peak = np.maximum.accumulate(v)
    dd = (v - peak) / peak
    longest = cur = 0
    for x in dd:
        cur = cur + 1 if x < 0 else 0
        longest = max(longest, cur)
    years = max((dates[-1] - dates[0]).days / 365.25, 1e-9)
    r = np.diff(v) / v[:-1]
    s = pd.Series(v, index=dates)
    yearly = s.groupby(s.index.year).agg(lambda x: 100 * (x.iloc[-1] / x.iloc[0] - 1))
    return {"return_pct": float(100 * (v[-1] / capital - 1)), "cagr_pct": float(100 * ((v[-1] / capital) ** (1 / years) - 1)),
            "max_drawdown_pct": float(100 * dd.min()), "longest_drawdown_days": int(longest),
            "ulcer_index": float(np.sqrt(np.mean((100 * dd) ** 2))),
            "sharpe": float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0.0,
            "annual_roi_pct": {int(k): round(float(x), 2) for k, x in yearly.items()},
            "annual_roi_mean": float(yearly.mean()), "annual_roi_std": float(yearly.std(ddof=0))}


def perf_score(res, prm, capital=C.INITIAL_CAPITAL):
    """Fonction objectif de ref/MyQTM (benchmark.run_benchmark → perf), sans les termes A/B."""
    m = metrics(res["dates"], res["values"], capital)
    allpos = res["history"] + res["open_positions"]
    long_share = sum(p["type"] == "long" for p in allpos) / (len(allpos) + 1)
    if m["annual_roi_mean"] <= 5.0 or m["longest_drawdown_days"] <= 5:
        return 0.0
    gauss = float(np.exp(-((long_share - 0.7) ** 2) / (2 * 0.2 ** 2)))
    safety = prm["long_open"] * prm["long_close"] * prm["short_open"] * prm["short_close"] * prm["increase"]
    return safety * gauss * m["return_pct"] ** 3 / ((1 + m["longest_drawdown_days"] / 100)
                                                    * (0.2 + m["annual_roi_std"] / 10)
                                                    * (1 + abs(m["max_drawdown_pct"]) / 10))
