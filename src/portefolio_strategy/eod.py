"""Lecture (seule) de data/eodhd : cours, splits, fondamentaux."""
import gzip
import json
from functools import lru_cache

import numpy as np
import pandas as pd

from . import config as C


def load_fund(code):
    p = C.EODHD / "fundamentals" / f"{code}.json.gz"
    if not p.exists():
        return None
    with gzip.open(p, "rt") as f:
        return json.load(f)


def split_factor(code, dates):
    """Facteur par lequel diviser un cours brut pour l'exprimer en actions d'aujourd'hui (splits postérieurs)."""
    f = np.ones(len(dates))
    p = C.EODHD / "splits" / f"{code}.parquet"
    if not p.exists():
        return f
    s = pd.read_parquet(p)
    for d, r in zip(pd.to_datetime(s["date"]), s["split"]):
        try:
            a, b = (float(x) for x in str(r).split("/"))
        except ValueError:
            continue
        if a > 0 and b > 0:
            f[dates < d] *= a / b
    return f


def load_eod(code, start=C.PRICE_START, end=None):
    """Cours quotidiens. Colonnes ajoutées :
    split_close  cours brut ramené au nombre d'actions actuel (pour la capitalisation) ;
    o/h/l/c      cours ajustés splits + dividendes (rendement total), v = volume ajusté des splits."""
    p = C.EODHD / "eod" / f"{code}.parquet"
    if not p.exists():
        return None
    d = pd.read_parquet(p)
    d["date"] = pd.to_datetime(d["date"])
    d = d[(d["date"] >= start) & (d["close"] > 0)]
    if end is not None:
        d = d[d["date"] <= end]
    d = d.drop_duplicates("date").sort_values("date").set_index("date")
    if d.empty:
        return None
    sf = split_factor(code, d.index)
    d["split_close"] = d["close"] / sf
    k = d["adjusted_close"] / d["close"]
    d["o"], d["h"], d["l"], d["c"] = d["open"] * k, d["high"] * k, d["low"] * k, d["adjusted_close"]
    d.loc[d["o"] <= 0, "o"] = d["c"]
    d["v"] = d["volume"] * sf
    return d


def shares_series(fund):
    """Nombre d'actions (ajusté des splits par EODHD), daté de sa disponibilité = fin de trimestre + délai."""
    q = (fund.get("outstandingShares") or {}).get("quarterly") or {}
    rows = [(v.get("dateFormatted"), v.get("shares")) for v in q.values()]
    rows = [(pd.Timestamp(d) + pd.Timedelta(days=C.SHARES_LAG_DAYS), float(s)) for d, s in rows if d and s]
    if not rows:
        return None
    s = pd.Series(dict(rows)).sort_index()
    return s[s > 0]


@lru_cache(maxsize=1)
def nasdaq_candidates():
    """Actions ordinaires cotées au NASDAQ, actives et radiées (pas de biais du survivant)."""
    fr = []
    for name, delisted in (("symbols_active", False), ("symbols_delisted", True)):
        d = pd.read_parquet(C.EODHD / f"{name}.parquet")
        d = d[(d["Exchange"] == "NASDAQ") & (d["Type"] == "Common Stock")][["Code", "Name"]].copy()
        d["delisted"] = delisted
        fr.append(d)
    d = pd.concat(fr).drop_duplicates("Code")
    ok = [(C.EODHD / "eod" / f"{c}.parquet").exists() and (C.EODHD / "fundamentals" / f"{c}.json.gz").exists()
          for c in d["Code"]]
    return d[ok].reset_index(drop=True)
