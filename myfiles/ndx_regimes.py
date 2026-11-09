"""Zones de tendance du NDX (daily) par zigzag sur les clôtures.

Deux niveaux :
- tendance primaire : zone baissière confirmée par une baisse ≥ 20 % depuis le sommet, zone
  haussière par une hausse ≥ 25 % depuis le creux (seuil haussier plus large pour qu'un rebond de
  marché baissier, ex. juin-août 2022 +23 %, ne coupe pas la zone rouge) ;
- mouvements secondaires : retournements ≥ 10 % à l'intérieur (teinte claire). Utilisé par myfiles/ndx_sentiment_chart.py.
"""
import numpy as np
import pandas as pd

PRIMARY = (0.20, 0.25)   # (baisse, hausse)
SECONDARY = (0.10, 0.10)


def zigzag(close, thr):
    dn, up = thr if isinstance(thr, tuple) else (thr, thr)
    """Pivots alternés : un extrême est confirmé comme pivot après un retournement ≥ thr."""
    x = close.values
    piv, trend, hi, lo = [], 0, 0, 0
    for i in range(1, len(x)):
        if x[i] > x[hi]:
            hi = i
        if x[i] < x[lo]:
            lo = i
        if trend >= 0 and x[i] <= x[hi] * (1 - dn):      # sommet confirmé
            if trend == 0:
                piv.append(0)
            piv.append(hi); trend, lo = -1, i
        elif trend <= 0 and x[i] >= x[lo] * (1 + up):    # creux confirmé
            if trend == 0:
                piv.append(0)
            piv.append(lo); trend, hi = 1, i
    piv.append(len(x) - 1)  # segment en cours (non confirmé)
    return [close.index[i] for i in piv]


def segments(close, thr):
    """[(debut, fin, +1/−1, variation)] entre pivots successifs."""
    p = [close.index[0]] + zigzag(close, thr)
    p = list(dict.fromkeys(p))
    out = []
    for a, b in zip(p[:-1], p[1:]):
        r = close[b] / close[a] - 1
        out.append((a, b, 1 if r > 0 else -1, r))
    return out


if __name__ == "__main__":
    import sys
    c = pd.read_parquet(sys.argv[1])["close"]
    c = c[c.index >= "2021-11-01"]
    for thr in (PRIMARY, SECONDARY):
        print(f"--- seuils {thr}")
        for a, b, s, r in segments(c, thr):
            print(f"  {a:%Y-%m-%d} → {b:%Y-%m-%d}  {'↑' if s > 0 else '↓'} {r:+.1%}  {(b - a).days} j")
