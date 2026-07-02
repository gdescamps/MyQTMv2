"""
decay2 : trim de l'expo AU-DESSUS de la MA quand le prix est trop etendu (gap grand).
Hypothese : protege contre la reversion d'un marche sur-etendu que le cap-vol rate
(melt-up a faible vol). Surtout utile en x2 (reversion amplifiee).

Juge : SHARPE (si plat -> decay2 ne fait que de-lever, sans valeur ; si monte ->
il time la reversion). + Calmar/maxDD (ce qui compte pour x2).

Causal / exec_lag=1. Usage: python myfiles/decay2_gap_test.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.risk_off_strategy.data import load_ohlc, load_macro
from src.risk_off_strategy.strategy import (
    yang_zhang_vol, SMA_LONG, VOL_TARGET, BELOW_SCALE, GAP_CUTOFF,
    NFCI_OFF, CPI_OFF, ANN, ABOVE_CAP,
)

o = load_ohlc("QQQ")
oo, hh, ll, c = (o[x].values for x in ["open", "high", "low", "close"])
nfci, cpi = load_macro(o.index)
ret = pd.Series(c).pct_change().fillna(0).values
yz = yang_zhang_vol(oo, hh, ll, c)
sma = pd.Series(c).rolling(SMA_LONG, min_periods=1).mean().values
gap = c / sma - 1.0
above = c > sma
decay = np.clip(1.0 + gap / GAP_CUTOFF, 0, 1)
below_alloc = BELOW_SCALE * np.clip(VOL_TARGET / np.maximum(yz, 1e-6), 0, 1) * decay

print("Distribution du gap (close/SMA-1) quand close > MA :")
g = gap[above]
print("  " + "  ".join(f"p{p}={np.percentile(g,p)*100:.0f}%" for p in [50, 75, 90, 95, 99]))
print()


def make_alloc(g0, gspan, floor):
    """decay2 = trim lineaire quand gap > g0 (plancher floor)."""
    if g0 is None:
        decay2 = np.ones_like(c)
    else:
        decay2 = np.clip(1.0 - np.maximum(0.0, gap - g0) / gspan, floor, 1.0)
    a = np.where(above, np.clip(ABOVE_CAP / np.maximum(yz, 1e-6), 0, 1) * decay2, below_alloc)
    a = np.clip(a, 0, 1)
    if nfci is not None:
        a = np.where(np.nan_to_num(nfci, nan=-9) > NFCI_OFF, 0.0, a)
    if cpi is not None:
        a = np.where(np.nan_to_num(cpi, nan=-9) > CPI_OFF, 0.0, a)
    return a


def metrics(alloc, L):
    pos = np.concatenate([[0.0], alloc[:-1]])
    r = ret * L * pos
    eq = np.cumprod(1 + r); yrs = len(c) / ANN
    peak = np.maximum.accumulate(eq); dd = ((eq - peak) / peak).min()
    cagr = eq[-1] ** (1 / yrs) - 1
    sh = r.mean() / r.std() * np.sqrt(ANN)
    tim = (pos > 0).mean()
    return cagr, dd, sh, cagr / abs(dd), tim


configs = [
    ("sans decay2 (deploye)", None, None, None),
    ("g0=0.10 span0.20 fl0.4", 0.10, 0.20, 0.4),
    ("g0=0.15 span0.20 fl0.4", 0.15, 0.20, 0.4),
    ("g0=0.20 span0.20 fl0.4", 0.20, 0.20, 0.4),
    ("g0=0.15 span0.15 fl0.5", 0.15, 0.15, 0.5),
    ("g0=0.10 span0.30 fl0.3", 0.10, 0.30, 0.3),
]
for L in (1.0, 2.0):
    print("=" * 92)
    print(f"LEVIER x{L:.1f}  (Sharpe plat entre configs => decay2 ne fait que de-lever)")
    print("=" * 92)
    print(f"{'config':<26} | CAGR   maxDD  Sharpe Calmar  TiM")
    print("-" * 92)
    for name, g0, gspan, floor in configs:
        cg, dd, sh, cal, tim = metrics(make_alloc(g0, gspan, floor), L)
        print(f"{name:<26} | {cg*100:5.1f}% {dd*100:6.1f}% {sh:.3f}  {cal:.2f}   {tim*100:.0f}%")
    print()
