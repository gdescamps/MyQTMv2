"""
Filtrer vol positive vs negative : la semi-volatilite downside (rendements < 0)
comme signal du cap above-MA, vs la vol totale (YZ / close-to-close).

Idee : une hausse volatile ne devrait pas declencher le risk-off. On ne coupe que
sur la vol des jours BAISSIERS. Question : ca recupere-t-il du CAGR a maxDD egal,
ou la vol clustering rend-elle le downside aussi predictif que le total ?

Causal / exec_lag=1. Usage: python myfiles/downside_vol_test.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.risk_off_strategy.data import load_ohlc, load_macro
from src.risk_off_strategy.strategy import (
    yang_zhang_vol, simulate, SMA_LONG, VOL_TARGET, BELOW_SCALE, GAP_CUTOFF,
    NFCI_OFF, CPI_OFF, ANN,
)

o = load_ohlc("QQQ")
oo, hh, ll, c = (o[x].values for x in ["open", "high", "low", "close"])
dates = o.index
nfci, cpi = load_macro(dates)
ret = pd.Series(c).pct_change().fillna(0).values


def semi(mask_sign, w=10):
    """Semi-vol annualisee (racine de la semi-variance sur fenetre w)."""
    sq = np.where(mask_sign, ret ** 2, 0.0)
    v = pd.Series(sq).rolling(w, min_periods=5).mean().bfill().values
    return np.maximum(np.sqrt(v * ANN), 1e-6)


yz = yang_zhang_vol(oo, hh, ll, c)                       # totale (range)
vol_tot_cc = semi(np.ones_like(ret, bool))               # totale (close-to-close)
vol_down = semi(ret < 0)                                 # downside seule
vol_up = semi(ret > 0)                                   # upside seule (curiosite)

signals = {
    "YZ totale (deploye)": yz,
    "cc totale":           vol_tot_cc,
    "cc DOWNSIDE":         vol_down,
    "cc UPSIDE":           vol_up,
    "max(YZ, cc-down)":    np.maximum(yz, vol_down),
}

sma = pd.Series(c).rolling(SMA_LONG, min_periods=1).mean().values
above = c > sma
decay = np.clip(1.0 + (c / sma - 1.0) / GAP_CUTOFF, 0, 1)
below_alloc = BELOW_SCALE * np.clip(VOL_TARGET / np.maximum(yz, 1e-6), 0, 1) * decay


def alloc_for(signal, cap):
    a = np.where(above, np.clip(cap / np.maximum(signal, 1e-6), 0, 1), below_alloc)
    a = np.clip(a, 0, 1)
    if nfci is not None:
        a = np.where(np.nan_to_num(nfci, nan=-9) > NFCI_OFF, 0.0, a)
    if cpi is not None:
        a = np.where(np.nan_to_num(cpi, nan=-9) > CPI_OFF, 0.0, a)
    return a


def frontier(signal):
    pts = []
    base = signal[above]
    for q in range(40, 96, 2):
        cap = np.percentile(base, q)
        m = simulate(c, alloc_for(signal, cap))
        pts.append((m["maxdd"], m["cagr"], m["calmar"], m["time_in_market"]))
    return np.array(pts)


print("=" * 92)
print("FRONTIERE above-MA (full 2000-2026) : vol totale vs semi-vol downside/upside")
print("=" * 92)
print(f"{'signal':<22} | CAGR@maxDD≈-19% | CAGR@maxDD≈-24% | Calmar max (DD, CAGR)")
print("-" * 92)
for name, s in signals.items():
    f = frontier(s)
    def cagr_at(dd_target):
        i = np.argmin(np.abs(f[:, 0] - dd_target)); return f[i, 1] * 100, f[i, 0] * 100
    c19, d19 = cagr_at(-0.19)
    c24, d24 = cagr_at(-0.24)
    ic = np.argmax(f[:, 2])
    print(f"{name:<22} |  {c19:5.1f}% (DD {d19:.0f}) |  {c24:5.1f}% (DD {d24:.0f}) |  "
          f"{f[ic,2]:.2f} (DD {f[ic,0]*100:.0f}, CAGR {f[ic,1]*100:.1f})")

print()
print("Comparaison directe au MEME maxDD -> si DOWNSIDE > totale, filtrer la vol positive aide.")
print("Note: vol clustering -> le downside peut rester aussi predictif que le total (a verifier ci-dessus).")
