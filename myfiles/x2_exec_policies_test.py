"""
Audit x2.0 (suite) : politiques d'execution plus intelligentes que le seuil fixe.

P0 continu sans frais (ref backtest) ; P1 live actuel (sell >= 0.20 alloc x1) ;
P2 = P1 + sortie TOTALE toujours executee quand cible = 0 (garde-fou macro) ;
P3 seuil 0.10 ; P4 asymetrique : seuil 0.05 sous la MA (urgence), 0.20 au-dessus ;
P5 = P4 + sortie totale cible 0.
+ sweep VOL_TARGET du bras sous-MA dedie x2 (0.08 actuel, calibre x1).

Usage : python myfiles/x2_exec_policies_test.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.risk_off_strategy.data import load_ohlc, load_macro
from src.risk_off_strategy.strategy import (
    compute_allocation, yang_zhang_vol, simulate, ANN, SMA_LONG,
    BELOW_SCALE, GAP_CUTOFF, NFCI_OFF, CPI_OFF, ABOVE_CAP, GAP2_START,
    GAP2_SPAN, DECAY2_FLOOR,
)

ohlc = load_ohlc("QQQ")
o, h, l, c = (ohlc[x].values for x in ["open", "high", "low", "close"])
dates = ohlc.index
nfci, cpi = load_macro(dates)
N = len(c)
ret = pd.Series(c).pct_change().fillna(0).values
s = pd.Series(c).rolling(SMA_LONG, min_periods=1).mean().values
a1 = compute_allocation(c, nfci, cpi, high=h, low=l, open_=o)
target2 = 2 * a1
below_ma = c <= s   # regime connu a la decision (meme info que la cible)


def sim_policy(sell_thr_fn, full_exit_on_zero, sell_fee=0.005, exec_lag=1):
    """sell_thr_fn(t) -> seuil de vente en EXPO au jour t (decision t-1)."""
    tgt = np.concatenate([np.zeros(exec_lag), target2[:-exec_lag]])
    reg = np.concatenate([[False] * exec_lag, below_ma[:-exec_lag]])
    pos, fees = np.zeros(N), np.zeros(N)
    cur, ns = 0.0, 0
    for t in range(N):
        d = tgt[t] - cur
        thr = sell_thr_fn(reg[t])
        if d > 0:
            cur = tgt[t]
        elif d < 0 and (-d >= thr or (full_exit_on_zero and tgt[t] == 0.0)):
            fees[t] = (-d / 2) * sell_fee
            cur = tgt[t]; ns += 1
        pos[t] = cur
    rr = ret * pos - fees
    e = np.cumprod(1 + rr)
    yrs = N / ANN
    pk = np.maximum.accumulate(e)
    mdd = float(((e - pk) / pk).min())
    cagr = e[-1] ** (1 / yrs) - 1
    return dict(cagr=cagr, maxdd=mdd, sharpe=rr.mean() / rr.std() * np.sqrt(ANN),
                calmar=cagr / abs(mdd), fees=fees.sum() / yrs * 100, s=ns / yrs)


print("=" * 96)
print("POLITIQUES D'EXECUTION x2 (fee vente 0.5%, achats gratuits ; seuils en expo x2)")
print("=" * 96)
print(f"{'politique':<44} | {'CAGR':>6} {'maxDD':>7} {'Shrp':>5} {'Calm':>5} {'fees/an':>7} {'ventes/an':>9}")
print("-" * 96)
pols = [
    ("P0 continu sans frais (ref backtest)", lambda r: 0.0, False, 0.0),
    ("P1 live actuel : sell >= 0.40", lambda r: 0.40, False, 0.005),
    ("P2 = P1 + sortie totale si cible = 0", lambda r: 0.40, True, 0.005),
    ("P3 sell >= 0.20", lambda r: 0.20, False, 0.005),
    ("P3b = P3 + sortie totale si cible = 0", lambda r: 0.20, True, 0.005),
    ("P4 asym : 0.10 sous MA / 0.40 au-dessus", lambda r: 0.10 if r else 0.40, False, 0.005),
    ("P5 = P4 + sortie totale si cible = 0", lambda r: 0.10 if r else 0.40, True, 0.005),
]
for name, fn, fz, fee in pols:
    m = sim_policy(fn, fz, sell_fee=fee)
    print(f"{name:<44} | {m['cagr']*100:5.1f}% {m['maxdd']*100:6.1f}% {m['sharpe']:5.2f} "
          f"{m['calmar']:5.2f} {m['fees']:6.2f}% {m['s']:9.1f}")

# ── VOL_TARGET dedie x2 pour le bras sous-MA ──────────────
rv = yang_zhang_vol(o, h, l, c)
gap = c / s - 1.0
decay_down = np.clip(1.0 + gap / GAP_CUTOFF, 0, 1)
decay_up = np.clip(1.0 - np.maximum(0.0, gap - GAP2_START) / GAP2_SPAN, DECAY2_FLOOR, 1.0)
macro_off = ((np.nan_to_num(np.asarray(nfci, float), nan=-9) > NFCI_OFF)
             | (np.nan_to_num(np.asarray(cpi, float), nan=-9) > CPI_OFF))
above = np.clip(2 * ABOVE_CAP / rv, 0, 2) * decay_up
half = N // 2
print("\n" + "=" * 96)
print("VOL_TARGET DEDIE x2 (bras sous-MA ; 0.08 = actuel, sans frais)  | Calmar H1 / H2")
print("=" * 96)
for vt in (0.04, 0.05, 0.06, 0.08, 0.10):
    below = 2 * BELOW_SCALE * np.clip(vt / rv, 0, 1) * decay_down
    e = np.where(macro_off, 0.0, np.where(c > s, above, below))
    m = simulate(c, e)
    m1 = simulate(c[:half], e[:half]); m2 = simulate(c[half:], e[half:])
    tag = "  <- actuel" if vt == 0.08 else ""
    print(f"VT = {vt:.2f} : CAGR {m['cagr']*100:5.1f}%  maxDD {m['maxdd']*100:6.1f}%  "
          f"Sharpe {m['sharpe']:.2f}  Calmar {m['calmar']:.2f}  | {m1['calmar']:.2f} / {m2['calmar']:.2f}{tag}")
