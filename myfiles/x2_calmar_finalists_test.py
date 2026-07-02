"""
Pistes Calmar x2, finale : sweep fin des 2 gagnants du banc anti-chop
(V1 pente SMA250 sur k jours ; V5 momentum absolu 12m) + combo, puis
verification NETTE D'EXECUTION (vente si delta expo >= 0.40, fee 0.5%).

Le sweep fin de k repond au drapeau rouge de non-monotonicite (k=20 pire que
k=10 et k=40 au 1er banc) : si le gain ne tient que sur un k isole, c'est du
bruit ; s'il tient sur une plage large, c'est structurel.

Usage : python myfiles/x2_calmar_finalists_test.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.risk_off_strategy.data import load_ohlc, load_macro
from src.risk_off_strategy.strategy import (
    yang_zhang_vol, SMA_LONG, ABOVE_CAP, VOL_TARGET, BELOW_SCALE, GAP_CUTOFF,
    GAP2_START, GAP2_SPAN, DECAY2_FLOOR, NFCI_OFF, CPI_OFF, ANN,
)

ohlc = load_ohlc("QQQ")
o, h, l, c = (ohlc[x].values for x in ["open", "high", "low", "close"])
dates = ohlc.index
nfci, cpi = load_macro(dates)
N = len(c)
half = N // 2
ret = pd.Series(c).pct_change().fillna(0).values
s = pd.Series(c).rolling(SMA_LONG, min_periods=1).mean().values
gap = c / s - 1.0
rv = yang_zhang_vol(o, h, l, c)
decay_down = np.clip(1.0 + gap / GAP_CUTOFF, 0, 1)
decay_up = np.clip(1.0 - np.maximum(0.0, gap - GAP2_START) / GAP2_SPAN, DECAY2_FLOOR, 1.0)
macro_off = ((np.nan_to_num(np.asarray(nfci, float), nan=-9) > NFCI_OFF)
             | (np.nan_to_num(np.asarray(cpi, float), nan=-9) > CPI_OFF))
above_arm = np.clip(2 * ABOVE_CAP / rv, 0, 2) * decay_up
below_arm = 2 * BELOW_SCALE * np.clip(VOL_TARGET / rv, 0, 1) * decay_down
raw_above = c > s


def assemble(regime_above):
    return np.where(macro_off, 0.0, np.where(regime_above, above_arm, below_arm))


def sim(expo, sl=slice(0, N)):
    pos = np.concatenate([[0.0], expo[:-1]])
    r = (ret * pos)[sl]
    eq = np.cumprod(1 + r)
    yrs = len(r) / ANN
    pk = np.maximum.accumulate(eq)
    dd = float(((eq - pk) / pk).min())
    cagr = eq[-1] ** (1 / yrs) - 1
    return dict(cagr=cagr, maxdd=dd, sharpe=r.mean() / r.std() * np.sqrt(ANN),
                calmar=cagr / abs(dd) if dd < 0 else np.inf)


def sim_net(expo, sell_thr=0.40, sell_fee=0.005):
    """Execution live : achats libres, vente seulement si delta expo >= seuil."""
    tgt = np.concatenate([[0.0], expo[:-1]])
    pos, fees = np.zeros(N), np.zeros(N)
    cur = 0.0
    for t in range(N):
        d = tgt[t] - cur
        if d > 0:
            cur = tgt[t]
        elif d < 0 and -d >= sell_thr:
            fees[t] = (-d / 2) * sell_fee
            cur = tgt[t]
        pos[t] = cur
    r = ret * pos - fees
    eq = np.cumprod(1 + r)
    yrs = N / ANN
    pk = np.maximum.accumulate(eq)
    dd = float(((eq - pk) / pk).min())
    cagr = eq[-1] ** (1 / yrs) - 1
    return dict(cagr=cagr, maxdd=dd, sharpe=r.mean() / r.std() * np.sqrt(ANN),
                calmar=cagr / abs(dd), fees=fees.sum() / yrs * 100)


def slope_reg(k):
    return raw_above & np.concatenate([[True] * k, s[k:] > s[:-k]])


def mom_reg(k):
    return raw_above & (np.concatenate([[1.0] * k, c[k:] / c[:-k] - 1.0]) > 0)


F, H1, H2 = slice(0, N), slice(0, half), slice(half, N)


def line(name, expo):
    m, m1, m2 = sim(expo, F), sim(expo, H1), sim(expo, H2)
    print(f"{name:<38} | {m['cagr']*100:5.1f}% {m['maxdd']*100:6.1f}% {m['sharpe']:5.2f} "
          f"{m['calmar']:5.2f} | {m1['calmar']:5.2f} {m2['calmar']:5.2f}")


hdr = f"{'variante':<38} | {'CAGR':>5} {'maxDD':>6} {'Shrp':>5} {'Calm':>5} | {'CalH1':>5} {'CalH2':>5}"
print("=" * 96)
print("SWEEP FIN — pente SMA250 sur k jours (V1)")
print("=" * 96)
print(hdr); print("-" * 96)
line("x2 actuelle (ref)", assemble(raw_above))
for k in (10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 80):
    line(f"pente k={k}", assemble(slope_reg(k)))

print()
print("=" * 96)
print("SWEEP FIN — momentum absolu k jours (V5)")
print("=" * 96)
print(hdr); print("-" * 96)
for k in (105, 126, 168, 189, 210, 252, 294):
    line(f"mom k={k}", assemble(mom_reg(k)))

print()
print("=" * 96)
print("COMBOS pente x momentum")
print("=" * 96)
print(hdr); print("-" * 96)
for ks, km in [(40, 252), (40, 189), (50, 252), (30, 252)]:
    line(f"pente{ks} + mom{km}", assemble(slope_reg(ks) & mom_reg(km)))

print()
print("=" * 96)
print("NET D'EXECUTION (vente si delta expo >= 0.40, fee 0.5% ; achats libres)")
print("=" * 96)
print(f"{'variante':<38} | {'CAGR':>5} {'maxDD':>6} {'Shrp':>5} {'Calm':>5} | {'fees/an':>7}")
print("-" * 96)
for name, e in [("x2 actuelle", assemble(raw_above)),
                ("pente k=40", assemble(slope_reg(40))),
                ("mom k=252", assemble(mom_reg(252))),
                ("pente40 + mom252", assemble(slope_reg(40) & mom_reg(252)))]:
    m = sim_net(e)
    print(f"{name:<38} | {m['cagr']*100:5.1f}% {m['maxdd']*100:6.1f}% {m['sharpe']:5.2f} "
          f"{m['calmar']:5.2f} | {m['fees']:6.2f}%")
