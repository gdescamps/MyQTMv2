"""
Pistes Calmar x2 : les 5 pires DD de la x2 sont des regimes de CHOP (2015-16,
2004-05, 2018, 2025, dot-com), pas les grandes crises. On teste des mecanismes
anti-chop ORTHOGONAUX a la formule actuelle (dont les parametres sont satures) :

  V1 pente SMA250 : pente <= 0 -> le bras "above" est traite en "below" (l'expo
     pleine exige prix > MA ET MA montante)
  V2 hysteresis : on n'entre en regime "above" que si prix > MA*(1+b), on ne
     sort que si prix < MA (tue les flip-flops autour de la MA)
  V3 compteur de whipsaw : n croisements de MA sur 60j -> expo reduite de moitie
  V4 structure par terme VIX : VIX/VIX3M >= seuil (backwardation) -> expo coupee
     (NaN avant 2006 -> neutre)
  V5 momentum absolu 12m : ret_252j < 0 -> bras "below" meme au-dessus de la MA
  V6 combo des 2 meilleurs si complementaires

Tout causal (signaux au close t, execution t+1 via exec_lag=1). Sans frais pour
comparabilite avec le chart. Full + 2 moities + detail des 5 pires episodes.

Usage : python myfiles/x2_calmar_antichop_test.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.risk_off_strategy.data import load_ohlc, load_macro, DATA_DIR
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


def assemble(regime_above, above=None, scale=None):
    """expo x2 : bras above/below selon `regime_above` (bool), garde-fous macro."""
    e = np.where(regime_above, above_arm if above is None else above, below_arm)
    if scale is not None:
        e = e * scale
    return np.where(macro_off, 0.0, e)


def sim(expo, sl=slice(0, N), exec_lag=1):
    pos = np.zeros_like(expo)
    pos[exec_lag:] = expo[:-exec_lag]
    r = (ret * pos)[sl]
    eq = np.cumprod(1 + r)
    yrs = len(r) / ANN
    pk = np.maximum.accumulate(eq)
    dd = float(((eq - pk) / pk).min())
    cagr = eq[-1] ** (1 / yrs) - 1
    return dict(cagr=cagr, maxdd=dd, sharpe=r.mean() / r.std() * np.sqrt(ANN),
                calmar=cagr / abs(dd) if dd < 0 else np.inf, eq=eq, pos=pos)


# 5 pires episodes de la x2 actuelle (bornes trouvees a l'audit precedent)
EPISODES = [("dot-com", "2000-05-03", "2003-04-09"), ("2004-05 chop", "2004-03-09", "2005-04-20"),
            ("2015-16 chop", "2015-08-20", "2016-06-27"), ("2018 T4", "2018-10-10", "2019-01-03"),
            ("2025 tarifs", "2025-01-13", "2025-04-21")]


def episode_dd(expo):
    """maxDD de la variante sur chaque fenetre d'episode (equity locale)."""
    pos = np.concatenate([[0.0], expo[:-1]])
    r = ret * pos
    out = []
    for name, d0, d1 in EPISODES:
        m = (dates >= d0) & (dates <= pd.Timestamp(d1) + pd.Timedelta(days=90))
        e = np.cumprod(1 + r[m])
        pk = np.maximum.accumulate(e)
        out.append(((e - pk) / pk).min())
    return out


F, H1, H2 = slice(0, N), slice(0, half), slice(half, N)
rows = []


def line(name, expo):
    m, m1, m2 = sim(expo, F), sim(expo, H1), sim(expo, H2)
    eps = episode_dd(expo)
    rows.append((name, m, m1, m2, eps))
    print(f"{name:<40} | {m['cagr']*100:5.1f}% {m['maxdd']*100:6.1f}% {m['sharpe']:5.2f} "
          f"{m['calmar']:5.2f} | {m1['calmar']:5.2f} {m2['calmar']:5.2f} | "
          + " ".join(f"{e*100:5.0f}%" for e in eps))


print("=" * 118)
print(f"PISTES CALMAR x2 (sans frais)  |  {dates[0].date()} -> {dates[-1].date()}  |  cesure = {dates[half].date()}")
print("=" * 118)
print(f"{'variante':<40} | {'CAGR':>5} {'maxDD':>6} {'Shrp':>5} {'Calm':>5} | {'CalH1':>5} {'CalH2':>5} | "
      "dotcom 04-05 15-16  2018  2025")
print("-" * 118)

raw_above = c > s
line("x2 actuelle", assemble(raw_above))

# V1 : pente de la SMA250
for k in (10, 20, 40):
    slope_up = np.concatenate([[True] * k, s[k:] > s[:-k]])
    line(f"V1 pente SMA>0 sur {k}j (above ET pente)", assemble(raw_above & slope_up))

# V2 : hysteresis sur le croisement
for b in (0.01, 0.02, 0.03):
    reg = np.zeros(N, bool)
    cur = False
    for t in range(N):
        if c[t] > s[t] * (1 + b):
            cur = True
        elif c[t] < s[t]:
            cur = False
        reg[t] = cur
    line(f"V2 hysteresis entree +{b*100:.0f}%", assemble(reg))

# V3 : compteur de croisements MA sur 60j -> demi-expo si chop
sign = np.sign(c - s)
flips = pd.Series((np.diff(sign, prepend=sign[0]) != 0).astype(float)).rolling(60).sum().fillna(0).values
for nf in (3, 4, 6):
    scale = np.where(flips >= nf, 0.5, 1.0)
    line(f"V3 whipsaw: >={nf} flips/60j -> expo x0.5", assemble(raw_above, scale=scale))

# V4 : VIX/VIX3M (backwardation) -> coupe
vix = pd.read_parquet(DATA_DIR / "vix_spot.parquet")["close"]
v3m = pd.read_parquet(DATA_DIR / "vix3m.parquet")["close"]
ts = (vix / v3m).reindex(dates).ffill().values     # NaN avant 2006
for thr, sc in [(1.0, 0.5), (1.0, 0.0), (0.95, 0.5)]:
    scale = np.where(np.nan_to_num(ts, nan=0.0) >= thr, sc, 1.0)
    line(f"V4 VIX/VIX3M>={thr:.2f} -> expo x{sc:.1f}", assemble(raw_above, scale=scale))

# V5 : momentum absolu 12m
mom = np.concatenate([[1.0] * 252, c[252:] / c[:-252] - 1.0])
line("V5 mom 12m<0 -> bras below", assemble(raw_above & (mom > 0)))

print("-" * 118)
# combos : (a) pente 20j + hysteresis 2% ; (b) pente 20j + VIX ts
k = 20
slope_up = np.concatenate([[True] * k, s[k:] > s[:-k]])
reg2 = np.zeros(N, bool); cur = False
for t in range(N):
    if c[t] > s[t] * 1.02:
        cur = True
    elif c[t] < s[t]:
        cur = False
    reg2[t] = cur
line("V6a pente20 + hysteresis 2%", assemble(reg2 & slope_up))
scale_v4 = np.where(np.nan_to_num(ts, nan=0.0) >= 1.0, 0.5, 1.0)
line("V6b pente20 + VIX/VIX3M>=1 -> x0.5", assemble(raw_above & slope_up, scale=scale_v4))
line("V6c hyst2% + VIX ts x0.5", assemble(reg2, scale=scale_v4))
