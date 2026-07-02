"""
Audit + pistes d'amelioration de la variante x2.0 (LQQ).

Constat de depart : la x2.0 du backtest = 2 * alloc_x1, SANS frais ni financement,
et tous les parametres (ABOVE_CAP, DECAY2_FLOOR...) ont ete calibres sur les
metriques x1. Ce script teste :
  1. le drag reel du LQQ (TER + financement + FX) vs le modele 2*ret sans frais ;
  2. le levier optimal theorique (Kelly) sur QQQ brut et sur la strategie x1 ;
  3. des variantes de sizing DEDIEES au x2 :
     - cap2 : saturation vol de l'expo au-dessus de la MA (0.30 = x2 actuel)
     - below_mode : levier sous la MA (x2 actuel / cap a 1 / x1 / cash)
     - floor2 : plancher du decay_up (0.4 = actuel)
  avec full-sample + 2 moities (anti-overfit) et sensibilite au drag.

Usage : python myfiles/x2_improvement_analysis.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.risk_off_strategy.data import load_ohlc, load_macro
from src.risk_off_strategy.strategy import (
    yang_zhang_vol, compute_allocation, SMA_LONG, ABOVE_CAP, VOL_TARGET,
    BELOW_SCALE, GAP_CUTOFF, GAP2_START, GAP2_SPAN, DECAY2_FLOOR, ANN,
    NFCI_OFF, CPI_OFF,
)

# ── donnees ───────────────────────────────────────────────
ohlc = load_ohlc("QQQ")
o, h, l, c = (ohlc[x].values for x in ["open", "high", "low", "close"])
dates = ohlc.index
nfci, cpi = load_macro(dates)
N = len(c)
half = N // 2
mid = dates[half].date()

ret = pd.Series(c).pct_change().fillna(0).values
s = pd.Series(c).rolling(SMA_LONG, min_periods=1).mean().values
gap = c / s - 1.0
rv = yang_zhang_vol(o, h, l, c)
decay_down = np.clip(1.0 + gap / GAP_CUTOFF, 0, 1)
macro_off = ((np.nan_to_num(np.asarray(nfci, float), nan=-9) > NFCI_OFF)
             | (np.nan_to_num(np.asarray(cpi, float), nan=-9) > CPI_OFF))


def build_expo(cap2=2 * ABOVE_CAP, floor2=DECAY2_FLOOR, below="x2",
               g2start=GAP2_START, g2span=GAP2_SPAN):
    """Expo x2 dans [0,2]. cap2=0.30/floor2=0.4/below='x2' == 2*alloc_x1 actuel."""
    decay_up = np.clip(1.0 - np.maximum(0.0, gap - g2start) / g2span, floor2, 1.0)
    above = np.clip(cap2 / rv, 0, 2) * decay_up
    bx1 = BELOW_SCALE * np.clip(VOL_TARGET / rv, 0, 1) * decay_down
    below_arm = {"x2": 2 * bx1, "cap1": np.minimum(2 * bx1, 1.0),
                 "x1": bx1, "cash": np.zeros_like(bx1)}[below]
    expo = np.where(c > s, above, below_arm)
    return np.where(macro_off, 0.0, expo)


def sim(expo, sl=slice(0, N), drag=0.0, exec_lag=1):
    """Metriques ; drag = cout annuel (TER+financement) applique pro rata de la
    fraction detenue en LQQ (= pos/2)."""
    pos = np.zeros_like(expo)
    pos[exec_lag:] = expo[:-exec_lag]
    r = (ret * pos - (pos / 2) * drag / ANN)[sl]
    eq = np.cumprod(1 + r)
    yrs = len(r) / ANN
    peak = np.maximum.accumulate(eq)
    dd = float(((eq - peak) / peak).min())
    cagr = eq[-1] ** (1 / yrs) - 1
    return dict(cagr=cagr, maxdd=dd,
                sharpe=r.mean() / r.std() * np.sqrt(ANN) if r.std() > 0 else 0.0,
                calmar=cagr / abs(dd) if dd < 0 else np.inf,
                turn=float(np.abs(np.diff(pos[sl])).sum() / yrs))


F, H1, H2 = slice(0, N), slice(0, half), slice(half, N)


def line(name, e, drag=0.0):
    m, m1, m2 = sim(e, F, drag), sim(e, H1, drag), sim(e, H2, drag)
    print(f"{name:<34} | {m['cagr']*100:5.1f}% {m['maxdd']*100:6.1f}% {m['sharpe']:5.2f} "
          f"{m['calmar']:5.2f} {m['turn']:5.1f} | {m1['calmar']:5.2f} {m2['calmar']:5.2f}")
    return m


# verif : build_expo par defaut == 2*alloc_x1 de strategy.py
a1 = compute_allocation(c, nfci, cpi, high=h, low=l, open_=o)
assert np.allclose(build_expo(), 2 * a1, atol=1e-12), "mismatch build_expo vs strategy.py"

hdr = f"{'variante':<34} | {'CAGR':>5} {'maxDD':>6} {'Shrp':>5} {'Calm':>5} {'turn':>5} | {'CalH1':>5} {'CalH2':>5}"
print("=" * 100)
print(f"AUDIT x2.0  |  QQQ {dates[0].date()} -> {dates[-1].date()}  |  cesure moities = {mid}")
print("=" * 100)

# ── 1. drag reel LQQ vs modele sans frais ─────────────────
lqq = pd.read_parquet(ROOT / "data" / "LQQ.parquet")["close"].dropna()
ndx = pd.read_parquet(ROOT / "data" / "^NDX.parquet")["close"].dropna()
t0, t1 = max(lqq.index[0], ndx.index[0]), min(lqq.index[-1], ndx.index[-1])
lqq_w, ndx_w = lqq.loc[t0:t1], ndx.loc[t0:t1]
r_theo = 2 * ndx_w.pct_change().fillna(0)
yrs_l = (t1 - t0).days / 365.25
cagr_theo = float(np.prod(1 + r_theo) ** (1 / yrs_l) - 1)
cagr_lqq = float((lqq_w.iloc[-1] / lqq_w.iloc[0]) ** (1 / yrs_l) - 1)
print(f"\n1) DRAG REEL LQQ ({t0.date()} -> {t1.date()}, {yrs_l:.1f} ans)")
print(f"   2x NDX quotidien sans frais : CAGR {cagr_theo*100:6.2f}%")
print(f"   LQQ reel (EUR)              : CAGR {cagr_lqq*100:6.2f}%")
print(f"   ecart = {(cagr_theo-cagr_lqq)*100:+.2f} pt/an  (TER 0.6% + financement + FX EUR/USD ;")
print(f"   USD s'est apprecie sur la periode -> l'ecart SOUS-estime le drag hors FX)")

# ── 2. Kelly ──────────────────────────────────────────────
def kelly(r):
    mu, sig2 = r.mean() * ANN, r.var() * ANN
    return mu / sig2

r_x1 = ret[1:] * a1[:-1]
print(f"\n2) LEVIER OPTIMAL (Kelly, L* = mu/sigma^2, plein-echantillon)")
print(f"   QQQ brut      : L* = {kelly(ret[1:]):.2f}")
print(f"   strategie x1  : L* = {kelly(r_x1):.2f}   (vol realisee x1 = {r_x1.std()*np.sqrt(ANN)*100:.1f}%/an)")
above_mask = (c > s)[SMA_LONG:]
r_above = (ret * np.concatenate([[0], a1[:-1]]))[SMA_LONG:][above_mask]
print(f"   x1, jours au-dessus MA seulement : L* = {kelly(r_above):.2f}")

# ── 3. baselines ──────────────────────────────────────────
print(f"\n3) BASELINES (sans frais, comme le chart actuel)")
print(hdr); print("-" * 100)
line("B&H QQQ x1", np.ones(N))
line("strategie x1 (deployee)", a1.copy())
m20 = line("x2 actuel = 2*alloc_x1", build_expo())
line("x2 actuel + drag 2.5%/an", build_expo(), drag=0.025)

# ── 4. sensibilite au drag ────────────────────────────────
print(f"\n4) SENSIBILITE AU DRAG (x2 actuel) — CAGR full")
for dg in (0.0, 0.015, 0.025, 0.035):
    m = sim(build_expo(), F, dg)
    print(f"   drag {dg*100:3.1f}%/an : CAGR {m['cagr']*100:5.1f}%  Sharpe {m['sharpe']:.2f}  Calmar {m['calmar']:.2f}")

# ── 5. variante A : cap2 dedie (saturation vol au-dessus MA) ──
print(f"\n5) CAP2 DEDIE x2 — expo_above = clip(cap2/vol, 0, 2)*decay_up   (0.30 = actuel)")
print(hdr); print("-" * 100)
for cap2 in (0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30):
    tag = "  <- actuel (=2*0.15)" if abs(cap2 - 0.30) < 1e-9 else ""
    line(f"cap2 = {cap2:.2f}{tag}", build_expo(cap2=cap2))

# ── 6. variante B : levier sous la MA ─────────────────────
print(f"\n6) LEVIER SOUS LA MA  (actuel = 'x2' : 2*0.8*clip(0.08/vol,1)*decay -> expo max 1.6)")
print(hdr); print("-" * 100)
for bm, lbl in [("x2", "below x2 (actuel)"), ("cap1", "below cap a 1.0"),
                ("x1", "below = x1 (0.8 max)"), ("cash", "below = cash")]:
    line(lbl, build_expo(below=bm))

# ── 7. variante C : plancher decay_up ─────────────────────
print(f"\n7) PLANCHER decay_up (floor2, actuel = {DECAY2_FLOOR})")
print(hdr); print("-" * 100)
for fl in (0.2, 0.3, 0.4, 0.5, 0.6, 1.0):
    tag = "  <- actuel" if abs(fl - DECAY2_FLOOR) < 1e-9 else ("  (= sans decay_up)" if fl == 1.0 else "")
    line(f"floor2 = {fl:.1f}{tag}", build_expo(floor2=fl))

# ── 8. combos prometteurs + validation croisee ────────────
print(f"\n8) COMBOS (cap2 x below x floor2) — selection par Calmar sur H1, verif H2 (et inversement)")
print(hdr); print("-" * 100)
combos = [(cap2, bm, fl) for cap2 in (0.22, 0.26, 0.30) for bm in ("x2", "cap1", "x1")
          for fl in (0.3, 0.4, 0.6)]
res = []
for cap2, bm, fl in combos:
    e = build_expo(cap2=cap2, below=bm, floor2=fl)
    res.append(dict(cap2=cap2, below=bm, floor2=fl,
                    f=sim(e, F), h1=sim(e, H1), h2=sim(e, H2)))
best_h1 = max(res, key=lambda x: x["h1"]["calmar"])
best_h2 = max(res, key=lambda x: x["h2"]["calmar"])
cur = next(x for x in res if x["cap2"] == 0.30 and x["below"] == "x2" and x["floor2"] == 0.4)
for x, lbl in [(cur, "actuel (0.30/x2/0.4)"),
               (best_h1, f"argmax H1 ({best_h1['cap2']}/{best_h1['below']}/{best_h1['floor2']})"),
               (best_h2, f"argmax H2 ({best_h2['cap2']}/{best_h2['below']}/{best_h2['floor2']})")]:
    print(f"{lbl:<34} | {x['f']['cagr']*100:5.1f}% {x['f']['maxdd']*100:6.1f}% {x['f']['sharpe']:5.2f} "
          f"{x['f']['calmar']:5.2f} {x['f']['turn']:5.1f} | {x['h1']['calmar']:5.2f} {x['h2']['calmar']:5.2f}")
print(f"\n   argmax-H1 applique sur H2 : Calmar {best_h1['h2']['calmar']:.2f} "
      f"(actuel sur H2 : {cur['h2']['calmar']:.2f})")
print(f"   argmax-H2 applique sur H1 : Calmar {best_h2['h1']['calmar']:.2f} "
      f"(actuel sur H1 : {cur['h1']['calmar']:.2f})")

# top-5 combos full par Calmar
print(f"\n   top-5 combos (Calmar full) :")
for x in sorted(res, key=lambda z: -z["f"]["calmar"])[:5]:
    print(f"   cap2={x['cap2']:.2f} below={x['below']:<4} floor2={x['floor2']:.1f} : "
          f"CAGR {x['f']['cagr']*100:5.1f}%  maxDD {x['f']['maxdd']*100:5.1f}%  "
          f"Sharpe {x['f']['sharpe']:.2f}  Calmar {x['f']['calmar']:.2f}  "
          f"| H1 {x['h1']['calmar']:.2f} H2 {x['h2']['calmar']:.2f}")
