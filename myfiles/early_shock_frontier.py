"""
Creuse toutes les pistes de détection précoce de choc de vol AU-DESSUS de la MA.
Compare des signaux de vol candidats par leur FRONTIÈRE CAGR-vs-maxDD (le vrai juge,
pas la précocité seule).

Signaux (branche above-MA : alloc = clip(cap / signal)) :
  S0 YZ                          vol réalisée range-based (déployé)
  S1 VIX/100                     vol implicite pure (forward-looking)
  S2 max(YZ, VIX/100)            le plus alarmant des deux
  S3 max(YZ, VIX/100, TLTvol)    + vol obligataire (proxy MOVE)
  S4 YZ · max(1, VIX/VIX3M)      + structure par terme (booste si backwardation)
  S5 max(YZ,VIX/100)·max(1,VIX/VIX3M)   tout combiné

Fenêtre commune 2006-07 -> 2026 (VIX3M dispo depuis 2006 ; inclut GFC/2018/Covid/2022).
Seuil normalisé par percentile du signal sur les jours above-MA -> comparaison à
agressivité comparable. Causal / exec_lag=1 (cf. lookahead_audit.py).

Sortie : tableau + frontière -> myfiles/early_shock_frontier.png
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.risk_off_strategy.data import load_ohlc, load_macro
from src.risk_off_strategy.strategy import (
    yang_zhang_vol, simulate, SMA_LONG, VOL_TARGET, BELOW_SCALE, GAP_CUTOFF,
    NFCI_OFF, CPI_OFF, ANN,
)

# ── données alignées sur QQQ (ffill = passé uniquement) ───
qqq = load_ohlc("QQQ")
dates_all = qqq.index
def align(s):
    return s.reindex(dates_all.union(s.index)).sort_index().ffill().reindex(dates_all)

yz_all = pd.Series(yang_zhang_vol(*(qqq[x].values for x in ["open", "high", "low", "close"])), index=dates_all)
vix = align(pd.read_parquet(ROOT / "data" / "vix_spot.parquet")["close"]) / 100.0
vix3m = align(pd.read_parquet(ROOT / "data" / "vix3m.parquet")["close"]) / 100.0
tlt = load_ohlc("TLT")
tltvol = pd.Series(yang_zhang_vol(*(tlt[x].reindex(dates_all).ffill().values for x in ["open", "high", "low", "close"])), index=dates_all)
backwardation = (vix / vix3m).clip(lower=1.0)   # >1 quand court > long (stress)

# ── fenêtre commune ───────────────────────────────────────
mask = dates_all >= "2006-07-17"
d = dates_all[mask]
c = qqq["close"].values[mask]
yz = yz_all.values[mask]
sig = {
    "S0 YZ (déployé)":              yz,
    "S1 VIX/100":                   vix.values[mask],
    "S2 max(YZ,VIX)":               np.maximum(yz, vix.values[mask]),
    "S3 max(YZ,VIX,TLTvol)":        np.maximum.reduce([yz, vix.values[mask], tltvol.values[mask]]),
    "S4 YZ·termstruct":             yz * backwardation.values[mask],
    "S5 max(YZ,VIX)·termstruct":    np.maximum(yz, vix.values[mask]) * backwardation.values[mask],
}
nfci, cpi = load_macro(d)
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
    """(maxdd, cagr, calmar, tim) en balayant le seuil = percentile du signal above-MA."""
    pts = []
    base = signal[above]
    for q in range(45, 96, 3):
        cap = np.percentile(base, q)
        m = simulate(c, alloc_for(signal, cap))
        pts.append((m["maxdd"], m["cagr"], m["calmar"], m["time_in_market"], q))
    return np.array(pts)

# ── tableau : au maxDD le plus proche de -18%, quel CAGR ? + Calmar max ──
print("=" * 96)
print(f"FRONTIÈRE above-MA sur {d[0].date()} -> {d[-1].date()}  (fenêtre commune, GFC/2018/Covid/2022)")
print("=" * 96)
print(f"{'signal':<28} | CAGR@maxDD≈-18% | Calmar max (maxDD, CAGR)")
print("-" * 96)
fr = {}
for name, s in sig.items():
    f = frontier(s)
    fr[name] = f
    # point le plus proche de maxDD -18%
    i18 = np.argmin(np.abs(f[:, 0] - (-0.18)))
    # Calmar-optimal
    ic = np.argmax(f[:, 2])
    print(f"{name:<28} |   {f[i18,1]*100:5.1f}%  (DD {f[i18,0]*100:.1f}%) |  "
          f"{f[ic,2]:.2f}  (DD {f[ic,0]*100:.1f}%, CAGR {f[ic,1]*100:.1f}%)")

# ── précocité de la structure par terme vs YZ ─────────────
print("\n" + "=" * 96)
print("Précocité structure par terme (VIX/VIX3M > seuil) vs YZ — 1er jour au 80e percentile")
print("=" * 96)
ts = (vix / vix3m)
def cross80(series, m):
    s = series[m]; thr = s.expanding(min_periods=200).quantile(0.80)
    return (s > thr).values, np.where(m)[0]
crises = {"GFC 2008": ("2007-10-01", "2008-12-01"), "2018Q4": ("2018-10-01", "2018-12-31"),
          "Covid 2020": ("2020-02-01", "2020-04-01"), "Rate 2022": ("2022-01-01", "2022-06-01")}
cr_ts, idx_ts = cross80(ts, dates_all >= "2006-07-17")
cr_yz, idx_yz = cross80(yz_all, dates_all >= "2006-07-17")
for cname, (a, b) in crises.items():
    win = (d >= a) & (d <= b)
    ii = np.where(win)[0]
    def first(cr):
        hit = ii[cr[ii]]; return hit[0] if len(hit) else None
    its, iyz = first(cr_ts), first(cr_yz)
    if its is None or iyz is None:
        print(f"  {cname:<12} termstruct {'-' if its is None else d[its].date()}  YZ {'-' if iyz is None else d[iyz].date()}")
    else:
        print(f"  {cname:<12} termstruct {d[its].date()}  vs YZ {d[iyz].date()}  -> {iyz-its:+d}j")

# ── graphe frontières ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 7))
colors = {"S0 YZ (déployé)": "black", "S1 VIX/100": "royalblue", "S2 max(YZ,VIX)": "green",
          "S3 max(YZ,VIX,TLTvol)": "purple", "S4 YZ·termstruct": "darkorange",
          "S5 max(YZ,VIX)·termstruct": "crimson"}
for name, f in fr.items():
    lw = 2.4 if name.startswith("S0") else 1.4
    ax.plot(-f[:, 0] * 100, f[:, 1] * 100, "o-", color=colors[name], lw=lw, ms=3, label=name)
ax.set_xlabel("maxDD (%, ↓ = mieux à gauche)")
ax.set_ylabel("CAGR (%, ↑ = mieux)")
ax.set_title(f"Frontière CAGR-vs-maxDD des signaux above-MA  ({d[0].date()}->{d[-1].date()})\n"
             "haut-gauche = domine")
ax.grid(True, alpha=0.25); ax.legend(fontsize=9)
out = ROOT / "myfiles" / "early_shock_frontier.png"
fig.tight_layout(); fig.savefig(out, dpi=140)
print(f"\nGraphe -> {out}")
