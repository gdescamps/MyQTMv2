"""
Décompose la vol Yang-Zhang en overnight / intraday et teste si le gap overnight
porte la précocité. Puis teste le gap brut comme déclencheur direct.

Causal, non-fuite : tout au jour t utilise O/H/L/C<=t et C_{t-1}.
Usage: python myfiles/overnight_gap_test.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.risk_off_strategy.data import load_macro
from src.risk_off_strategy.strategy import simulate, SMA_LONG, VOL_TARGET, BELOW_SCALE, GAP_CUTOFF, ANN

df = pd.read_parquet(ROOT / "data" / "QQQ.parquet").sort_index().loc["2000-01-01":]
o, h, l, c = df["open"], df["high"], df["low"], df["close"]
dates = df.index
nfci, cpi = load_macro(dates)
W = 10

# ── composantes (causales) ────────────────────────────────
lo = np.log(o / c.shift(1))          # overnight (O_t / C_{t-1})
lc = np.log(c / o)                    # open -> close (intraday close)
ho, ll = np.log(h / o), np.log(l / o)
rs = ho * (ho - lc) + ll * (ll - lc)  # Rogers-Satchell (intraday range)

k = 0.34 / (1.34 + (W + 1) / (W - 1))
var_on = lo.rolling(W, min_periods=5).var()          # overnight
var_oc = lc.rolling(W, min_periods=5).var()          # open-close
var_rs = rs.rolling(W, min_periods=5).mean()         # intraday range
var_yz = var_on + k * var_oc + (1 - k) * var_rs      # Yang-Zhang total

# part de la variance YZ portée par l'overnight (pondérée comme dans YZ : poids 1)
share_on = (var_on / var_yz).clip(0, 1)

print("=" * 90)
print("A. Part moyenne de la variance YZ portée par le terme OVERNIGHT")
print("=" * 90)
print(f"  moyenne globale : {share_on.mean()*100:.0f}%")
crises = {
    "Dotcom 2000": ("2000-07-01", "2001-04-01"), "GFC 2008": ("2007-10-01", "2008-12-01"),
    "Covid 2020": ("2020-02-01", "2020-04-01"), "Rate 2022": ("2022-01-01", "2022-06-01"),
    "2018Q4": ("2018-10-01", "2018-12-31"),
}
for name, (a, b) in crises.items():
    win = (dates >= a) & (dates <= b)
    print(f"  {name:<14s} : {share_on[win].mean()*100:.0f}%")

# ── précocité : chaque composante vs sa propre 80e percentile ─────
print()
print("=" * 90)
print("B. Précocité : 1er jour où la vol composante franchit SA 80e percentile glissante")
print("   (percentile propre -> normalise l'échelle). avance = jours avant l'intraday-RS")
print("=" * 90)
def pctl_cross(series):
    thr = series.expanding(min_periods=250).quantile(0.80)
    return (series > thr).values
von = np.sqrt(var_on.clip(0) * ANN)
vrs = np.sqrt(var_rs.clip(0) * ANN)
vyz = np.sqrt(var_yz.clip(0) * ANN)
cr_on, cr_rs, cr_yz = pctl_cross(von), pctl_cross(vrs), pctl_cross(vyz)
for name, (a, b) in crises.items():
    idx = np.where((dates >= a) & (dates <= b))[0]
    def first(cr):
        hit = idx[cr[idx]]
        return hit[0] if len(hit) else None
    ir, ion, iyz = first(cr_rs), first(cr_on), first(cr_yz)
    def tag(i, ref):
        if i is None: return "jamais"
        if ref is None: return f"{dates[i].date()}"
        d = ref - i
        return f"{dates[i].date()} ({'+' if d>0 else ''}{d}j)"
    print(f"  {name:<14s}  intraday-RS {tag(ir,None):<18s}  overnight {tag(ion,ir):<20s}  YZ {tag(iyz,ir)}")

# ── C. gap brut comme déclencheur DIRECT (même matin) ─────
print()
print("=" * 90)
print("C. Gap overnight brut comme trigger direct : si gap_t <= seuil -> alloc 0 ce jour")
print("   (réagit le matin même ; comparé à la stratégie déployée cap-0.25)")
print("=" * 90)
from src.risk_off_strategy.strategy import compute_allocation, ABOVE_CAP
price = c.values
base = compute_allocation(price, nfci, cpi, high=h.values, low=l.values, open_=o.values)
def fmt(n, a):
    m = simulate(price, a)
    return (f"{n:<34s} CAGR {m['cagr']*100:4.1f}%  maxDD {m['maxdd']*100:6.1f}%  "
            f"Sharpe {m['sharpe']:.2f}  Calmar {m['calmar']:.2f}  TiM {m['time_in_market']*100:.0f}%")
print(fmt("déployé (cap 0.25, pas de gap-trig)", base))
print("-" * 90)
gap_on = lo.values  # ln(O_t/C_{t-1}) ; connu à l'ouverture de t
for thr in [-0.02, -0.03, -0.04, -0.05]:
    a = base.copy()
    # overlay : gap-down brutal -> coupe ce jour (garde le min avec l'alloc de base)
    a = np.where(gap_on <= thr, np.minimum(a, 0.0), a)
    n_trig = int((gap_on <= thr).sum())
    print(fmt(f"+ gap-trigger <= {thr*100:.0f}%  ({n_trig} jours)", a))
