"""
Passe fine sur ABOVE_CAP (seuil vol au-dessus de la SMA250).
- grille fine full-sample + par moitié
- validation croisée : argmax-Calmar sur une moitié -> perf sur l'autre (anti-overfit)
- graphe de la courbe d'arbitrage -> myfiles/above_cap_sweep.png

Causal / non-fuite (cf. lookahead_audit.py). Usage: python myfiles/above_cap_sweep.py
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
    yang_zhang_vol, compute_allocation, simulate, SMA_LONG, ABOVE_CAP,
)

ohlc = load_ohlc("QQQ")
o, h, l, c = (ohlc[x].values for x in ["open", "high", "low", "close"])
dates = ohlc.index
nfci, cpi = load_macro(dates)
N = len(c)
half = N // 2
mid_date = dates[half].date()

CAPS = np.round(np.arange(0.18, 0.451, 0.01), 2)


def metr(cap, sl):
    a = compute_allocation(c, nfci, cpi, high=h, low=l, open_=o, above_cap=cap)
    m = simulate(c[sl], a[sl])
    return m


def row(cap, sl):
    m = metr(cap, sl)
    return dict(cap=cap, cagr=m["cagr"], maxdd=m["maxdd"], sharpe=m["sharpe"],
               calmar=m["calmar"], tim=m["time_in_market"], turn=m["turnover"])

full = slice(0, N); h1 = slice(0, half); h2 = slice(half, N)
rf = pd.DataFrame([row(cap, full) for cap in CAPS])
r1 = pd.DataFrame([row(cap, h1) for cap in CAPS])
r2 = pd.DataFrame([row(cap, h2) for cap in CAPS])

# baseline (pas de cap au-dessus de la MA)
def baseline(sl):
    a = compute_allocation(c, nfci, cpi)  # sans OHLC -> above=1
    return simulate(c[sl], a[sl])
bf = baseline(full)

print("=" * 92)
print(f"PASSE FINE ABOVE_CAP  |  full-sample {dates[0].date()} -> {dates[-1].date()}  (déployé = {ABOVE_CAP})")
print("=" * 92)
print(f"{'cap':>5} | {'CAGR':>6} {'maxDD':>7} {'Sharpe':>6} {'Calmar':>6} {'TiM':>4} {'turn':>4}")
print(f"{'base':>5} | {bf['cagr']*100:5.1f}% {bf['maxdd']*100:6.1f}% {bf['sharpe']:6.2f} "
      f"{bf['calmar']:6.2f} {bf['time_in_market']*100:3.0f}% {bf['turnover']:4.1f}")
print("-" * 92)
best_cal = rf.loc[rf["calmar"].idxmax(), "cap"]
best_dd = rf.loc[rf["maxdd"].idxmax(), "cap"]
for _, x in rf.iterrows():
    star = "  <- déployé" if abs(x["cap"] - ABOVE_CAP) < 1e-9 else ""
    star += "  [max Calmar]" if abs(x["cap"] - best_cal) < 1e-9 else ""
    print(f"{x['cap']:5.2f} | {x['cagr']*100:5.1f}% {x['maxdd']*100:6.1f}% {x['sharpe']:6.2f} "
          f"{x['calmar']:6.2f} {x['tim']*100:3.0f}% {x['turn']:4.1f}{star}")

# ── validation croisée ────────────────────────────────────
print("\n" + "=" * 92)
print("VALIDATION CROISÉE (anti-overfit) : on choisit le cap optimal-Calmar sur une")
print(f"moitié, on mesure sa perf sur l'AUTRE.  césure = {mid_date}")
print("=" * 92)
cap1 = r1.loc[r1["calmar"].idxmax(), "cap"]   # optimal sur moitié 1
cap2 = r2.loc[r2["calmar"].idxmax(), "cap"]   # optimal sur moitié 2
def calmar_at(rdf, cap):
    return float(rdf.loc[np.isclose(rdf["cap"], cap), "calmar"].iloc[0])
print(f"  cap optimal sur 2000-{mid_date.year} (crises dot-com+GFC) = {cap1:.2f}")
print(f"     -> appliqué sur {mid_date.year}-2026 : Calmar {calmar_at(r2, cap1):.2f} "
      f"(optimum in-sample de cette moitié = {r2['calmar'].max():.2f} @ {cap2:.2f})")
print(f"  cap optimal sur {mid_date.year}-2026 (Covid+2022)      = {cap2:.2f}")
print(f"     -> appliqué sur 2000-{mid_date.year} : Calmar {calmar_at(r1, cap2):.2f} "
      f"(optimum in-sample de cette moitié = {r1['calmar'].max():.2f} @ {cap1:.2f})")
print(f"  déployé {ABOVE_CAP} : Calmar {calmar_at(r1, ABOVE_CAP):.2f} (moitié 1) / "
      f"{calmar_at(r2, ABOVE_CAP):.2f} (moitié 2) / {calmar_at(rf, ABOVE_CAP):.2f} (full)")

# ── graphe ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6))
ax.plot(rf["cap"], rf["maxdd"] * 100, "o-", color="crimson", label="maxDD full (%)")
ax.plot(rf["cap"], r1["maxdd"] * 100, "--", color="salmon", lw=1, label=f"maxDD 2000-{mid_date.year}")
ax.plot(rf["cap"], r2["maxdd"] * 100, ":", color="darkred", lw=1, label=f"maxDD {mid_date.year}-2026")
ax.axhline(bf["maxdd"] * 100, color="grey", ls="-", lw=0.8, alpha=0.6, label="maxDD baseline (sans cap)")
ax.set_xlabel("ABOVE_CAP (seuil vol au-dessus de la SMA250)")
ax.set_ylabel("maxDD (%)", color="crimson")
ax.axvline(ABOVE_CAP, color="black", ls="--", lw=1, alpha=0.7)
ax.text(ABOVE_CAP, ax.get_ylim()[0], f" déployé {ABOVE_CAP}", fontsize=9, va="bottom")
ax.grid(True, alpha=0.25)

ax2 = ax.twinx()
ax2.plot(rf["cap"], rf["calmar"], "s-", color="royalblue", label="Calmar full")
ax2.plot(rf["cap"], rf["cagr"] * 100, "^-", color="green", lw=1, alpha=0.7, label="CAGR full (%)")
ax2.set_ylabel("Calmar  /  CAGR (%)", color="royalblue")

l1, la1 = ax.get_legend_handles_labels()
l2, la2 = ax2.get_legend_handles_labels()
ax.legend(l1 + l2, la1 + la2, loc="center right", fontsize=8)
ax.set_title("Passe ABOVE_CAP — arbitrage protection (maxDD) vs rendement (CAGR/Calmar)")
out = ROOT / "myfiles" / "above_cap_sweep.png"
fig.tight_layout(); fig.savefig(out, dpi=140); print(f"\nGraphe -> {out}")
