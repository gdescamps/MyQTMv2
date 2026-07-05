"""
Optimiseur stratégie x2 LQQ NET DE FRAIS.

Objectifs (cf. demande) :
  1. optimiser la x2 avec LQQ
  2. prendre en compte les frais REELS (drag de financement LQQ + TER + fee de vente)
  3. utiliser PUST ou LQQ pour limiter les frais (PUST=base, LQQ=part >100%, €STR=cash)
  4. afficher la x2 NETTE de frais (chart + metriques)
  5. limiter le nombre de revisions d'allocation (bande de non-action + comptage)

Modele des instruments (espace QQQ-USD, cf. convention du backtest du repo ;
le vrai LQQ/PUST subissent fuseau Paris + EURUSD, non calibrables proprement) :
  r_pust = r_qqq − TER_PUST/252                         (TER 0.30%)
  r_lqq  = 2·r_qqq − (DFF + SWAP_SPREAD + TER_LQQ)/252  (levier 2x finance au taux court)
  r_cash = DFF/252                                       (€STR ~ overnight, proxy USD)

Realisation drag-minimale d'une exposition cible E ∈ [0, e_max] :
  E ≤ 1 :  PUST=E,      LQQ=0,    cash=1−E     (aucun drag de levier)
  E > 1 :  PUST=2−E,    LQQ=E−1,  cash=0       (LQQ ne porte QUE la part >100%)

Usage : python myfiles/x2_lqq_optimize.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.risk_off_strategy.data import load_ohlc, load_macro
from src.risk_off_strategy.strategy import (
    compute_allocation, yang_zhang_vol, SMA_LONG, ABOVE_CAP, VOL_TARGET,
    BELOW_SCALE, GAP_CUTOFF, GAP2_START, GAP2_SPAN, DECAY2_FLOOR,
    NFCI_OFF, CPI_OFF, SLOPE_K, ANN,
)

TER_LQQ, TER_PUST, SWAP_SPREAD, SELL_FEE = 0.0060, 0.0030, 0.0040, 0.005
RATES_CACHE = ROOT / "myfiles" / "x2_rates.parquet"
OUT_PNG = ROOT / "myfiles" / "x2_lqq_net.png"

# ── Donnees ───────────────────────────────────────────────
ohlc = load_ohlc("QQQ")
o, h, l, c = (ohlc[x].values for x in ["open", "high", "low", "close"])
dates = ohlc.index
N = len(c)
half = N // 2
nfci, cpi = load_macro(dates)
rq = pd.Series(c).pct_change().fillna(0).values

dff = pd.read_parquet(RATES_CACHE)["DFF"]
dff = dff.reindex(dates.union(dff.index)).sort_index().ffill().reindex(dates).bfill().values

# rendements synthetiques des instruments (net de leurs frais internes)
r_pust = rq - TER_PUST / ANN
r_lqq = 2 * rq - (dff + SWAP_SPREAD + TER_LQQ) / ANN
r_cash = dff / ANN

# ── Composantes du signal (communes) ──────────────────────
s = pd.Series(c).rolling(SMA_LONG, min_periods=1).mean().values
gap = c / s - 1.0
rv = yang_zhang_vol(o, h, l, c)
decay_down = np.clip(1.0 + gap / GAP_CUTOFF, 0, 1)
decay_up = np.clip(1.0 - np.maximum(0.0, gap - GAP2_START) / GAP2_SPAN, DECAY2_FLOOR, 1.0)
macro_off = ((np.nan_to_num(np.asarray(nfci, float), nan=-9) > NFCI_OFF)
             | (np.nan_to_num(np.asarray(cpi, float), nan=-9) > CPI_OFF))
slope_up = np.concatenate([[True] * SLOPE_K, s[SLOPE_K:] > s[:-SLOPE_K]])
above_regime = (c > s) & slope_up
below_arm = BELOW_SCALE * np.clip(VOL_TARGET / rv, 0, 1) * decay_down   # ≤0.8, PUST/cash


def exposure_x2(above_cap, e_max):
    """Exposition cible E ∈ [0, e_max] : le levier n'est deploye QUE dans le
    regime de tendance confirmee (au-dessus MA + pente montante + vol basse).
    Le bras below reste ≤1 (defensif, jamais de LQQ) ; garde-fous macro -> 0."""
    above = np.clip(above_cap / rv, 0, e_max) * decay_up
    E = np.where(above_regime, above, below_arm)
    return np.where(macro_off, 0.0, E)


def realize(E):
    """Poids drag-minimaux (pust, lqq, cash) pour l'exposition E."""
    w_lqq = np.maximum(0.0, E - 1.0)
    w_pust = np.where(E <= 1.0, E, 2.0 - E)
    w_cash = np.maximum(0.0, 1.0 - E)
    return w_pust, w_lqq, w_cash


def sim_x2(E, band, naive=False):
    """Simulation NET de frais avec bande de non-action (limite les revisions).
    naive=True : realisation tout-LQQ (w_lqq=E/2) -> montre le drag economise
    par l'astuce PUST-base. Renvoie metriques + series (equity, expo, poids)."""
    Etgt = np.concatenate([[0.0], E[:-1]])            # exec_lag=1
    vp = vl = vc = 0.0
    V = 1.0
    vc = 1.0                                          # depart 100% cash
    eq = np.empty(N)
    e_eff_s = np.empty(N)
    wser = np.empty((N, 3))
    fees_tot = 0.0
    revis = 0
    for t in range(N):
        vp *= (1 + r_pust[t]); vl *= (1 + r_lqq[t]); vc *= (1 + r_cash[t])
        V = vp + vl + vc
        e_eff = (vp + 2 * vl) / V if V > 0 else 0.0
        et = Etgt[t]
        force = (et == 0.0 and e_eff > 1e-9) or (e_eff == 0.0 and et > 1e-9)
        if abs(et - e_eff) >= band or force:
            if naive:
                tp, tl, tc = 0.0, et / 2.0, 1.0 - et / 2.0
            else:
                wl = max(0.0, et - 1.0); wp = et if et <= 1 else 2.0 - et
                tp, tl, tc = wp, wl, max(0.0, 1.0 - et)
            np_, nl_, nc_ = tp * V, tl * V, tc * V
            fee = SELL_FEE * (max(0, vp - np_) + max(0, vl - nl_))   # ventes seulement
            fees_tot += fee / V                                      # fraction du portefeuille (comparable)
            V -= fee
            vp, vl, vc = tp * V, tl * V, tc * V
            revis += 1
        eq[t] = V
        e_eff_s[t] = (vp + 2 * vl) / V if V > 0 else 0.0
        wser[t] = (vp / V, vl / V, vc / V) if V > 0 else (0, 0, 1)
    return _metrics(eq, fees_tot, revis), eq, e_eff_s, wser


def sim_x1(band=0.20):
    """Baseline x1 deployee NET (PUST + cash, seuil de vente = band en alloc)."""
    a1 = compute_allocation(c, nfci, cpi, high=h, low=l, open_=o)
    Atgt = np.concatenate([[0.0], a1[:-1]])
    vp = 0.0; vc = 1.0
    eq = np.empty(N); fees_tot = 0.0; revis = 0
    for t in range(N):
        vp *= (1 + r_pust[t]); vc *= (1 + r_cash[t])
        V = vp + vc
        w = vp / V if V > 0 else 0.0
        at = Atgt[t]
        force = (at == 0.0 and w > 1e-9)
        if abs(at - w) >= band or force:
            np_ = at * V
            fee = SELL_FEE * max(0, vp - np_)
            fees_tot += fee / V; V -= fee
            vp, vc = at * V, (1 - at) * V
            revis += 1
        eq[t] = V
    return _metrics(eq, fees_tot, revis), eq


def _metrics(eq, fees_tot, revis, sl=None):
    e = eq if sl is None else eq[sl] / eq[sl.start if sl.start else 0]
    yrs = len(e) / ANN
    r = np.diff(np.concatenate([[e[0]], e])) / np.concatenate([[e[0]], e])[:-1]
    pk = np.maximum.accumulate(e)
    dd = float(((e - pk) / pk).min())
    cagr = e[-1] ** (1 / yrs) - 1
    return dict(cagr=cagr, maxdd=dd, sharpe=r.mean() / r.std() * np.sqrt(ANN) if r.std() else 0,
                calmar=cagr / abs(dd) if dd < 0 else np.inf,
                fees_yr=fees_tot / yrs * 100, revis_yr=revis / yrs)


def halves_calmar(eq):
    def cal(sl):
        e = eq[sl] / eq[sl][0]
        yrs = len(e) / ANN
        pk = np.maximum.accumulate(e)
        dd = ((e - pk) / pk).min()
        return (e[-1] ** (1 / yrs) - 1) / abs(dd) if dd < 0 else np.inf
    return cal(slice(0, half)), cal(slice(half, N))


# ── Baseline x1 net ───────────────────────────────────────
print("=" * 100)
print(f"BASELINE x1 (PUST) NET DE FRAIS  —  {dates[0].date()} -> {dates[-1].date()}  "
      f"(DFF moyen {dff.mean()*100:.2f}%/an)")
print("=" * 100)
m1, eq1 = sim_x1(band=0.20)
h1a, h1b = halves_calmar(eq1)
print(f"  CAGR {m1['cagr']*100:5.1f}%  maxDD {m1['maxdd']*100:6.1f}%  Sharpe {m1['sharpe']:.2f}  "
      f"Calmar {m1['calmar']:.2f}  (H1 {h1a:.2f} / H2 {h1b:.2f})  fees {m1['fees_yr']:.2f}%/an  "
      f"revis {m1['revis_yr']:.0f}/an")

# ── Grille x2 ─────────────────────────────────────────────
print("\n" + "=" * 100)
print("OPTIMISATION x2 NET (realisation PUST-base + LQQ-top)  — trie par Calmar net")
print("=" * 100)
print(f"{'above_cap':>9} {'e_max':>6} {'band':>5} | {'CAGR':>6} {'maxDD':>7} {'Shrp':>5} "
      f"{'Calm':>5} | {'CalH1':>5} {'CalH2':>5} | {'fees/an':>7} {'revis/an':>8}")
print("-" * 100)
results = []
for ac in (0.22, 0.26, 0.30, 0.34):
    for emax in (1.3, 1.5, 1.7, 2.0):
        for band in (0.15, 0.25, 0.40):
            E = exposure_x2(ac, emax)
            m, eq, _, _ = sim_x2(E, band)
            ca, cb = halves_calmar(eq)
            results.append((m["calmar"], ac, emax, band, m, ca, cb, eq))
results.sort(key=lambda x: -x[0])
for cal, ac, emax, band, m, ca, cb, _ in results[:14]:
    print(f"{ac:9.2f} {emax:6.1f} {band:5.2f} | {m['cagr']*100:5.1f}% {m['maxdd']*100:6.1f}% "
          f"{m['sharpe']:5.2f} {m['calmar']:5.2f} | {ca:5.2f} {cb:5.2f} | "
          f"{m['fees_yr']:6.2f}% {m['revis_yr']:7.0f}")

# ── Choix : meilleur Calmar net robuste (H1 & H2 > 0.5), revisions raisonnables ──
robust = [r for r in results if r[5] > 0.5 and r[6] > 0.5]
best = robust[0] if robust else results[0]
_, ac, emax, band, mB, caB, cbB, eqB = best
E_best = exposure_x2(ac, emax)
mB, eqB, eeff, wser = sim_x2(E_best, band)
mN, eqN, _, _ = sim_x2(E_best, band, naive=True)     # meme signal, realisation tout-LQQ

print("\n" + "=" * 100)
print(f"RETENU : above_cap={ac}  e_max={emax}  band={band}")
print("=" * 100)
print(f"  x1 PUST net        : CAGR {m1['cagr']*100:5.1f}%  maxDD {m1['maxdd']*100:6.1f}%  "
      f"Calmar {m1['calmar']:.2f}  revis {m1['revis_yr']:.0f}/an  fees {m1['fees_yr']:.2f}%/an")
print(f"  x2 PUST-base+LQQ   : CAGR {mB['cagr']*100:5.1f}%  maxDD {mB['maxdd']*100:6.1f}%  "
      f"Calmar {mB['calmar']:.2f}  revis {mB['revis_yr']:.0f}/an  fees {mB['fees_yr']:.2f}%/an")
print(f"  x2 tout-LQQ (naif) : CAGR {mN['cagr']*100:5.1f}%  maxDD {mN['maxdd']*100:6.1f}%  "
      f"Calmar {mN['calmar']:.2f}")
print(f"    -> l'astuce PUST-base ameliore surtout le RISQUE : maxDD {mN['maxdd']*100:.1f}% -> "
      f"{mB['maxdd']*100:.1f}%, Calmar {mN['calmar']:.2f} -> {mB['calmar']:.2f} (a exposition egale)")

print("\n  Arbitrage 'limiter les revisions' (meme signal above_cap={}, e_max={}) :".format(ac, emax))
print(f"    {'band':>5} | {'revis/an':>8} {'fees/an':>7} | {'CAGR':>6} {'maxDD':>7} {'Calmar':>6}")
for band_f in (0.05, 0.10, 0.15, 0.25, 0.40, 0.60):
    mf, eqf, _, _ = sim_x2(E_best, band_f)
    print(f"    {band_f:5.2f} | {mf['revis_yr']:8.0f} {mf['fees_yr']:6.2f}% | "
          f"{mf['cagr']*100:5.1f}% {mf['maxdd']*100:6.1f}% {mf['calmar']:6.2f}")

# ── Chart ─────────────────────────────────────────────────
d = dates
fig, ax = plt.subplots(4, 1, figsize=(13, 13), sharex=True,
                       gridspec_kw={"height_ratios": [3, 1.1, 1.1, 1.3]})
ax[0].semilogy(d, eq1, label=f"x1 PUST net (Calmar {m1['calmar']:.2f})", color="steelblue", lw=1.3)
ax[0].semilogy(d, eqB, label=f"x2 PUST+LQQ net (Calmar {mB['calmar']:.2f})", color="crimson", lw=1.3)
ax[0].semilogy(d, eqN, label=f"x2 tout-LQQ net (Calmar {mN['calmar']:.2f})", color="gray", lw=1.0, ls="--")
ax[0].set_title(f"Strategie x2 LQQ nette de frais — above_cap={ac}, e_max={emax}, band={band}  "
                f"(DFF moyen {dff.mean()*100:.1f}%/an, spread {SWAP_SPREAD*100:.1f}%, TER LQQ {TER_LQQ*100:.1f}%)",
                fontsize=11)
ax[0].legend(loc="upper left", fontsize=9); ax[0].grid(alpha=0.3, which="both")
ax[0].set_ylabel("equity (log)")

ax[1].plot(d, eeff, color="crimson", lw=0.8)
ax[1].axhline(1.0, color="k", ls=":", lw=0.8); ax[1].axhline(2.0, color="gray", ls=":", lw=0.6)
ax[1].set_ylabel("expo x2\neffective"); ax[1].grid(alpha=0.3)

ax[2].stackplot(d, wser[:, 0], wser[:, 1], wser[:, 2],
                labels=["PUST", "LQQ", "cash €STR"], colors=["steelblue", "crimson", "lightgray"], alpha=0.8)
ax[2].legend(loc="upper left", fontsize=8, ncol=3); ax[2].set_ylabel("poids"); ax[2].set_ylim(0, 1)

for eq, col, lab in [(eq1, "steelblue", "x1"), (eqB, "crimson", "x2")]:
    pk = np.maximum.accumulate(eq)
    ax[3].fill_between(d, (eq - pk) / pk * 100, 0, color=col, alpha=0.35, label=lab)
ax[3].legend(loc="lower left", fontsize=8); ax[3].set_ylabel("drawdown %"); ax[3].grid(alpha=0.3)
ax[3].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

fig.tight_layout()
fig.savefig(OUT_PNG, dpi=130, bbox_inches="tight")
print(f"\n  chart -> {OUT_PNG}")
