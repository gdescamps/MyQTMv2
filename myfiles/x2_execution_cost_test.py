"""
Suite de l'audit x2.0 : (a) ou se loge le maxDD de la x2 ; (b) cout d'execution
REEL : le backtest rebalance en continu sans frais, le live Bourso vend avec
0.5% de frais (achats gratuits) et ne vend que si delta >= SELL_THRESHOLD=0.20.
On simule cette execution discretisee pour mesurer le drag reel et chercher le
seuil optimal.

Usage : python myfiles/x2_execution_cost_test.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.risk_off_strategy.data import load_ohlc, load_macro
from src.risk_off_strategy.strategy import compute_allocation, ANN

ohlc = load_ohlc("QQQ")
o, h, l, c = (ohlc[x].values for x in ["open", "high", "low", "close"])
dates = ohlc.index
nfci, cpi = load_macro(dates)
N = len(c)
ret = pd.Series(c).pct_change().fillna(0).values
a1 = compute_allocation(c, nfci, cpi, high=h, low=l, open_=o)

# ── (a) episodes de drawdown de la x2 sans frais ──────────
pos = np.concatenate([[0.0], (2 * a1)[:-1]])
r = ret * pos
eq = np.cumprod(1 + r)
peak = np.maximum.accumulate(eq)
dd = (eq - peak) / peak
print("=" * 96)
print("(a) TOP-5 EPISODES DE DRAWDOWN — x2 sans frais")
print("=" * 96)
episodes = []
i = 0
while i < N:
    if dd[i] < -0.10:
        j0 = i
        while i < N and dd[i] < 0:
            i += 1
        j1 = j0 + int(np.argmin(dd[j0:i]))
        episodes.append((dd[j1], j0, j1))
    i += 1
for depth, j0, j1 in sorted(episodes)[:5]:
    # expo moyenne pendant l'episode
    e_mean = pos[j0:j1 + 1].mean()
    print(f"   {depth*100:6.1f}%  {dates[j0].date()} -> {dates[j1].date()}  "
          f"(expo moyenne {e_mean:.2f}, {j1-j0} jours)")

# ── (b) execution discretisee avec frais de vente ─────────
def sim_exec(target, sell_thr=0.20, buy_thr=0.0, sell_fee=0.005, exec_lag=1):
    """pos suit target avec bandes : on n'achete que si delta > buy_thr (gratuit),
    on ne vend que si delta >= sell_thr (fee 0.5% sur le notionnel vendu).
    target/pos en unites d'expo [0,2] ; le notionnel LQQ echange = delta/2."""
    tgt = np.concatenate([np.zeros(exec_lag), target[:-exec_lag]])
    pos_, fees = np.zeros(N), np.zeros(N)
    cur = 0.0
    trades_b = trades_s = 0
    for t in range(N):
        d = tgt[t] - cur
        if d > buy_thr:
            cur = tgt[t]; trades_b += 1
        elif d < -sell_thr:
            fees[t] = (-d / 2) * sell_fee   # notionnel LQQ vendu = delta_expo/2
            cur = tgt[t]; trades_s += 1
        pos_[t] = cur
    rr = ret * pos_ - fees
    e = np.cumprod(1 + rr)
    yrs = N / ANN
    pk = np.maximum.accumulate(e)
    mdd = float(((e - pk) / pk).min())
    cagr = e[-1] ** (1 / yrs) - 1
    return dict(cagr=cagr, maxdd=mdd, sharpe=rr.mean() / rr.std() * np.sqrt(ANN),
                calmar=cagr / abs(mdd), fees_yr=fees.sum() / yrs * 100,
                b=trades_b / yrs, s=trades_s / yrs)

target2 = 2 * a1
print()
print("=" * 96)
print("(b) EXECUTION DISCRETISEE x2 (achat gratuit, vente 0.5% du notionnel vendu)")
print("    seuil de vente = delta d'EXPO (le live utilise delta d'alloc x1 = expo/2)")
print("=" * 96)
base = sim_exec(target2, sell_thr=-1e-9, buy_thr=-1e-9, sell_fee=0.0)
print(f"{'seuils (buy/sell)':<26} | {'CAGR':>6} {'maxDD':>7} {'Shrp':>5} {'Calm':>5} "
      f"{'fees/an':>7} {'achats/an':>9} {'ventes/an':>9}")
print("-" * 96)
print(f"{'continu sans frais (ref)':<26} | {base['cagr']*100:5.1f}% {base['maxdd']*100:6.1f}% "
      f"{base['sharpe']:5.2f} {base['calmar']:5.2f} {base['fees_yr']:6.2f}% {base['b']:9.1f} {base['s']:9.1f}")
for bt, st in [(0.0, 0.0), (0.0, 0.10), (0.0, 0.20), (0.0, 0.30), (0.0, 0.40),
               (0.05, 0.20), (0.10, 0.20), (0.05, 0.40), (0.10, 0.40)]:
    m = sim_exec(target2, sell_thr=st, buy_thr=bt)
    cur_tag = "  <- ~live actuel" if (bt, st) == (0.0, 0.40) else ""
    print(f"buy>{bt:.2f} / sell>={st:.2f}      | {m['cagr']*100:5.1f}% {m['maxdd']*100:6.1f}% "
          f"{m['sharpe']:5.2f} {m['calmar']:5.2f} {m['fees_yr']:6.2f}% {m['b']:9.1f} {m['s']:9.1f}{cur_tag}")
print("\n   NB: SELL_THRESHOLD live = 0.20 en alloc x1 = 0.40 en expo x2.")
