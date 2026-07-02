"""
Audit look-ahead complet du backtest risk-off (post ajout Yang-Zhang + cap précoce).

Principe : une stratégie est causale ssi la décision au jour t ne dépend QUE des
données <= t. On le prouve de plusieurs façons indépendantes.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.risk_off_strategy.data import load_ohlc, load_macro, NFCI_LAG, CPI_LAG
from src.risk_off_strategy.strategy import (
    compute_allocation, yang_zhang_vol, realized_vol, simulate,
)

ohlc = load_ohlc("QQQ")
o, h, l, c = (ohlc[x].values for x in ["open", "high", "low", "close"])
dates = ohlc.index
nfci, cpi = load_macro(dates)
N = len(c)
full = compute_allocation(c, nfci, cpi, high=h, low=l, open_=o)

ok = True

# ── Test 1 : équivalence batch vs incrémental (LE test décisif) ───────────
# Pour chaque T, l'alloc recalculée sur data[:T] doit être IDENTIQUE à l'alloc
# full sur [:T]. Si une seule valeur future fuitait, ça casserait.
print("=" * 88)
print("TEST 1 — batch vs tronqué : compute_allocation(data[:T]) == full[:T] ?")
print("         (grille dense, y compris les 300 derniers jours = zone live)")
print("=" * 88)
worst = 0.0
grid = list(range(300, N, 500)) + list(range(N - 300, N))  # dense sur la fin
for T in grid:
    sub = compute_allocation(
        c[:T], None if nfci is None else nfci[:T], None if cpi is None else cpi[:T],
        high=h[:T], low=l[:T], open_=o[:T])
    d = np.abs(sub - full[:T]).max()
    worst = max(worst, d)
print(f"  {len(grid)} points de coupe testés — max |alloc[:T] - full[:T]| = {worst:.2e}")
t1 = worst < 1e-12
ok &= t1
print("  ->", "OK : chaque alloc[t] ne dépend que du passé" if t1 else "!!! FUITE")

# ── Test 2 : perturber le FUTUR ne change pas le PASSÉ ────────────────────
print("\n" + "=" * 88)
print("TEST 2 — on corrompt les données APRÈS t0 ; l'alloc AVANT t0 doit être intacte")
print("=" * 88)
t0 = N - 250
c2, h2, l2, o2 = c.copy(), h.copy(), l.copy(), o.copy()
rng_mult = np.linspace(0.5, 1.5, N - t0)  # bruit déterministe (pas de Random)
for arr in (c2, h2, l2, o2):
    arr[t0:] = arr[t0:] * rng_mult
nfci2 = None if nfci is None else nfci.copy()
cpi2 = None if cpi is None else cpi.copy()
if nfci2 is not None: nfci2[t0:] = nfci2[t0:] + 3.0
if cpi2 is not None: cpi2[t0:] = cpi2[t0:] + 10.0
pert = compute_allocation(c2, nfci2, cpi2, high=h2, low=l2, open_=o2)
d_before = np.abs(pert[:t0] - full[:t0]).max()
d_after = np.abs(pert[t0:] - full[t0:]).max()
print(f"  max diff AVANT t0 (doit être 0)  = {d_before:.2e}")
print(f"  max diff APRÈS t0 (doit être !=0) = {d_after:.2e}  (confirme que la perturbation agit)")
t2 = d_before < 1e-12 and d_after > 0
ok &= t2
print("  ->", "OK : le futur n'influence pas le passé" if t2 else "!!! FUITE")

# ── Test 3 : estimateurs de vol causaux (isolé) ──────────────────────────
print("\n" + "=" * 88)
print("TEST 3 — vol Yang-Zhang & realized_vol : v(data[:T])[-k:] == v(full)[T-k:T] ?")
print("=" * 88)
vf = yang_zhang_vol(o, h, l, c)
rf = realized_vol(pd.Series(c).pct_change().fillna(0).values)
wy = wr = 0.0
for T in range(1000, N, 800):
    wy = max(wy, np.abs(yang_zhang_vol(o[:T], h[:T], l[:T], c[:T])[-20:] - vf[T-20:T]).max())
    wr = max(wr, np.abs(realized_vol(pd.Series(c[:T]).pct_change().fillna(0).values)[-20:] - rf[T-20:T]).max())
t3 = wy < 1e-12 and wr < 1e-12
ok &= t3
print(f"  Yang-Zhang max diff = {wy:.2e} | realized_vol max diff = {wr:.2e}")
print("  ->", "OK : vols causales" if t3 else "!!! FUITE")

# ── Test 4 : décalage macro = vers le PASSÉ (publication lag) ─────────────
print("\n" + "=" * 88)
print("TEST 4 — macro décalée du lag de publication (on voit la donnée PLUS VIEILLE)")
print("=" * 88)
raw_nfci = pd.read_parquet(ROOT / "data" / "fred_nfci.parquet")["nfci"]
al = raw_nfci.reindex(dates.union(raw_nfci.index)).sort_index().ffill().reindex(dates)
# nfci[t] (utilisé) doit égaler al décalé de NFCI_LAG jours ouvrés (valeur passée)
expected = al.shift(NFCI_LAG).values
d = np.nanmax(np.abs(np.nan_to_num(nfci) - np.nan_to_num(expected)))
t4 = d < 1e-12
ok &= t4
print(f"  NFCI utilisé == NFCI(t - {NFCI_LAG}j ouvrés) : max diff = {d:.2e}  (IPC lag={CPI_LAG}j même schéma)")
print("  -> shift positif = décalage vers le passé =",
      "OK (conservateur, pas de fuite)" if t4 else "!!! shift inversé = FUITE")

# ── Test 5 : simulate applique bien pos[t] = alloc[t-1] (exec_lag=1) ──────
print("\n" + "=" * 88)
print("TEST 5 — simulate : position du jour t = alloc[t-1] (aucun rendement du jour de décision)")
print("=" * 88)
# reconstruit pos depuis simulate et compare au shift manuel
a = full.copy()
ret = pd.Series(c).pct_change().fillna(0).values
pos_manual = np.concatenate([[0.0], a[:-1]])          # pos[t]=alloc[t-1]
# vérifie que le rendement dépend de alloc DÉCALÉE : corrompre alloc[t] seul ne doit
# pas changer r[t], seulement r[t+1]
m_full = simulate(c, a)
eq_full = m_full["equity"]
a_pert = a.copy(); a_pert[t0] = 0.0                    # coupe l'alloc au jour t0
eq_pert = simulate(c, a_pert)["equity"]
change_at_t0 = abs(eq_pert[t0] / eq_full[t0] - 1)      # doit être ~0 (alloc[t0] agit à t0+1)
change_at_t0p1 = abs(eq_pert[t0 + 1] / eq_full[t0 + 1] - 1)
t5 = change_at_t0 < 1e-12 and change_at_t0p1 > 0
ok &= t5
print(f"  couper alloc[t0] change equity[t0]   : {change_at_t0:.2e}  (doit être 0)")
print(f"  couper alloc[t0] change equity[t0+1] : {change_at_t0p1:.2e}  (doit être !=0)")
print("  ->", "OK : décision à t exécutée à t+1" if t5 else "!!! exécution le jour même = FUITE")

print("\n" + "=" * 88)
print("RÉSULTAT GLOBAL :", "✅ AUCUN LOOK-AHEAD" if ok else "❌ FUITE DÉTECTÉE")
print("=" * 88)
