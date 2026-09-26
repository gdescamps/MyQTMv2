"""
RSI(14) journalier dans la strategie deployee : deux overlays testes separement
puis combines (prompts.txt) :

  A. DESALLOCATION sur-achat persistant : si le RSI reste > RSI_HI (~75) trop
     longtemps (>= D jours sur les W derniers), on multiplie le bras "above" par F.
  B. RE-ALLOCATION RAPIDE sous la SMA250 : prix < MA, vol en baisse (YZ < YZ il y a
     k jours) et RSI bas (min du RSI sur L jours < RSI_LO) -> on relache le
     decay_down (le bras below n'est plus coupe par l'enfoncement sous la MA) ou on
     applique le bras above (cap-vol ABOVE_CAP) a la place.

Mesure : backtest NET (simulate_net, bande asymetrique 0.25/0.50) x1 et x2,
historique complet + 2 moities (robustesse, comme les autres calibrations).

    python -m myfiles.rsi_overlay_test          # depuis la racine, venv actif
"""
import itertools

import numpy as np
import pandas as pd

from src.risk_off_strategy.data import load_ohlc, load_macro, load_funding
from src.risk_off_strategy.strategy import (
    compute_allocation, yang_zhang_vol, simulate_net, ANN,
    SMA_LONG, SLOPE_K, ABOVE_CAP, VOL_TARGET, BELOW_SCALE, GAP_CUTOFF,
    GAP2_START, GAP2_SPAN, DECAY2_FLOOR, NFCI_OFF, CPI_OFF,
)

ohlc = load_ohlc("QQQ")
price = ohlc["close"]
o, h, l, c = (ohlc[x].values for x in ["open", "high", "low", "close"])
nfci, cpi = load_macro(price.index)
funding = load_funding(price.index)
N = len(c)
MID = N // 2


def rsi_wilder(p, n=14):
    """RSI de Wilder (lissage exponentiel alpha=1/n), causal."""
    d = pd.Series(p).diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return (100 - 100 / (1 + up / dn.replace(0, np.nan))).fillna(50).values


# ── Briques de la strategie actuelle (identiques a compute_allocation) ──
s = pd.Series(c).rolling(SMA_LONG, min_periods=1).mean().values
slope_up = np.ones(N, bool)
slope_up[SLOPE_K:] = s[SLOPE_K:] > s[:-SLOPE_K]
gap = c / s - 1.0
decay_down = np.clip(1.0 + gap / GAP_CUTOFF, 0, 1)
decay_up = np.clip(1.0 - np.maximum(0.0, gap - GAP2_START) / GAP2_SPAN, DECAY2_FLOOR, 1.0)
rv = yang_zhang_vol(o, h, l, c)
above_arm = np.clip(ABOVE_CAP / rv, 0, 1) * decay_up
vm = np.clip(VOL_TARGET / rv, 0, 1)
below_arm = BELOW_SCALE * vm * decay_down
regime = (c > s) & slope_up
macro_off = ((np.nan_to_num(np.asarray(nfci, float), nan=-9) > NFCI_OFF)
             | (np.nan_to_num(np.asarray(cpi, float), nan=-9) > CPI_OFF))
rsi = rsi_wilder(c)

base = np.where(macro_off, 0.0, np.clip(np.where(regime, above_arm, below_arm), 0, 1))
ref = compute_allocation(c, nfci, cpi, high=h, low=l, open_=o)
assert np.allclose(base, ref), "la reconstruction diverge de compute_allocation"


def roll_count(mask, w):
    return pd.Series(mask.astype(float)).rolling(w, min_periods=1).sum().values


def overlay_a(thr, w, d, f):
    """Facteur multiplicatif du bras above : F si >= d jours RSI>thr sur w jours."""
    return np.where(roll_count(rsi > thr, w) >= d, f, 1.0)


def overlay_b(lo, look, k, mode):
    """Bras below de remplacement quand (RSI bas recent) & (vol en baisse)."""
    rsi_min = pd.Series(rsi).rolling(look, min_periods=1).min().values
    vol_down = np.concatenate([np.zeros(k, bool), rv[k:] < rv[:-k]])
    cond = (rsi_min < lo) & vol_down
    alt = BELOW_SCALE * vm if mode == "nodecay" else np.clip(ABOVE_CAP / rv, 0, 1)
    return cond, np.where(cond, np.maximum(below_arm, alt), below_arm)


def assemble(fa=None, below=None):
    a_arm = above_arm if fa is None else above_arm * fa
    b_arm = below_arm if below is None else below
    return np.where(macro_off, 0.0, np.clip(np.where(regime, a_arm, b_arm), 0, 1))


def metrics(eq):
    yrs = len(eq) / ANN
    pk = np.maximum.accumulate(eq)
    dd = float(((eq - pk) / pk).min())
    cagr = eq[-1] ** (1 / yrs) - 1
    r = eq[1:] / eq[:-1] - 1
    return cagr, dd, r.mean() / r.std() * np.sqrt(ANN), cagr / abs(dd)


def evaluate(alloc):
    out = {}
    for lev in (1, 2):
        for tag, sl in (("F", slice(0, N)), ("H1", slice(0, MID)), ("H2", slice(MID, N))):
            eq, fees, rev = simulate_net(price.iloc[sl], alloc[sl], leverage=lev,
                                         funding=funding[sl])
            cagr, dd, sh, cal = metrics(eq)
            out[(lev, tag)] = dict(cagr=cagr, dd=dd, sh=sh, cal=cal, fees=fees, rev=rev)
    return out


HDR = (f"{'variante':<34} | {'x1 CAGR':>7} {'maxDD':>6} {'Shrp':>5} {'Calm':>5} {'rev':>4} "
       f"{'CalH1':>5} {'CalH2':>5} | {'x2 CAGR':>7} {'maxDD':>6} {'Calm':>5} {'CalH1':>5} {'CalH2':>5}")


def line(name, m):
    a, a1, a2 = m[(1, "F")], m[(1, "H1")], m[(1, "H2")]
    b, b1, b2 = m[(2, "F")], m[(2, "H1")], m[(2, "H2")]
    print(f"{name:<34} | {a['cagr']*100:6.1f}% {a['dd']*100:6.1f}% {a['sh']:5.2f} {a['cal']:5.2f} "
          f"{a['rev']:4.1f} {a1['cal']:5.2f} {a2['cal']:5.2f} | {b['cagr']*100:6.1f}% "
          f"{b['dd']*100:6.1f}% {b['cal']:5.2f} {b1['cal']:5.2f} {b2['cal']:5.2f}")


def score(m, r):
    """Gain robuste : pire delta de Calmar net sur (x1, x2) x (H1, H2)."""
    return min(m[(lv, t)]["cal"] - r[(lv, t)]["cal"] for lv in (1, 2) for t in ("H1", "H2"))


def sweep(title, variants, ref_m, top=12):
    print("\n" + "=" * len(HDR)); print(title); print("=" * len(HDR)); print(HDR)
    print("-" * len(HDR))
    line("actuelle (ref)", ref_m)
    res = sorted(((score(m, ref_m), name, m) for name, m in variants), key=lambda x: -x[0])
    for sc, name, m in res[:top]:
        line(f"{name} [{sc:+.2f}]", m)
    return res


if __name__ == "__main__":
    ref_m = evaluate(base)
    print(f"RSI14 QQQ : %jours >70 = {(rsi > 70).mean()*100:.1f}%, >75 = {(rsi > 75).mean()*100:.1f}%, "
          f"<30 = {(rsi < 30).mean()*100:.1f}%")
    print(f"Dernier RSI ({price.index[-1].date()}) = {rsi[-1]:.1f}")

    va = []
    for thr, w, d, f in itertools.product((70, 75, 80), (10, 20, 40), (3, 5, 10, 15), (0.5, 0.7, 0.85)):
        if d <= w:
            va.append((f"A rsi>{thr} {d}/{w}j x{f}", evaluate(assemble(fa=overlay_a(thr, w, d, f)))))
    res_a = sweep("A. DESALLOCATION sur-achat persistant (bras above x F)", va, ref_m)

    vb = []
    for lo, look, k, mode in itertools.product((30, 35, 40, 45), (1, 5, 10, 20), (5, 10), ("nodecay", "above")):
        cond, b = overlay_b(lo, look, k, mode)
        vb.append((f"B rsi<{lo} L{look} k{k} {mode}", evaluate(assemble(below=b))))
    res_b = sweep("B. RE-ALLOCATION rapide sous MA (RSI bas + vol en baisse)", vb, ref_m)

    # Combo : meilleur A x meilleurs B (par nom -> reconstruit)
    print("\n" + "=" * len(HDR)); print("COMBOS A + B (3 meilleurs de chaque)"); print("=" * len(HDR))
    print(HDR); print("-" * len(HDR)); line("actuelle (ref)", ref_m)

    def parse_a(n):
        t = n.split()
        return int(t[1][4:]), int(t[2].split("/")[1][:-1]), int(t[2].split("/")[0]), float(t[3][1:])

    def parse_b(n):
        t = n.split()
        return int(t[1][4:]), int(t[2][1:]), int(t[3][1:]), t[4]

    for (_, na, _), (_, nb, _) in itertools.product(res_a[:3], res_b[:3]):
        _, b = overlay_b(*parse_b(nb))
        m = evaluate(assemble(fa=overlay_a(*parse_a(na)), below=b))
        line(f"{na[2:]} + {nb[2:]}"[:34] + f" [{score(m, ref_m):+.2f}]", m)
