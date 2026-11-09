"""Reprise PLUS RAPIDE après un choc de vol, mais CONDITIONNELLE au régime (x1.7 PUST+LQQ, frais 0.1 %/côté).

Idée : les reprises rapides se font piéger en 2000-02 / 2008 (rebonds de marché baissier). On n'accélère
donc la reprise que dans un régime « choc dans un marché haussier », connu au close J :
  choc   : max(YZ10 sur 30 séances) > V   (un choc de vol vient d'avoir lieu)
  pente  : SMA250 > SMA250 il y a 40 j     (tendance de fond intacte)
  NFCI<0 : pas de stress financier
Régimes testés : partout (témoin inconditionnel) | choc | choc+pente | choc+pente+NFCI<0 (V = 0.25, 0.30).
Leviers (actifs seulement dans le régime) :
  A vol   : après le pic (YZ10 < son max 10 j), vol = min(YZ10, YZ5) → la vol redescend plus vite
  B bande : seuil d'achat 0.25 au lieu de 0.50 (en expo) → on remonte sans attendre une grosse marche
  C prix  : close = plus haut 20 j (ou 10 j) ET > SMA250 → plafond de vol levé (alloc ≥ decay_up)
  D VIX   : VIX/VIX3M < 0.9 alors qu'il était > 1 dans les 20 j → plafond levé (≥ 0.5 sous la MA) ; 2006+
Garde-fous NFCI > 0.5 / CPI > 7 ré-appliqués après tout levier. exec J+1 (régime de J → seuil de J+1).
Validation : battre l'actuelle sur H1 2000-2012 ET H2 2013-2026 (Calmar) + épisodes.
    python -m myfiles.pust17_fast_reentry
"""
import numpy as np
import pandas as pd

import src.risk_off_strategy.strategy as st
from myfiles.pust17_vol_estimators import EPIS, HALVES, win
from src.risk_off_strategy.data import load_funding, load_macro, load_ohlc
from src.risk_off_strategy.strategy import E_MAX, compute_allocation, simulate_net

st.BUY_FEE = st.SELL_FEE = 0.001
YZ = st.yang_zhang_vol
EPIS = {**EPIS, "2024-08": ("2024-07-10", "2024-08-31")}


def sim(price, alloc, funding, buy_thr_arr, sell_thr=st.SELL_THR_E, e_max=E_MAX):
    """simulate_net(leverage=2) avec un seuil d'achat variable dans le temps (indexé sur le jour de décision)."""
    p = np.asarray(price, float)
    ret = pd.Series(p).pct_change().fillna(0).values
    n = len(p)
    fund = np.asarray(funding, float)
    r_pust = ret - st.TER_PUST / st.ANN
    r_lqq = 2 * ret - (fund + st.SWAP_SPREAD + st.TER_LQQ) / st.ANN
    E = np.clip(2 * np.asarray(alloc, float), 0, e_max)
    Etgt = np.concatenate([[0.0], E[:-1]])
    bthr = np.concatenate([[st.BUY_THR_E], np.asarray(buy_thr_arr, float)[:-1]])
    vp = vl = 0.0
    vc = 1.0
    eq = np.empty(n)
    fees = 0.0
    revis = 0
    for t in range(n):
        vp *= 1 + r_pust[t]
        vl *= 1 + r_lqq[t]
        V = vp + vl + vc
        e_eff = (vp + 2 * vl) / V if V > 0 else 0.0
        et = Etgt[t]
        force = (et == 0.0 and e_eff > 1e-9) or (e_eff == 0.0 and et > 1e-9)
        thr = bthr[t] if et > e_eff else sell_thr
        if abs(et - e_eff) >= thr or force:
            wl = max(0.0, et - 1.0)
            wp = et if et <= 1.0 else 2.0 - et
            tp, tl = wp * V, wl * V
            fee = (st.SELL_FEE * (max(0.0, vp - tp) + max(0.0, vl - tl))
                   + st.BUY_FEE * (max(0.0, tp - vp) + max(0.0, tl - vl)))
            fees += fee / V
            V -= fee
            vp, vl, vc = wp * V, wl * V, max(0.0, 1.0 - et) * V
            revis += 1
        eq[t] = V
    yrs = n / st.ANN
    return eq, fees / yrs * 100, revis / yrs


def main():
    o = load_ohlc("QQQ", "2000-01-01", None)
    p, idx = o["close"], o.index
    c, kw = p.values, dict(high=o.high.values, low=o.low.values, open_=o.open.values)
    nfci, cpi = load_macro(idx)
    fund = np.nan_to_num(np.asarray(load_funding(idx), float))
    yz10 = YZ(o.open.values, o.high.values, o.low.values, c)
    yz5 = YZ(o.open.values, o.high.values, o.low.values, c, w=5)
    s = pd.Series(c).rolling(st.SMA_LONG, min_periods=1).mean().values
    slope = np.ones(len(c), bool)
    slope[st.SLOPE_K:] = s[st.SLOPE_K:] > s[:-st.SLOPE_K]
    gap = c / s - 1
    decay_up = np.clip(1 - np.maximum(0, gap - st.GAP2_START) / st.GAP2_SPAN, st.DECAY2_FLOOR, 1)
    nf = np.nan_to_num(np.asarray(nfci, float), nan=9)
    macro_off = (np.nan_to_num(np.asarray(nfci, float), nan=-9) > st.NFCI_OFF) | \
                (np.nan_to_num(np.asarray(cpi, float), nan=-9) > st.CPI_OFF)
    yz_max30 = pd.Series(yz10).rolling(30, min_periods=1).max().values
    past_peak = yz10 < pd.Series(yz10).rolling(10, min_periods=1).max().values
    hi = {w: c >= pd.Series(c).rolling(w, min_periods=1).max().values for w in (10, 20)}
    vt = pd.read_parquet("data/vix/vix_term.parquet")
    ratio = (vt.vix / vt.vix3m).reindex(idx).ffill(limit=3)
    vix_ok = ((ratio < 0.9) & (ratio.rolling(20, min_periods=1).max() > 1.0)).fillna(False).values

    regimes = {"partout": np.ones(len(c), bool)}
    for V in (0.25, 0.30):
        sh = yz_max30 > V
        regimes[f"choc{V}"] = sh
        regimes[f"choc{V}+pente"] = sh & slope
        regimes[f"choc{V}+pente+NFCI<0"] = sh & slope & (nf < 0)

    base = compute_allocation(c, nfci, cpi, **kw)
    # contrôle : mon simulateur == simulate_net
    e_ref, _, _ = simulate_net(p, base, leverage=2, funding=fund, e_max=E_MAX)
    e_me, _, _ = sim(c, base, fund, np.full(len(c), st.BUY_THR_E))
    assert abs(e_ref[-1] / e_me[-1] - 1) < 1e-9, (e_ref[-1], e_me[-1])

    rows = []

    def run(name, reg, lev, alloc, bthr, on):
        eq, fz, rv = sim(c, alloc, fund, bthr)
        row = {"variante": name, "régime": reg, "levier": lev, "% jours régime": round(100 * on.mean(), 1),
               "frais %/an": round(fz, 2), "revis/an": round(rv, 1)}
        for hn, (a, b) in HALVES.items():
            cg, d_, k = win(eq, idx, a, b)
            row.update({f"{hn} %/an": round(cg, 1), f"{hn} maxDD": round(d_, 1), f"{hn} Calmar": round(k, 2)})
        for k_, (a, b) in EPIS.items():
            m = (idx >= a) & (idx <= b)
            row[k_] = round(100 * (eq[m][-1] / eq[m][0] - 1))
        rows.append(row)

    run("ACTUELLE", "-", "-", base, np.full(len(c), st.BUY_THR_E), np.zeros(len(c), bool))
    levers = ("A vol", "B bande", "C prix20", "C prix10", "D VIX", "A+B", "A+B+C20", "A+B+C20+D")
    for rn, R in regimes.items():
        for lev in levers:
            parts = lev.split(" ")[0].split("+")
            vol = np.where(R & past_peak, np.minimum(yz10, yz5), yz10) if "A" in parts else yz10
            st.yang_zhang_vol = lambda *a, _v=vol, **k: _v
            al = compute_allocation(c, nfci, cpi, **kw)
            st.yang_zhang_vol = YZ
            if "C" in parts or "C20" in parts or lev.startswith("C"):
                w = 10 if "prix10" in lev else 20
                cond = R & hi[w] & (c > s)
                al = np.where(cond, np.maximum(al, decay_up), al)
            if "D" in parts:
                cond = R & vix_ok
                al = np.where(cond, np.maximum(al, np.where(c > s, decay_up, 0.5)), al)
            al = np.where(macro_off, 0.0, al)
            bthr = np.where(R, 0.25, st.BUY_THR_E) if "B" in parts else np.full(len(c), st.BUY_THR_E)
            run(f"{lev} | {rn}", rn, lev, al, bthr, R)
    df = pd.DataFrame(rows)
    df.to_csv("myfiles/pust17_fast_reentry.csv", index=False)
    pd.set_option("display.width", 320)
    pd.set_option("display.max_rows", 200)
    cols = ["variante", "% jours régime", "H1 Calmar", "H2 Calmar", "tout %/an", "tout maxDD", "tout Calmar",
            "frais %/an", "revis/an"] + list(EPIS)
    print(df[cols].to_string(index=False))
    ref = df.iloc[0]
    both = df[(df["H1 Calmar"] > ref["H1 Calmar"]) & (df["H2 Calmar"] > ref["H2 Calmar"])]
    print(f"\n=== Battent l'actuelle sur LES DEUX moitiés : {len(both)} / {len(df) - 1} ===")
    print(both[cols].to_string(index=False) if len(both) else "  aucune")
    print("\n=== Validation croisée par levier (meilleur régime réglé sur une moitié, lu sur l'autre) ===")
    for lev in levers:
        g = df[df.levier == lev]
        for fit, test in (("H1", "H2"), ("H2", "H1")):
            b = g.loc[g[f"{fit} Calmar"].idxmax()]
            print(f"  {lev:11s} réglé {fit} → {b.régime:22s} {test} Calmar {b[f'{test} Calmar']:.2f} (actuelle {ref[f'{test} Calmar']:.2f})"
                  f"  %/an {b[f'{test} %/an']:.1f} ({ref[f'{test} %/an']:.1f})  maxDD {b[f'{test} maxDD']:.1f} ({ref[f'{test} maxDD']:.1f})")


if __name__ == "__main__":
    main()
