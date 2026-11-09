"""Overlay « coupe-circuit news » sur la stratégie déployée PUST+LQQ x1.7.

Base : allocation de src/risk_off_strategy (compute_allocation OHLC + garde-fous macro), réalisée
comme simulate_net(leverage=2, e_max=E_MAX) : bande asymétrique 0.50/1.00 en expo, frais 0.5 %
achat + 0.5 % vente, exec_lag=1 (close J → exécution J+1).

Overlay : un tweet DeltaOne classé par Haiku (data/text/x/deltaone/classified.parquet) avec
mover=True, sens=-1, confiance >= CONF, publié avant la clôture US de J (sinon rattaché à la
séance suivante), ET une vol Yang-Zhang du QQQ au close de J >= VOL (« volat déjà élevée ») →
plafond d'exposition CAP (0 = cash, 1 = PUST seul, LQQ vendu) pendant K séances, appliqué de
force (hors bande) au même rythme que la stratégie (décision au close J, exécution J+1).
Un nouveau signal pendant la coupure la prolonge.

Évaluation sur 2023-01 → fin (période où les tweets existent), et hors échantillon Haiku
(2026-07 → fin, coupure des connaissances du modèle = juin 2026).
"""
import itertools

import numpy as np
import pandas as pd

from src.risk_off_strategy.data import load_funding, load_macro, load_ohlc
from src.risk_off_strategy.strategy import (ANN, BUY_FEE, BUY_THR_ALLOC, E_MAX, SELL_FEE, SELL_THR_ALLOC,
                                            SWAP_SPREAD, TER_LQQ, TER_PUST, compute_allocation,
                                            yang_zhang_vol)

LEV = 2.0
CONFS = [50, 60, 70]
VOLS = [0.0, 0.20, 0.25, 0.30]
KS = [3, 5, 10]
CAPS = [1.0, 0.0]
EVAL_START = "2023-01-03"
OOS_START = "2026-07-01"


def simulate(ret, alloc, fund, cap=None):
    """Copie de simulate_net (leverage=2, e_max=E_MAX) + plafond d'expo forcé `cap` (NaN = aucun)."""
    n = len(ret)
    sell_thr, buy_thr = SELL_THR_ALLOC * LEV, BUY_THR_ALLOC * LEV
    r_pust = ret - TER_PUST / ANN
    r_lqq = 2 * ret - (fund + SWAP_SPREAD + TER_LQQ) / ANN
    E = np.clip(LEV * alloc, 0.0, E_MAX)
    if cap is not None:
        E = np.where(np.isnan(cap), E, np.minimum(E, cap))
        capd = np.concatenate([[np.nan], cap[:-1]])
    Etgt = np.concatenate([[0.0], E[:-1]])
    vp = vl = 0.0
    vc = 1.0
    eq, fees = np.empty(n), np.zeros(n)
    for t in range(n):
        vp *= 1 + r_pust[t]
        vl *= 1 + r_lqq[t]
        V = vp + vl + vc
        e_eff = (vp + 2 * vl) / V if V > 0 else 0.0
        et = Etgt[t]
        force = (et == 0.0 and e_eff > 1e-9) or (e_eff == 0.0 and et > 1e-9)
        if cap is not None and not np.isnan(capd[t]) and e_eff > capd[t] + 0.05:
            force = True  # coupe-circuit : on descend sous le plafond hors bande
        thr = buy_thr if et > e_eff else sell_thr
        if abs(et - e_eff) >= thr or force:
            wl = max(0.0, et - 1.0)
            wp = et if et <= 1.0 else 2.0 - et
            tp, tl = wp * V, wl * V
            fee = (SELL_FEE * (max(0.0, vp - tp) + max(0.0, vl - tl))
                   + BUY_FEE * (max(0.0, tp - vp) + max(0.0, tl - vl)))
            fees[t] = fee / V
            V -= fee
            vp, vl, vc = wp * V, wl * V, max(0.0, 1.0 - et) * V
        eq[t] = V
    return eq, fees


def metrics(eq, fees, idx, start, end=None):
    m = (idx >= pd.Timestamp(start)) & ((idx <= pd.Timestamp(end)) if end else True)
    e = eq[m] / eq[m][0]
    r = np.diff(e) / e[:-1]
    yrs = len(e) / ANN
    cagr = e[-1] ** (1 / yrs) - 1
    dd = (e / np.maximum.accumulate(e) - 1).min()
    sh = r.mean() / r.std() * np.sqrt(ANN)
    return {"%/an": 100 * cagr, "maxDD %": 100 * dd, "Sharpe": sh, "Calmar": cagr / -dd if dd < 0 else np.nan,
            "frais %/an": 100 * fees[m].sum() / yrs}


def signals(idx_close_utc, conf):
    """Séance (indice) à laquelle chaque tweet baissier mover est connu au close."""
    df = pd.read_parquet("data/text/x/deltaone/classified.parquet")
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    s = df[(df.mover == True) & (df.sens == -1) & (df.confiance >= conf)]  # noqa: E712
    pos = np.searchsorted(idx_close_utc.values, s.created_at.values)  # 1re clôture >= tweet
    return np.unique(pos[pos < len(idx_close_utc)])


def main():
    ohlc = load_ohlc("QQQ", "2000-01-01", None)
    price = ohlc["close"]
    idx = price.index
    o, h, l = ohlc["open"].values, ohlc["high"].values, ohlc["low"].values
    nfci, cpi = load_macro(idx)
    alloc = compute_allocation(price.values, nfci, cpi, high=h, low=l, open_=o)
    rv = yang_zhang_vol(o, h, l, price.values)
    fund = np.asarray(load_funding(idx), float)
    ret = price.pct_change().fillna(0).values
    close_utc = pd.DatetimeIndex([pd.Timestamp(d.date()).tz_localize("America/New_York") + pd.Timedelta(hours=16)
                                  for d in idx]).tz_convert("UTC")

    base_eq, base_fees = simulate(ret, alloc, fund)
    rows = [{"variante": "BASE x1.7 (déployée)", **metrics(base_eq, base_fees, idx, EVAL_START),
             "coupures": 0, **{f"OOS {k}": v for k, v in metrics(base_eq, base_fees, idx, OOS_START).items()
                               if k in ("%/an", "maxDD %")}}]
    for conf, vol, k, capv in itertools.product(CONFS, VOLS, KS, CAPS):
        sig = [p for p in signals(close_utc, conf) if rv[p] >= vol]
        cap = np.full(len(idx), np.nan)
        for p in sig:
            cap[p:p + k] = capv
        eq, fees = simulate(ret, alloc, fund, cap)
        # nb de coupures effectives = jours où le plafond force réellement une vente
        rows.append({"variante": f"conf≥{conf} vol≥{vol:.2f} K={k} cap={capv:g}",
                     **metrics(eq, fees, idx, EVAL_START), "coupures": len(sig),
                     **{f"OOS {kk}": v for kk, v in metrics(eq, fees, idx, OOS_START).items()
                        if kk in ("%/an", "maxDD %")}})
    res = pd.DataFrame(rows).round(2)
    pd.set_option("display.width", 200)
    print(f"Évaluation {EVAL_START} → {idx[-1].date()} (OOS = {OOS_START} → fin)\n")
    print(res.iloc[:1].to_string(index=False))
    print("\nTop 15 par Calmar :")
    print(res.iloc[1:].sort_values("Calmar", ascending=False).head(15).to_string(index=False))
    print("\nPire 5 par Calmar :")
    print(res.iloc[1:].sort_values("Calmar").head(5).to_string(index=False))
    better = (res.iloc[1:].Calmar > res.iloc[0].Calmar) & (res.iloc[1:]["%/an"] > res.iloc[0]["%/an"] - 0.5)
    print(f"\nVariantes battant la base en Calmar sans perdre > 0.5 pt de CAGR : {better.sum()}/{len(res) - 1}")
    res.to_csv("myfiles/pust17_tweet_overlay.csv", index=False)


if __name__ == "__main__":
    main()
