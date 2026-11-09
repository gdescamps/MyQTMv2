"""Frein « pression news cumulée » sur la stratégie déployée PUST+LQQ x1.7.

Score au close US de J : S = Σ sens·conf/100·exp(-(t_close - t_i)·ln2/HL) sur les tweets DeltaOne
classés (hors reaction_marche/hors_marche, sens≠0) publiés avant le close.
Frein : tant que S ≤ -SEUIL ET vol Yang-Zhang ≥ VOL → plafond d'expo CAP (1 = PUST seul, 0 = cash),
forcé hors bande, décision au close J / exécution J+1 (simulate de pust17_tweet_overlay).
Seuil exprimé en quantile de S sur 2023-2024 (calibrage) pour éviter de fixer un niveau sur toute la période.
"""
import itertools

import numpy as np
import pandas as pd

from myfiles.pust17_tweet_overlay import EVAL_START, OOS_START, metrics, simulate
from src.risk_off_strategy.data import load_funding, load_macro, load_ohlc
from src.risk_off_strategy.strategy import compute_allocation, yang_zhang_vol

HLS = [8, 24, 72]
QS = [0.02, 0.05, 0.10, 0.20]
VOLS = [0.0, 0.20, 0.25]
CAPS = [1.0, 0.0]


def score(close_utc, sub, hl):
    ts = sub.created_at.values.astype("datetime64[ns]").astype(np.int64)
    w = (sub.sens * sub.confiance / 100).values
    tc = close_utc.values.astype("datetime64[ns]").astype(np.int64)
    lam = np.log(2) / (hl * 3600e9)
    S = np.zeros(len(tc))
    k0 = np.searchsorted(ts, tc[0] - int(20 * hl * 3600e9))
    for i, t in enumerate(tc):
        if t < ts[0]:
            continue
        j1 = np.searchsorted(ts, t)
        j0 = max(k0, np.searchsorted(ts, t - int(10 * hl * 3600e9)))
        S[i] = (w[j0:j1] * np.exp(-lam * (t - ts[j0:j1]))).sum()
    return S


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
    df = pd.read_parquet("data/text/x/deltaone/classified.parquet")
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    df = df[~df.categorie.isin(["reaction_marche", "hors_marche"]) & (df.sens != 0)].dropna(subset=["sens"])
    df = df.sort_values("created_at")
    subs = {"tous": df, "movers": df[df.mover == True]}  # noqa: E712
    ev = idx >= pd.Timestamp(EVAL_START)
    calib = ev & (idx < pd.Timestamp("2025-01-01"))

    base_eq, base_fees = simulate(ret, alloc, fund)
    oos = lambda eq, f: {f"OOS {k}": v for k, v in metrics(eq, f, idx, OOS_START).items() if k in ("%/an", "maxDD %")}  # noqa: E731
    rows = [{"variante": "BASE x1.7", **metrics(base_eq, base_fees, idx, EVAL_START), "jours freinés": 0,
             "ventes forcées": 0, **oos(base_eq, base_fees)}]
    for (sname, sub), hl in itertools.product(subs.items(), HLS):
        S = score(close_utc, sub, hl)
        for q, vol, capv in itertools.product(QS, VOLS, CAPS):
            thr = np.quantile(S[calib], q)
            on = ev & (S <= min(thr, 0)) & (rv >= vol)
            cap = np.where(on, capv, np.nan)
            eq, fees = simulate(ret, alloc, fund, cap)
            E = np.clip(2 * alloc, 0, 1.7)
            rows.append({"variante": f"{sname} HL={hl}h q={q:.2f} vol≥{vol:.2f} cap={capv:g}",
                         **metrics(eq, fees, idx, EVAL_START), "jours freinés": int(on.sum()),
                         "ventes forcées": int((on & (E > capv + 0.05)).sum()), **oos(eq, fees)})
    res = pd.DataFrame(rows).round(2)
    pd.set_option("display.width", 220)
    print(f"Évaluation {EVAL_START} → {idx[-1].date()}, seuils calibrés 2023-2024, OOS = {OOS_START} →\n")
    print(res.iloc[:1].to_string(index=False))
    print("\nTop 15 Calmar :")
    print(res.iloc[1:].sort_values("Calmar", ascending=False).head(15).to_string(index=False))
    print("\nPire 5 :")
    print(res.iloc[1:].sort_values("Calmar").head(5).to_string(index=False))
    b = res.iloc[0]
    better = (res.iloc[1:].Calmar > b.Calmar) & (res.iloc[1:]["%/an"] > b["%/an"])
    print(f"\nBattent la base en Calmar ET CAGR : {better.sum()}/{len(res) - 1} ; "
          f"médiane Δ%/an {(res.iloc[1:]['%/an'] - b['%/an']).median():+.2f}, "
          f"médiane ΔmaxDD {(res.iloc[1:]['maxDD %'] - b['maxDD %']).median():+.2f}")
    res.to_csv("myfiles/pust17_news_brake.csv", index=False)


if __name__ == "__main__":
    main()
