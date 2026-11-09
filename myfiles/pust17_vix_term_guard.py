"""Structure par terme du VIX (VIX / VIX3M) comme garde-fou ou accélérateur de retour, stratégie x1.7 PUST+LQQ.

Données : yfinance ^VIX et ^VIX3M (2006-07 →) → data/vix/vix_term.parquet. Clôtures 16:15 ET ≈ clôture QQQ
→ connues le soir J, appliquées J+1 (exec_lag=1 de simulate_net) : pas de look-ahead.
Variantes (frais 0.1 %/côté, garde-fous NFCI+CPI conservés) :
  guard   : ratio (brut ou MM5) > T  → cash ou cap1 (expo ≤ 1)
  reentry : sous la SMA250 mais ratio MM5 < L (calme revenu) → alloc ≥ 0.5 (expo ≥ 1, PUST)
Validation : seuil réglé sur H1 2007-2016, lu sur H2 2017-2026 et inversement ; épisodes de crise.
    python -m myfiles.pust17_vix_term_guard
"""
import os

import numpy as np
import pandas as pd
import yfinance as yf

import src.risk_off_strategy.strategy as st
from src.risk_off_strategy.data import load_funding, load_macro, load_ohlc
from src.risk_off_strategy.strategy import E_MAX, compute_allocation, simulate_net

st.BUY_FEE = st.SELL_FEE = 0.001
OUT = "data/vix/vix_term.parquet"
HALVES = {"H1": ("2007-01-01", "2016-12-31"), "H2": ("2017-01-01", None), "2007+": ("2007-01-01", None)}
EPIS = {"2008": ("2007-10-01", "2009-03-31"), "2011": ("2011-07-01", "2011-10-31"), "2015-16": ("2015-08-01", "2016-02-29"),
        "2018T4": ("2018-10-01", "2018-12-31"), "2020": ("2020-02-15", "2020-04-30"), "2022": ("2021-11-15", "2022-12-31"),
        "2025": ("2025-02-15", "2025-05-31")}


def load_vix():
    if not os.path.exists(OUT):
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        d = {t: yf.download(t, start="2006-01-01", progress=False, auto_adjust=False)["Close"].squeeze()
             for t in ("^VIX", "^VIX3M")}
        pd.DataFrame({"vix": d["^VIX"], "vix3m": d["^VIX3M"]}).dropna().to_parquet(OUT)
    return pd.read_parquet(OUT)


def win(eq, idx, a, b):
    m = (idx >= pd.Timestamp(a)) & (idx <= (pd.Timestamp(b) if b else idx[-1]))
    e = eq[m] / eq[m][0]
    cagr = e[-1] ** (252 / m.sum()) - 1
    dd = (e / np.maximum.accumulate(e) - 1).min()
    return 100 * cagr, 100 * dd, cagr / abs(dd)


def main():
    o = load_ohlc("QQQ", "2000-01-01", None)
    p, idx = o["close"], o.index
    nfci, cpi = load_macro(idx)
    fund = load_funding(idx)
    v = load_vix()
    print(f"VIX/VIX3M : {v.index[0].date()} → {v.index[-1].date()}, dernier ratio {v.vix.iloc[-1] / v.vix3m.iloc[-1]:.3f}")
    r = (v.vix / v.vix3m).reindex(idx).ffill()
    sig = {"brut": np.nan_to_num(r.values, nan=0), "MM5": np.nan_to_num(r.rolling(5).mean().values, nan=0)}
    base = compute_allocation(p.values, nfci, cpi, high=o["high"].values, low=o["low"].values, open_=o["open"].values)
    sma = pd.Series(p.values).rolling(250).mean().values
    below = p.values < sma
    has = ~np.isnan(r.values)
    rows = []

    def run(name, fam, thr, a, on):
        eq, fees, rev = simulate_net(p, a, leverage=2, funding=fund, e_max=E_MAX)
        row = {"variante": name, "famille": fam, "seuil": thr, "% jours actifs": round(100 * on[idx >= "2007"].mean(), 1),
               "frais %/an": round(fees, 2), "revis/an": round(rev, 1)}
        for h, (x, y) in HALVES.items():
            c, d, k = win(eq, idx, x, y)
            row.update({f"{h} %/an": round(c, 1), f"{h} maxDD": round(d, 1), f"{h} Calmar": round(k, 2)})
        for k, (x, y) in EPIS.items():
            m = (idx >= x) & (idx <= y)
            row[k] = round(100 * (eq[m][-1] / eq[m][0] - 1))
        rows.append(row)

    run("base NFCI+CPI", "-", np.nan, base, np.zeros(len(idx), bool))
    for sm in ("brut", "MM5"):
        for t in (0.95, 1.0, 1.05, 1.1, 1.15, 1.2):
            on = has & (sig[sm] > t)
            run(f"guard {sm} >{t} cash", f"guard {sm} cash", t, np.where(on, 0.0, base), on)
            run(f"guard {sm} >{t} cap1", f"guard {sm} cap1", t, np.where(on, np.minimum(base, 0.5), base), on)
    for L in (0.8, 0.85, 0.9, 0.95):
        on = has & below & (sig["MM5"] < L) & (base > 0)          # base>0 : ne contourne pas NFCI/CPI
        run(f"reentry MM5 <{L} → expo≥1", "reentry", L, np.where(on, np.maximum(base, 0.5), base), on)
    df = pd.DataFrame(rows)
    df.to_csv("myfiles/pust17_vix_term_guard.csv", index=False)
    pd.set_option("display.width", 260)
    cols = ["variante", "% jours actifs", "H1 %/an", "H1 maxDD", "H1 Calmar", "H2 %/an", "H2 maxDD", "H2 Calmar",
            "2007+ %/an", "2007+ maxDD", "2007+ Calmar", "frais %/an", "revis/an"] + list(EPIS)
    print(df[cols].to_string(index=False))
    ref = df.iloc[0]
    print("\nValidation croisée (seuil au Calmar max sur une moitié, lu sur l'autre) :")
    for fam in df.famille.unique()[1:]:
        g = df[df.famille == fam]
        for fit, test in (("H1", "H2"), ("H2", "H1")):
            b = g.loc[g[f"{fit} Calmar"].idxmax()]
            print(f"  {fam:16s} réglé {fit} seuil {b.seuil:<5} → {test} Calmar {b[f'{test} Calmar']:.2f} "
                  f"(base {ref[f'{test} Calmar']:.2f}), maxDD {b[f'{test} maxDD']:.1f} (base {ref[f'{test} maxDD']:.1f}), "
                  f"%/an {b[f'{test} %/an']:.1f} (base {ref[f'{test} %/an']:.1f})")


if __name__ == "__main__":
    main()
