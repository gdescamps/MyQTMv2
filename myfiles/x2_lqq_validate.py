"""
Etape 1/2 de l'optimiseur x2 net de frais : recupere le taux court USD (FRED DFF)
et VALIDE le LQQ synthetique (2xQQQ - funding - TER) contre le vrai LQQ 2008-2026.

- regression LQQ_ret ~ QQQ_ret  -> beta (doit etre ~2.0) + drag annuel (intercept).
- compare le drag mesure a (DFF + TER 0.60%) pour caler le spread swap.

Usage : python myfiles/x2_lqq_validate.py
"""
import os
import ssl
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")
FRED_KEY = os.environ.get("FRED")
RATES_CACHE = ROOT / "myfiles" / "x2_rates.parquet"
ANN = 252
TER_LQQ = 0.0060
TER_PUST = 0.0030


def fetch_fred_series(series_id):
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    u = (f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}"
         f"&api_key={FRED_KEY}&file_type=json&observation_start=2000-01-01")
    req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"})
    obs = json.loads(urllib.request.urlopen(req, timeout=30, context=ctx).read())["observations"]
    idx, val = [], []
    for o in obs:
        if o["value"] not in (".", ""):
            idx.append(pd.Timestamp(o["date"]))
            val.append(float(o["value"]) / 100.0)
    return pd.Series(val, index=idx, name=series_id)


def load_rates(dates):
    """DFF (Fed funds, funding USD) aligne sur `dates`, ffill, en fraction annuelle."""
    if RATES_CACHE.exists():
        r = pd.read_parquet(RATES_CACHE)
    else:
        dff = fetch_fred_series("DFF")
        r = dff.to_frame()
        RATES_CACHE.parent.mkdir(exist_ok=True)
        r.to_parquet(RATES_CACHE)
        print(f"  taux caches -> {RATES_CACHE}")
    s = r["DFF"].reindex(dates.union(r.index)).sort_index().ffill().reindex(dates)
    return s.bfill().values


if __name__ == "__main__":
    from src.risk_off_strategy.data import load_ohlc

    qqq = load_ohlc("QQQ")
    lqq = pd.read_parquet(ROOT / "data" / "LQQ.parquet")["close"]

    # returns alignes sur l'intersection des dates
    rq = qqq["close"].pct_change()
    rl = lqq.pct_change()
    df = pd.DataFrame({"q": rq, "l": rl}).dropna()
    df = df.loc["2008-01-01":]
    dff = load_rates(df.index)

    print("=" * 84)
    print(f"VALIDATION LQQ synthetique — overlap {df.index[0].date()} -> {df.index[-1].date()} "
          f"({len(df)} jours)")
    print("=" * 84)

    # beta + drag (regression l ~ beta*q + alpha)
    beta, alpha = np.polyfit(df["q"].values, df["l"].values, 1)
    drag_meas = -alpha * ANN
    print(f"  beta (levier effectif)       = {beta:.3f}   (attendu ~2.0)")
    print(f"  drag annuel mesure (−alpha·252) = {drag_meas*100:+.2f}%/an")
    print(f"  DFF moyen sur la periode     = {dff.mean()*100:.2f}%/an")
    print(f"  TER LQQ                       = {TER_LQQ*100:.2f}%/an")
    implied_spread = drag_meas - dff.mean() - TER_LQQ
    print(f"  => spread swap implicite      = drag − DFF − TER = {implied_spread*100:+.2f}%/an")

    # drag par regime de taux (bas 2012-2015 vs haut 2023-2024)
    print("\n  Drag par sous-periode (2·QQQ − LQQ, annualise) :")
    for lo, hi, tag in [("2012-01-01", "2015-12-31", "taux ~0"),
                        ("2016-01-01", "2019-12-31", "taux 0.5-2.5%"),
                        ("2020-01-01", "2021-12-31", "taux ~0 (covid)"),
                        ("2022-06-01", "2024-12-31", "taux 3-5.5%")]:
        sub = df.loc[lo:hi]
        if len(sub) < 20:
            continue
        d = (2 * sub["q"] - sub["l"]).mean() * ANN
        dff_sub = load_rates(sub.index).mean()
        print(f"    {tag:18} {lo[:7]}→{hi[:7]} : drag {d*100:+5.2f}%/an  (DFF {dff_sub*100:.2f}%  "
              f"→ spread+TER {(d-dff_sub)*100:+.2f}%)")

    # tracking du LQQ synthetique cale
    spread = max(0.0, implied_spread)
    r_syn = 2 * df["q"].values - (dff + spread + TER_LQQ) / ANN
    eq_syn = np.cumprod(1 + r_syn)
    eq_real = np.cumprod(1 + df["l"].values)
    err = eq_syn[-1] / eq_real[-1] - 1
    print(f"\n  LQQ synthetique cale (spread={spread*100:.2f}%) : ecart cumule final vs reel = {err*100:+.1f}%")
    print(f"  (residuel = FX EURUSD + tracking, hors modele USD-index)")
