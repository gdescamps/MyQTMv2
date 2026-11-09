"""Telecharge les donnees gratuites de couverture / positionnement institutionnel -> myfiles/hedge_data/.

  cftc_{nq,es,vix}.parquet : CFTC Traders in Financial Futures (futures only), hebdo depuis 2006
                             E-mini Nasdaq-100 (209742), E-mini S&P 500 (13874A), VIX futures (1170E1)
                             positions du mardi, publiees le vendredi 15:30 ET
  dix.parquet              : SqueezeMetrics DIX (dark pool) / GEX (gamma dealers), quotidien depuis 2011
  cboe_idx.parquet         : ^SKEW, ^VVIX, ^VIX, ^VIX3M, ^VIX9D (yfinance), quotidien
Put/call CBOE : deja dans data/cboe_putcall.parquet. Non couverts : NAAIM (page JS), short interest FINRA.
    python -m myfiles.hedge_data_download
"""
import os

import pandas as pd
import requests

OUT = "myfiles/hedge_data"
CFTC = {"nq": "209742", "es": "13874A", "vix": "1170E1"}
CFTC_URL = "https://publicreporting.cftc.gov/resource/gpe5-46if.json"
UA = {"User-Agent": "Mozilla/5.0 (research script)"}


def cftc(code):
    r = requests.get(CFTC_URL, params={"cftc_contract_market_code": code, "$limit": 5000,
                                       "$order": "report_date_as_yyyy_mm_dd"}, timeout=120)
    r.raise_for_status()
    d = pd.DataFrame(r.json())
    d["date"] = pd.to_datetime(d["report_date_as_yyyy_mm_dd"]).dt.tz_localize(None).dt.normalize()
    num = [c for c in d.columns if c.startswith(("asset_mgr_", "lev_money_", "dealer_", "open_interest",
                                                 "nonrept_", "other_rept_", "traders_tot"))]
    d[num] = d[num].apply(pd.to_numeric, errors="coerce")
    return d.groupby("date")[num].last().sort_index()


def main():
    os.makedirs(OUT, exist_ok=True)
    for k, code in CFTC.items():
        d = cftc(code)
        d.to_parquet(f"{OUT}/cftc_{k}.parquet")
        print(f"cftc_{k}: {d.index[0].date()} -> {d.index[-1].date()} ({len(d)} semaines)")
    r = requests.get("https://squeezemetrics.com/monitor/static/DIX.csv", headers=UA, timeout=120)
    r.raise_for_status()
    from io import StringIO
    d = pd.read_csv(StringIO(r.text), parse_dates=["date"]).set_index("date")[["dix", "gex"]]
    d.to_parquet(f"{OUT}/dix.parquet")
    print(f"dix: {d.index[0].date()} -> {d.index[-1].date()}")
    import yfinance as yf
    t = {"^SKEW": "skew", "^VVIX": "vvix", "^VIX": "vix", "^VIX3M": "vix3m", "^VIX9D": "vix9d"}
    d = yf.download(list(t), start="2000-01-01", auto_adjust=False, progress=False)["Close"].rename(columns=t)
    d.index = pd.to_datetime(d.index).tz_localize(None).normalize()
    d.to_parquet(f"{OUT}/cboe_idx.parquet")
    print("cboe_idx:", {c: str(d[c].first_valid_index().date()) for c in d})


if __name__ == "__main__":
    main()
