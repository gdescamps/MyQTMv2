"""Correspondance des données d'entrée de ref/MyQTM (FMP) avec data/eodhd : contrôle sur l'univers TRADE_STOCKS."""
import collections
import gzip
import importlib.util
import json
import os
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests
from dotenv import load_dotenv

OUT = Path("data/eodhd")
spec = importlib.util.spec_from_file_location("myqtm_config", "ref/MyQTM/src/config.py")
cfg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cfg)
stocks = sorted(cfg.TRADE_STOCKS)
print(f"univers MyQTM : {len(stocks)} titres ; téléchargement depuis {cfg.DOWNLOAD_START_DATE}")

# 1. cours
miss, short = [], []
for s in stocks:
    f = OUT / "eod" / f"{s}.parquet"
    if not f.exists():
        miss.append(s)
        continue
    d = pd.read_parquet(f, columns=["date"])["date"]
    if d.min() > cfg.DOWNLOAD_START_DATE:
        short.append(f"{s}({d.min()})")
print(f"1. cours : {len(stocks) - len(miss)}/{len(stocks)} présents ; absents {miss} ; début après 2017 : {short}")
print("   colonnes :", list(pd.read_parquet(OUT / 'eod' / 'AAPL.parquet').columns))

# 2. fondamentaux
fmiss = [s for s in stocks if not (OUT / "fundamentals" / f"{s}.json.gz").exists()]
print(f"2. fondamentaux : {len(stocks) - len(fmiss)}/{len(stocks)} présents ; absents (téléchargement en cours) {fmiss}")
f = json.load(gzip.open(OUT / "fundamentals" / "AAPL.json.gz", "rt"))
print("   sections :", list(f.keys()))
g = f["General"]
print("   profil :", {k: str(g.get(k))[:40] for k in ("Name", "Exchange", "CurrencyCode", "Sector", "Industry", "Description")})
print("   AnalystRatings (instantané, pas d'historique) :", f.get("AnalystRatings"))
bs = f["Financials"]["Balance_Sheet"]["quarterly"]
k = max(bs)
print(f"   bilan trimestriel : {len(bs)} trimestres {min(bs)}..{k} ; filing_date={bs[k].get('filing_date')}")
print("   champs dette :", {x: bs[k].get(x) for x in ("totalAssets", "totalLiab", "totalStockholderEquity", "shortLongTermDebtTotal", "longTermDebt", "shortTermDebt")})
nfd = sum(1 for v in bs.values() if v.get("filing_date"))
print(f"   trimestres avec filing_date : {nfd}/{len(bs)}")
tr = f["Earnings"]["Trend"]
kt = max(tr)
print("   Earnings.Trend :", len(tr), "périodes ; champs", list(tr[kt].keys())[:14])
print("   Earnings.History ex :", list(f["Earnings"]["History"].items())[-1])

# 3. news : échantillon d'un jour par mois depuis 2017
target = {s + ".US" for s in stocks}
per_year = collections.Counter()
per_year_days = collections.Counter()
dom = collections.Counter()
per_stock = collections.Counter()
ntot = nshort = 0
keys = None
for m in pd.date_range("2017-01-01", "2026-10-01", freq="MS"):
    p = OUT / "news" / f"{m:%Y}" / f"{(m + pd.Timedelta(days=14)):%Y-%m-%d}.jsonl.gz"
    if not p.exists():
        continue
    per_year_days[m.year] += 1
    for line in gzip.open(p, "rt"):
        a = json.loads(line)
        keys = keys or list(a.keys())
        hit = target & set(a.get("symbols") or [])
        if not hit:
            continue
        ntot += 1
        per_year[m.year] += 1
        dom[urlparse(a.get("link") or "").netloc.replace("www.", "")] += 1
        nshort += len(a.get("content") or "") < 500
        for h in hit:
            per_stock[h] += 1
print("3. news, champs :", keys)
print("   articles/jour citant un titre MyQTM (jour 15 de chaque mois) :",
      {y: round(per_year[y] / per_year_days[y]) for y in sorted(per_year_days)})
rel = set(cfg.RELIABLE_NEWS_SITES)
inrel = sum(v for k2, v in dom.items() if k2 in rel)
print(f"   part des articles venant des RELIABLE_NEWS_SITES : {inrel}/{ntot} = {inrel / max(ntot, 1):.0%} ; contenu < 500 car. : {nshort / max(ntot, 1):.0%}")
print("   domaines principaux :", [(k2, v, "fiable" if k2 in rel else "-") for k2, v in dom.most_common(12)])
zero = [s for s in stocks if per_stock[s + ".US"] == 0]
print(f"   titres sans aucun article dans l'échantillon : {zero}")

# 4. macro : calendrier économique
ev = pd.concat([pd.read_parquet(p) for p in sorted((OUT / "econ").glob("*.parquet")) if pd.read_parquet(p).shape[0]])
us = ev[ev["country"] == "US"]
print(f"4. calendrier économique : {len(ev)} événements, {ev['date'].min()} .. {ev['date'].max()} ; US {len(us)}")
for pat in ("Durable Goods Orders", "Retail Sales", "Interest Rate Decision", "Inflation Rate", "Unemployment Rate",
            "Non Farm Payrolls", "Initial Jobless Claims", "GDP Growth Rate", "Michigan Consumer Sentiment",
            "Industrial Production", "Housing Starts", "Total Vehicle Sales", "30-Year Mortgage Rate"):
    x = us[us["type"].str.contains(pat, case=False, regex=False)]
    print(f"   {pat:28} {len(x):5} événements, depuis {x['date'].min() if len(x) else '-'} ; variantes {x['type'].nunique()}")

# 5. indices (non téléchargés : 2 appels)
load_dotenv()
for sym in ("VIX.INDX", "IXIC.INDX"):
    r = requests.get(f"https://eodhd.com/api/eod/{sym}", timeout=60,
                     params={"api_token": os.getenv("EODHD_API_KEY"), "fmt": "json", "from": "2017-01-01"})
    d = r.json() if r.status_code == 200 else []
    print(f"5. indice {sym} : HTTP {r.status_code}, {len(d)} jours", (d[0]["date"], d[-1]["date"]) if d else "")
