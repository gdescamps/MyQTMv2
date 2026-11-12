"""Sondage EODHD : profondeur/volume des news, radiées (cours + fondamentaux), calendrier économique."""
import json
import os

import requests
from dotenv import load_dotenv

load_dotenv()
K = os.getenv("EODHD_API_KEY")
B = "https://eodhd.com/api"


def get(p, **q):
    r = requests.get(f"{B}/{p}", params={"api_token": K, "fmt": "json", **q}, timeout=180)
    try:
        return r.status_code, r.json()
    except ValueError:
        return r.status_code, r.text[:200]


print("== volume du flux général par jour (limit 1000, offsets 0 / 1000 / 5000 / 9000)")
for day in ("2011-03-15", "2014-03-12", "2017-03-15", "2020-03-12", "2023-03-13", "2026-11-04"):
    out = []
    for off in (0, 1000, 5000, 9000):
        c, d = get("news", **{"from": day, "to": day, "limit": 1000, "offset": off})
        n = len(d) if isinstance(d, list) else f"ERR {c} {str(d)[:60]}"
        out.append(n)
        if n != 1000:
            break
    c, d = get("news", **{"from": day, "to": day, "limit": 50})
    if isinstance(d, list) and d:
        nosym = sum(1 for a in d if not a.get("symbols"))
        size = len(json.dumps(d)) // len(d)
        dates = sorted(a["date"] for a in d)
        print(day, out, f"| sans symbole {nosym}/50 | ~{size} octets/article | {dates[0]} .. {dates[-1]}")
    else:
        print(day, out, d if not isinstance(d, list) else "vide")

print("== news par ticker : première date disponible")
for s in ("AAPL.US", "JPM.US", "XOM.US"):
    for y in (2011, 2014, 2016, 2018, 2020):
        c, d = get("news", s=s, **{"from": f"{y}-01-01", "to": f"{y}-12-31", "limit": 1000})
        print(s, y, len(d) if isinstance(d, list) else d, end=" | ")
    print()

print("== radiées")
c, s = get("exchange-symbol-list/US", delisted=1)
byc = {x["Code"]: x for x in s}
for code in ("LEH", "LEHMQ", "ENRNQ", "BSC", "WAMUQ", "TWTR", "SIVB", "FRC", "ATVI", "YHOO", "CELG", "BBBY"):
    x = byc.get(code)
    if not x:
        print(code, "absent de la liste")
        continue
    c, e = get(f"eod/{code}.US")
    rng = f"{e[0]['date']} .. {e[-1]['date']} ({len(e)} j)" if isinstance(e, list) and e else e
    c, f = get(f"fundamentals/{code}.US")
    if isinstance(f, dict):
        q = (f.get("Financials", {}).get("Income_Statement", {}) or {}).get("quarterly", {}) or {}
        h = (f.get("Earnings", {}) or {}).get("History", {}) or {}
        fs = f"fond: {len(json.dumps(f))//1024} ko, {len(q)} trimestres ({min(q, default='-')}..{max(q, default='-')}), {len(h)} earnings"
        one = next(iter(q.values()), {})
        fs += f", filing_date={one.get('filing_date')}"
    else:
        fs = f"fond: {c} {str(f)[:60]}"
    print(code, x["Exchange"], x["Type"], "| eod", rng, "|", fs)

print("== calendrier économique")
for a, b in (("2012-03-01", "2012-03-31"), ("2019-03-01", "2019-03-31"), ("2026-10-01", "2026-10-31")):
    c, d = get("economic-events", country="US", limit=1000, **{"from": a, "to": b})
    print(a, len(d) if isinstance(d, list) else d, str(d[:1])[:260] if isinstance(d, list) else "")
c, u = get("user")
print("appels consommés:", u.get("apiRequests"))
