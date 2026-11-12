"""Reconnaissance EODHD : univers US (actifs + radiés) et accès aux endpoints. La clé n'est jamais affichée."""
import collections
import os

import requests
from dotenv import load_dotenv

load_dotenv()
K = os.getenv("EODHD_API_KEY")
B = "https://eodhd.com/api"


def get(p, **q):
    r = requests.get(f"{B}/{p}", params={"api_token": K, "fmt": "json", **q}, timeout=120)
    try:
        return r.status_code, r.json()
    except ValueError:
        return r.status_code, r.text[:200]


c, u = get("user")
print("user", c, {k: u.get(k) for k in ("subscriptionType", "apiRequests", "dailyRateLimit", "extraLimit")})
for dl in (0, 1):
    c, s = get("exchange-symbol-list/US", delisted=dl)
    print("delisted" if dl else "actifs", c, len(s) if isinstance(s, list) else s)
    if isinstance(s, list):
        print("  types", collections.Counter(x.get("Type") for x in s).most_common(8))
        print("  exch", collections.Counter(x.get("Exchange") for x in s).most_common(10))
        cs = [x for x in s if x.get("Type") == "Common Stock"
              and x.get("Exchange") in ("NYSE", "NASDAQ", "NYSE MKT", "NYSE ARCA", "AMEX", "BATS")]
        print("  actions ordinaires sur places principales:", len(cs))
        print("  ex", s[0])
TESTS = [
    ("eod AAPL", "eod/AAPL.US", {"from": "2026-11-01"}),
    ("eod LEHMQ radiée", "eod/LEHMQ.US", {"from": "2008-09-01", "to": "2008-09-20"}),
    ("fundamentals AAPL", "fundamentals/AAPL.US", {"filter": "General::Code"}),
    ("earnings cal", "calendar/earnings", {"symbols": "AAPL.US", "from": "2015-01-01", "to": "2015-12-31"}),
    ("economic-events", "economic-events", {"from": "2015-01-01", "to": "2015-01-10", "country": "US", "limit": 5}),
    ("news 2012", "news", {"from": "2012-03-01", "to": "2012-03-07", "limit": 3}),
    ("news AAPL 2014", "news", {"s": "AAPL.US", "from": "2014-03-01", "to": "2014-03-31", "limit": 3}),
    ("macro-indicator", "macro-indicator/USA", {"indicator": "inflation_consumer_prices_annual"}),
]
for name, p, q in TESTS:
    c, d = get(p, **q)
    n = len(d) if isinstance(d, (list, dict)) else d
    print(name, c, n, str(d)[:230].replace("\n", " "))
c, u = get("user")
print("appels consommés:", u.get("apiRequests"))
