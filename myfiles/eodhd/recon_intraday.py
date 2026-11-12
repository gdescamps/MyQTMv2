"""Sondage EODHD intraday : profondeur par intervalle, titres radiés, taille maximale d'une requête."""
import os
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv

load_dotenv()
K = os.getenv("EODHD_API_KEY")
B = "https://eodhd.com/api"


def ts(s):
    return int(datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())


def intr(sym, itv, a, b):
    r = requests.get(f"{B}/intraday/{sym}", timeout=180,
                     params={"api_token": K, "fmt": "json", "interval": itv, "from": ts(a), "to": ts(b)})
    try:
        d = r.json()
    except ValueError:
        return f"HTTP {r.status_code} {r.text[:80]}"
    if not isinstance(d, list) or not d:
        return f"HTTP {r.status_code} vide {str(d)[:80]}"
    return f"{len(d)} barres, {d[0]['datetime']} .. {d[-1]['datetime']}"


u0 = requests.get(f"{B}/user", params={"api_token": K}).json()["apiRequests"]
for sym, itv, a, b in [
    ("AAPL.US", "1m", "2011-03-01", "2011-03-10"), ("AAPL.US", "1m", "2005-03-01", "2005-03-10"),
    ("AAPL.US", "1m", "2026-01-01", "2026-06-30"), ("AAPL.US", "1m", "2026-07-01", "2026-11-10"),
    ("AAPL.US", "5m", "2011-03-01", "2011-03-10"), ("AAPL.US", "5m", "2019-03-01", "2019-03-10"),
    ("AAPL.US", "5m", "2021-01-01", "2026-11-10"), ("AAPL.US", "1h", "2011-03-01", "2011-06-10"),
    ("AAPL.US", "1h", "2019-03-01", "2019-06-10"), ("AAPL.US", "1h", "2011-01-01", "2026-11-10"),
    ("SIVB.US", "1m", "2023-03-01", "2023-03-10"), ("TWTR.US", "1m", "2022-10-01", "2022-10-28"),
    ("LEH.US", "1m", "2008-09-01", "2008-09-17"), ("CELG.US", "5m", "2019-10-01", "2019-11-20"),
    ("ACHN.US", "1m", "2019-06-01", "2019-06-20"),
]:
    print(f"{sym:9} {itv:3} {a} -> {b} : {intr(sym, itv, a, b)}")
u1 = requests.get(f"{B}/user", params={"api_token": K}).json()["apiRequests"]
print("appels consommés par ce sondage:", u1 - u0, "| total du jour:", u1)
