"""Sondage de l'API EODHD de sentiment agrégé par jour et par titre."""
import os

import requests
from dotenv import load_dotenv

load_dotenv()
K = os.getenv("EODHD_API_KEY")
u0 = requests.get("https://eodhd.com/api/user", params={"api_token": K}).json()["apiRequests"]
r = requests.get("https://eodhd.com/api/sentiments", timeout=120,
                 params={"api_token": K, "s": "AAPL.US,LEH.US,TWTR.US", "from": "2011-01-01", "to": "2026-11-10", "fmt": "json"})
d = r.json() if r.status_code == 200 else r.text[:200]
if isinstance(d, dict):
    for s, v in d.items():
        ds = sorted(x["date"] for x in v) if v else []
        print(s, len(v), "jours", (ds[0], ds[-1]) if ds else "", v[:1])
else:
    print(r.status_code, d)
u1 = requests.get("https://eodhd.com/api/user", params={"api_token": K}).json()["apiRequests"]
print("coût de la requête :", u1 - u0, "appels")
