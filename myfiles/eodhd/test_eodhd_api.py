"""Test minimal de l'API EODHD (clé lue dans .env, jamais affichée)."""
import os

import requests
from dotenv import load_dotenv

load_dotenv()
KEY = os.getenv("EODHD_API_KEY")
BASE = "https://eodhd.com/api"


def get(path, **params):
    try:
        r = requests.get(f"{BASE}/{path}", params={"api_token": KEY, "fmt": "json", **params}, timeout=30)
    except requests.RequestException as e:
        return None, type(e).__name__
    try:
        return r.status_code, r.json()
    except ValueError:
        return r.status_code, r.text[:200]


def main():
    if not KEY:
        print("EODHD_API_KEY absente de l'environnement")
        return
    code, user = get("user")
    print("user:", code)
    if isinstance(user, dict):
        for k in ("subscriptionType", "apiRequests", "apiRequestsDate", "dailyRateLimit", "extraLimit"):
            print(f"  {k}: {user.get(k)}")
    code, news = get("news", limit=5)
    print("news (flux général):", code)
    if isinstance(news, list):
        for a in news:
            print(f"  {a.get('date')} | {a.get('title', '')[:90]} | tags={a.get('tags', [])[:4]} | symbols={a.get('symbols', [])[:3]}")
    else:
        print("  ", news)
    code, user = get("user")
    if isinstance(user, dict):
        print("appels consommés aujourd'hui:", user.get("apiRequests"), "/", user.get("dailyRateLimit"))


if __name__ == "__main__":
    main()
