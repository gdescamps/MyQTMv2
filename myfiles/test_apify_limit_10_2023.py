"""Test Apify : les 10 PREMIERS tweets de 2023 du compte X/Twitter @unusual_whales.

Actor `apidojo/tweet-scraper` (Tweet Scraper V2) en mode recherche, jour par jour :
`from:<handle> since:<J> until:<J+1>`. La recherche X renvoie du plus récent au plus
ancien, donc on lit la journée entière (jusqu'à DAY_CAP tweets) et on garde les plus
anciens. Si le jour ne suffit pas à faire 10 tweets, on passe au jour suivant.
Nécessite un plan Apify payant (le plan FREE ne renvoie que des items `noResults`
sur toute recherche).

Sortie brute (JSON de l'Actor) -> outputs/apify/test_apify_limit_10_2023.json

Usage : python myfiles/test_apify_limit_10_2023.py [handle]
Nécessite APIFY_TOKEN dans .env et `pip install apify-client` (3.x).
"""
import json
import os
import sys
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

from apify_client import ApifyClient
from dotenv import load_dotenv

HANDLE = sys.argv[1] if len(sys.argv) > 1 else "unusual_whales"
LIMIT = 10                         # nombre de tweets voulus
DAY_CAP = 200                      # max items lus par jour (0,0004 $/tweet -> 0,08 $)
MAX_CHARGE = Decimal("0.20")       # plafond de dépense par run (USD)
MAX_DAYS = 7                       # garde-fou
START = date(2023, 1, 1)
START_DT = datetime(START.year, START.month, START.day, tzinfo=timezone.utc)
OUT = Path("outputs/apify/test_apify_limit_10_2023.json")


def created(tweet):
    """createdAt X ("Sun Jan 01 01:17:00 +0000 2023") -> datetime UTC."""
    return datetime.strptime(tweet["createdAt"], "%a %b %d %H:%M:%S %z %Y")


def search_day(client, day):
    query = f"from:{HANDLE} since:{day} until:{day + timedelta(days=1)}"
    run_input = {"searchTerms": [query], "maxItems": DAY_CAP, "sort": "Latest"}
    # apify-client 3.x : `call` renvoie un modèle `Run` typé ; logger=None coupe le log de l'Actor.
    run = client.actor("apidojo/tweet-scraper").call(
        run_input=run_input, max_total_charge_usd=MAX_CHARGE, logger=None)
    status = getattr(run, "status", None)
    if run is None or str(getattr(status, "value", status)) != "SUCCEEDED":
        sys.exit(f"run non réussi : {status} ({query})")
    items = client.dataset(run.default_dataset_id).list_items().items
    # `since:` de la recherche X n'est pas borné à minuit UTC (des tweets du 31/12/2022 23:39 UTC
    # remontent) -> on refiltre sur createdAt.
    tweets = [it for it in items if "text" in it and created(it) >= START_DT]
    print(f"  {query}  -> {len(tweets)} tweet(s) ({len(items)} item(s))")
    if len(tweets) >= DAY_CAP:
        print(f"  [WARN] {DAY_CAP} tweets = plafond atteint, les plus anciens du jour peuvent manquer")
    return tweets


def main():
    load_dotenv()
    token = os.getenv("APIFY_TOKEN")
    if not token:
        sys.exit("APIFY_TOKEN absent de l'environnement / .env")
    client = ApifyClient(token)

    print(f"-> {LIMIT} premiers tweets 2023 de @{HANDLE}")
    found = {}
    for k in range(MAX_DAYS):
        for t in search_day(client, START + timedelta(days=k)):
            found[t["id"]] = t
        if len(found) >= LIMIT:
            break

    tweets = sorted(found.values(), key=lambda t: int(t["id"]))[:LIMIT]  # id X croissant = chronologique
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(tweets, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{len(tweets)} tweet(s) -> {OUT}\n")
    for t in tweets:
        text = (t.get("text") or "").replace("\n", " ")
        print(f"[{t.get('createdAt')}] {text[:120]}")


if __name__ == "__main__":
    main()
