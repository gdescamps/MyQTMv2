"""Test Apify : les 5 derniers tweets du compte X/Twitter @unusual_whales.

Actor `apidojo/tweet-scraper` (Tweet Scraper V2), tri "Latest", 5 items.
Sortie brute (JSON de l'Actor) -> outputs/apify/test_apify_limit_5.json

Usage : python myfiles/test_apify_limit_5.py [handle]
Nécessite APIFY_TOKEN dans .env et `pip install apify-client` (3.x).
Plan gratuit Apify : 10 items max par run sur cet Actor.
"""
import json
import os
import sys
from pathlib import Path

from apify_client import ApifyClient
from dotenv import load_dotenv

HANDLE = sys.argv[1] if len(sys.argv) > 1 else "unusual_whales"
LIMIT = 5
OUT = Path("outputs/apify/test_apify_limit_5.json")


def main():
    load_dotenv()
    token = os.getenv("APIFY_TOKEN")
    if not token:
        sys.exit("APIFY_TOKEN absent de l'environnement / .env")
    client = ApifyClient(token)

    run_input = {"twitterHandles": [HANDLE], "maxItems": LIMIT, "sort": "Latest"}
    print(f"-> apidojo/tweet-scraper {run_input}")
    # apify-client 3.x : `call` renvoie un modèle `Run` typé ; logger=None coupe le log de l'Actor.
    run = client.actor("apidojo/tweet-scraper").call(run_input=run_input, logger=None)
    status = getattr(run, "status", None)
    if run is None or str(getattr(status, "value", status)) != "SUCCEEDED":
        sys.exit(f"run non réussi : {status}")

    items = client.dataset(run.default_dataset_id).list_items().items
    # L'Actor peut renvoyer une page entière (> maxItems) ou des items "noResults" : on filtre et on tronque.
    tweets = [it for it in items if it.get("type") == "tweet" or "text" in it]
    tweets.sort(key=lambda t: int(t.get("id", 0) or 0), reverse=True)  # id X croissant avec le temps
    tweets = tweets[:LIMIT]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(tweets, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(tweets)} tweet(s) (sur {len(items)} item(s) renvoyés) -> {OUT}\n")

    for t in tweets:
        text = (t.get("text") or "").replace("\n", " ")
        print(f"[{t.get('createdAt')}] {text[:140]}")


if __name__ == "__main__":
    main()
