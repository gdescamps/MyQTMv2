"""Débit journalier (tweets/jour) de comptes X via Apify Tweet Scraper V2 — un jour par compte.

Recherche `from:<handle> since:J until:J+2`, filtre sur la journée J en heure de New York.
Coût : 0,0004 $/tweet, plafonné à DAY_CAP par compte. Cache -> outputs/apify/daily_rate/<handle>_<J>.json

Usage : python myfiles/test_apify_daily_rate.py 2026-11-05 DeItaone Reuters markets
"""
import json
import os
import sys
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

from apify_client import ApifyClient
from dotenv import load_dotenv

DAY_CAP = 600
MAX_CHARGE = Decimal("0.30")
NY = ZoneInfo("America/New_York")
OUT = Path("outputs/apify/daily_rate")


def parse_x_date(s):
    return datetime.strptime(s, "%a %b %d %H:%M:%S %z %Y")


def main():
    load_dotenv()
    client = ApifyClient(os.environ["APIFY_TOKEN"])
    day = date.fromisoformat(sys.argv[1])
    OUT.mkdir(parents=True, exist_ok=True)
    for handle in sys.argv[2:]:
        cache = OUT / f"{handle}_{day}.json"
        if cache.exists():
            items = json.loads(cache.read_text())
        else:
            q = f"from:{handle} since:{day} until:{day + timedelta(days=2)}"
            run = client.actor("apidojo/tweet-scraper").call(
                run_input={"searchTerms": [q], "maxItems": DAY_CAP, "sort": "Latest"},
                max_total_charge_usd=MAX_CHARGE, logger=None)
            items = client.dataset(run.default_dataset_id).list_items().items
            cache.write_text(json.dumps(items, ensure_ascii=False))
        tweets = {it["id"]: it for it in items if "text" in it and it.get("createdAt")}
        kept = [t for t in tweets.values() if parse_x_date(t["createdAt"]).astimezone(NY).date() == day]
        cap = " [PLAFOND ATTEINT]" if len(items) >= DAY_CAP else ""
        print(f"@{handle} {day}: {len(items)} items, {len(kept)} tweets le {day} (ET){cap}")


if __name__ == "__main__":
    main()
