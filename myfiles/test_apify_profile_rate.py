"""Débit (tweets/jour) d'un compte X via la timeline Tweet Scraper V2 (mode twitterHandles).
Usage : python myfiles/test_apify_profile_rate.py markets [maxItems]"""
import json, os, sys
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from apify_client import ApifyClient
from dotenv import load_dotenv

handle = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 300
load_dotenv()
client = ApifyClient(os.environ["APIFY_TOKEN"])
out = Path(f"outputs/apify/daily_rate/{handle}_profile_{n}.json")
if out.exists():
    items = json.loads(out.read_text())
else:
    run = client.actor("apidojo/tweet-scraper").call(
        run_input={"twitterHandles": [handle], "maxItems": n, "sort": "Latest"},
        max_total_charge_usd=Decimal("0.15"), logger=None)
    items = client.dataset(run.default_dataset_id).list_items().items
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(items, ensure_ascii=False))
tw = [x for x in items if "text" in x and x.get("createdAt")]
print("items", len(items), "tweets", len(tw), "non-tweets", [x for x in items if "text" not in x][:2])
ts = sorted(datetime.strptime(x["createdAt"], "%a %b %d %H:%M:%S %z %Y") for x in tw)
if ts:
    span = (ts[-1] - ts[0]).total_seconds() / 86400
    print("span", ts[0], "->", ts[-1], f"= {span:.2f} j  -> {len(tw)/span:.0f} tweets/jour")
    a = tw[0].get("author", {})
    print("statusesCount", a.get("statusesCount"), "created", a.get("createdAt"))
