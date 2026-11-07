"""Liste les derniers runs Apify avec leur coût réel (pay-per-event) et le nombre d'items.
Usage : python myfiles/test_apify_runs_cost.py [n]"""
import os, sys
from apify_client import ApifyClient
from dotenv import load_dotenv
load_dotenv()
client = ApifyClient(os.environ["APIFY_TOKEN"])
n = int(sys.argv[1]) if len(sys.argv) > 1 else 15


def as_dict(obj):
    for m in ("model_dump", "to_dict", "dict"):
        if hasattr(obj, m):
            return getattr(obj, m)()
    return obj if isinstance(obj, dict) else vars(obj)


for r in client.runs().list(limit=n, desc=True).items:
    rid = as_dict(r).get("id")
    rr = as_dict(client.run(rid).get())
    stats = rr.get("stats") or {}
    ds = as_dict(client.dataset(rr.get("default_dataset_id") or rr.get("defaultDatasetId")).get())
    print(f"{rr.get('started_at') or rr.get('startedAt')} {str(rr.get('status')):22s} "
          f"usd={rr.get('usage_total_usd') or rr.get('usageTotalUsd')} "
          f"items={ds.get('item_count') or ds.get('itemCount')} "
          f"charged={rr.get('charged_event_counts') or rr.get('chargedEventCounts')} "
          f"pricing={(rr.get('pricing_info') or rr.get('pricingInfo') or {}).get('pricing_model') or (rr.get('pricing_info') or rr.get('pricingInfo') or {}).get('pricingModel')}")
