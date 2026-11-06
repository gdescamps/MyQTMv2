"""Test Apify : nombre total de tweets du compte X/Twitter @unusual_whales.

Le total est le compteur `statusesCount` du profil (tweets + retweets + réponses,
tel que X l'affiche) — on ne scrape pas tout l'historique (coûteux et plafonné).

1) Actor profil `apidojo/twitter-user-scraper` (1 item = le profil)
2) Repli : Actor `apidojo/tweet-scraper` (Tweet Scraper V2), 1 tweet récent,
   dont l'objet `author` porte aussi `statusesCount`.

Usage : python myfiles/test_apify_count.py [handle]
Nécessite APIFY_TOKEN dans .env et `pip install apify-client`.
"""
import os
import sys

from apify_client import ApifyClient
from dotenv import load_dotenv

HANDLE = sys.argv[1] if len(sys.argv) > 1 else "unusual_whales"


def find_key(obj, key):
    """Cherche récursivement `key` dans un dict/list JSON (schéma de sortie variable selon l'Actor)."""
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            found = find_key(v, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = find_key(v, key)
            if found is not None:
                return found
    return None


def run_actor(client, actor_id, run_input):
    print(f"-> lancement {actor_id} {run_input}")
    # apify-client 3.x : `call` renvoie un modèle `Run` typé (attributs, pas un dict) ;
    # logger=None coupe le streaming du log de l'Actor dans la console.
    run = client.actor(actor_id).call(run_input=run_input, logger=None)
    status = getattr(run, "status", None)
    if run is None or str(getattr(status, "value", status)) != "SUCCEEDED":
        print(f"   run non réussi : {status}")
        return []
    items = client.dataset(run.default_dataset_id).list_items().items
    print(f"   {len(items)} item(s) récupéré(s)")
    return items


def main():
    load_dotenv()
    token = os.getenv("APIFY_TOKEN")
    if not token:
        sys.exit("APIFY_TOKEN absent de l'environnement / .env")
    client = ApifyClient(token)

    attempts = [
        ("apidojo/twitter-user-scraper", {"twitterHandles": [HANDLE], "maxItems": 1,
                                          "getFollowers": False, "getFollowing": False}),
        ("apidojo/tweet-scraper", {"twitterHandles": [HANDLE], "maxItems": 1, "sort": "Latest"}),
    ]
    for actor_id, run_input in attempts:
        try:
            items = run_actor(client, actor_id, run_input)
        except Exception as exc:  # noqa: BLE001 — test : on passe à l'Actor suivant
            print(f"   erreur : {exc}")
            continue
        for item in items:
            count = find_key(item, "statusesCount")
            if count is not None:
                name = find_key(item, "userName") or HANDLE
                followers = find_key(item, "followers")
                print(f"\n@{name} : {int(count):,} tweets au total".replace(",", " "))
                if followers is not None:
                    print(f"followers : {int(followers):,}".replace(",", " "))
                return
        print("   pas de champ statusesCount dans la sortie, clés du 1er item :",
              list(items[0].keys()) if items else "-")
    sys.exit("Impossible de déterminer le nombre de tweets.")


if __name__ == "__main__":
    main()
