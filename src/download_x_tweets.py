"""
Historique des tweets d'un compte X/Twitter via Apify (Tweet Scraper V2, `apidojo/tweet-scraper`).

Recherche `from:<handle> since:J until:J+1` jour par jour (fenêtres contiguës, chaque tweet
n'est payé qu'une fois), DAYS_PER_RUN jours regroupés dans un même run (liste `searchTerms`).
Reprenable : chaque jour terminé est écrit dans <out>/days/<J>.json (liste brute des tweets,
fichier vide = 0 tweet) et noté dans <out>/progress.json ; un run échoué est rejoué au prochain
lancement. À la fin, consolidation dans <out>/tweets.parquet (colonnes utiles, 1 ligne par tweet,
dédoublonné par id, trié par date).

Coût : 0,0004 $/tweet (pay-per-event, pas de frais de run). Deux garde-fous :
  --budget  : dépense cumulée (USD, lue sur les runs) au-delà de laquelle le script s'arrête
              proprement (on relance plus tard : les jours faits ne sont pas repayés) ;
  max_total_charge_usd par run (plafond Apify côté acteur).

ATTENTION : `createdAt` est en UTC ; pour un usage en backtest, un tweet de la séance J n'est
connu qu'à son heure de publication (pas de look-ahead si on filtre par timestamp).

Usage :
  python -m src.download_x_tweets DeItaone --start 2023-01-01 --out data/text/x/deltaone --budget 33
  python -m src.download_x_tweets DeItaone --out data/text/x/deltaone --consolidate   # parquet seul
  python -m src.download_x_tweets DeItaone --out ... --retry-empty      # rejoue les jours à 0 tweet
Nécessite APIFY_TOKEN dans .env et `pip install apify-client` (3.x).
"""
import argparse
import json
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pandas as pd
from apify_client import ApifyClient
from dotenv import load_dotenv

ACTOR = "apidojo/tweet-scraper"
PRICE_PER_TWEET = 0.0004           # USD, tarif de base
DAYS_PER_RUN = 7                   # 7 requêtes journalières par run
DAY_CAP = 400                      # tweets max attendus par jour (maxItems = DAYS_PER_RUN × DAY_CAP)
RUN_MAX_CHARGE = Decimal("1.50")   # plafond Apify par run (USD)
RETRIES = 3                        # tentatives par run avant abandon du lot
COLUMNS = ["id", "created_at", "text", "is_retweet", "is_reply", "is_quote", "lang",
           "like_count", "retweet_count", "reply_count", "quote_count", "view_count",
           "in_reply_to", "quote_text", "retweet_text", "url"]


def parse_x_date(s):
    """Date X ("Sun Jan 01 01:17:00 +0000 2023") -> datetime UTC."""
    return datetime.strptime(s, "%a %b %d %H:%M:%S %z %Y").astimezone(timezone.utc)


def as_dict(obj):
    for m in ("model_dump", "to_dict", "dict"):
        if hasattr(obj, m):
            return getattr(obj, m)()
    return obj if isinstance(obj, dict) else vars(obj)


def log(msg):
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}", flush=True)


class Progress:
    def __init__(self, path):
        self.path = path
        self.data = json.loads(path.read_text()) if path.exists() else {"days": {}, "runs": [], "spent_usd": 0.0}

    def done(self, day):
        return str(day) in self.data["days"]

    def save(self):
        self.path.write_text(json.dumps(self.data, indent=1, ensure_ascii=False))


def run_window(client, handle, days):
    """Un run Apify = une requête `from:` par jour de `days`. Retourne (items, usd dépensés)."""
    terms = [f"from:{handle} since:{d} until:{d + timedelta(days=1)}" for d in days]
    run_input = {"searchTerms": terms, "maxItems": DAY_CAP * len(days), "sort": "Latest"}
    for attempt in range(1, RETRIES + 1):
        try:
            run = client.actor(ACTOR).call(run_input=run_input, max_total_charge_usd=RUN_MAX_CHARGE,
                                           logger=None)
            status = str(getattr(getattr(run, "status", None), "value", getattr(run, "status", None)))
            if status != "SUCCEEDED":
                raise RuntimeError(f"run {getattr(run, 'id', '?')} status {status}")
            items = list(client.dataset(run.default_dataset_id).iterate_items())
            # usage_total_usd est calculé avec retard côté Apify (souvent 0/None juste après le run) :
            # on garde le max avec le tarif de base × items (les items `noResults` sont facturés aussi).
            usd = as_dict(client.run(run.id).get()).get("usage_total_usd") or 0.0
            return items, max(float(usd), len(items) * PRICE_PER_TWEET)
        except Exception as exc:  # noqa: BLE001 — réseau / acteur : on réessaie
            log(f"  tentative {attempt}/{RETRIES} échouée : {exc}")
            time.sleep(30 * attempt)
    raise RuntimeError(f"lot {days[0]} -> {days[-1]} abandonné après {RETRIES} tentatives")


def split_by_day(items, days):
    """Répartit les tweets par jour UTC (les bornes since/until de X ne tombent pas pile à minuit :
    un tweet hors des jours demandés est rattaché au jour le plus proche de la fenêtre)."""
    buckets = {d: {} for d in days}
    for it in items:
        if "text" not in it or not it.get("createdAt") or not it.get("id"):
            continue
        d = parse_x_date(it["createdAt"]).date()
        d = min(max(d, days[0]), days[-1])
        buckets[d][it["id"]] = it
    return {d: sorted(b.values(), key=lambda t: int(t["id"])) for d, b in buckets.items()}


def row_of(t):
    q, r = t.get("quote") or {}, t.get("retweetedTweet") or t.get("retweet") or {}
    return {
        "id": int(t["id"]), "created_at": parse_x_date(t["createdAt"]),
        "text": t.get("fullText") or t.get("text"),
        "is_retweet": bool(t.get("isRetweet")), "is_reply": bool(t.get("isReply")), "is_quote": bool(t.get("isQuote")),
        "lang": t.get("lang"),
        "like_count": t.get("likeCount"), "retweet_count": t.get("retweetCount"),
        "reply_count": t.get("replyCount"), "quote_count": t.get("quoteCount"), "view_count": t.get("viewCount"),
        "in_reply_to": t.get("inReplyToUsername"),
        "quote_text": (q.get("fullText") or q.get("text")) if isinstance(q, dict) else None,
        "retweet_text": (r.get("fullText") or r.get("text")) if isinstance(r, dict) else None,
        "url": t.get("url"),
    }


def consolidate(out):
    rows = []
    for f in sorted((out / "days").glob("*.json")):
        rows += [row_of(t) for t in json.loads(f.read_text(encoding="utf-8"))]
    df = pd.DataFrame(rows, columns=COLUMNS)
    if len(df):
        df = df.drop_duplicates("id").sort_values("created_at").reset_index(drop=True)
        for c in ("like_count", "retweet_count", "reply_count", "quote_count", "view_count"):
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    df.to_parquet(out / "tweets.parquet", index=False)
    log(f"{len(df)} tweets -> {out / 'tweets.parquet'}"
        + (f" ({df.created_at.min():%Y-%m-%d} -> {df.created_at.max():%Y-%m-%d})" if len(df) else ""))
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("handle")
    ap.add_argument("--start", default="2023-01-01")
    ap.add_argument("--end", default=str(date.today()), help="dernier jour inclus (défaut : aujourd'hui)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget", type=float, default=30.0, help="dépense cumulée max (USD) pour ce script")
    ap.add_argument("--consolidate", action="store_true", help="ne télécharge rien, rebâtit tweets.parquet")
    ap.add_argument("--retry-empty", action="store_true", help="rejoue les jours enregistrés à 0 tweet")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "days").mkdir(parents=True, exist_ok=True)
    progress = Progress(out / "progress.json")
    if args.consolidate:
        consolidate(out)
        return

    load_dotenv()
    token = os.getenv("APIFY_TOKEN")
    if not token:
        sys.exit("APIFY_TOKEN absent de l'environnement / .env")
    client = ApifyClient(token)

    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
    todo = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    if args.retry_empty:
        todo = [d for d in todo if progress.data["days"].get(str(d)) == 0]
    else:
        todo = [d for d in todo if not progress.done(d)]
    log(f"@{args.handle} {start} -> {end} : {len(todo)} jour(s) à faire, "
        f"déjà dépensé {progress.data['spent_usd']:.2f} $, budget {args.budget:.2f} $")

    n_tweets_session, usd_session = 0, 0.0
    try:
        for i in range(0, len(todo), DAYS_PER_RUN):
            days = todo[i:i + DAYS_PER_RUN]
            if progress.data["spent_usd"] >= args.budget:
                log(f"budget atteint ({progress.data['spent_usd']:.2f} $ >= {args.budget:.2f} $) : arrêt, "
                    f"{len(todo) - i} jour(s) restants (relancer plus tard)")
                break
            items, usd = run_window(client, args.handle, days)
            per_day = split_by_day(items, days)
            for d, tweets in per_day.items():
                (out / "days" / f"{d}.json").write_text(json.dumps(tweets, ensure_ascii=False), encoding="utf-8")
                progress.data["days"][str(d)] = len(tweets)
            progress.data["spent_usd"] = round(progress.data["spent_usd"] + usd, 4)
            progress.data["runs"].append({"from": str(days[0]), "to": str(days[-1]), "items": len(items),
                                          "usd": round(usd, 4), "at": datetime.now(timezone.utc).isoformat()})
            progress.save()
            n_tweets_session += len(items)
            usd_session += usd
            counts = " ".join(f"{d.strftime('%a')[:2]}{len(t):>4}" for d, t in per_day.items())
            log(f"{days[0]} -> {days[-1]} : {len(items):4d} tweets {usd:5.2f} $ | {counts} | "
                f"cumul {progress.data['spent_usd']:.2f} $")
            if len(items) >= DAY_CAP * len(days):
                log("  [WARN] maxItems atteint sur ce run : des tweets peuvent manquer")
    except KeyboardInterrupt:
        log("interrompu (les jours terminés sont conservés)")
    finally:
        progress.save()
        log(f"session : {n_tweets_session} tweets, {usd_session:.2f} $ ; total {progress.data['spent_usd']:.2f} $")
        consolidate(out)


if __name__ == "__main__":
    main()
