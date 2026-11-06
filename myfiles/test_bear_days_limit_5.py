"""Tweets de @unusual_whales pendant 5 séances de baisse choisies et leurs veilles, + images.

1. Séances FIGÉES (liste DAYS ci-dessous, choisies à la main parmi les pires séances
   QQQ depuis 2023 + le 2026-11-05), + pour chacune la séance de bourse PRÉCÉDENTE
   (veille = contexte avant le choc, calendrier QQQ). La baisse affichée = rendement close-to-close de QQQ
   lu dans data/QQQ.parquet, ou via yfinance si la séance est postérieure au parquet
   (lecture seule, le parquet n'est pas modifié).
2. Pour chaque séance J : recherche `from:<handle> since:J until:J+2` (Tweet Scraper V2),
   puis filtre sur la journée J en heure de New York (00:00 -> 24:00 ET) — `since/until`
   de X ne tombe pas pile sur minuit UTC, et la soirée ET déborde sur J+1 en UTC.
   Un tweets.json déjà présent sert de cache (pas de nouveau coût Apify).
3. Images uniquement (pas de vidéos) : photos des tweets, des tweets cités et retweetés,
   en taille originale (`?name=orig`). Les miniatures de vidéo ne sont pas prises.

Sortie :
  outputs/apify/bear_days/<J>/tweets.json      tweets bruts de la séance (ordre chronologique)
  outputs/apify/bear_days/<J>/images/<tweet_id>_<n>.jpg
  outputs/apify/bear_days/summary.json         séances, baisse, nb de tweets / images

Coût Apify : 0,0004 $/tweet, plafonné à MAX_CHARGE par séance (DAY_CAP tweets).
Usage : python myfiles/test_bear_days_limit_5.py [handle]
"""
import json
import os
import sys
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from apify_client import ApifyClient
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_apify_download import download, photo_orig  # noqa: E402

HANDLE = sys.argv[1] if len(sys.argv) > 1 else "unusual_whales"
DAYS = [                          # séances figées (contexte de la baisse en commentaire)
    date(2025, 4, 4),             # riposte chinoise aux tarifs "Liberation Day" (QQQ -6,2 %)
    date(2026, 11, 5),            # choisie à la main
    date(2026, 6, 5),             # emploi trop fort -> peur de hausse de taux + krach des semis (-4,8 %)
    date(2024, 12, 18),           # Fed "hawkish cut" : 2 baisses seulement prévues en 2025 (-3,6 %)
    date(2024, 7, 24),            # résultats Alphabet / Tesla décevants, rotation hors mega-caps (-3,6 %)
]
TICKER = "QQQ"
DAY_CAP = 1000                    # tweets max lus par séance (fenêtre de 2 jours UTC)
MAX_CHARGE = Decimal("0.45")      # plafond Apify par séance (USD) ~ DAY_CAP x 0,0004 + marge
NY = ZoneInfo("America/New_York")
OUT = Path("outputs/apify/bear_days")


def call_actor(client, actor_id, run_input, max_charge=None):
    # apify-client 3.x : `call` renvoie un modèle `Run` typé ; logger=None coupe le log de l'Actor.
    run = client.actor(actor_id).call(run_input=run_input, max_total_charge_usd=max_charge, logger=None)
    status = getattr(run, "status", None)
    if run is None or str(getattr(status, "value", status)) != "SUCCEEDED":
        sys.exit(f"run {actor_id} non réussi : {status} ({run_input})")
    return client.dataset(run.default_dataset_id).list_items().items


def parse_x_date(s):
    """Date X ("Sun Jan 01 01:17:00 +0000 2023") -> datetime aware."""
    return datetime.strptime(s, "%a %b %d %H:%M:%S %z %Y")


def load_close(last_day):
    """Clôtures QQQ : parquet du repo, complété par yfinance au-delà (lecture seule)."""
    close = pd.read_parquet(Path("data") / f"{TICKER}.parquet")["close"].dropna().sort_index()
    if last_day > close.index.max().date():
        try:
            import yfinance as yf
            fresh = yf.download(TICKER, start=str(close.index.max().date()), progress=False, auto_adjust=True)["Close"]
            fresh = fresh.squeeze("columns") if isinstance(fresh, pd.DataFrame) else fresh
            fresh.index = pd.to_datetime(fresh.index).tz_localize(None)
            close = pd.concat([close, fresh[fresh.index > close.index.max()]]).sort_index()
        except Exception as exc:  # noqa: BLE001 — rendement = info, pas bloquant
            print(f"[WARN] yfinance indisponible ({exc}) : rendements récents inconnus")
    return close


def day_return(close, d):
    ret = close.pct_change()
    return float(ret[pd.Timestamp(d)]) if pd.Timestamp(d) in ret.index else None


def previous_session(close, d):
    """Séance de bourse précédant `d` (calendrier QQQ), sinon jour ouvré précédent."""
    before = close.index[close.index < pd.Timestamp(d)]
    if len(before) and pd.Timestamp(d) in close.index:
        return before[-1].date()
    prev = d - timedelta(days=1)
    while prev.weekday() >= 5:
        prev -= timedelta(days=1)
    return prev


def tweets_of_day(client, day, cache):
    """Tweets de HANDLE publiés le jour `day` (date de bourse, heure de New York).
    `cache` = tweets.json d'un run précédent : relu tel quel s'il existe (pas de nouveau coût Apify)."""
    if cache.exists():
        kept = json.loads(cache.read_text(encoding="utf-8"))
        print(f"  cache {cache} -> {len(kept)} tweet(s)")
        return kept
    query = f"from:{HANDLE} since:{day} until:{day + timedelta(days=2)}"
    items = call_actor(client, "apidojo/tweet-scraper",
                       {"searchTerms": [query], "maxItems": DAY_CAP, "sort": "Latest"}, MAX_CHARGE)
    tweets = {it["id"]: it for it in items if "text" in it and it.get("createdAt")}
    kept = [t for t in tweets.values() if parse_x_date(t["createdAt"]).astimezone(NY).date() == day]
    kept.sort(key=lambda t: int(t["id"]))
    print(f"  {query}  -> {len(items)} item(s), {len(kept)} tweet(s) le {day} (ET)")
    if len(items) >= DAY_CAP:
        print(f"  [WARN] {DAY_CAP} items = plafond atteint, des tweets de la séance peuvent manquer")
    return kept


def photos_of(tweet):
    """[(tweet_id, url orig)] des photos du tweet et de ses tweets cité / retweeté."""
    out = []
    for t in (tweet, tweet.get("quote"), tweet.get("retweet")):
        if isinstance(t, dict) and t.get("id"):
            out += [(t["id"], photo_orig(m["media_url_https"]))
                    for m in (t.get("extendedEntities") or {}).get("media", [])
                    if m.get("type") == "photo" and m.get("media_url_https")]
    return out


def fmt_ret(r):
    return f"{r:+.2%}" if r is not None else "   n/d"


def main():
    load_dotenv()
    token = os.getenv("APIFY_TOKEN")
    if not token:
        sys.exit("APIFY_TOKEN absent de l'environnement / .env")
    client = ApifyClient(token)

    close = load_close(max(DAYS))
    # Chaque séance de baisse + la séance qui la précède (veille = contexte avant le choc).
    jobs = []
    for d in DAYS:
        jobs += [(previous_session(close, d), d), (d, None)]
    summary = []
    with requests.Session() as session:
        for day, eve_of in jobs:
            r = day_return(close, day)
            label = f"veille du {eve_of}" if eve_of else "séance de baisse"
            print(f"\n== {day}  ({TICKER} {fmt_ret(r)}, {label})")
            day_dir = OUT / str(day)
            cache = day_dir / "tweets.json"
            tweets = tweets_of_day(client, day, cache)
            img_dir = day_dir / "images"
            img_dir.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(tweets, ensure_ascii=False, indent=2), encoding="utf-8")

            seen, counters, n_img, size = set(), {}, 0, 0
            for tweet in tweets:
                for tid, url in photos_of(tweet):
                    if url in seen:
                        continue
                    seen.add(url)
                    counters[tid] = counters.get(tid, 0) + 1
                    dest = img_dir / f"{tid}_{counters[tid]}.jpg"
                    if not dest.exists():
                        if download(session, url, dest) is None:
                            continue
                    n_img += 1
                    size += dest.stat().st_size
            print(f"  {n_img} image(s), {size / 1e6:.1f} Mo -> {img_dir}")
            summary.append({"date": str(day), "kind": "eve" if eve_of else "bear",
                            "eve_of": str(eve_of) if eve_of else None,
                            "return": round(r, 6) if r is not None else None,
                            "tweets": len(tweets), "images": n_img, "dir": str(day_dir)})

    (OUT / "summary.json").write_text(json.dumps({"days": summary}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nRésumé -> {OUT / 'summary.json'}")
    for s in summary:
        tag = f"veille du {s['eve_of']}" if s["eve_of"] else "BAISSE"
        print(f"  {s['date']}  {fmt_ret(s['return'])}  {s['tweets']:4d} tweets  {s['images']:4d} images  {tag}")


if __name__ == "__main__":
    main()
