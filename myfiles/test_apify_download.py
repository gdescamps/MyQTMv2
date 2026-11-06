"""Télécharge les images et vidéos des tweets Apify (Tweet Scraper V2) -> outputs/apify/download/.

Lit un ou plusieurs JSON produits par les scripts test_apify_*.py (liste de tweets bruts) et,
pour chaque tweet (et son tweet cité `quote` / retweeté `retweet`), parcourt
`extendedEntities.media` :
  - photo                 -> https://pbs.twimg.com/media/<id>.jpg?name=orig (taille originale)
  - video / animated_gif  -> la variante MP4 de plus haut débit (https://video.twimg.com/...)
                             (le .m3u8 HLS est ignoré)
Option --cards : ajoute aussi les images d'aperçu de lien (pbs.twimg.com/card_img, version `orig`).
Les photos de profil / bannières sont ignorées.

Fichiers nommés `<tweet_id>_<n>.<ext>` (tweet_id = celui qui porte le média, donc le tweet cité
pour un média de citation), déjà présents = sautés. Un `manifest.json` relie chaque fichier
à son tweet, son type et son URL source.

Usage :
  python myfiles/test_apify_download.py                       # tous les outputs/apify/*.json
  python myfiles/test_apify_download.py outputs/apify/test_apify_limit_10_2023.json --cards
"""
import argparse
import json
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

SRC_DIR = Path("outputs/apify")
OUT_DIR = SRC_DIR / "download"
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) media-downloader"}
TIMEOUT = 60
RETRIES = 3


def best_mp4(media):
    variants = [v for v in (media.get("video_info") or {}).get("variants", [])
                if v.get("content_type") == "video/mp4" and v.get("url")]
    return max(variants, key=lambda v: v.get("bitrate", 0))["url"] if variants else None


def photo_orig(url):
    """pbs.twimg.com/media/XXX.jpg -> XXX.jpg?name=orig (taille originale)."""
    return url.split("?")[0] + "?name=orig"


def ext_of(url, default):
    path = urlparse(url).path
    if "." in path.rsplit("/", 1)[-1]:
        return path.rsplit(".", 1)[-1].lower()
    fmt = dict(p.split("=", 1) for p in urlparse(url).query.split("&") if "=" in p).get("format")
    return fmt or default


def card_images(tweet):
    """URL `orig` des images d'aperçu de lien (card) du tweet."""
    urls = []
    for b in ((tweet.get("card") or {}).get("legacy") or {}).get("binding_values", []):
        url = ((b.get("value") or {}).get("image_value") or {}).get("url", "")
        if "pbs.twimg.com/card_img" in url and "name=orig" in url:
            urls.append(url)
    return urls


def media_of(tweet, cards):
    """[(tweet_id, kind, url)] pour le tweet et ses tweets cité / retweeté."""
    out = []
    for t in (tweet, tweet.get("quote"), tweet.get("retweet")):
        if not isinstance(t, dict) or not t.get("id"):
            continue
        for m in (t.get("extendedEntities") or {}).get("media", []):
            kind = m.get("type")
            if kind == "photo" and m.get("media_url_https"):
                out.append((t["id"], "photo", photo_orig(m["media_url_https"])))
            elif kind in ("video", "animated_gif"):
                url = best_mp4(m)
                if url:
                    out.append((t["id"], kind, url))
        if cards:
            out += [(t["id"], "card", u) for u in card_images(t)]
    return out


def download(session, url, dest):
    for attempt in range(1, RETRIES + 1):
        try:
            with session.get(url, headers=HEADERS, timeout=TIMEOUT, stream=True) as r:
                r.raise_for_status()
                tmp = dest.with_suffix(dest.suffix + ".part")
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 16):
                        f.write(chunk)
                tmp.rename(dest)
                return dest.stat().st_size
        except requests.RequestException as exc:
            if attempt == RETRIES:
                print(f"  [ERREUR] {url} : {exc}")
                return None
            time.sleep(2 ** attempt)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("files", nargs="*", type=Path, help="JSON de tweets (défaut : outputs/apify/*.json)")
    ap.add_argument("--cards", action="store_true", help="inclure les images d'aperçu de lien (card_img)")
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    files = args.files or sorted(SRC_DIR.glob("*.json"))
    if not files:
        sys.exit(f"aucun JSON dans {SRC_DIR}")
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    # Dédoublonnage par URL (un même média peut revenir via plusieurs tweets / fichiers).
    jobs, seen, counters = [], set(), {}
    for f in files:
        for tweet in json.loads(f.read_text(encoding="utf-8")):
            for tid, kind, url in media_of(tweet, args.cards):
                if url in seen:
                    continue
                seen.add(url)
                counters[tid] = counters.get(tid, 0) + 1
                ext = ext_of(url, "mp4" if kind in ("video", "animated_gif") else "jpg")
                jobs.append((tid, kind, url, args.out / f"{tid}_{counters[tid]}.{ext}"))

    print(f"{len(jobs)} média(s) dans {len(files)} fichier(s) -> {args.out}")
    ok = skipped = failed = total = 0
    with requests.Session() as session:
        for tid, kind, url, dest in jobs:
            if dest.exists():
                skipped += 1
            else:
                size = download(session, url, dest)
                if size is None:
                    failed += 1
                    continue
                ok += 1
                total += size
                print(f"  {kind:12s} {dest.name:32s} {size / 1024:8.0f} Ko")
            manifest[dest.name] = {"tweet_id": tid, "type": kind, "url": url}

    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{ok} téléchargé(s) ({total / 1e6:.1f} Mo), {skipped} déjà présent(s), {failed} erreur(s)"
          f" -> {manifest_path}")


if __name__ == "__main__":
    main()
