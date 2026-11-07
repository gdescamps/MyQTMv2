"""
Resume quotidien de cloture de Wall Street (CNBC), historique + mise a jour.

Pour chaque seance US (calendrier = index de data/QQQ.parquet), cherche l'article
CNBC qui raconte la cloture et en garde le titre, la description et le texte.

Decouverte des candidats (le nom de l'article de cloture a change au fil des ans) :
  - 2007 -> 2022 : sitemap XML de CNBC (https://www.cnbc.com/sitemapAll.xml, ~620k
    URLs, cache dans data/text/cnbc/sitemap_urls.txt). Les slugs sont filtres et
    classes (open-to-close / stock-market-today > "stocks close" > us-stocks > dow-,
    s-p-500-, stocks-, wall-street-...). Les articles pre-ouverture (futures,
    biggest-moves) sont exclus.
  - pages dediees absentes du sitemap (aucune depuis 2022-08) : a partir de 2020 on essaie
    directement les slugs connus (stock-market-today-live-updates, ...-open-to-close-news,
    stock-market-today-live) aux dates du soir precedent et du jour J (la page d'une
    seance est ouverte la veille vers 18h ET avec les futures, puis mise a jour
    jusqu'apres la cloture).
Validation d'un candidat (cf. validate) :
  - page dediee : seance deduite de l'heure de publication (apres 17h ET = seance
    suivante, sinon seance du jour) ;
  - autre article : verbe de mouvement + indice US dans titre / 1er paragraphe, pas de
    titre prospectif ni marche etranger, et le premier jour de la semaine cite doit
    etre celui de la seance ("on Friday") ; a defaut de jour cite, publication le
    jour J apres 15h30 ET.

Politesse : 2 workers, ~1-1.5 s entre requetes. Un 403 (blocage Akamai, deja subi a
4 workers) arrete le run proprement (exit 2) sans marquer de seance manquante.

ATTENTION look-ahead : les pages "live" sont modifiees apres la cloture, parfois
le lendemain. Les colonnes `published` / `modified` (UTC) sont gardees : pour un
backtest, n'utiliser le texte de la seance J qu'a partir de J+1.

Colonne `score` = qualite du candidat retenu : >= 9.5 page de cloture dediee
(live / open-to-close), 8-9 slug "stocks/dow ... close/falls/rises", 6 us-stocks /
stock-market-, 4 dow-/stocks-/s-p-500-, 2 = repli (article de marche du jour, pas
forcement le compte rendu : frequent en 2014-2019 ou CNBC n'a pas de wrap regulier).
Filtrer `score >= 4` pour un corpus plus propre.

Usage personnel / recherche uniquement (CGU CNBC : pas de redistribution).

Sortie  : data/text/cnbc/cnbc_close.parquet  (une ligne par seance trouvee)
          data/text/cnbc/missing.json        (seances sans article valide + URLs essayees)
Usage   : python -m src.download_cnbc_daily                     # incremental (depuis la derniere seance)
          python -m src.download_cnbc_daily --start 2007-01-01  # historique complet (~2h)
          python -m src.download_cnbc_daily --start 2020-03-01 --end 2020-03-31 --retry-missing
          python -m src.download_cnbc_daily --refresh-sitemap   # re-telecharge le sitemap
"""

import re
import json
import time
import random
import argparse
import threading
import datetime as dt
from pathlib import Path
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "text" / "cnbc"
OUT = OUT_DIR / "cnbc_close.parquet"
MISSING = OUT_DIR / "missing.json"
SITEMAP_CACHE = OUT_DIR / "sitemap_urls.txt"
CALENDAR = ROOT / "data" / "QQQ.parquet"

BASE = "https://www.cnbc.com"
SITEMAP_INDEX = f"{BASE}/sitemapAll.xml"
# pages de cloture dediees (slug fixe, date = veille au soir ou J), essayees directement
# quand le sitemap n'en propose pas (il n'en liste plus aucune depuis 2022-08)
DEDICATED_FROM = dt.date(2020, 1, 1)
DEDICATED_SLUGS = [   # (a partir de, slug), le plus recent d'abord
    (dt.date(2022, 6, 1), "stock-market-today-live-updates.html"),
    (dt.date(2020, 6, 1), "stock-market-futures-open-to-close-news.html"),
    (dt.date(2020, 6, 1), "stock-market-news-futures-open-to-close.html"),
    (dt.date(2020, 6, 1), "stock-market-open-to-close-news.html"),
    (dt.date(2020, 1, 1), "stock-market-today-live.html"),
]
DEFAULT_START = dt.date(2007, 1, 1)      # le sitemap CNBC demarre fin 2006
ET = ZoneInfo("America/New_York")
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; research-script)"}
WORKERS = 2          # 4 workers x 18 URLs/seance = blocage Akamai (403) apres ~10 min
PAUSE = 0.8          # s, pause entre deux requetes d'un worker (+ jitter de meme ordre)
MAX_CANDIDATES = 6
SUMMARY_MIN_CHARS = 1200
BODY_MAX_CHARS = 20000
SAVE_EVERY = 200

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MOVE_RE = re.compile(
    r"\b(clos(e|ed|es|ing)|end(ed|s)?|finish(ed|es)?|settled|rose|fell|gain(ed|s)|lost|drop(ped|s)|"
    r"plung(ed|es)|tumbl(ed|es)|jump(ed|s)|rall(ied|ies)|surg(ed|es)|slid|slides|climb(ed|s)|sank|sinks|"
    r"edg(ed|es)|slipp?(ed|s)|soar(ed|s)|rebound(ed|s)|skid(ded|s)|crater(ed|s)|dip(ped|s)|advanc(ed|es)|"
    r"declin(ed|es)|retreat(ed|s)|shed|snapp?(ed|s)|post(ed|s)|logs?|logged)\b", re.I)
PROSPECTIVE_RE = re.compile(r"\b(set to|futures|on track|poll|could|premarket|pre-market|before the bell|"
                            r"what to watch|stocks to watch|here's what|by (end )?20\d\d|Hong Kong|Asia|Europe|"
                            r"China|Japan|London|FTSE|DAX|Nikkei|Hang Seng)\b", re.I)
INDEX_RE = re.compile(r"\b(Dow|S&P|Nasdaq|stocks|equities|Wall Street|major averages)\b", re.I)
DEDICATED_RE = re.compile(r"/(stock-market-(today-)?live(-today|-updates)?(-[a-z0-9-]*)?|[a-z-]*open-to-close[a-z-]*)\.html$")
URL_RE = re.compile(r"^https://www\.cnbc\.com/(\d{4})/(\d\d)/(\d\d)/([^/]+)\.html$")
# slugs a exclure : pre-ouverture, intraday, opinion, autres marches
EXCLUDE_RE = re.compile(
    r"futures(?!.*open-to-close)|biggest-moves|premarket|pre-market|midday|cramer|asia|europe|"
    r"china-|japan|oil-|gold-|bitcoin|crypto|treasur|bond-|dollar-|week-ahead|what-to-watch|"
    r"\d+-things|before-the-bell|options-|etf-", re.I)
SLUG_SCORES = [   # (regex, score) : le premier qui matche donne le score
    (re.compile(r"^(stock-market-(today-)?live(-today|-updates)?(-[a-z0-9-]*)?|[a-z-]*open-to-close[a-z-]*)$"), 10),
    (re.compile(r"(^|-)(stocks?|dow|s-?p-500|sp-500|nasdaq|wall-street)-(.*-)?(clos(e|es|ed)|end(s|ed)?|finish(es|ed)?|settle)"), 9),
    (re.compile(r"(^(stocks?|s-?p-500|sp-500|nasdaq|us-stocks?)|(^|-)dow)-(falls?|rises?|jumps?|surges?|tumbles?|rall(y|ies)|drops?|"
                r"slides?|gains?|plunges?|soars?|sinks?|climbs?|edges?|slips?|rebounds?|sheds?|adds?|posts?)(-|$)"), 8),
    (re.compile(r"^us-stocks?|^us-markets?$|^stock-market-"), 6),
    (re.compile(r"^(dow|s-?p-500|stocks|wall-street|nasdaq)-"), 4),
    (re.compile(r"(^|-)(stocks?|dow|s-?p-500|sp-500|nasdaq|market)(-|$)"), 2),   # repli (2014-2019 surtout)
]

_tls = threading.local()
STOP = threading.Event()


class Blocked(Exception):
    """CNBC (Akamai) renvoie 403 : IP bloquee pour trop de requetes -> on arrete tout."""


def session():
    if not hasattr(_tls, "s"):
        _tls.s = requests.Session()
        _tls.s.headers.update(HEADERS)
    return _tls.s


def http_get(url, tries=4):
    """GET avec backoff sur 429/5xx/erreur reseau. Retourne la Response (404 compris) ou None.
    Leve Blocked sur 403 (blocage Akamai : inutile d'insister, il faut attendre)."""
    for k in range(tries):
        if STOP.is_set():
            raise Blocked(url)
        try:
            r = session().get(url, timeout=25)
            if r.status_code == 403:
                STOP.set()
                raise Blocked(url)
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"HTTP {r.status_code}")
            return r
        except requests.RequestException:
            if k == tries - 1:
                return None
            time.sleep(2 ** k * 2 + random.random())
    return None


# ---------------------------------------------------------------- sitemap ---

def load_sitemap(refresh=False):
    """Liste des URLs d'articles CNBC (cache local)."""
    if SITEMAP_CACHE.exists() and not refresh:
        return SITEMAP_CACHE.read_text().split()
    print("[sitemap] telechargement de l'index...")
    idx = re.findall(r"<loc>(.*?)</loc>", http_get(SITEMAP_INDEX).text)
    urls = []
    for u in idx:
        r = http_get(u)
        if r is None or not r.ok:
            print(f"[sitemap] WARN {u} indisponible")
            continue
        urls += re.findall(r"<loc>(.*?)</loc>", r.text)
        print(f"[sitemap] {u.rsplit('/', 1)[1]}: total {len(urls)}")
    urls = sorted(set(u for u in urls if URL_RE.match(u)))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SITEMAP_CACHE.write_text("\n".join(urls))
    return urls


def slug_score(slug):
    if EXCLUDE_RE.search(slug):
        return 0
    for rx, sc in SLUG_SCORES:
        if rx.search(slug):
            return sc
    return 0


def index_sitemap(urls):
    """{date: [(score, url), ...]} pour les slugs pertinents."""
    by_date = {}
    for u in urls:
        m = URL_RE.match(u)
        if not m:
            continue
        sc = slug_score(m.group(4))
        if sc:
            d = dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            by_date.setdefault(d, []).append((sc, u))
    return by_date


# ---------------------------------------------------------------- articles ---

def parse_article(html):
    s = BeautifulSoup(html, "lxml")

    def meta(p):
        el = s.find("meta", property=p) or s.find("meta", attrs={"name": p})
        return el.get("content") if el else None

    # page live : FeaturedContent = le resume en tete (mis a jour apres la cloture)
    root = (s.select_one("div.FeaturedContent-articleBody")
            or s.select_one("div.ArticleBody-articleBody")
            or s.select_one("div[data-module=ArticleBody]"))
    paras = []
    if root is not None:
        for p in root.find_all("p"):
            t = " ".join(p.get_text(" ", strip=True).split())
            if len(t) > 40:
                paras.append(t)
    summary, n = [], 0
    for p in paras:
        summary.append(p)
        n += len(p)
        if n >= SUMMARY_MIN_CHARS:
            break
    return {
        "title": (meta("og:title") or "").strip(),
        "description": (meta("og:description") or meta("description") or "").strip(),
        "published": meta("article:published_time"),
        "modified": meta("article:modified_time"),
        "summary": "\n".join(summary),
        "body": "\n".join(paras)[:BODY_MAX_CHARS],
    }


def to_et(ts):
    try:
        return pd.Timestamp(ts).tz_convert(ET)
    except Exception:
        return None


def validate(art, day, prev_day, dedicated):
    """True si l'article raconte la cloture de la seance `day`.

    `dedicated` = page de cloture dediee (open-to-close / stock-market-today) : la
    seance se deduit de l'heure de publication (page ouverte apres 17h ET = seance
    suivante, sinon seance du jour). Le jour de la semaine cite n'est pas fiable sur
    ces pages ("a report on Thursday" dans le compte rendu du vendredi).
    Sinon il faut un verbe de mouvement et un indice US dans le titre + 1er
    paragraphe (evite "Twitter stock falls below IPO price"), puis le premier jour
    de la semaine cite doit etre celui de la seance. La description n'est PAS
    utilisee pour le jour : sur les pages live elle resume la seance precedente.
    """
    if not art["summary"]:
        return False
    first = art["summary"].split("\n", 1)[0]
    head = f"{art['title']} {art['summary'][:900]}"
    pub = to_et(art["published"])
    if dedicated and pub is not None:
        if pub.hour >= 17:
            return prev_day <= pub.date() < day
        return pub.date() == day
    if not dedicated:
        lead = f"{art['title']} {first}"
        if PROSPECTIVE_RE.search(art["title"]) or not (MOVE_RE.search(lead) and INDEX_RE.search(lead)):
            return False
    wd = WEEKDAYS[day.weekday()]
    m = re.search(r"\b(" + "|".join(WEEKDAYS) + r")\b", head)
    if m:   # le premier jour cite doit etre celui de la seance
        return m.group(1) == wd
    return pub is not None and pub.date() == day and (pub.hour, pub.minute) >= (15, 30)


def candidates(day, prev_day, by_date):
    """URLs a essayer pour la seance `day`, la plus probable d'abord."""
    out = list(by_date.get(day, []))
    # pages "open-to-close"/live datees de la veille au soir
    d = day - dt.timedelta(days=1)
    while d >= prev_day:
        out += [(sc - 0.5, u) for sc, u in by_date.get(d, []) if sc >= 10]
        d -= dt.timedelta(days=1)
    if day >= DEDICATED_FROM and not any(sc >= 9.5 for sc, _ in out):
        # le sitemap ne liste pas toutes les pages dediees (aucune depuis 2022-08) :
        # on essaie les slugs connus, datees de la veille au soir (dimanche pour lundi) puis J
        dates = []
        d = day - dt.timedelta(days=1)
        while d >= prev_day:
            dates.append(d)
            d -= dt.timedelta(days=1)
        dates.append(day)
        slugs = [s for since, s in DEDICATED_SLUGS if day >= since]
        out = [(12 - 0.1 * i, f"{BASE}/{d:%Y/%m/%d}/{s}")
               for i, (s, d) in enumerate((s, d) for s in slugs for d in dates)] + out
        limit = MAX_CANDIDATES + len(slugs) * len(dates)
    else:
        limit = MAX_CANDIDATES
    seen, uniq = set(), []
    for sc, u in sorted(out, key=lambda x: -x[0]):
        if u not in seen:
            seen.add(u)
            uniq.append((sc, u))
    return uniq[:limit]


def fetch_day(day, prev_day, by_date):
    tried = []
    for sc, u in candidates(day, prev_day, by_date):
        r = http_get(u)
        time.sleep(PAUSE + random.random() * PAUSE)
        if r is None or r.status_code != 200:
            tried.append(f"{u} [{getattr(r, 'status_code', 'ERR')}]")
            continue
        art = parse_article(r.text)
        if validate(art, day, prev_day, bool(DEDICATED_RE.search(u))):
            return {"date": pd.Timestamp(day), "url": u, "score": float(sc), **art}, tried
        tried.append(f"{u} [rejet: {art['title'][:60]}]")
    return None, tried


# ---------------------------------------------------------------- main ---

def trading_days(start, end):
    idx = pd.read_parquet(CALENDAR, columns=[]).index
    days = [d.date() for d in idx]
    # seances posterieures au calendrier (QQQ pas encore rafraichi) : jours ouvres
    last = days[-1]
    for d in pd.bdate_range(last + dt.timedelta(days=1), end):
        days.append(d.date())
    pairs = list(zip(days[1:], days[:-1]))   # (seance, seance precedente)
    return [(d, p) for d, p in pairs if start <= d <= end]


def save(rows_df, missing):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows_df.sort_values("date").reset_index(drop=True).to_parquet(OUT, index=False)
    MISSING.write_text(json.dumps(dict(sorted(missing.items())), indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", type=dt.date.fromisoformat)
    ap.add_argument("--end", type=dt.date.fromisoformat)
    ap.add_argument("--retry-missing", action="store_true", help="re-essaie les seances deja marquees manquantes")
    ap.add_argument("--refetch", action="store_true", help="re-telecharge aussi les seances deja presentes")
    ap.add_argument("--refresh-sitemap", action="store_true")
    ap.add_argument("--workers", type=int, default=WORKERS)
    a = ap.parse_args()

    have = pd.read_parquet(OUT) if OUT.exists() else pd.DataFrame(columns=["date"])
    missing = json.loads(MISSING.read_text()) if MISSING.exists() else {}
    today_et = dt.datetime.now(ET).date()
    end = a.end or today_et
    if a.start:
        start = a.start
    elif len(have):
        start = pd.Timestamp(have["date"].max()).date() + dt.timedelta(days=1)
    else:
        start = DEFAULT_START

    done = set(pd.to_datetime(have["date"]).dt.date) if len(have) else set()
    # une seance presente n'est pas manquante ; une entree issue d'un blocage (403) non plus
    missing = {k: v for k, v in missing.items()
               if dt.date.fromisoformat(k) not in done and not any("[403]" in x for x in v)}
    todo = [(d, p) for d, p in trading_days(start, end)
            if (a.refetch or d not in done) and (a.retry_missing or str(d) not in missing)]
    # la seance du jour n'est racontee qu'apres 16h ET
    now = dt.datetime.now(ET)
    todo = [(d, p) for d, p in todo if d < today_et or now.hour >= 17]
    print(f"[cnbc] {len(todo)} seances a traiter ({start} -> {end}), {len(done)} deja presentes")
    if not todo:
        return

    need_sitemap = any(d < dt.date(2023, 1, 1) for d, _ in todo) or a.refresh_sitemap
    by_date = index_sitemap(load_sitemap(a.refresh_sitemap)) if need_sitemap else {}

    rows, n_ok, n_ko, blocked = [], 0, 0, False
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(fetch_day, d, p, by_date): d for d, p in todo}
        for i, f in enumerate(as_completed(futs), 1):
            if f.cancelled():
                continue
            d = futs[f]
            try:
                row, tried = f.result()
            except Blocked:
                if not blocked:
                    print("[cnbc] 403 : CNBC bloque l'IP (Akamai). Arret ; relancer plus tard "
                          "(les seances deja traitees sont sauvegardees).", flush=True)
                    for g in futs:
                        g.cancel()
                blocked = True
                continue
            except Exception as e:      # ne jamais perdre le lot pour une page
                row, tried = None, [f"exception: {e!r}"]
            if row:
                rows.append(row)
                missing.pop(str(d), None)
                n_ok += 1
            elif d not in done:         # un echec de --refetch ne rend pas la seance manquante
                missing[str(d)] = tried
                n_ko += 1
            if i % SAVE_EVERY == 0:
                have = flush(have, rows, missing)
                el = time.time() - t0
                print(f"[cnbc] {i}/{len(todo)}  ok={n_ok} manquants={n_ko}  "
                      f"{el / 60:.1f} min, reste ~{el / i * (len(todo) - i) / 60:.0f} min", flush=True)
    have = flush(have, rows, missing)
    print(f"[cnbc] {'INTERROMPU (403)' if blocked else 'termine'} : {len(have)} seances dans "
          f"{OUT.relative_to(ROOT)}, {len(missing)} manquantes dans {MISSING.relative_to(ROOT)}")
    if blocked:
        raise SystemExit(2)


def flush(have, rows, missing):
    """Fusionne les nouvelles lignes (remplacent les anciennes de meme date) et sauvegarde."""
    new = pd.DataFrame(rows)
    rows.clear()
    if len(new):
        keep = have[~pd.to_datetime(have["date"]).isin(new["date"])] if len(have) else have
        have = pd.concat([keep, new], ignore_index=True) if len(keep) else new
    save(have, missing)
    return have


if __name__ == "__main__":
    main()
