"""Téléchargement EODHD -> data/eodhd/ (reprenable, respecte le quota quotidien).

Univers : actions ordinaires US des places principales (NYSE, NASDAQ, NYSE MKT, NYSE ARCA, AMEX, BATS),
actives ET radiées (exchange-symbol-list/US, delisted=0/1).

Étapes (dans cet ordre ; chacune saute ce qui est déjà sur disque) :
  symbols   listes actifs / radiés                         -> symbols_active.parquet, symbols_delisted.parquet
  eod       cours quotidiens, historique complet (1 appel) -> eod/<CODE>.parquet
  earnings  calendrier des résultats, par mois (1 appel)   -> earnings/<YYYY-MM>.parquet
  econ      calendrier économique mondial, par semaine     -> econ/<YYYY>.parquet
  news      flux général (entreprises + macro), par jour   -> news/<YYYY>/<YYYY-MM-DD>.jsonl.gz   (5 appels/requête)
  fund      fondamentaux + historique des résultats        -> fundamentals/<CODE>.json.gz          (10 appels)
            (seulement les titres ayant des cours depuis START)
  corp      splits + dividendes (1 appel chacun)           -> splits/<CODE>.parquet, dividends/<CODE>.parquet
  intraday  barres 1 h depuis 2020-10 (début de l'historique 1 h chez EODHD ; 5 appels, une requête par titre)
                                                           -> intraday_1h/<CODE>.parquet
            (seulement les titres ayant des cours depuis INTRADAY_START)
  sentiment sentiment agrégé par jour et par titre (5 appels par titre, requêtes groupées par SENT_BATCH)
                                                           -> sentiment/<CODE>.parquet  (date, count, normalized)

Le script s'arrête proprement quand le quota du jour est presque épuisé (marge MARGIN) : il suffit de le
relancer le lendemain. La clé EODHD_API_KEY est lue dans .env et n'est jamais affichée.

    python -m myfiles.eodhd.download_eodhd                 # toutes les étapes
    python -m myfiles.eodhd.download_eodhd eod news        # étapes choisies
    python -m myfiles.eodhd.download_eodhd --wait          # au quota : attend minuit GMT et continue (plusieurs jours)
    python -m myfiles.eodhd.download_eodhd --region eu --wait   # actions européennes -> data/eodhd/eu/

Région `eu` (--region eu) : actions ordinaires des places de EU_EXCHANGES, actives ET radiées, une seule
cotation par société (dédoublonnage par ISIN, place d'origine en priorité), sociétés non européennes
cotées en Europe écartées (préfixe ISIN). Les titres sont nommés <CODE>.<PLACE> (ex. MC.PA) et rangés dans
data/eodhd/eu/ avec la même arborescence. Étapes : symbols, eod, fund, corp, intraday, sentiment — les
calendriers (earnings, econ) et le flux de news sont mondiaux et déjà dans data/eodhd/.
"""
import gzip
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()
KEY = os.getenv("EODHD_API_KEY")
BASE = "https://eodhd.com/api"
OUT = Path("data/eodhd")
START = "2011-01-01"            # début de la période utile (15 ans)
NEWS_START = "2011-01-01"
ECON_START = "2011-01-01"
EXCHANGES = {"NYSE", "NASDAQ", "NYSE MKT", "NYSE ARCA", "AMEX", "BATS"}
MARGIN = 3000                   # appels gardés en réserve sur le quota quotidien
WORKERS = 8
INTRADAY_START = "2020-10-01"   # les barres 1 h (et 5 min) n'existent pas avant chez EODHD
STAGES = ["symbols", "eod", "earnings", "econ", "news", "fund", "corp", "intraday", "sentiment"]
SENT_BATCH = 10
SUFFIX = ".US"                  # suffixe de place ajouté au code dans les appels (vide en région eu : déjà dans le code)
REGION = "us"

# Région eu : place EODHD -> préfixe ISIN du pays d'origine. L'ordre donne la priorité de dédoublonnage.
# Les places régionales allemandes (F, STU, MU, HA, DU, HM) sont exclues : doublons de XETRA et de titres étrangers.
EU_EXCHANGES = {"PA": "FR", "AS": "NL", "BR": "BE", "LS": "PT", "XETRA": "DE", "MC": "ES", "LSE": "GB",
                "SW": "CH", "ST": "SE", "CO": "DK", "HE": "FI", "OL": "NO", "VI": "AT", "IR": "IE",
                "LU": "LU", "WAR": "PL", "AT": "GR"}
# Préfixes ISIN gardés (sociétés européennes, y compris holdings à Jersey / Guernesey / Man, Italie cotée ailleurs).
EU_ISIN = set(EU_EXCHANGES.values()) | {"IT", "JE", "GG", "IM", "IS", "CZ", "HU", "MT", "CY", "LI", "EE", "LT", "LV"}
EU_STAGES = ["symbols", "eod", "fund", "corp", "intraday", "sentiment"]


def set_region(name):
    global OUT, SUFFIX, REGION
    if name == "eu":
        OUT, SUFFIX, REGION = Path("data/eodhd/eu"), "", "eu"
    elif name != "us":
        sys.exit(f"région inconnue : {name} (us ou eu)")


class QuotaReached(Exception):
    pass


class Client:
    def __init__(self):
        self.lock = threading.Lock()
        self.session = requests.Session()
        self.used = self.limit = 0
        self.since_sync = 0
        self.sync()

    def _raw(self, path, params):
        return self.session.get(f"{BASE}/{path}", params={"api_token": KEY, "fmt": "json", **params}, timeout=180)

    def sync(self):
        u = self._raw("user", {}).json()
        with self.lock:
            # le compteur EODHD repart à zéro à minuit GMT : apiRequestsDate = jour du compteur
            today = time.strftime("%Y-%m-%d", time.gmtime())
            self.used = int(u.get("apiRequests") or 0) if u.get("apiRequestsDate") == today else 0
            self.limit = int(u.get("dailyRateLimit") or 0)
            self.since_sync = 0

    def get(self, path, cost=1, **params):
        with self.lock:
            if self.used + cost > self.limit - MARGIN:
                raise QuotaReached(f"quota du jour presque atteint ({self.used}/{self.limit})")
            self.used += cost
            self.since_sync += 1
            need_sync = self.since_sync >= 400
        if need_sync:
            self.sync()
        for attempt in range(6):
            try:
                r = self._raw(path, params)
            except requests.RequestException:
                time.sleep(2 ** attempt)
                continue
            if r.status_code == 200:
                try:
                    return r.json()
                except ValueError:
                    return None
            if r.status_code == 404:
                return None
            if r.status_code == 402:
                raise QuotaReached("402 : quota épuisé côté serveur")
            time.sleep(min(60, 2 ** attempt * (5 if r.status_code == 429 else 1)))
        raise RuntimeError(f"échec répété sur {path} (HTTP {r.status_code})")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def fname(code):
    return code.replace("/", "_").replace("\\", "_")


def atomic_parquet(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.rename(path)


def atomic_gz(text, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        f.write(text)
    tmp.rename(path)


class EmptySet:
    """Codes / périodes interrogés sans donnée (pour ne pas les redemander)."""

    def __init__(self, name):
        self.path = OUT / "_state" / f"{name}_empty.txt"
        self.lock = threading.Lock()
        self.items = set(self.path.read_text().split("\n")) - {""} if self.path.exists() else set()

    def add(self, x):
        with self.lock:
            self.items.add(x)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a") as f:
                f.write(x + "\n")

    def __contains__(self, x):
        return x in self.items


def run_parallel(name, todo, fn):
    """Exécute fn(item) en parallèle ; s'arrête proprement au quota. Retourne True si tout est fait."""
    if not todo:
        log(f"[{name}] rien à faire")
        return True
    log(f"[{name}] {len(todo)} éléments à télécharger")
    done = errors = 0
    stop = threading.Event()

    def wrapped(x):
        if stop.is_set():
            return "skip"
        try:
            fn(x)
            return "ok"
        except QuotaReached:
            stop.set()
            return "quota"
        except Exception as e:                                   # noqa: BLE001
            return f"err {type(e).__name__}"

    with ThreadPoolExecutor(WORKERS) as ex:
        futs = [ex.submit(wrapped, x) for x in todo]
        for f in as_completed(futs):
            r = f.result()
            if r == "ok":
                done += 1
                if done % 1000 == 0:
                    log(f"[{name}] {done}/{len(todo)}")
            elif r.startswith("err"):
                errors += 1
    log(f"[{name}] faits {done}, erreurs {errors}, restants {len(todo) - done}" + (" — QUOTA" if stop.is_set() else ""))
    return not stop.is_set()


# ---------------------------------------------------------------- étapes
def stage_symbols(c):
    for dl, name in ((0, "active"), (1, "delisted")):
        p = OUT / f"symbols_{name}.parquet"
        if p.exists():
            continue
        if REGION == "us":
            df = pd.DataFrame(c.get("exchange-symbol-list/US", delisted=dl))
        else:
            parts = []
            for ex in EU_EXCHANGES:
                d = pd.DataFrame(c.get(f"exchange-symbol-list/{ex}", delisted=dl) or [])
                if len(d):
                    d["Code"] = d["Code"].astype(str) + "." + ex          # code complet, ex. MC.PA
                    parts.append(d.assign(Ex=ex))
            df = pd.concat(parts, ignore_index=True)
        atomic_parquet(df, p)
        log(f"[symbols] {name}: {len(df)}")
    return True


def universe():
    a = pd.read_parquet(OUT / "symbols_active.parquet").assign(delisted=False)
    d = pd.read_parquet(OUT / "symbols_delisted.parquet").assign(delisted=True)
    u = pd.concat([a, d], ignore_index=True)
    if REGION == "us":
        return u[(u["Type"] == "Common Stock") & u["Exchange"].isin(EXCHANGES)].drop_duplicates("Code")
    u = u[u["Type"] == "Common Stock"].drop_duplicates("Code")
    pref = u["Isin"].str[:2]
    u = u[u["Isin"].isna() | pref.isin(EU_ISIN)]                          # écarte les sociétés non européennes
    # une cotation par société : place d'origine d'abord, puis titre actif, puis ordre de EU_EXCHANGES
    order = {ex: i for i, ex in enumerate(EU_EXCHANGES)}
    u = u.assign(_home=u["Isin"].str[:2] != u["Ex"].map(EU_EXCHANGES), _rank=u["Ex"].map(order))
    u = u.sort_values(["_home", "delisted", "_rank"], kind="stable")
    u = pd.concat([u[u["Isin"].notna()].drop_duplicates("Isin"), u[u["Isin"].isna()]], ignore_index=True)
    return u.drop(columns=["_home", "_rank"])


def stage_eod(c):
    codes = universe()["Code"].tolist()
    empty = EmptySet("eod")
    todo = [x for x in codes if x not in empty and not (OUT / "eod" / f"{fname(x)}.parquet").exists()]

    def one(code):
        d = c.get(f"eod/{code}{SUFFIX}")
        if not d:
            empty.add(code)
            return
        atomic_parquet(pd.DataFrame(d), OUT / "eod" / f"{fname(code)}.parquet")

    return run_parallel("eod", todo, one)


def month_starts(start):
    d = pd.Timestamp(start).replace(day=1)
    end = pd.Timestamp(date.today()) + pd.offsets.MonthBegin(4)       # + résultats annoncés à venir
    return list(pd.date_range(d, end, freq="MS"))


def stage_earnings(c):
    cur = pd.Timestamp(date.today()).strftime("%Y-%m")
    todo = [m for m in month_starts(START)
            if m.strftime("%Y-%m") >= cur or not (OUT / "earnings" / f"{m:%Y-%m}.parquet").exists()]

    def one(m):
        rows = []
        # fenêtres d'une semaine : la réponse est plafonnée, un mois entier serait tronqué
        for a in pd.date_range(m, m + pd.offsets.MonthEnd(0), freq="7D"):
            b = min(a + pd.Timedelta(days=6), m + pd.offsets.MonthEnd(0))
            d = c.get("calendar/earnings", **{"from": f"{a:%Y-%m-%d}", "to": f"{b:%Y-%m-%d}"})
            rows += (d or {}).get("earnings", []) if isinstance(d, dict) else []
        atomic_parquet(pd.DataFrame(rows).astype(str) if rows else pd.DataFrame(), OUT / "earnings" / f"{m:%Y-%m}.parquet")

    return run_parallel("earnings", todo, one)


def stage_econ(c):
    this_year = date.today().year
    todo = [y for y in range(int(ECON_START[:4]), this_year + 1)
            if y == this_year or not (OUT / "econ" / f"{y}.parquet").exists()]

    def one(y):
        rows = []
        for a in pd.date_range(f"{y}-01-01", f"{y}-12-31", freq="7D"):
            b = min(a + pd.Timedelta(days=6), pd.Timestamp(f"{y}-12-31"))
            off = 0
            while True:
                d = c.get("economic-events", limit=1000, offset=off, **{"from": f"{a:%Y-%m-%d}", "to": f"{b:%Y-%m-%d}"})
                if not isinstance(d, list) or not d:
                    break
                rows += d
                if len(d) < 1000:
                    break
                off += 1000
        atomic_parquet(pd.DataFrame(rows).astype(str) if rows else pd.DataFrame(), OUT / "econ" / f"{y}.parquet")

    return run_parallel("econ", todo, one)


def stage_news(c):
    empty = EmptySet("news")
    last = date.today() - timedelta(days=1)                       # jours complets seulement
    days = [d.date() for d in pd.date_range(NEWS_START, last)]

    def path(d):
        return OUT / "news" / f"{d:%Y}" / f"{d:%Y-%m-%d}.jsonl.gz"

    # Avant 2016 le flux est quasi vide : un sondage par mois évite 5 appels par jour vide.
    todo = []
    by_month = {}
    for d in days:
        if path(d).exists() or f"{d}" in empty:
            continue
        if d.year < 2016:
            by_month.setdefault(f"{d:%Y-%m}", []).append(d)
        else:
            todo.append(d)
    for m, ds in sorted(by_month.items()):
        if m in empty:
            continue
        try:
            a = pd.Timestamp(m + "-01")
            probe = c.get("news", cost=5, limit=1, **{"from": f"{a:%Y-%m-%d}", "to": f"{a + pd.offsets.MonthEnd(0):%Y-%m-%d}"})
        except QuotaReached:
            log("[news] QUOTA pendant le sondage mensuel")
            return False
        if probe:
            todo += ds
        else:
            empty.add(m)
    todo.sort(reverse=True)                                       # du plus récent au plus ancien

    def one(d):
        arts, off = [], 0
        while True:
            r = c.get("news", cost=5, limit=1000, offset=off, **{"from": f"{d}", "to": f"{d}"})
            if not isinstance(r, list) or not r:
                break
            arts += r
            if len(r) < 1000:
                break
            off += 1000
            if off >= 10000:
                log(f"[news] {d} : plafond de pagination atteint (10 000 articles), jour tronqué")
                break
        if not arts:
            empty.add(f"{d}")
            return
        atomic_gz("\n".join(json.dumps(a, ensure_ascii=False) for a in arts) + "\n", path(d))

    return run_parallel("news", todo, one)


def codes_with_recent_prices(since=START):
    """Titres ayant au moins un cours depuis `since` (inutile de demander le reste pour les autres)."""
    cache = OUT / "_state" / "eod_last_date.parquet"
    known = pd.read_parquet(cache).set_index("code")["last"].to_dict() if cache.exists() else {}
    files = sorted((OUT / "eod").glob("*.parquet"))
    new = False
    for f in files:
        if f.stem not in known:
            known[f.stem] = str(pd.read_parquet(f, columns=["date"])["date"].max())
            new = True
    if new:
        atomic_parquet(pd.DataFrame({"code": list(known), "last": list(known.values())}), cache)
    u = universe()
    return [x for x in u["Code"] if known.get(fname(x), "") >= since]


def stage_fund(c):
    empty = EmptySet("fund")
    todo = [x for x in codes_with_recent_prices()
            if x not in empty and not (OUT / "fundamentals" / f"{fname(x)}.json.gz").exists()]

    def one(code):
        d = c.get(f"fundamentals/{code}{SUFFIX}", cost=10)
        if not d:
            empty.add(code)
            return
        atomic_gz(json.dumps(d, ensure_ascii=False), OUT / "fundamentals" / f"{fname(code)}.json.gz")

    return run_parallel("fund", todo, one)


def stage_corp(c):
    ok = True
    for kind, ep in (("splits", "splits"), ("dividends", "div")):
        empty = EmptySet(kind)
        todo = [x for x in codes_with_recent_prices()
                if x not in empty and not (OUT / kind / f"{fname(x)}.parquet").exists()]

        def one(code, ep=ep, kind=kind, empty=empty):
            d = c.get(f"{ep}/{code}{SUFFIX}")
            if not d:
                empty.add(code)
                return
            atomic_parquet(pd.DataFrame(d).astype(str), OUT / kind / f"{fname(code)}.parquet")

        ok = run_parallel(kind, todo, one) and ok
        if not ok:
            break
    return ok


def stage_intraday(c):
    empty = EmptySet("intraday_1h")
    todo = [x for x in codes_with_recent_prices(INTRADAY_START)
            if x not in empty and not (OUT / "intraday_1h" / f"{fname(x)}.parquet").exists()]
    t0 = int(pd.Timestamp(INTRADAY_START, tz="UTC").timestamp())

    def one(code):
        d = c.get(f"intraday/{code}{SUFFIX}", cost=5, interval="1h", **{"from": t0, "to": int(time.time())})
        if not isinstance(d, list) or not d:
            empty.add(code)
            return
        atomic_parquet(pd.DataFrame(d), OUT / "intraday_1h" / f"{fname(code)}.parquet")

    return run_parallel("intraday_1h", todo, one)


def stage_sentiment(c):
    empty = EmptySet("sentiment")
    codes = [x for x in codes_with_recent_prices()
             if x not in empty and not (OUT / "sentiment" / f"{fname(x)}.parquet").exists()]
    todo = [codes[i:i + SENT_BATCH] for i in range(0, len(codes), SENT_BATCH)]

    def one(batch):
        d = c.get("sentiments", cost=5 * len(batch), s=",".join(f"{x}{SUFFIX}" for x in batch),
                  **{"from": START, "to": f"{date.today()}"})
        d = d if isinstance(d, dict) else {}
        for code in batch:
            rows = d.get(f"{code}{SUFFIX}") or []
            if rows:
                atomic_parquet(pd.DataFrame(rows), OUT / "sentiment" / f"{fname(code)}.parquet")
            else:
                empty.add(code)

    return run_parallel("sentiment", todo, one)


def wait_for_reset():
    now = pd.Timestamp.now(tz="UTC")
    nxt = (now + pd.Timedelta(days=1)).normalize() + pd.Timedelta(minutes=5)
    log(f"attente de la remise à zéro du quota ({nxt:%Y-%m-%d %H:%M} GMT)")
    time.sleep(max(60, (nxt - now).total_seconds()))


def main():
    if not KEY:
        sys.exit("EODHD_API_KEY absente de l'environnement")
    if "--region" in sys.argv:
        set_region(sys.argv[sys.argv.index("--region") + 1])
    allowed = STAGES if REGION == "us" else EU_STAGES
    stages = [s for s in sys.argv[1:] if s in allowed] or allowed
    OUT.mkdir(parents=True, exist_ok=True)
    c = Client()
    log(f"région {REGION} ; quota : {c.used}/{c.limit} appels utilisés aujourd'hui ; étapes : {stages}")
    wait = "--wait" in sys.argv
    fns = {"symbols": stage_symbols, "eod": stage_eod, "earnings": stage_earnings, "econ": stage_econ,
           "news": stage_news, "fund": stage_fund, "corp": stage_corp, "intraday": stage_intraday,
           "sentiment": stage_sentiment}
    for s in stages:
        while True:
            try:
                complete = fns[s](c)
            except QuotaReached as e:
                log(f"[{s}] arrêt : {e}")
                complete = False
            if complete:
                break
            c.sync()
            if not wait:
                log(f"ARRÊT au quota pendant « {s} » ({c.used}/{c.limit}). Relancer demain : le script reprend où il s'est arrêté.")
                return
            log(f"quota atteint pendant « {s} » ({c.used}/{c.limit})")
            wait_for_reset()
            c.sync()
    c.sync()
    log(f"TERMINÉ. quota : {c.used}/{c.limit}")


if __name__ == "__main__":
    main()
