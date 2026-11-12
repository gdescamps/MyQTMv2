"""Extraction des news par (titre, jour) depuis data/eodhd/news (lecture seule) → data/portefolio_strategy/news/.

- seulement à partir de START_DATE (2020-01-01 : le LLM PIT ne connaît rien après 2019-12) ;
- seulement les titres de l'univers dynamique, pendant leur appartenance (+ NEWS_WARMUP_DAYS avant l'entrée) ;
- jour d'une news = date de (horodatage UTC + 4 h) : un article publié après 20:00 UTC (clôture US) compte
  pour le lendemain → la note d'un jour ne contient rien de postérieur à la clôture de ce jour ;
- articles généralistes (> NEWS_MAX_SYMBOLS titres cités) et trop courts écartés, doublons de titre retirés,
  au plus NEWS_MAX_ARTICLES articles par (titre, jour), les plus ciblés d'abord.

    python -m src.portefolio_strategy.news
"""
import ast
import gzip
import json
import re
from concurrent.futures import ProcessPoolExecutor
from urllib.parse import urlparse

import pandas as pd

from . import config as C
from . import universe as U

NEWS_DIR = C.DATA / "news"


def windows(u):
    """ticker de news (sans suffixe _old) → [(symbole univers, début, fin)]."""
    w = {}
    far = pd.Timestamp("2100-01-01")
    for s, a, b in u[["symbol", "start", "end"]].itertuples(index=False):
        w.setdefault(s.replace("_old", "") + ".US", []).append(
            (s, a - pd.Timedelta(days=C.NEWS_WARMUP_DAYS), b if pd.notna(b) else far))
    return w


def news_day(ts):
    return (pd.Timestamp(ts).tz_convert("UTC") + pd.Timedelta(hours=24 - C.NEWS_CUTOFF_HOUR_UTC)).tz_localize(None).normalize()


def _scan(args):
    files, w = args
    rows = []
    for p in files:
        with gzip.open(p, "rt") as f:
            for line in f:
                if '"symbols": "[]"' in line:
                    continue
                a = json.loads(line)
                try:
                    syms = a.get("symbols") or []
                    syms = ast.literal_eval(syms) if isinstance(syms, str) else syms
                except (ValueError, SyntaxError):
                    continue
                hits = [s for s in syms if s in w]
                content = a.get("content") or ""
                if not hits or len(syms) > C.NEWS_MAX_SYMBOLS or len(content) < C.NEWS_MIN_CHARS or not a.get("title"):
                    continue
                day = news_day(a["date"])
                for h in hits:
                    for sym, d0, d1 in w[h]:
                        if d0 <= day < d1:
                            rows.append((day, sym, a["date"], a["title"], content[:C.NEWS_MAX_CHARS], len(content),
                                         len(syms), urlparse(a.get("link") or "").netloc.replace("www.", "")))
    return rows


def select(df):
    """Dédoublonne et garde les NEWS_MAX_ARTICLES articles les plus ciblés par (titre, jour)."""
    df = df.copy()
    df["k"] = df["title"].str.lower().map(lambda t: re.sub(r"[^a-z0-9]+", " ", t).strip())
    df = df.drop_duplicates(["day", "symbol", "k"]).drop(columns="k")
    df = df.sort_values(["day", "symbol", "n_symbols", "full_len"], ascending=[True, True, True, False])
    return df.groupby(["day", "symbol"], sort=False).head(C.NEWS_MAX_ARTICLES).reset_index(drop=True)


def main():
    w = windows(U.load_universe())
    NEWS_DIR.mkdir(parents=True, exist_ok=True)
    start = pd.Timestamp(C.START_DATE)
    cols = ["day", "symbol", "ts", "title", "content", "full_len", "n_symbols", "site"]
    carry = pd.DataFrame(columns=cols)
    for ydir in sorted((C.EODHD / "news").iterdir()):
        if int(ydir.name) < start.year:
            continue
        files = sorted(p for p in ydir.glob("*.jsonl.gz") if p.name[:10] >= C.START_DATE)
        chunks = [(files[i:i + 10], w) for i in range(0, len(files), 10)]
        with ProcessPoolExecutor(C.WORKERS) as ex:
            rows = [r for part in ex.map(_scan, chunks) for r in part]
        df = pd.concat([carry, pd.DataFrame(rows, columns=cols)], ignore_index=True)
        df = df[df["day"] >= start]
        nxt = df["day"].dt.year > int(ydir.name)       # articles du 31/12 après 20:00 UTC → année suivante
        carry, df = df[nxt], select(df[~nxt])
        df.to_parquet(NEWS_DIR / f"{ydir.name}.parquet", index=False)
        print(f"{ydir.name} : {len(df)} articles, {df.groupby(['day', 'symbol']).ngroups} couples (titre, jour), "
              f"{df['symbol'].nunique()} titres", flush=True)


def load_news(years=None):
    files = sorted(NEWS_DIR.glob("*.parquet"))
    if years:
        files = [f for f in files if int(f.stem) in years]
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


if __name__ == "__main__":
    main()
