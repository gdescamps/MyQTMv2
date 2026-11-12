"""Sentiment du fournisseur (EODHD) et catégories de news par (titre, jour) → data/portefolio_strategy/vendor_sentiment.parquet.

Chaque article de data/eodhd/news porte un champ `sentiment` {polarity, neg, neu, pos} et une liste de `tags`
en texte libre. On les agrège par (titre, jour) avec les mêmes règles que news.py (univers dynamique, jour =
horodatage UTC + 4 h, articles généralistes écartés), mais sans plafond d'articles : aucun coût par article.

Catégories : les tags EODHD ne forment pas une liste fermée (≈ 150 000 valeurs) et manquent presque
totalement en 2024. On les ramène à CATEGORIES par mots-clés, cherchés dans les tags ET dans le titre
(toujours présent) ; un article peut appartenir à plusieurs catégories.

Par (titre, jour) : n, polarity/pos/neg/neu moyens, polarity min/max, et par catégorie le nombre d'articles
(`cat_<c>_n`) et leur polarité moyenne (`cat_<c>_pol`).

    python -m src.portefolio_strategy.vendor_sentiment
"""
import ast
import gzip
import json
import re
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from . import config as C
from . import news as N
from . import universe as U

VENDOR = C.DATA / "vendor_sentiment.parquet"

CATEGORIES = {
    "earnings": r"EARNINGS?|QUARTER(LY)?|FINANCIAL RESULTS|REVENUES?|EPS|GUIDANCE|NET INCOME|OUTLOOK",
    "analyst": r"PRICE[- ]TARGET|UPGRADES?D?|DOWNGRADES?D?|ANALYSTS?|RATINGS?|CONSENSUS|ESTIMATES?|ZACKS|OVERWEIGHT|UNDERWEIGHT",
    "legal": r"CLASS ACTION|LAWSUITS?|LEGAL|INVESTIGATIONS?|ANTITRUST|SETTLEMENT|FRAUD|SUED?|PROBE|LITIGATION",
    "mna": r"MERGERS?|ACQUISITIONS?|ACQUIRES?D?|M-A|TAKEOVER|BUYOUT",
    "capital": r"DIVIDENDS?|BUYBACKS?|REPURCHASE|OFFERING|STOCK SPLIT",
    "product": r"PRODUCTS?|LAUNCH(ES|ED)?|PARTNERSHIP|CONTRACTS?|FDA|APPROVAL|UNVEILS?",
    "management": r"CEO|CFO|DIRECTORS AND OFFICERS|APPOINTS?|RESIGNS?|EXECUTIVE|LAYOFFS?",
    "macro": r"INFLATION|FED|FEDERAL RESERVE|INTEREST RATES?|TARIFFS?|ECONOMY|RECESSION|MARKETS|S&P 500|NASDAQ COMPOSITE|TREASURY|JOBS REPORT",
    "press": r"PRESS RELEASES?|COMPANY ANNOUNCEMENT|ANNOUNCES?",
}
_RX = {c: re.compile(rf"\b(?:{p})\b") for c, p in CATEGORIES.items()}


def categories(tags, title):
    """Catégories d'un article d'après ses tags et son titre."""
    text = " | ".join(tags).upper() + " | " + (title or "").upper()
    return [c for c, rx in _RX.items() if rx.search(text)]


def _lit(x, default):
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            return default
    return x if x is not None else default


def _scan(args):
    files, w = args
    rows = []
    for p in files:
        with gzip.open(p, "rt") as f:
            for line in f:
                if '"symbols": "[]"' in line:
                    continue
                a = json.loads(line)
                syms = _lit(a.get("symbols"), [])
                hits = [s for s in syms if s in w]
                s = _lit(a.get("sentiment"), None)
                if not hits or len(syms) > C.NEWS_MAX_SYMBOLS or not a.get("title") or not isinstance(s, dict) \
                        or s.get("polarity") is None or len(a.get("content") or "") < C.NEWS_MIN_CHARS:
                    continue
                day = N.news_day(a["date"])
                cats = categories(_lit(a.get("tags"), []), a["title"])
                key = re.sub(r"[^a-z0-9]+", " ", a["title"].lower()).strip()
                for h in hits:
                    for sym, d0, d1 in w[h]:
                        if d0 <= day < d1:
                            rows.append((day, sym, key, s["polarity"], s.get("pos"), s.get("neg"), s.get("neu"),
                                         *[c in cats for c in CATEGORIES]))
    return rows


def aggregate(df):
    """Une ligne par article (day, symbol, polarity, pos, neg, neu, booléens de catégorie) → une par (titre, jour)."""
    g = df.groupby(["day", "symbol"], sort=True)
    out = g.agg(n=("polarity", "size"), polarity=("polarity", "mean"), pos=("pos", "mean"), neg=("neg", "mean"),
                neu=("neu", "mean"), polarity_min=("polarity", "min"), polarity_max=("polarity", "max"))
    for c in CATEGORIES:
        sub = df[df[c]].groupby(["day", "symbol"])["polarity"].agg(["size", "mean"])
        out[f"cat_{c}_n"] = sub["size"].reindex(out.index).fillna(0).astype(int)
        out[f"cat_{c}_pol"] = sub["mean"].reindex(out.index)
    return out.reset_index()


def main():
    w = N.windows(U.load_universe())
    start = pd.Timestamp(C.START_DATE)
    files = sorted(p for y in (C.EODHD / "news").iterdir() if int(y.name) >= start.year
                   for p in y.glob("*.jsonl.gz") if p.name[:10] >= C.START_DATE)
    chunks = [(files[i:i + 10], w) for i in range(0, len(files), 10)]
    with ProcessPoolExecutor(C.WORKERS) as ex:
        rows = [r for part in ex.map(_scan, chunks) for r in part]
    cols = ["day", "symbol", "key", "polarity", "pos", "neg", "neu", *CATEGORIES]
    df = pd.DataFrame(rows, columns=cols)
    df = df[df["day"] >= start].drop_duplicates(["day", "symbol", "key"])
    out = aggregate(df)
    out.to_parquet(VENDOR, index=False)
    share = {c: round(float(df[c].mean()), 2) for c in CATEGORIES}
    print(f"sentiment fournisseur : {len(df)} articles → {len(out)} couples (titre, jour), {out['symbol'].nunique()} titres")
    print(f"polarité moyenne {df['polarity'].mean():.3f} (écart-type {df['polarity'].std():.3f}) ; "
          f"part d'articles par catégorie : {share} ; sans catégorie : {1 - df[list(CATEGORIES)].any(axis=1).mean():.2f}")
    y = out["day"].dt.year
    print("couples par année :", out.groupby(y).size().to_dict())
    return out


def load_vendor():
    return pd.read_parquet(VENDOR) if VENDOR.exists() else None


if __name__ == "__main__":
    main()
