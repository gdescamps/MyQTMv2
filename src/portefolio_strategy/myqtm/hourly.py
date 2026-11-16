"""Version horaire (barres de 1 h) du classifieur myqtm : barres, news horodatées, features.

Barres EODHD intraday 1 h (depuis 2020-10), heure de New York : barres débutant à 9 h 30, 10 h 30 … 15 h 30 (la
dernière couvre 15 h 30 - 16 h), 7 par séance ; la barre partielle de 16 h est écartée. Cours bruts divisés par le
facteur de division (eod.split_factor). Une ligne = une barre, datée de sa CLÔTURE (10 h 30 … 16 h) : la décision
prise à la clôture d'une barre est exécutée à l'ouverture de la barre suivante (le lendemain 9 h 30 après 16 h).

Features d'une ligne (tout est connu à la clôture de la barre) :
- prix en unités de barres : mêmes familles que le quotidien (features.price_features : RSI 14 barres, EMA 4 … 200
  barres normalisées, cours / volumes des 6 dernières barres et des 6 dernières séances via le pas de 7 barres) ;
- intraday : rang de la barre dans la séance, jour de semaine, rendement de la barre, depuis l'ouverture de la séance,
  depuis la clôture de la veille, écart d'ouverture, amplitude (haut - bas), volume relatif à la même barre des
  20 séances précédentes ;
- coupe transversale : rendement du marché (moyenne des membres) sur 1 et 7 barres, rendement relatif du titre ;
- news horodatées (data/eodhd/news, mêmes filtres que news.py : ≤ NEWS_MAX_SYMBOLS titres cités, contenu ≥
  NEWS_MIN_CHARS) : articles publiés depuis la clôture de la barre précédente (la nuit comptée dans la première
  barre), sur 7 et 35 barres, par catégorie (vendor_sentiment.CATEGORIES) sur 7 barres, heures depuis le dernier
  article ; niveau marché : articles de tous les titres sur 1 et 7 barres ;
- contexte quotidien : toutes les features du modèle quotidien (features.parquet) de la séance PRÉCÉDENTE,
  préfixées d_ (connues avant l'ouverture).

Sorties : data/portefolio_strategy/myqtm_h/{bars,articles,features}.parquet.

    python -m src.portefolio_strategy.myqtm.hourly [--news] [--features]   # défaut : les deux
"""
import ast
import gzip
import json
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from . import config as C
from . import features as F
from .. import config as PC
from .. import eod
from .. import news as N
from .. import universe as U
from ..vendor_sentiment import CATEGORIES, categories

DATA = PC.DATA / "myqtm_h"
OUT = PC.OUT / "myqtm_h"
BARS = DATA / "bars.parquet"
ARTICLES = DATA / "articles.parquet"
FEATURES = DATA / "features.parquet"
STARTS = [f"{h:02d}:30" for h in range(9, 16)]          # débuts des 7 barres de la séance
NEWS_FROM = "2020-09-01"
# Tendances en barres : EMA 1 / 3 / 10 barres, au moins 14 barres (2 séances), 0,03 % par barre (≈ 0,15 % / jour).
TREND = dict(ma_short=1, ma_mid=3, ma_long=10, min_days=14, avg_daily_percent_thr=0.03, bars=True)


# --------------------------------------------------------------------------- barres
def load_bars(symbol):
    p = PC.EODHD / "intraday_1h" / f"{symbol}.parquet"
    if not p.exists():
        return None
    x = pd.read_parquet(p, columns=["timestamp", "open", "high", "low", "close", "volume"])
    t = pd.to_datetime(x["timestamp"], unit="s", utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
    keep = t.dt.strftime("%H:%M").isin(STARTS).to_numpy()
    x, t = x[keep].reset_index(drop=True), t[keep].reset_index(drop=True)
    if len(x) < 500:
        return None
    f = eod.split_factor(symbol, pd.DatetimeIndex(t.dt.normalize()))
    b = pd.DataFrame({"start": t, "o": x["open"] / f, "h": x["high"] / f, "l": x["low"] / f, "c": x["close"] / f,
                      "v": x["volume"].fillna(0) * f})
    b = b.drop_duplicates("start").sort_values("start").reset_index(drop=True)
    r = b["c"].pct_change(fill_method=None)
    b = b[~(r.abs() > 0.5) | r.isna()]                      # erreurs de cotation
    b["date"] = b["start"] + pd.Timedelta(hours=1)
    b.loc[b["start"].dt.strftime("%H:%M") == "15:30", "date"] = b["start"] + pd.Timedelta(minutes=30)
    b["day"] = b["start"].dt.normalize()
    b["bar"] = b.groupby("day").cumcount()
    return b.reset_index(drop=True)


# --------------------------------------------------------------------------- news horodatées
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
                if not hits or len(syms) > PC.NEWS_MAX_SYMBOLS or len(a.get("content") or "") < PC.NEWS_MIN_CHARS \
                        or not a.get("title"):
                    continue
                try:
                    tags = ast.literal_eval(a.get("tags") or "[]") if isinstance(a.get("tags"), str) else a.get("tags")
                except (ValueError, SyntaxError):
                    tags = []
                cats = categories(tags or [], a["title"])
                ts = pd.Timestamp(a["date"]).tz_convert("America/New_York").tz_localize(None)
                key = a["title"].lower()
                for h in hits:
                    for sym, d0, d1 in w[h]:
                        if d0 <= ts < d1:
                            rows.append((ts, sym, key, *[c in cats for c in CATEGORIES]))
    return rows


def build_articles():
    w = N.windows(U.load_universe())
    files = sorted(p for y in (PC.EODHD / "news").iterdir() if int(y.name) >= int(NEWS_FROM[:4])
                   for p in y.glob("*.jsonl.gz") if p.name[:10] >= NEWS_FROM)
    chunks = [(files[i:i + 10], w) for i in range(0, len(files), 10)]
    with ProcessPoolExecutor(C.WORKERS) as ex:
        rows = [r for part in ex.map(_scan, chunks) for r in part]
    a = pd.DataFrame(rows, columns=["ts", "symbol", "key"] + [f"cat_{c}" for c in CATEGORIES])
    a = a.drop_duplicates(["symbol", "key"]).drop(columns="key").sort_values(["symbol", "ts"]).reset_index(drop=True)
    DATA.mkdir(parents=True, exist_ok=True)
    a.to_parquet(ARTICLES, index=False)
    print(f"articles : {len(a)} (titre, article), {a['symbol'].nunique()} titres, {a['ts'].min()} → {a['ts'].max()}",
          flush=True)


def news_features(dates, art):
    """dates : clôtures des barres d'un titre (croissantes) ; art : ses articles → features de news horodatées."""
    t = dates.to_numpy()
    out = {}
    if art is None or art.empty:
        z = np.zeros(len(t))
        out.update({"hn_n1": z, "hn_n7": z, "hn_n35": z, "hn_hours_since": np.full(len(t), 72.0)})
        out.update({f"hn_{c}_n7": z for c in CATEGORIES})
        return pd.DataFrame(out, index=dates)
    ts = art["ts"].to_numpy()
    cum = np.searchsorted(ts, t, side="right")             # articles publiés ≤ clôture de la barre

    def window(cnt, k):
        prev = np.concatenate([np.zeros(k, dtype=cnt.dtype), cnt[:-k]]) if len(cnt) > k else np.zeros_like(cnt)
        return cnt - prev
    out["hn_n1"], out["hn_n7"], out["hn_n35"] = window(cum, 1), window(cum, 7), window(cum, 35)
    last = np.where(cum > 0, ts[np.maximum(cum - 1, 0)], np.datetime64("NaT", "ns"))
    hs = (t - last) / np.timedelta64(1, "h")
    out["hn_hours_since"] = np.where(np.isnan(hs), 72.0, np.minimum(hs, 72.0))
    for c in CATEGORIES:
        tc = art.loc[art[f"cat_{c}"], "ts"].to_numpy()
        out[f"hn_{c}_n7"] = window(np.searchsorted(tc, t, side="right"), 7)
    return pd.DataFrame(out, index=dates).astype("float32")


# --------------------------------------------------------------------------- features d'un titre
def intraday_features(b):
    c, o = b["c"], b["o"]
    first_o = o.groupby(b["day"]).transform("first")
    last_c = c.groupby(b["day"]).transform("last")
    prev_close = last_c.groupby(b["day"]).first().shift(1).reindex(b["day"]).to_numpy()
    v = b["v"].astype(float)
    vbar = v.groupby(b["bar"]).transform(lambda s: s.shift(1).rolling(20, min_periods=5).mean())
    return pd.DataFrame({"h_bar": b["bar"], "h_dow": b["day"].dt.dayofweek, "h_r1": c.pct_change(fill_method=None),
                         "h_r7": c.pct_change(7, fill_method=None), "h_r_open": c / first_o - 1, "h_r_prevclose": c / prev_close - 1,
                         "h_gap": first_o / prev_close - 1, "h_range": (b["h"] - b["l"]) / c,
                         "h_vol_rel": v / vbar.replace(0, np.nan)}, index=b.index)


def _one(args):
    symbol, art, daily = args
    b = load_bars(symbol)
    if b is None or daily is None or daily.empty:
        return None
    px = b.set_index("date")[["o", "c", "v"]]
    pf = F.price_features(px)                               # en unités de barres ; enlève les 200 premières
    b = b[b["date"].isin(pf.index)].reset_index(drop=True)
    pf = pf.reset_index(drop=True)
    idf = intraday_features(b).reset_index(drop=True)
    nf = news_features(pd.DatetimeIndex(b["date"]), art).reset_index(drop=True)
    out = pd.concat([b[["date", "day", "o", "c"]], pf, idf, nf], axis=1)
    d = daily.sort_values("date")
    out = pd.merge_asof(out.sort_values("day"), d.add_prefix("d_").rename(columns={"d_date": "day_d"}),
                        left_on="day", right_on="day_d", allow_exact_matches=False, direction="backward")
    member = daily.set_index("date")["member"].reindex(out["day"]).astype("boolean").fillna(False).to_numpy(bool)
    out = out.drop(columns=["day_d", "d_member", "d_o", "d_c", "d_symbol"], errors="ignore")
    out.insert(0, "member", member)
    out.insert(0, "symbol", symbol)
    num = out.columns.difference(["symbol", "member", "date", "day"])
    out[num] = out[num].astype("float32")
    out = out.sort_values("date")
    return out[out["date"] >= pd.Timestamp("2020-10-01")].reset_index(drop=True)


def add_market(df):
    m = df["member"]
    for k in (1, 7):
        mk = df["h_r1" if k == 1 else "h_r7"].where(m).groupby(df["date"]).transform("mean")
        df[f"h_mkt_r{k}"] = mk.astype("float32")
        df[f"h_rel_mkt_r{k}"] = (df["h_r1" if k == 1 else "h_r7"] - mk).astype("float32")
    n1 = df.groupby("date")["hn_n1"].transform("sum")
    df["hn_mkt_n1"] = n1.astype("float32")
    tot = df.groupby("date")["hn_n1"].sum().sort_index()
    df["hn_mkt_n7"] = df["date"].map(tot.rolling(7, min_periods=1).sum()).astype("float32")
    df["hn_mkt_macro_n7"] = df.groupby("date")["hn_macro_n7"].transform("sum").astype("float32")
    return df


def build_features():
    daily = pd.read_parquet(F.FEATURES)
    syms = sorted(daily.loc[daily["member"], "symbol"].unique())
    art = pd.read_parquet(ARTICLES)
    a_sym = dict(tuple(art.groupby("symbol")))
    d_sym = dict(tuple(daily.groupby("symbol")))
    del daily
    with ProcessPoolExecutor(C.WORKERS) as ex:
        parts = [p for p in ex.map(_one, [(s, a_sym.get(s), d_sym.get(s)) for s in syms]) if p is not None]
    df = pd.concat(parts, ignore_index=True).drop(columns="day")
    df = add_market(df).sort_values(["symbol", "date"]).reset_index(drop=True)
    DATA.mkdir(parents=True, exist_ok=True)
    df.to_parquet(FEATURES, index=False)
    m = df[df["member"]]
    print(f"features horaires : {len(df)} lignes ({len(m)} membres), {df['symbol'].nunique()} titres, "
          f"{len(F.feature_columns(df))} colonnes, {df['date'].min()} → {df['date'].max()} ; "
          f"barres avec ≥ 1 article : {(m['hn_n1'] > 0).mean():.1%}", flush=True)


def use_hourly():
    """Bascule train.py / features sur la version horaire (appelé par train.py --hourly)."""
    F.FEATURES = FEATURES
    C.DATA, C.OUT = DATA, OUT
    C.TREND = TREND


if __name__ == "__main__":
    do_news = "--news" in sys.argv or "--features" not in sys.argv
    do_feat = "--features" in sys.argv or "--news" not in sys.argv
    if do_news:
        build_articles()
    if do_feat:
        build_features()
