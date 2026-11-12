"""Features par (titre, jour) — portage de ref/MyQTM/src/data_transform_*.py sur data/eodhd.

Tout est point-in-time : une ligne datée J n'utilise que ce qui est connu à la clôture de J.
Fuites de ref/MyQTM corrigées ici :
- fondamentaux datés de leur dépôt (filing_date + 1 j), pas de la fin de trimestre ;
- indicateurs macro datés de leur publication (calendrier économique), pas de la période mesurée ;
- recommandations d'analystes et « ratings » FMP abandonnés (EODHD n'en a qu'un instantané d'aujourd'hui),
  remplacés par l'historique des surprises de résultats (daté de la publication + 1 j) ;
- VIX (absent de data/eodhd) remplacé par la volatilité réalisée 20 j de l'indice de l'univers ;
- colonnes « open » (cours brut) et « stock » (identifiant du titre) retirées des features.

    python -m src.portefolio_strategy.features
"""
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from . import config as C
from . import eod
from . import universe as U

FEATURES = C.DATA / "features.parquet"
INDEX = C.DATA / "index.parquet"
SENTIMENTS = C.DATA / "sentiments.parquet"
ROWS_START = "2019-06-01"
META = ["date", "symbol", "member", "o", "c"]
EPS = 1e-9


# --------------------------------------------------------------------------- prix
def _norm(x, mn, mx):
    return (100.0 * (x - mn) / (mx - mn + EPS)).round(2)


def rsi14(c):
    d = c.diff()
    g = d.clip(lower=0).rolling(14).mean()
    l = (-d).clip(lower=0).rolling(14).mean()
    r = 100 - 100 / (1 + g / l)
    r = r.where(l != 0, 100.0)
    return r.where(g.notna(), 50.0).round(2)


def price_features(px):
    """px : cours ajustés (o, c, v) indexés par date croissante. Normalisation min/max sur l'historique passé."""
    c, v = px["c"], px["v"].astype(float)
    mn, mx = c.expanding().min(), c.expanding().max()
    f = {"rsi_14": rsi14(c)}
    for n in (4, 8, 30, 50, 200):
        f[f"ema_{n}"] = _norm(c.rolling(n, min_periods=1).mean(), mn, mx)
    for k in range(C.TS_SIZE):
        f[f"ts_d_close_{k}"] = _norm(c.shift(k), mn, mx)
        f[f"ts_w_close_{k}"] = _norm(c.shift(7 * k), mn, mx)
    vmn, vmx = v.expanding().min(), v.expanding().max()
    vw = v.rolling(7).mean()
    for k in range(C.TS_SIZE):
        f[f"ts_d_volume_{k}"] = _norm(v.shift(k), vmn, vmx)
        f[f"ts_w_volume_{k}"] = _norm(vw.shift(7 * k), vmn, vmx)
    for col in ("rsi_14", "ema_4", "ema_8", "ema_30", "ema_50", "ema_200"):
        for k in range(C.TS_SIZE + 1):
            f[f"ts_{col}_{k}"] = f[col].shift(k)
    out = pd.DataFrame(f, index=px.index)
    out["month"] = px.index.month
    return out.iloc[200:]          # séries incomplètes (EMA 200)


# --------------------------------------------------------------------------- indice de l'univers
def build_index(u, caps, closes):
    """Indice pondéré par la capitalisation des membres (poids = capitalisation de la fin du mois précédent)."""
    ret = closes.pct_change(fill_method=None)
    cap = caps.pivot_table(index="month", columns="symbol", values="cap")
    cap.index = pd.PeriodIndex(cap.index, freq="M")
    prev_month = (ret.index.to_period("M") - 1)
    w = cap.reindex(prev_month).set_axis(ret.index).reindex(columns=ret.columns)
    member = pd.DataFrame({s: U.member_mask(u, s, ret.index) for s in ret.columns}, index=ret.index)
    w = w.where(member & ret.notna()).fillna(0.0)
    w = w.div(w.sum(axis=1).replace(0, np.nan), axis=0)
    r = (w * ret.fillna(0.0)).sum(axis=1)
    r[w.sum(axis=1).isna() | (w.sum(axis=1) == 0)] = 0.0
    level = 100 * (1 + r).cumprod()
    return pd.DataFrame({"ret": r, "level": level, "n": (w > 0).sum(axis=1)})


def market_features(index):
    """Remplace le VIX : volatilité réalisée 20 j annualisée de l'indice (en %)."""
    vol = (index["ret"].rolling(20).std() * np.sqrt(252) * 100).dropna()
    mn, mx = vol.expanding().min(), vol.expanding().max()
    f = {"vixp_ema_50": _norm(vol.rolling(50, min_periods=1).mean(), mn, mx)}
    for k in range(C.TS_SIZE):
        f[f"ts_vixp_w_close_{k}"] = _norm(vol.shift(7 * k), mn, mx)
    return pd.DataFrame(f, index=vol.index).iloc[60:]


# --------------------------------------------------------------------------- blocs datés (fusion « dernier connu »)
def _ts_block(df, cols, prefix, size):
    """df trié par date de disponibilité ; ajoute {prefix}{col}_{k} = valeur k publications plus tôt."""
    out = pd.DataFrame(index=df.index)
    for col in cols:
        for k in range(size):
            out[f"{prefix}{col}_{k}"] = df[col].shift(k)
    return out


def asof_join(dates, block, days_col=None):
    """Aligne un bloc indexé par date de disponibilité sur `dates` (dernière valeur connue <= date)."""
    left = pd.DataFrame({"date": pd.DatetimeIndex(dates).astype("datetime64[ns]")})
    if block is None or block.empty:
        return pd.DataFrame(index=dates)
    right = block.sort_index().reset_index(names="_avail")
    right["_avail"] = right["_avail"].astype("datetime64[ns]")
    m = pd.merge_asof(left, right, left_on="date", right_on="_avail", direction="backward")
    if days_col:
        m[days_col] = (m["date"] - m["_avail"]).dt.days.fillna(-1).astype(int)
    return m.drop(columns=["_avail"]).set_index("date")


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def key_metrics(fund):
    """debtToEquity / debtToAssets trimestriels, disponibles le lendemain du dépôt (filing_date)."""
    bs = ((fund.get("Financials") or {}).get("Balance_Sheet") or {}).get("quarterly") or {}
    rows = []
    for period, r in bs.items():
        avail = pd.Timestamp(r.get("filing_date") or period) + pd.Timedelta(days=1 if r.get("filing_date") else 46)
        debt = _num(r.get("shortLongTermDebtTotal"))
        if np.isnan(debt):
            debt = np.nansum([_num(r.get("longTermDebt")), _num(r.get("shortTermDebt"))]) \
                if r.get("longTermDebt") or r.get("shortTermDebt") else _num(r.get("totalLiab"))
        eq, assets = _num(r.get("totalStockholderEquity")), _num(r.get("totalAssets"))
        rows.append((avail, debt / eq if eq else np.nan, debt / assets if assets else np.nan))
    if not rows:
        return None
    d = pd.DataFrame(rows, columns=["avail", "debtToEquity", "debtToAssets"]).drop_duplicates("avail")
    d = d.set_index("avail").sort_index().round(4)
    return _ts_block(d, ["debtToEquity", "debtToAssets"], "ts_km_", 3)


def earnings_surprises(fund, today=None):
    """Surprise de BPA (%) des 4 dernières publications, disponible le lendemain de la publication."""
    h = ((fund.get("Earnings") or {}).get("History") or {}).values()
    today = pd.Timestamp(today or pd.Timestamp.today().normalize())
    rows = []
    for r in h:
        act, est, rd = _num(r.get("epsActual")), _num(r.get("epsEstimate")), r.get("reportDate")
        if not rd or np.isnan(act) or pd.Timestamp(rd) > today:
            continue
        sp = _num(r.get("surprisePercent"))
        if np.isnan(sp) and not np.isnan(est) and est != 0:
            sp = 100 * (act - est) / abs(est)
        rows.append((pd.Timestamp(rd) + pd.Timedelta(days=1), sp))
    if not rows:
        return None
    d = pd.DataFrame(rows, columns=["avail", "surprise"]).drop_duplicates("avail").set_index("avail").sort_index()
    d["surprise"] = d["surprise"].clip(-200, 200).round(2)
    return _ts_block(d, ["surprise"], "ts_es_", 4)


ECON = {"durableGoods": ("Durable Goods Orders", "mom"), "retailSales": ("Retail Sales", "mom"),
        "federalFunds": ("Fed Interest Rate Decision", None)}


def econ_blocks():
    """Trois indicateurs US de ref/MyQTM, datés de leur publication (règle de clôture des news)."""
    ev = pd.concat([pd.read_parquet(p) for p in sorted((C.EODHD / "econ").glob("*.parquet"))], ignore_index=True)
    ev = ev[ev["country"] == "US"]
    blocks = {}
    for name, (typ, comp) in ECON.items():
        x = ev[(ev["type"] == typ) & (ev["comparison"].isna() | (ev["comparison"] == "None") if comp is None
                                      else ev["comparison"] == comp)].copy()
        x["actual"] = pd.to_numeric(x["actual"], errors="coerce")
        x = x.dropna(subset=["actual"])
        x["avail"] = (pd.to_datetime(x["date"]) + pd.Timedelta(hours=24 - C.NEWS_CUTOFF_HOUR_UTC)).dt.normalize()
        x = x.drop_duplicates("avail", keep="last").set_index("avail").sort_index()
        blocks[name] = _ts_block(x.rename(columns={"actual": name}), [name], "ts_ei_", 4)
    return blocks


def sentiment_block(sent):
    """sent : notes d'un titre (day, company, competitors, global) → séries des 10 derniers jours de news."""
    if sent is None or sent.empty:
        return None
    d = sent.set_index("day").sort_index()
    d = d.rename(columns={c: f"sentiment_{c}" for c in C.SENT_CATEGORIES})
    return _ts_block(d, [f"sentiment_{c}" for c in C.SENT_CATEGORIES], "ts_news_", C.NEWS_TS_SIZE)


def _div(a, b):
    return np.where((b != 0) & ~np.isnan(b), a / np.where(b == 0, np.nan, b), np.nan)


def add_ratios(f):
    def get(col):
        return f[col].to_numpy(float) if col in f else np.full(len(f), np.nan)
    pairs = {"momentum_1w": ("ts_w_close_0", "ts_w_close_1"), "rsi_over_price": ("rsi_14", "ts_d_close_0"),
             "sentiment_vs_rate": ("ts_news_sentiment_global_0", "ts_ei_federalFunds_0"),
             "ratio_ts_rsi_14_0_over_ts_rsi_14_1": ("ts_rsi_14_0", "ts_rsi_14_1"),
             "ratio_ts_d_volume_1_over_ts_d_volume_2": ("ts_d_volume_1", "ts_d_volume_2"),
             "ratio_ts_d_close_0_over_ts_d_close_2": ("ts_d_close_0", "ts_d_close_2")}
    for name, (a, b) in pairs.items():
        f[name] = _div(get(a), get(b))
    return f


def stock_features(symbol, px, fund, market, econ, sent, member):
    """Assemble toutes les features d'un titre. `px` : cours ajustés depuis PRICE_START."""
    f = price_features(px)
    dates = f.index
    parts = [f, asof_join(dates, market.rename_axis(None), "vix_days"),
             asof_join(dates, key_metrics(fund) if fund else None, "km_days")]
    parts += [asof_join(dates, b) for b in econ.values()]
    parts.append(asof_join(dates, earnings_surprises(fund, today=dates[-1]) if fund else None, "es_days"))
    sb = sentiment_block(sent)
    if sb is None:
        sb = pd.DataFrame(columns=[f"ts_news_sentiment_{c}_{k}" for c in C.SENT_CATEGORIES
                                   for k in range(C.NEWS_TS_SIZE)], dtype=float)
    parts.append(asof_join(dates, sb, "news_days"))
    out = pd.concat([p.reindex(dates) for p in parts], axis=1)
    for col in ("km_days", "es_days", "news_days", "vix_days"):
        if col not in out:
            out[col] = -1
        out[col] = out[col].fillna(-1).astype(int)
    for c in C.SENT_CATEGORIES:                       # colonnes toujours présentes (NaN tant que non notées)
        for k in range(C.NEWS_TS_SIZE):
            if f"ts_news_sentiment_{c}_{k}" not in out:
                out[f"ts_news_sentiment_{c}_{k}"] = np.nan
    out = add_ratios(out)
    out.insert(0, "c", px["c"].reindex(dates))
    out.insert(0, "o", px["o"].reindex(dates))
    out.insert(0, "member", member(dates))
    out.insert(0, "symbol", symbol)
    return out.rename_axis("date").reset_index()


def feature_columns(df):
    return [c for c in df.columns if c not in META]


# --------------------------------------------------------------------------- construction
def _one(args):
    symbol, market, econ, sent, u = args
    px = eod.load_eod(symbol)
    if px is None or len(px) < 260:
        return None
    f = stock_features(symbol, px, eod.load_fund(symbol), market, econ, sent,
                       lambda dates: U.member_mask(u, symbol, dates))
    return f[f["date"] >= ROWS_START]


def load_sentiments():
    return pd.read_parquet(SENTIMENTS) if SENTIMENTS.exists() else None


def main():
    u = U.load_universe()
    symbols = sorted(u["symbol"].unique())
    caps = pd.read_parquet(U.CAPS)
    closes = pd.DataFrame({s: p["c"] for s in symbols
                           if (p := eod.load_eod(s, start=f"{C.UNIVERSE_FIRST_YEAR - 1}-12-01")) is not None})
    index = build_index(u, caps[caps["symbol"].isin(symbols)], closes)
    index.to_parquet(INDEX)
    market, econ = market_features(index), econ_blocks()
    sent = load_sentiments()
    by_sym = dict(tuple(sent.groupby("symbol"))) if sent is not None else {}
    with ProcessPoolExecutor(C.WORKERS) as ex:
        parts = [p for p in ex.map(_one, [(s, market, econ, by_sym.get(s), u) for s in symbols]) if p is not None]
    df = pd.concat(parts, ignore_index=True)
    df.to_parquet(FEATURES, index=False)
    m = df[df["member"]]
    cov = m["ts_news_sentiment_company_0"].notna().mean()
    print(f"features : {len(df)} lignes, {df['symbol'].nunique()} titres, {len(feature_columns(df))} colonnes, "
          f"{df['date'].min():%Y-%m-%d} → {df['date'].max():%Y-%m-%d}")
    print(f"lignes membres : {len(m)} ; couverture du sentiment : {cov:.1%}")
    return df


if __name__ == "__main__":
    main()
