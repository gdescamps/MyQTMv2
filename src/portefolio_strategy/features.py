"""Features par (titre, jour) pour classer les titres de l'univers Nasdaq dynamique.

Tout est point-in-time : une ligne datée J n'utilise que ce qui est connu à la clôture de J.
- cours : rendements passés, volatilité, distance aux extrêmes, volume, bêta (rien de normalisé par titre :
  le modèle compare des titres entre eux le même jour) ;
- fondamentaux datés de leur dépôt (filing_date + 1 j) : endettement, rendement bénéficiaire, croissance, marges ;
- surprises de résultats datées de la publication + 1 j ; indicateurs macro datés de leur publication ;
- capitalisation = nombre d'actions connu (fin de trimestre + délai) × cours du jour ;
- sentiment PIT : directement P(Yes) aux questions « bonne nouvelle ? » / « mauvaise nouvelle ? » (pit_*) ;
- types de news Qwen (qt_*), sentiment Qwen (qs_*, NON point-in-time : comparaison seulement), sentiment
  fournisseur (vs_*) ;
- rangs du jour parmi les membres (xs_*) des variables principales.

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
SENTIMENTS = C.PIT_DIR / "sentiment.parquet"      # notes du LLM PIT (sentiment.py)
QWEN_SENTIMENTS = C.QWEN_DIR / "sentiment.parquet"
QWEN_TYPES = C.QWEN_DIR / "categories.parquet"
ROWS_START = "2019-06-01"
META = ["date", "symbol", "member", "o", "c"]
EPS = 1e-9
# Variables dont on ajoute le rang du jour parmi les membres (xs_<nom>, entre 0 et 1).
XS_COLS = ["ret_5", "ret_21", "ret_63", "ret_126", "mom_12_1", "vol_20", "vol_60", "dist_high_252", "beta_126",
           "dvol_ratio", "log_cap", "earnings_yield", "sales_yield", "fcf_yield", "sales_growth", "op_margin",
           "ts_es_surprise_0", "pit_pos_company_5", "pit_neg_company_5"]


# --------------------------------------------------------------------------- prix
def rsi14(c):
    d = c.diff()
    g = d.clip(lower=0).rolling(14).mean()
    l = (-d).clip(lower=0).rolling(14).mean()
    r = 100 - 100 / (1 + g / l)
    r = r.where(l != 0, 100.0)
    return r.where(g.notna(), 50.0).round(2)


def price_features(px, index_ret=None):
    """px : cours ajustés (o, c, v) indexés par date croissante ; index_ret : rendement quotidien de l'indice."""
    c, v = px["c"], px["v"].astype(float)
    r = c.pct_change()
    f = {"rsi_14": rsi14(c)}
    for n in (1, 5, 10, 21, 63, 126, 252):
        f[f"ret_{n}"] = c / c.shift(n) - 1
    f["mom_12_1"] = c.shift(21) / c.shift(252) - 1
    f["vol_20"] = r.rolling(20).std() * np.sqrt(252)
    f["vol_60"] = r.rolling(60).std() * np.sqrt(252)
    f["max_ret_21"] = r.rolling(21).max()
    f["dist_high_252"] = c / c.rolling(252, min_periods=120).max() - 1
    f["dist_low_252"] = c / c.rolling(252, min_periods=120).min() - 1
    f["gap_sma_50"] = c / c.rolling(50).mean() - 1
    f["gap_sma_200"] = c / c.rolling(200).mean() - 1
    dv = (c * v)
    f["log_dvol_20"] = np.log(dv.rolling(20).mean().where(lambda x: x > 0))
    f["dvol_ratio"] = dv.rolling(5).mean() / dv.rolling(60).mean()
    if index_ret is not None:
        m = index_ret.reindex(px.index)
        f["beta_126"] = r.rolling(126).cov(m) / m.rolling(126).var()
        f["rel_ret_21"] = f["ret_21"] - (np.exp(np.log1p(m).rolling(21).sum()) - 1)
    out = pd.DataFrame(f, index=px.index)
    out["month"] = px.index.month
    return out.iloc[200:].round(5)          # historique incomplet (moyenne 200 j)


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
    """État du marché (identique pour tous les titres du jour) : rendement, volatilité, repli de l'indice."""
    lvl, ret = index["level"], index["ret"]
    f = {"idx_ret_21": lvl / lvl.shift(21) - 1, "idx_ret_63": lvl / lvl.shift(63) - 1,
         "idx_vol_20": ret.rolling(20).std() * np.sqrt(252), "idx_dd_252": lvl / lvl.rolling(252, min_periods=60).max() - 1}
    return pd.DataFrame(f, index=index.index).iloc[63:].round(5)


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


def fundamentals_ttm(fund):
    """Sommes sur 4 trimestres (chiffre d'affaires, résultat net, flux de trésorerie disponible) et ratios,
    disponibles le lendemain du dépôt du dernier trimestre (filing_date ; à défaut fin de trimestre + 46 j)."""
    fin = fund.get("Financials") or {}
    inc = (fin.get("Income_Statement") or {}).get("quarterly") or {}
    cf = (fin.get("Cash_Flow") or {}).get("quarterly") or {}
    rows = []
    for period, r in inc.items():
        fd = r.get("filing_date")
        avail = pd.Timestamp(fd) + pd.Timedelta(days=1) if fd else pd.Timestamp(period) + pd.Timedelta(days=46)
        rows.append((pd.Timestamp(period), avail, _num(r.get("totalRevenue")), _num(r.get("netIncome")),
                     _num(r.get("operatingIncome")), _num(r.get("grossProfit")),
                     _num((cf.get(period) or {}).get("freeCashFlow"))))
    if len(rows) < 4:
        return None
    d = pd.DataFrame(rows, columns=["period", "avail", "rev", "ni", "op", "gp", "fcf"]).sort_values("period")
    t = d[["rev", "ni", "op", "gp", "fcf"]].rolling(4, min_periods=4).sum()
    out = pd.DataFrame({"avail": d["avail"].cummax(), "f_rev_ttm": t["rev"], "f_ni_ttm": t["ni"], "f_fcf_ttm": t["fcf"],
                        "sales_growth": (t["rev"] / t["rev"].shift(4).where(lambda x: x > 0) - 1).clip(-1, 5),
                        "op_margin": (t["op"] / t["rev"].where(lambda x: x > 0)).clip(-2, 1),
                        "gross_margin": (t["gp"] / t["rev"].where(lambda x: x > 0)).clip(-1, 1)})
    return out.dropna(subset=["f_rev_ttm"]).drop_duplicates("avail", keep="last").set_index("avail").sort_index()


def prob_block(sent, prefix):
    """sent : notes d'un titre (une ligne par jour de news : p_pos_<cat>, p_neg_<cat>) → pour chaque catégorie,
    les probabilités P(Yes) brutes du dernier jour de news et leur moyenne sur les 5 et 20 derniers jours de
    news (rubriques vides ignorées), plus le nombre de jours calendaires couverts par ces 5 jours de news."""
    if sent is None or sent.empty:
        return None
    d = sent.set_index("day").sort_index()
    out = pd.DataFrame(index=d.index)
    for cat in C.SENT_CATEGORIES:
        for pol in ("pos", "neg"):
            x = d[f"p_{pol}_{cat}"].astype(float)
            out[f"{prefix}{pol}_{cat}"] = x
            out[f"{prefix}{pol}_{cat}_5"] = x.rolling(5, min_periods=1).mean()
            out[f"{prefix}{pol}_{cat}_20"] = x.rolling(20, min_periods=1).mean()
    days = pd.Series(d.index, index=d.index)
    out[f"{prefix}span_5"] = (days - days.shift(4)).dt.days
    return out.round(4)


def types_block(types):
    """types : P(type de news) d'un titre par jour de news (qwen.py) → dernier jour de news et somme sur les
    5 derniers jours de news (≈ nombre de news récentes de ce type)."""
    if types is None or types.empty:
        return None
    d = types.set_index("day").sort_index()
    out = pd.DataFrame(index=d.index)
    for t in C.NEWS_TYPES:
        x = d[f"p_{t}"].astype(float)
        out[f"qt_{t}"] = x
        out[f"qt_{t}_5"] = x.rolling(5, min_periods=1).sum()
    return out.round(4)


def vendor_block(vs):
    """vs : sentiment fournisseur d'un titre (vendor_sentiment.py), une ligne par jour de news → features :
    séries des derniers jours de news (polarité : 10 ; n, pos, neg : 3), extrêmes du dernier jour, et sur une
    fenêtre glissante de 5 jours de news : polarité pondérée, nombre d'articles, et par catégorie de news le
    nombre d'articles et leur polarité moyenne. Tout est calculé sur des jours de news passés."""
    if vs is None or vs.empty:
        return None
    from .vendor_sentiment import CATEGORIES
    d = vs.set_index("day").sort_index()
    out = pd.concat([_ts_block(d, ["polarity"], "ts_vs_", C.NEWS_TS_SIZE), _ts_block(d, ["n", "pos", "neg"], "ts_vs_", 3)],
                    axis=1)
    out["vs_polarity_min"], out["vs_polarity_max"] = d["polarity_min"], d["polarity_max"]
    n5 = d["n"].rolling(5, min_periods=1).sum()
    out["vs_n5"] = n5
    out["vs_pol5"] = (d["polarity"] * d["n"]).rolling(5, min_periods=1).sum() / n5
    for c in CATEGORIES:
        cn = d[f"cat_{c}_n"].rolling(5, min_periods=1).sum()
        out[f"vs_cat_{c}_n5"] = cn
        out[f"vs_cat_{c}_pol5"] = ((d[f"cat_{c}_pol"] * d[f"cat_{c}_n"]).fillna(0).rolling(5, min_periods=1).sum()
                                   / cn.where(cn > 0))
    return out.round(4)


def stock_features(symbol, px, fund, market, econ, sent, member, vendor=None, index_ret=None, qsent=None,
                   types=None, sector=-1):
    """Assemble toutes les features d'un titre. `px` : cours ajustés depuis PRICE_START."""
    f = price_features(px, index_ret)
    dates = f.index
    parts = [f, asof_join(dates, market.rename_axis(None)),
             asof_join(dates, key_metrics(fund) if fund else None, "km_days")]
    parts += [asof_join(dates, b) for b in econ.values()]
    parts.append(asof_join(dates, earnings_surprises(fund, today=dates[-1]) if fund else None, "es_days"))
    parts.append(asof_join(dates, fundamentals_ttm(fund) if fund else None))
    parts.append(asof_join(dates, prob_block(sent, "pit_"), "news_days"))
    parts.append(asof_join(dates, prob_block(qsent, "qs_")))
    parts.append(asof_join(dates, types_block(types)))
    vb = vendor_block(vendor)
    if vb is not None:
        parts.append(asof_join(dates, vb, "vs_days"))
    out = pd.concat([p.reindex(dates) for p in parts], axis=1)
    for col in ("km_days", "es_days", "news_days", "vs_days"):
        if col in out or col != "vs_days":
            out[col] = (out[col] if col in out else pd.Series(-1, index=dates)).fillna(-1).astype(int)
    # capitalisation et ratios de valorisation (nombre d'actions connu × cours du jour)
    sh = eod.shares_series(fund) if fund else None
    cap = (sh.reindex(sh.index.union(dates)).ffill().reindex(dates) * px["split_close"].reindex(dates)
           if sh is not None else pd.Series(np.nan, index=dates))
    cap = cap.where(cap > 0)
    out["log_cap"] = np.log(cap).round(4)
    for name, col in (("earnings_yield", "f_ni_ttm"), ("sales_yield", "f_rev_ttm"), ("fcf_yield", "f_fcf_ttm")):
        out[name] = (out[col] / cap).clip(-1, 5).round(5) if col in out else np.nan
    out = out.drop(columns=[c for c in ("f_rev_ttm", "f_ni_ttm", "f_fcf_ttm") if c in out])
    for c in ("sales_growth", "op_margin", "gross_margin"):
        if c not in out:
            out[c] = np.nan
    for p, src in (("pit_", sent), ("qs_", qsent)):   # colonnes toujours présentes (NaN tant que non notées)
        if src is None or src.empty:
            for cat in C.SENT_CATEGORIES:
                for pol in ("pos", "neg"):
                    for suf in ("", "_5", "_20"):
                        out[f"{p}{pol}_{cat}{suf}"] = np.nan
            out[f"{p}span_5"] = np.nan
    if types is None or types.empty:
        for t in C.NEWS_TYPES:
            out[f"qt_{t}"], out[f"qt_{t}_5"] = np.nan, np.nan
    out["sector_id"] = sector
    out.insert(0, "c", px["c"].reindex(dates))
    out.insert(0, "o", px["o"].reindex(dates))
    out.insert(0, "member", member(dates))
    out.insert(0, "symbol", symbol)
    return out.copy().rename_axis("date").reset_index()


def add_cross_section(df):
    """Rang du jour parmi les membres (0 = plus bas, 1 = plus haut ; NaN hors univers) des variables XS_COLS."""
    m = df["member"].to_numpy()
    g = df.loc[m].groupby("date")
    new = {}
    for c in XS_COLS:
        if c in df:
            new[f"xs_{c}"] = g[c].rank(pct=True).reindex(df.index).round(4)
    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)


# Familles de features de sentiment / news, et variantes d'évaluation (quelles familles entrent dans le modèle).
FAMILIES = {"pit": lambda c: c.startswith(("pit_", "xs_pit_")) or c == "news_days",
            "types": lambda c: c.startswith("qt_"),
            "qsent": lambda c: c.startswith("qs_"),
            "vendor": lambda c: c.startswith(("vs_", "ts_vs_"))}
VARIANTS = {"base": ((), "données structurées seules (cours, fondamentaux, macro)"),
            "pit": (("pit",), "structurées + sentiment PIT (probabilités brutes)"),
            "pit_types": (("pit", "types"), "structurées + sentiment PIT + types de news Qwen"),
            "all": (("pit", "types", "vendor"), "structurées + PIT + types Qwen + sentiment fournisseur"),
            "vendor": (("vendor",), "structurées + sentiment fournisseur EODHD"),
            "qsent": (("qsent", "types"), "structurées + sentiment et types Qwen (NON point-in-time)")}


def family_of(col):
    return next((k for k, f in FAMILIES.items() if f(col)), None)


def feature_columns(df, variant="all"):
    keep = VARIANTS[variant][0]
    return [c for c in df.columns if c not in META and family_of(c) in (None, *keep)]


# --------------------------------------------------------------------------- construction
def _one(args):
    symbol, market, econ, u, index_ret, sector, blocks = args
    px = eod.load_eod(symbol)
    if px is None or len(px) < 260:
        return None
    f = stock_features(symbol, px, eod.load_fund(symbol), market, econ, blocks.get("pit"),
                       lambda dates: U.member_mask(u, symbol, dates), blocks.get("vendor"), index_ret,
                       blocks.get("qsent"), blocks.get("types"), sector)
    return f[f["date"] >= ROWS_START]


def _by_symbol(path):
    if not path.exists():
        return {}
    return dict(tuple(pd.read_parquet(path).groupby("symbol")))


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
    from .vendor_sentiment import VENDOR
    src = {"pit": _by_symbol(SENTIMENTS), "qsent": _by_symbol(QWEN_SENTIMENTS), "types": _by_symbol(QWEN_TYPES),
           "vendor": _by_symbol(VENDOR)}
    sectors = caps.drop_duplicates("symbol", keep="last").set_index("symbol")["sector"]
    sector_id = {s: i for i, s in enumerate(sorted(sectors.dropna().unique()))}
    jobs = [(s, market, econ, u, index["ret"], sector_id.get(sectors.get(s), -1),
             {k: v.get(s) for k, v in src.items()}) for s in symbols]
    with ProcessPoolExecutor(C.WORKERS) as ex:
        parts = [p for p in ex.map(_one, jobs) if p is not None]
    df = pd.concat(parts, ignore_index=True)
    if "vs_days" in df:
        df["vs_days"] = df["vs_days"].fillna(-1).astype(int)
    df = add_cross_section(df)
    df.to_parquet(FEATURES, index=False)
    m = df[df["member"] & (df["date"] >= C.START_DATE)]
    print(f"features : {len(df)} lignes, {df['symbol'].nunique()} titres, {len(feature_columns(df))} colonnes, "
          f"{df['date'].min():%Y-%m-%d} → {df['date'].max():%Y-%m-%d}")
    cov = {k: f"{m[c].notna().mean():.0%}" for k, c in (("PIT", "pit_pos_company"), ("types Qwen", "qt_earnings"),
                                                        ("sentiment Qwen", "qs_pos_company"),
                                                        ("fournisseur", "ts_vs_polarity_0")) if c in m}
    last = {k: f"{pd.read_parquet(p, columns=['day'])['day'].max():%Y-%m-%d}" for k, p in
            (("PIT", SENTIMENTS), ("Qwen", QWEN_SENTIMENTS)) if p.exists()}
    print(f"lignes membres depuis {C.START_DATE} : {len(m)} ; couverture {cov} ; dernier jour de news noté {last}")
    return df


if __name__ == "__main__":
    main()
