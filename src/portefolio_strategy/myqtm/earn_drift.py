"""Modèle dédié aux publications de résultats : dérive post-annonce (PEAD) sur le S&P 500 reconstitué.

Un exemple = une publication de résultats d'un membre de l'univers (calendrier EODHD, ~2 000 par an depuis 2015).
Instant de décision T = clôture de la séance de réaction : publication avant l'ouverture → le jour même ; après la
clôture → la séance suivante ; moment inconnu → la séance qui suit la date de publication (fenêtre de 2 séances).
`base` = dernière clôture avant l'annonce. Entrée à l'ouverture de T+1 : on achète APRÈS la publication, sans
porter le risque du saut d'annonce.

Features connues à la clôture de T (aucun LLM) : surprise de BPA (et les 3 précédentes, série de dépassements),
réaction du cours relative au marché (base → T), écart d'ouverture, séance, position de la clôture dans la
fourchette du jour, volume relatif, réaction rapportée à la volatilité 20 j, momentum relatif avant l'annonce
(5 / 21 / 63 / 126 séances), distance au plus haut 252 j, RSI 14, marché (21 j, volatilité 20 j), taille,
moment de la publication, délai depuis la fin du trimestre, ton de la saison (réaction moyenne des publications
des 30 jours précédents, marché et secteur), nombre de publications du jour.

Cible : rendement relatif à l'univers équipondéré, de l'ouverture T+1 à l'ouverture T+1+H (H = 20 séances ;
5 et 60 aussi calculés). Marché = moyenne des membres du jour (ouverture à ouverture pour les cibles).

Walk-forward : coupes trimestrielles dès 2024 ; apprentissage sur les publications depuis 2015 dont la cible est
connue avant la coupe ; XGBoost en régression (cible bornée à ±30 %), moyenne de 3 graines. Seuils causaux : quantiles
80 / 90 % (et 20 %) des prédictions du modèle sur les publications d'apprentissage des 365 derniers jours.

Sorties (data/portefolio_strategy/myqtm_sp) : earn_events.parquet (features, cibles, prédictions hors échantillon),
earn_opens.parquet (ouvertures, pour simuler), earn_bench.parquet (rendements de l'univers équipondéré).

    python -m src.portefolio_strategy.myqtm.earn_drift
"""
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import xgboost as xgb

from . import features as F
from . import train as T
from .. import config as PC
from .. import eod
from .. import sp500 as SP
from .. import universe as U

DATA = PC.DATA / "myqtm_sp"
EVENTS, OPENS, BENCH = DATA / "earn_events.parquet", DATA / "earn_opens.parquet", DATA / "earn_bench.parquet"
PRICE_START = "2014-01-01"
EVENT_START = "2015-01-01"
FIRST = "2024-01-01"
HORIZONS = (5, 20, 60)
H = 20                         # horizon de la cible apprise
CLIP = 0.30
SEEDS = (42, 7, 123)
FEATS = ["surprise", "surprise_1", "surprise_2", "surprise_3", "beats4", "react", "gap", "intraday", "clv",
         "vol_ratio", "react_z", "pre5", "pre21", "pre63", "pre126", "vol20", "hi252", "rsi14", "mkt21", "mktvol20",
         "logcap", "amc", "lag_q", "days_prev", "tone_mkt", "tone_sect", "beat_mkt", "n_day"]


def _load(sym):
    d = eod.load_eod(sym, start=PRICE_START)
    return None if d is None else d[["o", "h", "l", "c", "v"]]


def matrices():
    u = pd.read_parquet(SP.UNIVERSE)
    syms = sorted(u["symbol"].unique())
    with ProcessPoolExecutor(PC.WORKERS) as ex:
        parts = dict(zip(syms, ex.map(_load, syms, chunksize=16)))
    parts = {s: d for s, d in parts.items() if d is not None}
    m = pd.concat(parts, axis=1)
    mats = {k: m.xs(k, axis=1, level=1).sort_index() for k in ("o", "h", "l", "c", "v")}
    dates = mats["c"].index
    member = pd.DataFrame({s: U.member_mask(u[u["symbol"] == s], s, dates) for s in mats["c"].columns}, index=dates)
    return mats, member


def benchmark(mats, member):
    """Univers équipondéré : ro[t] = ouverture t → t+1, rc[t] = clôture t−1 → t ; niveaux cumulés LO, LC."""
    o, c = mats["o"], mats["c"]
    ro = (o.shift(-1) / o - 1).clip(-0.5, 1.0).where(member).mean(axis=1)
    rc = (c / c.shift(1) - 1).clip(-0.5, 1.0).where(member).mean(axis=1)
    lo = (1 + ro.fillna(0)).cumprod().shift(1, fill_value=1.0)
    lc = (1 + rc.fillna(0)).cumprod()
    return pd.DataFrame({"ro": ro, "rc": rc, "lo": lo, "lc": lc})


def calendar(symbols):
    files = sorted((PC.EODHD / "earnings").glob("*.parquet"))
    e = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    e = e[e["code"].str.endswith(".US")].assign(symbol=lambda x: x["code"].str[:-3])
    e = e[e["symbol"].isin(symbols)].copy()
    e["report_date"] = pd.to_datetime(e["report_date"])
    e["period"] = pd.to_datetime(e["date"], errors="coerce")
    e["surprise"] = pd.to_numeric(e["percent"], errors="coerce").clip(-100, 100)
    e["actual"] = pd.to_numeric(e["actual"], errors="coerce")
    e = e.dropna(subset=["actual"]).drop_duplicates(["symbol", "report_date"]).sort_values(["symbol", "report_date"])
    g = e.groupby("symbol")
    for k in (1, 2, 3):
        e[f"surprise_{k}"] = g["surprise"].shift(k)
    beat = (e["surprise"] > 0).astype(float)
    e["beats4"] = sum(beat.groupby(e["symbol"]).shift(k).fillna(0) for k in (1, 2, 3, 4))
    e["days_prev"] = (e["report_date"] - g["report_date"].shift(1)).dt.days.clip(upper=200)
    e["lag_q"] = (e["report_date"] - e["period"]).dt.days.clip(0, 120)
    e["amc"] = e["before_after_market"].map({"AfterMarket": 1.0, "BeforeMarket": 0.0})
    return e.reset_index(drop=True)


def build_events(mats, member, bench):
    o, h, l, c, v = (mats[k] for k in ("o", "h", "l", "c", "v"))
    dates, n = c.index, len(c)
    e = calendar(set(c.columns))
    e = e[e["report_date"] >= dates[130]].reset_index(drop=True)
    rd = e["report_date"].to_numpy()
    pos = dates.searchsorted(rd, side="left")
    is_sess = (pos < n) & (dates[np.minimum(pos, n - 1)] == rd)
    amc = e["amc"].to_numpy()
    base = np.where(amc == 0, pos - 1, np.where(amc == 1, np.where(is_sess, pos, pos - 1), pos - 1))
    t = np.where(amc == 0, pos, np.where(amc == 1, base + 1, np.where(is_sess, pos + 1, pos)))
    ok = (base - 126 >= 0) & (t < n)
    e, base, t = e[ok].reset_index(drop=True), base[ok], t[ok]
    j = c.columns.get_indexer(e["symbol"])
    cn, on, hn, ln, vn = (x.to_numpy(float) for x in (c, o, h, l, v))
    ofill = o.ffill().to_numpy(float)                       # titre retiré de la cote : valorisé au dernier cours
    lc, lo = bench["lc"].to_numpy(), bench["lo"].to_numpy()
    ret = c.pct_change(fill_method=None)
    vol20 = ret.rolling(20).std().to_numpy()
    medv = v.rolling(20).median().to_numpy()
    hi = c.rolling(252, min_periods=120).max().to_numpy()
    rsi = c.apply(F.rsi14).to_numpy(float)
    e["tdate"], e["member"] = dates[t], member.to_numpy()[t, j]
    e["react"] = cn[t, j] / cn[base, j] - 1 - (lc[t] / lc[base] - 1)
    e["gap"] = on[base + 1, j] / cn[base, j] - 1
    e["intraday"] = cn[t, j] / on[t, j] - 1
    e["clv"] = (cn[t, j] - ln[t, j]) / np.where(hn[t, j] > ln[t, j], hn[t, j] - ln[t, j], np.nan)
    e["vol_ratio"] = vn[t, j] / np.where(medv[base, j] > 0, medv[base, j], np.nan)
    e["vol20"] = vol20[base, j]
    e["react_z"] = e["react"] / e["vol20"].replace(0, np.nan)
    for k in (5, 21, 63, 126):
        e[f"pre{k}"] = cn[base, j] / cn[base - k, j] - 1 - (lc[base] / lc[base - k] - 1)
    e["hi252"] = cn[t, j] / hi[t, j] - 1
    e["rsi14"] = rsi[t, j]
    e["mkt21"] = lc[t] / lc[t - 21] - 1
    e["mktvol20"] = bench["rc"].rolling(20).std().to_numpy()[t]
    for hz in HORIZONS:
        x = np.minimum(t + 1 + hz, n - 1)
        tg = ofill[x, j] / on[np.minimum(t + 1, n - 1), j] - 1 - (lo[x] / lo[np.minimum(t + 1, n - 1)] - 1)
        e[f"y{hz}"] = np.where(t + 1 + hz < n, tg, np.nan)
        e[f"end{hz}"] = dates[x]
    e["entry"] = dates[np.minimum(t + 1, n - 1)]
    e.loc[t + 1 >= n, "entry"] = pd.NaT
    caps = pd.read_parquet(SP.CAPS, columns=["month", "symbol", "cap", "sector"])
    caps["month"] = caps["month"].astype(str)
    e["month"] = (e["tdate"].dt.to_period("M") - 1).astype(str)
    e = e.merge(caps, on=["month", "symbol"], how="left")
    e["logcap"] = np.log(e["cap"].where(e["cap"] > 0))
    e["sector"] = e["sector"].fillna("?")
    e = e[e["member"] & e["react"].notna()].sort_values(["tdate", "symbol"]).reset_index(drop=True)
    e["n_day"] = e.groupby("tdate")["symbol"].transform("size")
    s = e.set_index("tdate")
    e["tone_mkt"] = s["react"].rolling("30D", closed="left").mean().to_numpy()
    e["beat_mkt"] = (s["surprise"] > 0).astype(float).rolling("30D", closed="left").mean().to_numpy()
    e["tone_sect"] = np.nan
    for _, idx in e.groupby("sector").indices.items():
        x = e.iloc[idx].set_index("tdate")["react"].rolling("30D", closed="left").mean()
        e.loc[e.index[idx], "tone_sect"] = x.to_numpy()
    return e[e["tdate"] >= EVENT_START].reset_index(drop=True)


def walk_forward(e):
    prm = dict(PC.XGB_PARAMS)
    e["pred"] = np.nan
    for q in (20, 80, 90):
        e[f"thr{q}"] = np.nan
    last = e["tdate"].max()
    cuts = T.cutoffs(FIRST, last)
    imp = {}
    for i, cut in enumerate(cuts):
        nxt = cuts[i + 1] if i + 1 < len(cuts) else last + pd.Timedelta(days=1)
        tr = (e[f"end{H}"] < cut) & e[f"y{H}"].notna() & (e[f"y{H}"].abs() < 1.5)
        te = (e["tdate"] >= cut) & (e["tdate"] < nxt)
        if not te.any():
            continue
        dtr = xgb.DMatrix(e.loc[tr, FEATS].to_numpy(np.float32), label=e.loc[tr, f"y{H}"].clip(-CLIP, CLIP))
        recent = tr & (e["tdate"] >= cut - pd.Timedelta(days=365))
        p_te, p_rec = 0.0, 0.0
        for s in SEEDS:
            m = xgb.train(dict(prm, seed=s), dtr, num_boost_round=PC.XGB_ROUNDS)
            p_te = p_te + m.predict(xgb.DMatrix(e.loc[te, FEATS].to_numpy(np.float32))) / len(SEEDS)
            p_rec = p_rec + m.predict(xgb.DMatrix(e.loc[recent, FEATS].to_numpy(np.float32))) / len(SEEDS)
            g = m.get_score(importance_type="total_gain")
            imp[(f"{cut:%Y-%m-%d}", s)] = {FEATS[int(k[1:])]: v for k, v in g.items()}
        e.loc[te, "pred"] = p_te
        for q in (20, 80, 90):
            e.loc[te, f"thr{q}"] = np.quantile(p_rec, q / 100)
        print(f"  coupe {cut:%Y-%m-%d} : {int(tr.sum())} publications d'apprentissage, {int(te.sum())} à prédire",
              flush=True)
    im = pd.DataFrame(imp).fillna(0)
    im = (im / im.sum()).mean(axis=1).sort_values(ascending=False)
    return e, im


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    mats, member = matrices()
    bench = benchmark(mats, member)
    e = build_events(mats, member, bench)
    print(f"publications : {len(e)} de {e['tdate'].min():%Y-%m-%d} à {e['tdate'].max():%Y-%m-%d}, "
          f"{e['symbol'].nunique()} titres ; par an {e.groupby(e['tdate'].dt.year).size().to_dict()}", flush=True)
    e, im = walk_forward(e)
    e.drop(columns=["code", "date", "currency", "difference", "percent", "estimate", "month"], errors="ignore") \
        .to_parquet(EVENTS, index=False)
    mats["o"].ffill().loc[FIRST:].astype("float32").to_parquet(OPENS)
    bench.to_parquet(BENCH)
    im.rename("part_du_gain").to_csv(PC.OUT / "myqtm_sp" / "earn_drift_importance.csv")
    print("importance des features (part du gain) :")
    print(im.head(12).round(3).to_string())


if __name__ == "__main__":
    (PC.OUT / "myqtm_sp").mkdir(parents=True, exist_ok=True)
    main()
