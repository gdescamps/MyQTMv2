"""Secteurs du S&P 500 reconstitué en données horaires (EODHD intraday 1 h, depuis octobre 2020) et oracle « top k ».

Points horaires gardés (heure de New York) : 9 h 30, 10 h 30 … 15 h 30, puis la clôture de 16 h (8 points par
séance) ; le premier rendement de la séance inclut la nuit. Les cours intraday EODHD ne sont pas ajustés des
divisions d'actions : ils sont divisés par le facteur de division à leur date (eod.split_factor). Rendement horaire
d'un titre > ±50 % = erreur de cotation, écarté. Rendement d'un secteur = moyenne de ses membres du jour pondérée
par la capitalisation de la fin du mois précédent (comme sector_oracle.py, en quotidien).

Oracle : à chaque point, les k secteurs qui feront le mieux jusqu'au point suivant, à poids égaux, frais 0,1 % par
sens sur les secteurs renouvelés ; comparé, sur la même période, aux versions quotidienne et hebdomadaire calculées
à partir des mêmes données.

Sorties : data/portefolio_strategy/sp500/sector_hourly.parquet (rendements), sector_hourly_weights.parquet,
outputs/portefolio_strategy/sp500/sector_hourly.md.

    python -m src.portefolio_strategy.sector_hourly
"""
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from . import config as C
from . import eod
from . import sector_oracle as SO
from . import sp500 as SP
from . import universe as U

OUT = C.OUT / "sp500"
RET = SP.DIR / "sector_hourly.parquet"
WEIGHTS = SP.DIR / "sector_hourly_weights.parquet"
TIMES = {f"{h:02d}:30" for h in range(9, 16)} | {"16:00"}
KS = range(1, 12)


def _one(args):
    symbol, u = args
    p = C.EODHD / "intraday_1h" / f"{symbol}.parquet"
    if not p.exists():
        return None
    x = pd.read_parquet(p, columns=["timestamp", "close"])
    t = pd.to_datetime(x["timestamp"], unit="s", utc=True).dt.tz_convert("America/New_York")
    keep = t.dt.strftime("%H:%M").isin(TIMES).to_numpy()
    x, t = x[keep], t[keep].dt.tz_localize(None)
    if x.empty:
        return None
    days = t.dt.normalize()
    c = x["close"].to_numpy(float) / eod.split_factor(symbol, pd.DatetimeIndex(days))
    s = pd.Series(c, index=pd.DatetimeIndex(t)).groupby(level=0).last()
    r = s.pct_change(fill_method=None)
    r[r.abs() > 0.5] = np.nan
    member = U.member_mask(u, symbol, pd.DatetimeIndex(r.index.normalize()))
    return r[member].astype("float32").rename(symbol)


def build():
    u = pd.read_parquet(SP.UNIVERSE)
    syms = sorted(u["symbol"].unique())
    with ProcessPoolExecutor(C.WORKERS) as ex:
        parts = [p for p in ex.map(_one, [(s, u[u["symbol"] == s]) for s in syms], chunksize=8) if p is not None]
    R = pd.concat(parts, axis=1).sort_index()
    n = R.notna().sum(axis=1)
    R = R[n >= 0.8 * n.median()]                      # points où presque aucun titre n'a de barre : écartés
    caps = pd.read_parquet(SP.CAPS)
    sector = caps.drop_duplicates("symbol", keep="last").set_index("symbol")["sector"].replace("", np.nan)
    cp = caps.pivot_table(index="month", columns="symbol", values="cap")
    cp.index = pd.PeriodIndex(cp.index, freq="M")
    w = cp.reindex(R.index.to_period("M") - 1).set_axis(R.index).reindex(columns=R.columns)
    w = w.where(R.notna())
    out, wout = {}, {}
    for g, cols in sector.reindex(R.columns).dropna().groupby(sector.reindex(R.columns).dropna()).groups.items():
        ww, rr = w[list(cols)], R[list(cols)]
        out[g] = (ww * rr).sum(axis=1, min_count=1) / ww.sum(axis=1).replace(0, np.nan)
        wout[g] = ww.sum(axis=1)
    S, W = pd.DataFrame(out), pd.DataFrame(wout)
    W = W.div(W.sum(axis=1), axis=0)
    S.to_parquet(RET)
    W.to_parquet(WEIGHTS)
    return S, W


def aggregate(S, W, periods):
    """Rendements composés par période (étiquettes `periods` alignées sur S) ; poids du premier point de la période."""
    g = (1 + S.fillna(0)).groupby(periods)
    return g.prod() - 1, W.groupby(periods).first()


def oracle_table(R, W):
    R = R.dropna(how="all")
    W = W.reindex(R.index)
    step = max((R.index[-1] - R.index[0]).days / len(R), 1 / 24)
    dates = list(R.index) + [R.index[-1] + pd.Timedelta(days=step)]
    rows = {"indice (tous les secteurs, capi)": SO.stats(SO.run(R, None, weights=W), dates, 0)}
    for k in KS:
        for fee, tag in ((C.FEE, ""), (0.0, " brut")):
            s = SO.stats(SO.run(R, lambda i: R.iloc[i].nlargest(k).index, fee=fee), dates, 0)
            if tag:
                rows[f"top {k}"]["CAGR brut %"] = s["CAGR %"]
            else:
                rows[f"top {k}"] = s
    t = pd.DataFrame(rows).T[["CAGR %", "CAGR brut %", "maxDD %", "Sharpe"]]
    top = R.rank(axis=1, ascending=False) <= 5
    ch = (top.astype(int).diff().abs().sum(axis=1) / 2).iloc[1:].mean()
    rk = R.rank(axis=1)
    pers = np.nanmean([rk.iloc[i].corr(rk.iloc[i + 1]) for i in range(0, len(rk) - 1, max(1, len(rk) // 3000))])
    return t, ch, pers


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    S, W = (pd.read_parquet(RET), pd.read_parquet(WEIGHTS)) if RET.exists() else build()
    S = S[[c for c in S.columns if S[c].notna().mean() > 0.9]]
    W = W[S.columns]
    days = S.index.normalize()
    dlist = pd.DatetimeIndex(np.sort(days.unique()))
    week = pd.Series(np.arange(len(dlist)) // 5, index=dlist).reindex(days).to_numpy()
    views = {"horaire": (S, W), "quotidien": aggregate(S, W, days), "hebdomadaire": aggregate(S, W, week)}
    views["hebdomadaire"] = tuple(v.set_axis(dlist[::5][:len(v)]) for v in views["hebdomadaire"])
    md = ["# Oracle « secteurs » du S&P 500 reconstitué : horaire, quotidien, hebdomadaire", "",
          f"Période {S.index[0]:%Y-%m-%d} → {S.index[-1]:%Y-%m-%d} ({len(dlist)} séances, {len(S)} points horaires, "
          f"{S.shape[1]} secteurs), données intraday EODHD agrégées à partir des actions. Décision et exécution au même "
          "cours (borne supérieure) ; frais 0,1 % par sens sur les secteurs renouvelés.", ""]
    for name, (R, Wv) in views.items():
        t, ch, pers = oracle_table(R, Wv)
        md += [f"## {name.capitalize()}", "",
               f"Changements moyens du top 5 d'une période à la suivante : {ch:.2f} sur 5 (hasard 2,73) ; persistance du "
               f"classement : {pers:+.3f}.", "", t.round(2).to_markdown(), ""]
    (OUT / "sector_hourly.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
