"""Liste Nasdaq dynamique, point-in-time, reconstruite depuis data/eodhd.

data/eodhd ne contient pas l'historique officiel des constituants du Nasdaq-100. On le reconstruit avec
la méthode de l'indice :
- candidats = actions ordinaires cotées au NASDAQ (actives ET radiées), hors financières ;
- capitalisation = cours × nombre d'actions connu à la date (fin de trimestre + 45 j) ;
- reconstitution annuelle : classement de fin novembre, effet le lundi suivant le 3e vendredi de décembre,
  avec la règle tampon du NDX (entrée d'office si rang <= 75, maintien d'un membre si rang <= 100) ;
- entre deux reconstitutions, un membre radié est remplacé à la fin du mois par la plus forte
  capitalisation non membre.

Limites (données) : la place de cotation et le secteur sont ceux d'aujourd'hui (ou de la radiation),
pas ceux de l'époque.

    python -m src.portefolio_strategy.universe
"""
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from . import config as C
from . import eod

CAPS = C.DATA / "mcap_monthly.parquet"
UNIVERSE = C.DATA / "universe.parquet"


def _caps_one(code):
    fund = eod.load_fund(code)
    if not fund:
        return None
    g = fund.get("General") or {}
    sector = g.get("GicSector") or g.get("Sector") or ""
    if sector in C.EXCLUDED_SECTORS:
        return None
    sh = eod.shares_series(fund)
    px = eod.load_eod(code, start=f"{C.UNIVERSE_FIRST_YEAR - 1}-06-01")
    if sh is None or px is None:
        return None
    dv = (px["split_close"] * px["v"]).rolling(60, min_periods=20).median()
    shares = sh.reindex(px.index.union(sh.index)).ffill().reindex(px.index)
    d = pd.DataFrame({"cap": px["split_close"] * shares, "dv": dv, "last": px.index}, index=px.index)
    m = d.groupby(px.index.to_period("M")).last().dropna(subset=["cap"])
    m = m[(m.index.to_timestamp("M") - m["last"]).dt.days <= 7]   # encore coté en fin de mois
    if m.empty:
        return None
    return pd.DataFrame({"month": m.index.astype(str), "symbol": code, "cap": m["cap"].values,
                         "dv": m["dv"].values, "name": g.get("Name") or code, "sector": sector})


def build_caps():
    codes = list(eod.nasdaq_candidates()["Code"])
    with ProcessPoolExecutor(C.WORKERS) as ex:
        parts = [p for p in ex.map(_caps_one, codes, chunksize=64) if p is not None]
    caps = pd.concat(parts, ignore_index=True)
    C.DATA.mkdir(parents=True, exist_ok=True)
    caps.to_parquet(CAPS, index=False)
    return caps


def effective_date(year):
    """Lundi suivant le 3e vendredi de décembre de l'année précédente."""
    fridays = pd.date_range(f"{year - 1}-12-01", f"{year - 1}-12-31", freq="W-FRI")
    return pd.Timestamp(fridays[2]) + pd.Timedelta(days=3)


def reconstitute(ranked, prev, size=C.UNIVERSE_SIZE, enter=C.UNIVERSE_ENTER_RANK, keep=C.UNIVERSE_KEEP_RANK):
    """ranked : symboles éligibles triés par capitalisation décroissante ; prev : membres actuels."""
    rank = {s: i + 1 for i, s in enumerate(ranked)}
    members = [s for s in ranked if (s in prev and rank[s] <= keep) or rank[s] <= enter]
    members = members[:size]
    for s in ranked:
        if len(members) >= size:
            break
        if s not in members:
            members.append(s)
    return members


def build_universe(caps=None, last_month=None):
    """→ DataFrame (symbol, name, start, end) ; end exclu, NaT = encore membre."""
    caps = pd.read_parquet(CAPS) if caps is None else caps
    caps = caps[caps["dv"].fillna(0) >= C.UNIVERSE_MIN_DOLLAR_VOL]
    # capitalisation incohérente avec les échanges (nombre d'actions faux, ratio d'ADR) : écartée
    caps = caps[caps["dv"] * 252 / caps["cap"] >= C.UNIVERSE_MIN_TURNOVER]
    months = sorted(caps["month"].unique())
    last_month = last_month or months[-1]
    by_month = {m: g.sort_values("cap", ascending=False)["symbol"].tolist() for m, g in caps.groupby("month")}
    names = caps.drop_duplicates("symbol", keep="last").set_index("symbol")["name"].to_dict()
    open_, rows, members = {}, [], []

    def set_members(new, when):
        for s in set(members) - set(new):
            rows.append((s, open_.pop(s), when))
        for s in set(new) - set(members):
            open_[s] = when
        members[:] = new

    year = C.UNIVERSE_FIRST_YEAR
    while f"{year - 1}-11" in by_month and f"{year - 1}-11" <= last_month:
        set_members(reconstitute(by_month[f"{year - 1}-11"], set(members)), effective_date(year))
        # remplacements mensuels jusqu'à la reconstitution suivante
        for m in pd.period_range(f"{year - 1}-12", f"{year}-10", freq="M").astype(str):
            if m not in by_month or m > last_month or m == months[-1]:
                continue                       # mois en cours : données incomplètes, pas de radiation
            alive = by_month[m]
            alive_set = set(alive)
            gone = [s for s in members if s not in alive_set]
            if gone:
                new = [s for s in members if s in alive_set]
                new += [s for s in alive if s not in new][:len(gone)]
                when = (pd.Period(m, "M") + 1).to_timestamp()
                set_members(new, max(when, effective_date(year)))
        year += 1
    rows += [(s, d, pd.NaT) for s, d in open_.items()]
    u = pd.DataFrame(rows, columns=["symbol", "start", "end"])
    u["name"] = u["symbol"].map(names)
    return u.sort_values(["start", "symbol"]).reset_index(drop=True)


def load_universe():
    return pd.read_parquet(UNIVERSE)


def members_on(u, date):
    date = pd.Timestamp(date)
    return sorted(u[(u["start"] <= date) & (u["end"].isna() | (u["end"] > date))]["symbol"])


def member_mask(u, symbol, dates):
    """Booléen par date : le titre est-il membre ?"""
    m = np.zeros(len(dates), dtype=bool)
    for s, e in u[u["symbol"] == symbol][["start", "end"]].itertuples(index=False):
        m |= (dates >= s) & ((dates < e) if pd.notna(e) else True)
    return m


def main():
    caps = build_caps()
    u = build_universe(caps)
    u.to_parquet(UNIVERSE, index=False)
    C.OUT.mkdir(parents=True, exist_ok=True)
    u.to_csv(C.OUT / "universe.csv", index=False)
    print(f"candidats NASDAQ avec capitalisation : {caps['symbol'].nunique()}")
    print(f"univers : {u['symbol'].nunique()} titres distincts, {len(u)} périodes d'appartenance")
    top = caps[caps["month"] == caps["month"].max()].sort_values("cap", ascending=False)
    for y in range(C.UNIVERSE_FIRST_YEAR + 1, pd.Timestamp.today().year + 1):
        d = effective_date(y) + pd.Timedelta(days=30)
        m = members_on(u, d)
        if m:
            c = caps[(caps["month"] == f"{y - 1}-11") & caps["symbol"].isin(m)].sort_values("cap", ascending=False)
            print(f"  {y} : {len(m)} membres ; 10 premiers : {' '.join(c['symbol'].head(10))}")
    print("dernier classement :", " ".join(top["symbol"].head(15)))


if __name__ == "__main__":
    main()
