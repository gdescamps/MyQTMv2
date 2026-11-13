"""Univers « S&P 500 » reconstitué point-in-time (substitut : pas de liste historique officielle dans le dépôt).

Règle (approximation de l'indice) :
- candidats : actions ordinaires NYSE + NASDAQ, actives et radiées (pas de biais du survivant), avec cours et
  fondamentaux EODHD, de sociétés « américaines » au sens de l'indice (us_company : ni ADR ni société étrangère ;
  les sociétés domiciliées en Irlande, Suisse, Pays-Bas, Jersey, Bermudes… mais cotées comme des américaines,
  comme Linde, Accenture, Medtronic, NXP, Chubb, restent éligibles) ;
- capitalisation = nombre d'actions connu (fin de trimestre + SHARES_LAG_DAYS) × cours brut ramené aux splits ;
- liquidité : médiane 60 j du volume en dollars >= MIN_DOLLAR_VOL et rotation annuelle >= UNIVERSE_MIN_TURNOVER ;
- reconstitution trimestrielle (lundi suivant le 3e vendredi de mars, juin, septembre, décembre) sur les
  capitalisations de la fin du mois précédent : 500 titres, entrée d'office si rang <= ENTER, maintien si rang <= KEEP ;
  un titre qui cesse d'être coté est remplacé le mois suivant par le premier non-membre.
Les financières sont incluses (contrairement au Nasdaq-100). Les deux classes d'actions d'une même société
(GOOG/GOOGL…) comptent chacune, comme dans l'indice.

Sorties : data/portefolio_strategy/sp500/caps.parquet, universe.parquet, panel.parquet (date, symbol, member,
o, c, rsi_14, ret_21), index.parquet (indice pondéré capitalisation des membres).

    python -m src.portefolio_strategy.sp500
"""
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from . import config as C
from . import eod
from . import features as F
from . import universe as U

DIR = C.DATA / "sp500"
CAPS, UNIVERSE, PANEL, INDEX = DIR / "caps.parquet", DIR / "universe.parquet", DIR / "panel.parquet", DIR / "index.parquet"
CAPS_START = "2013-06-01"
FIRST_RECON = "2013-12"        # première reconstitution : décembre 2013 (capitalisations de fin novembre)
SIZE, ENTER, KEEP = 500, 450, 600
MIN_DOLLAR_VOL = 10e6
PANEL_START = "2013-09-01"     # historique pour le RSI 14 j et le rendement 21 j


def candidates():
    fr = []
    for name in ("symbols_active", "symbols_delisted"):
        d = pd.read_parquet(C.EODHD / f"{name}.parquet")
        fr.append(d[(d["Type"] == "Common Stock") & d["Exchange"].isin(["NYSE", "NASDAQ"])][["Code"]])
    codes = pd.concat(fr)["Code"].drop_duplicates()
    return [c for c in codes if (C.EODHD / "eod" / f"{c}.parquet").exists()
            and (C.EODHD / "fundamentals" / f"{c}.json.gz").exists()]


TAX_HOMES = {"IE", "JE", "CH", "NL", "BM", "LR", "PA"}


def us_company(g):
    """General EODHD → société éligible : catégorie « Domestic » (pas ADR, canadienne…), adresse aux États-Unis ou
    domiciliation fiscale classique des grandes sociétés américaines. Écarte p. ex. AstraZeneca (GB), Arm, ASML."""
    if not str(g.get("HomeCategory") or "").startswith("Domestic"):
        return False
    country = (g.get("AddressData") or {}).get("Country") or ""
    return country == "United States" or str(g.get("ISIN") or "")[:2] in TAX_HOMES


def _caps_one(code):
    fund = eod.load_fund(code)
    g = (fund or {}).get("General") or {}
    if not us_company(g):
        return None
    sh = eod.shares_series(fund)
    px = eod.load_eod(code, start=CAPS_START)
    if sh is None or px is None:
        return None
    dv = (px["split_close"] * px["v"]).rolling(60, min_periods=20).median()
    shares = sh.reindex(px.index.union(sh.index)).ffill().reindex(px.index)
    d = pd.DataFrame({"cap": px["split_close"] * shares, "dv": dv, "last": px.index}, index=px.index)
    m = d.groupby(px.index.to_period("M")).last().dropna(subset=["cap"])
    m = m[(m.index.to_timestamp("M") - m["last"]).dt.days <= 7]
    if m.empty:
        return None
    return pd.DataFrame({"month": m.index.astype(str), "symbol": code, "cap": m["cap"].values, "dv": m["dv"].values,
                         "name": g.get("Name") or code, "sector": g.get("GicSector") or g.get("Sector") or "",
                         "group": g.get("GicGroup") or ""})


def build_caps():
    codes = candidates()
    print(f"candidats NYSE + NASDAQ avec cours et fondamentaux : {len(codes)}", flush=True)
    with ProcessPoolExecutor(C.WORKERS) as ex:
        parts = [p for p in ex.map(_caps_one, codes, chunksize=64) if p is not None]
    caps = pd.concat(parts, ignore_index=True)
    DIR.mkdir(parents=True, exist_ok=True)
    caps.to_parquet(CAPS, index=False)
    return caps


def effective(month):
    """Lundi suivant le 3e vendredi du mois `month` (AAAA-MM)."""
    p = pd.Period(month, "M")
    fridays = pd.date_range(p.start_time, p.end_time, freq="W-FRI")
    return pd.Timestamp(fridays[2]) + pd.Timedelta(days=3)


def build_universe(caps):
    caps = caps[(caps["dv"].fillna(0) >= MIN_DOLLAR_VOL) & (caps["dv"] * 252 / caps["cap"] >= C.UNIVERSE_MIN_TURNOVER)]
    months = sorted(caps["month"].unique())
    by_month = {m: g.sort_values("cap", ascending=False)["symbol"].tolist() for m, g in caps.groupby("month")}
    names = caps.drop_duplicates("symbol", keep="last").set_index("symbol")["name"].to_dict()
    open_, rows, members = {}, [], []

    def set_members(new, when):
        for s in set(members) - set(new):
            rows.append((s, open_.pop(s), when))
        for s in set(new) - set(members):
            open_[s] = when
        members[:] = new

    recon = [m for m in pd.period_range(FIRST_RECON, months[-1], freq="M").astype(str) if m[-2:] in ("03", "06", "09", "12")]
    for r in recon:
        prev = str(pd.Period(r, "M") - 1)
        if prev not in by_month:
            continue
        when = effective(r)
        set_members(U.reconstitute(by_month[prev], set(members), SIZE, ENTER, KEEP), when)
        for m in pd.period_range(r, pd.Period(r, "M") + 2, freq="M").astype(str):
            if m not in by_month or m == months[-1]:
                continue
            alive = set(by_month[m])
            gone = [s for s in members if s not in alive]
            if gone:
                new = [s for s in members if s in alive]
                new += [s for s in by_month[m] if s not in new][:len(gone)]
                set_members(new, max((pd.Period(m, "M") + 1).to_timestamp(), when))
    rows += [(s, d, pd.NaT) for s, d in open_.items()]
    u = pd.DataFrame(rows, columns=["symbol", "start", "end"])
    u["name"] = u["symbol"].map(names)
    return u.sort_values(["start", "symbol"]).reset_index(drop=True)


def _panel_one(args):
    symbol, u = args
    px = eod.load_eod(symbol, start=PANEL_START)
    if px is None or len(px) < 30:
        return None
    c = px["c"]
    d = pd.DataFrame({"symbol": symbol, "o": px["o"], "c": c, "rsi_14": F.rsi14(c), "ret_21": c / c.shift(21) - 1},
                     index=px.index)
    d["member"] = U.member_mask(u, symbol, d.index)
    return d.rename_axis("date").reset_index()


def build_panel(u):
    syms = sorted(u["symbol"].unique())
    with ProcessPoolExecutor(C.WORKERS) as ex:
        parts = [p for p in ex.map(_panel_one, [(s, u[u["symbol"] == s]) for s in syms], chunksize=16) if p is not None]
    return pd.concat(parts, ignore_index=True)


def main():
    caps = pd.read_parquet(CAPS) if CAPS.exists() else build_caps()
    u = build_universe(caps)
    u.to_parquet(UNIVERSE, index=False)
    panel = build_panel(u)
    panel.to_parquet(PANEL, index=False)
    closes = panel.pivot(index="date", columns="symbol", values="c")
    closes = closes[closes.index >= u["start"].min() - pd.Timedelta(days=40)]
    idx = F.build_index(u, caps[caps["symbol"].isin(closes.columns)], closes)
    idx.to_parquet(INDEX)
    n = panel[panel["member"]].groupby("date").size()
    print(f"univers : {u['symbol'].nunique()} titres distincts, {len(u)} périodes ; membres par jour : "
          f"min {n.min()}, médiane {int(n.median())}, max {n.max()} ({n.index.min():%Y-%m-%d} → {n.index.max():%Y-%m-%d})")
    last = caps[caps["month"] == caps["month"].max()]
    sec = last[last["symbol"].isin(set(panel.loc[panel["date"] == panel["date"].max()].query("member")["symbol"]))]
    print("répartition sectorielle actuelle :", sec["sector"].value_counts().to_dict())
    print("indice : CAGR", round(100 * ((idx["level"].iloc[-1] / idx["level"].iloc[0]) ** (365.25 / (idx.index[-1] - idx.index[0]).days) - 1), 1), "%")


if __name__ == "__main__":
    main()
