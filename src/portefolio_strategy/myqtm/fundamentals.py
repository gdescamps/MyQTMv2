"""Features fondamentales (comptes trimestriels EODHD) et sectorielles (groupe GICS), sans LLM.

Fondamentaux — point-in-time : un trimestre n'est connu qu'au lendemain de son dépôt (filing_date des trois états ;
à défaut fin de trimestre + 46 j). Les ratios instantanés d'EODHD (Highlights, Valuation…) ne sont pas utilisés :
ce sont des photos d'aujourd'hui. Réserve : les montants peuvent être des valeurs retraitées.
- fq_* (par trimestre) : croissance sur un an du chiffre d'affaires (et son accélération), du résultat net et
  opérationnel ; marges brute / opérationnelle / nette et leur variation sur un an ; marge de flux de trésorerie
  disponible ; rentabilité des fonds propres et des actifs ; écart bénéfice − flux opérationnel ; investissements,
  rémunération en actions et R&D rapportés aux ventes ; dette nette / EBITDA ; couverture des intérêts ; dilution ;
  croissance des actifs ; trésorerie / actifs ;
- fv_* (chaque jour, capitalisation du jour = actions connues × cours brut) : rendements bénéfice, ventes, actif net,
  flux opérationnel, flux disponible, dividendes, rachats nets, actionnaire (dividendes + rachats) ; EBITDA / valeur
  d'entreprise ; écart du rendement des ventes et du bénéfice à leur médiane de 3 ans ; log de la capitalisation ;
- fund_days : jours depuis le dernier dépôt.

Secteur — groupe GICS (25 groupes), calculé sur les membres du jour :
- momentum du titre relatif au marché sur 63 / 126 / 252 séances, distance au plus haut 252 j, volatilité 63 j ;
- rendement du groupe sur 5 / 21 / 63 séances, rendement du titre relatif au groupe, rang du groupe parmi les groupes ;
- dispersion des rendements 21 j dans le groupe ; part des titres au-dessus de leur EMA 50 (groupe et marché) ;
- ton des publications récentes (surprise moyenne et part de dépassements des titres ayant publié depuis ≤ 30 j,
  groupe et marché) ;
- chaque ratio fondamental clé relatif à la médiane de son groupe (suffixe _grp).
"""
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from . import config as C
from . import features as F
from .. import eod

ITEMS = {"Income_Statement": ["totalRevenue", "grossProfit", "operatingIncome", "ebitda", "netIncome",
                              "researchDevelopment", "interestExpense"],
         "Cash_Flow": ["totalCashFromOperatingActivities", "capitalExpenditures", "freeCashFlow", "dividendsPaid",
                       "salePurchaseOfStock", "stockBasedCompensation"],
         "Balance_Sheet": ["totalAssets", "totalStockholderEquity", "cash", "netDebt", "commonStockSharesOutstanding"]}
GROUP_REL = ["fv_ey", "fv_sy", "fv_by", "fv_ebitda_ev", "fv_fcfy", "fv_shy", "fq_rev_yoy", "fq_om", "fq_gm", "fq_roe",
             "fq_nd_ebitda", "fq_d_om"]
PREFIXES = ("fq_", "fv_", "fund_days", "grp_", "rel_grp_", "mkt_breadth", "mkt_surp30", "mkt_beat30", "hi252", "vol63",
            "rel_mkt_r63", "rel_mkt_r126", "rel_mkt_r252")


def _div(a, b, lo=None, hi=None, pos=False):
    b = b.where(b > 0) if pos else b.where(b != 0)
    x = a / b
    return x.clip(lo, hi) if lo is not None or hi is not None else x


def quarterly(fund):
    """Comptes trimestriels → une ligne par trimestre, indexée par date de disponibilité, colonnes fq_* et montants
    bruts (_*) pour les ratios de valorisation."""
    fin = (fund or {}).get("Financials") or {}
    rows, avail = {}, {}
    for st, keys in ITEMS.items():
        for per, r in ((fin.get(st) or {}).get("quarterly") or {}).items():
            if per < "2011-01-01":
                continue
            row = rows.setdefault(per, {})
            for k in keys:
                row[k] = F._num(r.get(k))
            fd, end = r.get("filing_date"), pd.Timestamp(per)
            a = pd.Timestamp(fd) + pd.Timedelta(days=1) if fd and pd.Timestamp(fd) > end else end + pd.Timedelta(days=46)
            avail[per] = max(avail.get(per, a), a)
    if len(rows) < 5:
        return None
    d = pd.DataFrame.from_dict(rows, orient="index").reindex(columns=[k for ks in ITEMS.values() for k in ks])
    d.index = pd.to_datetime(d.index)
    d = d.sort_index()
    idx = d.index.to_series()
    ok4 = ((idx - idx.shift(4)).dt.days.between(350, 380)).to_numpy()       # même trimestre un an plus tôt
    ok3 = ((idx - idx.shift(3)).dt.days <= 290).to_numpy()                   # 4 trimestres consécutifs
    rev, gp, opi, ni = d["totalRevenue"], d["grossProfit"], d["operatingIncome"], d["netIncome"]
    cfo, capex = d["totalCashFromOperatingActivities"], d["capitalExpenditures"].abs()
    fcf = d["freeCashFlow"].fillna(cfo - capex)
    assets, eq, shares = d["totalAssets"], d["totalStockholderEquity"], d["commonStockSharesOutstanding"]

    def ttm(s):
        return s.rolling(4, min_periods=4).sum().where(ok3)

    def yoy(s):
        return s.shift(4).where(ok4)

    o = pd.DataFrame(index=d.index)
    o["fq_rev_yoy"] = (_div(rev, yoy(rev), pos=True) - 1).clip(-1, 5)
    o["fq_rev_accel"] = o["fq_rev_yoy"] - o["fq_rev_yoy"].shift(1)
    o["fq_ni_yoy"] = _div(ni - yoy(ni), yoy(ni).abs(), -5, 5)
    o["fq_opi_yoy"] = _div(opi - yoy(opi), yoy(opi).abs(), -5, 5)
    for name, num in (("gm", gp), ("om", opi), ("nm", ni)):
        o[f"fq_{name}"] = _div(num, rev, -2, 1, pos=True)
        o[f"fq_d_{name}"] = o[f"fq_{name}"] - yoy(o[f"fq_{name}"])
    rev_t, ni_t, cfo_t, fcf_t, opi_t, ebitda_t = ttm(rev), ttm(ni), ttm(cfo), ttm(fcf), ttm(opi), ttm(d["ebitda"])
    o["fq_fcfm"] = _div(fcf_t, rev_t, -2, 1, pos=True)
    o["fq_roe"] = _div(ni_t, eq, -2, 2, pos=True)
    o["fq_roa"] = _div(ni_t, assets, -1, 1, pos=True)
    o["fq_accruals"] = _div(ni_t - cfo_t, assets, -1, 1, pos=True)
    o["fq_capex_rev"] = _div(ttm(capex), rev_t, 0, 2, pos=True)
    o["fq_sbc_rev"] = _div(ttm(d["stockBasedCompensation"].abs()), rev_t, 0, 1, pos=True)
    o["fq_rd_rev"] = _div(ttm(d["researchDevelopment"].abs()), rev_t, 0, 2, pos=True)
    o["fq_nd_ebitda"] = _div(d["netDebt"], ebitda_t, -10, 20, pos=True)
    o["fq_int_cov"] = _div(opi_t, ttm(d["interestExpense"].abs()), -50, 100, pos=True)
    o["fq_dilution"] = (_div(shares, yoy(shares), pos=True) - 1).clip(-0.5, 1)
    o["fq_asset_growth"] = (_div(assets, yoy(assets), pos=True) - 1).clip(-0.8, 3)
    o["fq_cash_assets"] = _div(d["cash"], assets, 0, 1, pos=True)
    o["_ni"], o["_rev"], o["_eq"], o["_cfo"], o["_fcf"], o["_ebitda"] = ni_t, rev_t, eq, cfo_t, fcf_t, ebitda_t
    o["_nd"], o["_div"], o["_bb"] = d["netDebt"], ttm(d["dividendsPaid"].abs().fillna(0)), ttm(d["salePurchaseOfStock"].fillna(0))
    o.index = pd.DatetimeIndex([avail[f"{p:%Y-%m-%d}"] for p in d.index])
    return o[~o.index.duplicated(keep="last")].sort_index()


def _one(symbol):
    fund = eod.load_fund(symbol)
    q = quarterly(fund)
    px = eod.load_eod(symbol)
    sh = eod.shares_series(fund) if fund else None
    if q is None or px is None:
        return None
    dates = px.index
    f = F.asof_join(dates, q, "fund_days")
    if sh is not None:
        cap = px["split_close"] * sh.reindex(dates.union(sh.index)).ffill().reindex(dates)
    else:
        cap = pd.Series(np.nan, index=dates)
    cap = cap.where(cap > 0)
    out = f[[c for c in f.columns if c.startswith("fq_")]].copy()
    out["fund_days"] = f["fund_days"].clip(upper=400).where(f["fund_days"] >= 0)
    out["fv_ey"] = (f["_ni"] / cap).clip(-1, 1)
    out["fv_sy"] = (f["_rev"] / cap).clip(0, 20)
    out["fv_by"] = (f["_eq"] / cap).clip(-2, 5)
    out["fv_cfy"] = (f["_cfo"] / cap).clip(-1, 1)
    out["fv_fcfy"] = (f["_fcf"] / cap).clip(-1, 1)
    ev = (cap + f["_nd"].fillna(0))
    out["fv_ebitda_ev"] = (f["_ebitda"] / ev.where(ev > 0)).clip(-1, 1)
    out["fv_divy"] = (f["_div"] / cap).clip(0, 0.5)
    out["fv_bby"] = (-f["_bb"] / cap).clip(-0.5, 0.5)
    out["fv_shy"] = out["fv_divy"] + out["fv_bby"]
    for k in ("fv_sy", "fv_ey"):
        out[f"{k}_hist"] = out[k] - out[k].rolling(750, min_periods=250).median()
    out["fv_logcap"] = np.log(cap)
    out = out.astype("float32")
    out.insert(0, "symbol", symbol)
    return out.rename_axis("date").reset_index()


def add_fundamentals(df):
    syms = sorted(df["symbol"].unique())
    with ProcessPoolExecutor(C.WORKERS) as ex:
        parts = [p for p in ex.map(_one, syms, chunksize=8) if p is not None]
    fund = pd.concat(parts, ignore_index=True)
    return df.merge(fund, on=["date", "symbol"], how="left")


def add_group_features(df, groups):
    """groups : Series symbole → groupe GICS."""
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    grp = df["symbol"].map(groups).fillna("?").rename("grp")
    mem, date = df["member"], df["date"]
    g = df.groupby("symbol")["c"]
    new = {}
    for h in (63, 126, 252):
        r = g.pct_change(h, fill_method=None)
        new[f"rel_mkt_r{h}"] = r - r.where(mem).groupby(date).transform("mean")
    new["hi252"] = df["c"] / g.transform(lambda c: c.rolling(252, min_periods=120).max()) - 1
    new["vol63"] = g.pct_change(fill_method=None).groupby(df["symbol"]).transform(lambda r: r.rolling(63, min_periods=30).std())
    for h in (5, 21, 63):
        r = g.pct_change(h, fill_method=None)
        rm = r.where(mem)
        gm = rm.groupby([date, grp]).transform("mean")
        new[f"grp_r{h}"], new[f"rel_grp_r{h}"] = gm, r - gm
        tab = rm.groupby([date, grp]).mean().rename("v").reset_index()
        tab["rk"] = tab.groupby("date")["v"].rank(pct=True)
        new[f"grp_rank{h}"] = pd.DataFrame({"date": date, "grp": grp}).merge(tab[["date", "grp", "rk"]], on=["date", "grp"],
                                                                          how="left")["rk"].to_numpy()
        if h == 21:
            new["grp_disp21"] = rm.groupby([date, grp]).transform("std")
    above = (df["c"] > g.transform(lambda c: c.ewm(span=50, adjust=False).mean())).astype(float).where(mem)
    new["grp_breadth"] = above.groupby([date, grp]).transform("mean")
    new["mkt_breadth"] = above.groupby(date).transform("mean")
    if "ts_es_surprise_0" in df:
        s = df["ts_es_surprise_0"].where(mem & df["es_days"].between(0, 30))
        beat = (s > 0).astype(float).where(s.notna())
        new["grp_surp30"], new["mkt_surp30"] = s.groupby([date, grp]).transform("mean"), s.groupby(date).transform("mean")
        new["grp_beat30"], new["mkt_beat30"] = beat.groupby([date, grp]).transform("mean"), beat.groupby(date).transform("mean")
    for c in GROUP_REL:
        if c in df:
            new[f"{c}_grp"] = df[c] - df[c].where(mem).groupby([date, grp]).transform("median")
    return pd.concat([df, pd.DataFrame(new, index=df.index).astype("float32")], axis=1)


def added_columns(df):
    return [c for c in df.columns if c.startswith(PREFIXES) or c.endswith("_grp")]
