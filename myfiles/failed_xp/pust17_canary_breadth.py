"""Canaris (Keller & Keuning DAA/BAA) et breadth comme garde-fous de la strategie deployee x1.7 PUST+LQQ.

Reference : strategie deployee (compute_allocation + NFCI/CPI), simulate_net(leverage=2, e_max=E_MAX),
frais Bourso 0.5 %/cote, sans saturation — memes chiffres que le chart.

Familles testees (le garde-fou s'applique a l'alloc AVANT les coupe-circuits macro, qui restent) :
  daa2     : canaris DAA = emergents (VEIEX) + obligations agregees (VBMFX), momentum 13612W
             (12·r1m + 4·r3m + 2·r6m + r12m)/4 ; "bad" = nb de canaris a momentum < 0.
  baa4     : canaris BAA = SPY + developpes hors US (VTMGX) + emergents + obligations.
             Actions : any_cash (bad>=1 -> 0), any_cap1 (bad>=1 -> expo<=1), frac (alloc·(1-bad/n)),
             all_cash (tous mauvais -> 0). Evaluation quotidienne ou mensuelle (fin de mois, comme Keller).
  sect     : breadth des 9 secteurs SPDR (XLB..XLY) = part au-dessus de leur SMA200 ; garde si < seuil.
  stk      : breadth des ~229 actions de data/stocks au-dessus de leur SMA200 (BIAIS DU SURVIVANT :
             univers actuel, 1 seul delisté) ; garde si < seuil.
  zweig    : breadth thrust de Zweig sur ces actions (EMA10 de la part de hausses : < 0.40 puis > 0.615
             en <= 10 seances) -> pendant 63 seances, alloc >= bras "au-dessus" (min(cap_vol, 1)).
             C'est un signal de RE-ENTREE, pas de protection.
Fonds Vanguard (VL publiee apres le run du soir) decales d'une seance ; ETF US au close du jour.
Validation : seuil choisi sur une moitie (Calmar), lu sur l'autre — H1 2000-2012 / H2 2013-2026.
    python -W ignore -m myfiles.failed_xp.pust17_canary_breadth      (log : myfiles/failed_xp/pust17_canary_breadth.log)

Resultats (2026-11-09, ref = 16.4 %/an, maxDD -29.5 %, Calmar 0.55, H1/H2 0.23/0.88) :
  - Canaris DAA/BAA : TOUS pires. Les canaris sont "mauvais" 45-55 % du temps (obligations en 2013, 2018,
    2022...) ; en cash -> ruine par les frais (jusqu'a 8 %/an) ; meilleure variante baa4/jour/any_cap1 =
    15.5 % / -30.1 % / Calmar 0.51. Redondant avec le filtre de tendance sur un Nasdaq deja suivi.
  - Breadth secteurs / actions : tous <= ref ; seuil regle sur une moitie -> pire ou egal sur l'autre.
  - Zweig thrust (re-entree) : hold 126 = 18.3 % / -31.3 % / 0.58, H1/H2 0.28/0.92 ; hold 63 = neutre.
    Le timing bat nettement le hasard (placebo : boosts aleatoires sous la MA -> maxDD ~ -70 %), mais le gain
    vs ref est modeste, maxDD un peu pire, duree choisie ex post, univers biaise du survivant, et le signal
    live demanderait un flux quotidien avance/declin. A ne PAS deployer en l'etat.
"""
import glob
import os

import numpy as np
import pandas as pd

from src.risk_off_strategy.data import load_funding, load_macro, load_ohlc
from src.risk_off_strategy.strategy import (CPI_OFF, E_MAX, NFCI_OFF, allocation_components, compute_allocation,
                                            simulate_net, yang_zhang_vol)

CACHE = "data/pust17_canary_close.parquet"
TICKERS = ["VEIEX", "VBMFX", "VTMGX", "XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
FUNDS = ["VEIEX", "VBMFX", "VTMGX"]   # VL apres le run du soir -> lag 1
SECTORS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
HALVES = {"H1": ("2000-01-01", "2012-12-31"), "H2": ("2013-01-01", None), "tout": ("2000-01-01", None)}
BREADTH_THR = [0.2, 0.3, 0.4, 0.5, 0.6]


def load_canaries(idx):
    if not os.path.exists(CACHE):
        import yfinance as yf
        yf.download(TICKERS, start="1997-01-01", auto_adjust=True, progress=False)["Close"].to_parquet(CACHE)
    d = pd.read_parquet(CACHE)
    d.index = pd.to_datetime(d.index).tz_localize(None).normalize()
    d[FUNDS] = d[FUNDS].shift(1)
    spy = pd.read_parquet("data/SPY.parquet")["close"]
    spy.index = pd.to_datetime(spy.index).tz_localize(None).normalize()
    d["SPY"] = spy
    return d.reindex(d.index.union(idx)).ffill().reindex(idx)


def mom13612(s):
    r = lambda n: s / s.shift(n) - 1  # noqa: E731
    return (12 * r(21) + 4 * r(63) + 2 * r(126) + r(252)) / 4


def load_stock_breadth(idx):
    closes = {}
    for f in glob.glob("data/stocks/*.parquet"):
        c = pd.read_parquet(f)["close"]
        c.index = pd.to_datetime(c.index, utc=True).tz_localize(None).normalize()
        closes[os.path.basename(f)[:-8]] = c[~c.index.duplicated(keep="last")]
    px = pd.DataFrame(closes).sort_index()
    # un fichier porte des lignes de week-end/jours feries : sans ce filtre, les NaN qu'elles creent dans la
    # fenetre glissante invalident la SMA200 de presque tous les titres -> ne garder que les vraies seances
    px = px[px.notna().sum(axis=1) >= 0.9 * px.shape[1]]
    ma = px.rolling(200, min_periods=200).mean()
    above = (px > ma).astype(float).where(ma.notna() & px.notna())
    breadth = above.mean(axis=1)
    ret = px.pct_change(fill_method=None)
    adv_ratio = (ret > 0).astype(float).where(ret.notna()).mean(axis=1)
    ema = adv_ratio.ewm(span=10, adjust=False).mean()
    low10 = ema.rolling(10).min()
    thrust = (ema > 0.615) & (low10 < 0.40)
    end = px.index.max()
    br = breadth.reindex(idx).where(idx <= end)
    th = thrust.reindex(idx).fillna(False).where(idx <= end, False).astype(bool)
    return br.values, th.values, end


def win(eq, idx, a, b):
    m = (idx >= pd.Timestamp(a)) & (idx <= (pd.Timestamp(b) if b else idx[-1]))
    e = eq[m] / eq[m][0]
    yrs = m.sum() / 252
    cagr = e[-1] ** (1 / yrs) - 1
    dd = (e / np.maximum.accumulate(e) - 1).min()
    return 100 * cagr, 100 * dd, cagr / abs(dd)


def episodes(on):
    on = np.asarray(on, bool)
    return int(((on[1:]) & (~on[:-1])).sum() + on[0])


def main():
    o = load_ohlc("QQQ", "2000-01-01", None)
    p, idx = o["close"], o.index
    kw = dict(high=o["high"].values, low=o["low"].values, open_=o["open"].values)
    nfci, cpi = load_macro(idx)
    fund = load_funding(idx)
    macro_off = np.zeros(len(idx), bool)
    if nfci is not None:
        macro_off |= np.nan_to_num(nfci, nan=-9) > NFCI_OFF
    if cpi is not None:
        macro_off |= np.nan_to_num(cpi, nan=-9) > CPI_OFF
    a0 = compute_allocation(p.values, None, None, **kw)
    rv = yang_zhang_vol(kw["open_"], kw["high"], kw["low"], p.values)
    cap_vol = allocation_components(p.values, rv)["cap_vol"]

    can = load_canaries(idx)
    mom = {t: mom13612(can[t]) for t in ["VEIEX", "VBMFX", "VTMGX", "SPY"]}
    bad = {"daa2": sum((mom[t] < 0).astype(int) for t in ["VEIEX", "VBMFX"]),
           "baa4": sum((mom[t] < 0).astype(int) for t in ["SPY", "VTMGX", "VEIEX", "VBMFX"])}
    n_can = {"daa2": 2, "baa4": 4}
    sect_above = pd.concat([(can[s] > can[s].rolling(200).mean()) for s in SECTORS], axis=1)
    sect_breadth = sect_above.mean(axis=1).values
    stk_breadth, thrust, stk_end = load_stock_breadth(idx)

    rows = []

    def run(fam, param, action, alloc, on):
        a = np.where(macro_off, 0.0, alloc)
        eq, fees, rev = simulate_net(p, a, leverage=2, funding=fund, e_max=E_MAX)
        r = {"famille": fam, "param": param, "action": action, "% jours": round(100 * np.mean(on), 1),
             "episodes": episodes(on), "frais %/an": round(fees, 2), "revis/an": round(rev, 1)}
        for h, (x, y) in HALVES.items():
            c, d, cal = win(eq, idx, x, y)
            r.update({f"{h} %/an": round(c, 1), f"{h} maxDD": round(d, 1), f"{h} Calmar": round(cal, 2)})
        rows.append(r)

    run("ref", "-", "-", a0, np.zeros(len(idx), bool))
    for fam in ("daa2", "baa4"):
        for freq in ("jour", "mois"):
            b = bad[fam].values.astype(float)
            if freq == "mois":
                s = pd.Series(b, idx)
                # valeur de fin du mois precedent, appliquee tout le mois
                per = idx.to_period("M")
                last_prev = s.groupby(per).last().shift(1)
                b = last_prev.reindex(per).fillna(0).values
            n = n_can[fam]
            run(fam, freq, "any_cash", np.where(b >= 1, 0.0, a0), b >= 1)
            run(fam, freq, "any_cap1", np.where(b >= 1, np.minimum(a0, 0.5), a0), b >= 1)
            run(fam, freq, "frac", a0 * (1 - b / n), b >= 1)
            run(fam, freq, "all_cash", np.where(b >= n, 0.0, a0), b >= n)
    for fam, br in (("sect", sect_breadth), ("stk", stk_breadth)):
        for t in BREADTH_THR:
            on = np.nan_to_num(br, nan=1.0) < t
            run(fam, t, "cash", np.where(on, 0.0, a0), on)
            run(fam, t, "cap1", np.where(on, np.minimum(a0, 0.5), a0), on)
    for hold in (21, 63, 126):
        boost = pd.Series(thrust.astype(float), idx).rolling(hold, min_periods=1).max().values > 0
        run("zweig", hold, "boost", np.where(boost, np.maximum(a0, np.minimum(cap_vol, 1.0)), a0), boost)

    df = pd.DataFrame(rows)
    df.to_csv("myfiles/failed_xp/pust17_canary_breadth.csv", index=False)
    pd.set_option("display.width", 260)
    pd.set_option("display.max_rows", 200)
    cols = ["famille", "param", "action", "% jours", "episodes", "H1 %/an", "H1 maxDD", "H1 Calmar",
            "H2 %/an", "H2 maxDD", "H2 Calmar", "tout %/an", "tout maxDD", "tout Calmar", "frais %/an", "revis/an"]
    print(f"Actions data/stocks jusqu'au {stk_end.date()} (breadth/zweig inactifs ensuite) ; "
          f"{int(thrust.sum())} jours de thrust Zweig : {[d.date() for d in idx[thrust]][:30]}")
    print(df[cols].to_string(index=False))

    ref = df.iloc[0]
    out = []
    for (fam, action), g in df[df.famille.isin(["sect", "stk"])].groupby(["famille", "action"]):
        for fit, test in (("H1", "H2"), ("H2", "H1")):
            b = g.loc[g[f"{fit} Calmar"].idxmax()]
            out.append({"famille": fam, "action": action, "regle sur": fit, "seuil": b.param,
                        "Calmar test": b[f"{test} Calmar"], "ref": ref[f"{test} Calmar"],
                        "maxDD test": b[f"{test} maxDD"], "ref maxDD": ref[f"{test} maxDD"],
                        "%/an test": b[f"{test} %/an"], "ref %/an": ref[f"{test} %/an"]})
    print("\n=== Breadth : seuil choisi sur une moitie, lu sur l'autre ===")
    print(pd.DataFrame(out).to_string(index=False))

    # Placebo Zweig : meme nb d'episodes et meme duree, departs tires au hasard parmi les jours ou le boost
    # change quelque chose (alloc < bras "au-dessus") -> le timing du thrust vaut-il mieux que le hasard ?
    t = pd.Series(thrust, idx)
    starts = np.where((t & ~t.shift(1, fill_value=False)).values)[0]
    cand = np.where(a0 < np.minimum(cap_vol, 1.0) - 1e-9)[0]
    rng = np.random.default_rng(1)
    print("\n=== Placebo Zweig (300 tirages) : rang centile du vrai signal (100 = meilleur que tous) ===")
    for hold in (63, 126):
        real = df[(df.famille == "zweig") & (df.param == hold)].iloc[0]
        res = []
        for _ in range(300):
            b = np.zeros(len(idx), bool)
            for s0 in rng.choice(cand[cand < len(idx) - hold], len(starts), replace=False):
                b[s0:s0 + hold] = True
            eq = simulate_net(p, np.where(macro_off, 0.0, np.where(b, np.maximum(a0, np.minimum(cap_vol, 1.0)), a0)),
                              leverage=2, funding=fund, e_max=E_MAX)[0]
            res.append([win(eq, idx, *HALVES[h])[2] for h in HALVES] + [win(eq, idx, *HALVES["tout"])[1]])
        r = np.array(res)
        print(f"hold {hold}: Calmar H1/H2/tout reel {real['H1 Calmar']}/{real['H2 Calmar']}/{real['tout Calmar']}"
              f" | placebo median {np.median(r[:, 0]):.2f}/{np.median(r[:, 1]):.2f}/{np.median(r[:, 2]):.2f}"
              f" maxDD {np.median(r[:, 3]):.1f} | rang H1 {100 * (r[:, 0] < real['H1 Calmar']).mean():.0f}"
              f" H2 {100 * (r[:, 1] < real['H2 Calmar']).mean():.0f} tout {100 * (r[:, 2] < real['tout Calmar']).mean():.0f}")


if __name__ == "__main__":
    main()
