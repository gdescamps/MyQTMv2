"""Version HORAIRE de la stratégie x1.7 PUST+LQQ : peut-on couper plus vite ?

Données : CFD Nasdaq-100 1h (data/stocks/USATECHIDXUSD_1h.parquet, 2012 →, ~23 h/24) pour le chemin
intraday ; QQQ daily (comme la prod) pour la MA250, la YZ, la macro.
Contrainte PEA : PUST/LQQ ne se traitent que 09:00-17:30 Paris → décisions aux clôtures des barres
horaires 09:00 … 17:00 Paris. Clôture US (16:00 New York) = référence journalière.

Instruments (espace index) : PUST = indice ; LQQ = 2x avec RESET quotidien à la clôture US
(NAV_t = NAV_ref·(1 + 2·(P_t/P_ref − 1))) ; drag quotidien (TER, financement + spread swap) à la clôture US.
Frais 0.1 %/côté sur le notionnel traité ; bande asymétrique 0.50 / 1.00 en exposition (comme la prod).

Variantes :
  journalier (prod)      : cible du close US J, exécutée à la barre de 09:00 Paris J+1
  09h prix du matin      : 1 décision/jour à 09:00 Paris, formule recalculée avec le prix futures du moment
  horaire complet        : formule recalculée à chaque barre 09:00-17:00 Paris (achats + ventes)
  horaire coupe seule    : cible journalière le matin ; en journée, seules les VENTES sont permises
  × vol : YZ de la veille (prix seul) | max(YZ veille, vol horaire 10 j mise à l'échelle YZ, causal)
Prix « nowcast » QQQ = close QQQ J × P_t / P_close_US_J.
    python -m myfiles.pust17_hourly
"""
import numpy as np
import pandas as pd

import src.risk_off_strategy.strategy as st
from src.risk_off_strategy.data import load_funding, load_macro, load_ohlc
from src.risk_off_strategy.strategy import E_MAX, compute_allocation

FEE = 0.001
BUY_THR, SELL_THR = st.BUY_THR_E, st.SELL_THR_E
SRC = "data/stocks/USATECHIDXUSD_1h.parquet"
START = "2013-01-01"
HALVES = {"2013-2019": ("2013-01-01", "2019-12-31"), "2020-2026": ("2020-01-01", None), "tout": ("2013-01-01", None)}
EPIS = {"2015-08": ("2015-08-01", "2015-10-31"), "2016-01": ("2015-12-15", "2016-03-31"), "2018-02": ("2018-01-25", "2018-04-30"),
        "2018T4": ("2018-10-01", "2018-12-31"), "2020": ("2020-02-15", "2020-04-30"), "2022": ("2021-11-15", "2022-12-31"),
        "2024-08": ("2024-07-10", "2024-08-31"), "2025": ("2025-02-15", "2025-05-31")}


def formula(q, s, slope_up, rv, nf, cp):
    gap = q / s - 1
    dd = np.clip(1 + gap / st.GAP_CUTOFF, 0, 1)
    du = np.clip(1 - np.maximum(0, gap - st.GAP2_START) / st.GAP2_SPAN, st.DECAY2_FLOOR, 1)
    above = np.clip(st.ABOVE_CAP / rv, 0, 1) * du
    below = st.BELOW_SCALE * np.clip(st.VOL_TARGET / rv, 0, 1) * dd
    a = np.clip(np.where((q > s) & slope_up, above, below), 0, 1)
    a = np.where(np.nan_to_num(nf, nan=-9) > st.NFCI_OFF, 0, a)
    a = np.where(np.nan_to_num(cp, nan=-9) > st.CPI_OFF, 0, a)
    return np.minimum(2 * a, E_MAX)


def main():
    h = pd.read_parquet(SRC)[["close"]].sort_index()
    h = h[h.index >= pd.Timestamp("2012-01-01", tz="UTC")]
    ny = h.index.tz_convert("America/New_York")
    paris_close = (h.index + pd.Timedelta(hours=1)).tz_convert("Europe/Paris")   # horodatage = début de barre
    is_usclose = np.asarray(ny.hour == 15)                                         # barre 15:00-16:00 NY
    ph = np.asarray(paris_close.hour)
    is_trade = (ph >= 9) & (ph <= 17)
    is_trade &= np.asarray(paris_close.dayofweek) < 5
    P = h.close.values
    # séance US de rattachement : date NY de la dernière clôture US passée
    us_date = pd.Series(ny.normalize().tz_localize(None), index=h.index).where(is_usclose).ffill()
    # ── journalier QQQ (prod) ──
    o = load_ohlc("QQQ", "2000-01-01", None)
    qi = o.index
    nfci, cpi = load_macro(qi)
    fund = pd.Series(np.asarray(load_funding(qi), float), qi)
    alloc_d = compute_allocation(o.close.values, nfci, cpi, high=o.high.values, low=o.low.values, open_=o.open.values)
    E_d = pd.Series(np.minimum(2 * alloc_d, E_MAX), qi)
    yz = pd.Series(st.yang_zhang_vol(o.open.values, o.high.values, o.low.values, o.close.values), qi)
    closes = o.close
    sma = closes.rolling(st.SMA_LONG, min_periods=1).mean()
    sma_k = sma.shift(st.SLOPE_K - 1)                                  # s(J+1−K) pour la pente
    nf_s, cp_s = pd.Series(nfci, qi), pd.Series(cpi, qi)
    # vol horaire 10 j (230 barres) annualisée, mise à l'échelle YZ par ratio médian causal (250 séances)
    rh = np.log(h.close).diff().fillna(0)
    rvh = np.sqrt((rh ** 2).rolling(230, min_periods=100).sum() / 10 * 252)
    rvh_close = pd.Series(rvh.values[is_usclose], pd.DatetimeIndex(us_date.values[is_usclose]))
    rvh_close = rvh_close[~rvh_close.index.duplicated()]
    k = (yz.reindex(rvh_close.index) / rvh_close).rolling(250, min_periods=60).median().shift(1)
    # ── tableau par barre : infos de la dernière clôture US J ──
    D = pd.DatetimeIndex(us_date.values)
    ok = D.isin(qi) & (h.index >= pd.Timestamp(START, tz="UTC"))
    ref_close_P = pd.Series(np.where(is_usclose, P, np.nan), h.index).ffill().values
    qJ = closes.reindex(D).values
    qnow = qJ * P / ref_close_P
    s_now = (closes.rolling(st.SMA_LONG - 1).sum().reindex(D).values + qnow) / st.SMA_LONG
    slope = s_now > sma_k.reindex(D).values
    yzJ = yz.reindex(D).values
    rv_h = (rvh.values * k.reindex(D).values)
    rv_mix = np.where(np.isnan(rv_h), yzJ, np.maximum(yzJ, rv_h))
    nfJ, cpJ = nf_s.reindex(D).values, cp_s.reindex(D).values
    E_now = {"YZ veille": formula(qnow, s_now, slope, yzJ, nfJ, cpJ),
             "max(YZ, vol 1h)": formula(qnow, s_now, slope, rv_mix, nfJ, cpJ)}
    E_daily = E_d.reindex(D).values
    fundJ = fund.reindex(D).values
    # première barre tradable après chaque clôture US (= 09:00 Paris J+1)
    first = np.zeros(len(h), bool)
    seen = None
    for i in np.where(is_trade)[0]:
        if D[i] != seen:
            first[i], seen = True, D[i]

    def sim(mode, vol):
        En = E_now[vol] if vol else None
        vp = vl = 0.0
        vc = 1.0
        lqq_nav, lqq_ref_nav, ref_p = 1.0, 1.0, None
        pust_nav = 1.0
        last_p = None
        eq = np.full(len(h), np.nan)
        fees = revis = 0
        for i in range(len(h)):
            if not ok[i]:
                continue
            p = P[i]
            if last_p is None:
                last_p, ref_p = p, p
            # valorisation
            new_pust = pust_nav * p / last_p
            new_lqq = lqq_ref_nav * (1 + 2 * (p / ref_p - 1))
            vp *= new_pust / pust_nav
            vl *= new_lqq / lqq_nav if lqq_nav > 0 else 1
            pust_nav, lqq_nav, last_p = new_pust, new_lqq, p
            if is_usclose[i]:                                   # reset LQQ + drag journalier
                f = fundJ[i] if not np.isnan(fundJ[i]) else 0.0
                vp *= 1 - st.TER_PUST / 252
                vl *= 1 - (f + st.SWAP_SPREAD + st.TER_LQQ) / 252
                lqq_ref_nav, ref_p = lqq_nav, p
            V = vp + vl + vc
            if is_trade[i]:
                e_eff = (vp + 2 * vl) / V
                et = None
                if mode == "journalier" and first[i]:
                    et = E_daily[i]
                elif mode == "09h" and first[i]:
                    et = En[i]
                elif mode == "horaire":
                    et = En[i]
                elif mode == "coupe":
                    if first[i]:
                        et = E_daily[i]
                    elif En[i] < e_eff:
                        et = En[i]
                if et is not None and not np.isnan(et):
                    force = (et == 0 and e_eff > 1e-9) or (e_eff == 0 and et > 1e-9)
                    thr = BUY_THR if et > e_eff else SELL_THR
                    if abs(et - e_eff) >= thr or force:
                        wl = max(0.0, et - 1)
                        wp = et if et <= 1 else 2 - et
                        fee = FEE * (abs(wp * V - vp) + abs(wl * V - vl))
                        fees += fee / V
                        V -= fee
                        vp, vl, vc = wp * V, wl * V, max(0.0, 1 - et) * V
                        revis += 1
            eq[i] = V
        e = pd.Series(eq[is_usclose & ok], pd.DatetimeIndex(D[is_usclose & ok]))
        e = e[~e.index.duplicated()].dropna()
        yrs = len(e) / 252
        return e, 100 * fees / yrs, revis / yrs

    rows = []
    for name, mode, vol in (("journalier (prod)", "journalier", None),
                            ("09h prix du matin, YZ veille", "09h", "YZ veille"),
                            ("09h prix du matin, max(YZ,1h)", "09h", "max(YZ, vol 1h)"),
                            ("horaire complet, YZ veille", "horaire", "YZ veille"),
                            ("horaire complet, max(YZ,1h)", "horaire", "max(YZ, vol 1h)"),
                            ("horaire coupe seule, YZ veille", "coupe", "YZ veille"),
                            ("horaire coupe seule, max(YZ,1h)", "coupe", "max(YZ, vol 1h)")):
        e, fz, rv = sim(mode, vol)
        row = {"variante": name, "frais %/an": round(fz, 2), "revis/an": round(rv, 1)}
        for hn, (a, b) in HALVES.items():
            m = e.loc[a:b] if b else e.loc[a:]
            x = m / m.iloc[0]
            cg = x.iloc[-1] ** (252 / len(x)) - 1
            dd = (x / x.cummax() - 1).min()
            row.update({f"{hn} %/an": round(100 * cg, 1), f"{hn} maxDD": round(100 * dd, 1), f"{hn} Calmar": round(cg / abs(dd), 2)})
        for kk, (a, b) in EPIS.items():
            m = e.loc[a:b]
            row[kk] = round(100 * (m.iloc[-1] / m.iloc[0] - 1))
        rows.append(row)
        print(f"  {name} fait", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv("myfiles/pust17_hourly.csv", index=False)
    pd.set_option("display.width", 280)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
