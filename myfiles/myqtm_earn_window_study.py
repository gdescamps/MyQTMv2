"""Où se trouve l'effet « résultats » ? Étude d'événement sur le S&P 500 reconstitué (2015 → fin).

Pour chaque publication d'un membre, rendement relatif à l'univers équipondéré sur trois fenêtres :
- avant   : clôture base−10 → clôture base (les 10 séances qui précèdent l'annonce) ;
- annonce : clôture base → clôture T (la séance de réaction, 1 à 2 séances) ;
- après   : ouverture T+1 → ouverture T+21 (dérive post-annonce, cible de earn_drift.py) ;
selon le rang du momentum 63 séances du titre parmi les membres, mesuré à base−10 (connu à l'entrée).
Puis portefeuilles causaux : achat à l'ouverture de base−9, vente à l'ouverture de T+1 (on porte l'annonce),
parts ≤ 1/10, frais 0,05 % par sens ; toutes les publications, ou momentum dans le premier quintile / tercile.
Hypothèse : la date de publication est connue 10 séances avant (calendrier EODHD tel qu'enregistré).

    PYTHONPATH=.:myfiles python myfiles/myqtm_earn_window_study.py
"""
import numpy as np
import pandas as pd

from myqtm_alpha_eval import rel_stats
from myqtm_earn_drift_eval import simulate, weekly_t
from src.portefolio_strategy.myqtm import earn_drift as E

PRE = 10


def main():
    mats, member = E.matrices()
    bench = E.benchmark(mats, member)
    c, o = mats["c"], mats["o"]
    dates, n = c.index, len(c)
    e = pd.read_parquet(E.EVENTS)
    t = dates.get_indexer(e["tdate"])
    # base : recalculée comme dans earn_drift (dernière clôture avant l'annonce)
    rd = e["report_date"].to_numpy()
    pos = dates.searchsorted(rd, side="left")
    is_sess = dates[np.minimum(pos, n - 1)] == rd
    amc = e["amc"].to_numpy()
    base = np.where(amc == 0, pos - 1, np.where(amc == 1, np.where(is_sess, pos, pos - 1), pos - 1))
    j = c.columns.get_indexer(e["symbol"])
    cn, on, lc, lo = c.to_numpy(float), o.ffill().to_numpy(float), bench["lc"].to_numpy(), bench["lo"].to_numpy()
    rank = (c / c.shift(63) - 1).where(member).rank(axis=1, pct=True).to_numpy()
    ok = base - PRE - 63 >= 0
    e, t, base, j = e[ok].reset_index(drop=True), t[ok], base[ok], j[ok]
    e["mom_rank"] = rank[base - PRE, j]
    e["avant"] = cn[base, j] / cn[base - PRE, j] - 1 - (lc[base] / lc[base - PRE] - 1)
    e["annonce"] = e["react"]
    e["après"] = e["y20"]
    e["port"] = on[np.minimum(t + 1, n - 1), j] / on[base - PRE + 1, j] - 1 \
        - (lo[np.minimum(t + 1, n - 1)] / lo[base - PRE + 1] - 1)           # ce que le portefeuille capte
    e["entry_w"], e["exit_w"] = dates[base - PRE + 1], dates[np.minimum(t + 1, n - 1)]
    e = e[e["mom_rank"].notna() & (t + 1 < n)]
    e["quintile"] = np.ceil(e["mom_rank"] * 5).clip(1, 5).astype(int)
    pd.set_option("display.width", 250, "display.float_format", "{:.2f}".format)
    rows = []
    y24, y26 = pd.Timestamp("2024-01-01"), pd.Timestamp("2026-01-01")
    for per, d in (("2015-2023", e[e["tdate"] < y24]), ("2024-25", e[(e["tdate"] >= y24) & (e["tdate"] < y26)]),
                   ("2026", e[e["tdate"] >= y26])):
        for q, x in [("tous", d)] + [(f"quintile {k}", d[d["quintile"] == k]) for k in (1, 3, 5)]:
            r = dict(période=per, momentum=q, publications=len(x))
            for col in ("avant", "annonce", "après", "port"):
                y = x[col].dropna()
                y = y[y.abs() < 1.5]
                r[f"{col}_pct"], r[f"{col}_t"] = 100 * y.mean(), weekly_t(y, x.loc[y.index, "tdate"])
            r["port_gagnants_pct"] = 100 * (x["port"] > 0).mean()
            rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv("myfiles/myqtm_earn_window_study.csv", index=False)
    print("rendement relatif moyen par fenêtre (%), t sur moyennes hebdomadaires ; quintile 5 = plus fort momentum")
    print(out.to_string(index=False))
    opens = pd.read_parquet(E.OPENS)
    ref = bench["ro"].loc[opens.index[0]:]
    x = e[(e["entry_w"] >= opens.index[0])].drop(columns=["entry"]).rename(columns={"entry_w": "entry", "exit_w": "exit"})
    ports = {"toutes": x, "momentum quintile 5": x[x["quintile"] == 5], "momentum rang ≥ 0,9": x[x["mom_rank"] >= 0.9],
             "momentum quintile 1": x[x["quintile"] == 1]}
    prow = []
    for name, cnd in ports.items():
        v, npos = simulate(cnd[["entry", "exit", "symbol"]], opens)
        prow.append(dict(portefeuille=name, trades=len(cnd), pos_moy=npos.mean(), **rel_stats(v, ref)))
    pf = pd.DataFrame(prow)
    pf.to_csv("myfiles/myqtm_earn_window_portfolios.csv", index=False)
    print("\nportefeuilles « porter l'annonce » 2024 → fin (achat 10 séances avant, vente à l'ouverture après la réaction)")
    print(pf.to_string(index=False))


if __name__ == "__main__":
    main()
