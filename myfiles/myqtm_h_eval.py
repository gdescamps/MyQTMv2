"""Évaluation du classifieur horaire (train.py --hourly) : trades, portefeuille, avantage des news récentes.

Signal à la clôture d'une barre (10 h 30 … 16 h, heure de New York) ; exécution à l'ouverture de la barre suivante
du titre (le lendemain 9 h 30 après la dernière barre). Entrée quand le titre entre dans le top k de la barre ;
sortie : EMA 3 < EMA 10 sur les clôtures horaires (trade.ema_exit), exécutée à l'ouverture suivante.
Deux modes :
- « nuit permise » : les positions peuvent rester ouvertes d'une séance à l'autre ;
- « intraday » : pas d'entrée sur la dernière barre de la séance, toute position est vendue à la clôture de la
  séance (cours de clôture de la dernière barre).
Portefeuille = lots dynamiques 1/n (myqtm_topk_dynamic.simulate) sur la grille des cours d'exécution, frais
0,05 % par sens ; Sharpe / drawdown sur la valeur de fin de séance. Tops : k premiers de la barre (k = 1, 2, 3, 5)
et top 1 sélectif (rang du score du n° 1 parmi les barres précédentes du trimestre, ≥ 1 − q).
Filtre résultats : prochaine réaction aux résultats ≤ 20 jours (d_earn_days_to_next, veille).
News : gain des trades selon qu'un article du titre a été publié dans la barre du signal (hn_n1) ou dans les
7 dernières barres (hn_n7).

    PYTHONPATH=.:myfiles python myfiles/myqtm_h_eval.py [--pred=ts0_from2024]
"""
import sys

import numpy as np
import pandas as pd

from myqtm_topk_dynamic import simulate
from myqtm_topk_slots import stats
from src.portefolio_strategy.myqtm import config as C
from src.portefolio_strategy.myqtm import hourly as H
from src.portefolio_strategy.myqtm.trade import ema_exit

H.use_hourly()
FIRST = "2024-01-01"
KS = (1, 2, 3, 5)
QS = (0.7, 0.5, 0.3, 0.1)
MIN_BARS = 70                  # pas de top sélectif dans les 70 premières barres (10 séances) du trimestre


def load(pred):
    f = pd.read_parquet(H.FEATURES, columns=["date", "symbol", "o", "c", "h_bar", "hn_n1", "hn_n7",
                                             "d_earn_days_to_next"]).sort_values(["symbol", "date"])
    f["start"] = f["date"] - pd.Timedelta(hours=1)
    last_day = f["date"].dt.normalize()
    f["last_bar"] = last_day.ne(last_day.groupby(f["symbol"]).shift(-1))       # dernière barre de la séance
    f["x_ema"] = f.groupby("symbol")["c"].transform(ema_exit).astype(bool)
    f["next_date"] = f.groupby("symbol")["start"].shift(-1)                     # ouverture de la barre suivante
    # grille des cours d'exécution : ouvertures des barres + clôture de séance (sortie intraday)
    g_open = f[["start", "symbol", "o"]].rename(columns={"start": "t", "o": "px"})
    g_close = f.loc[f["last_bar"], ["date", "symbol", "c"]].rename(columns={"date": "t", "c": "px"})
    grid = pd.concat([g_open, g_close]).drop_duplicates(["t", "symbol"], keep="last")
    grid = grid.pivot(index="t", columns="symbol", values="px").sort_index()
    p = pd.read_parquet(C.DATA / f"predictions_{pred}.parquet", columns=["date", "symbol", "ybull"])
    p = p.merge(f, on=["date", "symbol"]).sort_values(["symbol", "date"]).reset_index(drop=True)
    p = p[p["date"] >= FIRST].reset_index(drop=True)
    return p, grid.loc[p["start"].min():]


def signals_h(p, k, intraday):
    """→ trades (signal, entry, exit, symbol, prob, hn_n1, hn_n7, earn, bar)."""
    rank = p.groupby("date")["ybull"].rank(ascending=False, method="first")
    flag = (rank <= k).to_numpy()
    sym = p["symbol"].to_numpy()
    prev = np.concatenate([[False], flag[:-1]]) & np.concatenate([[False], sym[1:] == sym[:-1]])
    entry = flag & ~prev
    if intraday:
        entry &= ~p["last_bar"].to_numpy()
    ex, last, dates = p["x_ema"].to_numpy(), p["last_bar"].to_numpy(), p["date"].to_numpy()
    nxt = p["next_date"].to_numpy()
    rows, i, n = [], 0, len(p)
    while i < n:
        if not entry[i] or pd.isna(nxt[i]):
            i += 1
            continue
        j = i + 1
        while j < n and sym[j] == sym[i] and not ex[j] and not (intraday and last[j]):
            j += 1
        if j >= n or sym[j] != sym[i]:
            i = j
            continue
        x = dates[j] if intraday and last[j] else nxt[j]     # intraday : vente à la clôture de la séance
        if pd.isna(x):
            i = j
            continue
        rows.append((dates[i], nxt[i], x, sym[i], p["ybull"].iat[i], p["hn_n1"].iat[i], p["hn_n7"].iat[i],
                     p["d_earn_days_to_next"].iat[i], p["h_bar"].iat[i]))
        i = j + 1
    return pd.DataFrame(rows, columns=["signal", "entry", "exit", "symbol", "prob", "hn_n1", "hn_n7", "earn", "bar"])


def bar_scores(p):
    """Rang trimestriel du score du n° 1 de chaque barre parmi les barres précédentes du trimestre."""
    top = p.sort_values(["date", "ybull"], ascending=[True, False]).groupby("date").head(1).set_index("date")["ybull"]
    rk = []
    for _, x in top.groupby(top.index.to_period("Q")):
        v = x.to_numpy()
        rk.append(pd.Series([np.nan if i < MIN_BARS else (v[:i] < v[i]).mean() for i in range(len(v))], index=x.index))
    return pd.concat(rk).rename("rang_trim")


def run(c, grid):
    v, n, rot = simulate(c, grid)
    daily = v.groupby(v.index.normalize()).last()
    ret = [grid.at[x, s] / grid.at[e, s] * (1 - C.FEE) ** 2 - 1 for e, x, s in zip(c["entry"], c["exit"], c["symbol"])]
    c = c.assign(gain=ret)
    hold = (c["exit"] - c["entry"]).dt.total_seconds() / 3600
    years = (daily.index[-1] - daily.index[0]).days / 365.25
    return c, dict(trades=len(c), trades_an=len(c) / years, gain_moy_pct=100 * np.nanmean(ret),
                   gain_med_pct=100 * np.nanmedian(ret), gagnants_pct=100 * np.nanmean(np.array(ret) > 0),
                   duree_med_h=hold.median(), pos_moy=n.mean(), **stats(daily))


def main():
    pred = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--pred=")), "ts0_from2024")
    p, grid = load(pred)
    end = grid.index[-1]
    rk = bar_scores(p)
    rows, news = [], []
    for intraday in (False, True):
        mode = "intraday" if intraday else "nuit permise"
        for k in KS:
            cand = signals_h(p, k, intraday)
            cand = cand[cand["exit"] <= end].merge(rk, left_on="signal", right_index=True, how="left")
            sets = {f"top {k}": cand, f"top {k} + résultats ≤ 20 j": cand[cand["earn"] <= 20]}
            if k == 1:
                for q in QS:
                    sets[f"top 1 sélectif {q}"] = cand[cand["rang_trim"] >= 1 - q - 1e-9]
            for name, c in sets.items():
                if len(c) < 3:
                    continue
                c, st = run(c, grid)
                rows.append(dict(mode=mode, règle=name, **st))
                if name == f"top {k}":
                    for lab, msk in (("article dans la barre", c["hn_n1"] > 0), ("article sur 7 barres", c["hn_n7"] > 0),
                                     ("aucun article sur 7 barres", c["hn_n7"] == 0)):
                        x = c.loc[msk, "gain"]
                        news.append(dict(mode=mode, top=k, groupe=lab, trades=len(x), gain_moy_pct=100 * x.mean(),
                                         gain_med_pct=100 * x.median(), gagnants_pct=100 * (x > 0).mean()))
                    if k == 3:
                        print(f"\n{mode}, top 3 : gain moyen par barre du signal (%)")
                        gb = c.groupby("bar")["gain"]
                        print(pd.DataFrame({"trades": gb.size(), "moy": 100 * gb.mean(),
                                            "méd": 100 * gb.median()}).round(3).to_string())
            print(f"{mode} top {k} ok", flush=True)
    out, nw = pd.DataFrame(rows), pd.DataFrame(news)
    tag = pred.replace("_from2024", "")
    out.to_csv(f"myfiles/myqtm_h_eval_{tag}.csv", index=False)
    nw.to_csv(f"myfiles/myqtm_h_eval_news_{tag}.csv", index=False)
    pd.set_option("display.width", 250, "display.float_format", "{:.2f}".format, "display.max_rows", 200)
    print(out.to_string(index=False))
    print(nw.to_string(index=False))


if __name__ == "__main__":
    main()
