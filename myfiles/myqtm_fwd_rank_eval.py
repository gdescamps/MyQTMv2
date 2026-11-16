"""Évaluation du classement à horizon fixe (src/portefolio_strategy/myqtm/fwd_rank.py), S&P 500, 2024 → fin.

Par horizon h (21 / 63 séances) :
- corrélation de rang prédiction ↔ rendement relatif, par jour ; moyenne, et t de Student sur des dates espacées de h
  séances (cibles non recouvrantes) ;
- rendement relatif moyen par décile de prédiction (dates espacées de h), premier − dernier décile, par période ;
- portefeuille : chaque semaine (5 séances), achat à l'ouverture suivante des N premiers titres, gardés h séances
  (tranches qui se recouvrent), parts égales, frais 0,05 % par sens ; alpha et bêta contre l'univers équipondéré.

    PYTHONPATH=.:myfiles python myfiles/myqtm_fwd_rank_eval.py
"""
import numpy as np
import pandas as pd

from myqtm_alpha_eval import rel_stats
from myqtm_earn_drift_eval import simulate
from src.portefolio_strategy.myqtm import config as C
from src.portefolio_strategy.myqtm import features as F
from src.portefolio_strategy.myqtm import fwd_rank as R
from src.portefolio_strategy.myqtm.sp import use_sp

SPLIT = pd.Timestamp("2026-01-01")
TOPS = (10, 25, 50)


def main():
    use_sp()
    f = pd.read_parquet(F.FEATURES, columns=["date", "symbol", "member", "o"]).sort_values(["symbol", "date"])
    opens = f.pivot(index="date", columns="symbol", values="o").sort_index().ffill()
    m = f[f["member"]].assign(o_next=f.groupby("symbol")["o"].shift(-1))
    pd.set_option("display.width", 250, "display.float_format", "{:.2f}".format)
    rows, dec_rows = [], []
    for h in R.HORIZONS:
        p = pd.read_parquet(C.DATA / f"fwd_pred_h{h}.parquet")
        days = np.sort(p["date"].unique())
        opens_h = opens.loc[days[0]:]
        ref = (m["o_next"] / m["o"] - 1).clip(-0.5, 1.0).groupby(m["date"]).mean().loc[days[0]:days[-1]]
        q = p.dropna(subset=["y"])
        ic = q.groupby("date")[["pred", "y"]].apply(lambda d: d["pred"].corr(d["y"], method="spearman"))
        for per, msk in (("2024-25", ic.index < SPLIT), ("2026", ic.index >= SPLIT), ("2024-26", ic.index >= ic.index[0])):
            x = ic[msk]
            xs = x.iloc[::h]
            rows.append(dict(horizon=h, période=per, ic_moyen=x.mean(), ic_t_non_recouvrant=xs.mean() / xs.std() * np.sqrt(len(xs)),
                             dates_indépendantes=len(xs)))
        spaced = q[q["date"].isin(days[::h])].copy()
        spaced["décile"] = spaced.groupby("date")["pred"].transform(lambda s: np.ceil(s.rank(pct=True) * 10).clip(1, 10))
        for per, d in (("2024-25", spaced[spaced["date"] < SPLIT]), ("2026", spaced[spaced["date"] >= SPLIT]), ("2024-26", spaced)):
            by = 100 * d.groupby("décile")["y"].mean()
            sp_d = d[d["décile"] == 10].groupby("date")["y"].mean() - d[d["décile"] == 1].groupby("date")["y"].mean()
            dec_rows.append(dict(horizon=h, période=per, **{f"d{int(k)}": v for k, v in by.items()},
                                 écart_10_1=100 * sp_d.mean(), t=sp_d.mean() / sp_d.std() * np.sqrt(len(sp_d))))
        nd = pd.Series(list(days[1:]) + [pd.NaT], index=days)
        xd = pd.Series(list(days[1 + h:]) + [pd.NaT] * (1 + h), index=days)
        week = p[p["date"].isin(days[::5])].copy()
        week["rk"] = week.groupby("date")["pred"].rank(ascending=False, method="first")
        week["entry"], week["exit"] = week["date"].map(nd), week["date"].map(xd)
        week["exit"] = week["exit"].fillna(days[-1])
        week = week.dropna(subset=["entry"])
        for n in TOPS:
            c = week[(week["rk"] <= n) & (week["exit"] > week["entry"])]
            v, npos = simulate(c[["entry", "exit", "symbol"]], opens_h, n_min=n)
            rows.append(dict(horizon=h, période=f"portefeuille top {n} hebdo", trades=len(c), pos_moy=npos.mean(),
                             **rel_stats(v, ref)))
    out, dec = pd.DataFrame(rows), pd.DataFrame(dec_rows)
    out.to_csv("myfiles/myqtm_fwd_rank_eval.csv", index=False)
    dec.to_csv("myfiles/myqtm_fwd_rank_deciles.csv", index=False)
    print(out.to_string(index=False))
    print("\nrendement relatif moyen (%) par décile de prédiction (10 = meilleures prédictions)")
    print(dec.to_string(index=False))


if __name__ == "__main__":
    main()
