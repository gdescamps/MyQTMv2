"""Étape 1 : une feature du jour du signal est-elle liée au gain des trades ?

Signaux d'entrée des tops 1 … 15 du jour (modèle de référence : sous-échantillonnage sévère, 164 features,
predictions_ts0_from2024), sortie EMA 3 < EMA 10, 2024 → fin ; trades distincts (titre, date d'entrée).
Pour chaque feature (et P(haussier)) au jour du signal : corrélation de rang (Spearman) avec
- le gain brut du trade (ouverture → ouverture) ;
- le gain relatif à l'indice de l'univers sur la même période (retire l'effet marché).
p-valeur, seuil de Bonferroni sur le nombre de features, signe par année (2024 / 2025 / 2026), écart de gain
moyen entre le quintile haut et le quintile bas de la feature.

    PYTHONPATH=.:myfiles python myfiles/myqtm_signal_feature_ic.py
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import myqtm_close_threshold as B
from myqtm_topk_slots import signals
from src.portefolio_strategy.myqtm import features as F
from src.portefolio_strategy.myqtm import train as T
from src.portefolio_strategy.myqtm.trade import ema_exit

FIRST = "2024-01-01"


def main():
    T.PREDICTIONS = T.predictions_path(first=FIRST, ts=0.0)
    p = B.load()
    p = p[p["date"] >= FIRST].sort_values(["symbol", "date"]).reset_index(drop=True)
    p["x_ema"] = p.groupby("symbol")["c"].transform(ema_exit).astype(bool)
    allf = pd.read_parquet(F.FEATURES)
    opens = allf.pivot(index="date", columns="symbol", values="o").sort_index().ffill()
    nd = pd.Series(opens.index[1:].append(pd.DatetimeIndex([pd.NaT])), index=opens.index)
    p["next_date"] = p["date"].map(nd)
    end = p["date"].max()
    cand = pd.concat([signals(p, k).assign(k=k) for k in range(1, 16)])
    cand = cand[cand["exit"] <= end].sort_values("k").drop_duplicates(["symbol", "entry"])
    idx = pd.read_parquet(F.INDEX)["level"].reindex(opens.index).ffill()
    cand["ret"] = [opens.at[x, s] / opens.at[e, s] - 1 for e, x, s in zip(cand["entry"], cand["exit"], cand["symbol"])]
    cand["ret_rel"] = cand["ret"] - (idx.loc[cand["exit"]].to_numpy() / idx.loc[cand["entry"]].to_numpy() - 1)
    cols = F.feature_columns(allf)
    X = cand.merge(allf[["date", "symbol"] + cols], left_on=["signal", "symbol"], right_on=["date", "symbol"])
    X["p_haussier"] = X["prob"]
    cols = ["p_haussier"] + cols
    year = X["signal"].dt.year
    rows = []
    for c in cols:
        x = X[c]
        ok = x.notna()
        if x[ok].nunique() < 5:
            continue
        rb, pb = spearmanr(x[ok], X.loc[ok, "ret"])
        rr, pr = spearmanr(x[ok], X.loc[ok, "ret_rel"])
        signs = [np.sign(spearmanr(x[ok & (year == y)], X.loc[ok & (year == y), "ret_rel"])[0]) for y in (2024, 2025, 2026)]
        q = pd.qcut(x[ok].rank(method="first"), 5, labels=False)
        g = X.loc[ok, "ret_rel"].groupby(q).mean()
        rows.append(dict(feature=c, ic_brut=rb, p_brut=pb, ic_rel=rr, p_rel=pr, signe_meme_3_ans=len(set(signs)) == 1,
                         q5_moins_q1_rel_pct=100 * (g.iloc[-1] - g.iloc[0])))
    r = pd.DataFrame(rows)
    bonf = 0.05 / len(r)
    r["signif_bonferroni"] = r["p_rel"] < bonf
    r = r.sort_values("p_rel")
    r.to_csv("myfiles/myqtm_signal_feature_ic.csv", index=False)
    pd.set_option("display.width", 220, "display.float_format", "{:.4f}".format)
    print(f"{len(X)} trades distincts (tops 1 à 15), {X['signal'].min():%Y-%m-%d} → {X['signal'].max():%Y-%m-%d} ; "
          f"gain médian {100 * X['ret'].median():.2f} %, relatif médian {100 * X['ret_rel'].median():.2f} %")
    print(f"{len(r)} features testées ; seuil de Bonferroni p < {bonf:.1e} ; "
          f"significatives : {r['signif_bonferroni'].sum()} ; p < 0,05 : {(r['p_rel'] < 0.05).sum()} "
          f"(≈ {0.05 * len(r):.0f} attendues par hasard)\n")
    print(r.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
