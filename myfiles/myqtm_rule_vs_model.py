"""Le modèle myqtm fait-il mieux qu'une règle simple RSI / momentum ?

Même moteur que le top 1 sélectif (n° 1 du jour, rang trimestriel, lots dynamiques, sortie EMA 3 < EMA 10,
frais 0,05 %), en remplaçant P(haussier) par un score simple calculé sur les features du jour :
rsi_14, momentum_1w, moyenne des rangs (RSI, momentum). Comparé aux modèles (normal, sous-éch. sévère),
prédictions sur les 156 features d'origine (suffixe _noctx). 2024 → fin.

    PYTHONPATH=.:myfiles python myfiles/myqtm_rule_vs_model.py
"""
import pandas as pd

import myqtm_close_threshold as B
import myqtm_top1_selective as S
from myqtm_topk_dynamic import simulate
from myqtm_topk_slots import signals, stats
from src.portefolio_strategy.myqtm import features as F
from src.portefolio_strategy.myqtm import train as T
from src.portefolio_strategy.myqtm.trade import ema_exit

TOPS = (1.0, 0.7, 0.5, 0.4, 0.3, 0.2, 0.1)


def run(p, opens, end):
    cand = signals(p, 1)
    cand = cand[cand["exit"] <= end].merge(S.day_scores(p)[["rang_trim"]], left_on="signal", right_index=True)
    out = {}
    for k in TOPS:
        c = cand[cand["rang_trim"] >= 1 - k - 1e-9]
        v, _, _ = simulate(c, opens)
        out[k] = dict(trades=len(c), **stats(v))
    return out


def main():
    T.PREDICTIONS = T.C.DATA / "predictions_from2024_noctx.parquet"
    base = B.load()
    base = base.merge(pd.read_parquet(T.PREDICTIONS, columns=["date", "symbol", "ybear"]), on=["date", "symbol"])
    base = base[base["date"] >= S.FIRST].sort_values(["symbol", "date"]).reset_index(drop=True)
    base["x_ema"] = base.groupby("symbol")["c"].transform(ema_exit).astype(bool)
    allf = pd.read_parquet(F.FEATURES, columns=["date", "symbol", "o", "rsi_14", "momentum_1w"])
    opens = allf.pivot(index="date", columns="symbol", values="o").sort_index()
    nd = pd.Series(opens.index[1:].append(pd.DatetimeIndex([pd.NaT])), index=opens.index)
    base["next_date"] = base["date"].map(nd)
    base = base.merge(allf[["date", "symbol", "rsi_14", "momentum_1w"]], on=["date", "symbol"], how="left")
    end = base["date"].max()
    opens = opens.loc[base["date"].min():end]
    ts0 = pd.read_parquet(T.C.DATA / "predictions_ts0_from2024_noctx.parquet", columns=["date", "symbol", "ybull"])
    scores = {
        "modèle normal": base["ybull"],
        "modèle sous-éch. sévère": base[["date", "symbol"]].merge(ts0, how="left")["ybull"].to_numpy(),
        "règle RSI": base["rsi_14"] / 100,
        "règle momentum 1 sem.": base["momentum_1w"],
        "règle RSI + momentum (rangs)": (base.groupby("date")["rsi_14"].rank(pct=True)
                                         + base.groupby("date")["momentum_1w"].rank(pct=True)) / 2,
    }
    rows = []
    for name, sc in scores.items():
        p = base.copy()
        p["ybull"] = sc
        for k, r in run(p, opens, end).items():
            rows.append(dict(score=name, top=k, **r))
    out = pd.DataFrame(rows)
    out.to_csv("myfiles/myqtm_rule_vs_model.csv", index=False)
    pd.set_option("display.width", 220, "display.float_format", "{:.2f}".format)
    for v in ("rendement_an_pct", "sharpe", "max_drawdown_pct", "trades"):
        print(f"\n{v}")
        print(out.pivot(index="top", columns="score", values=v)[list(scores)].sort_index(ascending=False).to_string())
    idx = pd.read_parquet(F.INDEX)["level"].loc[base["date"].min():end]
    print("\nindice :", {k: round(float(v), 2) for k, v in stats(idx).items()})
    m = base.groupby("date").apply(lambda g: pd.Series({
        "n1_modele_est_n1_rsi": g["ybull"].idxmax() == g["rsi_14"].idxmax(),
        "n1_modele_dans_top5_rsi": g.loc[g["ybull"].idxmax(), "rsi_14"] >= g["rsi_14"].nlargest(5).min(),
        "ecart_p_1_2": g["ybull"].nlargest(2).diff().abs().iloc[-1]}), include_groups=False)
    print(f"\nle n° 1 du modèle normal est aussi le n° 1 en RSI : {m['n1_modele_est_n1_rsi'].mean():.0%} des jours ; "
          f"dans les 5 premiers en RSI : {m['n1_modele_dans_top5_rsi'].mean():.0%} ; "
          f"écart médian de P entre n° 1 et n° 2 : {m['ecart_p_1_2'].median():.4f}")


if __name__ == "__main__":
    main()
