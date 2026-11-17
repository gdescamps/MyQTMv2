"""Indice concentré classé par le modèle « ensemble » quotidien (src/portefolio_strategy/myqtm/set_rank.py).

Rotation mensuelle (scores de fin de mois), top 10/20/30/40, parts égales ou inverse de la volatilité, sans et avec
la limitation « risk-off + garde-fous macro » de myqtm_index_vol_overlay.py. Références : XGBoost futurs leaders,
momentum 6 mois, plus grosses capitalisations, univers équipondéré. But : battre le Sharpe de l'univers équipondéré
sur 2019-2023 (choix) ET 2024-26 (vérification).

    PYTHONPATH=.:myfiles python myfiles/myqtm_set_eval.py [--tag=nom]
"""
import sys

import numpy as np
import pandas as pd

import myqtm_index_sharpe as IS
import myqtm_index_vol_overlay as VO
from src.portefolio_strategy.myqtm import monthly_rank as M
from src.portefolio_strategy.myqtm import set_rank as SR

NS = (10, 20, 30, 40)
SCHEMES = ("égal", "inverse volatilité")
LIMITS = ("sans limitation", "risk-off + garde-fous macro")


def main():
    for a in sys.argv[1:]:
        if a.startswith("--tag="):
            SR.TAG = "_" + a.split("=")[1]
    opens, bench = pd.read_parquet(M.OPENS), pd.read_parquet(M.BENCH)
    p = pd.read_parquet(SR.monthly_path())
    p = p[p["exit"].notna() & p["pred_set"].notna()].reset_index(drop=True)
    big = p["cap"].groupby(p["date"]).rank(ascending=False)
    r = lambda col: p[col].groupby(p["date"]).rank(pct=True)        # noqa: E731
    sc = {"modèle ensemble": p["pred_set"],
          "ensemble parmi les 100 plus grosses": p["pred_set"].where(big <= 100),
          "ensemble parmi les 50 plus grosses": p["pred_set"].where(big <= 50),
          "ensemble + XGBoost futurs leaders": r("pred_set") + r("pred_mom6n"),
          "XGBoost futurs leaders": p["pred_mom6n"], "momentum 6 mois": p["mom6"], "grosses capitalisations": p["cap"]}
    periods, groups = IS.prepare(p, opens)
    sig = VO.signals(bench["ro"])
    ref = bench["ro"].loc[periods[0][0][0]:periods[-1][0][-1]]
    uni_v = (1 + ref.fillna(0)).cumprod().shift(1, fill_value=1.0).iloc[:-1]
    rows = [dict(classement="univers équipondéré", top=500, allocation="égal", limitation=lim,
                 **IS.stats(VO.overlay(uni_v, sig[lim])[0])) for lim in LIMITS]
    for name, s in sc.items():
        p["_s"] = s
        tops = [g.assign(_s=p.loc[g.index, "_s"]).nlargest(max(NS), "_s") for g in groups]
        for n in NS:
            for scheme in SCHEMES:
                picks = [(t["j"].to_numpy()[:n], IS.weights(t.iloc[:n], scheme)) for t in tops]
                v, turn = IS.simulate(periods, picks, opens.shape[1])
                for lim in LIMITS:
                    rows.append(dict(classement=name, top=n, allocation=scheme, limitation=lim, rotation_an_x=12 * turn,
                                     **IS.stats(VO.overlay(v, sig[lim])[0])))
    out = pd.DataFrame(rows)
    out.to_csv(f"myfiles/myqtm_set_eval{SR.TAG}.csv", index=False)
    pd.set_option("display.width", 250, "display.float_format", "{:.2f}".format, "display.max_rows", 400)
    ic = p.groupby("date")[["pred_set", "pred_mom6n", "mom6", "y_next"]].apply(
        lambda g: g[["pred_set", "pred_mom6n", "mom6"]].corrwith(g["y_next"], method="spearman"))
    a, b = ic[ic.index < IS.SPLIT], ic[ic.index >= IS.SPLIT]
    print(f"{len(ic)} mois, {p['date'].min():%Y-%m} → {p['date'].max():%Y-%m}")
    print("corrélation de rang mensuelle avec le rendement du mois suivant (moyenne ; t)")
    print(pd.DataFrame({"2019-23": a.mean(), "2024-26": b.mean(), "ensemble": ic.mean(),
                        "t": ic.mean() / ic.std() * np.sqrt(len(ic))}).round(3).to_string())
    for lim in LIMITS:
        d = out[out["limitation"] == lim]
        u = d[d["classement"] == "univers équipondéré"].iloc[0]
        print(f"\n=== {lim} — univers équipondéré : Sharpe 2019-23 {u['sharpe_2019_23']:.2f}, 2024-26 {u['sharpe_2024_26']:.2f}, "
              f"ensemble {u['sharpe']:.2f}, {u['rendement_an_pct']:.1f} %/an, DD {u['max_dd_pct']:.1f} %")
        d = d[d["classement"] != "univers équipondéré"]
        for col in ("sharpe_2019_23", "sharpe_2024_26", "sharpe", "rendement_an_pct", "max_dd_pct"):
            print(f"\n{col}")
            print(d.pivot_table(index="classement", columns=["allocation", "top"], values=col).reindex(list(sc)).to_string())
    print("\nrotation annuelle (×) :", out.groupby("classement")["rotation_an_x"].mean().round(1).dropna().to_dict())


if __name__ == "__main__":
    main()
