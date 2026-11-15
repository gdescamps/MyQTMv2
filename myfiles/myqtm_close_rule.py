"""Fermeture des positions longues myqtm par la règle experte des moyennes mobiles (version causale).

trends.detect_trends définit une hausse entre deux croisements EMA moyenne / EMA longue et recule sa fin au
croisement court / moyen précédent : c'est un label ex post (il faut connaître la suite). En temps réel, on ne
peut utiliser que les croisements eux-mêmes, constatés à la clôture de J et exécutés à l'ouverture de J+1 :
  court/moyen : EMA(ma_short) < EMA(ma_mid)   (ma_short = 1 → le cours passe sous son EMA 3)
  moyen/long  : EMA(ma_mid) < EMA(ma_long)    (EMA 3 sous EMA 10 : la fin de hausse au sens du label)
Comparé à la sortie P(haussier) < c, avec la même entrée (P(haussier) franchit 0,6) et les mêmes mesures
que myqtm_close_threshold.py (partie B).

    PYTHONPATH=. python myfiles/myqtm_close_rule.py
"""
import numpy as np
import pandas as pd

import myqtm_close_threshold as B
from src.portefolio_strategy.myqtm import config as C


def add_rules(p):
    g = p.groupby("symbol")["c"]
    ema = {k: g.transform(lambda s, k=k: s.ewm(span=k, adjust=False).mean()) for k in
           (C.TREND["ma_short"], C.TREND["ma_mid"], C.TREND["ma_long"])}
    s, m, lg = ema[C.TREND["ma_short"]], ema[C.TREND["ma_mid"]], ema[C.TREND["ma_long"]]
    p["x_court_moyen"] = s < m
    p["x_moyen_long"] = m < lg
    p["x_p037"] = p["ybull"] < 0.37
    p["x_p037_ou_moyen_long"] = p["x_p037"] | p["x_moyen_long"]
    p["x_court_moyen_2j"] = p["x_court_moyen"] & p.groupby("symbol")["x_court_moyen"].shift(1, fill_value=False)
    return p


def simulate(p, col):
    res = []
    for sym, g in p.groupby("symbol", sort=False):
        ent, ex = g["entry"].to_numpy(), g[col].to_numpy()
        i, n = 0, len(g)
        while i < n:
            if not ent[i]:
                i += 1
                continue
            j = i + 1
            while j < n and not ex[j]:
                j += 1
            if j >= n:
                break
            e, x = g.iloc[i], g.iloc[j]
            if x["bull"] and x["phase"] >= 1 - B.TAIL_B:
                where = "fin (20 % derniers)"
            elif x["bull"] and e["bull"] and x["seg"] == e["seg"]:
                where = "prématurée"
            elif not e["bull"] and not x["bull"]:
                where = "entrée hors hausse"
            else:
                where = "après la fin"
            late = np.nan
            if where == "après la fin" and pd.notna(x["last_end"]):
                late = int(((g["date"] > x["last_end"]) & (g["date"] <= x["date"])).sum())
            r = x["o_next"] / e["o_next"] - 1
            res.append(dict(where=where, days=j - i, ret=r, rel=r - (x["mkt"] / e["mkt"] - 1), late=late))
            i = j + 1
    r = pd.DataFrame(res).dropna(subset=["ret"])
    share = r["where"].value_counts(normalize=True)
    return dict(positions=len(r), duree_med=r["days"].median(),
                **{f"% {k}": 100 * share.get(k, 0) for k in
                   ("prématurée", "fin (20 % derniers)", "après la fin", "entrée hors hausse")},
                retard_med_j=r["late"].median(), gain_moy_pct=100 * r["ret"].mean(),
                rel_moy_pct=100 * r["rel"].mean(), rel_par_jour_pb=1e4 * r["rel"].sum() / r["days"].sum(),
                gagnantes_pct=100 * (r["ret"] > 0).mean(), gain_net_moy_pct=100 * (r["ret"].mean() - 2 * C.FEE))


def main():
    p = B.load()
    above = p["ybull"] >= B.OPEN
    p["entry"] = above & ~above.groupby(p["symbol"]).shift(1, fill_value=False)
    p = add_rules(p)
    rules = {"P(haussier) < 0,37 (défaut)": "x_p037",
             "EMA1 < EMA3 (court/moyen)": "x_court_moyen",
             "EMA1 < EMA3 deux jours de suite": "x_court_moyen_2j",
             "EMA3 < EMA10 (moyen/long)": "x_moyen_long",
             "P < 0,37 ou EMA3 < EMA10": "x_p037_ou_moyen_long"}
    out = pd.DataFrame({k: simulate(p, c) for k, c in rules.items()}).T
    out.to_csv("myfiles/myqtm_close_rule.csv")
    pd.set_option("display.width", 250, "display.float_format", "{:.2f}".format)
    print(f"EMA : court {C.TREND['ma_short']}, moyen {C.TREND['ma_mid']}, long {C.TREND['ma_long']} ; "
          f"entrée P(haussier) franchit {B.OPEN} ; frais {C.FEE:.2%} par sens")
    print(out.to_string())


if __name__ == "__main__":
    main()
