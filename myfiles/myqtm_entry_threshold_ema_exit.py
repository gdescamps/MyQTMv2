"""Seuil d'entrée P(haussier) ≥ t avec fermeture EMA 3 < EMA 10 : précision sur le début des hausses,
nombre de trades et gain moyen.

Entrée = premier jour où P(haussier) franchit t (achat à l'ouverture suivante) ; sortie = premier jour
suivant où EMA 3 < EMA 10 à la clôture (vente à l'ouverture suivante) ; un trade à la fois par titre, sans
limite de positions. Précision d'entrée sur palier X = part des entrées tombant dans les X % premiers jours
d'une hausse. Le tableau jour par jour (myqtm_confidence_head.csv) est rappelé pour le palier haussier.

    PYTHONPATH=.:myfiles python myfiles/myqtm_entry_threshold_ema_exit.py
"""
import numpy as np
import pandas as pd

import myqtm_close_threshold as B
from src.portefolio_strategy.myqtm import config as C
from src.portefolio_strategy.myqtm.trade import ema_exit

THRESHOLDS = (0.33, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
HEADS = (0.05, 0.10, 0.15, 0.20, 0.25)


def trades(p, t):
    above = p["ybull"] >= t
    entry = (above & ~above.groupby(p["symbol"]).shift(1, fill_value=False)).to_numpy()
    ex = p["x_ema"].to_numpy()
    sym = p["symbol"].to_numpy()
    rows, i, n = [], 0, len(p)
    while i < n:
        if not entry[i]:
            i += 1
            continue
        j = i + 1
        while j < n and sym[j] == sym[i] and not ex[j]:
            j += 1
        if j >= n or sym[j] != sym[i]:                 # pas de sortie avant la fin des données du titre
            i = j
            continue
        e, x = p.iloc[i], p.iloc[j]
        r = x["o_next"] / e["o_next"] - 1
        rows.append(dict(date=e["date"], bull=e["bull"], phase=e["phase"], days=j - i, ret=r,
                         rel=r - (x["mkt"] / e["mkt"] - 1)))
        i = j + 1
    return pd.DataFrame(rows).dropna(subset=["ret"])


def main():
    p = B.load()
    p["x_ema"] = p.groupby("symbol")["c"].transform(ema_exit).astype(bool)
    rows = []
    for t in THRESHOLDS:
        r = trades(p, t)
        row = dict(seuil=t, trades=len(r), trades_par_an=len(r) / ((p["date"].max() - p["date"].min()).days / 365.25),
                   duree_med_j=r["days"].median(), dans_hausse_pct=100 * r["bull"].mean())
        for h in HEADS:
            row[f"prec_{int(h * 100)}%"] = 100 * (r["bull"] & (r["phase"] < h)).mean()
        row.update(gain_moy_pct=100 * r["ret"].mean(), gain_net_pct=100 * (r["ret"].mean() - 2 * C.FEE),
                   gain_med_pct=100 * r["ret"].median(), rel_moy_pct=100 * r["rel"].mean(),
                   gagnants_pct=100 * (r["ret"] > 0).mean())
        m = r.groupby(r["date"].dt.to_period("M"))["rel"].mean()
        row["t_rel_mensuel"] = m.mean() / m.std() * np.sqrt(len(m))
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv("myfiles/myqtm_entry_threshold_ema_exit.csv", index=False)
    pd.set_option("display.width", 250, "display.float_format", "{:.2f}".format)
    day = pd.read_csv("myfiles/myqtm_confidence_head.csv")
    day = day[day["classe"] == "haussier"].pivot(index="seuil", columns="palier", values="precision") * 100
    print("=== Précision jour par jour sur le début des hausses (rappel du tableau précédent, %) ===")
    print(day[[f"{int(h * 100)} %" for h in HEADS]].to_string())
    print(f"\n=== Entrée P(haussier) franchit t, sortie EMA {C.TREND['ma_mid']} < EMA {C.TREND['ma_long']} "
          f"(frais {C.FEE:.2%} par sens) ===")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
