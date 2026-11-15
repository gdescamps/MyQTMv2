"""Entrées myqtm indépendantes du calibrage de chaque modèle, sortie EMA 3 < EMA 10.

Option 1 — classement du jour : entrée quand un titre entre dans le top k des P(haussier) du jour parmi les
  membres (il n'y était pas la veille). Seul l'ordre entre titres compte.
(L'option 2, seuil = quantile des prédictions de validation de chaque modèle, a été retirée : les modèles sont
plus confiants en validation qu'en test, les seuils obtenus étaient trop hauts.)

Pour chaque règle : trades déclenchés, gain moyen net de frais (0,05 % par sens), gain total = (1 + gain moyen
net)^trades (tout le capital sur chaque trade, sans notion de calendrier, irréaliste), part de gagnants, et
nombre de modèles (coupes trimestrielles) sans aucun trade. Entrées 2021-01 → 2026-11.
Colonnes « 1 à la fois » : masquage temporel, une seule position à la fois sur tout l'univers ; tant qu'un
trade est ouvert, aucun autre signal n'est pris (signaux du même jour : la plus forte P(haussier) d'abord ;
un signal le jour du signal de sortie est pris, achat à l'ouverture où l'on vend).

    PYTHONPATH=.:myfiles python myfiles/myqtm_entry_rank.py
"""
import json

import numpy as np
import pandas as pd

import myqtm_close_threshold as B
from src.portefolio_strategy.myqtm import config as C
from src.portefolio_strategy.myqtm.trade import ema_exit

TOPS = (1, 2, 3, 5, 10, 20)


def trades_from_entries(p, entry):
    entry, ex, sym = entry.to_numpy(), p["x_ema"].to_numpy(), p["symbol"].to_numpy()
    rows, i, n = [], 0, len(p)
    while i < n:
        if not entry[i]:
            i += 1
            continue
        j = i + 1
        while j < n and sym[j] == sym[i] and not ex[j]:
            j += 1
        if j >= n or sym[j] != sym[i]:
            i = j
            continue
        rows.append((p["date"].iat[i], p["date"].iat[j], p["ybull"].iat[i],
                     p["o_next"].iat[j] / p["o_next"].iat[i] - 1))
        i = j + 1
    return pd.DataFrame(rows, columns=["signal", "exit_signal", "prob", "ret"]).dropna()


def one_at_a_time(tr):
    """Masquage temporel : un trade n'est pris que si aucune position n'est ouverte."""
    taken, free_from = [], pd.Timestamp.min
    for r in tr.sort_values(["signal", "prob"], ascending=[True, False]).itertuples():
        if r.signal >= free_from:
            taken.append(r.ret)
            free_from = r.exit_signal
    return pd.Series(taken, dtype=float)


def crossing(p, flag):
    return flag & ~flag.groupby(p["symbol"]).shift(1, fill_value=False)


def summary(name, tr, cuts):
    net = tr["ret"] - 2 * C.FEE
    model = pd.cut(tr["signal"], list(cuts) + [pd.Timestamp("2100-01-01")], right=False)
    return dict(regle=name, trades=len(tr),
                gain_moy_net_pct=100 * net.mean(), total_net_x=(1 + net.mean()) ** len(tr),
                gagnants_pct=100 * (tr["ret"] > 0).mean(),
                modeles_sans_trade=int(len(cuts) - model.nunique()),
                **one_summary(one_at_a_time(tr)))


def one_summary(r):
    net = r - 2 * C.FEE
    return {"1_a_la_fois_trades": len(r), "1_a_la_fois_gain_net_pct": 100 * net.mean() if len(r) else np.nan,
            "1_a_la_fois_total_x": (1 + net.mean()) ** len(r) if len(r) else 1.0,
            "1_a_la_fois_reel_x": float(np.prod(1 + net))}


def main():
    p = B.load()
    p["x_ema"] = p.groupby("symbol")["c"].transform(ema_exit).astype(bool)
    rep = json.loads((C.OUT / "train_report.json").read_text())
    cuts = pd.to_datetime([c["cutoff"] for c in rep["cutoffs"]])
    p["cutoff"] = pd.cut(p["date"], list(cuts) + [pd.Timestamp("2100-01-01")], right=False,
                         labels=list(cuts)).astype("datetime64[ns]")
    rows = []
    rank = p.groupby("date")["ybull"].rank(ascending=False, method="first")
    for k in TOPS:
        rows.append(summary(f"top {k} du jour", trades_from_entries(p, crossing(p, rank <= k)), cuts))
    out = pd.DataFrame(rows)
    out.to_csv("myfiles/myqtm_entry_rank.csv", index=False)
    pd.set_option("display.width", 220, "display.float_format", "{:.2f}".format)
    print(out.to_string(index=False))

if __name__ == "__main__":
    main()
