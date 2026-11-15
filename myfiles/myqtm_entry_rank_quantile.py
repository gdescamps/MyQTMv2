"""Entrées myqtm indépendantes du calibrage de chaque modèle, sortie EMA 3 < EMA 10.

Option 1 — classement du jour : entrée quand un titre entre dans le top k des P(haussier) du jour parmi les
  membres (il n'y était pas la veille). Seul l'ordre entre titres compte.
Option 2 — seuil relatif par modèle : seuil = quantile q des P(haussier) de validation du modèle de la coupe
  (data/portefolio_strategy/myqtm/val_predictions.parquet : blocs B prédits par A et inversement, tous
  antérieurs à la coupe → causal). Entrée quand P(haussier) franchit le seuil de son modèle.

Pour chaque règle : trades déclenchés, gain moyen brut et net (0,05 % par sens), gain total = (1 + gain moyen
net)^trades (tout le capital sur chaque trade, sans notion de calendrier, irréaliste), part de gagnants, et
nombre de modèles (coupes trimestrielles) sans aucun trade. Entrées 2021-01 → 2026-11.

    PYTHONPATH=.:myfiles python myfiles/myqtm_entry_rank_quantile.py
"""
import json

import numpy as np
import pandas as pd

import myqtm_close_threshold as B
from src.portefolio_strategy.myqtm import config as C
from src.portefolio_strategy.myqtm import train as T
from src.portefolio_strategy.myqtm.trade import ema_exit

TOPS = (1, 2, 3, 5, 10, 20)
QUANTILES = (0.90, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999)
FIXED = (0.80, 0.90)                                  # rappel : seuils fixes


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
        rows.append((p["date"].iat[i], p["o_next"].iat[j] / p["o_next"].iat[i] - 1))
        i = j + 1
    return pd.DataFrame(rows, columns=["signal", "ret"]).dropna()


def crossing(p, flag):
    return flag & ~flag.groupby(p["symbol"]).shift(1, fill_value=False)


def summary(name, tr, cuts):
    net = tr["ret"] - 2 * C.FEE
    model = pd.cut(tr["signal"], list(cuts) + [pd.Timestamp("2100-01-01")], right=False)
    return dict(regle=name, trades=len(tr), gain_moy_brut_pct=100 * tr["ret"].mean(),
                gain_moy_net_pct=100 * net.mean(), total_net_x=(1 + net.mean()) ** len(tr),
                gagnants_pct=100 * (tr["ret"] > 0).mean(),
                modeles_sans_trade=int(len(cuts) - model.nunique()))


def main():
    p = B.load()
    p["x_ema"] = p.groupby("symbol")["c"].transform(ema_exit).astype(bool)
    rep = json.loads((C.OUT / "train_report.json").read_text())
    cuts = pd.to_datetime([c["cutoff"] for c in rep["cutoffs"]])
    p["cutoff"] = pd.cut(p["date"], list(cuts) + [pd.Timestamp("2100-01-01")], right=False,
                         labels=list(cuts)).astype("datetime64[ns]")
    rows = []
    for t in FIXED:
        rows.append(summary(f"seuil fixe {t:.2f}", trades_from_entries(p, crossing(p, p["ybull"] >= t)), cuts))
    rank = p.groupby("date")["ybull"].rank(ascending=False, method="first")
    for k in TOPS:
        rows.append(summary(f"option 1 : top {k} du jour", trades_from_entries(p, crossing(p, rank <= k)), cuts))
    val = pd.read_parquet(T.VAL_PREDICTIONS, columns=["cutoff", "ybull"])
    thr = {}
    for q in QUANTILES:
        th = val.groupby("cutoff")["ybull"].quantile(q)
        thr[q] = th
        flag = p["ybull"] >= p["cutoff"].map(th)
        rows.append(summary(f"option 2 : quantile {q:.3f} de validation", trades_from_entries(p, crossing(p, flag)), cuts))
    out = pd.DataFrame(rows)
    out.to_csv("myfiles/myqtm_entry_rank_quantile.csv", index=False)
    pd.set_option("display.width", 220, "display.float_format", "{:.2f}".format)
    print(out.to_string(index=False))
    th = pd.DataFrame({f"q{q}": thr[q] for q in (0.95, 0.99, 0.995)})
    th.index = th.index.strftime("%Y-%m")
    print("\nseuils P(haussier) par modèle (quantiles de validation) :")
    print(th.round(3).T.to_string())


if __name__ == "__main__":
    main()
