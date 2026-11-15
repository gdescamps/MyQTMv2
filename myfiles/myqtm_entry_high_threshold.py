"""Seuils d'entrée élevés (0,80 → 0,99) avec sortie EMA 3 < EMA 10 : rendement par trade, nombre de trades,
rendement total.

- Rendement total théorique = (1 + gain moyen)^nombre de trades = chaque trade pris avec tout le capital,
  l'un après l'autre. Irréalisable quand des trades se chevauchent dans le temps.
- Rendement réalisable « une position à la fois » : on ne prend un trade que si aucune position n'est
  ouverte (achat à l'ouverture du lendemain du signal) ; si plusieurs signaux tombent le même jour, celui
  de plus forte P(haussier). Tout le capital sur la position, frais 0,05 % par sens.

    PYTHONPATH=.:myfiles python myfiles/myqtm_entry_high_threshold.py
"""
import numpy as np
import pandas as pd

import myqtm_close_threshold as B
from src.portefolio_strategy.myqtm import config as C
from src.portefolio_strategy.myqtm import features as F
from src.portefolio_strategy.myqtm.trade import ema_exit

THRESHOLDS = (0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98, 0.99)


def all_trades(p, t):
    above = p["ybull"] >= t
    entry = (above & ~above.groupby(p["symbol"]).shift(1, fill_value=False)).to_numpy()
    ex, sym = p["x_ema"].to_numpy(), p["symbol"].to_numpy()
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
        e, x = p.iloc[i], p.iloc[j]
        r = x["o_next"] / e["o_next"] - 1
        rows.append(dict(signal=e["date"], exit_signal=x["date"], prob=e["ybull"], ret=r,
                         rel=r - (x["mkt"] / e["mkt"] - 1), days=j - i))
        i = j + 1
    return pd.DataFrame(rows).dropna(subset=["ret"])


def one_at_a_time(tr):
    """Une seule position : un trade n'est pris que si son signal tombe après la sortie du précédent."""
    tr = tr.sort_values(["signal", "prob"], ascending=[True, False])
    taken, free_from = [], pd.Timestamp.min
    for r in tr.itertuples():
        if r.signal >= free_from:                 # vente à l'ouverture J+1 du signal de sortie = achat possible
            taken.append(r)
            free_from = r.exit_signal
    return pd.DataFrame(taken)


def main():
    p = B.load()
    p["x_ema"] = p.groupby("symbol")["c"].transform(ema_exit).astype(bool)
    years = (p["date"].max() - p["date"].min()).days / 365.25
    idx = p.groupby("date")["mkt"].first()                   # moyenne des ouvertures (référence équipondérée)
    rows = []
    for t in THRESHOLDS:
        tr = all_trades(p, t)
        if tr.empty:
            continue
        net = tr["ret"] - 2 * C.FEE
        one = one_at_a_time(tr)
        onet = one["ret"] - 2 * C.FEE
        tot1 = np.prod(1 + onet) - 1
        m = tr.groupby(tr["signal"].dt.to_period("M"))["rel"].mean()
        rows.append(dict(seuil=t, trades=len(tr), gain_moy_net_pct=100 * net.mean(), rel_moy_pct=100 * tr["rel"].mean(),
                         gagnants_pct=100 * (tr["ret"] > 0).mean(), duree_med_j=tr["days"].median(),
                         total_theorique_pct=100 * ((1 + net.mean()) ** len(tr) - 1),
                         une_pos_trades=len(one), une_pos_gain_moy_pct=100 * onet.mean(),
                         une_pos_total_pct=100 * tot1, une_pos_cagr_pct=100 * ((1 + tot1) ** (1 / years) - 1),
                         une_pos_temps_investi_pct=100 * one["days"].sum() / len(idx),
                         t_rel_mensuel=m.mean() / m.std() * np.sqrt(len(m)) if len(m) > 2 else np.nan))
    out = pd.DataFrame(rows)
    out.to_csv("myfiles/myqtm_entry_high_threshold.csv", index=False)
    lvl = pd.read_parquet(F.INDEX)["level"]
    lvl = lvl[(lvl.index >= p["date"].min()) & (lvl.index <= p["date"].max())]
    ew = lvl.iloc[-1] / lvl.iloc[0] - 1
    pd.set_option("display.width", 250, "display.float_format", "{:.2f}".format)
    print(f"période {p['date'].min():%Y-%m-%d} → {p['date'].max():%Y-%m-%d} ({years:.1f} ans) ; "
          f"indice de l'univers (pondéré par les capitalisations) : {100 * ew:+.1f} %, "
          f"CAGR {100 * ((1 + ew) ** (1 / years) - 1):.1f} %")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
