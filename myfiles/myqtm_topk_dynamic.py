"""Portefeuille à lots dynamiques : TOUS les signaux d'entrée « top k du jour » sont pris, sortie EMA 3 < EMA 10.

- Pas de nombre de lots fixe : le portefeuille est réparti à parts égales entre toutes les positions ouvertes
  (1/n chacune) ; à chaque entrée ou sortie, rééquilibrage à l'ouverture vers 1/n (frais sur le volume échangé).
- Aucune position ouverte → 100 % cash (rémunération 0).
- Signal à la clôture de J → exécution à l'ouverture suivante (mêmes signaux que myqtm_topk_slots.signals).
- Frais 0,05 % par sens sur |Δ poids| ; entre deux changements, les poids dérivent avec les cours (pas de
  rééquilibrage quotidien). Valorisation quotidienne aux cours d'ouverture.
- Mesures en plus : nombre moyen / max de positions, part des jours en cash, rotation annuelle.

    PYTHONPATH=.:myfiles python myfiles/myqtm_topk_dynamic.py [--no-ab] [--severe] [--first=2024-01-01] [--head-only]
"""
import sys

import numpy as np
import pandas as pd

import myqtm_close_threshold as B
from myqtm_topk_slots import TOPS, signals, stats
from src.portefolio_strategy.myqtm import config as C
from src.portefolio_strategy.myqtm import features as F
from src.portefolio_strategy.myqtm import train as T
from src.portefolio_strategy.myqtm.trade import ema_exit


def simulate(cand, opens, fee=C.FEE):
    """→ (valeur quotidienne, nb de positions par jour, rotation annuelle)."""
    dates, syms = opens.index, list(opens.columns)
    px = opens.ffill().to_numpy()
    held = np.zeros(px.shape, dtype=bool)        # held[t, s] : position tenue de l'ouverture t à l'ouverture t+1
    pos, col = {d: i for i, d in enumerate(dates)}, {s: i for i, s in enumerate(syms)}
    for r in cand.itertuples():
        held[pos[r.entry]:pos[r.exit], col[r.symbol]] = True
    n = held.sum(axis=1)
    v, w = np.ones(len(dates)), np.zeros(len(syms))
    turnover = 0.0
    for t in range(len(dates) - 1):
        target = held[t] / n[t] if n[t] else np.zeros(len(syms))
        if not np.array_equal(target > 0, w > 0):             # entrée ou sortie → rééquilibrage 1/n
            tr = np.abs(target - w).sum()
            v[t] *= 1 - fee * tr
            turnover += tr
            w = target
        g = np.where(w > 0, px[t + 1] / px[t], 1.0)
        g = np.nan_to_num(g, nan=1.0)
        gross = (w * g).sum() + (1 - w.sum())
        v[t + 1] = v[t] * gross
        w = w * g / gross
    years = (dates[-1] - dates[0]).days / 365.25
    return pd.Series(v, index=dates), n, turnover / years


def main():
    noab, severe = "--no-ab" in sys.argv, "--severe" in sys.argv
    first = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--first=")), None)
    head = "--head-only" in sys.argv
    if noab or severe or first or head:
        T.PREDICTIONS = T.predictions_path("all", ab=not noab, severe=severe, first=first, head=head)
    p = B.load()
    p["x_ema"] = p.groupby("symbol")["c"].transform(ema_exit).astype(bool)
    allf = pd.read_parquet(F.FEATURES, columns=["date", "symbol", "o"])
    opens = allf.pivot(index="date", columns="symbol", values="o").sort_index()
    nd = pd.Series(opens.index[1:].append(pd.DatetimeIndex([pd.NaT])), index=opens.index)
    p["next_date"] = p["date"].map(nd)
    start, end = p["date"].min(), p["date"].max()
    opens = opens.loc[start:end]
    rows = []
    for k in TOPS:
        cand = signals(p, k)
        cand = cand[cand["exit"] <= end]
        v, n, rot = simulate(cand, opens)
        rows.append(dict(regle=f"top {k} du jour", trades=len(cand), pos_moy=n.mean(), pos_max=int(n.max()),
                         jours_cash_pct=100 * (n == 0).mean(), rotation_an_x=rot, **stats(v)))
    idx = pd.read_parquet(F.INDEX)["level"].loc[start:end]
    rows.append(dict(regle="indice de l'univers", **stats(idx)))
    out = pd.DataFrame(rows)
    name = "myfiles/myqtm_topk_dynamic" + ("_noab" if noab else "") + ("_severe" if severe else "") + ("_head" if head else "") + T._first_sfx(first)
    out.to_csv(name + ".csv", index=False)
    pd.set_option("display.width", 200, "display.float_format", "{:.2f}".format)
    print(f"{'sans blocs A/B — ' if noab else ''}{'filtrage A/B sévère — ' if severe else ''}"
          f"{f'walk-forward dès {first} — ' if first else ''}lots dynamiques 1/n, tous les trades, "
          f"{start:%Y-%m-%d} → {end:%Y-%m-%d}, frais {C.FEE:.2%} par sens")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
