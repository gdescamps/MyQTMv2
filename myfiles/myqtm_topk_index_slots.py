"""5 lots investis par défaut dans l'indice de l'univers ; un signal « top k du jour » remplace l'indice d'un lot.

- Capital partagé en N_SLOTS = 5 lots égaux, tous achetés en indice au départ (frais d'achat).
- Signal à la clôture de J (titre entrant dans le top k des P(haussier)) : un lot en indice vend l'indice et
  achète le titre à l'ouverture de J+1 (frais sur la vente de l'indice et sur l'achat du titre). Signaux du
  même jour : la plus forte P(haussier) d'abord ; si les 5 lots portent déjà un titre, le signal est perdu.
- Sortie EMA 3 < EMA 10 à la clôture → vente du titre à l'ouverture suivante et rachat de l'indice (frais).
- Frais 0,05 % par sens. Indice = indice de l'univers pondéré par les capitalisations ; il est exécuté au
  niveau de la clôture de la veille (proxy de l'ouverture), les titres aux cours d'ouverture.
- Mesures : rendement annualisé, max drawdown, Sharpe (quotidien, taux sans risque 0, √252).

    PYTHONPATH=.:myfiles python myfiles/myqtm_topk_index_slots.py           # modèles A/B
    PYTHONPATH=.:myfiles python myfiles/myqtm_topk_index_slots.py --no-ab   # un modèle par coupe, sans A/B
"""
import sys

import numpy as np
import pandas as pd

import myqtm_close_threshold as B
from myqtm_topk_slots import signals, stats
from src.portefolio_strategy.myqtm import config as C
from src.portefolio_strategy.myqtm import features as F
from src.portefolio_strategy.myqtm import train as T
from src.portefolio_strategy.myqtm.trade import ema_exit

TOPS = (1, 2, 3, 4, 5)
N_SLOTS = 5


def simulate(cand, opens, idx, n_slots=N_SLOTS, fee=C.FEE):
    dates = opens.index
    free_at = [pd.Timestamp.min] * n_slots
    taken = []
    for r in cand.sort_values(["signal", "prob"], ascending=[True, False]).itertuples():
        for s in range(n_slots):
            if r.signal >= free_at[s]:
                taken.append((s, r.entry, r.exit, r.symbol))
                free_at[s] = r.exit_signal
                break
    pos = {d: i for i, d in enumerate(dates)}
    lvl = idx.to_numpy()
    val = np.zeros((len(dates), n_slots))
    in_stock = np.zeros(len(dates))
    for s in range(n_slots):
        trs = sorted([t for t in taken if t[0] == s], key=lambda t: t[1])
        units, last = (1.0 / n_slots) * (1 - fee) / lvl[0], 0       # achat initial de l'indice
        for _, e, x, sym in trs:
            ie, ix = pos[e], pos[x]
            val[last:ie, s] = units * lvl[last:ie]
            cap = units * lvl[ie] * (1 - fee)                          # vente de l'indice
            px = pd.Series(opens[sym].to_numpy()[ie:ix + 1]).ffill().to_numpy()
            path = cap * (1 - fee) * px / px[0]                        # achat du titre
            val[ie:ix, s] = path[:-1]
            cap = path[-1] * (1 - fee)                                 # vente du titre
            units = cap * (1 - fee) / lvl[ix]                          # rachat de l'indice
            val[ix, s] = units * lvl[ix]
            in_stock[ie:ix] += 1
            last = ix + 1
        val[last:, s] = units * lvl[last:]
    return pd.Series(val.sum(axis=1), index=dates), len(taken), in_stock.mean() / n_slots


def main():
    noab = "--no-ab" in sys.argv
    if noab:
        T.PREDICTIONS = T.predictions_path("all", ab=False)
    p = B.load()
    p["x_ema"] = p.groupby("symbol")["c"].transform(ema_exit).astype(bool)
    allf = pd.read_parquet(F.FEATURES, columns=["date", "symbol", "o"])
    opens = allf.pivot(index="date", columns="symbol", values="o").sort_index()
    nd = pd.Series(opens.index[1:].append(pd.DatetimeIndex([pd.NaT])), index=opens.index)
    p["next_date"] = p["date"].map(nd)
    start, end = p["date"].min(), p["date"].max()
    level = pd.read_parquet(F.INDEX)["level"].reindex(opens.index).ffill()
    idx_open = level.shift(1).loc[start:end]                         # clôture de la veille ≈ ouverture
    opens = opens.loc[start:end]
    rows = []
    for k in TOPS:
        cand = signals(p, k)
        cand = cand[cand["exit"] <= end]
        v, n_taken, occ = simulate(cand, opens, idx_open)
        rows.append(dict(regle=f"top {k} du jour", signaux=len(cand), trades_pris=n_taken,
                         lots_en_titres_pct=100 * occ, **stats(v)))
    rows.append(dict(regle="indice de l'univers seul", lots_en_titres_pct=0.0, **stats(idx_open)))
    out = pd.DataFrame(rows)
    out.to_csv(f"myfiles/myqtm_topk_index_slots{'_noab' if noab else ''}.csv", index=False)
    pd.set_option("display.width", 200, "display.float_format", "{:.2f}".format)
    print(f"{'sans blocs A/B — ' if noab else ''}{N_SLOTS} lots en indice par défaut, {start:%Y-%m-%d} → {end:%Y-%m-%d}, frais {C.FEE:.2%} par sens")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
