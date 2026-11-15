"""Portefeuille à lots : chaque signal d'entrée « top k du jour » prend un lot libre, sortie EMA 3 < EMA 10.

- Nombre de lots = k (top k ↔ k lots) par défaut, ou N_SLOTS fixe avec --fixed.
- Capital initial partagé en lots égaux ; chaque lot porte une position à la fois et réinvestit son propre
  capital (pas de rééquilibrage entre lots) ; un lot libre reste en cash (rémunération 0).
- Signal à la clôture de J → achat à l'ouverture suivante du titre ; signal de sortie à la clôture → vente à
  l'ouverture suivante ; le lot est réutilisable pour un signal du jour du signal de sortie.
- Plusieurs signaux le même jour : la plus forte P(haussier) d'abord ; sans lot libre, le signal est perdu.
- Frais 0,05 % par sens. Valorisation quotidienne aux cours d'ouverture.
- Mesures : rendement annualisé, max drawdown, Sharpe (rendements quotidiens, taux sans risque 0, √252),
  trades pris / signaux, taux d'occupation des lots. Référence : indice de l'univers (pondéré par les
  capitalisations).

    PYTHONPATH=.:myfiles python myfiles/myqtm_topk_slots.py            # k lots pour le top k
    PYTHONPATH=.:myfiles python myfiles/myqtm_topk_slots.py --fixed    # N_SLOTS = 10 lots
"""
import sys

import numpy as np
import pandas as pd

import myqtm_close_threshold as B
from src.portefolio_strategy.myqtm import config as C
from src.portefolio_strategy.myqtm import features as F
from src.portefolio_strategy.myqtm.trade import ema_exit

TOPS = tuple(range(1, 16))
N_SLOTS = 10


def signals(p, k):
    """→ trades candidats (signal, entry_date, exit_signal, exit_date, symbol, prob)."""
    rank = p.groupby("date")["ybull"].rank(ascending=False, method="first")
    flag = rank <= k
    entry = (flag & ~flag.groupby(p["symbol"]).shift(1, fill_value=False)).to_numpy()
    ex, sym, dates = p["x_ema"].to_numpy(), p["symbol"].to_numpy(), p["date"].to_numpy()
    nxt = p["next_date"].to_numpy()
    rows, i, n = [], 0, len(p)
    while i < n:
        if not entry[i]:
            i += 1
            continue
        j = i + 1
        while j < n and sym[j] == sym[i] and not ex[j]:
            j += 1
        if j >= n or sym[j] != sym[i] or pd.isna(nxt[j]) or pd.isna(nxt[i]):
            i = j
            continue
        rows.append((dates[i], nxt[i], dates[j], nxt[j], sym[i], p["ybull"].iat[i]))
        i = j + 1
    return pd.DataFrame(rows, columns=["signal", "entry", "exit_signal", "exit", "symbol", "prob"])


def simulate(cand, opens, n_slots=N_SLOTS, fee=C.FEE):
    """Lots indépendants ; → (valeur quotidienne du portefeuille, trades pris, occupation moyenne)."""
    dates = opens.index
    free_at = [pd.Timestamp.min] * n_slots            # date du signal à partir duquel le lot est libre
    taken = []
    for r in cand.sort_values(["signal", "prob"], ascending=[True, False]).itertuples():
        for s in range(n_slots):
            if r.signal >= free_at[s]:
                taken.append((s, r.entry, r.exit, r.symbol))
                free_at[s] = r.exit_signal
                break
    slot_val = np.full((len(dates), n_slots), 1.0 / n_slots)
    busy = np.zeros(len(dates))
    pos = {d: i for i, d in enumerate(dates)}
    by_slot = {s: sorted([t for t in taken if t[0] == s], key=lambda t: t[1]) for s in range(n_slots)}
    for s, trs in by_slot.items():
        cap, last = 1.0 / n_slots, 0
        for _, e, x, sym in trs:
            ie, ix = pos[e], pos[x]
            slot_val[last:ie, s] = cap                 # cash entre deux trades
            px = opens[sym].to_numpy()
            size = cap * (1 - fee)
            path = size * px[ie:ix + 1] / px[ie]
            path = pd.Series(path).ffill().to_numpy()  # jour sans cours : dernier cours connu
            slot_val[ie:ix, s] = path[:-1]
            cap = path[-1] * (1 - fee)
            slot_val[ix, s] = cap
            busy[ie:ix] += 1
            last = ix + 1
        slot_val[last:, s] = cap
    return pd.Series(slot_val.sum(axis=1), index=dates), len(taken), busy.mean() / n_slots


def stats(v):
    r = v.pct_change().dropna()
    years = (v.index[-1] - v.index[0]).days / 365.25
    dd = v / v.cummax() - 1
    return dict(rendement_an_pct=100 * ((v.iloc[-1] / v.iloc[0]) ** (1 / years) - 1),
                max_drawdown_pct=100 * dd.min(), sharpe=r.mean() / r.std() * np.sqrt(252),
                total_x=v.iloc[-1] / v.iloc[0])


def main():
    p = B.load()
    p["x_ema"] = p.groupby("symbol")["c"].transform(ema_exit).astype(bool)
    allf = pd.read_parquet(F.FEATURES, columns=["date", "symbol", "o"])
    opens = allf.pivot(index="date", columns="symbol", values="o").sort_index()
    nd = pd.Series(opens.index[1:].append(pd.DatetimeIndex([pd.NaT])), index=opens.index)
    p["next_date"] = p["date"].map(nd)                  # prochaine séance (exécution à l'ouverture)
    start, end = p["date"].min(), p["date"].max()
    opens = opens.loc[start:end]
    fixed = "--fixed" in sys.argv
    rows = []
    for k in TOPS:
        cand = signals(p, k)
        cand = cand[cand["exit"] <= end]
        n_slots = N_SLOTS if fixed else k
        v, n_taken, occ = simulate(cand, opens, n_slots=n_slots)
        rows.append(dict(regle=f"top {k} du jour", lots=n_slots, signaux=len(cand), trades_pris=n_taken,
                         occupation_lots_pct=100 * occ, **stats(v)))
    idx = pd.read_parquet(F.INDEX)["level"].loc[start:end]
    rows.append(dict(regle="indice de l'univers", **stats(idx)))
    out = pd.DataFrame(rows)
    out.to_csv("myfiles/myqtm_topk_slots.csv" if fixed else "myfiles/myqtm_topk_slots_k.csv", index=False)
    pd.set_option("display.width", 200, "display.float_format", "{:.2f}".format)
    print(f"{'%d lots' % N_SLOTS if fixed else 'k lots pour le top k'}, {start:%Y-%m-%d} → {end:%Y-%m-%d}, frais {C.FEE:.2%} par sens")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
