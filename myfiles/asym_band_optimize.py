"""
Optimisation de la bande de non-action ASYMETRIQUE du backtest net de frais.

Idee (prompts.txt) : les achats sont GRATUITS sur PEA, les ventes coutent 0.5%.
Donc on veut suivre la cible A LA HAUSSE par petits pas (beaucoup de niveaux
d'allocation, meilleur tracking de la montee), mais ne DE-lever que par grands pas
(peu de ventes -> moins de frais). => seuil d'achat buy_thr bas, seuil de vente
sell_thr haut.

Ce script balaie la grille (buy_thr, sell_thr) sur le backtest QQQ NET DE FRAIS
(simulate_net) et classe par Sharpe / Calmar. Pour x1 et x2.

    python -m myfiles.asym_band_optimize          # (depuis la racine, venv actif)
"""
import numpy as np
import pandas as pd

from src.risk_off_strategy.data import load_ohlc, load_macro, load_funding
from src.risk_off_strategy.strategy import (
    compute_allocation, yang_zhang_vol, simulate_net, ANN,
    BUY_THR_ALLOC, SELL_THR_ALLOC,
)

START = "2000-01-01"


def metrics(eq):
    """CAGR / maxDD / Sharpe / Calmar depuis une equity pleine."""
    yrs = len(eq) / ANN
    peak = np.maximum.accumulate(eq)
    dd = float(((eq - peak) / peak).min())
    cagr = eq[-1] ** (1 / yrs) - 1
    r = eq[1:] / eq[:-1] - 1
    sh = r.mean() / r.std() * np.sqrt(ANN) if r.std() > 0 else 0.0
    cal = cagr / abs(dd) if dd < 0 else np.inf
    return cagr, dd, sh, cal


def half_split_sharpe(price, alloc, leverage, funding, buy, sell):
    """Sharpe net sur chaque moitie de l'historique (robustesse)."""
    n = len(price)
    mid = n // 2
    out = []
    for sl in (slice(0, mid), slice(mid, n)):
        res = simulate_net(price.iloc[sl], alloc[sl], leverage=leverage,
                           funding=None if funding is None else funding[sl],
                           buy_thr=buy, sell_thr=sell)
        if res is None:
            return None, None
        eq = res[0]
        _, _, sh, _ = metrics(eq)
        out.append(sh)
    return out[0], out[1]


def main():
    ohlc = load_ohlc("QQQ", START, None)
    price = ohlc["close"]
    o, h, l = ohlc["open"].values, ohlc["high"].values, ohlc["low"].values
    nfci, cpi = load_macro(price.index)
    funding = load_funding(price.index)
    alloc = compute_allocation(price.values, nfci, cpi, high=h, low=l, open_=o)
    print(f"QQQ {price.index[0].date()} -> {price.index[-1].date()}  "
          f"({len(price)} j)  funding={'oui' if funding is not None else 'NON'}\n")

    buy_grid = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    sell_grid = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 1.00]

    for lev in (1, 2):
        if lev == 2 and funding is None:
            print("x2 : funding indispo -> saute\n")
            continue
        rows = []
        for buy in buy_grid:
            for sell in sell_grid:
                res = simulate_net(price, alloc, leverage=lev, funding=funding,
                                   buy_thr=buy * lev, sell_thr=sell * lev)
                if res is None:
                    continue
                eq, fees, rev = res
                cagr, dd, sh, cal = metrics(eq)
                sh1, sh2 = half_split_sharpe(price, alloc, lev, funding,
                                             buy * lev, sell * lev)
                rows.append(dict(buy=buy, sell=sell, cagr=cagr, dd=dd, sh=sh,
                                 cal=cal, fees=fees, rev=rev,
                                 sh_min=min(sh1, sh2) if sh1 is not None else sh))
        df = pd.DataFrame(rows)

        # score = Sharpe + Calmar normalises (les deux comptent, cf. demande)
        df["score"] = df["sh"] / df["sh"].max() + df["cal"] / df["cal"].max()

        print(f"{'='*88}\nx{lev}  — top 12 par SCORE (Sharpe+Calmar net), colonne sh_min = pire moitie")
        print(f"{'='*88}")
        hdr = f"{'buy':>5} {'sell':>5} | {'CAGR':>7} {'maxDD':>7} {'Sharpe':>7} {'Calmar':>7} {'frais/an':>8} {'rev/an':>7} {'sh_min':>7} {'score':>6}"
        print(hdr)
        top = df.sort_values("score", ascending=False).head(12)
        for _, r in top.iterrows():
            print(f"{r.buy:5.2f} {r.sell:5.2f} | {r.cagr*100:6.1f}% {r.dd*100:6.1f}% "
                  f"{r.sh:7.3f} {r.cal:7.3f} {r.fees:7.2f}% {r.rev:6.0f} {r.sh_min:7.3f} {r.score:6.3f}")

        # references : bande symetrique (ancien defaut 0.20/0.20) et defaut courant
        for label, buy, sell in (("SYMETRIQUE 0.20/0.20", 0.20, 0.20),
                                  (f"defaut courant {BUY_THR_ALLOC}/{SELL_THR_ALLOC}",
                                   BUY_THR_ALLOC, SELL_THR_ALLOC)):
            res = simulate_net(price, alloc, leverage=lev, funding=funding,
                               buy_thr=buy * lev, sell_thr=sell * lev)
            eq, fees, rev = res
            cagr, dd, sh, cal = metrics(eq)
            print(f"  [ref {label:26}] CAGR {cagr*100:5.1f}%  maxDD {dd*100:5.1f}%  "
                  f"Sharpe {sh:.3f}  Calmar {cal:.3f}  frais {fees:.2f}%/an  rev {rev:.0f}")
        print()


if __name__ == "__main__":
    main()
