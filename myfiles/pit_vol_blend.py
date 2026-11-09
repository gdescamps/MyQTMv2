"""Vol prédite par les news (PIT, MM20 → vol20, hors échantillon) injectée DANS la stratégie x1.7 PUST+LQQ.

La vol Yang-Zhang (YZ) pilote les deux bras de compute_allocation (cap vol au-dessus de la MA,
cible de vol en dessous). On la remplace par un mélange avec la vol prédite par les embeddings :
  lin w   : vol = w·YZ + (1−w)·news
  max     : vol = max(YZ, news)          (prudent : la news ne peut que réduire l'expo)
Jours sans prédiction (avant 2014, années de test pas encore calculées, news manquante) → YZ seule
(= stratégie de base, le fallback de prod). Prédiction de l'article du soir J utilisée pour l'alloc
de J, exécutée J+1 par simulate_net (pas de look-ahead). Frais 0.1 %/côté.
    python -m myfiles.pit_vol_blend
"""
import numpy as np
import pandas as pd

import src.risk_off_strategy.strategy as st
from src.risk_off_strategy.data import load_funding, load_macro, load_ohlc
from src.risk_off_strategy.strategy import E_MAX, compute_allocation, simulate_net

st.BUY_FEE = st.SELL_FEE = 0.001
P = "data/text/cnbc/pit_vintage/oos_predictions.parquet"
YZ = st.yang_zhang_vol


def win(eq, m):
    e = eq[m] / eq[m][0]
    cagr = e[-1] ** (252 / m.sum()) - 1
    dd = (e / np.maximum.accumulate(e) - 1).min()
    return 100 * cagr, 100 * dd, cagr / abs(dd)


def main():
    o = load_ohlc("QQQ", "2000-01-01", None)
    p, idx = o["close"], o.index
    nfci, cpi = load_macro(idx)
    fund = load_funding(idx)
    kw = dict(high=o["high"].values, low=o["low"].values, open_=o["open"].values)
    yz = YZ(o["open"].values, o["high"].values, o["low"].values, p.values)
    rows = []
    for xn in ("MM20", "jour"):
        g = pd.read_parquet(P).query("X == @xn and cible == 'vol20'").set_index("date").sort_index()
        news = (g.pred / 100).reindex(idx).ffill(limit=3).clip(lower=0.03).values   # %/an → fraction
        has = ~np.isnan(news)
        years = sorted(set(idx[has].year))
        variants = {"base YZ": yz}
        for w in (0.75, 0.5, 0.25, 0.0):
            variants[f"lin w={w}"] = np.where(has, w * yz + (1 - w) * np.nan_to_num(news), yz)
        variants["max(YZ,news)"] = np.where(has, np.maximum(yz, np.nan_to_num(news)), yz)
        for name, vol in variants.items():
            if xn == "jour" and name == "base YZ":
                continue
            st.yang_zhang_vol = lambda *a, _v=vol, **k: _v
            a = compute_allocation(p.values, nfci, cpi, **kw)
            st.yang_zhang_vol = YZ
            eq, fees, rev = simulate_net(p, a, leverage=2, funding=fund, e_max=E_MAX)
            r = {"X": xn if name != "base YZ" else "-", "variante": name}
            m = np.isin(idx.year, years)            # années couvertes par les news seulement
            c, d, k = win(eq, m) if False else (None, None, None)
            mt = idx >= pd.Timestamp(f"{years[0]}-01-01")
            c, d, k = win(eq, mt)
            r.update({"%/an": round(c, 1), "maxDD": round(d, 1), "Calmar": round(k, 2), "frais %/an": round(fees, 2),
                      "revis/an": round(rev, 1)})
            for y in years:                          # rendement par année de test couverte
                my = idx.year == y
                r[str(y)] = round(100 * (eq[my][-1] / eq[my][0] - 1), 1)
            rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv("myfiles/pit_vol_blend.csv", index=False)
    pd.set_option("display.width", 250)
    print(f"Fenêtre {years[0]} → fin ; années avec news : {years} (autres années = YZ seule)")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
