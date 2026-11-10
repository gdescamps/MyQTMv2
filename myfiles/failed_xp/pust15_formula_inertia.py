"""Remplacer la bande d'exécution (achat 0.50 / vente 1.00) par de l'inertie DANS la formule.
Base : E_MAX 1.5 + saturation. Exécution des variantes avec une bande quasi nulle (0.10/0.10 ou 0.20/0.20).
Familles :
  A. vol lissée   : rv = EMA(YZ-vol10, demi-vie h) -> les deux bras (cap 0.15/vol et 0.08/vol) bougent moins
  B. vol longue   : Yang-Zhang sur fenêtre N (20/40/60) au lieu de 10
  C. alloc lissée : EMA asymétrique de l'allocation (demi-vie montée hu / descente hd)
  D. paliers      : exposition quantifiée en paliers (pas d) avec hystérésis h (seuils dans la formule)
Frais 0.5 % et 0.1 %/côté.
    python -m myfiles.failed_xp.pust15_formula_inertia
"""
import numpy as np
import pandas as pd

import src.risk_off_strategy.strategy as st
from myfiles.failed_xp.pust17_emax_saturate import metr
from src.risk_off_strategy.data import load_funding, load_macro, load_ohlc

EM = 1.5


def yz(o, h, l, c, n):
    lo, lh, ll, lc = np.log(o), np.log(h), np.log(l), np.log(c)
    pc = np.concatenate([[lc[0]], lc[:-1]])
    ov, oc = lo - pc, lc - lo
    rs = (lh - lc) * (lh - lo) + (ll - lc) * (ll - lo)
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    v = (pd.Series(ov).rolling(n, min_periods=2).var() + k * pd.Series(oc).rolling(n, min_periods=2).var()
         + (1 - k) * pd.Series(rs).rolling(n, min_periods=2).mean())
    return np.sqrt(np.maximum(v.values, 1e-12) * 252)


def alloc_from_rv(p, rv, nf, cp):
    c = st.allocation_components(p, rv)
    a = np.clip(np.where(c["arm_above"], c["above"], c["below"]), 0, 1)
    a = np.where(np.nan_to_num(np.asarray(nf, float), nan=-9) > st.NFCI_OFF, 0.0, a)
    return np.where(np.nan_to_num(np.asarray(cp, float), nan=-9) > st.CPI_OFF, 0.0, a)


def ema_asym(a, hu, hd):
    au, ad = 1 - 0.5 ** (1 / hu), 1 - 0.5 ** (1 / hd)
    out = np.empty_like(a)
    s = 0.0
    for i, x in enumerate(a):
        if x == 0.0:
            s = 0.0                                   # garde-fou macro / sortie : immédiat
        else:
            s += (au if x > s else ad) * (x - s)
        out[i] = s
    return out


def steps(a, d, h):
    """Expo cible quantifiée par paliers de d (espace expo), avec hystérésis h : on ne change
    de palier que si l'expo brute dépasse le palier voisin de h·d au-delà du milieu."""
    E = np.minimum(2 * a, EM)
    out = np.empty_like(E)
    q = 0.0
    for i, e in enumerate(E):
        if e == 0.0:
            q = 0.0
        elif e >= EM - 1e-9:
            q = EM
        elif e > q + d * (0.5 + h) or e < q - d * (0.5 + h):
            q = min(EM, np.round(e / d) * d)
        out[i] = q
    return out / 2.0


def main():
    o = load_ohlc("QQQ", "2000-01-01", None)
    p, idx = o.close, o.index
    O, H, L, C = o.open.values, o.high.values, o.low.values, p.values
    nf, cp = load_macro(idx)
    fund = np.nan_to_num(np.asarray(load_funding(idx), float))
    base = st.compute_allocation(C, nf, cp, high=H, low=L, open_=O)
    rv10 = st.yang_zhang_vol(O, H, L, C)
    assert np.allclose(alloc_from_rv(C, rv10, nf, cp), base)

    sigs = {"ACTUELLE (formule brute)": base}
    for h in (5, 10, 20, 40):
        sigs[f"A vol EMA demi-vie {h}j"] = alloc_from_rv(C, pd.Series(rv10).ewm(halflife=h).mean().values, nf, cp)
    for n in (20, 40, 60):
        sigs[f"B vol YZ {n}j"] = alloc_from_rv(C, yz(O, H, L, C, n), nf, cp)
    for hu, hd in ((5, 5), (10, 10), (20, 20), (5, 20), (10, 40), (20, 5)):
        sigs[f"C alloc EMA montée {hu}j / descente {hd}j"] = ema_asym(base, hu, hd)
    for d, h in ((0.25, 0.5), (0.5, 0.5), (0.5, 1.0), (0.3, 1.0)):
        sigs[f"D paliers {d} hyst {h}"] = steps(base, d, h)

    hz = {"tout": ("2000-01-01", None), "H1": ("2000-01-01", "2012-12-31"), "H2": ("2013-01-01", None),
          "10ans": (str(idx[-2520].date()), None), "5ans": (str(idx[-1260].date()), None)}
    rows = []
    for fee in (0.005, 0.001):
        st.BUY_FEE = st.SELL_FEE = fee
        for name, al in sigs.items():
            bands = [("0.50/1.00", 0.50, 1.00), ("0.10/0.10", 0.10, 0.10), ("0.20/0.20", 0.20, 0.20)]
            for bn, b, s in bands:
                eq, fz, rv = st.simulate_net(p, al, leverage=2, funding=fund, e_max=EM, sat_gap=st.SAT_GAP_E, buy_thr=b, sell_thr=s)
                r = {"frais": f"{fee*100:.1f}%", "formule": name, "bande": bn, "frais/an": round(fz, 2), "ordres/an": round(rv, 1)}
                for k_, (x, y) in hz.items():
                    cg, dd, sh, cal = metr(eq, idx, x, y)
                    r.update({f"{k_} %/an": round(cg, 1), f"{k_} DD": round(dd, 1), f"{k_} Sh": round(sh, 2), f"{k_} Cal": round(cal, 2)})
                rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv("myfiles/failed_xp/pust15_formula_inertia.csv", index=False)
    pd.set_option("display.width", 320)
    pd.set_option("display.max_rows", 500)
    cols = ["formule", "bande", "frais/an", "ordres/an", "tout %/an", "tout DD", "tout Sh", "tout Cal", "H1 Cal", "H2 Cal",
            "10ans %/an", "10ans Cal", "5ans %/an", "5ans DD"]
    for fee in df.frais.unique():
        print(f"\n=== frais {fee} ===")
        print(df[df.frais == fee][cols].to_string(index=False))


if __name__ == "__main__":
    main()
