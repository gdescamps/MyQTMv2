"""Ensemble de lookbacks (Levine & Pedersen 2016 ; Hurst, Ooi & Pedersen 2017) a la place de la SMA250 unique,
sur la strategie deployee x1.7 PUST+LQQ (frais 0.5 %/cote, sans saturation, macro NFCI/CPI inchangee).

  single_L : formule actuelle avec SMA_LONG = L (sensibilite au choix de 250)
  avg      : moyenne des allocations completes calculees avec chaque SMA_L de l'ensemble
             (= chaque membre garde son bras haut/bas, son decay_down et son filtre de pente)
  frac     : f = part des L ou le bras "au-dessus" est actif (prix > SMA_L ET SMA_L montante) ;
             alloc = f · cap_vol + (1 − f) · moyenne_L(bras bas_L)
Validation : une SMA unique "choisie" sur une moitie (Calmar) puis lue sur l'autre, vs ensembles non regles.
    python -W ignore -m myfiles.failed_xp.pust17_lookback_ensemble      (log : myfiles/failed_xp/pust17_lookback_ensemble.log)

Resultats (2026-11-09, ref SMA250 = 16.4 %/an, maxDD -29.5 %, Calmar 0.55, H1/H2 0.23/0.88) :
  - SMA unique 100-300 : Calmar 0.25 -> 0.55, mediane 0.45 ; 250 = le MAX (in-sample) -> 0.55 est optimiste.
    Choisie sur H1 -> L=225 -> H2 0.70 (vs 0.88 pour 250).
  - Meilleur ensemble au point deploye : avg 100-300 x9 = 15.9 % / -29.1 % / 0.55, H1/H2 0.27/0.84,
    frais 0.67 %/an (vs 0.89) — egale le meilleur L sans en choisir un. Mais l'ensemble x5 sur la meme plage
    = 0.47 : au point deploye, les ecarts de ~0.05-0.1 sont du bruit de discretisation de la bande.
  - Horizons courts (21/63, 50) : desastreux sur H1 (whipsaw 2000-2002, maxDD -40 a -55 %).
  - Robustesse (27 reglages voisins bande x E_MAX) : SMA250 mediane 0.44, min 0.29 ; ensembles 100-300 x9 /
    100-300 x5 / 200-300 : mediane 0.47/0.46/0.45, min 0.40/0.39/0.34 (150-300 x4 = au niveau de SMA250).
    Gain modeste (surtout sur le pire cas) : l'ensemble reduit la dependance au chemin, pas un gain de perf.
  - Lecon transversale : le Calmar 0.55 du point deploye est ~+0.1 au-dessus de la mediane de ses voisins ;
    des decisions passees fondees sur des ecarts < 0.05-0.1 de Calmar sont dans le bruit.
"""
import itertools

import numpy as np
import pandas as pd

import src.risk_off_strategy.strategy as st
from src.risk_off_strategy.data import load_funding, load_macro, load_ohlc
from src.risk_off_strategy.strategy import CPI_OFF, E_MAX, NFCI_OFF, simulate_net

HALVES = {"H1": ("2000-01-01", "2012-12-31"), "H2": ("2013-01-01", None), "tout": ("2000-01-01", None)}
SINGLES = [100, 125, 150, 175, 200, 225, 250, 275, 300]
ENSEMBLES = {
    "50-250":      [50, 100, 150, 200, 250],
    "100-250":     [100, 150, 200, 250],
    "100-300":     [100, 150, 200, 250, 300],
    "150-300":     [150, 200, 250, 300],
    "200-300":     [200, 250, 300],
    "125-275 x7":  [125, 150, 175, 200, 225, 250, 275],
    "100-300 x9":  [100, 125, 150, 175, 200, 225, 250, 275, 300],
    "21/63/252":   [21, 63, 252],             # horizons mensuel / trimestriel / annuel facon TSMOM
    "63/126/252":  [63, 126, 252],
}


def win(eq, idx, a, b):
    m = (idx >= pd.Timestamp(a)) & (idx <= (pd.Timestamp(b) if b else idx[-1]))
    e = eq[m] / eq[m][0]
    yrs = m.sum() / 252
    cagr = e[-1] ** (1 / yrs) - 1
    dd = (e / np.maximum.accumulate(e) - 1).min()
    return 100 * cagr, 100 * dd, cagr / abs(dd)


def components(p, rv, L):
    old = st.SMA_LONG
    st.SMA_LONG = L
    try:
        c = st.allocation_components(p, rv)
    finally:
        st.SMA_LONG = old
    return c


def main():
    o = load_ohlc("QQQ", "2000-01-01", None)
    p, idx = o["close"], o.index
    pv = p.values.astype(float)
    nfci, cpi = load_macro(idx)
    fund = load_funding(idx)
    rv = st.yang_zhang_vol(o["open"].values, o["high"].values, o["low"].values, pv)
    macro_off = (np.nan_to_num(nfci, nan=-9) > NFCI_OFF) | (np.nan_to_num(cpi, nan=-9) > CPI_OFF)
    lbs = sorted(set(SINGLES) | {L for v in ENSEMBLES.values() for L in v})
    comp = {L: components(pv, rv, L) for L in lbs}
    alloc_L = {L: np.clip(np.where(c["arm_above"], c["above"], c["below"]), 0, 1) for L, c in comp.items()}

    # garde-fou : la SMA250 unique doit redonner compute_allocation
    ref_alloc = st.compute_allocation(pv, nfci, cpi, high=o["high"].values, low=o["low"].values,
                                      open_=o["open"].values)
    assert np.allclose(np.where(macro_off, 0, alloc_L[250]), ref_alloc)

    rows = []

    def run(kind, name, a):
        a = np.where(macro_off, 0.0, a)
        eq, fees, rev = simulate_net(p, a, leverage=2, funding=fund, e_max=E_MAX)
        r = {"type": kind, "nom": name, "alloc moy": round(a.mean(), 2), "frais %/an": round(fees, 2),
             "revis/an": round(rev, 1)}
        for h, (x, y) in HALVES.items():
            c, d, cal = win(eq, idx, x, y)
            r.update({f"{h} %/an": round(c, 1), f"{h} maxDD": round(d, 1), f"{h} Calmar": round(cal, 2)})
        rows.append(r)

    for L in SINGLES:
        run("single", str(L), alloc_L[L])
    for name, Ls in ENSEMBLES.items():
        run("avg", name, np.mean([alloc_L[L] for L in Ls], axis=0))
        f = np.mean([comp[L]["arm_above"] for L in Ls], axis=0)
        below = np.mean([comp[L]["below"] for L in Ls], axis=0)
        run("frac", name, np.clip(f * comp[250]["above"] + (1 - f) * below, 0, 1))

    df = pd.DataFrame(rows)
    df.to_csv("myfiles/failed_xp/pust17_lookback_ensemble.csv", index=False)
    pd.set_option("display.width", 260)
    cols = ["type", "nom", "alloc moy", "H1 %/an", "H1 maxDD", "H1 Calmar", "H2 %/an", "H2 maxDD", "H2 Calmar",
            "tout %/an", "tout maxDD", "tout Calmar", "frais %/an", "revis/an"]
    print(df[cols].to_string(index=False))

    sg = df[df.type == "single"]
    print("\n=== Sensibilite SMA unique (100-300) ===")
    for h in HALVES:
        print(f"{h:5s} Calmar min/med/max {sg[f'{h} Calmar'].min():.2f}/{sg[f'{h} Calmar'].median():.2f}/"
              f"{sg[f'{h} Calmar'].max():.2f}  (250 : {sg[sg.nom == '250'][f'{h} Calmar'].iloc[0]:.2f})")
    print("\n=== SMA unique choisie sur une moitie, lue sur l'autre ===")
    for fit, test in (("H1", "H2"), ("H2", "H1")):
        b = sg.loc[sg[f"{fit} Calmar"].idxmax()]
        print(f"regle sur {fit} -> L={b.nom} : {test} Calmar {b[f'{test} Calmar']} "
              f"(%/an {b[f'{test} %/an']}, maxDD {b[f'{test} maxDD']})")

    # Robustesse au bruit de discretisation : 27 voisins du point deploye (bande achat 0.4/0.5/0.6 x vente
    # 0.8/1.0/1.2 x E_MAX 1.5/1.7/1.9) -> distribution du Calmar plutot qu'un point unique
    cands = {f"SMA{L}": alloc_L[L] for L in (200, 225, 250, 275)}
    for name in ("100-300 x9", "100-300", "150-300", "200-300"):
        cands[f"ens {name}"] = np.mean([alloc_L[L] for L in ENSEMBLES[name]], axis=0)
    rob = []
    for name, a in cands.items():
        a = np.where(macro_off, 0.0, a)
        for b, sl, e in itertools.product([0.4, 0.5, 0.6], [0.8, 1.0, 1.2], [1.5, 1.7, 1.9]):
            eq = simulate_net(p, a, leverage=2, funding=fund, e_max=e, buy_thr=b, sell_thr=sl)[0]
            rob.append([name] + [win(eq, idx, *HALVES[h])[2] for h in HALVES])
    rd = pd.DataFrame(rob, columns=["candidat", "H1", "H2", "tout"]).groupby("candidat")
    print("\n=== Robustesse : Calmar sur 27 reglages voisins (bande x E_MAX) ===")
    print(pd.concat({"median": rd.median(), "p25": rd.quantile(0.25), "min": rd.min()}, axis=1).round(2)
          .sort_values(("median", "tout"), ascending=False).to_string())


if __name__ == "__main__":
    main()
