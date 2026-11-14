"""Test : le score de RegimeRank (corrélation de rang 0,10 sur le Sharpe relatif à 1 an des secteurs du S&P 500)
s'explique-t-il par la seule persistance de la volatilité ?

Chaque semaine (tous les 5 jours de bourse), pour les 11 secteurs du S&P 500 reconstitué (paniers pondérés par la
capitalisation, rendements quotidiens de clôture à clôture, 2014 → 2026), on calcule des règles d'une ligne sur le
passé (fenêtre de 252 jours, ou 63 jours) et on mesure leur corrélation de rang avec l'avenir :
- cible principale, comme dans l'article : Sharpe relatif = moyenne / écart-type des rendements quotidiens
  « secteur − indice » sur les H jours suivants ;
- décomposition : rendement relatif cumulé sur H jours, et volatilité relative sur H jours.
Aucune règle n'est ajustée sur les données (signe fixé a priori), donc pas de fuite possible. Corrélation moyenne sur
toutes les semaines ; t calculé sur le nombre de périodes indépendantes (semaines × 5 / H).

Sortie : outputs/portefolio_strategy/sp500/sector_vol_test.md

    python -m src.portefolio_strategy.sector_vol_test
"""
import numpy as np
import pandas as pd

from . import config as C
from . import rotation_compare as RC
from . import sector_oracle as SO
from . import sector_rank_path as RP
from . import sp500 as SP

OUT = C.OUT / "sp500"
STEP = 5


def daily_active():
    data = RC.datasets()
    d, maps, _ = data["S&P 500 (reconstitué)"]
    caps = SO.monthly_caps(SP.CAPS)
    start = "2013-12-23"
    Rw, _, Ww = SO.sector_returns(d, maps["secteur"], caps, start, 5)
    R = RP.daily_sector_returns(d, maps["secteur"], caps, start, list(Rw.columns)).dropna(how="all")
    W = Ww.reindex(R.index, method="ffill").bfill()
    idx = (R * W[R.columns]).sum(axis=1)
    return R, R.sub(idx, axis=0)


def signals(R, A, win):
    """Règles calculées sur les `win` derniers jours (connues à la clôture du jour)."""
    neg = A.where(A < 0, 0.0)
    pos = A.where(A > 0, 0.0)
    return {"− volatilité totale du secteur": -R.rolling(win).std(),
            "− volatilité relative (secteur − indice)": -A.rolling(win).std(),
            "− semi-volatilité des baisses relatives": -np.sqrt((neg ** 2).rolling(win).mean()),
            "+ semi-volatilité des hausses relatives": np.sqrt((pos ** 2).rolling(win).mean()),
            "+ rendement relatif passé (momentum)": A.rolling(win).sum(),
            "− rendement relatif passé (retour à la moyenne)": -A.rolling(win).sum(),
            "+ Sharpe relatif passé": A.rolling(win).mean() / A.rolling(win).std()}


def targets(A, h):
    m = A.rolling(h).mean().shift(-h)
    s = A.rolling(h).std().shift(-h)
    return {"Sharpe relatif futur": m / s, "rendement relatif futur": A.rolling(h).sum().shift(-h),
            "volatilité relative future (signe −)": -s}


def ic(sig, tgt, dates):
    x, y = sig.loc[dates], tgt.loc[dates]
    ok = x.notna().sum(axis=1).ge(8) & y.notna().sum(axis=1).ge(8)
    return x[ok].rank(axis=1).corrwith(y[ok].rank(axis=1), axis=1).dropna()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    R, A = daily_active()
    md = ["# La persistance de la volatilité explique-t-elle le score de RegimeRank ?", "",
          f"S&P 500 reconstitué, 11 secteurs pondérés par la capitalisation, {R.index[0]:%Y-%m-%d} → {R.index[-1]:%Y-%m-%d}. "
          "Corrélation de rang moyenne entre une règle d'une ligne (sur le passé) et l'avenir des secteurs, décision chaque "
          "semaine. Repères de l'article (2002-2024, cible = Sharpe relatif à 52 semaines) : RegimeRank 0,100 ; XGBoost "
          "0,039 ; momentum 12 mois −0,050 ; hasard 0,010.", ""]
    for h, win in ((252, 252), (126, 126), (63, 63)):
        T = targets(A, h)
        S = signals(R, A, win)
        dates = R.index[win::STEP]
        dates = dates[dates <= R.index[-h - 1]]
        n_ind = len(dates) * STEP / h
        rows = {}
        for sn, sv in S.items():
            row = {}
            for tn, tv in T.items():
                c = ic(sv, tv, dates)
                row[tn] = f"{c.mean():+.3f} (t {c.mean() / c.std() * np.sqrt(n_ind):+.1f})"
            rows[sn] = row
        md += [f"## Horizon {h} jours ({h // 21} mois), règles sur les {win} jours passés — {len(dates)} semaines, "
               f"≈ {n_ind:.0f} périodes indépendantes", "", pd.DataFrame(rows).T.to_markdown(), ""]
    (OUT / "sector_vol_test.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
