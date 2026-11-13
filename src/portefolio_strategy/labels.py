"""Cible du modèle : rendement futur réalisable, comparé aux autres membres de l'univers le même jour.

Le signal de J est connu à la clôture de J et exécuté à l'ouverture de J+1 :
    fwd_h(J) = ouverture(J+1+h) / ouverture(J+1) − 1
Un titre qui cesse d'être coté pendant la fenêtre garde son dernier cours (rendement figé, comme en portefeuille).
Le label y_h est le rang du jour de fwd_h parmi les membres, centré (−0,5 … +0,5). Il n'est connu qu'à
l'ouverture de J+1+h : l'entraînement à la date T n'utilise que les jours dont cette date est < T (label_known).
"""
import numpy as np
import pandas as pd


def open_matrix(df):
    """df (date, symbol, o) → cours d'ouverture ajustés, dates × titres."""
    return df.pivot(index="date", columns="symbol", values="o").sort_index()


def forward_returns(opens, h):
    """→ DataFrame dates × titres. NaN si J+1 n'est pas coté ou si J+1+h dépasse la fin des données."""
    last = opens.ffill(limit=h + 1)
    return last.shift(-1 - h) / opens.shift(-1) - 1


def add_labels(df, horizons):
    """Ajoute à df : pos (rang de la date dans le calendrier), fwd_h et y_h pour chaque horizon."""
    opens = open_matrix(df)
    pos = pd.Series(np.arange(len(opens)), index=opens.index)
    out = df.copy()
    out["pos"] = out["date"].map(pos).to_numpy()
    key = pd.MultiIndex.from_frame(out[["date", "symbol"]])
    for h in horizons:
        f = forward_returns(opens, h).stack(future_stack=True)
        out[f"fwd_{h}"] = f.reindex(key).to_numpy()
        ok = out["member"] & out[f"fwd_{h}"].notna()
        out[f"y_{h}"] = out.loc[ok].groupby("date")[f"fwd_{h}"].rank(pct=True) - 0.5
    return out


def label_known(pos, h, cutoff_pos):
    """Le label du jour de rang `pos` est-il connu avec les cours strictement antérieurs à la date de coupe ?"""
    return pos + 1 + h < cutoff_pos
