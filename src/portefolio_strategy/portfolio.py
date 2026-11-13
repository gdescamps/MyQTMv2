"""Portefeuille long : les n_top titres les mieux classés, rééquilibré tous les `every` jours, à poids égaux
ou proportionnels à la capitalisation (`size`, comme l'indice auquel on se compare).

Le soir d'un jour de rééquilibrage J, on classe les titres éligibles (score non NaN = membres de l'univers) ;
les ordres sont exécutés à l'ouverture de J+1. Un titre déjà détenu est gardé tant que son rang reste
<= n_top × buffer (buffer > 1 : moins de rotation). Entre deux rééquilibrages les poids dérivent ; un titre
qui sort de l'univers ou n'est plus coté reste valorisé à son dernier cours jusqu'au rééquilibrage suivant.
Frais : `fee` × montant échangé (achats + ventes) à chaque rééquilibrage.
"""
import numpy as np
import pandas as pd

from . import config as C


def select(score, held, n_top, buffer=1.0):
    """score : vecteur (NaN = non éligible) ; held : booléens. → indices retenus (au plus n_top)."""
    elig = np.isfinite(score)
    n = int(elig.sum()) if n_top is None else min(n_top, int(elig.sum()))
    if n == 0:
        return np.array([], dtype=int)
    order = np.argsort(-np.where(elig, score, -np.inf), kind="stable")
    rank = np.empty(len(score), dtype=int)
    rank[order] = np.arange(len(score))
    keep = [i for i in order if held[i] and elig[i] and rank[i] < n * buffer][:n]
    kept = set(keep)
    return np.array(keep + [i for i in order[:int(elig.sum())] if i not in kept][:n - len(keep)], dtype=int)


def simulate(scores, opens, n_top, every, buffer=1.0, fee=C.FEE, offset=0, record=False, size=None, cash_if_short=False):
    """scores, opens : DataFrames dates × titres (mêmes dates) ; size : poids relatifs (capitalisation) ou None
    pour des poids égaux ; cash_if_short : s'il y a moins de n_top titres éligibles, chaque ligne garde 1/n_top
    et le reste est en liquidités (règles avec filtre). → dict(values (base 1), turnover, n, weights)."""
    cols = scores.columns
    S = scores.to_numpy(float)
    P = opens.reindex(index=scores.index, columns=cols).ffill().to_numpy(float)
    Z = None if size is None else size.reindex(index=scores.index, columns=cols).ffill().to_numpy(float)
    n, k = S.shape
    h, cash, target = np.zeros(k), 1.0, None
    values, turnover, count, hist = np.empty(n), np.zeros(n), np.zeros(n, dtype=int), []
    for i in range(n):
        if i:
            r = P[i] / P[i - 1]
            h *= np.where(np.isfinite(r), r, 1.0)
        if target is not None:                        # ordres décidés la veille, exécutés à l'ouverture
            v = cash + h.sum()
            traded = np.abs(target * v - h).sum()
            v_net = v - fee * traded
            h = target * v_net
            cash = v_net - h.sum()
            turnover[i] = traded / v
            if record:
                hist.append(pd.Series(target[target > 0], index=cols[target > 0], name=scores.index[i]))
            target = None
        values[i] = cash + h.sum()
        count[i] = int((h > 0).sum())
        if i >= offset and (i - offset) % every == 0:
            sel = select(S[i], h > 0, n_top, buffer)
            if len(sel) or cash_if_short:
                w = np.ones(len(sel)) if Z is None else Z[i, sel]
                w = np.where(np.isfinite(w) & (w > 0), w, np.nanmin(w) if np.isfinite(w).any() else 1.0)
                target = np.zeros(k)
                target[sel] = w / w.sum() * (len(sel) / n_top if cash_if_short and n_top else 1.0)
    return {"values": pd.Series(values, index=scores.index), "turnover": pd.Series(turnover, index=scores.index),
            "n": pd.Series(count, index=scores.index), "weights": hist}


def simulate_offsets(scores, opens, n_top, every, buffer=1.0, fee=C.FEE, max_offsets=C.MAX_OFFSETS, size=None,
                     cash_if_short=False):
    """Moyenne de portefeuilles identiques démarrés à des jours décalés (réduit la chance du calendrier).
    → (valeur moyenne, rotation annuelle moyenne en multiples du capital)."""
    offs = sorted({int(round(j * every / min(every, max_offsets))) for j in range(min(every, max_offsets))})
    runs = [simulate(scores, opens, n_top, every, buffer, fee, o, size=size, cash_if_short=cash_if_short) for o in offs]
    values = pd.concat([r["values"] for r in runs], axis=1).mean(axis=1)
    years = max((scores.index[-1] - scores.index[0]).days / 365.25, 1e-9)
    return values, float(np.mean([r["turnover"].sum() for r in runs]) / years)


def metrics(values):
    """values : Series de valeur du portefeuille indexée par date."""
    v, dates = values.to_numpy(float), values.index
    peak = np.maximum.accumulate(v)
    dd = (v - peak) / peak
    years = max((dates[-1] - dates[0]).days / 365.25, 1e-9)
    r = np.diff(v) / v[:-1]
    yearly = values.groupby(dates.year).agg(lambda x: 100 * (x.iloc[-1] / x.iloc[0] - 1))
    return {"return_pct": float(100 * (v[-1] / v[0] - 1)), "cagr_pct": float(100 * ((v[-1] / v[0]) ** (1 / years) - 1)),
            "max_drawdown_pct": float(100 * dd.min()), "vol_pct": float(100 * r.std() * np.sqrt(252)),
            "sharpe": float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0.0,
            "annual_roi_pct": {int(k): round(float(x), 2) for k, x in yearly.items()}}


def active(values, bench):
    """Performance relative à l'indice : écart de CAGR, ratio d'information, part des années gagnantes."""
    a = values.pct_change().dropna() - bench.pct_change().dropna()
    m, mb = metrics(values), metrics(bench)
    ys, yb = m["annual_roi_pct"], mb["annual_roi_pct"]
    return {"excess_cagr_pct": m["cagr_pct"] - mb["cagr_pct"],
            "info_ratio": float(a.mean() / a.std() * np.sqrt(252)) if a.std() > 0 else 0.0,
            "tracking_error_pct": float(100 * a.std() * np.sqrt(252)),
            "years_beating_index": f"{sum(ys[y] > yb[y] for y in ys)}/{len(ys)}"}
