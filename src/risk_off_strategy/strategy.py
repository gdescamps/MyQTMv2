"""
Strategie deployee : trend-following long + vol-managed + garde-fous anticrise.

Cœur (x1, close-to-close) :
    s      = SMA(prix, 250)
    rvol20 = ecart-type 20j des rendements, annualise
    decay  = clip(1 + (prix/s - 1) / GAP_CUTOFF, 0, 1)     # ecrase l'expo quand
                                                            # le prix est tres sous la MA
    alloc  = where(prix > s, 1.0, BELOW_SCALE * clip(VOL_TARGET/rvol20, 0, 1) * decay)

Garde-fous macro (cash total, ignores si la donnee est absente) :
    NFCI    > NFCI_OFF   -> stress credit (capte 2008)
    IPC YoY > CPI_OFF %  -> inflation extreme (capte 2022)

exec_lag=1 est gere en aval : la valeur alloc[-1] calculee au close du soir
s'execute a l'ouverture du lendemain, donc pas de look-ahead.

Historique/backtest depuis 2000 (le QQQ demarre le 2000-01-03).
"""
import numpy as np
import pandas as pd

ANN = 252

# ── Parametres de la strategie ────────────────────────────
SMA_LONG = 250       # MA de tendance
VOL_TARGET = 0.08    # V : cible de vol du bras vol-managed
BELOW_SCALE = 0.8    # L2 : facteur d'expo sous la MA
GAP_CUTOFF = 0.15    # G : alloc -> 0 lorsque le prix est G en dessous de la MA
NFCI_OFF = 0.5       # coupe-circuit conditions financieres (NFCI)
CPI_OFF = 7.0        # coupe-circuit inflation (IPC YoY, %)


def realized_vol(ret, w=20):
    """Vol realisee annualisee (plancher a 1e-6 pour eviter les divisions)."""
    return np.maximum(
        pd.Series(ret).rolling(w, min_periods=5).std().bfill().values * np.sqrt(ANN),
        1e-6,
    )


def compute_allocation(price, nfci=None, cpi=None):
    """Allocation continue dans [0, 1] pour une serie de prix (close).

    price : array-like des cloture.
    nfci  : array aligne (deja decale du lag de publication) ou None -> arm ignore.
    cpi   : idem pour l'IPC YoY (%).
    """
    p = np.asarray(price, dtype=float)
    ret = pd.Series(p).pct_change().fillna(0).values
    s = pd.Series(p).rolling(SMA_LONG, min_periods=1).mean().values
    rv = realized_vol(ret)
    decay = np.clip(1.0 + (p / s - 1.0) / GAP_CUTOFF, 0, 1)
    alloc = np.clip(
        np.where(p > s, 1.0, BELOW_SCALE * np.clip(VOL_TARGET / rv, 0, 1) * decay),
        0, 1,
    )
    if nfci is not None:
        alloc = np.where(np.nan_to_num(np.asarray(nfci, float), nan=-9) > NFCI_OFF, 0.0, alloc)
    if cpi is not None:
        alloc = np.where(np.nan_to_num(np.asarray(cpi, float), nan=-9) > CPI_OFF, 0.0, alloc)
    return alloc


def simulate(price, alloc, exec_lag=1):
    """Backtest x1, close-to-close, sans frais. Renvoie un dict de metriques + equity.

    exec_lag=1 : la position du jour t est alloc[t-1] (decision au close t-1,
    executee au close t) -> aucun look-ahead.
    """
    p = np.asarray(price, dtype=float)
    ret = pd.Series(p).pct_change().fillna(0).values
    a = np.asarray(alloc, dtype=float)
    pos = np.zeros_like(a)
    pos[exec_lag:] = a[:-exec_lag] if exec_lag else a
    r = ret * pos
    eq = np.cumprod(1 + r)
    years = len(p) / ANN
    peak = np.maximum.accumulate(eq)
    dd = ((eq - peak) / peak).min()
    return {
        "equity": eq,
        "cagr": eq[-1] ** (1 / years) - 1,
        "sharpe": r.mean() / r.std() * np.sqrt(ANN) if r.std() > 0 else 0.0,
        "maxdd": float(dd),
        "calmar": (eq[-1] ** (1 / years) - 1) / abs(dd) if dd < 0 else np.inf,
        "time_in_market": float((pos > 0).mean()),
        "turnover": float(np.abs(np.diff(pos)).sum() / years),
    }
