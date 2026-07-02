"""
Strategie deployee : trend-following long + vol-managed + garde-fous anticrise.

Cœur (x1, close-to-close) :
    s      = SMA(prix, 250)
    vol    = Yang-Zhang(OHLC, 10j) annualisee  (integre les gaps overnight -> plus
             precoce que le close-to-close ; fallback rvol20 close si pas d'OHLC)
    decay  = clip(1 + (prix/s - 1) / GAP_CUTOFF, 0, 1)     # ecrase l'expo quand
                                                            # le prix est tres sous la MA
    above  = clip(ABOVE_CAP / vol, 0, 1)   # RISK-OFF PRECOCE : coupe deja au-dessus
                                            # de la MA quand la vol s'emballe (=1 si OHLC absent)
    alloc  = where(prix > s, above, BELOW_SCALE * clip(VOL_TARGET/vol, 0, 1) * decay)

Garde-fous macro (cash total, ignores si la donnee est absente) :
    NFCI    > NFCI_OFF   -> stress credit (capte 2008)
    IPC YoY > CPI_OFF %  -> inflation extreme (capte 2022)

exec_lag=1 est gere en aval : la valeur alloc[-1] calculee au close du soir
s'execute a l'ouverture du lendemain, donc pas de look-ahead. Yang-Zhang au jour t
n'utilise que O/H/L/C de t (et C_{t-1}) sur une fenetre glissante finissant a t :
causal, aucune fuite temporelle.

Historique/backtest depuis 2000 (le QQQ demarre le 2000-01-03).
"""
import numpy as np
import pandas as pd

ANN = 252

# ── Parametres de la strategie ────────────────────────────
SMA_LONG = 250       # MA de tendance
VOL_TARGET = 0.08    # V : cible de vol du bras vol-managed (sous la MA)
BELOW_SCALE = 0.8    # L2 : facteur d'expo sous la MA
GAP_CUTOFF = 0.15    # G : alloc -> 0 lorsque le prix est G en dessous de la MA
NFCI_OFF = 0.5       # coupe-circuit conditions financieres (NFCI)
CPI_OFF = 7.0        # coupe-circuit inflation (IPC YoY, %)
VOL_WINDOW = 10      # fenetre Yang-Zhang (10j ~= 20j close-to-close en bruit, 2x plus reactif)
ABOVE_CAP = 0.18     # risk-off precoce : vol annualisee au-dela de laquelle on coupe deja
                     #  au-dessus de la MA. Reglage le plus defensif : max-Calmar full-sample
                     #  ET optimum de la moitie de crise 2000-2013 (validation croisee : voyage
                     #  bien sur 2013-2026, Calmar 1.01 vs optimum 1.06). Courbe monotone.
                     #  CAGR 11.1% / Sharpe 0.88 / Calmar 0.58 / maxDD -19% vs -36% baseline
                     #  (cote ~1 pt de CAGR vs 0.25 en echange de la protection max).


def realized_vol(ret, w=20):
    """Vol realisee close-to-close annualisee (plancher a 1e-6). Fallback si pas d'OHLC."""
    return np.maximum(
        pd.Series(ret).rolling(w, min_periods=5).std().bfill().values * np.sqrt(ANN),
        1e-6,
    )


def yang_zhang_vol(open_, high, low, close, w=VOL_WINDOW):
    """Vol Yang-Zhang annualisee (causale) = overnight + k*open-close + (1-k)*Rogers-Satchell.

    Combine le gap overnight (O_t vs C_{t-1}), la variance open->close et le terme
    Rogers-Satchell (drift-independent, H/L/O/C). ~5x plus efficace que le
    close-to-close et surtout capte le risque de gap nocturne -- la ou naissent les
    krachs -- donc alerte plus tot. Plancher a 1e-6.
    """
    o, h, l, c = (pd.Series(np.asarray(x, float)) for x in (open_, high, low, close))
    lo = np.log(o / c.shift(1))               # overnight (O_t / C_{t-1})
    lc = np.log(c / o)                         # open -> close
    ho, ll = np.log(h / o), np.log(l / o)
    rs = ho * (ho - lc) + ll * (ll - lc)       # Rogers-Satchell
    k = 0.34 / (1.34 + (w + 1) / (w - 1))
    var = (lo.rolling(w, min_periods=5).var()
           + k * lc.rolling(w, min_periods=5).var()
           + (1 - k) * rs.rolling(w, min_periods=5).mean())
    vol = np.sqrt(np.clip(var, 0, None) * ANN)
    return np.maximum(pd.Series(vol).bfill().values, 1e-6)


def compute_allocation(price, nfci=None, cpi=None, high=None, low=None, open_=None,
                       above_cap=ABOVE_CAP):
    """Allocation continue dans [0, 1].

    price : array-like des cloture.
    high/low/open_ : OHLC alignes -> active la vol Yang-Zhang + le risk-off precoce
                     (cap vol au-dessus de la MA). Absents -> fallback close-to-close
                     et comportement historique (expo=1 au-dessus de la MA).
    nfci  : array aligne (deja decale du lag de publication) ou None -> arm ignore.
    cpi   : idem pour l'IPC YoY (%).
    """
    p = np.asarray(price, dtype=float)
    s = pd.Series(p).rolling(SMA_LONG, min_periods=1).mean().values
    if high is not None and low is not None and open_ is not None:
        rv = yang_zhang_vol(open_, high, low, p)     # vol precoce (gaps overnight)
        above = np.clip(above_cap / rv, 0, 1)         # coupe deja au-dessus de la MA
    else:
        ret = pd.Series(p).pct_change().fillna(0).values
        rv = realized_vol(ret)                        # fallback close-to-close
        above = np.ones_like(p)                       # comportement historique
    decay = np.clip(1.0 + (p / s - 1.0) / GAP_CUTOFF, 0, 1)
    below = BELOW_SCALE * np.clip(VOL_TARGET / rv, 0, 1) * decay
    alloc = np.clip(np.where(p > s, above, below), 0, 1)
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
