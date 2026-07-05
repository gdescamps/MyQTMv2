"""
Chargement des donnees pour la strategie deployee (trend + vol-managed + macro).

- load_price  : cloture d'un ticker (QQQ par defaut), depuis 2000.
- load_macro  : NFCI et IPC YoY alignes sur les dates de prix, decales du lag de
                publication FRED (NFCI ~mardi publie vendredi ; IPC ~3 semaines).
                Renvoie None pour un arm si le fichier est absent (degrade
                gracieusement : le garde-fou correspondant est simplement ignore).
"""
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

NFCI_LAG = 5    # jours ouvres (snapshot mardi -> publie vendredi)
CPI_LAG = 15    # jours ouvres (~3 semaines de delai de publication IPC)


def load_price(ticker="QQQ", start="2000-01-01", end=None):
    price = pd.read_parquet(DATA_DIR / f"{ticker}.parquet")["close"]
    price = price.loc[start:end] if end else price.loc[start:]
    return price.dropna().sort_index()


def load_ohlc(ticker="QQQ", start="2000-01-01", end=None):
    """OHLC (open/high/low/close) d'un ticker -> DataFrame trie, sans NaN.
    Requis par la vol Yang-Zhang / le risk-off precoce."""
    df = pd.read_parquet(DATA_DIR / f"{ticker}.parquet")[["open", "high", "low", "close"]]
    df = df.loc[start:end] if end else df.loc[start:]
    return df.dropna().sort_index()


def _load_fred(fname, col, dates, lag):
    fp = DATA_DIR / fname
    if not fp.exists():
        return None
    s = pd.read_parquet(fp)[col]
    s = s.reindex(dates.union(s.index)).sort_index().ffill().reindex(dates)
    return s.shift(lag).values


def load_macro(dates, nfci_lag=NFCI_LAG, cpi_lag=CPI_LAG):
    """(nfci, cpi) alignes sur `dates`, decales du lag de publication.
    Un arm vaut None si son fichier FRED est absent."""
    nfci = _load_fred("fred_nfci.parquet", "nfci", dates, nfci_lag)
    cpi = _load_fred("fred_cpi.parquet", "cpi", dates, cpi_lag)
    return nfci, cpi


def load_funding(dates):
    """Taux court USD (FRED DFF, Fed funds) aligne sur `dates`, ffill, en FRACTION
    annuelle. Cout de financement du levier LQQ pour le backtest net de frais.
    None si le fichier est absent (le backtest net retombe alors sur le brut)."""
    fp = DATA_DIR / "fred_dff.parquet"
    if not fp.exists():
        return None
    s = pd.read_parquet(fp)["dff"]
    s = s.reindex(dates.union(s.index)).sort_index().ffill().reindex(dates).bfill()
    return s.values / 100.0


def load_ndx_excess_snapshot():
    """Snapshot courant de l'excess earnings yield NDX top-5/top-10 (trailing +
    forward vs taux reel), produit par download_ndx_excess_yield. Contexte du
    panneau valorisation. None si le JSON est absent."""
    import json
    fp = DATA_DIR / "pe" / "ndx_excess_yield.json"
    if not fp.exists():
        return None
    try:
        return json.loads(fp.read_text())
    except Exception:  # noqa: BLE001
        return None


def load_cape_ecy(dates, lag_days=5):
    """CAPE (P/E10) et Excess CAPE Yield du S&P 500, mensuels, alignes sur `dates`
    (ffill intra-mois) et decales de `lag_days` (petite marge d'honnetete ; le CAPE
    multpl du mois courant est estime en direct). Historique complet -> aujourd'hui.

    Variable de CONTEXTE de valorisation, ajustee des taux et comparable entre
    epoques -- PAS dans la strategie (meme statut que l'ancien PE top-5). La serie
    s'arrete a la derniere date Shiller reelle : au-dela, NaN (pas de prolongation
    a plat). Renvoie (None, None) si le fichier est absent (degrade gracieusement).
    """
    fp = DATA_DIR / "shiller" / "cape_ecy.parquet"
    if not fp.exists():
        return None, None
    df = pd.read_parquet(fp)

    def _align(col):
        s = df[col].dropna()
        if s.empty:
            return None
        last = s.index.max()
        a = s.reindex(dates.union(s.index)).sort_index().ffill().reindex(dates)
        a = a.shift(lag_days)              # lag de publication (~1 mois)
        a[a.index > last] = np.nan         # ne pas prolonger a plat au-dela de Shiller
        return a.values

    return _align("cape"), _align("ecy")
