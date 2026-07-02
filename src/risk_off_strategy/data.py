"""
Chargement des donnees pour la strategie deployee (trend + vol-managed + macro).

- load_price  : cloture d'un ticker (QQQ par defaut), depuis 2000.
- load_macro  : NFCI et IPC YoY alignes sur les dates de prix, decales du lag de
                publication FRED (NFCI ~mardi publie vendredi ; IPC ~3 semaines).
                Renvoie None pour un arm si le fichier est absent (degrade
                gracieusement : le garde-fou correspondant est simplement ignore).
"""
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


def load_pe(dates):
    """PE cap-weighted du top-5 NASDAQ-100, aligne sur `dates` (contexte
    valorisation, pas dans la strategie). None si le fichier est absent."""
    fp = DATA_DIR / "pe" / "pe_top5_daily.parquet"
    if not fp.exists():
        return None
    s = pd.read_parquet(fp)["pe_top5_daily"]
    return s.reindex(dates.union(s.index)).sort_index().ffill().reindex(dates).values
