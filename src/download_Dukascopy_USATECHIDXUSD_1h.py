"""Télécharge l'historique horaire du CFD Dukascopy USATECHIDXUSD (Nasdaq-100, ~24h/24).

Source : flux public gratuit Dukascopy (sans clé), fichiers binaires `.bi5` (LZMA) :
  - bougies 1h  : un fichier par mois  -> {inst}/{YYYY}/{MM-1}/BID_candles_hour_1.bi5
  - bougies 1min: un fichier par jour  -> {inst}/{YYYY}/{MM-1}/{DD}/BID_candles_min_1.bi5
    (utilisées pour le mois en cours, dont le fichier mensuel n'est pas encore publié)
Enregistrement = 24 octets big-endian `>5if` : (secondes depuis le début de la période,
open, close, low, high, volume) ; prix ×1000. Horodatage UTC.

HTTP simple (le HTTPS de datafeed.dukascopy.com expire depuis ce réseau).

Les bougies « de remplissage » (week-ends / fermetures : volume 0 et O=H=L=C) sont retirées.

Usage :
    python -m src.download_Dukascopy_USATECHIDXUSD_1h            # historique complet
    python -m src.download_Dukascopy_USATECHIDXUSD_1h --start 2020
"""
import argparse
import datetime as dt
import lzma
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

INSTRUMENT = "USATECHIDXUSD"
BASE_URL = "http://datafeed.dukascopy.com/datafeed"
OUT_PATH = Path("data/stocks") / f"{INSTRUMENT}_1h.parquet"
FIRST_YEAR_PROBE = 2005  # recherche du premier mois disponible à partir de cette année
REC = struct.Struct(">5if")
COLS = ["open", "high", "low", "close", "volume"]


def _fetch(url: str, retries: int = 5) -> bytes:
    """Retourne le contenu décompressé ('' si absent / vide)."""
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code == 404:
                return b""
            r.raise_for_status()
            return lzma.decompress(r.content) if r.content else b""
        except (requests.RequestException, lzma.LZMAError) as e:
            if attempt == retries - 1:
                raise RuntimeError(f"échec {url}: {e}") from e
            time.sleep(2 ** attempt)
    return b""


def _decode(raw: bytes, base: dt.datetime) -> list[tuple]:
    out = []
    for i in range(0, len(raw) - len(raw) % REC.size, REC.size):
        t, o, c, lo, h, v = REC.unpack_from(raw, i)
        out.append((base + dt.timedelta(seconds=t), o / 1e3, h / 1e3, lo / 1e3, c / 1e3, v))
    return out


def fetch_month(inst: str, year: int, month: int) -> list[tuple]:
    url = f"{BASE_URL}/{inst}/{year}/{month - 1:02d}/BID_candles_hour_1.bi5"
    return _decode(_fetch(url), dt.datetime(year, month, 1))


def fetch_day_minutes(inst: str, day: dt.date) -> list[tuple]:
    url = f"{BASE_URL}/{inst}/{day.year}/{day.month - 1:02d}/{day.day:02d}/BID_candles_min_1.bi5"
    return _decode(_fetch(url), dt.datetime(day.year, day.month, day.day))


def _is_real(rows: list[tuple]) -> bool:
    return any(r[5] > 0 or not (r[1] == r[2] == r[3] == r[4]) for r in rows)


def find_first_month(inst: str, start_year: int) -> tuple[int, int]:
    """Premier mois contenant de vraies bougies (année par année, puis mois par mois)."""
    this_year = dt.date.today().year
    year = next(y for y in range(start_year, this_year + 1) if _is_real(fetch_month(inst, y, 12)))
    if year > start_year and _is_real(fetch_month(inst, year - 1, 12)):
        year -= 1  # garde-fou (ne devrait pas arriver)
    for m in range(1, 13):
        if _is_real(fetch_month(inst, year, m)):
            return year, m
    return year, 12


def to_frame(rows: list[tuple]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["datetime", *COLS])
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    return df.drop_duplicates("datetime").set_index("datetime").sort_index()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", type=int, default=None,
                    help="année de départ (défaut : premier mois disponible détecté)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args()

    if args.start is None:
        y0, m0 = find_first_month(INSTRUMENT, FIRST_YEAR_PROBE)
        print(f"[INFO] premier mois disponible : {y0}-{m0:02d}")
    else:
        y0, m0 = args.start, 1

    today = dt.date.today()
    months = [(y, m) for y in range(y0, today.year + 1) for m in range(1, 13)
              if (y, m) >= (y0, m0) and (y, m) < (today.year, today.month)]
    print(f"[INFO] {len(months)} mois à télécharger ({months[0][0]}-{months[0][1]:02d} → "
          f"{months[-1][0]}-{months[-1][1]:02d}) + mois en cours en 1min agrégé")

    with ThreadPoolExecutor(args.workers) as ex:
        monthly = list(ex.map(lambda ym: fetch_month(INSTRUMENT, *ym), months))
        days = [today.replace(day=d) for d in range(1, today.day + 1)]
        minute = list(ex.map(lambda d: fetch_day_minutes(INSTRUMENT, d), days))

    df = to_frame([r for rows in monthly for r in rows])
    missing = [f"{y}-{m:02d}" for (y, m), rows in zip(months, monthly) if not rows]
    if missing:
        print(f"[WARN] mois sans fichier : {missing}")

    # Mois en cours : 1min -> 1h (fichier mensuel pas encore publié)
    mdf = to_frame([r for rows in minute for r in rows])
    if not mdf.empty:
        mdf = mdf[(mdf["volume"] > 0) | (mdf[["open", "high", "low"]].nunique(axis=1) > 1)]
        hourly = mdf.resample("1h").agg({"open": "first", "high": "max", "low": "min",
                                          "close": "last", "volume": "sum"}).dropna()
        df = pd.concat([df, hourly[~hourly.index.isin(df.index)]]).sort_index()

    flat = (df["volume"] == 0) & (df["open"] == df["high"]) & (df["high"] == df["low"]) & (df["low"] == df["close"])
    df = df[~flat]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out)
    print(f"[OK] {len(df)} bougies 1h ({flat.sum()} bougies de remplissage retirées) → {args.out}")
    print(f"[OK] plage : {df.index[0]} → {df.index[-1]} (UTC)")


if __name__ == "__main__":
    main()
