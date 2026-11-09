"""Séries FRED supplémentaires pour tester de nouveaux garde-fous PUST x1.7 (crédit + choc de taux).

Historique long (depuis 1990, pour avoir le recul de 6-12 mois avant la bulle 2000).
- BAA10Y       : écart de crédit Moody's BAA − Treasury 10 ans (quotidien, 1986+) — proxy crédit long
                 (le spread HY ICE BofA BAMLH0A0HYM2 n'est plus diffusé par FRED que sur ~3 ans)
- BAMLH0A0HYM2 : spread HY ICE BofA (récupéré pour mémoire, historique tronqué)
- DGS2         : taux Treasury 2 ans (choc de taux / anticipations Fed)
- DGS10        : taux Treasury 10 ans
- DFII10       : taux réel 10 ans (TIPS, 2003+)
Sortie : data/fred_extra/{label}.parquet (date index, 1 colonne). Clé : FRED / FRED_API_KEY du .env.
    python -m myfiles.download_extra_fred
"""
import os
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
KEY = os.getenv("FRED_API_KEY") or os.getenv("FRED")
OUT = ROOT / "data" / "fred_extra"
SERIES = {"BAA10Y": "baa10y_spread", "BAMLH0A0HYM2": "hy_oas", "DGS2": "y2", "DGS10": "y10", "DFII10": "real10"}
START = "1990-01-01"


def fetch(sid):
    r = requests.get("https://api.stlouisfed.org/fred/series/observations",
                     params={"series_id": sid, "api_key": KEY, "file_type": "json", "observation_start": START},
                     timeout=60)
    r.raise_for_status()
    d = pd.DataFrame(r.json()["observations"])[["date", "value"]]
    d["date"] = pd.to_datetime(d["date"])
    d["value"] = pd.to_numeric(d["value"], errors="coerce")
    return d.dropna().set_index("date")["value"]


def main():
    if not KEY:
        raise SystemExit("Clé FRED absente du .env")
    OUT.mkdir(parents=True, exist_ok=True)
    for sid, label in SERIES.items():
        s = fetch(sid).rename(label)
        s.to_frame().to_parquet(OUT / f"{label}.parquet")
        print(f"  {sid:13s} → {label:14s} {len(s):6d} obs  {s.index[0].date()} → {s.index[-1].date()}  dernier {s.iloc[-1]:.2f}")
        time.sleep(0.2)


if __name__ == "__main__":
    main()
