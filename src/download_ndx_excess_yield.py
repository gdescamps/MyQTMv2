"""
Excess earnings yield du top-5 / top-10 du Nasdaq-100 : rendement benefices
cap-pondere (courant ET forward) MOINS le taux reel 10 ans (FRED DFII10).

Analogue "specifie correctement" de l'ECY pour les leaders Nasdaq. Un vrai CAPE
(benefices lisses 10 ans) est INUTILISABLE sur des compounders : leur moyenne 10a
de benefices est une fraction du niveau actuel -> CAPE absurde (150-400). On garde
donc la structure "rendement - taux reel" SANS lissage cyclique :
  - trailing : 1/PE trailing  (benefices actuels)          -> "les leaders sont-ils
  - forward  : 1/PE forward    (benefices attendus)            chers vs les oblig ?"
Le saut trailing -> forward materialise la prime de croissance (le "credit IA").
Agregation = rendement benefices cap-pondere (Sw_i . EY_i), economiquement correct
(= benefices totaux / capitalisation totale), contrairement a une moyenne de PE.

Contexte de valorisation (webapp), PAS dans la strategie.
Source  : yfinance (trailingPE, forwardPE, marketCap -- couvre tous les titres, y
          compris ceux gates "premium" sur FMP) + FRED DFII10 (taux reel 10a).
Univers : top-20 NDX en cache (data/pe/ndx_constituents.json), re-classe par market
          cap courant.

Sortie  : data/pe/ndx_excess_yield.json
Usage   : python -m src.download_ndx_excess_yield        (quotidien, cf. cron)
"""

import os
import json
import ssl
import urllib.request
import datetime as dt
from pathlib import Path

import yfinance as yf
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
PE_DIR = ROOT / "data" / "pe"
OUT = PE_DIR / "ndx_excess_yield.json"

load_dotenv(ROOT / ".env")
FRED_KEY = os.environ.get("FRED")


def fetch_real_rate():
    """Taux reel 10 ans (FRED DFII10, TIPS) en fraction. None si indispo."""
    if not FRED_KEY:
        return None
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    try:
        u = (f"https://api.stlouisfed.org/fred/series/observations?series_id=DFII10"
             f"&api_key={FRED_KEY}&file_type=json&sort_order=desc&limit=8")
        req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"})
        obs = json.loads(urllib.request.urlopen(req, timeout=25, context=ctx).read())["observations"]
        for o in obs:
            if o["value"] not in (".", ""):
                return float(o["value"]) / 100.0
    except Exception as e:  # noqa: BLE001
        print(f"  WARN DFII10 : {repr(e)[:80]}")
    return None


def _universe():
    cons = json.load(open(PE_DIR / "ndx_constituents.json"))
    return [c["symbol"] for c in cons]


def _yield_from_pe(pe):
    """1/PE en fraction ; None si PE absent/<=0 (pertes -> pas de rendement)."""
    try:
        pe = float(pe)
    except (TypeError, ValueError):
        return None
    return 1.0 / pe if pe > 0 else None


def build():
    today = dt.date.today()
    real_rate = fetch_real_rate()
    if real_rate is None:
        raise RuntimeError("taux reel 10 ans indisponible (FRED DFII10)")

    rows = []
    for sym in _universe():
        try:
            info = yf.Ticker(sym).info
            mcap = info.get("marketCap")
            if not mcap:
                print(f"  {sym:6} SKIP : pas de marketCap")
                continue
            ey_trail = _yield_from_pe(info.get("trailingPE"))
            ey_fwd = _yield_from_pe(info.get("forwardPE"))
            rows.append({"symbol": sym, "mktCap": float(mcap),
                         "ey_trail": ey_trail, "ey_fwd": ey_fwd})
            print(f"  {sym:6} cap={float(mcap)/1e9:6.0f}B  "
                  f"EY_trail={ey_trail*100 if ey_trail else float('nan'):5.2f}%  "
                  f"EY_fwd={ey_fwd*100 if ey_fwd else float('nan'):5.2f}%")
        except Exception as e:  # noqa: BLE001
            print(f"  {sym:6} SKIP : {repr(e)[:70]}")

    if not rows:
        raise RuntimeError("aucun titre recupere (yfinance)")
    rows.sort(key=lambda r: r["mktCap"], reverse=True)

    def agg(subset, key):
        vals = [(r["mktCap"], r[key]) for r in subset if r.get(key) is not None]
        if not vals:
            return None, 0
        w = sum(c for c, _ in vals)
        return sum(c * v for c, v in vals) / w, len(vals)

    def block(n):
        sub = rows[:n]
        eyt, nt = agg(sub, "ey_trail")
        eyf, nf = agg(sub, "ey_fwd")
        return {
            "symbols": [r["symbol"] for r in sub],
            "ey_trailing": eyt, "ey_forward": eyf,
            "excess_trailing": (eyt - real_rate) if eyt is not None else None,
            "excess_forward": (eyf - real_rate) if eyf is not None else None,
            "n": len(sub), "n_trailing": nt, "n_forward": nf,
        }

    cap10 = sum(x["mktCap"] for x in rows[:10])
    return {
        "date": today.isoformat(),
        "real_rate": real_rate,
        "top5": block(5),
        "top10": block(10),
        "constituents": [
            {"symbol": r["symbol"], "weight_top10": r["mktCap"] / cap10,
             "ey_trail": r["ey_trail"], "ey_fwd": r["ey_fwd"]}
            for r in rows[:10]
        ],
    }


def main():
    print("=== Excess earnings yield NDX top-5 / top-10 (yfinance + FRED) ===")
    data = build()
    OUT.write_text(json.dumps(data, indent=2))
    rr = data["real_rate"] * 100

    def pct(x):
        return f"{x*100:+.2f}%" if x is not None else "  n/a"

    print(f"\n  taux reel 10a = {rr:.2f}%   ({data['date']})")
    for k in ("top5", "top10"):
        b = data[k]
        print(f"  {k:5} : EY trailing {pct(b['ey_trailing'])} -> excess {pct(b['excess_trailing'])}"
              f"  |  EY forward {pct(b['ey_forward'])} -> excess {pct(b['excess_forward'])}"
              f"   (couv. {b['n_trailing']}/{b['n_forward']} sur {b['n']})")
    print(f"  -> {OUT}")
    return data


if __name__ == "__main__":
    main()
