"""
Strategie de crise deployee — trend250 + vol-managed + garde-fous macro.
Ecrit outputs/<ticker>_strategy/signal.json pour l'execution PEA du matin.

Usage: python -m src.risk_off_strategy.run [QQQ] [--no-refresh]   (QQQ par defaut)
"""

import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.risk_off_strategy.data import load_ohlc, load_macro, load_cape_ecy, load_ndx_excess_snapshot
from src.risk_off_strategy.strategy import (
    compute_allocation, yang_zhang_vol, NFCI_OFF, CPI_OFF, ABOVE_CAP,
)
from src.risk_off_strategy.backtest import plot_backtest


# ── Download fresh data ──────────────────────────────────
US_CLOSE_HOUR = 22  # US market close in Paris time (22h00 = 16h00 ET)


def retry(fn, max_retries=5, initial_wait=60, max_wait=900,
          hourly_until=None, label=""):
    """Retry with exponential backoff, then hourly until deadline."""
    import time
    from datetime import datetime
    wait = initial_wait
    for attempt in range(1, max_retries + 1):
        try:
            return fn()
        except Exception as e:
            if attempt == max_retries:
                break
            print(f"[RETRY] {label}: tentative {attempt}/{max_retries} echouee: {e}")
            print(f"  Prochaine tentative dans {wait}s...")
            time.sleep(wait)
            wait = min(wait * 2, max_wait)

    if hourly_until is None:
        print(f"[ERREUR] {label}: echec apres {max_retries} tentatives")
        raise
    attempt = max_retries
    while datetime.now() < hourly_until:
        attempt += 1
        print(f"[RETRY] {label}: tentative {attempt} (horaire), prochaine dans 1h...")
        time.sleep(3600)
        try:
            return fn()
        except Exception as e:
            print(f"[RETRY] {label}: tentative {attempt} echouee: {e}")
    print(f"[ERREUR] {label}: echec, deadline {hourly_until} atteinte")
    raise


def refresh_data():
    """Re-download OHLCV + FRED macro (dont NFCI et IPC). Retry sur echec reseau."""
    from src.download_ohlcv import download_risk_off, RISK_OFF_TICKERS
    from src.download_macro_data import fetch_fred, FRED_SERIES
    from datetime import datetime, timezone, timedelta

    DATA_DIR = ROOT / "data"
    tomorrow_830 = (datetime.now() + timedelta(hours=10)).replace(hour=8, minute=30, second=0)

    retry(lambda: download_risk_off(force=True),
          hourly_until=tomorrow_830, label="OHLCV download")

    def _fetch_fred():
        for series_id, spec in FRED_SERIES.items():
            label = spec[0] if isinstance(spec, (tuple, list)) else spec
            units = spec[1] if isinstance(spec, (tuple, list)) and len(spec) > 1 else "lin"
            fetch_fred(series_id, label, force=True, units=units)
    retry(_fetch_fred, hourly_until=tomorrow_830, label="FRED download")

    paris_tz = timezone(timedelta(hours=2))
    now_paris = datetime.now(paris_tz)
    today = pd.Timestamp(now_paris.date())
    before_close = now_paris.hour < US_CLOSE_HOUR and today.weekday() < 5
    expected_date = None if (before_close or today.weekday() >= 5) else today
    if before_close:
        print(f"Before US close ({now_paris.strftime('%H:%M')} Paris) — excluding today")

    if expected_date is not None:
        import time as _time
        wait_tickers = [t for t in RISK_OFF_TICKERS if t != "vix_ohlc"]
        for data_attempt in range(3):
            download_risk_off(force=True)
            missing = []
            for t in wait_tickers:
                p = DATA_DIR / f"{t}.parquet"
                if not p.exists() or pd.read_parquet(p).index[-1] < expected_date:
                    missing.append(t)
            if not missing:
                print(f"Toutes les donnees du {expected_date.date()} disponibles")
                break
            if data_attempt < 2:
                print(f"[ATTENTE] Donnees {expected_date.date()} manquantes: {', '.join(missing)}. Retry 300s...")
                _time.sleep(300)
        else:
            print(f"[INFO] Donnees {expected_date.date()} indisponibles — probable jour ferie US")

    last_dates = []
    for t in RISK_OFF_TICKERS:
        p = DATA_DIR / f"{t}.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            if before_close:
                df = df[df.index < today]
            if len(df) > 0:
                last_dates.append(df.index[-1])
    end_date = max(last_dates).strftime("%Y-%m-%d") if last_dates else "2026-12-31"
    print(f"Data up to: {end_date}\n")
    return end_date


def refresh_valuation_context():
    """Rafraichit le contexte de valorisation (CAPE/ECY S&P + excess yield NDX)
    en best-effort : contexte de graphe uniquement, jamais dans la decision, donc
    un echec (multpl.com/yfinance/FRED indispo) ne doit PAS faire echouer le run.
    Le backtest lit ensuite les fichiers via load_cape_ecy/load_ndx_excess_snapshot."""
    from src.download_shiller_cape import main as cape_main
    from src.download_ndx_excess_yield import main as ndx_main
    for label, fn in (("CAPE/ECY S&P", cape_main), ("excess yield NDX", ndx_main)):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            print(f"[WARN] contexte {label} non rafraichi ({repr(e)[:90]}) — on garde la version precedente")


# ── Config ────────────────────────────────────────────────
START = "2000-01-01"

NO_REFRESH = "--no-refresh" in sys.argv
if NO_REFRESH:
    sys.argv.remove("--no-refresh")
    END = None
    print("Skipping data refresh (utilise toutes les donnees locales)\n")
else:
    END = refresh_data()

# QQQ uniquement par defaut (le systeme live ne trade que PUST/Nasdaq).
# Un autre ticker peut etre passe explicitement (ex: SPY) pour exploration.
ticker = sys.argv[1].upper() if len(sys.argv) > 1 else "QQQ"
TICKERS = [ticker]

# Contexte de valorisation (CAPE/ECY + excess yield NDX) rafraichi ici, juste
# avant le backtest, pour que le graphe lise toujours la version du jour (evite la
# dependance implicite a un cron 22:20 separe). QQQ seulement, best-effort.
if not NO_REFRESH and ticker == "QQQ":
    refresh_valuation_context()


def run_ticker(ticker):
    prefix = ticker.lower().replace("-", "_")
    OUT = ROOT / "outputs" / f"{prefix}_strategy"
    OUT.mkdir(parents=True, exist_ok=True)
    signal_path = OUT / "signal.json"

    # Marque "running" pour invalider un "ok" perime si on plante
    with open(signal_path, "w") as f:
        json.dump({"status": "running", "ticker": ticker,
                   "timestamp": pd.Timestamp.now().isoformat()}, f)

    print(f"\n{'='*70}\n=== {ticker} — strategie deployee (trend250 + VM + garde-fous) ===\n{'='*70}\n")

    ohlc = load_ohlc(ticker, START, END)
    price = ohlc["close"]
    o, h, l = ohlc["open"].values, ohlc["high"].values, ohlc["low"].values
    nfci, cpi = load_macro(price.index)
    cape, ecy = load_cape_ecy(price.index) if ticker == "QQQ" else (None, None)
    ndx_ey = load_ndx_excess_snapshot() if ticker == "QQQ" else None
    alloc = compute_allocation(price.values, nfci, cpi, high=h, low=l, open_=o)
    vol = yang_zhang_vol(o, h, l, price.values)   # meme vol pour le panneau du chart

    kw = dict(ticker=ticker, vol=vol, above_cap=ABOVE_CAP, ndx_ey=ndx_ey)
    m = plot_backtest(price, alloc, nfci, cpi, cape, ecy, save_path=str(OUT / "backtest.png"), **kw)
    plot_backtest(price, alloc, nfci, cpi, cape, ecy, save_path=str(OUT / "backtest_10y.png"), last_days=10 * 252, **kw)
    plot_backtest(price, alloc, nfci, cpi, cape, ecy, save_path=str(OUT / "backtest_5y.png"), last_days=5 * 252, **kw)
    plot_backtest(price, alloc, nfci, cpi, cape, ecy, save_path=str(OUT / "backtest_1y.png"), last_days=252, **kw)
    plot_backtest(price, alloc, nfci, cpi, cape, ecy, save_path=str(OUT / "backtest_1m.png"), last_days=21, **kw)

    last_alloc = float(alloc[-1])
    macro_off = bool(
        (nfci is not None and np.nan_to_num(nfci[-1], nan=-9) > NFCI_OFF) or
        (cpi is not None and np.nan_to_num(cpi[-1], nan=-9) > CPI_OFF)
    )
    signal = {
        "status": "ok",
        "ticker": ticker,
        "date": str(price.index[-1].date()),
        "probability": last_alloc,      # compat real_bourso/webapp (= allocation)
        "allocation": last_alloc,
        "macro_off": macro_off,
        "timestamp": pd.Timestamp.now().isoformat(),
    }
    with open(signal_path, "w") as f:
        json.dump(signal, f, indent=2)

    print(f"Backtest {price.index[0].date()}->{price.index[-1].date()}  "
          f"CAGR {m['cagr']*100:.1f}%  maxDD {m['maxdd']*100:.0f}%  "
          f"Sharpe {m['sharpe']:.2f}  Calmar {m['calmar']:.2f}")
    print(f"Signal: alloc={last_alloc*100:.0f}%  macro_off={macro_off}  -> {signal_path}")
    return price, alloc


# ── Run ───────────────────────────────────────────────────
for t in TICKERS:
    run_ticker(t)
