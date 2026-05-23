"""
Robot for MyQTM-ETF heuristic strategy.

Executes top-5 Sharpe-weighted 2y rolling allocation with VIX regime management.
No predictive model needed -- runs the same heuristic as backtest.py calm mode.

Regime logic (shared with backtest.py):
  VIX EMA100 < 19          -> heuristic top-5 Sharpe-weighted 2y
  VIX EMA100 >= 20 + slope -> cash (no SM model available)
  VIX spike 5d > 6         -> cash for 5 days
  else                     -> heuristic fallback

Usage:
    python robot.py --data                # refresh OHLCV + VIX data
    python robot.py --trade               # compute allocation + execute via IB
    python robot.py --dry-run             # show what would trade, no execution
    python robot.py --allocation          # show current target allocation only
    python robot.py --data --trade        # refresh data then trade
"""

import argparse
import json
import os
import random
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from backtest import (
    _load_daily_returns,
    _load_vix,
    CALM_TOP_N,
    VIX_CALM_THRESHOLD,
    VIX_CASH_THRESHOLD,
    VIX_SPIKE_MIN,
    VIX_SPIKE_CASH_DAYS,
    TEMPERATURE,
    USE_SOFTMAX,
)
from etf import UNIVERSE

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

DATA = Path(__file__).parent / "data"
ROBOT_DATA = DATA / "robot"
ROBOT_DATA.mkdir(parents=True, exist_ok=True)

REBAL_MIN_CHANGE = 0.03  # minimum weight change to trigger rebalance

# ---------------------------------------------------------------------------
#  IB contract mapping: etf bourso id -> (symbol, exchange, currency)
# ---------------------------------------------------------------------------
IB_CONTRACT_MAP = {}
for _etf in UNIVERSE:
    _t = _etf.bourso
    if _t == "ISF.L":
        IB_CONTRACT_MAP[_t] = ("ISF", "LSEETF", "GBP")
    elif _t == "SXRS.DE":
        IB_CONTRACT_MAP[_t] = ("SXRS", "SMART", "EUR")
    elif _t.endswith(".DE"):
        IB_CONTRACT_MAP[_t] = (_t.replace(".DE", ""), "IBIS", "EUR")
    elif _t.endswith(".AS"):
        IB_CONTRACT_MAP[_t] = (_t.replace(".AS", ""), "AEB", "EUR")
    else:
        IB_CONTRACT_MAP[_t] = (_t, "SMART", "USD")

# Reverse map: IB symbol -> etf_id (for reading back positions)
IB_SYMBOL_TO_ETF = {}
for _etf_id, (_sym, _exch, _curr) in IB_CONTRACT_MAP.items():
    IB_SYMBOL_TO_ETF[_sym] = _etf_id


# ===================================================================
#  Data refresh
# ===================================================================

def refresh_ohlcv():
    """Update OHLCV parquet files with latest yfinance data."""
    import yfinance as yf
    print("Refreshing OHLCV data...")
    tickers = sorted(set(e.proxy for e in UNIVERSE))

    for ticker in tickers:
        fname = DATA / f"{ticker.replace('.', '_')}.parquet"
        if not fname.exists():
            print(f"  {ticker}: no existing data, run download_ohlcv.py first")
            continue

        existing = pd.read_parquet(fname)
        last_date = existing.index[-1]
        today = pd.Timestamp.now().normalize()

        if last_date >= today - pd.Timedelta(days=1):
            print(f"  {ticker}: up to date ({last_date.date()})")
            continue

        try:
            new_data = yf.download(
                ticker,
                start=(last_date - pd.Timedelta(days=5)).strftime("%Y-%m-%d"),
                progress=False, auto_adjust=True,
            )
        except Exception as e:
            print(f"  {ticker}: download error: {e}")
            continue

        if new_data is None or new_data.empty:
            print(f"  {ticker}: no new data")
            continue

        if isinstance(new_data.columns, pd.MultiIndex):
            new_data.columns = new_data.columns.get_level_values(0)
        new_data.columns = [c.lower() for c in new_data.columns]
        new_data.index = pd.to_datetime(new_data.index).tz_localize(None)
        new_data.index.name = "date"
        cols = [c for c in ("open", "high", "low", "close", "volume")
                if c in new_data.columns]
        new_data = new_data[cols]

        combined = pd.concat([existing, new_data])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        combined.to_parquet(fname)
        print(f"  {ticker}: {last_date.date()} -> {combined.index[-1].date()} "
              f"({len(combined)} rows)")

        time.sleep(0.3)


def refresh_vix():
    """Update VIX data from FRED."""
    import requests

    FRED_KEY = os.getenv("FRED_API_KEY") or os.getenv("FRED")
    if not FRED_KEY:
        print("  VIX: FRED_API_KEY/FRED not set, skipping")
        return

    fname = DATA / "fred_vix.parquet"
    if not fname.exists():
        print("  VIX: no existing data, run download_macro_data.py first")
        return

    existing = pd.read_parquet(fname)
    last_date = existing.index[-1]

    try:
        r = requests.get(
            "https://api.stlouisfed.org/fred/series/observations",
            params={
                "series_id": "VIXCLS",
                "api_key": FRED_KEY,
                "file_type": "json",
                "observation_start": (last_date - pd.Timedelta(days=5))
                    .strftime("%Y-%m-%d"),
            },
            timeout=30,
        )
        r.raise_for_status()
        obs = r.json().get("observations", [])
    except Exception as e:
        print(f"  VIX: FRED error: {e}")
        return

    if not obs:
        print(f"  VIX: already up to date ({last_date.date()})")
        return

    df = pd.DataFrame(obs)[["date", "value"]]
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    col_name = existing.columns[0]
    df = df.dropna().rename(columns={"value": col_name})

    combined = pd.concat([existing, df])
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    combined.to_parquet(fname)
    print(f"  VIX: updated -> {combined.index[-1].date()} ({len(combined)} rows)")


def refresh_data():
    refresh_ohlcv()
    print()
    refresh_vix()
    print("Data refresh complete.\n")


# ===================================================================
#  Heuristic allocation (shared logic with backtest.py)
# ===================================================================

def compute_heuristic_allocation() -> dict:
    """
    Compute today's target allocation using the heuristic strategy.

    Uses identical logic to backtest.py calm mode:
      - 504d (2y) rolling Sharpe for all ETFs
      - top-5 by Sharpe, softmax-weighted
      - VIX regime: calm -> heuristic, turbulent -> cash (no SM model)

    Returns dict:
      weights     : {etf_id: weight}  (sums to ~1.0, or empty dict if cash)
      regime      : "calm" | "turbulent_heuristic" | "cash"
      vix_ema100  : current value
      top_etfs    : selected ETF ids
      sharpe_all  : {etf_id: sharpe} for all ETFs
      date        : data date
    """
    daily_ret = _load_daily_returns()
    vix_s, vix_ema100, _ = _load_vix()

    etf_ids = [e.bourso for e in UNIVERSE]

    # 2-year (504d) rolling Sharpe -- same as backtest.py _process_step
    roll_mean = daily_ret.rolling(504, min_periods=400).mean().iloc[-1] * 252
    roll_std = (daily_ret.rolling(504, min_periods=400).std().iloc[-1]
                * np.sqrt(252))
    roll_sharpe = (roll_mean / roll_std.replace(0, np.nan)).clip(0.0).fillna(0.0)

    sharpe_all = {etf: float(roll_sharpe.get(etf, 0.0)) for etf in etf_ids}

    # Top-N selection
    heuristic_sharpe = roll_sharpe.reindex(etf_ids).fillna(0.0)
    heur_top = [e for e in heuristic_sharpe.nlargest(CALM_TOP_N).index
                if heuristic_sharpe[e] > 0]

    # --- VIX regime ---
    data_date = daily_ret.index[-1]

    current_vix_ema100 = 0.0
    vix_slope = 0.0
    if not vix_ema100.empty:
        ema_reindexed = vix_ema100.reindex(daily_ret.index, method="ffill").dropna()
        if len(ema_reindexed) > 1:
            current_vix_ema100 = float(ema_reindexed.iloc[-1])
            vix_slope = float(ema_reindexed.diff().iloc[-1])

    # VIX spike (5d change > VIX_SPIKE_MIN)
    is_spike = False
    if not vix_s.empty:
        vix_close = vix_s.reindex(daily_ret.index, method="ffill").dropna()
        if len(vix_close) >= 6:
            is_spike = float(vix_close.diff(5).iloc[-1]) > VIX_SPIKE_MIN

    # Regime decision (same as backtest.py without SM/FL models)
    if is_spike:
        regime = "cash"
    elif current_vix_ema100 < VIX_CALM_THRESHOLD:
        regime = "calm"
    elif current_vix_ema100 >= VIX_CASH_THRESHOLD and vix_slope > 0:
        regime = "cash"
    else:
        # 19 <= VIX < 20, or slope not rising: stay with heuristic
        regime = "turbulent_heuristic"

    # Compute weights (softmax of Sharpe, same as backtest.py run_backtest)
    if regime == "cash":
        weights = {}
    else:
        scores = np.array([heuristic_sharpe[e] for e in heur_top])
        if USE_SOFTMAX and len(scores) > 0:
            s = scores / max(TEMPERATURE, 1e-6)
            s = s - s.max()
            e_s = np.exp(s)
            w = e_s / e_s.sum()
        elif len(scores) > 0:
            w = np.ones(len(scores)) / len(scores)
        else:
            w = np.array([])
        weights = {etf: float(w[i]) for i, etf in enumerate(heur_top)}

    return {
        "weights": weights,
        "regime": regime,
        "vix_ema100": current_vix_ema100,
        "vix_slope": vix_slope,
        "top_etfs": heur_top,
        "sharpe_all": sharpe_all,
        "date": str(data_date.date()),
    }


# ===================================================================
#  IB connection & helpers
# ===================================================================

_ib = None


def ib_connect(host="127.0.0.1", port=4002, timeout=30):
    global _ib
    from ib_insync import IB
    if _ib is not None and _ib.isConnected():
        return _ib
    if _ib is not None:
        _ib.disconnect()
    _ib = IB()
    client_id = random.randint(1, 10000)
    _ib.connect(host, port, clientId=client_id, timeout=timeout)
    _ib.errorEvent += lambda reqId, code, msg, adv: print(
        f"  [IB] reqId={reqId} code={code} msg={msg}")
    return _ib


def ib_disconnect():
    global _ib
    if _ib is not None:
        _ib.disconnect()
    _ib = None


def make_contract(etf_id: str):
    from ib_insync import Stock
    sym, exch, curr = IB_CONTRACT_MAP[etf_id]
    return Stock(sym, exch, curr)


def get_ib_positions() -> dict:
    """Current IB positions as {etf_id: shares}."""
    positions = {}
    for pos in _ib.positions():
        symbol = pos.contract.symbol
        etf_id = IB_SYMBOL_TO_ETF.get(symbol)
        if etf_id is not None:
            positions[etf_id] = float(pos.position)
    return positions


def get_account_value() -> dict:
    net_liq = None
    cash = None
    for v in _ib.accountValues():
        if v.tag == "NetLiquidationByCurrency" and v.currency == "USD":
            net_liq = float(v.value)
        if v.tag == "TotalCashBalance" and v.currency == "USD":
            cash = float(v.value)
    return {"net_liquidation": net_liq, "cash": cash}


def get_last_price(etf_id: str) -> float:
    """Get last close from parquet (reliable, no market data subscription)."""
    proxy = next(e.proxy for e in UNIVERSE if e.bourso == etf_id)
    fname = DATA / f"{proxy.replace('.', '_')}.parquet"
    if fname.exists():
        return float(pd.read_parquet(fname)["close"].iloc[-1])
    return 0.0


def market_is_open() -> bool:
    """Check if US market is open via IB."""
    from ib_insync import Stock
    try:
        c = Stock("AAPL", "SMART", "USD", primaryExchange="NASDAQ")
        c = _ib.qualifyContracts(c)[0]
        cd = _ib.reqContractDetails(c)[0]
        tzid = cd.timeZoneId or "America/New_York"
        et = ZoneInfo(tzid)
        now_utc = _ib.reqCurrentTime()
        if now_utc.tzinfo is None:
            from datetime import timezone
            now_utc = now_utc.replace(tzinfo=timezone.utc)
        now_et = now_utc.astimezone(et)
        raw = cd.liquidHours
        if not raw:
            return False
        for day_block in raw.split(";"):
            day_block = day_block.strip()
            if not day_block or ":" not in day_block:
                continue
            date_part, times_part = day_block.split(":", 1)
            if times_part.strip().upper() == "CLOSED":
                continue
            for span in times_part.split(","):
                span = span.strip()
                if not span or "-" not in span:
                    continue
                start_tok, end_tok = span.split("-", 1)
                start_dt = _parse_ib_time(start_tok, date_part, et)
                end_dt = _parse_ib_time(end_tok, date_part, et)
                if start_dt and end_dt and start_dt <= now_et < end_dt:
                    return True
    except Exception as e:
        print(f"  market_is_open error: {e}")
    return False


def _parse_ib_time(token, default_date, tz):
    token = token.strip()
    if ":" in token:
        dpart, tpart = token.split(":", 1)
    else:
        dpart, tpart = default_date, token
    try:
        y, m, d = int(dpart[:4]), int(dpart[4:6]), int(dpart[6:8])
        h, mi = int(tpart[:2]), int(tpart[2:])
        return datetime(y, m, d, h, mi, tzinfo=tz)
    except (ValueError, IndexError):
        return None


# ===================================================================
#  Trade execution
# ===================================================================

def execute_trades(target_weights: dict, dry_run: bool = False):
    """
    Rebalance portfolio to target_weights.

    1. Get current positions + account value
    2. Compute target shares from weights * net_liquidation
    3. Close positions not in target
    4. Rebalance existing + open new
    """
    from ib_insync import MarketOrder

    account = get_account_value()
    net_liq = account["net_liquidation"]
    current_cash = account["cash"]
    current_pos = get_ib_positions()

    print(f"\n  Account: net_liq={net_liq:,.0f} USD  cash={current_cash:,.0f} USD")
    print(f"  Current positions ({len(current_pos)}):")
    for etf_id, shares in sorted(current_pos.items()):
        price = get_last_price(etf_id)
        value = shares * price
        print(f"    {etf_id:<10} {shares:>6.0f} sh  "
              f"@ ~{price:.2f}  = {value:>10,.0f} USD")

    # Compute current weights
    current_weights = {}
    if net_liq and net_liq > 0:
        for etf_id, shares in current_pos.items():
            current_weights[etf_id] = shares * get_last_price(etf_id) / net_liq

    # Check if rebalance needed
    all_etfs = set(list(target_weights.keys()) + list(current_weights.keys()))
    weight_change = sum(abs(target_weights.get(e, 0) - current_weights.get(e, 0))
                        for e in all_etfs)
    if weight_change < REBAL_MIN_CHANGE:
        print(f"\n  Weight change {weight_change:.1%} < {REBAL_MIN_CHANGE:.0%} "
              f"-- no rebalance needed.")
        return

    # Target shares
    target_shares = {}
    for etf_id, weight in target_weights.items():
        price = get_last_price(etf_id)
        if price > 0:
            target_shares[etf_id] = int(net_liq * weight / price)

    print(f"\n  Target allocation (weight change: {weight_change:.1%}):")
    for etf_id, shares in sorted(target_shares.items()):
        w = target_weights[etf_id]
        price = get_last_price(etf_id)
        print(f"    {etf_id:<10} {shares:>6d} sh  "
              f"@ ~{price:.2f}  w={w:.1%}")

    if dry_run:
        print("\n  [DRY RUN] No trades executed.")
        _print_trade_plan(current_pos, target_shares)
        return

    # --- Execute: close first, then open/adjust ---
    trades_log = []

    # 1. Close positions not in target
    for etf_id, shares in current_pos.items():
        if etf_id not in target_shares and abs(shares) > 0:
            action = "SELL" if shares > 0 else "BUY"
            qty = int(abs(shares))
            print(f"  CLOSE {etf_id}: {action} {qty} shares")
            fill = _place_order(etf_id, action, qty)
            trades_log.append({"etf": etf_id, "action": action, "qty": qty,
                               "fill": fill})

    # 2. Adjust existing + open new
    for etf_id, target in target_shares.items():
        current = int(current_pos.get(etf_id, 0))
        delta = target - current
        if abs(delta) < 1:
            continue

        action = "BUY" if delta > 0 else "SELL"
        qty = abs(delta)
        print(f"  {'OPEN' if current == 0 else 'ADJUST'} {etf_id}: "
              f"{action} {qty} shares (current={current}, target={target})")
        fill = _place_order(etf_id, action, qty)
        trades_log.append({"etf": etf_id, "action": action, "qty": qty,
                           "fill": fill})

    # Save trade log
    now_str = datetime.now(ZoneInfo("Europe/Paris")).strftime("%Y-%m-%d_%H%M")
    log_file = ROBOT_DATA / f"{now_str}_trades.json"
    with open(log_file, "w") as f:
        json.dump({
            "datetime": now_str,
            "account": account,
            "target_weights": target_weights,
            "target_shares": target_shares,
            "current_positions": current_pos,
            "trades": trades_log,
        }, f, indent=2, default=str)
    print(f"\n  Trade log: {log_file}")


def _place_order(etf_id: str, action: str, qty: int) -> float:
    """Place a market order and wait for fill. Returns fill price."""
    from ib_insync import MarketOrder
    contract = make_contract(etf_id)
    _ib.qualifyContracts(contract)
    trade = _ib.placeOrder(contract, MarketOrder(action, qty))
    timeout = 60
    start = time.time()
    while not trade.isDone():
        _ib.waitOnUpdate(timeout=5)
        if time.time() - start > timeout:
            print(f"    WARNING: order timeout for {etf_id}")
            break
    fill_price = trade.orderStatus.avgFillPrice
    status = trade.orderStatus.status
    print(f"    -> {status} @ {fill_price:.2f}")
    return fill_price


def _print_trade_plan(current_pos: dict, target_shares: dict):
    """Print planned trades without executing."""
    print("\n  Trade plan:")
    # Closes
    for etf_id, shares in current_pos.items():
        if etf_id not in target_shares and abs(shares) > 0:
            action = "SELL" if shares > 0 else "BUY"
            print(f"    CLOSE {etf_id}: {action} {abs(shares):.0f} shares")
    # Opens / adjustments
    for etf_id, target in target_shares.items():
        current = int(current_pos.get(etf_id, 0))
        delta = target - current
        if abs(delta) < 1:
            print(f"    HOLD  {etf_id}: {current} shares (no change)")
        else:
            action = "BUY" if delta > 0 else "SELL"
            label = "OPEN" if current == 0 else "ADJUST"
            print(f"    {label:6s} {etf_id}: {action} {abs(delta)} shares "
                  f"(current={current}, target={target})")


# ===================================================================
#  Docker gateway management
# ===================================================================

def wait_for_port(host, port, timeout=60):
    start = time.time()
    while time.time() - start < timeout:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex((host, port)) == 0:
                return True
        time.sleep(2)
    return False


def ensure_gateway():
    """Start IB Gateway via Docker if not already running."""
    if wait_for_port("127.0.0.1", 4002, timeout=2):
        print("IB Gateway already running.")
        return True

    print("Starting IB Gateway via Docker...")
    compose_dir = Path(__file__).parent
    subprocess.run(["docker", "compose", "up", "-d"],
                   cwd=str(compose_dir), check=True)

    print("Waiting for gateway (port 4002)...")
    if wait_for_port("127.0.0.1", 4002, timeout=120):
        print("IB Gateway ready, waiting 20s for full init...")
        time.sleep(20)
        return True

    print("ERROR: IB Gateway did not start in time.")
    return False


def stop_gateway():
    compose_dir = Path(__file__).parent
    subprocess.run(["docker", "compose", "down"],
                   cwd=str(compose_dir), check=True)


# ===================================================================
#  Main
# ===================================================================

def print_allocation(alloc: dict):
    """Pretty-print the heuristic allocation."""
    print(f"\nDate: {alloc['date']}")
    print(f"Regime: {alloc['regime']}")
    print(f"VIX EMA100: {alloc['vix_ema100']:.1f}  "
          f"(slope: {alloc['vix_slope']:+.3f})")

    if alloc["regime"] == "cash":
        print("\n  -> CASH regime (no positions)")
        return

    print(f"\nTop {CALM_TOP_N} ETFs by 2y rolling Sharpe:")
    for etf_id in alloc["top_etfs"]:
        sh = alloc["sharpe_all"][etf_id]
        w = alloc["weights"].get(etf_id, 0.0)
        name = next((e.name for e in UNIVERSE if e.bourso == etf_id), etf_id)
        print(f"  {etf_id:<10} {name:<25} Sharpe={sh:.3f}  Weight={w:.1%}")


def main():
    parser = argparse.ArgumentParser(
        description="MyQTM-ETF Robot (heuristic strategy)")
    parser.add_argument("--data", action="store_true",
                        help="Refresh OHLCV + VIX data")
    parser.add_argument("--trade", action="store_true",
                        help="Execute trades via IB")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show trades without executing")
    parser.add_argument("--allocation", action="store_true",
                        help="Show current target allocation")
    parser.add_argument("--stop", action="store_true",
                        help="Stop IB Gateway docker")
    args = parser.parse_args()

    if args.stop:
        stop_gateway()
        return

    if args.data:
        refresh_data()

    if args.allocation:
        alloc = compute_heuristic_allocation()
        print_allocation(alloc)
        if not args.trade and not args.dry_run:
            return

    if args.trade or args.dry_run:
        # Weekend check
        paris_now = datetime.now(ZoneInfo("Europe/Paris"))
        if paris_now.weekday() >= 5:
            print("Weekend -- no trading.")
            return

        alloc = compute_heuristic_allocation()
        print_allocation(alloc)

        # Check last processed date (avoid double-trading)
        last_file = ROBOT_DATA / "last_processed.json"
        if last_file.exists() and not args.dry_run:
            with open(last_file) as f:
                last_date = json.load(f)
            if last_date == alloc["date"]:
                print(f"\nAlready processed {last_date} -- skipping.")
                return

        # Start gateway + connect
        if not ensure_gateway():
            return

        try:
            ib_connect()
            print("IB connected.\n")

            # Wait for US market if trading for real
            if args.trade:
                print("Checking market status...")
                wait_count = 0
                while not market_is_open():
                    if wait_count == 0:
                        print("  US market not open yet, waiting...")
                    time.sleep(30)
                    wait_count += 1
                    if wait_count > 120:  # ~1h
                        print("  Market did not open in time, aborting.")
                        return
                print("  US market is open.")

            execute_trades(alloc["weights"], dry_run=args.dry_run)

            # Mark as processed
            if not args.dry_run:
                with open(last_file, "w") as f:
                    json.dump(alloc["date"], f)

        except Exception as e:
            print(f"\nERROR: {e}")
            import traceback
            traceback.print_exc()
        finally:
            ib_disconnect()

    if not any([args.data, args.trade, args.dry_run,
                args.allocation, args.stop]):
        parser.print_help()


if __name__ == "__main__":
    main()
