"""
Robot for MyQTM-ETF strategy with SM + FL models and heuristic fallback.

Regime logic (same as backtest.py):
  VIX spike 5d > 6           -> cash (unless SM gate open)
  VIX EMA100 < 19 + FL open  -> Follow Leads model
  VIX EMA100 < 19 + FL closed-> heuristic top-5 Sharpe 2y
  VIX EMA100 >= 19 + SM open -> Smart Money model
  VIX EMA100 >= 20 + slope   -> cash
  else                       -> heuristic fallback

Usage:
    python robot.py --data                # refresh OHLCV + VIX + iShares XLS
    python robot.py --allocation          # show current target allocation
    python robot.py --trade               # execute trades via IB
    python robot.py --dry-run             # show trade plan without executing
    python robot.py --train               # retrain if end of 21-day block
    python robot.py --stop                # stop IB Gateway docker
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
    IC_GATE_THRESHOLD,
    IC_GATE_MIN_STEPS,
    FL_IC_GATE_THRESHOLD,
    FL_IC_GATE_MIN_STEPS,
    TOP_N_ALLOC,
    FL_SHARPE_POWER,
)
from etf import UNIVERSE, TRADING_MAP

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

DATA = Path(__file__).parent / "data"
OUTPUTS = Path(__file__).parent / "outputs"
ROBOT_DIR = OUTPUTS / "robot"
ROBOT_DATA = DATA / "robot"
ROBOT_DATA.mkdir(parents=True, exist_ok=True)

REBAL_MIN_CHANGE = 0.03
ACCOUNT_CURRENCY = "EUR"

# ---------------------------------------------------------------------------
#  IB contract mapping from TRADING_MAP (UCITS EUR)
# ---------------------------------------------------------------------------
IB_CONTRACT_MAP = {
    etf_id: (sym, exch, curr)
    for etf_id, (sym, exch, curr, _issuer, _fee) in TRADING_MAP.items()
}
IB_SYMBOL_TO_ETF = {sym: etf_id for etf_id, (sym, _, _) in IB_CONTRACT_MAP.items()}


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


def refresh_ishares():
    """Download + parse iShares XLS (shares outstanding for smart-money features)."""
    print("Refreshing iShares XLS data...")
    base = Path(__file__).parent
    try:
        subprocess.run([sys.executable, str(base / "download_ishares_xls.py")],
                       cwd=str(base), check=True, timeout=600)
        subprocess.run([sys.executable, str(base / "parse_ishares_xls.py")],
                       cwd=str(base), check=True, timeout=120)
    except subprocess.TimeoutExpired:
        print("  WARNING: iShares download timed out")
    except Exception as e:
        print(f"  iShares error: {e}")


def refresh_features():
    """Regenerate feature engineering (needed for model inference)."""
    print("Regenerating features...")
    base = Path(__file__).parent
    subprocess.run([sys.executable, str(base / "feature_engineering.py")],
                   cwd=str(base), check=True, timeout=600)


def refresh_data(full: bool = True):
    """Refresh all data sources."""
    refresh_ohlcv()
    print()
    refresh_vix()
    if full:
        print()
        refresh_ishares()
        print()
        refresh_features()
    print("Data refresh complete.\n")


# ===================================================================
#  Model inference
# ===================================================================

def load_robot_artifacts() -> dict | None:
    """Load exported model artifacts from outputs/robot/."""
    if not ROBOT_DIR.exists():
        return None

    ic_path = ROBOT_DIR / "ic_gate_state.json"
    info_path = ROBOT_DIR / "last_step_info.json"
    sm_feat_path = ROBOT_DIR / "sm_features.json"
    fl_feat_path = ROBOT_DIR / "fl_features.json"

    if not all(p.exists() for p in [ic_path, info_path, sm_feat_path, fl_feat_path]):
        return None

    with open(ic_path) as f:
        ic_state = json.load(f)
    with open(info_path) as f:
        step_info = json.load(f)
    with open(sm_feat_path) as f:
        sm_features = json.load(f)
    with open(fl_feat_path) as f:
        fl_features = json.load(f)

    # Load SM models
    sm_models = _load_models(ROBOT_DIR / "sm_models")
    fl_models = _load_models(ROBOT_DIR / "fl_models")

    return {
        "ic_state": ic_state,
        "step_info": step_info,
        "sm_features": sm_features,
        "fl_features": fl_features,
        "sm_models": sm_models,
        "fl_models": fl_models,
        "sm_gate_open": ic_state["smart_money"]["gate_open"],
        "fl_gate_open": ic_state["follow_leads"]["gate_open"],
    }


def _load_models(model_dir: Path) -> list:
    """Load XGBoost models from .ubj files."""
    import xgboost as xgb
    models = []
    for i in range(20):
        path = model_dir / f"model_{i}.ubj"
        if path.exists():
            model = xgb.XGBRegressor()
            model.load_model(str(path))
            models.append(model)
    return models


def run_model_inference(features_df: pd.DataFrame, models: list,
                        feature_cols: list) -> pd.Series:
    """Run inference with an ensemble of XGBoost models. Returns mean scores."""
    available = [c for c in feature_cols if c in features_df.columns]
    if not available or not models:
        return pd.Series(dtype=float)

    X = features_df[available].astype(np.float32).replace(
        [np.inf, -np.inf], np.nan)

    scores_stack = [m.predict(X.values) for m in models]
    mean_scores = np.mean(scores_stack, axis=0)
    return pd.Series(mean_scores, index=features_df.index)


# ===================================================================
#  Allocation computation (full regime logic)
# ===================================================================

def compute_allocation() -> dict:
    """
    Compute today's target allocation using the full regime logic.

    Same as backtest.py _process_step:
      1. Compute heuristic (top-5 Sharpe 2y)
      2. Check VIX regime
      3. If SM/FL models available and IC gate open, use model predictions
      4. Otherwise fall back to heuristic

    Returns dict with weights, regime, VIX info, model info.
    """
    daily_ret = _load_daily_returns()
    vix_s, vix_ema100, _ = _load_vix()
    etf_ids = [e.bourso for e in UNIVERSE]
    data_date = daily_ret.index[-1]

    # --- Heuristic allocation (always computed as fallback) ---
    roll_mean = daily_ret.rolling(504, min_periods=400).mean().iloc[-1] * 252
    roll_std = (daily_ret.rolling(504, min_periods=400).std().iloc[-1]
                * np.sqrt(252))
    roll_sharpe = (roll_mean / roll_std.replace(0, np.nan)).clip(0.0).fillna(0.0)
    sharpe_all = {etf: float(roll_sharpe.get(etf, 0.0)) for etf in etf_ids}

    heuristic_sharpe = roll_sharpe.reindex(etf_ids).fillna(0.0)
    heur_top = [e for e in heuristic_sharpe.nlargest(CALM_TOP_N).index
                if heuristic_sharpe[e] > 0]

    # --- VIX regime ---
    current_vix_ema100 = 0.0
    vix_slope = 0.0
    if not vix_ema100.empty:
        ema_reindexed = vix_ema100.reindex(daily_ret.index, method="ffill").dropna()
        if len(ema_reindexed) > 1:
            current_vix_ema100 = float(ema_reindexed.iloc[-1])
            vix_slope = float(ema_reindexed.diff().iloc[-1])

    is_spike = False
    if not vix_s.empty:
        vix_close = vix_s.reindex(daily_ret.index, method="ffill").dropna()
        if len(vix_close) >= 6:
            is_spike = float(vix_close.diff(5).iloc[-1]) > VIX_SPIKE_MIN

    is_calm = current_vix_ema100 < VIX_CALM_THRESHOLD

    # --- Load model artifacts ---
    artifacts = load_robot_artifacts()
    sm_gate_open = artifacts["sm_gate_open"] if artifacts else False
    fl_gate_open = artifacts["fl_gate_open"] if artifacts else False

    # --- Determine regime and compute weights ---
    regime = "heuristic"
    model_used = None
    weights = {}
    top_etfs = heur_top

    if is_spike and not sm_gate_open:
        regime = "cash"
    elif is_calm and fl_gate_open and artifacts and artifacts["fl_models"]:
        # Follow Leads model in calm markets
        regime = "follow_leads"
        model_used = "FL"
        weights, top_etfs = _compute_fl_weights(
            artifacts, etf_ids, heuristic_sharpe)
    elif is_calm:
        # Calm, no FL -> heuristic
        regime = "heuristic"
    elif sm_gate_open and artifacts and artifacts["sm_models"]:
        # Turbulent with SM model validated
        regime = "smart_money"
        model_used = "SM"
        weights, top_etfs = _compute_sm_weights(artifacts, etf_ids)
    elif current_vix_ema100 >= VIX_CASH_THRESHOLD and vix_slope > 0:
        regime = "cash"
    else:
        # Turbulent fallback -> heuristic
        regime = "heuristic"

    # Heuristic weights (for heuristic regime or empty model weights)
    if regime == "heuristic" or (regime in ("follow_leads", "smart_money") and not weights):
        regime = "heuristic" if not weights else regime
        top_etfs = heur_top
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

    if regime == "cash":
        weights = {}
        top_etfs = []

    return {
        "weights": weights,
        "regime": regime,
        "model_used": model_used,
        "vix_ema100": current_vix_ema100,
        "vix_slope": vix_slope,
        "is_spike": is_spike,
        "sm_gate_open": sm_gate_open,
        "fl_gate_open": fl_gate_open,
        "sm_ic_ema": artifacts["ic_state"]["smart_money"]["ic_ema"] if artifacts else None,
        "fl_ic_ema": artifacts["ic_state"]["follow_leads"]["ic_ema"] if artifacts else None,
        "top_etfs": top_etfs,
        "sharpe_all": sharpe_all,
        "date": str(data_date.date()),
    }


def _load_today_features() -> pd.DataFrame:
    """Load features for the most recent date."""
    features_df = pd.read_parquet(DATA / "features.parquet")
    last_date = features_df.index.get_level_values("date").max()
    today = features_df.loc[last_date]
    today.index = today.index.droplevel("date") if "date" in today.index.names else today.index
    return today


def _compute_sm_weights(artifacts: dict, etf_ids: list) -> tuple:
    """Compute Smart Money model weights (same as backtest _process_step)."""
    today_features = _load_today_features()

    scores = run_model_inference(today_features, artifacts["sm_models"],
                                artifacts["sm_features"])
    if scores.empty:
        return {}, []

    # Z-score cross-sectionally
    mu, sigma = scores.mean(), scores.std()
    if sigma > 0:
        scores = (scores - mu) / sigma

    # Top-N positive scores
    pos = scores[scores > 0].nlargest(TOP_N_ALLOC)
    if pos.empty:
        return {}, []

    # Softmax weights
    s = pos.values / max(TEMPERATURE, 1e-6)
    s = s - s.max()
    e_s = np.exp(s)
    w = e_s / e_s.sum()

    weights = {etf: float(w[i]) for i, etf in enumerate(pos.index)}
    return weights, list(pos.index)


def _compute_fl_weights(artifacts: dict, etf_ids: list,
                        heuristic_sharpe: pd.Series) -> tuple:
    """Compute Follow Leads model weights (same as backtest _process_step)."""
    today_features = _load_today_features()

    scores = run_model_inference(today_features, artifacts["fl_models"],
                                artifacts["fl_features"])
    if scores.empty:
        return {}, []

    fl_top = scores.nlargest(TOP_N_ALLOC).index.tolist()

    # Weight by expanding Sharpe ^ FL_SHARPE_POWER
    exp_sharpe = heuristic_sharpe.reindex(etf_ids).fillna(0.0)
    fl_sharpe = exp_sharpe ** FL_SHARPE_POWER
    fl_vals = np.array([fl_sharpe.get(e, 0) for e in fl_top])
    total = fl_vals.sum()
    if total > 0:
        fl_vals = fl_vals / total

    weights = {etf: float(fl_vals[i]) for i, etf in enumerate(fl_top)}
    return weights, fl_top


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
    currency = ACCOUNT_CURRENCY
    for v in _ib.accountValues():
        if v.tag == "NetLiquidationByCurrency" and v.currency == currency:
            net_liq = float(v.value)
        if v.tag == "TotalCashBalance" and v.currency == currency:
            cash = float(v.value)
    if net_liq is None:
        for v in _ib.accountValues():
            if v.tag == "NetLiquidation" and v.currency == "BASE":
                net_liq = float(v.value)
            if v.tag == "TotalCashBalance" and v.currency == "BASE":
                cash = float(v.value)
    return {"net_liquidation": net_liq, "cash": cash, "currency": currency}


def get_last_price(etf_id: str) -> float:
    proxy = next(e.proxy for e in UNIVERSE if e.bourso == etf_id)
    fname = DATA / f"{proxy.replace('.', '_')}.parquet"
    if fname.exists():
        return float(pd.read_parquet(fname)["close"].iloc[-1])
    return 0.0


def market_is_open() -> bool:
    """Check if European market (XETRA) is open via IB RTH hours."""
    from ib_insync import Stock
    try:
        c = Stock("SXR8", "IBIS2", "EUR")
        c = _ib.qualifyContracts(c)[0]
        cd = _ib.reqContractDetails(c)[0]
        tzid = cd.timeZoneId or "Europe/Berlin"
        tz = ZoneInfo(tzid)
        now_utc = _ib.reqCurrentTime()
        if now_utc.tzinfo is None:
            from datetime import timezone
            now_utc = now_utc.replace(tzinfo=timezone.utc)
        now_local = now_utc.astimezone(tz)
        raw = cd.tradingHours
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
                start_dt = _parse_ib_time(start_tok, date_part, tz)
                end_dt = _parse_ib_time(end_tok, date_part, tz)
                if start_dt and end_dt and start_dt <= now_local < end_dt:
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
    from ib_insync import MarketOrder

    account = get_account_value()
    net_liq = account["net_liquidation"]
    current_cash = account["cash"]
    current_pos = get_ib_positions()

    curr = account.get("currency", ACCOUNT_CURRENCY)
    print(f"\n  Account: net_liq={net_liq:,.0f} {curr}  cash={current_cash:,.0f} {curr}")
    print(f"  Current positions ({len(current_pos)}):")
    for etf_id, shares in sorted(current_pos.items()):
        ucits_sym = IB_CONTRACT_MAP.get(etf_id, (etf_id,))[0]
        price = get_last_price(etf_id)
        value = shares * price
        print(f"    {etf_id:<10} ({ucits_sym:<6}) {shares:>6.0f} sh  "
              f"@ ~{price:.2f}  = {value:>10,.0f}")

    current_weights = {}
    if net_liq and net_liq > 0:
        for etf_id, shares in current_pos.items():
            current_weights[etf_id] = shares * get_last_price(etf_id) / net_liq

    all_etfs = set(list(target_weights.keys()) + list(current_weights.keys()))
    weight_change = sum(abs(target_weights.get(e, 0) - current_weights.get(e, 0))
                        for e in all_etfs)
    if weight_change < REBAL_MIN_CHANGE:
        print(f"\n  Weight change {weight_change:.1%} < {REBAL_MIN_CHANGE:.0%} "
              f"-- no rebalance needed.")
        return

    target_shares = {}
    for etf_id, weight in target_weights.items():
        price = get_last_price(etf_id)
        if price > 0:
            target_shares[etf_id] = int(net_liq * weight / price)

    print(f"\n  Target allocation (weight change: {weight_change:.1%}):")
    for etf_id, shares in sorted(target_shares.items()):
        ucits_sym = IB_CONTRACT_MAP.get(etf_id, (etf_id,))[0]
        w = target_weights[etf_id]
        price = get_last_price(etf_id)
        print(f"    {etf_id:<10} ({ucits_sym:<6}) {shares:>6d} sh  "
              f"@ ~{price:.2f}  w={w:.1%}")

    if dry_run:
        print("\n  [DRY RUN] No trades executed.")
        _print_trade_plan(current_pos, target_shares)
        return

    trades_log = []

    for etf_id, shares in current_pos.items():
        if etf_id not in target_shares and abs(shares) > 0:
            action = "SELL" if shares > 0 else "BUY"
            qty = int(abs(shares))
            print(f"  CLOSE {etf_id}: {action} {qty} shares")
            fill = _place_order(etf_id, action, qty)
            trades_log.append({"etf": etf_id, "action": action, "qty": qty,
                               "fill": fill})

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
    print("\n  Trade plan:")
    for etf_id, shares in current_pos.items():
        if etf_id not in target_shares and abs(shares) > 0:
            action = "SELL" if shares > 0 else "BUY"
            print(f"    CLOSE {etf_id}: {action} {abs(shares):.0f} shares")
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
#  Incremental training (--train)
# ===================================================================

def check_and_train():
    """Check if we're at end of a 21-day block and retrain if needed."""
    step_info_path = ROBOT_DIR / "last_step_info.json"
    if not step_info_path.exists():
        print("No robot artifacts found. Run full training first:")
        print("  python train_smart_money.py && python train_follow_leads.py")
        print("  python export_robot_artifacts.py")
        return

    with open(step_info_path) as f:
        step_info = json.load(f)

    last_test_end = pd.Timestamp(step_info["test_end"])
    today = pd.Timestamp.now().normalize()
    days_since = (today - last_test_end).days
    trading_days_since = int(days_since * 5 / 7)  # approximate

    print(f"Last training step: {step_info['last_step']} "
          f"(test ended {step_info['test_end']})")
    print(f"Trading days since: ~{trading_days_since}")

    if trading_days_since < 21:
        print(f"Not yet end of block ({trading_days_since}/21 days). No retraining needed.")
        return

    print(f"End of block reached! Retraining...")
    base = Path(__file__).parent

    # Retrain both models
    print("\n--- Smart Money ---")
    subprocess.run([sys.executable, str(base / "train_smart_money.py")],
                   cwd=str(base), check=True)
    print("\n--- Follow Leads ---")
    subprocess.run([sys.executable, str(base / "train_follow_leads.py")],
                   cwd=str(base), check=True)

    # Re-export artifacts
    print("\n--- Export artifacts ---")
    subprocess.run([sys.executable, str(base / "export_robot_artifacts.py")],
                   cwd=str(base), check=True)

    print("\nRetraining complete.")


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
    print(f"\nDate: {alloc['date']}")
    print(f"Regime: {alloc['regime']}")
    print(f"VIX EMA100: {alloc['vix_ema100']:.1f}  "
          f"(slope: {alloc['vix_slope']:+.3f})"
          f"{'  SPIKE!' if alloc.get('is_spike') else ''}")

    # IC gate status
    sm_ema = alloc.get("sm_ic_ema")
    fl_ema = alloc.get("fl_ic_ema")
    sm_str = f"ema={sm_ema:.3f} {'OPEN' if alloc['sm_gate_open'] else 'CLOSED'}" if sm_ema is not None else "no artifacts"
    fl_str = f"ema={fl_ema:.3f} {'OPEN' if alloc['fl_gate_open'] else 'CLOSED'}" if fl_ema is not None else "no artifacts"
    print(f"SM gate: {sm_str}  (threshold {IC_GATE_THRESHOLD})")
    print(f"FL gate: {fl_str}  (threshold {FL_IC_GATE_THRESHOLD})")

    if alloc["regime"] == "cash":
        print("\n  -> CASH (no positions)")
        return

    model_label = f" [{alloc['model_used']} model]" if alloc.get("model_used") else ""
    print(f"\nAllocation{model_label}:")
    for etf_id in alloc["top_etfs"]:
        sh = alloc["sharpe_all"].get(etf_id, 0)
        w = alloc["weights"].get(etf_id, 0.0)
        name = next((e.name for e in UNIVERSE if e.bourso == etf_id), etf_id)
        ucits = TRADING_MAP.get(etf_id, (etf_id,))[0]
        exch = TRADING_MAP.get(etf_id, ("", "", "", "", ""))[1]
        print(f"  {etf_id:<10} -> {ucits:<6} ({exch:<8}) {name:<22} "
              f"Sharpe={sh:.3f}  Weight={w:.1%}")


def main():
    parser = argparse.ArgumentParser(
        description="MyQTM-ETF Robot (SM + FL + heuristic)")
    parser.add_argument("--data", action="store_true",
                        help="Refresh OHLCV + VIX + iShares + features")
    parser.add_argument("--data-quick", action="store_true",
                        help="Refresh OHLCV + VIX only (no iShares)")
    parser.add_argument("--trade", action="store_true",
                        help="Execute trades via IB")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show trades without executing")
    parser.add_argument("--allocation", action="store_true",
                        help="Show current target allocation")
    parser.add_argument("--train", action="store_true",
                        help="Retrain if end of 21-day block")
    parser.add_argument("--stop", action="store_true",
                        help="Stop IB Gateway docker")
    args = parser.parse_args()

    if args.stop:
        stop_gateway()
        return

    if args.data:
        refresh_data(full=True)
    elif args.data_quick:
        refresh_data(full=False)

    if args.train:
        check_and_train()

    if args.allocation:
        alloc = compute_allocation()
        print_allocation(alloc)
        if not args.trade and not args.dry_run:
            return

    if args.trade or args.dry_run:
        paris_now = datetime.now(ZoneInfo("Europe/Paris"))
        if paris_now.weekday() >= 5:
            print("Weekend -- no trading.")
            return

        alloc = compute_allocation()
        print_allocation(alloc)

        last_file = ROBOT_DATA / "last_processed.json"
        if last_file.exists() and not args.dry_run:
            with open(last_file) as f:
                last_date = json.load(f)
            if last_date == alloc["date"]:
                print(f"\nAlready processed {last_date} -- skipping.")
                return

        if not ensure_gateway():
            return

        try:
            ib_connect()
            print("IB connected.\n")

            if args.trade:
                print("Checking XETRA market status...")
                wait_count = 0
                while not market_is_open():
                    if wait_count == 0:
                        print("  European market not open yet, waiting...")
                    time.sleep(30)
                    wait_count += 1
                    if wait_count > 120:
                        print("  Market did not open in time, aborting.")
                        return
                print("  European market is open.")

            execute_trades(alloc["weights"], dry_run=args.dry_run)

            if not args.dry_run:
                with open(last_file, "w") as f:
                    json.dump(alloc["date"], f)

        except Exception as e:
            print(f"\nERROR: {e}")
            import traceback
            traceback.print_exc()
        finally:
            ib_disconnect()

    if not any([args.data, args.data_quick, args.trade, args.dry_run,
                args.allocation, args.train, args.stop]):
        parser.print_help()


if __name__ == "__main__":
    main()
