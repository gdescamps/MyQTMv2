"""
Real trading on Boursorama PEA via bourso-cli.

Workflow (daily, Paris time):
  08:00 — Download US close data, run model inference
  09:00 — Execute orders at Euronext open

ETF: PUST (Amundi PEA Nasdaq-100 UCITS ETF, FR0013412269)
     Symbol on Bourso: 1rTPUST

Fees:
  - Buy:  0% (free ETF on Bourso PEA)
  - Sell: 0.5% (Bourso PEA fee)
  → Only sell if allocation delta >= 20% to limit fees

Usage:
  python src/real_bourso.py                  # dry-run (no orders)
  python src/real_bourso.py --execute        # live execution
  python src/real_bourso.py --status         # show current positions
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Config ────────────────────────────────────────────────
BOURSO_CLI = Path.home() / ".local" / "bin" / "bourso-cli"
CREDENTIALS_FILE = Path.home() / ".bourso" / "credentials.json"  # optional

ETF_SYMBOL = "1rTPUST"       # Amundi PEA Nasdaq-100
ETF_NAME = "PUST (Nasdaq-100 PEA)"
PEA_ACCOUNT_ID = "e0aeafb04e60bdbe140479e499fd79d2"  # from bourso-cli accounts

SELL_THRESHOLD = 0.20         # only sell if delta_alloc >= 20%
PROB_CASH = 0.70
PROB_FULL = 0.75
TEMPERATURE = 3.0
LOOKAHEAD = 6

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
TRADE_LOG = LOG_DIR / "trades.jsonl"


# ── Bourso CLI wrapper ────────────────────────────────────
def _run_bourso(args, password=None):
    """Run bourso-cli command and return stdout."""
    cmd = [str(BOURSO_CLI)] + args
    if CREDENTIALS_FILE.exists():
        cmd = [str(BOURSO_CLI), "--credentials", str(CREDENTIALS_FILE)] + args

    env = None
    inp = None
    if password:
        inp = password + "\n"

    result = subprocess.run(cmd, capture_output=True, text=True, input=inp, timeout=60)
    if result.returncode != 0:
        print(f"[ERROR] bourso-cli failed: {result.stderr}")
    return result.stdout, result.stderr


def get_etf_price():
    """Get current ETF price (no login needed)."""
    stdout, stderr = _run_bourso(["quote", "--symbol", ETF_SYMBOL, "last"])
    # bourso-cli outputs INFO logs to stderr
    output = stdout + "\n" + stderr
    for line in output.split("\n"):
        if "current:" in line:
            parts = line.split("current:")[1].split(",")[0].strip()
            return float(parts)
    return None


def get_accounts(password):
    """Get trading accounts."""
    stdout, _ = _run_bourso(["accounts", "--trading", "true"], password=password)
    return stdout


def place_order(side, quantity, password, dry_run=True):
    """Place buy or sell order."""
    if quantity <= 0:
        return None

    action = f"{'BUY' if side == 'buy' else 'SELL'} {quantity} x {ETF_NAME}"

    if dry_run:
        print(f"  [DRY-RUN] {action}")
        return {"status": "dry-run", "side": side, "quantity": quantity}

    print(f"  [EXECUTE] {action}")
    stdout, stderr = _run_bourso([
        "trade", "order", "new",
        "--side", side,
        "--account", PEA_ACCOUNT_ID,
        "--symbol", ETF_SYMBOL,
        "--quantity", str(quantity),
    ], password=password)

    result = {"status": "executed", "side": side, "quantity": quantity,
              "stdout": stdout, "stderr": stderr}
    return result


# ── Model inference ───────────────────────────────────────
def run_inference():
    """Download latest data, run model, return allocation probability."""
    from src.risk_off_strategy.data import load_data, build_features, build_realtime_target
    from src.risk_off_strategy.backtest import walk_forward

    print("Downloading latest data...")
    from src.download_ohlcv import download_risk_off
    download_risk_off(force=True)

    # Refresh FRED
    from src.download_macro_data import fetch_fred, FRED_SERIES
    DATA_DIR = ROOT / "data"
    for series_id, label in FRED_SERIES.items():
        path = DATA_DIR / f"fred_{label}.parquet"
        if path.exists():
            path.unlink()
        fetch_fred(series_id, label)

    print("Running model inference...")
    price, vix, spread, tlt = load_data("QQQ", "2000-01-01", "2030-12-31")
    df = build_features(price, vix, spread, tlt, prefix="qqq")
    target, _ = build_realtime_target(price.values, -0.10, -0.05, lookahead=LOOKAHEAD)
    df["target"] = target
    df = df.dropna()

    feature_cols = [c for c in df.columns if c != "target"]
    X = df[feature_cols].values
    y = df["target"].values

    wf_pred, wf_proba, _ = walk_forward(
        X, y, feature_cols, min_train=504, step=21, temperature=TEMPERATURE
    )

    # Get latest probability
    last_idx = np.where(wf_pred >= 0)[0][-1]
    last_prob = wf_proba[last_idx]
    last_date = df.index[last_idx]

    # Compute target allocation
    alloc = np.clip((last_prob - PROB_CASH) / (PROB_FULL - PROB_CASH), 0, 1)

    print(f"\nModel inference:")
    print(f"  Date:        {last_date.date()}")
    print(f"  Probability: {last_prob:.4f}")
    print(f"  Allocation:  {alloc*100:.0f}%")

    return last_prob, alloc, last_date


# ── Trading logic ─────────────────────────────────────────
def compute_orders(target_alloc, current_shares, etf_price, total_equity):
    """Compute orders needed to reach target allocation.

    Args:
        target_alloc: 0.0 to 1.0
        current_shares: number of ETF shares currently held
        etf_price: current ETF price
        total_equity: total account value (cash + positions)

    Returns:
        side: 'buy', 'sell', or None
        quantity: number of shares
        reason: explanation string
    """
    current_value = current_shares * etf_price
    current_alloc = current_value / total_equity if total_equity > 0 else 0
    target_value = target_alloc * total_equity
    delta_value = target_value - current_value
    delta_alloc = target_alloc - current_alloc

    print(f"\nAllocation:")
    print(f"  Current: {current_alloc*100:.1f}% ({current_shares} shares, {current_value:.0f} EUR)")
    print(f"  Target:  {target_alloc*100:.1f}% ({target_value:.0f} EUR)")
    print(f"  Delta:   {delta_alloc*100:+.1f}% ({delta_value:+.0f} EUR)")

    if delta_value > etf_price:
        # BUY — no fee, execute freely
        quantity = int(delta_value / etf_price)
        return "buy", quantity, f"Buy {quantity} shares (+{delta_alloc*100:.1f}%)"

    elif delta_value < -etf_price and abs(delta_alloc) >= SELL_THRESHOLD:
        # SELL — 0.5% fee, only if delta >= 20%
        quantity = int(abs(delta_value) / etf_price)
        return "sell", quantity, f"Sell {quantity} shares ({delta_alloc*100:.1f}%, fee ~{quantity*etf_price*0.005:.1f} EUR)"

    elif delta_value < -etf_price:
        return None, 0, f"Sell skipped: delta {abs(delta_alloc)*100:.1f}% < threshold {SELL_THRESHOLD*100:.0f}%"

    else:
        return None, 0, "No action needed (within 1 share)"


def log_trade(record):
    """Append trade record to JSONL log."""
    record["timestamp"] = datetime.now().isoformat()
    with open(TRADE_LOG, "a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"  Logged → {TRADE_LOG}")


# ── Main ──────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Risk-off real trading on Bourso PEA")
    parser.add_argument("--execute", action="store_true", help="Execute orders (default: dry-run)")
    parser.add_argument("--status", action="store_true", help="Show current status only")
    parser.add_argument("--shares", type=int, default=0, help="Current number of ETF shares held")
    parser.add_argument("--equity", type=float, default=0, help="Total account equity (EUR)")
    parser.add_argument("--password", type=str, default=None, help="Bourso password (or use credentials file)")
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"Risk-Off Trading — {ETF_NAME}")
    print(f"Date: {date.today()}")
    print(f"Mode: {'LIVE' if args.execute else 'DRY-RUN'}")
    print(f"{'='*60}")

    # 1. Get ETF price
    etf_price = get_etf_price()
    if etf_price:
        print(f"\nETF price: {etf_price:.2f} EUR")
    else:
        print("[ERROR] Could not get ETF price")
        return

    # 2. Status only
    if args.status:
        if args.password:
            print("\nAccounts:")
            print(get_accounts(args.password))
        return

    # 3. Run inference
    prob, alloc, model_date = run_inference()

    # 4. Compute orders
    if args.equity <= 0:
        print(f"\n[INFO] Specify --equity and --shares to compute orders")
        print(f"  Example: python src/real_bourso.py --equity 10000 --shares 50")
        return

    side, quantity, reason = compute_orders(alloc, args.shares, etf_price, args.equity)
    print(f"\nDecision: {reason}")

    # 5. Execute or dry-run
    if side and quantity > 0:
        result = place_order(side, quantity, args.password, dry_run=not args.execute)

        log_trade({
            "date": str(date.today()),
            "model_date": str(model_date.date()),
            "probability": float(prob),
            "target_alloc": float(alloc),
            "side": side,
            "quantity": quantity,
            "etf_price": etf_price,
            "equity": args.equity,
            "current_shares": args.shares,
            "executed": args.execute,
            "result": result,
        })
    else:
        log_trade({
            "date": str(date.today()),
            "model_date": str(model_date.date()),
            "probability": float(prob),
            "target_alloc": float(alloc),
            "side": None,
            "quantity": 0,
            "etf_price": etf_price,
            "equity": args.equity,
            "current_shares": args.shares,
            "reason": reason,
        })

    print(f"\nDone.")


if __name__ == "__main__":
    main()
