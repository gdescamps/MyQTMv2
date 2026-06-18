"""
Morning PEA execution on BoursoBank — reads signal from evening backtest.

Workflow:
  22:30  cron backtest (run.py) → outputs/qqq_strategy/signal.json
  09:05  this script → reads signal, checks PEA, executes PUST at Euronext open

ETF: PUST (Amundi PEA Nasdaq-100 UCITS ETF, FR0011871110)
Leverage: x1 only (no LQQ)

Usage:
  python -m src.real_bourso                  # dry-run
  python -m src.real_bourso --execute        # live execution
"""

import argparse
import json
import sys
from datetime import datetime, date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SIGNAL_PATH = ROOT / "outputs" / "qqq_strategy" / "signal.json"
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
TRADE_LOG = LOG_DIR / "trades.jsonl"
OVERRIDE_FILE = LOG_DIR / "emergency_off.json"

SELL_THRESHOLD = 0.20  # only sell if delta_alloc >= 20% (0.5% fee on sells)
MAX_SIGNAL_AGE_HOURS = 18  # signal must be < 18h old (evening to morning)


def load_signal():
    """Load latest signal from backtest output."""
    if not SIGNAL_PATH.exists():
        print(f"[ERREUR] Signal introuvable: {SIGNAL_PATH}")
        return None

    with open(SIGNAL_PATH) as f:
        signal = json.load(f)

    # Check signal freshness
    ts = datetime.fromisoformat(signal["timestamp"])
    age = datetime.now() - ts
    if age > timedelta(hours=MAX_SIGNAL_AGE_HOURS):
        print(f"[ERREUR] Signal trop ancien: {signal['date']} ({age.total_seconds()/3600:.0f}h)")
        print(f"  Le backtest du soir a-t-il tourne? Verifier logs/cron_backtest.log")
        return None

    print(f"Signal du {signal['date']} (age: {age.total_seconds()/3600:.1f}h)")
    print(f"  Probabilite: {signal['probability']:.4f}")
    print(f"  Allocation:  {signal['allocation']*100:.0f}%")
    return signal


def get_pea_state():
    """Get PEA cash and PUST position via bourso-cli prepare."""
    from src.bourso.prepare import prepare_order, PEA_ACCOUNT_ID, SYMBOLS

    data = prepare_order(PEA_ACCOUNT_ID, SYMBOLS["PUST"])
    acct = data["account"]
    sym = data["symbol"]

    state = {
        "cash": acct["cash"],
        "stocks": acct["stocks"],
        "equity": acct["cash"] + acct["stocks"],
        "pust_price": sym["last_price"],
        "pust_shares": data["quantity_held"],
        "pust_value": data["quantity_held"] * sym["last_price"],
    }

    print(f"\nPEA {acct['name']}:")
    print(f"  Especes:  {state['cash']:>10.2f} EUR")
    print(f"  Titres:   {state['stocks']:>10.2f} EUR")
    print(f"  Total:    {state['equity']:>10.2f} EUR")
    print(f"  PUST:     {state['pust_shares']} parts @ {state['pust_price']:.2f} EUR")
    return state


def is_emergency_off():
    """Check if emergency override is active."""
    if OVERRIDE_FILE.exists():
        with open(OVERRIDE_FILE) as f:
            data = json.load(f)
        if data.get("active", False):
            print(f"\n{'!'*60}")
            print(f"  EMERGENCY OFF — allocation forcee a 0%")
            print(f"  Supprimer {OVERRIDE_FILE} pour reprendre")
            print(f"{'!'*60}")
            return True
    return False


def compute_orders(target_alloc, pea_state):
    """Compute buy/sell orders to reach target allocation."""
    equity = pea_state["equity"]
    price = pea_state["pust_price"]
    current_shares = pea_state["pust_shares"]
    current_value = current_shares * price
    current_alloc = current_value / equity if equity > 0 else 0

    target_value = target_alloc * equity
    delta_value = target_value - current_value
    delta_alloc = target_alloc - current_alloc

    print(f"\nAllocation:")
    print(f"  Actuelle: {current_alloc*100:.1f}% ({current_shares} parts, {current_value:.0f} EUR)")
    print(f"  Cible:    {target_alloc*100:.1f}% ({target_value:.0f} EUR)")
    print(f"  Delta:    {delta_alloc*100:+.1f}% ({delta_value:+.0f} EUR)")

    if delta_value > price:
        quantity = int(delta_value / price)
        # Check cash available
        if quantity * price > pea_state["cash"]:
            quantity = int(pea_state["cash"] / price)
            if quantity <= 0:
                return None, 0, "Cash insuffisant pour acheter"
        return "buy", quantity, f"Acheter {quantity} parts (+{delta_alloc*100:.1f}%)"

    elif delta_value < -price and abs(delta_alloc) >= SELL_THRESHOLD:
        quantity = int(abs(delta_value) / price)
        quantity = min(quantity, current_shares)
        fee = quantity * price * 0.005
        return "sell", quantity, f"Vendre {quantity} parts ({delta_alloc*100:.1f}%, frais ~{fee:.1f} EUR)"

    elif delta_value < -price:
        return None, 0, f"Vente ignoree: delta {abs(delta_alloc)*100:.1f}% < seuil {SELL_THRESHOLD*100:.0f}%"

    else:
        return None, 0, "Aucune action (dans la marge d'1 part)"


def execute_order(side, quantity, dry_run=True):
    """Execute order via bourso-cli."""
    from src.bourso.prepare import PEA_ACCOUNT_ID, SYMBOLS, _run_cli_raw

    action = f"{'ACHAT' if side == 'buy' else 'VENTE'} {quantity}x PUST"

    if dry_run:
        print(f"  [DRY-RUN] {action}")
        return {"status": "dry-run", "side": side, "quantity": quantity}

    print(f"  [EXECUTE] {action}")
    stdout, stderr, rc = _run_cli_raw(
        "trade", "order", "new",
        "--side", side,
        "--account", PEA_ACCOUNT_ID,
        "--symbol", SYMBOLS["PUST"],
        "--quantity", str(quantity),
    )

    output = (stdout + stderr).strip()
    if rc != 0:
        print(f"  [ERREUR] bourso-cli code {rc}: {output}")
        return {"status": "error", "side": side, "quantity": quantity, "error": output}

    print(f"  [OK] {output}")
    return {"status": "executed", "side": side, "quantity": quantity, "output": output}


def log_trade(record):
    """Append trade record to JSONL log."""
    record["timestamp"] = datetime.now().isoformat()
    with open(TRADE_LOG, "a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"  Log → {TRADE_LOG}")


def main():
    parser = argparse.ArgumentParser(description="Execution PEA matin — signal du backtest soir")
    parser.add_argument("--execute", action="store_true", help="Executer les ordres (defaut: dry-run)")
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"PEA PUST (Nasdaq x1) — {date.today()}")
    print(f"Mode: {'LIVE' if args.execute else 'DRY-RUN'}")
    print(f"{'='*60}")

    # 1. Load signal from evening backtest
    signal = load_signal()
    if signal is None:
        sys.exit(1)

    # 2. Emergency override
    target_alloc = signal["allocation"]
    if is_emergency_off():
        target_alloc = 0.0

    # 3. Get PEA state (cash, positions)
    try:
        pea_state = get_pea_state()
    except Exception as e:
        print(f"[ERREUR] Impossible de lire le PEA: {e}")
        sys.exit(1)

    if pea_state["equity"] <= 0:
        print("[ERREUR] PEA vide (equity=0)")
        sys.exit(1)

    # 4. Compute orders
    side, quantity, reason = compute_orders(target_alloc, pea_state)
    print(f"\nDecision: {reason}")

    # 5. Execute
    result = None
    if side and quantity > 0:
        result = execute_order(side, quantity, dry_run=not args.execute)

    # 6. Log
    log_trade({
        "date": str(date.today()),
        "signal_date": signal["date"],
        "probability": signal["probability"],
        "target_alloc": target_alloc,
        "side": side,
        "quantity": quantity,
        "pust_price": pea_state["pust_price"],
        "equity": pea_state["equity"],
        "current_shares": pea_state["pust_shares"],
        "executed": args.execute,
        "reason": reason,
        "result": result,
    })

    print("\nTermine.")


if __name__ == "__main__":
    main()
