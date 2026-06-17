"""List BoursoBank account balances via bourso-cli."""

import json
import os
import subprocess
import tempfile


def _get_credentials():
    """Load BOURSO_ID and BOURSO_CODE from environment or .env file."""
    bourso_id = os.environ.get("BOURSO_ID")
    bourso_code = os.environ.get("BOURSO_CODE")
    if bourso_id and bourso_code:
        return bourso_id, bourso_code

    # Try loading from .env
    env_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
    if not os.path.exists(env_path):
        env_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("BOURSO_ID="):
                    bourso_id = line.split("=", 1)[1].strip('"').strip("'")
                elif line.startswith("BOURSO_CODE="):
                    bourso_code = line.split("=", 1)[1].strip('"').strip("'")
    if not bourso_id or not bourso_code:
        raise RuntimeError("BOURSO_ID and BOURSO_CODE must be set in environment or .env")
    return bourso_id, bourso_code


def _run_cli(*args):
    """Run bourso-cli with credentials file, return stdout."""
    bourso_id, bourso_code = _get_credentials()
    creds = json.dumps({"clientId": bourso_id, "password": bourso_code})

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write(creds)
        creds_path = f.name
    try:
        result = subprocess.run(
            ["bourso-cli", "--credentials", creds_path, *args],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(f"bourso-cli failed: {result.stderr}")
        return result.stdout, result.stderr
    finally:
        os.unlink(creds_path)


def list_accounts(kind=None):
    """List accounts. kind: 'trading', 'saving', 'banking', 'loans' or None for all.

    Returns list of dicts with keys: id, name, balance_cents, balance_eur, bank_name, kind.
    """
    args = ["accounts"]
    if kind:
        args.append(f"--{kind}")
        args.append("true")

    stdout, stderr = _run_cli(*args)

    # Parse the Rust debug output: Account { id: "...", name: "...", balance: N, ... }
    import re
    accounts = []
    for m in re.finditer(
        r'Account \{[^}]*id: "(?P<id>[^"]+)".*?name: "(?P<name>[^"]+)".*?'
        r'balance: (?P<balance>-?\d+).*?bank_name: "(?P<bank>[^"]+)".*?'
        r'kind: (?P<kind>\w+)',
        stdout + stderr, re.DOTALL
    ):
        balance_cents = int(m.group("balance"))
        accounts.append({
            "id": m.group("id"),
            "name": m.group("name"),
            "balance_cents": balance_cents,
            "balance_eur": balance_cents / 100,
            "bank_name": m.group("bank"),
            "kind": m.group("kind").lower(),
        })
    return accounts


def get_trading_summary(account_id):
    """Get trading account summary (cash, positions, valuation).

    Returns dict with keys: account (dict), positions (list).
    """
    stdout, stderr = _run_cli("trade", "summary", "--account", account_id)

    # The CLI outputs JSON after the log lines
    json_start = stdout.find("[")
    if json_start == -1:
        # Try stderr (logs go to stderr, JSON to stdout)
        json_start = stderr.find("[")
        if json_start == -1:
            raise RuntimeError(f"No JSON in output.\nstdout: {stdout}\nstderr: {stderr}")
        raw = stderr[json_start:]
    else:
        raw = stdout[json_start:]

    data = json.loads(raw)

    result = {"account": None, "positions": []}
    for item in data:
        if item.get("account"):
            acct = item["account"]
            result["account"] = {
                "name": acct["name"],
                "currency": acct["currency"],
                "activation_date": acct["activationDate"],
                "balance": acct["balance"]["value"],
                "cash": acct["cash"]["value"],
                "valuation": acct["valuation"]["value"],
                "total": acct["total"]["value"],
                "gain_loss": acct["gainLoss"]["value"],
                "gain_loss_pct": acct["gainLossPercent"]["value"],
                "contribution": acct["contribution"],
            }
        if item.get("positions"):
            for pos in item["positions"]:
                result["positions"].append({
                    "symbol": pos["symbol"],
                    "label": pos["label"],
                    "quantity": pos["quantity"]["value"],
                    "buying_price": pos["buyingPrice"]["value"],
                    "last_price": pos["last"]["value"],
                    "amount": pos["amount"]["value"],
                    "gain_loss": pos["gainLoss"]["value"],
                    "gain_loss_pct": pos["gainLossPercent"]["value"],
                })
    return result


def print_balances():
    """Print all account balances in a readable format."""
    accounts = list_accounts()
    if not accounts:
        print("Aucun compte trouve.")
        return

    print(f"{'Compte':<35} {'Type':<10} {'Solde':>12}")
    print("-" * 60)
    total = 0
    for a in accounts:
        print(f"{a['name']:<35} {a['kind']:<10} {a['balance_eur']:>10.2f} EUR")
        total += a['balance_eur']
    print("-" * 60)
    print(f"{'TOTAL':<35} {'':10} {total:>10.2f} EUR")

    # Show trading details
    trading = [a for a in accounts if a["kind"] == "trading"]
    for t in trading:
        print(f"\n--- {t['name']} (detail) ---")
        try:
            summary = get_trading_summary(t["id"])
            acct = summary["account"]
            if acct:
                print(f"  Especes:    {acct['cash']:>10.2f} EUR")
                print(f"  Titres:     {acct['valuation']:>10.2f} EUR")
                print(f"  Total:      {acct['total']:>10.2f} EUR")
                print(f"  +/- value:  {acct['gain_loss']:>10.2f} EUR ({acct['gain_loss_pct']:.2f}%)")
                print(f"  Apports:    {acct['contribution']:>10.2f} EUR")
            for pos in summary["positions"]:
                print(f"  {pos['label'][:30]:<30} x{pos['quantity']:.0f}  "
                      f"@ {pos['last_price']:.2f}  = {pos['amount']:.2f} EUR  "
                      f"({pos['gain_loss_pct']:+.2f}%)")
        except Exception as e:
            print(f"  Erreur detail: {e}")


if __name__ == "__main__":
    print_balances()
