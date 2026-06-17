"""Prepare (dry-run) a trade order via bourso-cli to check price, fees, cash."""

import json
import os
import re
import subprocess
import tempfile


def _get_credentials():
    bourso_id = os.environ.get("BOURSO_ID")
    bourso_code = os.environ.get("BOURSO_CODE")
    if bourso_id and bourso_code:
        return bourso_id, bourso_code
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


def _run_cli_raw(*args):
    """Run bourso-cli, return stdout+stderr combined."""
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
        return result.stdout, result.stderr, result.returncode
    finally:
        os.unlink(creds_path)


def _extract_json(text):
    """Extract first JSON object from text (skip log lines)."""
    # Try "Response: {json}" pattern first (error message from Rust CLI)
    m = re.search(r'Response: ({.*)', text, re.DOTALL)
    raw = m.group(1) if m else text
    # Find first { or [ and try progressively shorter substrings
    for start_char in ['{', '[']:
        idx = raw.find(start_char)
        if idx == -1:
            continue
        s = raw[idx:]
        decoder = json.JSONDecoder()
        try:
            obj, _ = decoder.raw_decode(s)
            return obj
        except json.JSONDecodeError:
            continue
    return None


def prepare_order(account_id, symbol):
    """Prepare an order (dry-run) and return parsed data.

    The Rust CLI may fail to deserialize the response (null fields),
    but the raw JSON is in the error message — we parse it from there.

    Returns dict with keys: cash, last_price, symbol_info, account_info, order_types, etc.
    """
    stdout, stderr, rc = _run_cli_raw("trade", "prepare", "--account", account_id, "--symbol", symbol)

    combined = stdout + stderr
    data = _extract_json(combined)
    if not data:
        raise RuntimeError(f"Could not extract JSON from bourso-cli output:\n{combined}")

    result = {
        "resource_id": data.get("resourceId"),
        "cash": data.get("position", {}).get("cash"),
        "quantity_held": data.get("position", {}).get("quantity", 0),
    }

    sym = data.get("symbol", {})
    result["symbol"] = {
        "id": sym.get("symbol"),
        "label": sym.get("label"),
        "isin": sym.get("isin"),
        "last_price": sym.get("lastPrice"),
        "currency": sym.get("currency"),
        "exchange": sym.get("exchangeLabel"),
        "is_tracker": sym.get("details", {}).get("tracker", False),
    }

    acct = data.get("account", {})
    details = acct.get("details", {})
    result["account"] = {
        "name": acct.get("name"),
        "type": acct.get("type"),
        "balance": acct.get("balance"),
        "cash": details.get("cash"),
        "stocks": details.get("stocks"),
        "iban": acct.get("iban"),
        "fees_profile": data.get("accountFeesProfile"),
    }

    prefill = data.get("prefillOrderData", {})
    result["prefill"] = {
        "order_type": prefill.get("orderType"),
        "order_amount": prefill.get("orderAmount"),
        "order_validity": prefill.get("orderValidity"),
    }

    prep = data.get("prepareOrderData", {})
    result["order_config"] = {
        "buy_types": prep.get("listOrdType", {}).get("b", []),
        "sell_types": prep.get("listOrdType", {}).get("s", []),
        "min_expiry": prep.get("minExpireTm"),
        "max_expiry": prep.get("maxExpireTm"),
    }

    return result


def print_prepare(account_id, symbol):
    """Print prepare order info in a readable format."""
    data = prepare_order(account_id, symbol)

    sym = data["symbol"]
    acct = data["account"]

    print(f"=== Preparation ordre: {sym['label']} ({sym['id']}) ===")
    print(f"ISIN:        {sym['isin']}")
    print(f"Marche:      {sym['exchange']}")
    print(f"Cours:       {sym['last_price']:.4f} {sym['currency']}")
    print(f"Tracker:     {'Oui' if sym['is_tracker'] else 'Non'}")
    print()
    print(f"=== Compte: {acct['name']} ({acct['type']}) ===")
    print(f"Especes:     {acct['cash']:.2f} EUR")
    print(f"Titres:      {acct['stocks']:.2f} EUR")
    print(f"Quantite:    {data['quantity_held']} parts detenues")
    print(f"Profil:      {acct['fees_profile']}")
    print()

    max_qty = int(acct["cash"] // sym["last_price"]) if sym["last_price"] else 0
    print(f"=== Capacite d'achat ===")
    print(f"Avec {acct['cash']:.2f} EUR -> max {max_qty} parts @ {sym['last_price']:.4f}")
    print(f"Types ordre: {', '.join(data['order_config']['buy_types'])}")
    print(f"Validite:    {data['order_config']['min_expiry']} -> {data['order_config']['max_expiry']}")


# Constantes comptes (a adapter)
PEA_ACCOUNT_ID = "faab190372918f26c5d2d518fd307d05"
CTO_ACCOUNT_ID = "e0aeafb04e60bdbe140479e499fd79d2"

SYMBOLS = {
    "PUST": "1rTPUST",   # Amundi PEA Nasdaq-100 x1 (PEA)
    "LQQ": "1rTLQQ",     # Amundi Nasdaq-100 2x Leveraged (PEA)
    "CW8": "1rTCW8",     # Amundi MSCI World
    "PE500": "1rTPE500",  # Amundi PEA S&P 500
}


if __name__ == "__main__":
    for name in ["PUST", "LQQ"]:
        print_prepare(PEA_ACCOUNT_ID, SYMBOLS[name])
        print()
