"""Lecture (dry-run) de l'etat du compte PEA via bourso-cli, sans passer d'ordre.

S'appuie sur `bourso-cli trade summary` (patch maison exposant get_trading_summary
de la lib bourso_api) pour recuperer cash, titres et positions. Remplace
l'ancien `trade prepare` (qui venait d'un fork perdu et n'existe pas en amont).
"""

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


def _extract_json_array(text):
    """Extract the first JSON array from text (skip log lines)."""
    decoder = json.JSONDecoder()
    start = 0
    while True:
        idx = text.find("[", start)
        if idx == -1:
            return None
        try:
            obj, _ = decoder.raw_decode(text[idx:])
            if isinstance(obj, list):
                return obj
        except json.JSONDecodeError:
            pass
        start = idx + 1


def _val(d, key):
    """Extrait la valeur d'un champ SummaryValue ({value, decimals, currency}) ou brut."""
    x = d.get(key)
    return x.get("value") if isinstance(x, dict) else x


def get_account_summary(account_id):
    """Lit l'etat du compte trading via `bourso-cli trade summary` (read-only).

    Retourne:
      {"account": {name, type, cash, stocks, equity, balance},
       "positions": {symbol: {quantity, last_price, label, amount, currency}}}
    """
    stdout, stderr, rc = _run_cli_raw("trade", "summary", "--account", account_id)
    combined = stdout + stderr
    data = _extract_json_array(combined)
    if data is None:
        raise RuntimeError(f"Could not extract JSON from bourso-cli output:\n{combined}")

    account = {}
    positions = {}
    for item in data:
        if item.get("id") == "account" and item.get("account"):
            a = item["account"]
            account = {
                "name": a.get("name"),
                "type": a.get("typeCategory"),
                "cash": _val(a, "cash"),
                "stocks": _val(a, "valuation"),
                "equity": _val(a, "total"),
                "balance": _val(a, "balance"),
            }
        elif item.get("id") == "positions" and item.get("positions"):
            for p in item["positions"]:
                positions[p.get("symbol")] = {
                    "quantity": _val(p, "quantity"),
                    "last_price": _val(p, "last"),
                    "label": p.get("label"),
                    "amount": _val(p, "amount"),
                    "currency": (p.get("last") or {}).get("currency", "EUR"),
                }
    return {"account": account, "positions": positions}


def prepare_order(account_id, symbol):
    """Etat du compte + cours d'un symbole (dry-run, aucun ordre).

    Reconstruit a partir de `trade summary`. Si le symbole n'est pas detenu
    (0 part, donc absent des positions), le cours est recupere via le scraping
    HTTP Boursorama (`quote.py`, car le `quote` du CLI est en 410). Conserve la
    forme de retour attendue par les consommateurs (real_bourso, notify, etc.).
    """
    summary = get_account_summary(account_id)
    acct = summary["account"]
    pos = summary["positions"].get(symbol, {})

    qty = pos.get("quantity")
    qty = int(qty) if qty is not None else 0
    price = pos.get("last_price")
    label = pos.get("label")
    currency = pos.get("currency", "EUR")

    if price is None:
        # symbole non detenu -> cours via scraping HTTP (le CLI `quote` renvoie 410)
        from src.bourso.quote import get_quote
        q = get_quote(symbol)
        price = q.get("last")
        label = label or q.get("name")

    return {
        "resource_id": None,
        "cash": acct.get("cash"),
        "quantity_held": qty,
        "symbol": {
            "id": symbol,
            "label": label,
            "isin": None,
            "last_price": price,
            "currency": currency,
            "exchange": None,
            "is_tracker": True,
        },
        "account": {
            "name": acct.get("name"),
            "type": acct.get("type"),
            "balance": acct.get("balance"),
            "cash": acct.get("cash"),
            "stocks": acct.get("stocks"),
            "iban": None,
            "fees_profile": None,
        },
        "prefill": {},
        "order_config": {"buy_types": [], "sell_types": [], "min_expiry": None, "max_expiry": None},
    }


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
