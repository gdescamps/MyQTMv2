"""Liste les PEA de tous les comptes Bourso geres (slots du .env) et leurs soldes.

Usage: python -m src.bourso.list_accounts
"""

from src.bourso.accounts import load_accounts, resolve_pea
from src.bourso.prepare import prepare_order, SYMBOLS


def get_account_state(account, symbol_id=SYMBOLS["PUST"]):
    """Etat du PEA d'un compte (cash, titres, position sur `symbol_id`)."""
    resolve_pea(account)
    data = prepare_order(account.pea_account_id, symbol_id, creds=account.creds)
    acct = data["account"]
    sym = data["symbol"]
    return {
        "slot": account.slot,
        "name": acct["name"],
        "type": acct["type"],
        "cash": acct["cash"],
        "stocks": acct["stocks"],
        "total": acct["cash"] + acct["stocks"],
        "shares": data["quantity_held"],
        "price": sym["last_price"],
        "value": data["quantity_held"] * sym["last_price"],
    }


def print_balances(symbol="PUST"):
    """Affiche le PEA de chaque compte gere."""
    accounts = load_accounts()
    if not accounts:
        print("Aucun compte gere (BOURSO_ID_n / BOURSO_CODE_n vides dans .env).")
        return
    total = 0
    print(f"{'Slot':<5} {'Compte':<28} {'Type':<8} {'Especes':>10} {'Titres':>10} {'Total':>10}")
    print("-" * 76)
    for acc in accounts:
        try:
            s = get_account_state(acc, SYMBOLS[symbol])
            print(f"{s['slot']:<5} {s['name']:<28} {s['type']:<8} {s['cash']:>10.2f} {s['stocks']:>10.2f} {s['total']:>10.2f}")
            if s["shares"] > 0:
                print(f"      {symbol}: {s['shares']} parts @ {s['price']:.2f} EUR = {s['value']:.2f} EUR")
            total += s["total"]
        except Exception as e:  # noqa: BLE001
            print(f"{acc.slot:<5} {acc.label:<28} {'?':<8} {'erreur':>10}")
            print(f"      {e}")
    print("-" * 76)
    print(f"{'TOTAL':<5} {'':28} {'':8} {'':10} {'':10} {total:>10.2f}")


if __name__ == "__main__":
    import sys
    print_balances(sys.argv[1].upper() if len(sys.argv) > 1 else "PUST")
