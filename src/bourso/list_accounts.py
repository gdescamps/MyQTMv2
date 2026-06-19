"""List BoursoBank PEA and CTO balances via bourso-cli prepare (more accurate)."""

from src.bourso.prepare import prepare_order, PEA_ACCOUNT_ID, CTO_ACCOUNT_ID


ACCOUNTS = [
    ("PEA", PEA_ACCOUNT_ID),
    ("CTO", CTO_ACCOUNT_ID),
]


def get_account_state(account_id, symbol_id="1rTPUST"):
    """Get account state via prepare (cash, stocks, positions)."""
    data = prepare_order(account_id, symbol_id)
    acct = data["account"]
    sym = data["symbol"]
    return {
        "name": acct["name"],
        "type": acct["type"],
        "cash": acct["cash"],
        "stocks": acct["stocks"],
        "total": acct["cash"] + acct["stocks"],
        "pust_shares": data["quantity_held"],
        "pust_price": sym["last_price"],
        "pust_value": data["quantity_held"] * sym["last_price"],
    }


def print_balances():
    """Print PEA and CTO balances."""
    total = 0
    print(f"{'Compte':<30} {'Type':<6} {'Especes':>10} {'Titres':>10} {'Total':>10}")
    print("-" * 70)

    for label, acct_id in ACCOUNTS:
        try:
            s = get_account_state(acct_id)
            print(f"{s['name']:<30} {s['type']:<6} {s['cash']:>10.2f} {s['stocks']:>10.2f} {s['total']:>10.2f}")
            if s["pust_shares"] > 0:
                print(f"  PUST: {s['pust_shares']} parts @ {s['pust_price']:.2f} EUR = {s['pust_value']:.2f} EUR")
            total += s["total"]
        except Exception as e:
            print(f"{label:<30} {'?':<6} {'erreur':>10}")
            print(f"  {e}")

    print("-" * 70)
    print(f"{'TOTAL':<30} {'':6} {'':10} {'':10} {total:>10.2f}")


if __name__ == "__main__":
    print_balances()
