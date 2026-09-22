"""Execute a buy/sell order on BoursoBank CTO or PEA via bourso-cli."""

import argparse
import sys

from src.bourso.prepare import (
    PEA_ACCOUNT_ID, CTO_ACCOUNT_ID, SYMBOLS,
    prepare_order, _run_cli_raw,
)


ACCOUNTS = {
    "pea": PEA_ACCOUNT_ID,
    "cto": CTO_ACCOUNT_ID,
}


def execute_order(account_key, symbol_name, side, quantity, slot=1):
    """Execute an order after showing prepare info and asking confirmation.

    account_key: 'pea' or 'cto'
    symbol_name: key in SYMBOLS (e.g. 'PUST', 'LQQ')
    side: 'buy' or 'sell'
    quantity: number of shares
    slot: compte Bourso (1..4, cf. accounts.py) ; le PEA du slot est decouvert.
          'cto' n'est connu que pour le slot 1 (CTO_ACCOUNT_ID).
    """
    from src.bourso.accounts import get_account, resolve_pea
    account = get_account(slot)
    if account is None:
        print(f"ERREUR: slot {slot} non gere (BOURSO_ID_{slot}/BOURSO_CODE_{slot} vides)")
        return False
    if account_key == "pea":
        resolve_pea(account)
        account_id = account.pea_account_id
    elif slot == 1:
        account_id = ACCOUNTS[account_key]
    else:
        print("ERREUR: le CTO n'est connu que pour le slot 1")
        return False
    symbol_id = SYMBOLS[symbol_name]

    # 1. Prepare (dry-run) to show current state
    print(f"Preparation de l'ordre {side.upper()} {quantity}x {symbol_name} sur {account_key.upper()} "
          f"({account.label})...")
    print()
    data = prepare_order(account_id, symbol_id, creds=account.creds)

    sym = data["symbol"]
    acct = data["account"]
    price = sym["last_price"]

    print(f"  ETF:       {sym['label']} ({sym['isin']})")
    print(f"  Marche:    {sym['exchange']}")
    print(f"  Cours:     {price:.4f} {sym['currency']}")
    print(f"  Compte:    {acct['name']} ({acct['type']})")
    print(f"  Especes:   {acct['cash']:.2f} EUR")
    print(f"  Detenues:  {data['quantity_held']} parts")
    print()

    cost = price * quantity
    if side == "buy":
        if cost > acct["cash"]:
            print(f"ERREUR: cout estime {cost:.2f} EUR > especes {acct['cash']:.2f} EUR")
            return False
        print(f"  Cout estime: {cost:.2f} EUR (hors frais)")
        print(f"  Reste apres: ~{acct['cash'] - cost:.2f} EUR")
    else:
        if quantity > data["quantity_held"]:
            print(f"ERREUR: vente de {quantity} parts mais seulement {data['quantity_held']} detenues")
            return False
        print(f"  Vente estimee: {cost:.2f} EUR (hors frais)")

    print()
    confirm = input(f"Confirmer {side.upper()} {quantity}x {symbol_name} ? (oui/non) > ").strip().lower()
    if confirm not in ("oui", "o", "yes", "y"):
        print("Ordre annule.")
        return False

    # 2. Execute
    print()
    print("Envoi de l'ordre...")
    stdout, stderr, rc = _run_cli_raw(
        "trade", "order", "new",
        "--side", side,
        "--account", account_id,
        "--symbol", symbol_id,
        "--quantity", str(quantity),
        creds=account.creds,
    )

    output = stdout + stderr
    print(output)

    if rc != 0:
        print(f"ERREUR: bourso-cli retourne code {rc}")
        return False

    print("Ordre envoye avec succes.")
    return True


def main():
    parser = argparse.ArgumentParser(description="Executer un ordre BoursoBank")
    parser.add_argument("account_key", metavar="account", choices=["pea", "cto"], help="Compte: pea ou cto")
    parser.add_argument("symbol", choices=list(SYMBOLS.keys()), help=f"ETF: {', '.join(SYMBOLS.keys())}")
    parser.add_argument("side", choices=["buy", "sell"], help="Sens: buy ou sell")
    parser.add_argument("quantity", type=int, help="Nombre de parts")
    parser.add_argument("--account", type=int, default=1, metavar="N",
                        help="Slot du compte Bourso (defaut 1)")

    args = parser.parse_args()

    if args.quantity <= 0:
        print("ERREUR: la quantite doit etre > 0")
        sys.exit(1)

    success = execute_order(args.account_key, args.symbol, args.side, args.quantity, slot=args.account)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
