"""Ordre manuel NON interactif sur un compte Bourso gere (slot 1..4), en ordre LIMITE.

Pensé pour un compte jamais connecte depuis cette machine : la premiere connexion
declenche un MFA BoursoBank (validation dans l'application mobile). L'etape de
connexion/decouverte du PEA est lancee SANS capture de sortie et avec 300 s de delai,
pour voir le message "open the BoursoBank app" et valider sur le telephone.

    python myfiles/manual_order.py --account 3 --symbol LQQ --side buy --quantity 24
    python myfiles/manual_order.py --account 3 --symbol LQQ --side buy --quantity 24 --execute

Sans --execute : connexion + etat du compte + apercu de l'ordre, rien n'est envoye.
La limite par defaut = dernier cours x (1 + 3%) a l'achat (x (1 - 3%) a la vente), comme
real_bourso ; --limit la fixe explicitement. L'ordre n'est PAS journalise dans
logs/trades.jsonl (le run du matin repartira de l'etat reel du compte).
Bourso : achat minimum 200 EUR (code 5110) ; le cash reserve = quantite x limite + ~1.5%.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # racine du repo

from src.bourso.accounts import get_account, resolve_pea
from src.bourso.prepare import SYMBOLS, prepare_order
import src.bourso.prepare as prepare

LOGIN_TIMEOUT = 300    # MFA : bourso-cli attend jusqu'a 300 s la validation dans l'app
ORDER_TIMEOUT = 120
TOLERANCE_PCT = 3.0


def _patch_cli_timeout(seconds):
    """_run_cli_raw a un timeout fixe de 30 s : trop court pour un MFA."""
    orig = subprocess.run

    def run(cmd, **kw):
        kw["timeout"] = seconds
        return orig(cmd, **kw)
    prepare.subprocess.run = run


def login_and_discover(account):
    """Connexion visible (MFA) : `accounts --trading` sans capture, puis cache du PEA."""
    bid, code = account.creds
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write(json.dumps({"clientId": bid, "password": code}))
        path = f.name
    try:
        print(f"Connexion du slot {account.slot} (validez le MFA dans l'app BoursoBank si demande)...")
        rc = subprocess.run(["bourso-cli", "--credentials", path, "accounts", "--trading", "1"],
                            timeout=LOGIN_TIMEOUT).returncode
    finally:
        os.unlink(path)
    if rc != 0:
        sys.exit(f"ERREUR: connexion du slot {account.slot} echouee (code {rc})")
    _patch_cli_timeout(LOGIN_TIMEOUT)
    resolve_pea(account, force=True)
    print(f"PEA du slot {account.slot} : {account.name} ({account.pea_account_id[:8]}...)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--account", type=int, required=True, help="slot Bourso (1..4)")
    p.add_argument("--symbol", choices=list(SYMBOLS), default="LQQ")
    p.add_argument("--side", choices=["buy", "sell"], default="buy")
    p.add_argument("--quantity", type=int, required=True)
    p.add_argument("--limit", type=float, default=None, help="prix limite (defaut : cours +/-3%%)")
    p.add_argument("--execute", action="store_true", help="envoyer l'ordre (sinon apercu)")
    a = p.parse_args()

    account = get_account(a.account)
    if account is None:
        sys.exit(f"ERREUR: slot {a.account} non gere (BOURSO_ID_{a.account}/BOURSO_CODE_{a.account} vides)")
    login_and_discover(account)

    data = prepare_order(account.pea_account_id, SYMBOLS[a.symbol], creds=account.creds)
    last = float(data["symbol"]["last_price"])
    cash = float(data["account"]["cash"])
    held = int(data["quantity_held"])
    limit = a.limit if a.limit else round(last * (1 + TOLERANCE_PCT / 100 * (1 if a.side == "buy" else -1)), 3)
    value = a.quantity * last
    reserved = a.quantity * limit * 1.015
    print(f"\n  {data['symbol']['label']}  cours {last:.3f} EUR")
    print(f"  Compte {account.label} : especes {cash:.2f} EUR, {held} parts detenues")
    print(f"  Ordre : {a.side.upper()} {a.quantity} x {a.symbol}  limite {limit:.3f}  "
          f"(~{value:.2f} EUR au cours, frais ~{value*0.005:.2f} EUR)")
    if a.side == "buy":
        if value < 200:
            sys.exit(f"ERREUR: {value:.2f} EUR < minimum Bourso 200 EUR (code 5110)")
        if reserved > cash:
            sys.exit(f"ERREUR: cash reserve ~{reserved:.2f} EUR (limite + 1.5%) > especes {cash:.2f} EUR")
        print(f"  Cash reserve par Bourso ~{reserved:.2f} EUR, reste ~{cash - value*1.005:.2f} EUR apres execution")
    elif a.quantity > held:
        sys.exit(f"ERREUR: vente de {a.quantity} parts, {held} detenues")

    if not a.execute:
        print("\n[APERCU] rien n'est envoye (ajoutez --execute).")
        return
    print("\nEnvoi de l'ordre...")
    _patch_cli_timeout(ORDER_TIMEOUT)
    out, err, rc = prepare._run_cli_raw(
        "trade", "order", "new", "--side", a.side, "--account", account.pea_account_id,
        "--symbol", SYMBOLS[a.symbol], "--quantity", str(a.quantity),
        "--order-type", "LIM", "--limit", str(limit), "--validity", date.today().isoformat(),
        creds=account.creds)
    print(out + err)
    sys.exit(0 if rc == 0 else f"ERREUR: bourso-cli code {rc}")


if __name__ == "__main__":
    main()
