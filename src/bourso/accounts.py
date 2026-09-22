"""Comptes BoursoBank multiples — jusqu'a MAX_ACCOUNTS logins, declares dans .env :

    BOURSO_ID_1="..."       identifiant client BoursoBank
    BOURSO_CODE_1="..."     mot de passe (code a 8 chiffres)
    BOURSO_MAIL_1="..."     adresse qui recoit les mails de CE compte (connexion, allocation)
    BOURSO_ID_2=""          ... slot 2 vide -> ignore (le compte n'est pas gere)
    ...

Un slot n'est GERE que si `BOURSO_ID_n` ET `BOURSO_CODE_n` sont renseignes. Le mail est
optionnel (repli sur la MAILING_LIST de notify.py). L'ancienne paire `BOURSO_ID` /
`BOURSO_CODE` (mono-compte) reste acceptee comme slot 1 si `BOURSO_ID_1` est absent.

Chaque login Bourso a son propre PEA (id hexa de 32 caracteres) : il est decouvert une
fois via `bourso-cli accounts --trading` (compte de trading dont le nom contient "PEA",
hors PEA-PME) puis mis en cache dans `logs/accounts.json` (relu par la webapp pour
afficher le nom du compte). Surcharge possible avec `BOURSO_PEA_ID_n` dans .env.

Le meme signal (allocation) est REPLIQUE sur tous les comptes geres : chaque compte est
lu, decide et execute independamment (bande de non-action sur SA propre allocation
reelle), avec un mail par compte a son adresse.
"""

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
ENV_PATH = ROOT / ".env"
ACCOUNTS_CACHE = ROOT / "logs" / "accounts.json"

MAX_ACCOUNTS = 4


@dataclass
class BoursoAccount:
    slot: int
    bourso_id: str
    bourso_code: str
    mail: str = ""
    name: str = ""              # nom du PEA (decouvert), ex "PEA DESCAMPS"
    pea_account_id: str = ""    # id hexa du PEA (decouvert / cache / .env)
    extra: dict = field(default_factory=dict)

    @property
    def creds(self):
        """Tuple (id, code) pour prepare._run_cli_raw."""
        return self.bourso_id, self.bourso_code

    @property
    def label(self):
        """Libelle humain : "compte 2 (PEA DUPONT)" — jamais l'identifiant client."""
        return f"compte {self.slot}" + (f" ({self.name})" if self.name else "")

    @property
    def recipients(self):
        """Destinataires des mails de ce compte : son adresse, sinon la MAILING_LIST."""
        if self.mail:
            return [self.mail]
        from src.bourso.notify import MAILING_LIST
        return list(MAILING_LIST)


def _read_dotenv():
    """Lit .env en dict (sans ecraser l'environnement) — repli si dotenv n'a pas ete charge."""
    values = {}
    if ENV_PATH.exists():
        with open(ENV_PATH) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                values[k.strip()] = v.strip().strip('"').strip("'")
    return values


def _env(key, dotenv):
    v = os.environ.get(key)
    if v is None:
        v = dotenv.get(key, "")
    return (v or "").strip().strip('"').strip("'")


def load_accounts(dotenv=None):
    """Liste des comptes GERES (slots 1..MAX_ACCOUNTS avec id ET code renseignes).

    Compat mono-compte : si aucun `BOURSO_ID_1` mais `BOURSO_ID`/`BOURSO_CODE` existent,
    ils forment le slot 1. Les slots vides sont ignores silencieusement.
    """
    dotenv = _read_dotenv() if dotenv is None else dotenv
    cache = _load_cache()
    accounts = []
    for n in range(1, MAX_ACCOUNTS + 1):
        bid = _env(f"BOURSO_ID_{n}", dotenv)
        code = _env(f"BOURSO_CODE_{n}", dotenv)
        if n == 1 and not bid and not code:
            bid = _env("BOURSO_ID", dotenv)
            code = _env("BOURSO_CODE", dotenv)
        if not bid or not code:
            continue
        cached = cache.get(str(n), {})
        acc = BoursoAccount(
            slot=n, bourso_id=bid, bourso_code=code,
            mail=_env(f"BOURSO_MAIL_{n}", dotenv),
            name=cached.get("name", ""),
            pea_account_id=_env(f"BOURSO_PEA_ID_{n}", dotenv) or cached.get("pea_account_id", ""),
        )
        accounts.append(acc)
    return accounts


def get_account(slot):
    """Le compte gere du slot `slot`, ou None."""
    for acc in load_accounts():
        if acc.slot == slot:
            return acc
    return None


# ── Decouverte du PEA de chaque login ─────────────────────

_ACCOUNT_BLOCK = re.compile(r"Account\s*\{(.*?)\}", re.DOTALL)


def parse_accounts_output(text):
    """Parse la sortie Debug (`{:#?}`) de `bourso-cli accounts --trading` en liste de dicts.

    Format Rust :  Account { id: "abc...", name: "PEA DESCAMPS", balance: 5439364,
                             bank_name: "BoursoBank", kind: Trading, }
    (balance en centimes). Robuste aux lignes de log intercalees.
    """
    out = []
    for m in _ACCOUNT_BLOCK.finditer(text):
        block = m.group(1)
        d = {}
        for key in ("id", "name", "bank_name"):
            mm = re.search(rf'\b{key}:\s*"([^"]*)"', block)
            d[key] = mm.group(1) if mm else ""
        mm = re.search(r"\bbalance:\s*(-?\d+)", block)
        d["balance"] = int(mm.group(1)) / 100.0 if mm else None
        mm = re.search(r"\bkind:\s*(\w+)", block)
        d["kind"] = mm.group(1) if mm else ""
        if d["id"]:
            out.append(d)
    return out


def pick_pea(accounts):
    """Le compte PEA parmi une liste de comptes de trading (nom contenant "PEA", hors PEA-PME)."""
    peas = [a for a in accounts if re.search(r"\bPEA\b", a.get("name", ""), re.I)
            and not re.search(r"PEA[\s-]*PME", a.get("name", ""), re.I)]
    return peas[0] if peas else None


def list_trading_accounts(account):
    """Comptes de trading du login `account` via `bourso-cli accounts --trading` (lecture seule)."""
    from src.bourso.prepare import _run_cli_raw
    stdout, stderr, rc = _run_cli_raw("accounts", "--trading", "1", creds=account.creds)
    found = parse_accounts_output(stdout + stderr)
    if rc != 0 and not found:
        raise RuntimeError(f"bourso-cli accounts (code {rc}): {(stdout + stderr).strip()[-300:]}")
    return found


def resolve_pea(account, force=False):
    """Renseigne `account.pea_account_id` / `account.name` (cache, sinon decouverte) et
    met a jour `logs/accounts.json`. Leve RuntimeError si aucun PEA n'est trouve."""
    if account.pea_account_id and account.name and not force:
        return account
    trading = list_trading_accounts(account)
    pea = None
    if account.pea_account_id:           # id fixe par .env -> on ne fait que recuperer le nom
        pea = next((a for a in trading if a["id"] == account.pea_account_id), None)
    if pea is None:
        pea = pick_pea(trading)
    if pea is None:
        names = ", ".join(a.get("name", "?") for a in trading) or "aucun"
        raise RuntimeError(f"aucun PEA trouve pour le {account.label} (comptes trading: {names})")
    account.pea_account_id = pea["id"]
    account.name = pea["name"]
    _update_cache(account)
    return account


# ── Cache logs/accounts.json ──────────────────────────────

def _load_cache():
    if not ACCOUNTS_CACHE.exists():
        return {}
    try:
        return json.loads(ACCOUNTS_CACHE.read_text())
    except Exception:  # noqa: BLE001
        return {}


def _update_cache(account):
    cache = _load_cache()
    cache[str(account.slot)] = {
        "pea_account_id": account.pea_account_id,
        "name": account.name,
        "mail": account.mail,
        "discovered_at": datetime.now().isoformat(timespec="seconds"),
    }
    ACCOUNTS_CACHE.parent.mkdir(exist_ok=True)
    ACCOUNTS_CACHE.write_text(json.dumps(cache, indent=2))


def load_cached_accounts():
    """Cache {slot(int): {pea_account_id, name, mail, discovered_at}} — pour la webapp
    (qui n'a pas acces au .env Bourso) ; ne contient jamais d'identifiant client."""
    return {int(k): v for k, v in _load_cache().items() if k.isdigit()}


if __name__ == "__main__":
    accs = load_accounts()
    if not accs:
        print("Aucun compte gere (BOURSO_ID_n / BOURSO_CODE_n vides).")
    for acc in accs:
        try:
            resolve_pea(acc)
            print(f"slot {acc.slot}: {acc.name} (PEA {acc.pea_account_id[:8]}...)  mail={acc.mail or '(MAILING_LIST)'}")
        except Exception as e:  # noqa: BLE001
            print(f"slot {acc.slot}: ERREUR {e}")
