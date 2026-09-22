"""
Morning PEA execution on BoursoBank — reads signal from evening backtest.

Workflow:
  22:30  cron backtest (run.py) → outputs/qqq_strategy/signal.json
  09:05  this script → reads signal, checks PEA, executes PUST at Euronext open

Instrument (switch): PUST (Nasdaq x1, defaut prod) ou LQQ (Nasdaq x2 leverage).
Bascule via l'env TRADE_INSTRUMENT=LQQ. Le SIGNAL (allocation) est identique ;
seul l'instrument tradé change : LQQ donne 2x l'expo pour la meme fraction de
capital (= la variante "strategie x2.0" du backtest). x1 reste en prod par defaut.

Multicompte : le meme signal est REPLIQUE sur chaque compte Bourso gere (slots
BOURSO_ID_n/CODE_n/MAIL_n du .env, cf. src/bourso/accounts.py). Chaque compte est lu,
decide (bande de non-action sur SA propre allocation reelle) et execute a son tour ;
un mail par compte (connexion + allocation + action) part a l'adresse du compte.
Un compte injoignable n'empeche pas les autres de s'executer.

Usage:
  python -m src.real_bourso                  # dry-run (instrument = $TRADE_INSTRUMENT ou PUST)
  python -m src.real_bourso --execute        # live execution
  python -m src.real_bourso --account 2      # un seul slot
  TRADE_INSTRUMENT=LQQ python -m src.real_bourso   # dry-run en x2 (LQQ)
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SIGNAL_PATH = ROOT / "outputs" / "qqq_strategy" / "signal.json"
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
TRADE_LOG = LOG_DIR / "trades.jsonl"
OVERRIDE_FILE = LOG_DIR / "emergency_off.json"

# ── Switch d'instrument : x1 PUST (prod) <-> x2 LQQ ───────────────────────
# Le signal (allocation) ne change pas : seul l'instrument tradé change. LQQ
# (Amundi Nasdaq-100 x2 Leveraged, PEA) donne 2x l'expo pour la meme fraction
# de capital. Bascule via l'env TRADE_INSTRUMENT=LQQ ; defaut PUST (x1 en prod).
# ATTENTION a la bascule : liquider d'abord l'ancien instrument (sinon on
# detient PUST ET LQQ) -- le script ne trade que l'instrument actif.
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except Exception:
    pass
INSTRUMENTS = {
    "PUST": {"leverage": 1.0, "label": "Nasdaq x1"},
    "LQQ": {"leverage": 2.0, "label": "Nasdaq x2 (leverage)"},
}
INSTRUMENT = os.environ.get("TRADE_INSTRUMENT", "PUST").strip().upper()
if INSTRUMENT not in INSTRUMENTS:
    print(f"[WARN] TRADE_INSTRUMENT='{INSTRUMENT}' inconnu -> PUST (x1)")
    INSTRUMENT = "PUST"
LEVERAGE = INSTRUMENTS[INSTRUMENT]["leverage"]

# Bande de non-action ASYMETRIQUE, source unique = la strategie backtestee
# (strategy.py). En espace POIDS de portefeuille (= ce que compute_orders manipule),
# le seuil vaut BUY_/SELL_THR_ALLOC quel que soit l'instrument : pour LQQ l'expo = 2x
# le poids, mais le seuil d'expo du backtest = seuil_alloc . levier, donc en poids on
# retombe sur seuil_alloc. => achat si delta >= 0.25 (gratuit), vente si delta <= -0.50.
from src.risk_off_strategy.strategy import BUY_THR_ALLOC, SELL_THR_ALLOC
# Signal must be fresh enough, but tolerate weekend/holiday gaps so Monday (and
# post-long-weekend) mornings still execute. The backtest runs each weekday
# evening, so the worst legitimate gap is a Monday morning reading Friday's
# signal (~58h), or a long holiday weekend (Thu eve → Tue morning, ~82h).
# Anything older than ~3.75 days means the evening backtest stopped → refuse.
MAX_SIGNAL_AGE_HOURS = 90
MAX_RETRIES = 5
INITIAL_WAIT = 60  # seconds
MAX_WAIT = 900  # 15 min max between retries

# Detection de split / anomalie de prix : si le prix de l'instrument saute d'un
# facteur >= SPLIT_DETECT_FACTOR (x1.5 a la hausse, ou /1.5) vs la derniere seance,
# c'est quasi surement un split (ex: LQQ /200) ou une incoherence d'affichage broker,
# PAS un mouvement de marche (un ETF x2 ne fait pas +/-50% en une seance : coupe-circuits).
# Dans ce cas on NE PREND PAS de position ce jour-la (on evite d'acheter/vendre sur un
# prix fausse le jour du split) et on attend la seance suivante. Le prix de reference par
# instrument est stocke dans LAST_PRICE_FILE (mis a jour a chaque execution LIVE).
SPLIT_DETECT_FACTOR = 1.5
LAST_PRICE_FILE = LOG_DIR / "last_price.json"

# Tolerance de l'ordre LIMITE : la limite = dernier cours ± LIMIT_TOLERANCE_PCT%
# (achat: +, vente: -). Un ordre limite pile au cours ne se remplit pas si le prix
# s'ecarte a l'ouverture (cf. incident 07-07 : limite sous le marche -> non execute).
# Une tolerance de 3% tampon le gap d'ouverture -> remplissage fiable, tout en
# bornant le prix (contrairement a un ordre au marche non maitrise sur un gap).
LIMIT_TOLERANCE_PCT = 3.0


def retry(fn, label="", hourly_until=None):
    """Retry with exponential backoff (60s, 120s, 240s, 480s, 900s), then hourly."""
    wait = INITIAL_WAIT
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn()
        except Exception as e:
            if attempt == MAX_RETRIES:
                break
            print(f"[RETRY] {label}: tentative {attempt}/{MAX_RETRIES} echouee: {e}")
            print(f"  Prochaine tentative dans {wait}s...")
            time.sleep(wait)
            wait = min(wait * 2, MAX_WAIT)

    if hourly_until is None:
        print(f"[ERREUR] {label}: echec apres {MAX_RETRIES} tentatives")
        raise
    attempt = MAX_RETRIES
    while datetime.now() < hourly_until:
        attempt += 1
        print(f"[RETRY] {label}: tentative {attempt} (horaire), prochaine dans 1h...")
        time.sleep(3600)
        try:
            return fn()
        except Exception as e:
            print(f"[RETRY] {label}: tentative {attempt} echouee: {e}")

    print(f"[ERREUR] {label}: echec, deadline {hourly_until} atteinte")
    raise


def load_signal():
    """Load latest signal from backtest output."""
    if not SIGNAL_PATH.exists():
        print(f"[ERREUR] Signal introuvable: {SIGNAL_PATH}")
        return None

    with open(SIGNAL_PATH) as f:
        signal = json.load(f)

    # Check backtest completed without error
    if signal.get("status") != "ok":
        print(f"[ERREUR] Backtest en erreur (status={signal.get('status', 'missing')})")
        print(f"  Verifier logs/cron_backtest.log")
        return None

    # Check signal freshness
    ts = datetime.fromisoformat(signal["timestamp"])
    age = datetime.now() - ts
    if age > timedelta(hours=MAX_SIGNAL_AGE_HOURS):
        print(f"[ERREUR] Signal trop ancien: {signal['date']} ({age.total_seconds()/3600:.0f}h)")
        print(f"  Le backtest du soir a-t-il tourne? Verifier logs/cron_backtest.log")
        return None

    print(f"Signal du {signal['date']} (status=ok, age: {age.total_seconds()/3600:.1f}h)")
    print(f"  Probabilite: {signal['probability']:.4f}")
    print(f"  Allocation:  {signal['allocation']*100:.0f}%")
    return signal


def _default_account():
    """Slot 1 (compat mono-compte des anciens appels sans `account`)."""
    from src.bourso.accounts import load_accounts
    accs = load_accounts()
    if not accs:
        raise RuntimeError("aucun compte Bourso gere (BOURSO_ID_1/BOURSO_CODE_1 vides)")
    return accs[0]


def get_pea_state(account=None):
    """Get PEA cash and position (instrument actif) via bourso-cli `trade summary`.

    account: BoursoAccount (cf. accounts.py) ; None -> slot 1. L'id du PEA du login est
    decouvert/cache par `resolve_pea` (chaque login Bourso a son propre PEA)."""
    from src.bourso.prepare import prepare_order, SYMBOLS
    from src.bourso.accounts import resolve_pea

    account = account or _default_account()
    resolve_pea(account)
    data = prepare_order(account.pea_account_id, SYMBOLS[INSTRUMENT], creds=account.creds)
    acct = data["account"]
    sym = data["symbol"]

    state = {
        "cash": acct["cash"],
        "stocks": acct["stocks"],
        "equity": acct["cash"] + acct["stocks"],
        "etf_price": sym["last_price"],
        "etf_shares": data["quantity_held"],
        "etf_value": data["quantity_held"] * sym["last_price"],
        "account": account.slot,
        "account_name": acct["name"] or account.name,
    }

    print(f"\n{acct['name']} (compte {account.slot}):")
    print(f"  Especes:  {state['cash']:>10.2f} EUR")
    print(f"  Titres:   {state['stocks']:>10.2f} EUR")
    print(f"  Total:    {state['equity']:>10.2f} EUR")
    print(f"  {INSTRUMENT}:  {state['etf_shares']} parts @ {state['etf_price']:.2f} EUR "
          f"({INSTRUMENTS[INSTRUMENT]['label']})")
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
    price = pea_state["etf_price"]
    current_shares = pea_state["etf_shares"]
    current_value = current_shares * price
    current_alloc = current_value / equity if equity > 0 else 0

    target_value = target_alloc * equity
    delta_value = target_value - current_value
    delta_alloc = target_alloc - current_alloc

    print(f"\nAllocation:")
    print(f"  Actuelle: {current_alloc*100:.1f}% ({current_shares} parts, {current_value:.0f} EUR)")
    print(f"  Cible:    {target_alloc*100:.1f}% ({target_value:.0f} EUR)")
    print(f"  Delta:    {delta_alloc*100:+.1f}% ({delta_value:+.0f} EUR)")

    # FORCE vers/depuis le cash, comme le backtest (clause `force` de simulate_net) :
    # un passage a 0% (garde-fous macro NFCI/IPC, emergency-off) doit TOUJOURS liquider,
    # meme si le delta est sous le seuil de vente ; et une premiere entree depuis le cash
    # total doit s'executer meme sous le seuil d'achat. Sinon une crise laisserait la
    # position ouverte (0.30 < seuil 0.50).
    to_cash = target_alloc <= 1e-9 and current_shares > 0
    from_cash = current_shares == 0 and target_alloc > 0
    if to_cash:
        fee = current_shares * price * 0.005
        return "sell", current_shares, (f"Vendre TOUT {current_shares} parts -> cash "
                                        f"({delta_alloc*100:.1f}%, frais ~{fee:.1f} EUR)")

    # Bande asymetrique (cf. strategy.py) : on ne re-monte l'expo que si delta >=
    # BUY_THR_ALLOC (achats gratuits) et on ne la baisse que si delta <= -SELL_THR_ALLOC
    # (vente 0.5% -> on ne DE-lève que par grands pas). Rebalancement vers la cible
    # (int parts) une fois la bande franchie ; marge d'1 part = plancher pratique.
    if delta_alloc >= BUY_THR_ALLOC or from_cash:
        if delta_value <= price:
            return None, 0, f"Achat dans la marge d'1 part (+{delta_alloc*100:.1f}%)"
        quantity = int(delta_value / price)
        # Check cash available
        if quantity * price > pea_state["cash"]:
            quantity = int(pea_state["cash"] / price)
            if quantity <= 0:
                return None, 0, "Cash insuffisant pour acheter"
        return "buy", quantity, f"Acheter {quantity} parts (+{delta_alloc*100:.1f}%)"

    elif delta_value > price:
        return None, 0, f"Achat ignore: delta +{delta_alloc*100:.1f}% < seuil {BUY_THR_ALLOC*100:.0f}%"

    elif abs(delta_alloc) >= SELL_THR_ALLOC and delta_value < -price:
        quantity = int(abs(delta_value) / price)
        quantity = min(quantity, current_shares)
        fee = quantity * price * 0.005
        return "sell", quantity, f"Vendre {quantity} parts ({delta_alloc*100:.1f}%, frais ~{fee:.1f} EUR)"

    elif delta_value < -price:
        return None, 0, f"Vente ignoree: delta {abs(delta_alloc)*100:.1f}% < seuil {SELL_THR_ALLOC*100:.0f}%"

    else:
        return None, 0, "Aucune action (dans la bande de non-action)"


def execute_order(side, quantity, dry_run=True, tolerance=LIMIT_TOLERANCE_PCT, account=None):
    """Execute order via bourso-cli, sur le PEA du compte `account` (None -> slot 1).

    Ordre LIMITE avec tolerance : la limite = cours ± tolerance% (achat +, vente -),
    ce qui tampon le gap d'ouverture et fiabilise le remplissage tout en bornant le
    prix. tolerance=None -> ordre limite pile au cours (ancien comportement)."""
    from src.bourso.prepare import SYMBOLS, _run_cli_raw

    tol_txt = f" (limite ±{tolerance}%)" if tolerance is not None else ""
    action = f"{'ACHAT' if side == 'buy' else 'VENTE'} {quantity}x {INSTRUMENT}{tol_txt}"

    if dry_run:
        print(f"  [DRY-RUN] {action}")
        return {"status": "dry-run", "side": side, "quantity": quantity,
                "order_type": "LIM", "tolerance": tolerance}

    from src.bourso.accounts import resolve_pea
    account = resolve_pea(account or _default_account())
    print(f"  [EXECUTE] {action} — {account.label}")
    cli_args = [
        "trade", "order", "new",
        "--side", side,
        "--account", account.pea_account_id,
        "--symbol", SYMBOLS[INSTRUMENT],
        "--quantity", str(quantity),
        "--order-type", "LIM",
    ]
    if tolerance is not None:
        cli_args += ["--tolerance", str(tolerance)]
    stdout, stderr, rc = _run_cli_raw(*cli_args, creds=account.creds)

    output = (stdout + stderr).strip()
    if rc != 0:
        print(f"  [ERREUR] bourso-cli code {rc}: {output}")
        return {"status": "error", "side": side, "quantity": quantity,
                "order_type": "LIM", "tolerance": tolerance, "error": output}

    print(f"  [OK] {output}")
    return {"status": "executed", "side": side, "quantity": quantity,
            "order_type": "LIM", "tolerance": tolerance, "output": output}


def log_trade(record):
    """Append trade record to JSONL log."""
    record["timestamp"] = datetime.now().isoformat()
    with open(TRADE_LOG, "a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"  Log → {TRADE_LOG}")


def _price_key(instrument, slot):
    """Cle du prix de reference : par (instrument, compte) — les comptes lisent le cours
    a des instants differents et un split doit etre absorbe compte par compte."""
    return f"{instrument}#{slot}"


def load_last_price(instrument, slot=1):
    """Dernier prix connu de l'instrument sur ce compte (seance precedente), ou None.
    Repli sur l'ancienne cle mono-compte `instrument` pour le slot 1 (transition)."""
    if not LAST_PRICE_FILE.exists():
        return None
    try:
        data = json.loads(LAST_PRICE_FILE.read_text())
    except Exception:  # noqa: BLE001
        return None
    v = data.get(_price_key(instrument, slot))
    if v is None and slot == 1:
        v = data.get(instrument)
    return v


def save_last_price(instrument, price, slot=1):
    """Memorise le prix de reference de l'instrument pour ce compte (detection de split)."""
    data = {}
    if LAST_PRICE_FILE.exists():
        try:
            data = json.loads(LAST_PRICE_FILE.read_text())
        except Exception:  # noqa: BLE001
            data = {}
    data[_price_key(instrument, slot)] = price
    if slot == 1:
        data[instrument] = price   # ancienne cle, gardee a jour pour compat
    LAST_PRICE_FILE.write_text(json.dumps(data, indent=2))


def detect_split(prev_price, cur_price, factor=SPLIT_DETECT_FACTOR):
    """Renvoie le facteur de saut si le prix a bouge d'au moins `factor` (x ou /)
    entre `prev_price` et `cur_price` (= split / anomalie probable), sinon None.
    Renvoie None si l'un des prix est manquant/non positif (pas de reference -> pas
    de detection possible, on ne bloque pas)."""
    if not prev_price or not cur_price or prev_price <= 0 or cur_price <= 0:
        return None
    ratio = max(cur_price / prev_price, prev_price / cur_price)
    return ratio if ratio >= factor else None


def _send_account_email(account, subject, body):
    """Mail a l'adresse du compte (repli MAILING_LIST) — jamais bloquant."""
    try:
        from src.bourso.notify import send_email
        send_email(subject, body, to=account.recipients)
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] Email non envoye ({account.label}): {e}")


def _base_record(account, signal, target_alloc, execute):
    return {
        "date": str(date.today()), "model_date": signal["date"],
        "probability": signal["probability"], "target_alloc": target_alloc,
        "instrument": INSTRUMENT, "leverage": LEVERAGE,
        "account": account.slot, "account_name": account.name,
        "executed": execute,
    }


def process_account(account, signal, target_alloc, execute):
    """Lit le PEA du compte, decide et execute l'ordre du jour, journalise et envoie le
    mail du compte. Leve si le PEA reste illisible (le mail d'echec est envoye par main).

    Retourne le dict {side, quantity, reason, result, state}."""
    mode = "LIVE" if execute else "DRY-RUN"
    print(f"\n{'-'*60}\n  {account.label.upper()}\n{'-'*60}")

    # 1. Etat du PEA (cash, position) — backoff court ; la relance horaire est geree
    #    par main() sur l'ensemble des comptes en echec (un compte KO ne bloque pas
    #    les autres pendant des heures).
    pea_state = retry(lambda: get_pea_state(account), label=f"PEA state {account.label}")
    if pea_state["equity"] <= 0:
        raise RuntimeError("PEA vide (equity=0)")

    # 2. Detection de split / anomalie de prix -> aucune position aujourd'hui
    cur_price = pea_state["etf_price"]
    prev_price = load_last_price(INSTRUMENT, account.slot)
    split_ratio = detect_split(prev_price, cur_price)
    if execute:
        save_last_price(INSTRUMENT, cur_price, account.slot)
    if split_ratio is not None:
        msg = (f"SPLIT / anomalie de prix detecte sur {INSTRUMENT} : "
               f"{prev_price:.2f} -> {cur_price:.2f} EUR (facteur x{split_ratio:.1f}). "
               f"Aucune position prise aujourd'hui — on attend la prochaine seance.")
        print(f"\n{'!'*60}\n  {msg}\n{'!'*60}")
        _send_account_email(account,
            f"[MyQTM] SPLIT detecte {INSTRUMENT} — aucune position prise — {account.name}",
            f"{account.name} (compte {account.slot}) — {date.today()}\n\n"
            f"  Connexion:   OK\n"
            f"  ⚠️ {msg}\n\n"
            f"  Instrument:  {INSTRUMENT} ({INSTRUMENTS[INSTRUMENT]['label']})\n"
            f"  Prix veille: {prev_price:.2f} EUR\n"
            f"  Prix actuel: {cur_price:.2f} EUR\n"
            f"  Facteur:     x{split_ratio:.1f}\n"
            f"  Action:      AUCUNE (garde-fou split) — reprise a la prochaine seance\n"
            f"  Mode:        {mode}\n")
        rec = _base_record(account, signal, target_alloc, execute)
        rec.update(side=None, quantity=0, etf_price=cur_price, equity=pea_state["equity"],
                   current_shares=pea_state["etf_shares"],
                   reason=f"split detecte (x{split_ratio:.1f}, {prev_price:.2f}->{cur_price:.2f}) — no trade",
                   result={"status": "split_detected", "prev_price": prev_price,
                           "cur_price": cur_price, "factor": split_ratio})
        log_trade(rec)
        return {"side": None, "quantity": 0, "reason": rec["reason"], "result": rec["result"],
                "state": pea_state}

    # 3. Decision (bande asymetrique sur l'allocation reelle de CE compte)
    side, quantity, reason = compute_orders(target_alloc, pea_state)
    print(f"\nDecision: {reason}")

    # 4. Execution — backoff court (relance horaire par main si echec)
    result = None
    if side and quantity > 0:
        try:
            result = retry(
                lambda: execute_order(side, quantity, dry_run=not execute, account=account),
                label=f"Execution ordre {account.label}",
            )
        except Exception as e:  # noqa: BLE001
            print(f"[ERREUR] Ordre echoue: {e}")
            result = {"status": "error", "error": str(e)}

    # 5. Mail du compte : connexion + allocation + action (tous les jours)
    equity = pea_state["equity"]
    current_alloc = pea_state["etf_shares"] * cur_price / equity if equity > 0 else 0
    # position apres l'ordre du jour (inchangee si l'ordre est parti en erreur)
    order_ok = (result or {}).get("status") in ("executed", "dry-run")
    delta = (quantity if side == "buy" else -quantity if side == "sell" else 0) if order_ok else 0
    post_alloc = (pea_state["etf_shares"] + delta) * cur_price / equity if equity > 0 else 0
    if side and quantity > 0:
        action = f"{'ACHAT' if side == 'buy' else 'VENTE'} {quantity} parts {INSTRUMENT}"
        status = (result or {}).get("status", "?")
        action_line = f"{action} — {status.upper()}"
        if status == "error":
            action_line += f" : {(result or {}).get('error', '')[:200]}"
        subject_tail = f"{'ACHAT' if side == 'buy' else 'VENTE'} {quantity}x {INSTRUMENT}"
        if status == "error":
            subject_tail = "ORDRE EN ERREUR " + subject_tail
    else:
        action_line = f"aucun ordre ({reason})"
        subject_tail = "aucun changement"
    subject = (f"[MyQTM] {mode} {account.name} — alloc {target_alloc*100:.0f}% — {subject_tail}")
    body = (
        f"{account.name} (compte {account.slot}) — {date.today()}\n\n"
        f"  Connexion:   OK\n"
        f"  Instrument:  {INSTRUMENT} ({INSTRUMENTS[INSTRUMENT]['label']}, levier x{LEVERAGE:.0f})\n"
        f"  Signal du:   {signal['date']}  (probabilite {signal['probability']:.4f})\n\n"
        f"  Allocation conseillee: {target_alloc*100:.0f}%  ->  exposition cible {target_alloc*LEVERAGE*100:.0f}% (x{LEVERAGE:.0f})\n"
        f"  Allocation reelle:     {current_alloc*100:.0f}%  ({pea_state['etf_shares']} parts @ {cur_price:.2f} EUR)\n"
        f"  Apres ordre du jour:   {post_alloc*100:.0f}%\n\n"
        f"  Action:      {action_line}\n"
        + (f"  Type ordre:  limite ±{LIMIT_TOLERANCE_PCT}% (tampon d'ouverture)\n" if side and quantity > 0 else "")
        + f"\n  Especes:     {pea_state['cash']:.2f} EUR\n"
        f"  Titres:      {pea_state['stocks']:.2f} EUR\n"
        f"  Total:       {equity:.2f} EUR\n"
        f"  Mode:        {mode}\n"
    )
    _send_account_email(account, subject, body)

    # 6. Journal
    rec = _base_record(account, signal, target_alloc, execute)
    rec.update(side=side, quantity=quantity, etf_price=cur_price, equity=equity,
               current_shares=pea_state["etf_shares"], reason=reason, result=result)
    log_trade(rec)
    return {"side": side, "quantity": quantity, "reason": reason, "result": result, "state": pea_state}


def main():
    parser = argparse.ArgumentParser(description="Execution PEA matin — signal du backtest soir")
    parser.add_argument("--execute", action="store_true", help="Executer les ordres (defaut: dry-run)")
    parser.add_argument("--account", type=int, default=None, metavar="N",
                        help="Ne traiter que le slot N (defaut: tous les comptes geres)")
    args = parser.parse_args()

    from src.bourso.accounts import load_accounts
    accounts = load_accounts()
    if args.account is not None:
        accounts = [a for a in accounts if a.slot == args.account]

    print(f"{'='*60}")
    print(f"PEA {INSTRUMENT} ({INSTRUMENTS[INSTRUMENT]['label']}) — {date.today()}")
    print(f"Mode: {'LIVE' if args.execute else 'DRY-RUN'}")
    print(f"Comptes geres: {', '.join(f'slot {a.slot}' for a in accounts) or 'AUCUN'}")
    print(f"{'='*60}")
    if not accounts:
        print("[ERREUR] Aucun compte Bourso gere (BOURSO_ID_n / BOURSO_CODE_n vides dans .env)")
        sys.exit(1)

    # 1. Load signal from evening backtest
    signal = load_signal()
    if signal is None:
        sys.exit(1)

    # 2. Emergency override (global : s'applique a tous les comptes)
    target_alloc = signal["allocation"]
    if is_emergency_off():
        target_alloc = 0.0

    # 3. Chaque compte a son tour ; ceux en echec (PEA illisible) sont relances toutes
    #    les heures jusqu'a la cloture Euronext, sans bloquer les autres.
    euronext_close = datetime.now().replace(hour=17, minute=0, second=0)
    pending = list(accounts)
    failures = {}
    while pending:
        failures = {}
        for account in pending:
            try:
                process_account(account, signal, target_alloc, args.execute)
            except Exception as e:  # noqa: BLE001
                print(f"[ERREUR] {account.label}: {e}")
                failures[account.slot] = (account, e)
        pending = [a for a, _ in failures.values()]
        if not pending or datetime.now() >= euronext_close:
            break
        print(f"\n[RETRY] {len(pending)} compte(s) en echec, nouvelle tentative dans 1h...")
        time.sleep(3600)

    # 4. Comptes definitivement en echec : mail "connexion KO" + trace dans le journal
    for account, err in failures.values():
        msg = str(err)[:300]
        _send_account_email(account,
            f"[MyQTM] {'LIVE' if args.execute else 'DRY-RUN'} {account.name or account.label} — CONNEXION KO",
            f"{account.name or account.label} (compte {account.slot}) — {date.today()}\n\n"
            f"  Connexion:   ECHEC — {msg}\n"
            f"  Allocation conseillee: {target_alloc*100:.0f}%  (signal du {signal['date']})\n"
            f"  Action:      AUCUNE — le compte n'a pas pu etre lu, aucun ordre passe.\n"
            f"  Verifier logs/cron_pea.log.\n")
        rec = _base_record(account, signal, target_alloc, args.execute)
        rec.update(side=None, quantity=0, etf_price=0, equity=0, current_shares=None,
                   reason=f"connexion KO: {msg}", result={"status": "connection_error", "error": msg})
        log_trade(rec)

    print("\nTermine." + (f"  ({len(failures)} compte(s) en echec)" if failures else ""))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
