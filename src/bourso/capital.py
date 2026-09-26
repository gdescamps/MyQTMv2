"""Detection d'un APPORT de capital sur le PEA + deploiement progressif (DCA / RSI).

Principe de detection : bourso-cli ne liste pas les mouvements d'especes, mais
`trade summary` (position=INSTANT) donne le cash TEMPS REEL du compte. Chaque matin
on compare le cash lu au cash ATTENDU = cash de la veille +/- les flux des ordres
passes depuis (achats -qte.prix, ventes +qte.prix.(1-0.5%)). Un ecart positif
au-dela d'un seuil de bruit = apport ; un ecart negatif = retrait. Le bruit vient
du prix de remplissage des ordres limites (±3% de tolerance) : le seuil est donc
max(DEPOSIT_MIN_EUR, DEPOSIT_MIN_FRAC . equity, DEPOSIT_TRADE_TOL . notionnel traite
depuis la derniere lecture). Un apport le jour meme d'un gros ordre peut passer sous
ce seuil : il est alors simplement traite comme du cash ordinaire (investi par la
regle "cash oisif" de real_bourso, sans DCA) — jamais perdu.

Deploiement (strategy.py : DCA_*) : l'apport est RESERVE (exclu de l'equity que la
strategie alloue) puis libere par tranches hebdomadaires quand le RSI(14) du QQQ est
sous DCA_RSI_GATE (tranche plus grosse si le RSI est plus bas) ; tout est libere au
plus tard apres DCA_MAX_WEEKS. Le jour d'une tranche, real_bourso achete sans bande
(force_buy) vers la composition cible sur l'equity liberee.

Etat persistant : logs/capital.json  {slot: {expected_cash, as_of, traded_notional,
deposits: [...], dca: {...} | None}}. Aucune donnee sensible.
"""
import json
from datetime import date, datetime
from pathlib import Path

from src.risk_off_strategy.strategy import (
    SELL_FEE, DCA_MAX_WEEKS, DCA_WEEK_DAYS, dca_tranche,
)

ROOT = Path(__file__).resolve().parent.parent.parent
CAPITAL_FILE = ROOT / "logs" / "capital.json"

DEPOSIT_MIN_EUR = 100.0     # sous 100 EUR : bruit (arrondis, frais), pas un apport
DEPOSIT_MIN_FRAC = 0.005    # ... ni sous 0.5% de l'equity
DEPOSIT_TRADE_TOL = 0.05    # ... ni sous 5% du notionnel traite depuis la derniere lecture
                            #     (ordre limite ±3% + frais de vente 0.5% -> ecart de prix)


# ── Persistance ───────────────────────────────────────────

def load_capital_state(path=CAPITAL_FILE):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def save_capital_state(state, path=CAPITAL_FILE):
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(state, indent=2))


def get_entry(state, slot):
    return state.setdefault(str(slot), {"expected_cash": None, "as_of": None,
                                        "traded_notional": 0.0, "deposits": [], "dca": None})


# ── Detection ─────────────────────────────────────────────

def deposit_threshold(entry, equity):
    return max(DEPOSIT_MIN_EUR, DEPOSIT_MIN_FRAC * max(equity, 0.0),
               DEPOSIT_TRADE_TOL * float(entry.get("traded_notional") or 0.0))


def detect_deposit(entry, cash_now, equity, today=None):
    """Compare le cash lu au cash attendu. Renvoie (apport, retrait) en EUR (>= 0),
    et enregistre l'apport dans entry["deposits"]. Premiere lecture (pas de cash
    attendu) -> (0, 0) : on ne fait qu'initialiser la reference."""
    today = today or date.today()
    expected = entry.get("expected_cash")
    if expected is None:
        return 0.0, 0.0
    dev = float(cash_now) - float(expected)
    thr = deposit_threshold(entry, equity)
    if dev >= thr:
        entry["deposits"].append({"date": str(today), "amount": round(dev, 2),
                                  "expected_cash": round(float(expected), 2),
                                  "cash": round(float(cash_now), 2)})
        return dev, 0.0
    if dev <= -thr:
        entry["deposits"].append({"date": str(today), "amount": round(dev, 2),
                                  "expected_cash": round(float(expected), 2),
                                  "cash": round(float(cash_now), 2)})
        return 0.0, -dev
    return 0.0, 0.0


def record_expected_cash(entry, cash_now, orders, today=None):
    """Cash attendu a la prochaine lecture = cash lu aujourd'hui +/- flux des ordres
    passes (estimes au dernier cours ; l'ecart de remplissage est couvert par
    DEPOSIT_TRADE_TOL sur le notionnel traite)."""
    today = today or date.today()
    flow = 0.0
    notional = 0.0
    for o in orders or []:
        if o.get("status") not in ("executed",):
            continue
        amt = float(o["quantity"]) * float(o["price"])
        notional += amt
        flow += amt * (1 - SELL_FEE) if o["side"] == "sell" else -amt
    entry["expected_cash"] = round(float(cash_now) + flow, 2)
    entry["as_of"] = str(today)
    entry["traded_notional"] = round(notional, 2)


# ── DCA de l'apport ───────────────────────────────────────

def start_dca(entry, amount, today=None):
    """Nouvel apport : ouvre (ou complete) le DCA. Un apport pendant un DCA en cours
    s'ajoute au montant restant a liberer (la fraction liberee est recalculee)."""
    today = today or date.today()
    dca = entry.get("dca")
    if dca and not dca.get("done"):
        remaining = dca["deposit"] * (1 - dca["released"]) + amount
        dca["deposit"] = round(dca["deposit"] + amount, 2)
        dca["released"] = round(1 - remaining / dca["deposit"], 6) if dca["deposit"] > 0 else 1.0
        dca["events"].append({"date": str(today), "type": "deposit", "amount": round(amount, 2)})
    else:
        entry["dca"] = {"deposit": round(amount, 2), "released": 0.0, "detected": str(today),
                        "last_tranche": None, "weeks": 0, "done": False,
                        "events": [{"date": str(today), "type": "deposit", "amount": round(amount, 2)}]}
    return entry["dca"]


def apply_withdrawal(entry, amount):
    """Retrait : il sort d'abord de la part reservee (non encore investie)."""
    dca = entry.get("dca")
    if not dca or dca.get("done"):
        return
    reserved = dca["deposit"] * (1 - dca["released"])
    new_reserved = max(0.0, reserved - amount)
    if new_reserved <= 0:
        dca["done"] = True
        dca["released"] = 1.0
    else:
        dca["released"] = round(1 - new_reserved / dca["deposit"], 6)
    dca["events"].append({"date": str(date.today()), "type": "withdrawal", "amount": round(amount, 2)})


def dca_due(dca, today):
    """Une tranche est due le jour de la detection, puis tous les DCA_WEEK_DAYS jours."""
    if dca is None or dca.get("done"):
        return False
    last = dca.get("last_tranche")
    if last is None:
        return True
    return (today - datetime.strptime(last, "%Y-%m-%d").date()).days >= DCA_WEEK_DAYS


def dca_step(entry, rsi, today=None):
    """Evalue la tranche hebdo si elle est due. Renvoie la fraction liberee aujourd'hui
    (0 si rien / pas due). Marque `done` quand tout est libere ou apres DCA_MAX_WEEKS."""
    today = today or date.today()
    dca = entry.get("dca")
    if not dca_due(dca, today):
        return 0.0
    dca["weeks"] += 1
    dca["last_tranche"] = str(today)
    if dca["weeks"] >= DCA_MAX_WEEKS:
        tranche = 1.0 - dca["released"]            # filet : tout ce qui reste
    else:
        tranche = min(dca_tranche(rsi), 1.0 - dca["released"])
    dca["released"] = round(min(1.0, dca["released"] + tranche), 6)
    dca["events"].append({"date": str(today), "type": "tranche", "rsi": None if rsi is None else round(float(rsi), 1),
                          "tranche": round(tranche, 4), "released": dca["released"]})
    if dca["released"] >= 1.0 - 1e-9:
        dca["released"] = 1.0
        dca["done"] = True
    return float(tranche)


def reserved_cash(entry, cash_now):
    """Part de l'apport pas encore liberee (exclue de l'equity allouee), bornee au cash."""
    dca = entry.get("dca")
    if not dca or dca.get("done"):
        return 0.0
    return float(min(max(cash_now, 0.0), dca["deposit"] * (1 - dca["released"])))


def dca_summary(entry, cash_now):
    """Resume pour mails / journal / webapp (None si aucun DCA en cours)."""
    dca = entry.get("dca")
    if not dca or dca.get("done"):
        return None
    next_date = None
    if dca.get("last_tranche"):
        d = datetime.strptime(dca["last_tranche"], "%Y-%m-%d").date()
        next_date = str(date.fromordinal(d.toordinal() + DCA_WEEK_DAYS))
    return {"deposit": dca["deposit"], "released": dca["released"],
            "reserved": round(reserved_cash(entry, cash_now), 2),
            "detected": dca["detected"], "weeks": dca["weeks"], "next_tranche": next_date}
