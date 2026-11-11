"""Detection d'un APPORT (ou retrait) de capital sur le PEA.

Principe de detection : bourso-cli ne liste pas les mouvements d'especes, mais
`trade summary` (position=INSTANT) donne le cash TEMPS REEL du compte. Chaque matin
on compare le cash lu au cash ATTENDU = cash de la veille +/- les flux des ordres
passes depuis (achats -qte.prix.(1+0.5%), ventes +qte.prix.(1-0.5%)). Un ecart positif
au-dela d'un seuil de bruit = apport ; un ecart negatif = retrait. Le bruit vient
du prix de remplissage des ordres limites (±3% de tolerance) : le seuil est donc
max(DEPOSIT_MIN_EUR, DEPOSIT_MIN_FRAC . equity, DEPOSIT_TRADE_TOL . notionnel traite
depuis la derniere lecture). Un apport le jour meme d'un gros ordre peut passer sous
ce seuil : il est alors simplement traite comme du cash ordinaire — jamais perdu.

Utilisation (real_bourso) : un apport detecte est investi EN ENTIER le jour de la
detection (= le matin qui suit le virement), sans bande, a la composition cible du
signal. (L'ancien deploiement progressif DCA / RSI a ete retire le 2026-11-11.)

Etat persistant : logs/capital.json  {slot: {expected_cash, as_of, traded_notional,
deposits: [...]}}. Aucune donnee sensible.
"""
import json
from datetime import date
from pathlib import Path

from src.risk_off_strategy.strategy import SELL_FEE, BUY_FEE

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
    entry = state.setdefault(str(slot), {"expected_cash": None, "as_of": None,
                                         "traded_notional": 0.0, "deposits": []})
    entry.pop("dca", None)          # ancien etat DCA (retire le 2026-11-11)
    entry.setdefault("deposits", [])
    return entry


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
        flow += amt * (1 - SELL_FEE) if o["side"] == "sell" else -amt * (1 + BUY_FEE)
    entry["expected_cash"] = round(float(cash_now) + flow, 2)
    entry["as_of"] = str(today)
    entry["traded_notional"] = round(notional, 2)
