"""
Morning PEA execution on BoursoBank — reads signal from evening backtest.

Workflow:
  22:30  cron backtest (run.py) → outputs/qqq_strategy/signal.json (allocation, RSI14)
  09:05  this script → reads signal, checks each PEA, executes PUST / LQQ at Euronext open

Strategie deployee : PUST (Nasdaq x1) + LQQ (Nasdaq x2), exposition plafonnee.
  Exposition cible E = min(2 . alloc, E_MAX)  (E_MAX = 1.7, cf. strategy.py), realisee
  "drag-minimale" :  E <= 1 : PUST = E, cash = 1 - E   |   E > 1 : PUST = 2 - E, LQQ = E - 1
  (LQQ ne porte que la part > 100%, tout le cash travaille). Bande de non-action
  asymetrique en ESPACE EXPOSITION, identique au backtest net (simulate_net) : on remonte
  l'expo si cible - reelle >= 0.50, on la baisse si reelle - cible >= 1.00, force
  vers/depuis le cash total. Frais Bourso 0.5% a l'achat comme a la vente (BUY_FEE /
  SELL_FEE). Les VENTES partent en premier, les ACHATS attendent que le produit des
  ventes soit credite (trade summary temps reel) et sont dimensionnes sur le cash au
  prix RESERVE par Bourso (limite +3% + provision BUY_CASH_MARGIN 3%) ; si le cash
  n'arrive pas avant 17h, ou si Bourso refuse l'achat ("solde especes insuffisant"),
  l'achat est differe : reprise horaire puis lendemain (logs/pending_orders.json).

Restructuration : si le compte est levier (E > 1) avec du cash oisif (>= 5%, = l'ancienne
realisation "LQQ + cash", un achat refuse ou un residu), on passe a la composition
drag-minimale vers clip(cible, E_eff, E_eff + cash) plafonne E_MAX : le cash seul
remonte l'expo (achat de PUST, sans vente) ; du LQQ n'est vendu que si la cible est
sous E_eff (expo constante).
C'est la migration immediate LQQ x2 -> PUST + LQQ x1.7 ; elle ne se redeclenche pas
ensuite (plus de cash oisif quand E > 1).

Apport de capital : detecte par ecart entre le cash lu et le cash attendu (cf.
src/bourso/capital.py) au run du matin qui suit le virement, et investi EN ENTIER ce
jour-la, sans bande, a la composition cible du signal (composition(cible)) ; la bande
continue de s'appliquer au portefeuille existant, evaluee hors apport.

Multicompte : le meme signal est REPLIQUE sur chaque compte Bourso gere (slots
BOURSO_ID_n/CODE_n/MAIL_n du .env, cf. src/bourso/accounts.py). Chaque compte est lu,
decide (bande sur SA propre exposition reelle) et execute a son tour ; un mail par compte.
Un compte injoignable n'empeche pas les autres de s'executer.

Usage:
  python -m src.real_bourso                  # dry-run
  python -m src.real_bourso --execute        # live execution
  python -m src.real_bourso --account 2      # un seul slot
  TRADE_E_MAX=1.0 python -m src.real_bourso  # plafond d'expo different (1.0 = PUST seul)
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
PENDING_FILE = LOG_DIR / "pending_orders.json"

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except Exception:
    pass

# ── Instruments (les deux sont detenus simultanement quand E > 1) ─────────────
INSTRUMENTS = {
    "PUST": {"leverage": 1.0, "label": "Nasdaq x1"},
    "LQQ": {"leverage": 2.0, "label": "Nasdaq x2 (leverage)"},
}
STRATEGY_LABEL = "PUST+LQQ"

# Bande + plafond : source unique = la strategie backtestee (strategy.py).
from src.risk_off_strategy.strategy import (  # noqa: E402,F401  (BUY/SELL_THR_ALLOC re-exportes pour les tests)
    BUY_THR_ALLOC, SELL_THR_ALLOC, BUY_THR_E, SELL_THR_E, SELL_FEE, BUY_FEE,
    E_MAX as STRATEGY_E_MAX, target_exposure, composition,
)

# Plafond d'exposition live : E_MAX de strategy.py, surchargeable par TRADE_E_MAX
# (ex. 1.0 = PUST seul, sans levier). L'ancien switch TRADE_INSTRUMENT est ignore.
E_MAX = STRATEGY_E_MAX
_env_emax = os.environ.get("TRADE_E_MAX", "").strip()
if _env_emax:
    try:
        E_MAX = min(max(float(_env_emax), 0.0), 2.0)
    except ValueError:
        print(f"[WARN] TRADE_E_MAX='{_env_emax}' invalide -> plafond strategie {STRATEGY_E_MAX}")
if os.environ.get("TRADE_INSTRUMENT", "").strip():
    print(f"[INFO] TRADE_INSTRUMENT ignore : realisation {STRATEGY_LABEL}, plafond d'expo x{E_MAX}")

# Cash oisif tolere quand le compte est levier (E > 1) : au-dela, restructuration vers
# la composition drag-minimale (cf. compute_orders). Sous ce seuil = residu d'arrondi
# (marge de cash des achats, frais).
IDLE_CASH_TOL = 0.05

# Signal must be fresh enough, but tolerate weekend/holiday gaps so Monday (and
# post-long-weekend) mornings still execute. The backtest runs each weekday
# evening, so the worst legitimate gap is a Monday morning reading Friday's
# signal (~58h), or a long holiday weekend (Thu eve → Tue morning, ~82h).
# Anything older than ~3.75 days means the evening backtest stopped → refuse.
MAX_SIGNAL_AGE_HOURS = 90
MAX_RETRIES = 5
INITIAL_WAIT = 60  # seconds
MAX_WAIT = 900  # 15 min max between retries

# Detection de split / anomalie de prix : si le prix d'un instrument saute d'un
# facteur >= SPLIT_DETECT_FACTOR (x1.5 a la hausse, ou /1.5) vs la derniere seance,
# c'est quasi surement un split (ex: LQQ /200) ou une incoherence d'affichage broker,
# PAS un mouvement de marche (un ETF x2 ne fait pas +/-50% en une seance : coupe-circuits).
# Dans ce cas on NE PREND PAS de position ce jour-la (on evite d'acheter/vendre sur un
# prix fausse le jour du split) et on attend la seance suivante. Le prix de reference par
# instrument et par compte est stocke dans LAST_PRICE_FILE (mis a jour a chaque run LIVE).
SPLIT_DETECT_FACTOR = 1.5
LAST_PRICE_FILE = LOG_DIR / "last_price.json"

# Tolerance de l'ordre LIMITE : la limite = dernier cours ± LIMIT_TOLERANCE_PCT%
# (achat: +, vente: -). Un ordre limite pile au cours ne se remplit pas si le prix
# s'ecarte a l'ouverture (cf. incident 07-07 : limite sous le marche -> non execute).
# Une tolerance de 3% tampon le gap d'ouverture -> remplissage fiable, tout en
# bornant le prix (contrairement a un ordre au marche non maitrise sur un gap).
LIMIT_TOLERANCE_PCT = 3.0

# Attente du credit des ventes avant les achats : on considere les ventes creditees
# quand le cash a augmente d'au moins cette fraction du produit attendu (le reste =
# ecart de prix de remplissage, jusqu'a -3% de limite et -0.5% de frais).
SELL_CREDIT_MIN_FRAC = 0.90

# Dimensionnement des ACHATS : Bourso reserve quantite x LIMITE (cours +3%) plus une
# provision (~1.5% constatee : frais 0.5% + marge) et refuse l'ordre au-dela du cash
# ("solde especes insuffisant", code 5010 — incident du 2026-09-28 : 173 PUST dimensionnes
# au dernier cours = 19 162 EUR reserves pour 18 676 EUR de cash ; 167 parts a 1.5% de
# marge encore refusees par l'API). On plafonne donc la quantite par
# cash / (cours x (1 + tolerance) x (1 + BUY_CASH_MARGIN)) : au pire ~6% du cash reste
# non deploye sur un achat plein-cash, soit ~1 point d'expo, negligeable devant la bande.
BUY_CASH_MARGIN = 0.03

# Montant MINIMUM d'un achat chez Bourso : 200 EUR (code 5110, "Le montant minimum pour
# un achat est de 200.0 euros", constate le 2026-09-28 sur 1 PUST a 110 EUR). Une jambe
# d'achat dont la valeur (quantite x cours) est sous ce plancher n'est PAS envoyee : le
# cash reste sur le compte (il sera deploye avec le prochain achat, quand l'ecart a la
# cible depassera 200 EUR). Aucun plancher n'est applique aux
# ventes (non constate).
MIN_BUY_EUR = 200.0


def buy_reserve_price(price, tolerance=LIMIT_TOLERANCE_PCT, margin=BUY_CASH_MARGIN):
    """Cash a prevoir par part achetee : limite (+tolerance%) + provision Bourso (marge)."""
    return float(price) * (1 + tolerance / 100.0) * (1 + margin)


def is_cash_refusal(msg):
    """Refus Bourso 'solde especes insuffisant' (code 5010) sur un ordre d'achat."""
    m = (msg or "").lower()
    return "5010" in m or "insuffisant" in m


def is_min_amount_refusal(msg):
    """Refus Bourso 'montant minimum pour un achat' (code 5110, 200 EUR)."""
    m = (msg or "").lower()
    return "5110" in m or "montant minimum" in m


def retry(fn, label="", hourly_until=None):
    """Retry with exponential backoff (60s, 120s, 240s, 480s, 900s), then hourly."""
    wait = INITIAL_WAIT
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            if attempt == MAX_RETRIES:
                break
            print(f"[RETRY] {label}: tentative {attempt}/{MAX_RETRIES} echouee: {e}")
            print(f"  Prochaine tentative dans {wait}s...")
            time.sleep(wait)
            wait = min(wait * 2, MAX_WAIT)

    if hourly_until is None:
        print(f"[ERREUR] {label}: echec apres {MAX_RETRIES} tentatives")
        raise last_exc
    attempt = MAX_RETRIES
    while datetime.now() < hourly_until:
        attempt += 1
        print(f"[RETRY] {label}: tentative {attempt} (horaire), prochaine dans 1h...")
        time.sleep(3600)
        try:
            return fn()
        except Exception as e:
            last_exc = e
            print(f"[RETRY] {label}: tentative {attempt} echouee: {e}")

    print(f"[ERREUR] {label}: echec, deadline {hourly_until} atteinte")
    raise last_exc


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
        print("  Verifier logs/cron_backtest.log")
        return None

    # Check signal freshness
    ts = datetime.fromisoformat(signal["timestamp"])
    age = datetime.now() - ts
    if age > timedelta(hours=MAX_SIGNAL_AGE_HOURS):
        print(f"[ERREUR] Signal trop ancien: {signal['date']} ({age.total_seconds()/3600:.0f}h)")
        print("  Le backtest du soir a-t-il tourne? Verifier logs/cron_backtest.log")
        return None

    rsi = signal.get("rsi14")
    print(f"Signal du {signal['date']} (status=ok, age: {age.total_seconds()/3600:.1f}h)")
    print(f"  Probabilite: {signal['probability']:.4f}")
    print(f"  Allocation:  {signal['allocation']*100:.0f}%  ->  exposition cible "
          f"{target_exposure(signal['allocation'], E_MAX)*100:.0f}% (plafond x{E_MAX})")
    print(f"  RSI14 QQQ:   {rsi:.1f}" if rsi is not None else "  RSI14 QQQ:   (absent du signal)")
    return signal


def _default_account():
    """Slot 1 (compat mono-compte des anciens appels sans `account`)."""
    from src.bourso.accounts import load_accounts
    accs = load_accounts()
    if not accs:
        raise RuntimeError("aucun compte Bourso gere (BOURSO_ID_1/BOURSO_CODE_1 vides)")
    return accs[0]


def exposure_of(positions, equity):
    """Exposition reelle = (PUST + 2 . LQQ) / equity."""
    if equity <= 0:
        return 0.0
    return sum(INSTRUMENTS[k]["leverage"] * positions[k]["shares"] * positions[k]["price"]
               for k in INSTRUMENTS) / equity


def get_pea_state(account=None, quiet=False):
    """Etat du PEA (cash + positions PUST et LQQ) via `bourso-cli trade summary`
    (position=INSTANT : temps reel, les ordres executes sont deja refletes).

    account: BoursoAccount (cf. accounts.py) ; None -> slot 1. L'id du PEA du login est
    decouvert/cache par `resolve_pea` (chaque login Bourso a son propre PEA).
    Un instrument non detenu (absent des positions) est cote via quote.py."""
    from src.bourso.prepare import get_account_summary, SYMBOLS
    from src.bourso.accounts import resolve_pea

    account = account or _default_account()
    resolve_pea(account)
    summary = get_account_summary(account.pea_account_id, creds=account.creds)
    acct = summary["account"]

    positions = {}
    for ins in INSTRUMENTS:
        pos = summary["positions"].get(SYMBOLS[ins], {})
        qty = int(pos.get("quantity") or 0)
        price = pos.get("last_price")
        if price is None:
            from src.bourso.quote import get_quote
            price = get_quote(SYMBOLS[ins]).get("last")
        if not price or float(price) <= 0:
            raise RuntimeError(f"cours {ins} indisponible")
        positions[ins] = {"shares": qty, "price": float(price), "value": qty * float(price)}

    cash = float(acct.get("cash") or 0.0)
    stocks = float(acct.get("stocks") or 0.0)
    managed = sum(p["value"] for p in positions.values())
    state = {
        "cash": cash,
        "stocks": stocks,
        "equity": cash + stocks,
        "other_stocks": max(0.0, stocks - managed),    # autres lignes du PEA (hors strategie)
        "positions": positions,
        "account": account.slot,
        "account_name": acct.get("name") or account.name,
    }
    state["exposure"] = exposure_of(positions, cash + managed)

    if not quiet:
        print(f"\n{state['account_name']} (compte {account.slot}):")
        print(f"  Especes:  {cash:>10.2f} EUR")
        print(f"  Titres:   {stocks:>10.2f} EUR")
        print(f"  Total:    {state['equity']:>10.2f} EUR")
        for ins, p in positions.items():
            print(f"  {ins:<5} {p['shares']:>6} parts @ {p['price']:.3f} EUR = {p['value']:>9.0f} EUR "
                  f"({INSTRUMENTS[ins]['label']})")
        if state["other_stocks"] > 0.01 * state["equity"]:
            print(f"  [INFO] autres titres hors strategie: {state['other_stocks']:.0f} EUR (ignores)")
        print(f"  Exposition reelle: {state['exposure']*100:.0f}%")
    return state


def is_emergency_off():
    """Check if emergency override is active."""
    if OVERRIDE_FILE.exists():
        with open(OVERRIDE_FILE) as f:
            data = json.load(f)
        if data.get("active", False):
            print(f"\n{'!'*60}")
            print("  EMERGENCY OFF — allocation forcee a 0%")
            print(f"  Supprimer {OVERRIDE_FILE} pour reprendre")
            print(f"{'!'*60}")
            return True
    return False


# ── Decision ──────────────────────────────────────────────

def compute_orders(target_e, state, deposit=0.0, force_buy=False, buys_only=False, e_max=None):
    """Ordres (ventes puis achats) pour amener l'exposition reelle vers `target_e`,
    dans la composition drag-minimale, sur l'equity GEREE (PUST + LQQ + cash).

    deposit   : apport de capital detecte ce matin (EUR). La bande est evaluee sur le
                portefeuille HORS apport (l'apport ne doit pas declencher a lui seul un
                rebalancement) ; l'apport est ensuite investi en entier, sans bande, a la
                composition CIBLE du jour (composition(target_e)) — achats seulement.
    force_buy : achats sans bande (achat reporte de la veille).
    buys_only : jamais de vente (reprise d'un achat reporte le jour meme).
    Renvoie un plan {orders: [{instrument, side, quantity, price, value, fee}], mode,
    reason, e_eff, e_new, target_e, equity, deposit, cash_avail, weights, target_weights}."""
    e_max = E_MAX if e_max is None else e_max
    pos = state["positions"]
    px = {k: float(pos[k]["price"]) for k in INSTRUMENTS}
    sh = {k: int(pos[k]["shares"]) for k in INSTRUMENTS}
    val = {k: sh[k] * px[k] for k in INSTRUMENTS}
    cash = max(0.0, float(state["cash"]))
    deposit = min(max(0.0, float(deposit or 0.0)), cash)
    V = val["PUST"] + val["LQQ"] + cash
    plan = {"orders": [], "mode": None, "reason": "", "target_e": target_e, "e_eff": 0.0,
            "e_new": 0.0, "equity": V, "deposit": deposit, "cash_avail": cash,
            "weights": {"PUST": 0.0, "LQQ": 0.0, "cash": 1.0}, "target_weights": None}
    if V <= 0:
        plan["reason"] = "PEA vide (equity geree = 0)"
        return plan

    e_eff = (val["PUST"] + 2.0 * val["LQQ"]) / V
    w = {"PUST": val["PUST"] / V, "LQQ": val["LQQ"] / V, "cash": cash / V}
    plan.update(e_eff=e_eff, weights=w)
    # Decision de bande sur le portefeuille HORS apport (V0) : un apport fait baisser
    # l'exposition reelle mais n'est pas un signal de rebalancement.
    V0 = V - deposit
    e_dec = (val["PUST"] + 2.0 * val["LQQ"]) / V0 if V0 > 1e-9 else 0.0
    cash0_w = (cash - deposit) / V0 if V0 > 1e-9 else 0.0
    invested = sh["PUST"] > 0 or sh["LQQ"] > 0
    to_cash = target_e <= 1e-9 and invested
    from_cash = (not invested) and target_e > 1e-9
    delta = target_e - e_dec

    print(f"\nExposition (equity geree {V:.0f} EUR" + (f", dont apport {deposit:.0f} EUR" if deposit else "") + "):")
    print(f"  Reelle:  {e_eff*100:.1f}%  (PUST {w['PUST']*100:.0f}% | LQQ {w['LQQ']*100:.0f}% | cash {w['cash']*100:.0f}%)")
    if deposit:
        print(f"  Hors apport: {e_dec*100:.1f}%  (base de la decision de bande)")
    print(f"  Cible:   {target_e*100:.1f}%")
    print(f"  Delta:   {delta*100:+.1f}%")

    # FORCE vers/depuis le cash, comme le backtest (clause `force` de simulate_net) :
    # un passage a 0% (garde-fous macro NFCI/IPC, emergency-off) doit TOUJOURS liquider,
    # meme si le delta est sous le seuil de vente ; et une premiere entree depuis le cash
    # total doit s'executer meme sous le seuil d'achat.
    if to_cash:
        mode = "to_cash"
    elif from_cash:
        mode = "from_cash"
    # Bande asymetrique en espace exposition (cf. strategy.py) : on ne re-monte que si
    # delta >= BUY_THR_E (achats gratuits) et on ne baisse que si delta <= -SELL_THR_E.
    elif delta >= BUY_THR_E:
        mode = "buy_band"
    elif -delta >= SELL_THR_E:
        mode = "sell_band"
    elif force_buy and delta > 1e-9:
        mode = "force_buy"                  # achat reporte de la veille : sans bande
    # Cash oisif alors que le compte est levier (E > 1) : pas la realisation drag-
    # minimale (ex. ancienne "LQQ + cash", ou achat refuse/partiel) -> restructuration
    # vers la cible bornee par [E_eff, E_eff + cash] : le cash seul remonte l'expo
    # (achats, pas de vente) ; on ne revend du LQQ que si la cible est sous E_eff.
    # (Evalue hors apport : l'apport est investi a part, a la composition cible.)
    elif e_dec > 1.0 and cash0_w >= IDLE_CASH_TOL and not buys_only:
        mode = "restructure"
    # Apport seul (portefeuille existant dans la bande) : on n'investit QUE l'apport.
    elif deposit > 0 and target_e > 1e-9:
        mode = "deposit"
    else:
        if delta > 0:
            plan["reason"] = f"Achat ignore: delta +{delta*100:.1f}% < seuil {BUY_THR_E*100:.0f}%"
        elif delta < 0:
            plan["reason"] = f"Vente ignoree: delta {abs(delta)*100:.1f}% < seuil {SELL_THR_E*100:.0f}%"
        else:
            plan["reason"] = "Aucune action (dans la bande de non-action)"
        plan["e_new"] = e_eff
        return plan

    # Cible en EUR par instrument. Modes "globaux" (bande, force, entree/sortie du cash) :
    # tout le compte, apport compris, vers composition(cible). Restructuration / apport
    # seul : portefeuille existant (hors apport) + apport x composition(cible).
    dep_w = composition(target_e)
    if mode == "restructure":
        # clip(cible, E_eff, E_eff + cash) : E_eff + w_cash = tout le cash en PUST (sans
        # vente) ; en dessous de E_eff il faudrait vendre du LQQ -> on garde E_eff.
        e0 = min(max(target_e, e_dec), e_dec + cash0_w, e_max)
        cp, cl, _ = composition(e0)
        tgt = {"PUST": cp * V0 + dep_w[0] * deposit, "LQQ": cl * V0 + dep_w[1] * deposit}
    elif mode == "deposit":
        tgt = {"PUST": val["PUST"] + dep_w[0] * deposit, "LQQ": val["LQQ"] + dep_w[1] * deposit}
    else:
        tgt = {"PUST": dep_w[0] * V, "LQQ": dep_w[1] * V}
    e_new = (tgt["PUST"] + 2.0 * tgt["LQQ"]) / V
    wp, wl = tgt["PUST"] / V, tgt["LQQ"] / V
    wc = 1.0 - wp - wl
    plan.update(mode=mode, e_new=e_new, target_weights={"PUST": wp, "LQQ": wl, "cash": wc if wc > 1e-9 else 0.0})
    buys_only = buys_only or mode in ("force_buy", "from_cash", "deposit")

    orders = []
    if not buys_only:
        for ins in ("LQQ", "PUST"):                       # ventes d'abord
            d = tgt[ins] - val[ins]
            if mode == "to_cash":
                q = sh[ins]
            elif d < -px[ins]:                            # marge d'1 part
                q = min(int(-d / px[ins] + 1e-9), sh[ins])
            else:
                q = 0
            if q > 0:
                orders.append({"instrument": ins, "side": "sell", "quantity": q, "price": px[ins],
                               "value": q * px[ins], "fee": q * px[ins] * SELL_FEE})
    proceeds = sum(o["value"] * (1 - SELL_FEE) for o in orders)
    avail = cash + proceeds
    below_min = []
    if mode != "to_cash":
        for ins in ("PUST", "LQQ"):                       # achats : base PUST, puis LQQ
            d = tgt[ins] - val[ins]
            if d > px[ins]:
                # plafond de cash au prix RESERVE par Bourso (limite + provision), pas au cours
                q = min(int(d / px[ins] + 1e-9), int(avail / buy_reserve_price(px[ins]) + 1e-9))
                if q > 0 and q * px[ins] < MIN_BUY_EUR:
                    # sous le minimum Bourso (200 EUR) : pas d'ordre, le cash reste en attente
                    below_min.append(f"{q} {ins} ({q * px[ins]:.0f} EUR)")
                    print(f"  [INFO] achat {q} {ins} = {q * px[ins]:.0f} EUR < minimum "
                          f"{MIN_BUY_EUR:.0f} EUR : conserve en cash")
                    q = 0
                if q > 0:
                    orders.append({"instrument": ins, "side": "buy", "quantity": q, "price": px[ins],
                                   "value": q * px[ins], "fee": q * px[ins] * BUY_FEE})
                    avail -= q * px[ins] * (1 + BUY_FEE)
    if mode == "restructure" and not any(o["side"] == "buy" for o in orders):
        # La restructuration ne fait que changer la composition (vente LQQ -> achat PUST) ;
        # si aucun achat n'est possible (jambe < 1 part ou sous le minimum 200 EUR, petit
        # compte), vendre reviendrait a baisser l'exposition hors bande -> on ne fait rien.
        if orders:
            print("  [INFO] restructuration sans achat possible : ventes annulees, composition conservee")
        orders = []
    plan["orders"] = orders
    plan["cash_avail"] = cash + proceeds
    plan["below_min"] = below_min

    labels = {"to_cash": "passage en cash (force)", "from_cash": "entree depuis le cash (force)",
              "buy_band": f"achat (delta +{delta*100:.1f}% >= {BUY_THR_E*100:.0f}%)",
              "sell_band": f"vente (delta {delta*100:.1f}% <= -{SELL_THR_E*100:.0f}%)",
              "force_buy": f"achat sans bande (delta +{delta*100:.1f}%)",
              "restructure": f"restructuration vers expo {e_new*100:.0f}% (cash oisif {cash0_w*100:.0f}%, "
                             f"cible {target_e*100:.0f}%)",
              "deposit": f"investissement de l'apport +{deposit:.0f} EUR a la composition cible "
                         f"(PUST {dep_w[0]*100:.0f}% | LQQ {dep_w[1]*100:.0f}% | especes {dep_w[2]*100:.0f}%)"}
    label = labels[mode]
    if deposit and mode != "deposit":
        label += f" [apport +{deposit:.0f} EUR inclus]"
    min_txt = (f" ; achat sous le minimum {MIN_BUY_EUR:.0f} EUR conserve en cash : "
               + ", ".join(below_min)) if below_min else ""
    if not orders:
        plan["reason"] = f"{label} : rien a faire (marge d'1 part / cash insuffisant){min_txt}"
        plan["mode"] = None
        return plan
    plan["reason"] = label + " : " + " ; ".join(describe_order(o) for o in orders) + min_txt
    return plan


def describe_order(o):
    s = f"{'VENTE' if o['side'] == 'sell' else 'ACHAT'} {o['quantity']} {o['instrument']} ({o['value']:.0f} EUR"
    if o.get("fee"):
        s += f", frais ~{o['fee']:.0f} EUR"
    return s + ")"


def exposure_after(plan, state):
    """Exposition reelle apres les ordres du plan qui ont abouti (executed / dry-run)."""
    pos = state["positions"]
    val = {k: pos[k]["shares"] * pos[k]["price"] for k in INSTRUMENTS}
    cash = max(0.0, float(state["cash"]))
    for o in plan["orders"]:
        if o.get("status") not in ("executed", "dry-run"):
            continue
        q = o.get("executed_quantity", o["quantity"])
        amt = q * o["price"]
        if o["side"] == "sell":
            val[o["instrument"]] -= amt
            cash += amt * (1 - SELL_FEE)
        else:
            val[o["instrument"]] += amt
            cash -= amt * (1 + BUY_FEE)
    V = val["PUST"] + val["LQQ"] + cash
    return (val["PUST"] + 2 * val["LQQ"]) / V if V > 0 else 0.0


def balances_after(plan, state):
    """(titres, especes) apres les ordres du plan qui ont abouti (executed / dry-run),
    valorises au cours lu ce matin, frais deduits du cash."""
    stocks, cash = float(state["stocks"]), float(state["cash"])
    for o in plan["orders"]:
        if o.get("status") not in ("executed", "dry-run"):
            continue
        amt = o.get("executed_quantity", o["quantity"]) * o["price"]
        if o["side"] == "sell":
            stocks -= amt
            cash += amt * (1 - SELL_FEE)
        else:
            stocks += amt
            cash -= amt * (1 + BUY_FEE)
    return stocks, cash


def balance_lines(stocks, cash, stocks_after=None, cash_after=None, indent="  "):
    """Titres / especes / total : avant ET apres execution s'il y a eu execution
    (stocks_after renseigne), sinon une seule ligne."""
    def one(t, c):
        return f"titres {_eur(t)} EUR | especes {_eur(c)} EUR | total {_eur(t + c)} EUR"
    if stocks_after is None:
        return f"{indent}{one(stocks, cash)}\n"
    return (f"{indent}Avant execution : {one(stocks, cash)}\n"
            f"{indent}Apres execution : {one(stocks_after, cash_after)}\n")


# ── Execution ─────────────────────────────────────────────

def execute_order(instrument, side, quantity, dry_run=True, tolerance=LIMIT_TOLERANCE_PCT, account=None):
    """Execute order via bourso-cli, sur le PEA du compte `account` (None -> slot 1).

    Ordre LIMITE avec tolerance : la limite = cours ± tolerance% (achat +, vente -),
    ce qui tampon le gap d'ouverture et fiabilise le remplissage tout en bornant le
    prix. tolerance=None -> ordre limite pile au cours (ancien comportement)."""
    from src.bourso.prepare import SYMBOLS, _run_cli_raw

    tol_txt = f" (limite ±{tolerance}%)" if tolerance is not None else ""
    action = f"{'ACHAT' if side == 'buy' else 'VENTE'} {quantity}x {instrument}{tol_txt}"

    if dry_run:
        print(f"  [DRY-RUN] {action}")
        return {"status": "dry-run", "instrument": instrument, "side": side, "quantity": quantity,
                "order_type": "LIM", "tolerance": tolerance}

    from src.bourso.accounts import resolve_pea
    account = resolve_pea(account or _default_account())
    print(f"  [EXECUTE] {action} — {account.label}")
    cli_args = [
        "trade", "order", "new",
        "--side", side,
        "--account", account.pea_account_id,
        "--symbol", SYMBOLS[instrument],
        "--quantity", str(quantity),
        "--order-type", "LIM",
    ]
    if tolerance is not None:
        cli_args += ["--tolerance", str(tolerance)]
    stdout, stderr, rc = _run_cli_raw(*cli_args, creds=account.creds)

    output = (stdout + stderr).strip()
    if rc != 0:
        print(f"  [ERREUR] bourso-cli code {rc}: {output}")
        return {"status": "error", "instrument": instrument, "side": side, "quantity": quantity,
                "order_type": "LIM", "tolerance": tolerance, "error": output}

    print(f"  [OK] {output}")
    return {"status": "executed", "instrument": instrument, "side": side, "quantity": quantity,
            "order_type": "LIM", "tolerance": tolerance, "output": output}


def _run_order(o, account, execute):
    try:
        res = retry(lambda: execute_order(o["instrument"], o["side"], o["quantity"],
                                          dry_run=not execute, account=account),
                    label=f"Execution {o['side']} {o['instrument']} {account.label}")
    except Exception as e:  # noqa: BLE001
        print(f"[ERREUR] Ordre echoue: {e}")
        res = {"status": "error", "error": str(e)}
    o["status"] = res.get("status", "error")
    o["result"] = res
    if o["status"] == "error":
        o["error"] = res.get("error", "")[:300]
    return o


def execute_plan(plan, state, account, execute):
    """Execute le plan : VENTES d'abord, puis attente du credit des ventes (cash temps
    reel), puis ACHATS plafonnes par le cash disponible au prix reserve par Bourso
    (limite + provision). Renvoie True si des achats restent en attente de cash
    (produit des ventes non credite, ou refus 'solde especes insuffisant') — ils sont
    repris toutes les heures puis reportes au lendemain."""
    sells = [o for o in plan["orders"] if o["side"] == "sell"]
    buys = [o for o in plan["orders"] if o["side"] == "buy"]
    for o in sells:
        _run_order(o, account, execute)
    if not buys:
        return False

    cash_avail = max(0.0, float(state["cash"]))
    sold = [o for o in sells if o["status"] in ("executed", "dry-run")]
    if not execute:
        # dry-run : on suppose les ventes creditees (produit net de frais)
        cash_avail += sum(o["value"] * (1 - SELL_FEE) for o in sold)
    elif sold:
        expected = sum(o["value"] * (1 - SELL_FEE) for o in sold)
        cash0 = float(state["cash"])

        def _wait_cash():
            st = get_pea_state(account, quiet=True)
            gained = st["cash"] - cash0
            if gained < SELL_CREDIT_MIN_FRAC * expected:
                raise RuntimeError(f"ventes non encore creditees (cash {st['cash']:.0f} EUR, "
                                   f"+{gained:.0f} / attendu +{expected:.0f} EUR)")
            return st

        try:
            st = retry(_wait_cash, label=f"Attente credit ventes {account.label}")
            cash_avail = max(0.0, st["cash"])
        except Exception as e:  # noqa: BLE001
            print(f"[ATTENTE] {e} -> achats reportes (reprise horaire, puis lendemain)")
            for o in buys:
                o["status"] = "pending_cash"
            return True

    pending = False
    for o in buys:
        # plafond au prix RESERVE par Bourso (limite +3% + provision), cf. BUY_CASH_MARGIN
        q = min(o["quantity"], int(cash_avail / buy_reserve_price(o["price"]) + 1e-9))
        if q <= 0:
            o["status"] = "skipped"
            o["error"] = "cash insuffisant"
            print(f"  [SKIP] {describe_order(o)} : cash insuffisant ({cash_avail:.0f} EUR)")
            continue
        if q * o["price"] < MIN_BUY_EUR:
            # reduit par le cash sous le minimum Bourso (200 EUR) : on n'envoie pas, cash conserve
            o["status"] = "skipped"
            o["error"] = f"montant {q * o['price']:.0f} EUR < minimum {MIN_BUY_EUR:.0f} EUR (conserve en cash)"
            print(f"  [SKIP] {describe_order(o)} : {o['error']}")
            continue
        if q < o["quantity"]:
            print(f"  [INFO] achat {o['instrument']} reduit {o['quantity']} -> {q} parts (cash {cash_avail:.0f} EUR)")
            o["executed_quantity"] = q
            o["quantity"] = q
            o["value"] = q * o["price"]
            o["fee"] = o["value"] * BUY_FEE
        _run_order(o, account, execute)
        if o["status"] in ("executed", "dry-run"):
            cash_avail -= o["value"] * (1 + BUY_FEE)
        elif o["status"] == "error" and is_cash_refusal(o.get("error")):
            # Refus 'solde especes insuffisant' : pas une erreur definitive -> achat
            # differe, re-planifie sur l'etat frais (reprise horaire, puis lendemain).
            o["status"] = "pending_cash"
            pending = True
            print(f"  [ATTENTE] {describe_order(o)} : refus Bourso 'solde especes insuffisant' "
                  f"-> achat differe (re-dimensionne a la prochaine tentative)")
        elif o["status"] == "error" and is_min_amount_refusal(o.get("error")):
            # Refus 5110 'montant minimum 200 EUR' (cours bouge sous le plancher entre le
            # dimensionnement et l'envoi) : pas une erreur, le cash reste sur le compte.
            o["status"] = "skipped"
            print(f"  [SKIP] {describe_order(o)} : refus Bourso 'montant minimum {MIN_BUY_EUR:.0f} EUR' "
                  f"-> conserve en cash")
    return pending


# ── Achats reportes (cash non credite avant la deadline) ──

def load_pending():
    if not PENDING_FILE.exists():
        return {}
    try:
        return json.loads(PENDING_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def set_pending(slot, info=None):
    """info=None -> efface l'entree du slot."""
    data = load_pending()
    if info is None:
        data.pop(str(slot), None)
    else:
        data[str(slot)] = info
    PENDING_FILE.write_text(json.dumps(data, indent=2))


# ── Journal, split, mails ─────────────────────────────────

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


# Mails du matin : plus d'envoi par compte. Chaque compte AJOUTE son rapport a ce
# tampon ; main() envoie UN mail de synthese des PEA (a BOURSO_MAIL_1) apres chaque
# passe sur les comptes (puis apres chaque relance horaire eventuelle).
_MAIL_DIGEST = []

# Les mails ne parlent que d'EXPOSITION cible (deduite de l'allocation du signal :
# min(2 x allocation, E_MAX)) ; cette note rappelle qu'elle inclut le levier.
EXPO_NOTE = (f"levier inclus : 100% = PUST seul, au-dela la part > 100% est portee par "
             f"LQQ x2, plafond {E_MAX*100:.0f}%")

# Regle de la bande (rappelee dans les mails du soir et du matin) — seuils de strategy.py.
BAND_NOTE = (
    f"Regle d'execution (bande asymetrique, ecart = expo cible - expo reelle de chaque PEA) :\n"
    f"    - ACHAT si la cible depasse l'expo reelle d'au moins {BUY_THR_E*100:.0f} pts\n"
    f"    - VENTE seulement si l'expo reelle depasse la cible d'au moins {SELL_THR_E*100:.0f} pts\n"
    f"      (ou si la cible est forcee a 0% : garde-fou macro / arret d'urgence)\n"
    f"    - sinon : aucun ordre, le PEA garde son exposition\n"
    f"    Consequence : sans levier (expo ~100%), une baisse de cible ne vend rien sauf passage a 0%.\n"
    f"    Pourquoi : meilleur reglage net de frais du backtest 2000-2026 ; vendre plus tot (50-90 pts)\n"
    f"    ne reduit pas le drawdown et coute 2 a 3 pts de CAGR/an (frais + rebonds rates).")

# Ecart de cible (en points d'exposition) a partir duquel l'objet du mail annonce le
# changement « avant -> apres » (le corps l'affiche toujours).
TARGET_CHANGE_SUBJECT_PTS = 5


def target_change(signal, target_e=None):
    """Changement de cible veille -> signal du jour, lu dans `exposure_history` de
    signal.json (ecrit par risk_off_strategy.run). `target_e` = cible reellement
    appliquee si elle differe du signal (arret d'urgence / garde-fou -> 0).

    Renvoie (subject_part, line) :
      subject_part : "expo cible 102% -> 40%" si |ecart| >= TARGET_CHANGE_SUBJECT_PTS,
                     sinon "expo cible 40%" ;
      line         : "Exposition cible : avant 102% (signal du 2026-11-11) -> apres 40%
                     (signal du 2026-11-12), -62 pts" + l'historique des seances.
    Sans historique (ancien signal.json) : valeur seule, comme avant."""
    new = signal.get("exposure", 0.0) if target_e is None else target_e
    hist = [h for h in signal.get("exposure_history") or [] if h.get("date") != signal.get("date")]
    if not hist:
        return f"expo cible {new*100:.0f}%", f"Exposition cible : {new*100:.0f}%"
    prev = hist[-1]
    d = round(new * 100) - round(prev["exposure"] * 100)
    if d == 0:
        line = (f"Exposition cible : {new*100:.0f}% — inchangee "
                f"(signal du {prev['date']} : {prev['exposure']*100:.0f}%)")
    else:
        line = (f"Exposition cible : avant {prev['exposure']*100:.0f}% (signal du {prev['date']})"
                f" -> apres {new*100:.0f}% (signal du {signal.get('date')}), {d:+d} pts")
    seq = " / ".join(f"{h['exposure']*100:.0f}%" for h in hist[-4:]) + f" / {new*100:.0f}%"
    line += f"\n  Cible des dernieres seances : {seq}"
    subj = (f"expo cible {prev['exposure']*100:.0f}% -> {new*100:.0f}%"
            if abs(d) >= TARGET_CHANGE_SUBJECT_PTS else f"expo cible {new*100:.0f}%")
    return subj, line


def _send_account_email(account, subject, body, short=None, stocks=None, cash=None,
                        e_before=None, e_after=None, n_done=0, n_error=0,
                        stocks_after=None, cash_after=None):
    """Rapport du compte -> tampon du mail de synthese (cf. flush_mail_digest).
    stocks / cash = titres / especes lus avant ordres ; *_after = apres execution
    (seulement s'il y a eu execution). n_done / n_error = ordres passes / en erreur."""
    _MAIL_DIGEST.append({"account": account, "subject": subject, "body": body,
                         "short": short or subject.replace("[MyQTM] ", ""), "stocks": stocks,
                         "cash": cash, "stocks_after": stocks_after, "cash_after": cash_after,
                         "e_before": e_before, "e_after": e_after,
                         "n_done": n_done, "n_error": n_error})


def _eur(x):
    return f"{x:,.0f}".replace(",", " ")


def digest_mail(entries, mode, target_e, signal, notes=()):
    """(sujet, corps) du mail de synthese : un tableau d'etat des PEA + totaux, puis le
    detail de chaque compte. Pur (teste sans envoi)."""
    lines, n_read, any_exec = [], 0, False
    tot = {"stocks": 0.0, "cash": 0.0, "stocks_after": 0.0, "cash_after": 0.0}
    for e in entries:
        acc = e["account"]
        name = f"compte {acc.slot} — {acc.name or acc.label}"
        if e.get("stocks") is not None:
            n_read += 1
            executed = e.get("stocks_after") is not None
            any_exec = any_exec or executed
            tot["stocks"] += e["stocks"]
            tot["cash"] += e["cash"] or 0.0
            tot["stocks_after"] += e["stocks_after"] if executed else e["stocks"]
            tot["cash_after"] += e["cash_after"] if executed else (e["cash"] or 0.0)
            expo = (f"expo {e['e_before']*100:.0f}% -> {e['e_after']*100:.0f}%"
                    if e["e_before"] is not None else "expo --")
            lines.append(f"  {name}\n      {expo} — {e['short']}\n"
                         + balance_lines(e["stocks"], e["cash"] or 0.0, e.get("stocks_after"),
                                         e.get("cash_after"), indent="      ").rstrip("\n"))
        else:
            lines.append(f"  {name}\n      {e['short']}")
    rsi = signal.get("rsi14")
    t_subj, t_line = target_change(signal, target_e)
    head = (f"Synthese des {len(entries)} PEA — {date.today()} ({mode})\n"
            f"Changement de cible applique ce matin (annonce dans le recap d'hier soir ; les ordres dependent de la bande) :\n"
            f"  {t_line}\n"
            f"  (signal du {signal['date']}" + (f", RSI14 {rsi:.1f}" if rsi is not None else "")
            + f" ; {EXPO_NOTE})\n"
            f"  {BAND_NOTE}\n\n")
    body = head + "\n".join(lines) + "\n"
    if n_read:
        body += f"\n  TOTAL {n_read} PEA\n" + balance_lines(
            tot["stocks"], tot["cash"],
            tot["stocks_after"] if any_exec else None, tot["cash_after"] if any_exec else None,
            indent="      ")
    for n in notes:
        body += f"\n  {n}"
    if notes:
        body += "\n"
    body += "\n" + "=" * 60 + "\nDETAIL PAR COMPTE\n" + "=" * 60 + "\n"
    body += ("\n" + "-" * 60 + "\n").join(e["body"] for e in entries)
    # Objet : expo cible puis expo REELLE de chaque PEA apres les ordres du jour
    # (KO = compte illisible ; les ordres et leur statut sont dans le corps).
    reals = " | ".join(f"c{e['account'].slot} " + (f"{e['e_after']*100:.0f}%" if e["e_after"] is not None else "KO")
                       for e in entries)
    done = [f"c{e['account'].slot}" for e in entries if e.get("n_done")]
    errs = [f"c{e['account'].slot}" for e in entries if e.get("n_error")]
    ops = (f"EXECUTION {' '.join(done)}" if done else "aucune execution")
    if errs:
        ops += f" — ORDRE EN ERREUR {' '.join(errs)}"
    subject = (f"[MyQTM] {mode} {len(entries)} PEA — {t_subj} — "
               f"expo reelle {reals} — {ops}")
    return subject, body


def flush_mail_digest(mode, target_e, signal, notes=()):
    """Envoie le mail de synthese des rapports en tampon (BOURSO_MAIL_1) — jamais bloquant."""
    if not _MAIL_DIGEST:
        return
    entries = list(_MAIL_DIGEST)
    _MAIL_DIGEST.clear()
    subject, body = digest_mail(entries, mode, target_e, signal, notes)
    try:
        from src.bourso.notify import send_email
        send_email(subject, body)
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] Email de synthese non envoye: {e}")


def _base_record(account, signal, target_e, execute):
    return {
        "date": str(date.today()), "model_date": signal["date"],
        "probability": signal["probability"], "target_alloc": signal["allocation"],
        "target_exposure": target_e, "e_max": E_MAX, "rsi14": signal.get("rsi14"),
        "instrument": STRATEGY_LABEL, "leverage": E_MAX,
        "account": account.slot, "account_name": account.name,
        "executed": execute,
    }


def _positions_record(state):
    return {k: {"shares": p["shares"], "price": p["price"]} for k, p in state["positions"].items()}


def composition_line(state):
    pos = state["positions"]
    cash = max(0.0, state["cash"])
    V = sum(p["value"] for p in pos.values()) + cash
    if V <= 0:
        return "PEA vide"
    parts = [f"{k} {p['shares']} parts ({p['value']/V*100:.0f}%)" for k, p in pos.items()]
    return " | ".join(parts) + f" | especes {cash/V*100:.0f}%"


def plan_for_account(account, target_e, state=None):
    """Lecture + plan SANS effet de bord (apercu du soir) : un apport deja visible sur le
    compte est detecte (sur une copie de l'etat capital, rien n'est persiste) et inclus
    dans l'apercu, comme le ferait le run du matin. Renvoie (state, plan)."""
    import copy
    from src.bourso.capital import load_capital_state, get_entry, detect_deposit
    state = state or get_pea_state(account)
    entry = copy.deepcopy(get_entry(load_capital_state(), account.slot))
    deposit, _ = detect_deposit(entry, state["cash"], state["equity"])
    pending = load_pending().get(str(account.slot))
    plan = compute_orders(target_e, state, deposit=deposit, force_buy=pending is not None)
    return state, plan


def process_account(account, signal, target_e, execute, force_buy=False, buys_only=False):
    """Lit le PEA du compte, detecte un apport (investi le jour meme a la composition
    cible), decide, execute (ventes,
    attente du cash, achats), journalise et envoie le mail du compte. Leve si le PEA
    reste illisible (le mail d'echec est envoye par main).

    Renvoie {plan, state, pending} (pending = achats en attente de cash)."""
    from src.bourso.capital import (
        load_capital_state, save_capital_state, get_entry, detect_deposit, record_expected_cash,
    )
    mode = "LIVE" if execute else "DRY-RUN"
    print(f"\n{'-'*60}\n  {account.label.upper()}\n{'-'*60}")

    # 1. Etat du PEA (cash, positions) — backoff court ; la relance horaire est geree
    #    par main() sur l'ensemble des comptes en echec (un compte KO ne bloque pas
    #    les autres pendant des heures).
    state = retry(lambda: get_pea_state(account), label=f"PEA state {account.label}")
    if state["equity"] <= 0:
        raise RuntimeError("PEA vide (equity=0)")
    today = date.today()

    # 2. Achat reporte d'une veille (cash non credite) -> achats sans bande aujourd'hui
    pending_prev = load_pending().get(str(account.slot))
    if pending_prev and pending_prev.get("date") != str(today):
        print(f"  [REPRISE] achat reporte du {pending_prev.get('date')} : achats sans bande")
        force_buy = True

    # 3. Apport / retrait de capital (ecart cash lu vs cash attendu, persiste dans
    #    logs/capital.json en LIVE seulement : un dry-run ne deplace pas la reference).
    #    Un apport est investi AUJOURD'HUI (lendemain du virement) en entier, a la
    #    composition cible du signal — plus de DCA ni de reserve.
    cap_state = load_capital_state()
    entry = get_entry(cap_state, account.slot)
    deposit, withdrawal = detect_deposit(entry, state["cash"], state["equity"], today)
    if deposit > 0:
        print(f"\n  *** APPORT DE CAPITAL DETECTE : +{deposit:.0f} EUR -> investi aujourd'hui a la composition cible ***")
    if withdrawal > 0:
        print(f"\n  [INFO] retrait detecte : -{withdrawal:.0f} EUR")
    capital_info = {"deposit_today": round(deposit, 2), "withdrawal_today": round(withdrawal, 2)}

    # 4. Detection de split / anomalie de prix (par instrument) -> aucune position aujourd'hui
    splits = {}
    for ins, p in state["positions"].items():
        prev = load_last_price(ins, account.slot)
        ratio = detect_split(prev, p["price"])
        if execute:
            save_last_price(ins, p["price"], account.slot)
        if ratio is not None:
            splits[ins] = (prev, p["price"], ratio)
    if splits:
        det = " ; ".join(f"{ins} {pv:.2f} -> {cur:.2f} EUR (x{r:.1f})" for ins, (pv, cur, r) in splits.items())
        msg = (f"SPLIT / anomalie de prix detecte : {det}. "
               f"Aucune position prise aujourd'hui — on attend la prochaine seance.")
        print(f"\n{'!'*60}\n  {msg}\n{'!'*60}")
        _send_account_email(account,
            f"[MyQTM] SPLIT detecte — aucune position prise — {account.name}",
            f"{account.name} (compte {account.slot}) — {today}\n\n"
            f"  Connexion:   OK\n"
            f"  ⚠️ {msg}\n\n"
            f"  Action:      AUCUNE (garde-fou split) — reprise a la prochaine seance\n"
            f"  Mode:        {mode}\n",
            short="SPLIT detecte — aucune position", stocks=state["stocks"], cash=state["cash"],
            e_before=state["exposure"], e_after=state["exposure"])
        rec = _base_record(account, signal, target_e, execute)
        rec.update(side=None, quantity=0, orders=[], positions=_positions_record(state),
                   cash=state["cash"], equity=state["equity"], exposure_before=state["exposure"],
                   exposure_after=state["exposure"], capital=capital_info,
                   reason=f"split detecte ({det}) — no trade",
                   result={"status": "split_detected", "splits": {k: list(v) for k, v in splits.items()}})
        log_trade(rec)
        if execute and deposit <= 0:
            # un apport du jour n'est PAS acte : il sera re-detecte (et investi) au
            # prochain run, au lieu de rester en cash oisif
            record_expected_cash(entry, state["cash"], [], today)
            save_capital_state(cap_state)
        return {"plan": None, "state": state, "pending": False}

    # 5. Decision (bande asymetrique sur l'exposition reelle de CE compte)
    plan = compute_orders(target_e, state, deposit=deposit, force_buy=force_buy, buys_only=buys_only)
    print(f"\nDecision: {plan['reason']}")

    # 6. Execution — ventes, attente du cash, achats (backoff court par ordre)
    pending = False
    if plan["orders"]:
        pending = execute_plan(plan, state, account, execute)
    if execute:
        if pending:
            set_pending(account.slot, {"date": str(today), "target_exposure": target_e,
                                       "buys": [{"instrument": o["instrument"], "quantity": o["quantity"]}
                                                for o in plan["orders"] if o["side"] == "buy"],
                                       "reason": "cash non credite ou refus Bourso (solde especes insuffisant)"})
        else:
            set_pending(account.slot, None)
        # cash attendu a la prochaine lecture (detection d'apport) ; en cas d'achats
        # reportes, le produit des ventes est deja compte dans le cash attendu.
        record_expected_cash(entry, state["cash"], plan["orders"], today)
        save_capital_state(cap_state)

    # 7. Mail du compte : connexion + exposition + composition + action (tous les jours)
    e_after = exposure_after(plan, state)
    done = [o for o in plan["orders"] if o.get("status") in ("executed", "dry-run")]
    errors = [o for o in plan["orders"] if o.get("status") == "error"]
    waiting = [o for o in plan["orders"] if o.get("status") == "pending_cash"]
    if plan["orders"]:
        action_lines = [f"{describe_order(o)} — {str(o.get('status', '?')).upper()}"
                        + (f" : {o.get('error', '')[:160]}" if o.get("status") in ("error", "skipped") else "")
                        for o in plan["orders"]]
        summary = " + ".join(f"{'VENTE' if o['side'] == 'sell' else 'ACHAT'} {o['quantity']}x {o['instrument']}"
                             for o in plan["orders"])
        if errors:
            subject_tail = "ORDRE EN ERREUR " + summary
        elif waiting:
            subject_tail = "ACHAT DIFFERE (cash en attente) " + summary
        else:
            subject_tail = summary
    else:
        action_lines = [f"aucun ordre ({plan['reason']})"]
        subject_tail = "aucun changement"
    if capital_info["deposit_today"] > 0:
        subject_tail = f"APPORT +{capital_info['deposit_today']:.0f} EUR — " + subject_tail
    tw = plan.get("target_weights") or dict(zip(("PUST", "LQQ", "cash"), composition(target_e)))
    rsi = signal.get("rsi14")
    rsi_txt = f", RSI14 {rsi:.1f}" if rsi is not None else ""
    subject = f"[MyQTM] {mode} {account.name} — expo cible {target_e*100:.0f}% — {subject_tail}"
    body = (
        f"{account.name} (compte {account.slot}) — {today}\n\n"
        f"  Connexion:   OK\n"
        f"  Strategie:   {STRATEGY_LABEL} (PUST x1 + LQQ x2), exposition plafonnee x{E_MAX}\n"
        f"  Signal du:   {signal['date']}{(' (' + rsi_txt[2:] + ')') if rsi_txt else ''}\n\n"
    )
    body += (
        f"  Exposition cible:  {target_e*100:.0f}%  ({EXPO_NOTE})\n"
        f"  Exposition reelle: {plan['e_eff']*100:.0f}%  ->  {e_after*100:.0f}% apres ordres du jour\n"
        f"  Composition:  {composition_line(state)}\n"
        f"  Cible:        PUST {tw['PUST']*100:.0f}% | LQQ {tw['LQQ']*100:.0f}% | especes {tw['cash']*100:.0f}%\n\n"
        f"  Action:      " + "\n               ".join(action_lines) + "\n"
        + (f"  Type ordre:  limite ±{LIMIT_TOLERANCE_PCT}% (tampon d'ouverture), ventes puis achats\n" if plan["orders"] else "")
    )
    if capital_info["deposit_today"] > 0:
        body += (f"\n  *** APPORT DE CAPITAL DETECTE : +{capital_info['deposit_today']:.0f} EUR ***\n"
                 f"  Investi en entier aujourd'hui a la composition cible du signal (sans bande).\n")
    if capital_info["withdrawal_today"] > 0:
        body += f"\n  Retrait detecte : -{capital_info['withdrawal_today']:.0f} EUR\n"
    s_after, c_after = balances_after(plan, state) if done else (None, None)
    body += "\n" + balance_lines(state["stocks"], state["cash"], s_after, c_after)
    body += f"  Mode:        {mode}\n"
    _send_account_email(account, subject, body, short=subject_tail, stocks=state["stocks"],
                        cash=state["cash"], e_before=plan["e_eff"], e_after=e_after,
                        n_done=len(done), n_error=len(errors),
                        stocks_after=s_after, cash_after=c_after)

    # 8. Journal
    sides = {o["side"] for o in plan["orders"]}
    side = "rebalance" if len(sides) == 2 else (sides.pop() if sides else None)
    if errors:
        status = "error"
    elif waiting:
        status = "pending_cash"
    elif done:
        status = "executed" if execute else "dry-run"
    else:
        status = None
    rec = _base_record(account, signal, target_e, execute)
    rec.update(side=side, quantity=sum(o["quantity"] for o in plan["orders"]),
               orders=[{k: o.get(k) for k in ("instrument", "side", "quantity", "price", "value", "fee", "status", "error")}
                       for o in plan["orders"]],
               positions=_positions_record(state), cash=state["cash"], equity=state["equity"],
               exposure_before=plan["e_eff"], exposure_after=e_after,
               mode=plan["mode"], capital=capital_info, reason=plan["reason"],
               result=None if status is None else {"status": status,
                                                   "errors": [o.get("error") for o in errors] or None})
    log_trade(rec)
    return {"plan": plan, "state": state, "pending": pending}


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
    print(f"PEA {STRATEGY_LABEL} (PUST x1 + LQQ x2, plafond d'expo x{E_MAX}) — {date.today()}")
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

    # 2. Exposition cible (plafond E_MAX) + emergency override (global : tous les comptes)
    target_e = target_exposure(signal["allocation"], E_MAX)
    if is_emergency_off():
        target_e = 0.0

    # 3. Chaque compte a son tour ; ceux en echec (PEA illisible) ou dont les achats
    #    attendent le credit des ventes sont relances toutes les heures jusqu'a la
    #    cloture Euronext, sans bloquer les autres.
    euronext_close = datetime.now().replace(hour=17, minute=0, second=0)
    mode = "LIVE" if args.execute else "DRY-RUN"
    pending = [(a, False) for a in accounts]
    failures = {}
    while pending:
        failures = {}
        waiting = []
        for account, resume in pending:
            try:
                out = process_account(account, signal, target_e, args.execute,
                                      force_buy=resume, buys_only=resume)
                if out["pending"]:
                    waiting.append((account, True))
            except Exception as e:  # noqa: BLE001
                print(f"[ERREUR] {account.label}: {e}")
                failures[account.slot] = (account, e)
        pending = [(a, False) for a, _ in failures.values()] + waiting
        retrying = bool(pending) and datetime.now() < euronext_close
        notes = []
        if retrying:
            notes = ([f"{a.label} : connexion en echec ({str(e)[:120]}) — nouvelle tentative dans 1h"
                      for a, e in failures.values()]
                     + [f"{a.label} : achats en attente de cash — nouvelle tentative dans 1h"
                        for a, _ in waiting])
        flush_mail_digest(mode, target_e, signal, notes)
        if not retrying:
            break
        print(f"\n[RETRY] {len(failures)} compte(s) en echec, {len(waiting)} en attente de cash, "
              f"nouvelle tentative dans 1h...")
        time.sleep(3600)

    # 4. Comptes definitivement en echec : "connexion KO" (mail de synthese) + journal
    for account, err in failures.values():
        msg = str(err)[:300]
        _send_account_email(account,
            f"[MyQTM] {'LIVE' if args.execute else 'DRY-RUN'} {account.name or account.label} — CONNEXION KO",
            f"{account.name or account.label} (compte {account.slot}) — {date.today()}\n\n"
            f"  Connexion:   ECHEC — {msg}\n"
            f"  Exposition cible: {target_e*100:.0f}%  (signal du {signal['date']})\n"
            f"  Action:      AUCUNE — le compte n'a pas pu etre lu, aucun ordre passe.\n"
            f"  Verifier logs/cron_pea.log.\n",
            short="CONNEXION KO — aucun ordre")
        rec = _base_record(account, signal, target_e, args.execute)
        rec.update(side=None, quantity=0, orders=[], positions=None, cash=0, equity=0,
                   reason=f"connexion KO: {msg}", result={"status": "connection_error", "error": msg})
        log_trade(rec)

    flush_mail_digest(mode, target_e, signal)
    print("\nTermine." + (f"  ({len(failures)} compte(s) en echec)" if failures else ""))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
