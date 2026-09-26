"""
Risk-Off Strategy Dashboard — NiceGUI webapp.

Layout inspired by dashboard/ — sidebar photo + tabbed content.

Shows:
  - Backtests (full, 1y, 1m)
  - Real Bourso allocations & trade history
  - Gains normalized (% and EUR)

Usage:  python src/webapp.py
"""

import asyncio
import json
import os
import secrets
from datetime import datetime
from pathlib import Path

import plotly.graph_objects as go
from fastapi import Request
from fastapi.responses import FileResponse, RedirectResponse
from nicegui import app, ui
from starlette.middleware.base import BaseHTTPMiddleware

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
QQQ_OUT = OUTPUTS / "qqq_strategy"
TRADE_LOG = ROOT / "logs" / "trades.jsonl"
ASSETS_DIR = ROOT / "assets"
SIDEBAR_IMAGE = ASSETS_DIR / "sidebar.jpg"

# Backtest images
BACKTEST_FULL = QQQ_OUT / "backtest.png"
BACKTEST_10Y = QQQ_OUT / "backtest_10y.png"
BACKTEST_5Y = QQQ_OUT / "backtest_5y.png"
BACKTEST_1Y = QQQ_OUT / "backtest_1y.png"
BACKTEST_1M = QQQ_OUT / "backtest_1m.png"
CAPE_CHART = OUTPUTS / "shiller" / "cape_ecy.png"


# ── Auth (mot de passe unique) ───────────────────────────
# WEBAPP_PASSWORD : mot de passe d'acces (env ou .env). Le webapp refuse de
# demarrer sans, pour ne jamais tourner ouvert par accident (le bouton Emergency
# OFF peut liquider le PEA). WEBAPP_SECRET : cle de signature du cookie de session
# (optionnel ; sans, une cle aleatoire par demarrage -> re-login apres restart).
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

WEBAPP_PASSWORD = os.environ.get("WEBAPP_PASSWORD", "")
if not WEBAPP_PASSWORD:
    raise SystemExit("WEBAPP_PASSWORD manquant (env ou .env) — refus de demarrer sans mot de passe")
WEBAPP_SECRET = os.environ.get("WEBAPP_SECRET") or secrets.token_hex(32)

UNRESTRICTED_PREFIXES = ("/login", "/_nicegui", "/assets/")


class AuthMiddleware(BaseHTTPMiddleware):
    """Redirige vers /login toute requete (pages ET images /img/*) non authentifiee."""

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if not path.startswith(UNRESTRICTED_PREFIXES) and not app.storage.user.get("authenticated", False):
            return RedirectResponse("/login")
        return await call_next(request)


app.add_middleware(AuthMiddleware)


@ui.page("/login", title="Connexion")
async def login_page():
    if app.storage.user.get("authenticated", False):
        return RedirectResponse("/")

    async def try_login():
        if secrets.compare_digest(pwd.value or "", WEBAPP_PASSWORD):
            app.storage.user["authenticated"] = True
            ui.navigate.to("/")
        else:
            await asyncio.sleep(1)  # freine le brute-force
            pwd.value = ""
            ui.notify("Mot de passe incorrect", type="negative")

    ui.add_head_html("""<style>
        body { background: #1a1a1a; font-family: 'Inter', system-ui, sans-serif; }
    </style>""")
    with ui.card().classes("absolute-center items-center").style(
            "width: 320px; padding: 32px 28px; background: #262626; color: #ddd; border-radius: 12px"):
        # Volontairement sans titre ni sous-titre : ne rien reveler sur l'application
        # a qui n'a pas le mot de passe (le titre d'onglet est neutre aussi, cf. @ui.page).
        pwd = ui.input("Mot de passe", password=True, password_toggle_button=True) \
            .props("dark outlined autofocus").classes("w-full") \
            .on("keydown.enter", try_login)
        ui.button("Entrer", on_click=try_login).props("color=green unelevated").classes("w-full mt-2")


# ── Data loaders ──────────────────────────────────────────

SHILLER_DIR = ROOT / "data" / "shiller"


def load_cape_ecy_latest():
    """Dernier CAPE / ECY (S&P, Shiller) + rang percentile du CAPE sur tout
    l'historique. None si le parquet est absent (lancer download_shiller_cape)."""
    fp = SHILLER_DIR / "cape_ecy.parquet"
    if not fp.exists():
        return None
    import pandas as pd
    df = pd.read_parquet(fp)
    cape = df["cape"].dropna()
    ecy = df["ecy"].dropna()
    if cape.empty:
        return None
    return {"cape": float(cape.iloc[-1]),
            "ecy": float(ecy.iloc[-1]) if not ecy.empty else None,
            "date": str(cape.index.max().date()),
            "cape_pct": float((cape <= cape.iloc[-1]).mean() * 100),
            "cape_median": float(cape.median())}


def load_trades():
    if not TRADE_LOG.exists():
        return []
    trades = []
    with open(TRADE_LOG) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    trades.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return trades


ACCOUNTS_CACHE = ROOT / "logs" / "accounts.json"


def load_account_names():
    """{slot: {name, mail, ...}} ecrit par src/bourso/accounts.py (decouverte des PEA).
    La webapp n'a pas acces au .env Bourso : c'est sa seule source de noms de comptes."""
    if not ACCOUNTS_CACHE.exists():
        return {}
    try:
        return {int(k): v for k, v in json.loads(ACCOUNTS_CACHE.read_text()).items() if k.isdigit()}
    except (json.JSONDecodeError, ValueError, OSError):
        return {}


def _slot(t):
    """Slot du compte d'une ligne de trades.jsonl (anciennes lignes mono-compte -> 1)."""
    return int(t.get("account") or 1)


def _is_connection_error(t):
    return ((t.get("result") or {}).get("status") == "connection_error")


def split_by_account(trades):
    """Comptes -> {slot: {"name", "trades" (lignes exploitables), "last" (derniere ligne,
    y compris un echec de connexion), "mail"}}, tries par slot. Les echecs de
    connexion (equity=0, pas de prix) sont exclus des historiques/perf mais gardes
    comme dernier statut."""
    names = load_account_names()
    accounts = {}
    for t in trades:
        slot = _slot(t)
        acc = accounts.setdefault(slot, {"slot": slot, "trades": [], "last": None,
                                         "name": "", "mail": ""})
        acc["last"] = t
        if not _is_connection_error(t):
            acc["trades"].append(t)
        if t.get("account_name"):
            acc["name"] = t["account_name"]
    for slot, info in names.items():
        acc = accounts.setdefault(slot, {"slot": slot, "trades": [], "last": None,
                                         "name": "", "mail": ""})
        acc["name"] = info.get("name") or acc["name"]
        acc["mail"] = info.get("mail", "")
    for acc in accounts.values():
        acc["name"] = acc["name"] or f"Compte {acc['slot']}"
    return dict(sorted(accounts.items()))


def account_status(acc):
    """Statut synthetique d'un compte a partir de sa derniere ligne de journal."""
    t = acc["last"]
    if t is None:
        return {"state": "unknown", "label": "JAMAIS EXECUTE", "color": "grey",
                "date": "--", "detail": "Aucune ligne dans trades.jsonl"}
    date = t.get("date", "--")
    mode = "LIVE" if t.get("executed") else "DRY-RUN"
    res = t.get("result") or {}
    if _is_connection_error(t):
        return {"state": "ko", "label": "CONNEXION KO", "color": "red", "date": date,
                "mode": mode, "detail": res.get("error") or t.get("reason", "")}
    if res.get("status") == "error":
        errs = [e for e in (res.get("errors") or []) if e] or [res.get("error", "")]
        return {"state": "error", "label": "ORDRE EN ERREUR", "color": "orange", "date": date,
                "mode": mode, "detail": " ; ".join(str(e)[:200] for e in errs)}
    if res.get("status") == "pending_cash":
        return {"state": "pending", "label": "ACHAT DIFFERE (cash en attente)", "color": "orange",
                "date": date, "mode": mode, "detail": t.get("reason", "")}
    if res.get("status") == "split_detected":
        return {"state": "split", "label": "SPLIT DETECTE", "color": "orange", "date": date,
                "mode": mode, "detail": t.get("reason", "")}
    return {"state": "ok", "label": "CONNEXION OK", "color": "green", "date": date,
            "mode": mode, "detail": t.get("reason", "")}


LEVERAGE = {"PUST": 1.0, "LQQ": 2.0}
INSTRUMENT_ORDER = ("PUST", "LQQ")
CAPITAL_FILE = ROOT / "logs" / "capital.json"
PENDING_FILE = ROOT / "logs" / "pending_orders.json"


def _load_json(path):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def load_capital_entry(slot):
    """Entree logs/capital.json du slot (apport detecte / DCA en cours), ou {}."""
    return _load_json(CAPITAL_FILE).get(str(slot), {})


def load_pending_entry(slot):
    """Achat reporte (cash non credite) en attente pour ce slot, ou None."""
    return _load_json(PENDING_FILE).get(str(slot))


def row_view(t):
    """Vue unifiee d'une ligne de trades.jsonl, ancien schema (un instrument :
    current_shares / etf_price / side / quantity) ou nouveau (PUST+LQQ : positions /
    orders / cash / exposure_*). positions = AVANT les ordres du jour."""
    equity = float(t.get("equity") or 0)
    if isinstance(t.get("positions"), dict):
        positions = {k: {"shares": int(v.get("shares") or 0), "price": float(v.get("price") or 0)}
                     for k, v in t["positions"].items()}
        orders = [{"instrument": o.get("instrument"), "side": o.get("side"),
                   "quantity": int(o.get("quantity") or 0), "price": float(o.get("price") or 0),
                   "status": o.get("status"),
                   "done": o.get("status") in ("executed", "dry-run") and bool(t.get("executed"))}
                  for o in (t.get("orders") or [])]
        cash = float(t.get("cash") if t.get("cash") is not None else
                     equity - sum(p["shares"] * p["price"] for p in positions.values()))
        legacy = False
    else:
        ins = t.get("instrument", "PUST")
        shares = int(t.get("current_shares") or 0)
        price = float(t.get("etf_price") or 0)
        positions = {ins: {"shares": shares, "price": price}}
        side = (t.get("side") or "").lower()
        orders = ([{"instrument": ins, "side": side, "quantity": int(t.get("quantity") or 0),
                    "price": price, "status": (t.get("result") or {}).get("status"),
                    "done": bool(t.get("executed"))}]
                  if side in ("buy", "sell") and (t.get("quantity") or 0) > 0 else [])
        cash = equity - shares * price
        legacy = True
    return {"date": t.get("date", ""), "equity": equity, "cash": cash, "positions": positions,
            "orders": orders, "legacy": legacy, "target_alloc": float(t.get("target_alloc") or 0),
            "target_exposure": t.get("target_exposure"), "probability": float(t.get("probability") or 0),
            "executed": bool(t.get("executed")), "capital": t.get("capital") or {},
            "reserved": float(t.get("reserved") or 0)}


def positions_after(view):
    """Positions apres les ordres du jour qui ont abouti (LIVE seulement)."""
    pos = {k: dict(v) for k, v in view["positions"].items()}
    cash = view["cash"]
    for o in view["orders"]:
        if not o["done"] or not o["instrument"]:
            continue
        p = pos.setdefault(o["instrument"], {"shares": 0, "price": o["price"]})
        if o["side"] == "buy":
            p["shares"] += o["quantity"]; cash -= o["quantity"] * o["price"]
        elif o["side"] == "sell":
            p["shares"] -= o["quantity"]; cash += o["quantity"] * o["price"] * (1 - SELL_FEE)
    return pos, cash


def exposure_of(pos, equity):
    if equity <= 0:
        return 0.0
    return sum(LEVERAGE.get(k, 1.0) * p["shares"] * p["price"] for k, p in pos.items()) / equity


def compute_portfolio_history(trades):
    """Historique position / exposition reelle, ligne par ligne de trades.jsonl.

    Les positions du journal sont lues par real_bourso via bourso-cli AVANT l'ordre
    du jour -> position reelle du compte, a laquelle on applique les ordres executes
    le jour meme (position post-ordre). Fonctionne pour l'ancien schema (un
    instrument) comme pour le nouveau (PUST + LQQ + cash, exposition plafonnee).
    """
    history = []
    for t in trades:
        v = row_view(t)
        pos, cash = positions_after(v)
        position_value = sum(p["shares"] * p["price"] for p in pos.values())
        equity = v["equity"]
        target_e = v["target_exposure"]
        if target_e is None:                      # anciennes lignes : expo = levier . alloc
            target_e = v["target_alloc"] * max([LEVERAGE.get(k, 1.0) for k in pos] or [1.0])
        history.append({
            "date": v["date"], "positions": pos, "cash": cash,
            "shares": sum(p["shares"] for p in pos.values()),
            "price": next((p["price"] for k, p in pos.items() if p["shares"] > 0), 0.0),
            "position_value": position_value, "equity": equity,
            "target_alloc": v["target_alloc"], "target_exposure": target_e,
            "actual_alloc": position_value / equity if equity > 0 else 0,
            "exposure": exposure_of(pos, equity),
            "weights": {k: (p["shares"] * p["price"] / equity if equity > 0 else 0) for k, p in pos.items()},
            "cash_weight": cash / equity if equity > 0 else 0,
            "probability": v["probability"],
            "side": (t.get("side") or "hold"),
            "orders": v["orders"], "executed": v["executed"],
            "capital": v["capital"], "reserved": v["reserved"],
        })
    return history


SELL_FEE = 0.005  # 0.5% sur les ventes (achats gratuits) — cf. real_bourso.py


def compute_real_performance(trades):
    """Performance réelle (time-weighted) du compte PEA.

    Utilise l'equity mesurée chaque matin (cash + titres) comme vérité terrain
    et neutralise les apports/retraits externes, pour que la courbe reflète le
    rendement de la stratégie sur le capital disponible à l'instant — et non
    l'effet d'un ajout d'argent. Les frais de vente restent comptés (coût réel).
    """
    # Ne garder que les vraies lectures du compte : equity > 0, et soit un ordre
    # exécuté, soit un snapshot d'état (side=None). Les plans dry-run (side=buy/sell,
    # executed=False) et placeholders sont écartés.
    raw = [t for t in trades
           if t.get("equity", 0) > 0 and t.get("date") and not _is_connection_error(t)
           and (t.get("executed") or t.get("side") is None)]
    # Un seul point faisant foi par date (la dernière lecture l'emporte)
    by_date = {}
    for t in raw:
        by_date[t["date"]] = row_view(t)
    points = [by_date[d] for d in sorted(by_date)]
    if len(points) < 2:
        return [], {}

    series = [{"date": points[0]["date"], "equity": points[0]["equity"],
               "cum_return": 0.0, "period_return": 0.0, "deposit": 0.0}]
    cum = 1.0
    total_gain_eur = 0.0
    total_deposits = 0.0
    prev = points[0]
    for t in points[1:]:
        equity = t["equity"]
        # Cash attendu = cash de la veille apres les flux de ses ordres executes
        _, expected_cash = positions_after(prev)
        # Écart cash inexpliqué = apport/retrait externe. Bruit = prix de remplissage
        # des ordres limites (±3%) : seuil 5% du notionnel traite, plancher 5 EUR.
        traded = sum(o["quantity"] * o["price"] for o in prev["orders"] if o["done"])
        deposit = t["cash"] - expected_cash
        if abs(deposit) < max(5.0, 0.05 * traded):
            deposit = 0.0

        period_gain = equity - prev["equity"] - deposit
        base = prev["equity"]
        r = period_gain / base if base > 0 else 0.0
        cum *= (1 + r)
        total_gain_eur += period_gain
        total_deposits += deposit
        series.append({"date": t["date"], "equity": equity,
                       "cum_return": cum - 1, "period_return": r, "deposit": deposit})
        prev = t

    try:
        d0 = datetime.strptime(points[0]["date"], "%Y-%m-%d")
        d1 = datetime.strptime(points[-1]["date"], "%Y-%m-%d")
        days = max((d1 - d0).days, 1)
        annualized = cum ** (365 / days) - 1
    except (ValueError, OverflowError):
        days, annualized = 0, 0.0

    stats = {
        "cum_return": cum - 1,
        "total_gain_eur": total_gain_eur,
        "start_equity": points[0]["equity"],
        "current_equity": points[-1]["equity"],
        "total_deposits": total_deposits,
        "annualized": annualized,
        "days": days,
    }
    return series, stats


# ── Serve images ──────────────────────────────────────────

def _serve(path):
    if not path.exists():
        return None
    return FileResponse(path, media_type="image/png", headers={
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache", "Expires": "0",
    })

app.add_static_files("/assets", str(ASSETS_DIR))

@app.get("/img/backtest_full")
def _bf():
    return _serve(BACKTEST_FULL)

@app.get("/img/backtest_10y")
def _b10y():
    return _serve(BACKTEST_10Y)

@app.get("/img/backtest_5y")
def _b5y():
    return _serve(BACKTEST_5Y)

@app.get("/img/backtest_1y")
def _b1y():
    return _serve(BACKTEST_1Y)

@app.get("/img/backtest_1m")
def _b1m():
    return _serve(BACKTEST_1M)

@app.get("/img/cape_ecy")
def _cape():
    return _serve(CAPE_CHART)


# ── Styles (same as dashboard) ────────────────────────────

PAGE_CSS = f"""
<style>
    html, body {{
        margin: 0;
        height: 100%;
        overflow: hidden;
        background: #f0f0f0;
        font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }}
    #app, .nicegui-content {{
        height: 100%;
        margin: 0;
        padding: 0;
    }}
    .layout {{
        display: flex;
        height: 100vh;
        width: 100%;
    }}
    .sidebar {{
        flex: 0 0 240px;
        height: 100%;
        background: #1a1a1a;
        color: #ddd;
        display: flex;
        flex-direction: column;
        overflow: hidden;
    }}
    .sidebar-photo {{
        width: 100%;
        height: 200px;
        background-image: url('/assets/{SIDEBAR_IMAGE.name}');
        background-size: cover;
        background-position: top center;
        flex-shrink: 0;
    }}
    .sidebar-title {{
        padding: 20px 20px 8px;
        text-align: center;
    }}
    .sidebar-title h2 {{
        margin: 0;
        font-size: 1.1rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #ccc;
    }}
    .sidebar-title .sub {{
        font-size: 0.7rem;
        color: #777;
        margin-top: 4px;
        letter-spacing: 0.05em;
    }}
    .sidebar-divider {{
        width: 40px;
        height: 2px;
        background: #555;
        margin: 12px auto;
    }}
    .sidebar-metrics {{
        padding: 0 20px;
        display: flex;
        flex-direction: column;
        gap: 14px;
        flex: 1;
    }}
    .sidebar-metric {{
        display: flex;
        justify-content: space-between;
        align-items: baseline;
    }}
    .sidebar-metric .label {{
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #888;
    }}
    .sidebar-metric .value {{
        font-size: 1.1rem;
        font-weight: 700;
        color: #eee;
    }}
    .sidebar-emergency {{
        padding: 16px 20px;
        text-align: center;
        flex-shrink: 0;
    }}
    .emergency-btn {{
        width: 100%;
        padding: 12px;
        border: none;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.85rem;
        letter-spacing: 0.08em;
        cursor: pointer;
        transition: all 0.2s;
    }}
    .emergency-btn:hover {{
        transform: scale(1.03);
    }}
    .emergency-btn.off {{
        background: #c62828;
        color: white;
        box-shadow: 0 0 12px rgba(198, 40, 40, 0.5);
    }}
    .emergency-btn.on {{
        background: #2e7d32;
        color: white;
        box-shadow: 0 0 12px rgba(46, 125, 50, 0.5);
    }}
    .emergency-status {{
        margin-top: 8px;
        font-size: 0.7rem;
        letter-spacing: 0.05em;
    }}
    .sidebar-footer {{
        padding: 16px 20px;
        font-size: 0.65rem;
        color: #555;
        text-align: center;
        flex-shrink: 0;
    }}
    .content-panel {{
        flex: 1;
        background: #ffffff;
        color: #222;
        display: flex;
        flex-direction: column;
        padding: 0;
        min-height: 0;
        overflow: hidden;
        box-sizing: border-box;
    }}
    .gain-positive {{ color: #4caf50; }}
    .gain-negative {{ color: #ef5350; }}
    .custom-tabs .q-tab__label {{
        font-size: clamp(0.72rem, 0.5vw + 0.45rem, 0.95rem);
        font-weight: 600;
    }}
    .custom-tabs .q-tab {{
        min-height: 42px;
    }}
    .custom-tab-panels {{
        flex: 1;
        min-height: 0;
        overflow-y: auto;
    }}
    .custom-tab-panels .q-tab-panel {{
        min-height: 100%;
        padding-bottom: 16px;
    }}
    .tab-content {{
        width: 100%;
        max-width: 1100px;
        margin: 0 auto;
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        gap: 20px;
        padding: 12px 0 24px;
    }}
    .card {{
        background: white;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        padding: 20px;
        width: 100%;
    }}
    .mono {{
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    }}
    @media (max-width: 768px) {{
        .layout {{ flex-direction: column; }}
        .sidebar {{ display: none; }}
    }}
</style>
"""


# ── Emergency override ────────────────────────────────────

OVERRIDE_FILE = ROOT / "logs" / "emergency_off.json"


def is_emergency_off():
    if OVERRIDE_FILE.exists():
        with open(OVERRIDE_FILE) as f:
            data = json.load(f)
        return data.get("active", False)
    return False


def set_emergency_off(active):
    OVERRIDE_FILE.parent.mkdir(exist_ok=True)
    with open(OVERRIDE_FILE, "w") as f:
        json.dump({"active": active, "timestamp": datetime.now().isoformat()}, f)


# ── Page layout ───────────────────────────────────────────

def render_gain(trades):
    """Onglet Gain reel pour UN compte (courbe time-weighted + KPI)."""
    perf_series, perf_stats = compute_real_performance(trades)
    if not perf_series:
        ui.label("Pas encore assez d'historique réel.").classes("text-gray-500")
        return
    gcls = "gain-positive" if perf_stats["cum_return"] >= 0 else "gain-negative"
    eur_sign = "+" if perf_stats["total_gain_eur"] >= 0 else ""

    def kpi(label, value, cls=""):
        with ui.element("div").classes("card").style("flex:1; text-align:center"):
            ui.html(f'<div style="font-size:0.7rem;text-transform:uppercase;'
                    f'letter-spacing:0.06em;color:#888">{label}</div>')
            ui.html(f'<div class="{cls}" style="font-size:1.6rem;'
                    f'font-weight:700;margin-top:6px">{value}</div>')

    with ui.row().classes("w-full gap-4"):
        kpi("Capital actuel", f"{perf_stats['current_equity']:.0f} EUR")
        kpi("Gain cumulé", f"{perf_stats['cum_return']*100:+.1f}%", gcls)
        kpi("Gain réel", f"{eur_sign}{perf_stats['total_gain_eur']:.0f} EUR", gcls)
        kpi("Annualisé", f"{perf_stats['annualized']*100:+.1f}%", gcls)

    xs = [datetime.strptime(s["date"], "%Y-%m-%d") for s in perf_series]
    ys = [s["cum_return"] * 100 for s in perf_series]

    # Insérer les points de croisement à y=0 pour clipper net
    ax, ay = [], []
    for i in range(len(xs)):
        if i > 0 and ((ys[i-1] < 0 < ys[i]) or (ys[i-1] > 0 > ys[i])):
            frac = -ys[i-1] / (ys[i] - ys[i-1])
            ax.append(xs[i-1] + (xs[i] - xs[i-1]) * frac)
            ay.append(0.0)
        ax.append(xs[i])
        ay.append(ys[i])

    pos_fill = [y if y >= 0 else 0.0 for y in ay]
    neg_fill = [y if y <= 0 else 0.0 for y in ay]
    pos_line = [y if y >= 0 else None for y in ay]
    neg_line = [y if y <= 0 else None for y in ay]

    GREEN, RED = "#2e7d32", "#ef5350"
    fig_gain = go.Figure()
    # Remplissages (sous la courbe), vert au-dessus / rouge en dessous
    fig_gain.add_trace(go.Scatter(
        x=ax, y=neg_fill, fill="tozeroy", mode="none",
        fillcolor="rgba(239,83,80,0.20)", hoverinfo="skip", showlegend=False))
    fig_gain.add_trace(go.Scatter(
        x=ax, y=pos_fill, fill="tozeroy", mode="none",
        fillcolor="rgba(46,125,50,0.20)", hoverinfo="skip", showlegend=False))
    # Lignes colorées par signe (se rejoignent au croisement)
    fig_gain.add_trace(go.Scatter(
        x=ax, y=neg_line, mode="lines", line=dict(color=RED, width=2),
        connectgaps=False, hoverinfo="skip", showlegend=False))
    fig_gain.add_trace(go.Scatter(
        x=ax, y=pos_line, mode="lines", line=dict(color=GREEN, width=2),
        connectgaps=False, hoverinfo="skip", showlegend=False))
    # Marqueurs sur les vrais points, colorés par signe + hover
    fig_gain.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers",
        marker=dict(size=6, color=[RED if y < 0 else GREEN for y in ys]),
        hovertemplate="%{x|%d %b %Y}<br>%{y:+.2f}%<extra></extra>",
        showlegend=False))
    fig_gain.add_hline(y=0, line=dict(color="#888", width=1, dash="dot"))
    fig_gain.update_layout(
        title=dict(text="Rendement cumulé (time-weighted) du compte PEA", x=0.5),
        margin=dict(t=46, b=30, l=50, r=20), height=420,
        yaxis=dict(title="Gain (%)", ticksuffix="%", zeroline=False),
        xaxis=dict(title="", type="date"),
        plot_bgcolor="white", hovermode="x unified",
    )
    ui.plotly(fig_gain).classes("w-full")

    if abs(perf_stats["total_deposits"]) >= 5.0:
        ui.label(
            f"Apports/retraits neutralisés : {perf_stats['total_deposits']:+.0f} EUR "
            f"(exclus du rendement)"
        ).style("font-size:0.75rem; color:#999")
    ui.label(
        f"Capital initial {perf_stats['start_equity']:.0f} EUR · "
        f"{perf_stats['days']} jours · frais de vente (0.5%) inclus"
    ).style("font-size:0.75rem; color:#999")


def render_trades(trades, portfolio):
    """Onglet Trades pour UN compte : uniquement les ordres (BUY / SELL), une ligne par
    instrument — les jours sans mouvement restent visibles dans l'onglet Allocations."""
    rows = []
    for t, p in zip(reversed(trades), reversed(portfolio)):
        for o in reversed(p["orders"]):
            if o["side"] not in ("buy", "sell") or o["quantity"] <= 0:
                continue
            status = "LIVE" if o["done"] else ("DRY-RUN" if not t.get("executed") else (o.get("status") or "?").upper())
            rows.append({
                "date": p["date"], "side": o["side"].upper(), "instrument": o["instrument"],
                "quantity": o["quantity"], "price": f"{o['price']:.3f}",
                "value": f"{o['quantity'] * o['price']:.0f} EUR",
                "target": f"{p['target_exposure']*100:.0f}%",
                "real": f"{p['exposure']*100:.0f}%", "status": status,
            })
    if not rows:
        ui.label("No trades yet.").classes("text-gray-500")
        return
    columns = [
        {"name": "date", "label": "Date", "field": "date", "align": "left"},
        {"name": "side", "label": "Action", "field": "side", "align": "left"},
        {"name": "instrument", "label": "ETF", "field": "instrument", "align": "left"},
        {"name": "quantity", "label": "Qty", "field": "quantity", "align": "right"},
        {"name": "price", "label": "Price", "field": "price", "align": "right"},
        {"name": "value", "label": "Value", "field": "value", "align": "right"},
        {"name": "target", "label": "Expo cible", "field": "target", "align": "right"},
        {"name": "real", "label": "Expo réelle", "field": "real", "align": "right"},
        {"name": "status", "label": "Status", "field": "status", "align": "center"},
    ]
    table = ui.table(columns=columns, rows=rows).classes("w-full")
    table.add_slot("body-cell-side", """
        <q-td :props="props">
            <span :style="{ color: props.value === 'BUY' ? '#1f8f4c' : props.value === 'SELL' ? '#c0392b' : '#888',
                             fontWeight: 600 }">
                {{ props.value }}
            </span>
        </q-td>
    """)
    table.add_slot("body-cell-status", """
        <q-td :props="props">
            <q-badge :color="props.value === 'LIVE' ? 'green' : props.value === 'DRY-RUN' ? 'grey' : 'orange'" :label="props.value" />
        </q-td>
    """)


def render_allocations(portfolio):
    """Onglet Allocations pour UN compte : composition PUST / LQQ / cash et exposition
    reelle vs cible, jour par jour (position post-ordre)."""
    if not portfolio:
        ui.label("No allocation history yet.").classes("text-gray-500")
        return
    columns = [
        {"name": "date", "label": "Date", "field": "date", "align": "left"},
        {"name": "pust", "label": "PUST", "field": "pust", "align": "right"},
        {"name": "lqq", "label": "LQQ", "field": "lqq", "align": "right"},
        {"name": "cash", "label": "Cash", "field": "cash", "align": "right"},
        {"name": "equity", "label": "Equity", "field": "equity", "align": "right"},
        {"name": "alloc", "label": "Alloc x1", "field": "alloc", "align": "right"},
        {"name": "target", "label": "Expo cible", "field": "target", "align": "right"},
        {"name": "actual", "label": "Expo réelle", "field": "actual", "align": "right"},
        {"name": "action", "label": "Action", "field": "action", "align": "center"},
    ]

    def _cell(p, ins):
        pos = p["positions"].get(ins)
        if not pos or pos["shares"] <= 0:
            return "—"
        return f"{pos['shares']} ({p['weights'].get(ins, 0)*100:.0f}%)"

    rows = []
    for p in reversed(portfolio):
        action = " + ".join(f"{o['side'].upper()} {o['quantity']} {o['instrument']}" for o in p["orders"]
                            if o["side"] in ("buy", "sell") and o["quantity"] > 0) or "HOLD"
        if p.get("reserved"):
            action += f" · réserve DCA {p['reserved']:.0f} EUR"
        rows.append({
            "date": p["date"], "pust": _cell(p, "PUST"), "lqq": _cell(p, "LQQ"),
            "cash": f"{p['cash']:.0f} EUR ({p['cash_weight']*100:.0f}%)",
            "equity": f"{p['equity']:.0f} EUR",
            "alloc": f"{p['target_alloc']*100:.0f}%",
            "target": f"{p['target_exposure']*100:.0f}%",
            "actual": f"{p['exposure']*100:.0f}%",
            "action": action,
        })
    ui.table(columns=columns, rows=rows, row_key="date").classes("w-full")


def render_account_card(acc):
    """Carte de statut d'UN compte (onglet Comptes) : connexion, dernier run, exposition
    cible vs reelle, repartition PUST / LQQ / cash, capital a allouer detecte."""
    st = account_status(acc)
    last = acc["last"] or {}
    hist = compute_portfolio_history(acc["trades"])
    cur = hist[-1] if hist else {}
    equity = last.get("equity", 0) or cur.get("equity", 0)
    target_e = cur.get("target_exposure", 0)
    real_e = cur.get("exposure", 0)
    n_live = sum(1 for p in hist for o in p["orders"] if o["done"] and o["side"] in ("buy", "sell"))
    cap = load_capital_entry(acc["slot"])
    dca = cap.get("dca") if cap else None
    dca_active = bool(dca and not dca.get("done"))
    pending = load_pending_entry(acc["slot"])

    with ui.element("div").classes("card"):
        with ui.row().classes("w-full items-center justify-between"):
            with ui.row().classes("items-baseline gap-3"):
                ui.label(acc["name"]).classes("text-lg font-semibold")
                ui.label(f"compte {acc['slot']}").style("font-size:0.75rem; color:#999")
                if acc.get("mail"):
                    ui.label(acc["mail"]).style("font-size:0.75rem; color:#999")
            with ui.row().classes("items-center gap-2"):
                if dca_active:
                    ui.badge("CAPITAL À ALLOUER DÉTECTÉ", color="blue")
                if pending:
                    ui.badge("ACHAT DIFFÉRÉ", color="orange")
                if st.get("mode"):
                    ui.badge(st["mode"], color="green" if st["mode"] == "LIVE" else "grey")
                ui.badge(st["label"], color=st["color"]).props("outline" if st["state"] == "unknown" else "")

        def cell(label, value, cls=""):
            with ui.element("div").style("flex:1; min-width:120px"):
                ui.html(f'<div style="font-size:0.65rem;text-transform:uppercase;'
                        f'letter-spacing:0.06em;color:#888">{label}</div>')
                ui.html(f'<div class="{cls}" style="font-size:1.15rem;font-weight:700;'
                        f'margin-top:4px">{value}</div>')

        with ui.row().classes("w-full gap-4 mt-3"):
            cell("Dernier run", st["date"], "mono")
            cell("Capital", f"{equity:.0f} EUR" if equity else "--")
            cell("Expo cible", f"{target_e*100:.0f}%" if hist else "--")
            cell("Expo réelle", f"{real_e*100:.0f}%" if hist else "--",
                 "gain-positive" if real_e >= 1.0 else "gain-negative")
            cell("Trades LIVE", f"{n_live}")

        # Repartition PUST / LQQ / cash (barre + detail)
        if hist:
            w = cur.get("weights", {})
            wp, wl = w.get("PUST", 0), w.get("LQQ", 0)
            wc = max(0.0, cur.get("cash_weight", 0))
            with ui.row().classes("w-full gap-4 mt-3 items-end"):
                for ins, color in (("PUST", "#1565c0"), ("LQQ", "#6a1b9a")):
                    pos = cur["positions"].get(ins, {"shares": 0, "price": 0})
                    cell(f"{ins} (x{LEVERAGE[ins]:.0f})",
                         f"{pos['shares']} parts · {w.get(ins, 0)*100:.0f}%" if pos["shares"] else "0",
                         "")
                cell("Cash", f"{cur.get('cash', 0):.0f} EUR · {wc*100:.0f}%")
            ui.html(
                '<div style="display:flex;width:100%;height:12px;border-radius:6px;overflow:hidden;'
                'margin-top:8px;background:#eee">'
                f'<div title="PUST {wp*100:.0f}%" style="width:{wp*100:.1f}%;background:#1565c0"></div>'
                f'<div title="LQQ {wl*100:.0f}%" style="width:{wl*100:.1f}%;background:#6a1b9a"></div>'
                f'<div title="cash {wc*100:.0f}%" style="width:{wc*100:.1f}%;background:#bdbdbd"></div>'
                '</div>'
                '<div style="font-size:0.7rem;color:#888;margin-top:4px">'
                '<span style="color:#1565c0">■</span> PUST &nbsp; '
                '<span style="color:#6a1b9a">■</span> LQQ &nbsp; '
                '<span style="color:#bdbdbd">■</span> cash &nbsp;·&nbsp; '
                f'exposition = PUST + 2 × LQQ = {real_e*100:.0f}%</div>')

        # Capital a allouer (apport detecte, deploiement progressif DCA / RSI)
        if dca_active:
            reserved = dca["deposit"] * (1 - dca["released"])
            last_tr = dca.get("last_tranche")
            with ui.element("div").style(
                    "width:100%; margin-top:12px; padding:10px 14px; border-radius:8px; "
                    "background:#e3f2fd; border:1px solid #90caf9"):
                ui.html(f'<div style="font-weight:700;color:#0d47a1">Capital à allouer détecté : '
                        f'+{dca["deposit"]:.0f} EUR (le {dca.get("detected", "?")})</div>')
                ui.html(f'<div style="font-size:0.8rem;color:#333;margin-top:4px">'
                        f'Déployé {dca["released"]*100:.0f}% · réservé {reserved:.0f} EUR · '
                        f'{dca.get("weeks", 0)} tranche(s) · dernière tranche {last_tr or "—"}<br>'
                        f'Règle : chaque semaine, tranche = 20% + 2% × (50 − RSI14) si RSI14 &lt; 50 '
                        f'(0 sinon) ; tout investi au plus tard après 26 semaines.</div>')
                ui.html('<div style="width:100%;height:8px;border-radius:4px;background:#bbdefb;margin-top:6px">'
                        f'<div style="width:{dca["released"]*100:.1f}%;height:8px;border-radius:4px;'
                        'background:#1976d2"></div></div>')
        elif cap.get("deposits"):
            d = cap["deposits"][-1]
            ui.label(f"Dernier apport détecté : {d.get('amount', 0):+.0f} EUR le {d.get('date', '?')} (entièrement déployé)") \
                .style("font-size:0.75rem; color:#666; margin-top:8px")
        if pending:
            ui.label(f"Achat différé depuis le {pending.get('date')} : "
                     + ", ".join(f"{b.get('quantity')} {b.get('instrument')}" for b in pending.get("buys", []))
                     + f" — {pending.get('reason', '')} ; repris au prochain run.") \
                .style("font-size:0.8rem; color:#e65100; margin-top:8px")
        if st.get("detail"):
            color = "#c62828" if st["state"] in ("ko", "error") else "#666"
            ui.label(st["detail"]).style(f"font-size:0.8rem; color:{color}; margin-top:8px")


def account_header(acc, n_accounts):
    """Sous-titre d'une section par compte (masque s'il n'y a qu'un compte)."""
    if n_accounts > 1:
        st = account_status(acc)
        with ui.row().classes("w-full items-center gap-3 mt-2"):
            ui.label(f"{acc['name']}").classes("text-base font-semibold")
            ui.label(f"compte {acc['slot']}").style("font-size:0.75rem; color:#999")
            ui.badge(st["label"], color=st["color"])


@ui.page("/")
def index_page():
    """Dashboard (protege par AuthMiddleware). Les donnees sont relues a chaque
    visite -> plus besoin de redemarrer le conteneur apres un trade."""
    ui.add_head_html(PAGE_CSS)

    trades = load_trades()
    accounts = split_by_account(trades)
    n_accounts = len(accounts)
    # Signal (allocation conseillee, proba, date modele) = commun a tous les comptes :
    # on le lit sur la derniere ligne exploitable du journal, tous comptes confondus.
    usable = [t for t in trades if not _is_connection_error(t)]
    last_trade = usable[-1] if usable else {}

    current_alloc = last_trade.get("target_alloc", 0)
    current_prob = last_trade.get("probability", 0)
    last_view = row_view(last_trade) if last_trade else {"positions": {}, "target_exposure": None}
    current_expo = last_view["target_exposure"]
    if current_expo is None:
        current_expo = current_alloc * max([LEVERAGE.get(k, 1.0) for k in last_view["positions"]] or [1.0])
    prices = {k: p["price"] for k, p in last_view["positions"].items() if p.get("price")}
    instrument = last_trade.get("instrument", "PUST")
    model_date = last_trade.get("model_date", "--")
    n_exec = sum(1 for t in usable for o in row_view(t)["orders"] if o["done"] and o["side"] in ("buy", "sell"))
    n_ko = sum(1 for acc in accounts.values() if account_status(acc)["state"] != "ok")
    n_dca = sum(1 for acc in accounts.values()
                if (load_capital_entry(acc["slot"]).get("dca") or {}).get("done") is False)

    with ui.element("div").classes("layout"):
        # ── Sidebar ──
        with ui.element("div").classes("sidebar"):
            ui.element("div").classes("sidebar-photo")
            with ui.element("div").classes("sidebar-title"):
                ui.html("<h2>Risk-Off</h2>")
                ui.html(f'<div class="sub">QQQ / {instrument} — Boursorama PEA</div>')
            ui.element("div").classes("sidebar-divider")
            with ui.element("div").classes("sidebar-metrics"):
                alloc_color = "gain-positive" if current_alloc >= 0.5 else "gain-negative"
                acc_color = "gain-negative" if n_ko else ("gain-positive" if n_accounts else "")
                metrics = [
                    ("Allocation", f"{current_alloc*100:.0f}%", alloc_color),
                    ("Exposition", f"{current_expo*100:.0f}%", alloc_color),
                    ("Probability", f"{current_prob:.3f}", ""),
                ]
                metrics += [(k, f"{v:.2f} EUR", "") for k, v in prices.items()]
                metrics += [
                    ("Trades", f"{n_exec}", ""),
                    ("Comptes", f"{n_accounts - n_ko}/{n_accounts} OK" if n_accounts else "0", acc_color),
                ]
                if n_dca:
                    metrics.append(("Capital à allouer", f"{n_dca} compte(s)", "gain-positive"))
                metrics.append(("Model", model_date, "mono"))
                for label, value, extra_class in metrics:
                    with ui.element("div").classes("sidebar-metric"):
                        ui.html(f'<span class="label">{label}</span>')
                        ui.html(f'<span class="value {extra_class}">{value}</span>')
            # Emergency OFF button
            with ui.element("div").classes("sidebar-emergency"):
                emergency_active = is_emergency_off()

                status_label = ui.label(
                    "EMERGENCY OFF" if emergency_active else "MODEL ACTIVE"
                ).classes("emergency-status").style(
                    f"color: {'#ef5350' if emergency_active else '#4caf50'}"
                )

                def toggle_emergency():
                    if not is_emergency_off():
                        # Activate emergency — show confirmation dialog
                        with ui.dialog() as dlg, ui.card():
                            ui.label("Desallocation d'urgence").style("font-weight: 700; font-size: 1.1rem")
                            ui.label("Cela va forcer l'allocation a 0% et vendre toutes les positions "
                                     "(sur TOUS les comptes geres).").style("color: #666")
                            ui.label("Confirmer ?").style("font-weight: 600; margin-top: 8px")
                            with ui.row().classes("gap-4 mt-4"):
                                def confirm():
                                    set_emergency_off(True)
                                    btn.classes("on", remove="off")
                                    btn.text = "REACTIVER LE MODEL"
                                    status_label.text = "EMERGENCY OFF"
                                    status_label.style("color: #ef5350")
                                    dlg.close()
                                    ui.notify("Emergency OFF active — allocation forcee a 0%", type="warning")
                                ui.button("CONFIRMER", on_click=confirm).props("color=red")
                                ui.button("Annuler", on_click=dlg.close).props("flat")
                        dlg.open()
                    else:
                        # Deactivate emergency — back to model
                        set_emergency_off(False)
                        btn.classes("off", remove="on")
                        btn.text = "EMERGENCY OFF"
                        status_label.text = "MODEL ACTIVE"
                        status_label.style("color: #4caf50")
                        ui.notify("Model reactif — le model reprend le controle demain matin", type="positive")

                btn = ui.button(
                    "REACTIVER LE MODEL" if emergency_active else "EMERGENCY OFF",
                    on_click=toggle_emergency,
                )
                btn.classes(f"emergency-btn {'on' if emergency_active else 'off'}")

            with ui.element("div").classes("sidebar-footer"):
                ui.html(f'{datetime.now().strftime("%d %b %Y &nbsp; %H:%M")}')

        # ── Content ──
        with ui.column().classes("content-panel"):
            # Tabs
            with ui.tabs().classes("w-full custom-tabs").props("dense") as tabs:
                tab_full = ui.tab("Backtest")
                tab_10y = ui.tab("10 Years")
                tab_5y = ui.tab("5 Years")
                tab_1y = ui.tab("1 Year")
                tab_1m = ui.tab("1 Month")
                tab_accounts = ui.tab("Comptes")
                tab_gain = ui.tab("Gain réel")
                tab_trades = ui.tab("Trades")
                tab_alloc = ui.tab("Allocations")
                tab_val = ui.tab("Valorisation")

            with ui.tab_panels(tabs, value=tab_full).classes("w-full flex-1 custom-tab-panels"):

                # ── Backtest Full ──
                with ui.tab_panel(tab_full):
                    with ui.column().classes("tab-content"):
                        ui.element("div").classes("w-full h-0.5 bg-black")
                        ui.label("Backtest depuis 2000 — stratégie déployée PUST + LQQ x1.7 (net de frais) "
                                 "vs B&H QQQ").classes("text-base font-semibold")
                        if BACKTEST_FULL.exists():
                            ui.image("/img/backtest_full").classes("w-full rounded-lg shadow-lg")
                        else:
                            ui.label("Run: python src/risk_off_strategy/run.py QQQ").classes("text-gray-500")

                # ── 10Y ──
                with ui.tab_panel(tab_10y):
                    with ui.column().classes("tab-content"):
                        ui.element("div").classes("w-full h-0.5 bg-black")
                        ui.label("Last 10 Years").classes("text-base font-semibold")
                        if BACKTEST_10Y.exists():
                            ui.image("/img/backtest_10y").classes("w-full rounded-lg shadow-lg")

                # ── 5Y ──
                with ui.tab_panel(tab_5y):
                    with ui.column().classes("tab-content"):
                        ui.element("div").classes("w-full h-0.5 bg-black")
                        ui.label("Last 5 Years").classes("text-base font-semibold")
                        if BACKTEST_5Y.exists():
                            ui.image("/img/backtest_5y").classes("w-full rounded-lg shadow-lg")

                # ── 1Y ──
                with ui.tab_panel(tab_1y):
                    with ui.column().classes("tab-content"):
                        ui.element("div").classes("w-full h-0.5 bg-black")
                        ui.label("Last 252 Trading Days").classes("text-base font-semibold")
                        if BACKTEST_1Y.exists():
                            ui.image("/img/backtest_1y").classes("w-full rounded-lg shadow-lg")

                # ── 1M ──
                with ui.tab_panel(tab_1m):
                    with ui.column().classes("tab-content"):
                        ui.element("div").classes("w-full h-0.5 bg-black")
                        ui.label("Last 21 Trading Days").classes("text-base font-semibold")
                        if BACKTEST_1M.exists():
                            ui.image("/img/backtest_1m").classes("w-full rounded-lg shadow-lg")

                # ── Comptes (statut separe par compte Bourso) ──
                with ui.tab_panel(tab_accounts):
                    with ui.column().classes("tab-content"):
                        ui.element("div").classes("w-full h-0.5 bg-black")
                        ui.label("Comptes Bourso — statut par compte").classes("text-base font-semibold")
                        ui.label("Le meme signal est replique sur chaque compte gere ; chaque compte "
                                 "est lu, decide et execute independamment (bande de non-action sur "
                                 "sa propre exposition reelle). Strategie PUST + LQQ, exposition = "
                                 "min(2 x allocation, 170%) ; un apport de capital detecte est deploye "
                                 "par tranches hebdomadaires quand le RSI14 < 50.").style("font-size: 0.8rem; color: #666")
                        if not accounts:
                            ui.label("Aucun compte connu (trades.jsonl / logs/accounts.json vides).") \
                                .classes("text-gray-500")
                        for acc in accounts.values():
                            render_account_card(acc)

                # ── Valorisation (CAPE / ECY S&P, Shiller) ──
                with ui.tab_panel(tab_val):
                    with ui.column().classes("tab-content"):
                        ui.element("div").classes("w-full h-0.5 bg-black")
                        ui.label("Valorisation — CAPE & Excess CAPE Yield (S&P 500, Shiller)") \
                            .classes("text-base font-semibold")
                        val = load_cape_ecy_latest()
                        if not val:
                            ui.label("Pas de données. Lancer : python -m src.download_shiller_cape") \
                                .classes("text-gray-500")
                        else:
                            ecy_s = f"{val['ecy']*100:+.2f}%" if val["ecy"] is not None else "—"
                            with ui.row().classes("gap-6 items-baseline"):
                                ui.label(f"CAPE {val['cape']:.1f}").classes("text-lg font-semibold")
                                ui.label(f"percentile {val['cape_pct']:.0f} "
                                         f"(médiane {val['cape_median']:.1f})") \
                                    .style("font-size: 0.8rem; color: #999")
                                ui.label(f"ECY {ecy_s}").classes("text-lg font-semibold")
                                ui.label(f"données Shiller au {val['date']}") \
                                    .style("font-size: 0.8rem; color: #999")
                            ui.label("Valorisation ajustée des taux, comparable entre époques "
                                     "(≠ PE brut). Contexte — hors stratégie.") \
                                .style("font-size: 0.8rem; color: #666")
                        if CAPE_CHART.exists():
                            ui.image("/img/cape_ecy").classes("w-full rounded-lg shadow-lg")

                # ── Gain réel (une section par compte) ──
                with ui.tab_panel(tab_gain):
                    with ui.column().classes("tab-content"):
                        ui.element("div").classes("w-full h-0.5 bg-black")
                        ui.label("Gain réel du compte — rendement sur le capital").classes("text-base font-semibold")
                        if not accounts:
                            ui.label("Pas encore assez d'historique réel.").classes("text-gray-500")
                        for acc in accounts.values():
                            account_header(acc, n_accounts)
                            render_gain(acc["trades"])

                # ── Trades (une section par compte) ──
                with ui.tab_panel(tab_trades):
                    with ui.column().classes("tab-content"):
                        ui.element("div").classes("w-full h-0.5 bg-black")
                        ui.label("Trade History").classes("text-base font-semibold")
                        if not accounts:
                            ui.label("No trades yet.").classes("text-gray-500")
                        for acc in accounts.values():
                            account_header(acc, n_accounts)
                            render_trades(acc["trades"], compute_portfolio_history(acc["trades"]))

                # ── Allocation History (une section par compte) ──
                with ui.tab_panel(tab_alloc):
                    with ui.column().classes("tab-content"):
                        ui.element("div").classes("w-full h-0.5 bg-black")
                        ui.label("Allocation History").classes("text-base font-semibold")
                        if not accounts:
                            ui.label("No allocation history yet.").classes("text-gray-500")
                        for acc in accounts.values():
                            account_header(acc, n_accounts)
                            render_allocations(compute_portfolio_history(acc["trades"]))


ui.run(title="Risk-Off Strategy — Gregory Descamps", port=int(os.environ.get("WEBAPP_PORT", 8080)),
       reload=False, storage_secret=WEBAPP_SECRET)
