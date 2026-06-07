"""
Risk-Off Strategy Dashboard — NiceGUI webapp.

Shows:
  - Backtests (full, 1y, 1m)
  - Real Bourso allocations & trade history
  - Gains normalized (% and EUR)
  - Capital added and reallocated

Usage:  python src/webapp.py
"""

import json
from datetime import datetime
from pathlib import Path

from fastapi.responses import FileResponse
from nicegui import app, ui

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
QQQ_OUT = OUTPUTS / "qqq_strategy"
PE_OUT = OUTPUTS / "pe"
TRADE_LOG = ROOT / "logs" / "trades.jsonl"

# Backtest images
BACKTEST_FULL = QQQ_OUT / "backtest.png"
BACKTEST_1Y = QQQ_OUT / "backtest_1y.png"
BACKTEST_1M = QQQ_OUT / "backtest_1m.png"
PE_CHART = PE_OUT / "ndx_top5_pe_daily.png"

# ── Data loaders ──────────────────────────────────────────

def load_trades():
    """Load trade log from JSONL."""
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


def compute_portfolio_history(trades):
    """Compute portfolio value history from trade log."""
    history = []
    shares = 0
    cash_invested = 0  # total capital added

    for t in trades:
        side = t.get("side")
        qty = t.get("quantity", 0)
        price = t.get("etf_price", 0)
        equity = t.get("equity", 0)
        alloc = t.get("target_alloc", 0)
        prob = t.get("probability", 0)
        date = t.get("date", "")
        executed = t.get("executed", False)

        if side == "buy" and executed:
            shares += qty
        elif side == "sell" and executed:
            shares -= qty

        position_value = shares * price if price else 0

        history.append({
            "date": date,
            "shares": shares,
            "price": price,
            "position_value": position_value,
            "equity": equity,
            "target_alloc": alloc,
            "actual_alloc": position_value / equity if equity > 0 else 0,
            "probability": prob,
            "side": side or "hold",
            "quantity": qty,
            "executed": executed,
        })

    return history


# ── Serve images without cache ────────────────────────────

def serve_image(path):
    """Serve image with no-cache headers."""
    if not path.exists():
        return None
    return FileResponse(path, media_type="image/png", headers={
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache", "Expires": "0",
    })


@app.get("/img/backtest_full")
def _bf():
    return serve_image(BACKTEST_FULL)

@app.get("/img/backtest_1y")
def _b1y():
    return serve_image(BACKTEST_1Y)

@app.get("/img/backtest_1m")
def _b1m():
    return serve_image(BACKTEST_1M)

@app.get("/img/pe_chart")
def _pe():
    return serve_image(PE_CHART)


# ── Styles ────────────────────────────────────────────────

ui.add_head_html("""
<style>
    html, body {
        margin: 0;
        background: #f5f5f5;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    .header-bar {
        background: #111;
        color: white;
        padding: 16px 32px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .header-title {
        font-size: 1.3rem;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    .header-sub {
        font-size: 0.85rem;
        color: #aaa;
    }
    .card {
        background: white;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        padding: 20px;
    }
    .metric-card {
        background: white;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        padding: 16px 20px;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #111;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #888;
        margin-top: 4px;
    }
    .gain-positive { color: #1f8f4c; }
    .gain-negative { color: #c0392b; }
    .custom-tabs .q-tab__label {
        font-size: 0.9rem;
        font-weight: 600;
    }
    .mono {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    }
</style>
""")


# ── Page ──────────────────────────────────────────────────

# Header
with ui.element("div").classes("header-bar"):
    with ui.column().style("gap: 2px"):
        ui.label("Risk-Off Strategy").classes("header-title")
        ui.label("QQQ / PUST — Boursorama PEA").classes("header-sub")
    ui.label(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}").classes("header-sub")

# Load data
trades = load_trades()
portfolio = compute_portfolio_history(trades)

# ── Metrics row ───────────────────────────────────────────

with ui.row().classes("w-full max-w-6xl mx-auto mt-4 gap-4 px-4"):
    # Current allocation
    last_trade = trades[-1] if trades else {}
    current_alloc = last_trade.get("target_alloc", 0)
    current_prob = last_trade.get("probability", 0)
    current_price = last_trade.get("etf_price", 0)
    model_date = last_trade.get("model_date", "—")

    with ui.element("div").classes("metric-card flex-1"):
        color = "gain-positive" if current_alloc >= 0.5 else "gain-negative"
        ui.label(f"{current_alloc*100:.0f}%").classes(f"metric-value {color}")
        ui.label("Target Allocation").classes("metric-label")

    with ui.element("div").classes("metric-card flex-1"):
        ui.label(f"{current_prob:.3f}").classes("metric-value")
        ui.label("Model Probability").classes("metric-label")

    with ui.element("div").classes("metric-card flex-1"):
        ui.label(f"{current_price:.2f} EUR").classes("metric-value")
        ui.label("PUST Price").classes("metric-label")

    with ui.element("div").classes("metric-card flex-1"):
        n_trades = sum(1 for t in trades if t.get("executed"))
        ui.label(f"{n_trades}").classes("metric-value")
        ui.label("Trades Executed").classes("metric-label")

    with ui.element("div").classes("metric-card flex-1"):
        ui.label(model_date).classes("metric-value mono").style("font-size: 1.2rem")
        ui.label("Model Date").classes("metric-label")

# ── Tabs ──────────────────────────────────────────────────

with ui.column().classes("w-full max-w-6xl mx-auto mt-4 px-4 flex-1"):
    with ui.tabs().classes("w-full custom-tabs") as tabs:
        tab_full = ui.tab("Backtest Full")
        tab_1y = ui.tab("1 Year")
        tab_1m = ui.tab("1 Month")
        tab_pe = ui.tab("PE Top 5")
        tab_trades = ui.tab("Trades")
        tab_alloc = ui.tab("Allocation History")

    with ui.tab_panels(tabs, value=tab_full).classes("w-full"):

        # ── Backtest Full ──
        with ui.tab_panel(tab_full):
            with ui.element("div").classes("card w-full"):
                ui.label("Walk-Forward Backtest — Full History").style("font-weight: 600; margin-bottom: 12px")
                if BACKTEST_FULL.exists():
                    ui.image("/img/backtest_full").classes("w-full rounded-lg")
                else:
                    ui.label("No backtest available. Run: python src/risk_off_strategy/run.py QQQ")

        # ── Backtest 1Y ──
        with ui.tab_panel(tab_1y):
            with ui.element("div").classes("card w-full"):
                ui.label("Last 252 Trading Days").style("font-weight: 600; margin-bottom: 12px")
                if BACKTEST_1Y.exists():
                    ui.image("/img/backtest_1y").classes("w-full rounded-lg")
                else:
                    ui.label("No 1Y backtest available.")

        # ── Backtest 1M ──
        with ui.tab_panel(tab_1m):
            with ui.element("div").classes("card w-full"):
                ui.label("Last 21 Trading Days").style("font-weight: 600; margin-bottom: 12px")
                if BACKTEST_1M.exists():
                    ui.image("/img/backtest_1m").classes("w-full rounded-lg")
                else:
                    ui.label("No 1M backtest available.")

        # ── PE Top 5 ──
        with ui.tab_panel(tab_pe):
            with ui.element("div").classes("card w-full"):
                ui.label("NASDAQ-100 PE Top 5 (Daily Interpolated)").style("font-weight: 600; margin-bottom: 12px")
                if PE_CHART and PE_CHART.exists():
                    ui.image("/img/pe_chart").classes("w-full rounded-lg")
                else:
                    ui.label("No PE chart available. Run: python src/download_pe_qqq_top5.py")

        # ── Trades ──
        with ui.tab_panel(tab_trades):
            with ui.element("div").classes("card w-full"):
                ui.label("Trade History").style("font-weight: 600; margin-bottom: 12px")

                if not trades:
                    ui.label("No trades yet. Run: python src/real_bourso.py")
                else:
                    columns = [
                        {"name": "date", "label": "Date", "field": "date", "align": "left"},
                        {"name": "side", "label": "Action", "field": "side", "align": "left"},
                        {"name": "quantity", "label": "Qty", "field": "quantity", "align": "right"},
                        {"name": "etf_price", "label": "Price", "field": "etf_price", "align": "right"},
                        {"name": "value", "label": "Value (EUR)", "field": "value", "align": "right"},
                        {"name": "probability", "label": "Prob", "field": "probability", "align": "right"},
                        {"name": "target_alloc", "label": "Target", "field": "target_alloc", "align": "right"},
                        {"name": "executed", "label": "Status", "field": "executed", "align": "center"},
                    ]
                    rows = []
                    for t in reversed(trades):
                        qty = t.get("quantity", 0)
                        price = t.get("etf_price", 0)
                        rows.append({
                            "date": t.get("date", ""),
                            "side": (t.get("side") or "hold").upper(),
                            "quantity": qty,
                            "etf_price": f"{price:.2f}",
                            "value": f"{qty * price:.0f}",
                            "probability": f"{t.get('probability', 0):.3f}",
                            "target_alloc": f"{t.get('target_alloc', 0)*100:.0f}%",
                            "executed": "LIVE" if t.get("executed") else "DRY-RUN",
                        })
                    table = ui.table(columns=columns, rows=rows, row_key="date").classes("w-full")
                    table.add_slot("body-cell-side", """
                        <q-td :props="props">
                            <span :style="{ color: props.value === 'BUY' ? '#1f8f4c' : props.value === 'SELL' ? '#c0392b' : '#888' }">
                                {{ props.value }}
                            </span>
                        </q-td>
                    """)
                    table.add_slot("body-cell-executed", """
                        <q-td :props="props">
                            <q-badge :color="props.value === 'LIVE' ? 'green' : 'grey'" :label="props.value" />
                        </q-td>
                    """)

        # ── Allocation History ──
        with ui.tab_panel(tab_alloc):
            with ui.element("div").classes("card w-full"):
                ui.label("Allocation History").style("font-weight: 600; margin-bottom: 12px")

                if not portfolio:
                    ui.label("No allocation history yet.")
                else:
                    columns = [
                        {"name": "date", "label": "Date", "field": "date", "align": "left"},
                        {"name": "shares", "label": "Shares", "field": "shares", "align": "right"},
                        {"name": "price", "label": "Price", "field": "price", "align": "right"},
                        {"name": "position", "label": "Position (EUR)", "field": "position", "align": "right"},
                        {"name": "equity", "label": "Equity (EUR)", "field": "equity", "align": "right"},
                        {"name": "target", "label": "Target %", "field": "target", "align": "right"},
                        {"name": "actual", "label": "Actual %", "field": "actual", "align": "right"},
                        {"name": "action", "label": "Action", "field": "action", "align": "center"},
                    ]
                    rows = []
                    for p in reversed(portfolio):
                        rows.append({
                            "date": p["date"],
                            "shares": p["shares"],
                            "price": f"{p['price']:.2f}",
                            "position": f"{p['position_value']:.0f}",
                            "equity": f"{p['equity']:.0f}",
                            "target": f"{p['target_alloc']*100:.0f}%",
                            "actual": f"{p['actual_alloc']*100:.0f}%",
                            "action": p["side"].upper(),
                        })
                    ui.table(columns=columns, rows=rows, row_key="date").classes("w-full")


ui.run(title="Risk-Off Strategy — Dashboard", port=8081, reload=False)
