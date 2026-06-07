"""
Risk-Off Strategy Dashboard — NiceGUI webapp.

Layout inspired by dashboard/ — sidebar photo + tabbed content.

Shows:
  - Backtests (full, 1y, 1m)
  - PE Top 5 chart
  - Real Bourso allocations & trade history
  - Gains normalized (% and EUR)

Usage:  python src/webapp.py
"""

import json
from datetime import datetime
from pathlib import Path

import plotly.graph_objects as go
from fastapi.responses import FileResponse
from nicegui import app, ui

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
QQQ_OUT = OUTPUTS / "qqq_strategy"
PE_OUT = OUTPUTS / "pe"
TRADE_LOG = ROOT / "logs" / "trades.jsonl"
ASSETS_DIR = ROOT / "dashboard" / "assets"
SIDEBAR_IMAGE = ASSETS_DIR / "sidebar.jpg"

# Backtest images
BACKTEST_FULL = QQQ_OUT / "backtest.png"
BACKTEST_1Y = QQQ_OUT / "backtest_1y.png"
BACKTEST_1M = QQQ_OUT / "backtest_1m.png"
PE_CHART = PE_OUT / "ndx_top5_pe_daily.png"


# ── Data loaders ──────────────────────────────────────────

PE_DIR = ROOT / "data" / "pe"


def load_ndx_top(n=20):
    """Load top N NDX constituents with latest PE and PE date."""
    constituents_file = PE_DIR / "ndx_constituents.json"
    if not constituents_file.exists():
        return [], None

    with open(constituents_file) as f:
        top = json.load(f)

    results = []
    pe_date = None
    for t in top[:n]:
        sym = t["symbol"]
        qfile = PE_DIR / f"{sym}_quarterly.json"
        pe = None
        q_date = None
        if qfile.exists():
            with open(qfile) as f:
                quarters = json.load(f)
            if quarters:
                latest = quarters[0]
                pe = latest.get("peRatio")
                q_date = latest.get("date")
                if q_date and (pe_date is None or q_date > pe_date):
                    pe_date = q_date

        results.append({
            "symbol": sym,
            "name": t.get("name", "")[:25],
            "mktCap": t.get("mktCap", 0),
            "pe": pe,
        })

    # Interpolation date = last date in pe_top5_daily.parquet
    interp_date = None
    daily_file = PE_DIR / "pe_top5_daily.parquet"
    if daily_file.exists():
        import pandas as pd
        pe_daily = pd.read_parquet(daily_file)
        if len(pe_daily) > 0:
            interp_date = str(pe_daily.index[-1].date())

    return results, pe_date, interp_date


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


def compute_portfolio_history(trades):
    history = []
    shares = 0
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
            "date": date, "shares": shares, "price": price,
            "position_value": position_value, "equity": equity,
            "target_alloc": alloc,
            "actual_alloc": position_value / equity if equity > 0 else 0,
            "probability": prob, "side": side or "hold",
            "quantity": qty, "executed": executed,
        })
    return history


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

@app.get("/img/backtest_1y")
def _b1y():
    return _serve(BACKTEST_1Y)

@app.get("/img/backtest_1m")
def _b1m():
    return _serve(BACKTEST_1M)

@app.get("/img/pe_chart")
def _pe():
    return _serve(PE_CHART)


# ── Styles (same as dashboard) ────────────────────────────

ui.add_head_html(f"""
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
""")


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

trades = load_trades()
portfolio = compute_portfolio_history(trades)
last_trade = trades[-1] if trades else {}

current_alloc = last_trade.get("target_alloc", 0)
current_prob = last_trade.get("probability", 0)
current_price = last_trade.get("etf_price", 0)
model_date = last_trade.get("model_date", "--")
n_exec = sum(1 for t in trades if t.get("executed"))

with ui.element("div").classes("layout"):
    # ── Sidebar ──
    with ui.element("div").classes("sidebar"):
        ui.element("div").classes("sidebar-photo")
        with ui.element("div").classes("sidebar-title"):
            ui.html("<h2>Risk-Off</h2>")
            ui.html('<div class="sub">QQQ / PUST — Boursorama PEA</div>')
        ui.element("div").classes("sidebar-divider")
        with ui.element("div").classes("sidebar-metrics"):
            alloc_color = "gain-positive" if current_alloc >= 0.5 else "gain-negative"
            for label, value, extra_class in [
                ("Allocation", f"{current_alloc*100:.0f}%", alloc_color),
                ("Probability", f"{current_prob:.3f}", ""),
                ("PUST", f"{current_price:.2f} EUR", ""),
                ("Trades", f"{n_exec}", ""),
                ("Model", model_date, "mono"),
            ]:
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
                        ui.label("Cela va forcer l'allocation a 0% et vendre toutes les positions.").style("color: #666")
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
            tab_1y = ui.tab("1 Year")
            tab_1m = ui.tab("1 Month")
            tab_ndx5 = ui.tab("NDX Top 5")
            tab_ndx20 = ui.tab("NDX Top 20")
            tab_trades = ui.tab("Trades")
            tab_alloc = ui.tab("Allocations")

        with ui.tab_panels(tabs, value=tab_full).classes("w-full flex-1 custom-tab-panels"):

            # ── Backtest Full ──
            with ui.tab_panel(tab_full):
                with ui.column().classes("tab-content"):
                    ui.element("div").classes("w-full h-0.5 bg-black")
                    ui.label("Walk-Forward Backtest — Full History").classes("text-base font-semibold")
                    if BACKTEST_FULL.exists():
                        ui.image("/img/backtest_full").classes("w-full rounded-lg shadow-lg")
                    else:
                        ui.label("Run: python src/risk_off_strategy/run.py QQQ").classes("text-gray-500")

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

            # ── NASDAQ-100 Top 5 & Top 20 (shared render function) ──
            def render_ndx_tab(n):
                ndx_data, pe_date, interp_date = load_ndx_top(n)
                if not ndx_data:
                    ui.label("No data. Run: python src/download_pe_qqq_top5.py").classes("text-gray-500")
                    return

                with ui.row().classes("gap-6"):
                    ui.label(f"PE trimestriel : {pe_date or '—'}").style("font-size: 0.8rem; color: #999")
                    ui.label(f"Interpolation : {interp_date or '—'}").style("font-size: 0.8rem; color: #999")

                labels = [d["symbol"] for d in ndx_data]
                caps = [d["mktCap"] / 1e9 for d in ndx_data]
                pes = [d["pe"] if d["pe"] and d["pe"] > 0 else 0 for d in ndx_data]
                total_cap = sum(caps)

                def pe_color(pe):
                    if pe <= 0: return "#999"
                    if pe < 25: return "#4caf50"
                    if pe < 35: return "#8bc34a"
                    if pe < 50: return "#ff9800"
                    return "#f44336"

                colors = [pe_color(p) for p in pes]
                hover_text = [
                    f"{d['name']}<br>${d['mktCap']/1e9:.0f}B<br>PE: {d['pe']:.1f}" if d['pe'] and d['pe'] > 0
                    else f"{d['name']}<br>${d['mktCap']/1e9:.0f}B<br>PE: N/A"
                    for d in ndx_data
                ]

                with ui.row().classes("w-full gap-4"):
                    fig_cap = go.Figure(go.Pie(
                        labels=labels, values=caps, textinfo="label+percent",
                        textposition="inside", hovertext=hover_text, hoverinfo="text",
                        marker=dict(colors=colors, line=dict(color="#fff", width=1)), hole=0.35,
                    ))
                    fig_cap.update_layout(
                        title=dict(text=f"Market Cap (${total_cap:.0f}B)", x=0.5),
                        showlegend=False, margin=dict(t=40, b=10, l=10, r=10), height=420,
                    )
                    ui.plotly(fig_cap).classes("flex-1")

                    pe_labels = [f"{s}\nPE {p:.0f}" if p > 0 else f"{s}\nN/A"
                                 for s, p in zip(labels, pes)]
                    fig_pe = go.Figure(go.Pie(
                        labels=pe_labels, values=caps, textinfo="label",
                        textposition="inside", hovertext=hover_text, hoverinfo="text",
                        marker=dict(colors=colors, line=dict(color="#fff", width=1)), hole=0.35,
                    ))
                    valid = [(c, p) for c, p in zip(caps, pes) if p > 0]
                    w_pe = sum(c * p for c, p in valid) / sum(c for c, p in valid) if valid else 0
                    fig_pe.update_layout(
                        title=dict(text=f"PE Ratio (weighted: {w_pe:.1f})", x=0.5),
                        showlegend=False, margin=dict(t=40, b=10, l=10, r=10), height=420,
                        annotations=[dict(text=f"{w_pe:.1f}", x=0.5, y=0.5,
                                          font_size=24, showarrow=False, font_color="#333")],
                    )
                    ui.plotly(fig_pe).classes("flex-1")

                with ui.row().classes("gap-4 mt-2"):
                    for color, lbl in [("#4caf50", "PE < 25"), ("#8bc34a", "PE 25-35"),
                                       ("#ff9800", "PE 35-50"), ("#f44336", "PE > 50"), ("#999", "N/A")]:
                        with ui.row().classes("items-center gap-1"):
                            ui.element("div").style(f"width:12px; height:12px; border-radius:50%; background:{color}")
                            ui.label(lbl).style("font-size: 0.75rem; color: #666")

                ui.element("div").classes("w-full h-0.5 bg-black mt-4")
                ndx_columns = [
                    {"name": "rank", "label": "#", "field": "rank", "align": "right"},
                    {"name": "symbol", "label": "Ticker", "field": "symbol", "align": "left"},
                    {"name": "name", "label": "Name", "field": "name", "align": "left"},
                    {"name": "cap", "label": "Cap ($B)", "field": "cap", "align": "right"},
                    {"name": "weight", "label": "Weight", "field": "weight", "align": "right"},
                    {"name": "pe", "label": "PE TTM", "field": "pe", "align": "right"},
                ]
                ndx_rows = []
                for i, d in enumerate(ndx_data, 1):
                    pe_val = d["pe"]
                    ndx_rows.append({
                        "rank": i, "symbol": d["symbol"], "name": d["name"],
                        "cap": f"{d['mktCap']/1e9:.0f}",
                        "weight": f"{d['mktCap']/1e9/total_cap*100:.1f}%",
                        "pe": f"{pe_val:.1f}" if pe_val and pe_val > 0 else "N/A",
                    })
                ui.table(columns=ndx_columns, rows=ndx_rows, row_key="rank").classes("w-full")

            with ui.tab_panel(tab_ndx5):
                with ui.column().classes("tab-content"):
                    ui.element("div").classes("w-full h-0.5 bg-black")
                    ui.label("NASDAQ-100 Top 5 — Market Cap & PE").classes("text-base font-semibold")
                    render_ndx_tab(5)

            with ui.tab_panel(tab_ndx20):
                with ui.column().classes("tab-content"):
                    ui.element("div").classes("w-full h-0.5 bg-black")
                    ui.label("NASDAQ-100 Top 20 — Market Cap & PE").classes("text-base font-semibold")
                    render_ndx_tab(20)

            # ── Trades ──
            with ui.tab_panel(tab_trades):
                with ui.column().classes("tab-content"):
                    ui.element("div").classes("w-full h-0.5 bg-black")
                    ui.label("Trade History").classes("text-base font-semibold")
                    if not trades:
                        ui.label("No trades yet.").classes("text-gray-500")
                    else:
                        columns = [
                            {"name": "date", "label": "Date", "field": "date", "align": "left"},
                            {"name": "side", "label": "Action", "field": "side", "align": "left"},
                            {"name": "quantity", "label": "Qty", "field": "quantity", "align": "right"},
                            {"name": "etf_price", "label": "Price", "field": "etf_price", "align": "right"},
                            {"name": "value", "label": "Value", "field": "value", "align": "right"},
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
                                "value": f"{qty * price:.0f} EUR",
                                "probability": f"{t.get('probability', 0):.3f}",
                                "target_alloc": f"{t.get('target_alloc', 0)*100:.0f}%",
                                "executed": "LIVE" if t.get("executed") else "DRY-RUN",
                            })
                        table = ui.table(columns=columns, rows=rows, row_key="date").classes("w-full")
                        table.add_slot("body-cell-side", """
                            <q-td :props="props">
                                <span :style="{ color: props.value === 'BUY' ? '#1f8f4c' : props.value === 'SELL' ? '#c0392b' : '#888',
                                                 fontWeight: 600 }">
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
                with ui.column().classes("tab-content"):
                    ui.element("div").classes("w-full h-0.5 bg-black")
                    ui.label("Allocation History").classes("text-base font-semibold")
                    if not portfolio:
                        ui.label("No allocation history yet.").classes("text-gray-500")
                    else:
                        columns = [
                            {"name": "date", "label": "Date", "field": "date", "align": "left"},
                            {"name": "shares", "label": "Shares", "field": "shares", "align": "right"},
                            {"name": "price", "label": "Price", "field": "price", "align": "right"},
                            {"name": "position", "label": "Position", "field": "position", "align": "right"},
                            {"name": "equity", "label": "Equity", "field": "equity", "align": "right"},
                            {"name": "target", "label": "Target", "field": "target", "align": "right"},
                            {"name": "actual", "label": "Actual", "field": "actual", "align": "right"},
                            {"name": "action", "label": "Action", "field": "action", "align": "center"},
                        ]
                        rows = []
                        for p in reversed(portfolio):
                            rows.append({
                                "date": p["date"],
                                "shares": p["shares"],
                                "price": f"{p['price']:.2f}",
                                "position": f"{p['position_value']:.0f} EUR",
                                "equity": f"{p['equity']:.0f} EUR",
                                "target": f"{p['target_alloc']*100:.0f}%",
                                "actual": f"{p['actual_alloc']*100:.0f}%",
                                "action": p["side"].upper(),
                            })
                        ui.table(columns=columns, rows=rows, row_key="date").classes("w-full")


ui.run(title="Risk-Off Strategy — Gregory Descamps", port=8081, reload=False)
