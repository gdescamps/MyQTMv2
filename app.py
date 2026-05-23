"""
MyQTM-ETF Dashboard — NiceGUI web app.

Tabs:
  1. Robot     — current regime, IC gates, allocation, trade history
  2. Equity    — backtest equity chart (2006-2026)
  3. Robustness— robustness Monte Carlo chart
  4. Universe  — 26 ETF universe with UCITS mapping

Usage:
    python app.py                  # start on port 8080
    python app.py --port 8888      # custom port
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.responses import FileResponse
from nicegui import app, ui

sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).parent
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
ROBOT_DIR = OUTPUTS / "robot"
ROBOT_DATA = DATA / "robot"


# ===================================================================
#  Data loading helpers
# ===================================================================

def load_ic_gate() -> dict | None:
    path = ROBOT_DIR / "ic_gate_state.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_step_info() -> dict | None:
    path = ROBOT_DIR / "last_step_info.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_trade_logs() -> list[dict]:
    """Load all trade logs from data/robot/, newest first."""
    logs = []
    if not ROBOT_DATA.exists():
        return logs
    for f in sorted(ROBOT_DATA.glob("*_trades.json"), reverse=True):
        try:
            with open(f) as fh:
                logs.append(json.load(fh))
        except (json.JSONDecodeError, OSError):
            pass
    return logs


def load_allocation() -> dict | None:
    """Run allocation computation and return result."""
    try:
        from robot import compute_allocation
        return compute_allocation()
    except Exception as e:
        return {"error": str(e)}


def load_backtest_stats() -> dict:
    """Load backtest stats from equity CSV."""
    stats = {}
    eq_path = OUTPUTS / "backtest_equity.csv"
    if eq_path.exists():
        eq = pd.read_csv(eq_path, parse_dates=["date"])
        if len(eq) > 0:
            returns = eq["equity"].pct_change().dropna()
            n_years = len(returns) / 252
            total_ret = eq["equity"].iloc[-1] - 1
            ann_ret = (1 + total_ret) ** (1 / n_years) - 1
            sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
            max_dd = (eq["equity"] / eq["equity"].cummax() - 1).min()
            stats = {
                "period": f"{eq['date'].iloc[0].date()} -> {eq['date'].iloc[-1].date()}",
                "sharpe": f"{sharpe:.3f}",
                "ann_return": f"{ann_ret:+.1%}",
                "max_dd": f"{max_dd:.1%}",
                "total_return": f"{total_ret:.0%}",
            }
    return stats


def load_robustness_stats() -> dict:
    """Load robustness stats from CSV."""
    path = OUTPUTS / "backtest_robustness.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return {
        "n_runs": len(df),
        "median_sharpe": f"{df['sharpe'].median():.2f}",
        "min_sharpe": f"{df['sharpe'].min():.2f}",
        "max_sharpe": f"{df['sharpe'].max():.2f}",
        "median_ann": f"{df['ann_pct'].median():+.1f}%",
        "median_dd": f"{df['max_dd_pct'].median():.1f}%",
    }


# ===================================================================
#  Static file serving
# ===================================================================

@app.get("/chart/equity")
def serve_equity():
    path = OUTPUTS / "backtest_equity.jpg"
    if path.exists():
        return FileResponse(path, media_type="image/jpeg",
                            headers={"Cache-Control": "no-cache"})

@app.get("/chart/robustness")
def serve_robustness():
    path = OUTPUTS / "backtest_robustness.jpg"
    if path.exists():
        return FileResponse(path, media_type="image/jpeg",
                            headers={"Cache-Control": "no-cache"})


# ===================================================================
#  Styles
# ===================================================================

ui.add_head_html(shared=True, code="""
<style>
    body { font-family: 'Inter', system-ui, sans-serif; }
    .stat-card {
        background: #f8f8f8; border-radius: 12px; padding: 16px 20px;
        border: 1px solid #e0e0e0;
    }
    .stat-value { font-size: 1.8rem; font-weight: 700; color: #d62728; }
    .stat-label { font-size: 0.85rem; color: #666; margin-top: 2px; }
    .gate-open { color: #1f8f4c; font-weight: 700; }
    .gate-closed { color: #c0392b; font-weight: 700; }
    .regime-badge {
        display: inline-block; padding: 4px 14px; border-radius: 20px;
        font-weight: 700; font-size: 0.9rem;
    }
    .regime-heuristic { background: #2ca02c22; color: #2ca02c; border: 2px solid #2ca02c; }
    .regime-smart_money { background: #ff7f0e22; color: #ff7f0e; border: 2px solid #ff7f0e; }
    .regime-follow_leads { background: #1f77b422; color: #1f77b4; border: 2px solid #1f77b4; }
    .regime-cash { background: #d6272822; color: #d62728; border: 2px solid #d62728; }
</style>
""")


# ===================================================================
#  Pages
# ===================================================================

@ui.page("/")
def main_page():
    with ui.header().classes("bg-gray-900 text-white items-center"):
        ui.label("MyQTM-ETF").classes("text-xl font-bold")
        ui.label("Quantitative ETF Momentum Strategy").classes("text-sm text-gray-400 ml-4")

    with ui.tabs().classes("w-full").props("dense active-color=red") as tabs:
        tab_robot = ui.tab("Robot")
        tab_equity = ui.tab("Equity Backtest")
        tab_robust = ui.tab("Robustness")
        tab_universe = ui.tab("Universe")

    with ui.tab_panels(tabs, value=tab_robot).classes("w-full flex-1"):

        # ---- TAB 1: Robot ----
        with ui.tab_panel(tab_robot):
            with ui.column().classes("w-full max-w-5xl mx-auto gap-4 p-4"):
                ui.label("Robot State").classes("text-2xl font-bold")

                alloc = load_allocation()
                ic_gate = load_ic_gate()
                step_info = load_step_info()

                if alloc and "error" not in alloc:
                    # Regime badge
                    regime = alloc.get("regime", "unknown")
                    regime_labels = {
                        "heuristic": "Heuristic (top-5 Sharpe 2y)",
                        "smart_money": "Smart Money Model",
                        "follow_leads": "Follow Leads Model",
                        "cash": "Cash (no positions)",
                    }
                    with ui.row().classes("items-center gap-4"):
                        ui.label("Regime:").classes("text-lg font-semibold")
                        ui.html(f'<span class="regime-badge regime-{regime}">'
                                f'{regime_labels.get(regime, regime)}</span>')
                        ui.label(f"Data date: {alloc.get('date', '?')}").classes(
                            "text-sm text-gray-500")

                    # Stats cards
                    with ui.row().classes("gap-4 flex-wrap"):
                        _stat_card("VIX EMA100", f"{alloc.get('vix_ema100', 0):.1f}",
                                   f"slope: {alloc.get('vix_slope', 0):+.3f}")

                        if ic_gate:
                            sm = ic_gate["smart_money"]
                            sm_cls = "gate-open" if sm["gate_open"] else "gate-closed"
                            _stat_card("SM IC Gate",
                                       f"{'OPEN' if sm['gate_open'] else 'CLOSED'}",
                                       f"EMA={sm['ic_ema']:.4f} (thr={sm['threshold']})",
                                       value_class=sm_cls)

                            fl = ic_gate["follow_leads"]
                            fl_cls = "gate-open" if fl["gate_open"] else "gate-closed"
                            _stat_card("FL IC Gate",
                                       f"{'OPEN' if fl['gate_open'] else 'CLOSED'}",
                                       f"EMA={fl['ic_ema']:.4f} (thr={fl['threshold']})",
                                       value_class=fl_cls)

                        if step_info:
                            _stat_card("Last Training",
                                       f"Step {step_info['last_step']}",
                                       f"{step_info['test_start']} -> {step_info['test_end']}")

                    # Current allocation table
                    if alloc.get("weights"):
                        ui.label("Current Allocation").classes(
                            "text-lg font-semibold mt-4")
                        from etf import UNIVERSE as _U, TRADING_MAP as _TM
                        alloc_rows = []
                        for etf_id in alloc.get("top_etfs", []):
                            w = alloc["weights"].get(etf_id, 0)
                            sh = alloc["sharpe_all"].get(etf_id, 0)
                            name = next((e.name for e in _U if e.bourso == etf_id),
                                        etf_id)
                            ucits = _TM.get(etf_id, (etf_id,))[0]
                            exch = _TM.get(etf_id, ("", "", "", "", ""))[1]
                            alloc_rows.append({
                                "proxy": etf_id,
                                "ucits": ucits,
                                "exchange": exch,
                                "name": name,
                                "weight": f"{w:.1%}",
                                "sharpe": f"{sh:.3f}",
                            })
                        ui.table(
                            columns=[
                                {"name": "proxy", "label": "Proxy", "field": "proxy"},
                                {"name": "ucits", "label": "UCITS", "field": "ucits"},
                                {"name": "exchange", "label": "Exchange",
                                 "field": "exchange"},
                                {"name": "name", "label": "Name", "field": "name"},
                                {"name": "weight", "label": "Weight", "field": "weight",
                                 "align": "right"},
                                {"name": "sharpe", "label": "Sharpe 2y",
                                 "field": "sharpe", "align": "right"},
                            ],
                            rows=alloc_rows,
                            row_key="proxy",
                        ).classes("w-full")
                    elif regime == "cash":
                        ui.label("No positions (cash regime)").classes(
                            "text-lg text-red-600 font-semibold mt-4")
                else:
                    err = alloc.get("error", "Unknown error") if alloc else "No allocation data"
                    ui.label(f"Allocation error: {err}").classes("text-red-600")

                # Trade history
                logs = load_trade_logs()
                if logs:
                    ui.label("Recent Trades").classes("text-lg font-semibold mt-6")
                    for log in logs[:5]:
                        with ui.expansion(log.get("datetime", "?")).classes("w-full"):
                            trades = log.get("trades", [])
                            if trades:
                                ui.table(
                                    columns=[
                                        {"name": "etf", "label": "ETF", "field": "etf"},
                                        {"name": "action", "label": "Action",
                                         "field": "action"},
                                        {"name": "qty", "label": "Qty", "field": "qty",
                                         "align": "right"},
                                        {"name": "fill", "label": "Fill Price",
                                         "field": "fill", "align": "right"},
                                    ],
                                    rows=trades,
                                    row_key="etf",
                                ).classes("w-full")
                            else:
                                ui.label("No trades executed")

        # ---- TAB 2: Equity Backtest ----
        with ui.tab_panel(tab_equity):
            with ui.column().classes("w-full max-w-6xl mx-auto gap-4 p-4"):
                ui.label("Equity Backtest (OOS 2006-2026)").classes(
                    "text-2xl font-bold")

                stats = load_backtest_stats()
                if stats:
                    with ui.row().classes("gap-4 flex-wrap"):
                        _stat_card("Sharpe", stats["sharpe"])
                        _stat_card("Ann. Return", stats["ann_return"])
                        _stat_card("Max DD", stats["max_dd"])
                        _stat_card("Total Return", stats["total_return"])
                    ui.label(f"Period: {stats['period']}").classes(
                        "text-sm text-gray-500")

                if (OUTPUTS / "backtest_equity.jpg").exists():
                    ui.image("/chart/equity").classes(
                        "w-full rounded-lg shadow-lg")
                else:
                    ui.label("No equity chart found. Run: python backtest.py").classes(
                        "text-red-600")

        # ---- TAB 3: Robustness ----
        with ui.tab_panel(tab_robust):
            with ui.column().classes("w-full max-w-6xl mx-auto gap-4 p-4"):
                ui.label("Robustness (Monte Carlo)").classes("text-2xl font-bold")

                rob = load_robustness_stats()
                if rob:
                    with ui.row().classes("gap-4 flex-wrap"):
                        _stat_card("Runs", str(rob["n_runs"]))
                        _stat_card("Median Sharpe", rob["median_sharpe"])
                        _stat_card("Sharpe Range",
                                   f"{rob['min_sharpe']} - {rob['max_sharpe']}")
                        _stat_card("Median Ann.", rob["median_ann"])
                        _stat_card("Median DD", rob["median_dd"])

                if (OUTPUTS / "backtest_robustness.jpg").exists():
                    ui.image("/chart/robustness").classes(
                        "w-full rounded-lg shadow-lg")
                else:
                    ui.label("No robustness chart. Run: python backtest.py --robustness"
                             ).classes("text-red-600")

        # ---- TAB 4: Universe ----
        with ui.tab_panel(tab_universe):
            with ui.column().classes("w-full max-w-6xl mx-auto gap-4 p-4"):
                ui.label("ETF Universe (26 ETFs)").classes("text-2xl font-bold")
                ui.label("US proxy tickers for backtesting, UCITS EUR for live trading"
                         ).classes("text-sm text-gray-500")

                from etf import UNIVERSE as _UNIV, TRADING_MAP as _TMAP
                rows = []
                for etf in _UNIV:
                    tm = _TMAP.get(etf.bourso, ("?", "?", "?", "?", "?"))
                    rows.append({
                        "proxy": etf.bourso,
                        "name": etf.name,
                        "section": etf.section,
                        "ucits": tm[0],
                        "exchange": tm[1],
                        "currency": tm[2],
                        "issuer": tm[3],
                        "fee_model": tm[4],
                    })

                ui.table(
                    columns=[
                        {"name": "proxy", "label": "Proxy (backtest)",
                         "field": "proxy", "sortable": True},
                        {"name": "name", "label": "Name", "field": "name",
                         "sortable": True},
                        {"name": "section", "label": "Section", "field": "section",
                         "sortable": True},
                        {"name": "ucits", "label": "UCITS (live)", "field": "ucits",
                         "sortable": True},
                        {"name": "exchange", "label": "Exchange", "field": "exchange",
                         "sortable": True},
                        {"name": "currency", "label": "Curr", "field": "currency"},
                        {"name": "issuer", "label": "Issuer", "field": "issuer",
                         "sortable": True},
                        {"name": "fee_model", "label": "Fee Model",
                         "field": "fee_model"},
                    ],
                    rows=rows,
                    row_key="proxy",
                    pagination={"rowsPerPage": 30},
                ).classes("w-full")


def _stat_card(label: str, value: str, subtitle: str = "",
               value_class: str = "stat-value"):
    with ui.column().classes("stat-card min-w-[140px]"):
        ui.label(value).classes(value_class)
        ui.label(label).classes("stat-label")
        if subtitle:
            ui.label(subtitle).classes("text-xs text-gray-400")


# ===================================================================
#  Entry point
# ===================================================================

if __name__ in {"__main__", "__mp_main__"}:
    parser = argparse.ArgumentParser(description="MyQTM-ETF Dashboard")
    parser.add_argument("--port", type=int, default=8080)
    args, _ = parser.parse_known_args()
    ui.run(title="MyQTM-ETF Dashboard", port=args.port, reload=False)
