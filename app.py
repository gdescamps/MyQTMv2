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
ASSETS_DIR = ROOT / "dashboard" / "assets"
SIDEBAR_IMAGE = ASSETS_DIR / "sidebar.jpg"
TITLE_IMAGE = ASSETS_DIR / "title.jpg"

# Expose dashboard assets
if ASSETS_DIR.exists():
    app.add_static_files("/assets", str(ASSETS_DIR))


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
    try:
        from robot import compute_allocation
        return compute_allocation()
    except Exception as e:
        return {"error": str(e)}


def load_backtest_stats() -> dict:
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

@app.get("/chart/equity-12m")
def serve_equity_12m():
    """Generate 12-month equity + VIX overlay + allocation stackplot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    from io import BytesIO
    from starlette.responses import Response

    eq_path = OUTPUTS / "backtest_equity.csv"
    weights_path = OUTPUTS / "backtest_weights.parquet"
    regime_path = OUTPUTS / "backtest_regime_dates.json"
    vix_path = DATA / "fred_vix.parquet"
    if not eq_path.exists():
        return Response(status_code=404)

    eq = pd.read_csv(eq_path, parse_dates=["date"])
    last_date = eq["date"].iloc[-1]
    start_1y = last_date - pd.DateOffset(years=1)
    mask_1y = eq["date"] >= start_1y
    if mask_1y.sum() == 0:
        return Response(status_code=404)

    eq_1y = eq[mask_1y].copy()
    init = 150_000.0
    eq_1y["value"] = eq_1y["equity"] / eq_1y["equity"].iloc[0] * init
    returns_1y = eq_1y["equity"].pct_change().dropna()
    ret_1y = eq_1y["equity"].iloc[-1] / eq_1y["equity"].iloc[0] - 1
    sharpe_1y = (returns_1y.mean() / returns_1y.std() * np.sqrt(252)
                 if returns_1y.std() > 0 else 0)
    max_dd = (eq_1y["equity"] / eq_1y["equity"].cummax() - 1).min()
    start_str = eq_1y["date"].iloc[0].strftime("%b %Y")
    end_str = eq_1y["date"].iloc[-1].strftime("%b %Y")

    # Load ETF color map and short names
    from backtest import ETF_COLOR_MAP, _short_name
    from etf import UNIVERSE as _UNIV
    SHORT_NAMES = {e.bourso: _short_name(e.name) for e in _UNIV}

    # --- Figure: 3 rows (equity, regime table, allocation) + right legend ---
    fig = plt.figure(figsize=(22, 13))
    gs = fig.add_gridspec(3, 2, height_ratios=[3, 0.45, 2],
                          width_ratios=[3, 1],
                          hspace=0.06, wspace=0.02,
                          top=0.94, bottom=0.04, left=0.05, right=0.98)
    ax1 = fig.add_subplot(gs[0, 0])
    ax_tbl = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[2, 0], sharex=ax1)
    ax_leg = fig.add_subplot(gs[:, 1])
    ax_leg.axis("off")

    # Panel 1: Equity curve (linear scale)
    ax1.plot(eq_1y["date"], eq_1y["value"], lw=3, color="#d62728", zorder=10)
    ax1.axhline(init, color="#999", lw=1, ls="--", alpha=0.3)
    ax1.set_ylabel("Portfolio (linear scale, EUR)")
    ax1.yaxis.set_major_formatter(
        mtick.FuncFormatter(lambda x, _: f"{x:,.0f}".replace(",", " ")))
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor("#f8f8f8")

    fig.suptitle(
        f"Last 12 months ({start_str} -> {end_str})  |  {ret_1y:+.1%}  |  "
        f"Sharpe {sharpe_1y:.2f}  |  DD {max_dd:.1%}  |  "
        f"Last: {eq_1y['value'].iloc[-1]:,.0f} EUR",
        fontsize=13, fontweight="bold")

    # VIX overlay on equity panel
    if vix_path.exists():
        vix_s = pd.read_parquet(vix_path).iloc[:, 0]
        vix_12m = vix_s.reindex(pd.to_datetime(eq_1y["date"]),
                                method="ffill").dropna()
        if not vix_12m.empty:
            ax_vix = ax1.twinx()
            vix_ema100 = vix_12m.ewm(span=100).mean()

            # Color VIX by regime
            regime_dates = {}
            if regime_path.exists():
                with open(regime_path) as f:
                    rd = json.load(f)
                regime_dates = {k: set(v) for k, v in rd.items()}

            colors = pd.Series("#2ca02c", index=vix_12m.index)  # green default
            for d in vix_12m.index:
                ds = str(d.date())
                if ds in regime_dates.get("model", set()):
                    colors[d] = "#ff7f0e"   # orange SM
                elif ds in regime_dates.get("fl", set()):
                    colors[d] = "#1f77b4"   # blue FL
                elif ds in regime_dates.get("cash", set()):
                    colors[d] = "#d62728"   # red cash

            # Draw colored segments
            prev_c = colors.iloc[0]
            seg_start = 0
            for i in range(1, len(vix_12m)):
                if colors.iloc[i] != prev_c or i == len(vix_12m) - 1:
                    seg = slice(seg_start, i + 1)
                    ax_vix.fill_between(vix_12m.index[seg], vix_12m.values[seg],
                                        alpha=0.15, color=prev_c, linewidth=0)
                    ax_vix.plot(vix_12m.index[seg], vix_12m.values[seg],
                                color=prev_c, alpha=0.6, linewidth=0.8)
                    seg_start = i
                    prev_c = colors.iloc[i]

            ax_vix.plot(vix_ema100.index, vix_ema100.values,
                        color="#d62728", alpha=0.7, lw=2, ls=(0, (8, 4)))
            ax_vix.set_ylim(0, max(60, vix_12m.max() * 1.2))
            ax_vix.set_ylabel("VIX", color="#d62728", fontsize=8)
            ax_vix.tick_params(axis="y", labelcolor="#d62728", labelsize=7)

    # Regime table (between equity and allocation, same as backtest.py)
    from backtest import _draw_regime_table
    _draw_regime_table(ax_tbl)

    # Panel 2: Allocation stackplot
    if weights_path.exists():
        w_all = pd.read_parquet(weights_path)
        w_12m = w_all.loc[w_all.index >= start_1y].copy()
        if not w_12m.empty:
            w_12m = w_12m.reindex(pd.to_datetime(eq_1y["date"]),
                                  method="ffill").fillna(0)
            cash_w = (1 - w_12m.sum(axis=1)).clip(0, 1)

            # Only show allocated ETFs
            universe_order = [e.bourso for e in _UNIV]
            allocated = [t for t in universe_order
                         if t in w_12m.columns and w_12m[t].mean() > 0.001]
            w_sorted = w_12m[allocated]
            col_names = {e: SHORT_NAMES.get(e, e) for e in w_sorted.columns}
            w_named = w_sorted.rename(columns=col_names)
            plot_data = w_named.copy()
            plot_data["Cash"] = cash_w

            color_map = {SHORT_NAMES.get(t, t): c
                         for t, c in ETF_COLOR_MAP.items()}
            color_map["Cash"] = "#e8e8e8"
            colors_alloc = [color_map.get(c, "#7f7f7f")
                            for c in plot_data.columns]

            ax2.stackplot(plot_data.index, plot_data.values.T,
                          labels=plot_data.columns,
                          colors=colors_alloc, alpha=0.85)
            ax2.set_ylabel("Allocation")
            ax2.set_ylim(0, 1)
            ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
            ax2.set_facecolor("#f8f8f8")
            ax2.grid(True, alpha=0.3)

            # Right column: ETF list sorted by mean allocation
            mean_alloc = w_sorted.mean().sort_values(ascending=False)
            max_a = mean_alloc.max() if len(mean_alloc) > 0 else 1.0

            # Header
            y = 0.97
            ax_leg.text(0.0, y, "ETF Allocation (12m avg)",
                        fontsize=13, fontweight="bold", va="top",
                        transform=ax_leg.transAxes, color="#d62728")
            y -= 0.05
            ax_leg.plot([0.0, 0.97], [y + 0.01, y + 0.01],
                        color="#999999", lw=0.8, transform=ax_leg.transAxes)

            marker_max = 22
            marker_min = 6
            y_step = min(0.04, 0.85 / max(len(mean_alloc), 1))

            for etf_id in mean_alloc.index:
                short = SHORT_NAMES.get(etf_id, etf_id)
                color = ETF_COLOR_MAP.get(etf_id, "#7f7f7f")
                alloc_pct = mean_alloc[etf_id]
                frac = alloc_pct / max_a if max_a > 0 else 0
                m_fs = marker_min + (marker_max - marker_min) * frac

                ax_leg.text(0.0, y, "\u25A0", fontsize=m_fs, color=color,
                            va="center", transform=ax_leg.transAxes)
                ax_leg.text(0.08, y, short, fontsize=10, va="center",
                            transform=ax_leg.transAxes)
                ax_leg.text(0.70, y, f"{alloc_pct:.0%}", fontsize=10,
                            va="center", ha="right",
                            transform=ax_leg.transAxes, color="#333333")
                y -= y_step

    plt.setp(ax1.get_xticklabels(), visible=False)
    fig.align_ylabels([ax1, ax2])

    buf = BytesIO()
    fig.savefig(buf, format="jpeg", dpi=120, pil_kwargs={"quality": 85})
    plt.close(fig)
    buf.seek(0)
    return Response(content=buf.read(), media_type="image/jpeg",
                    headers={"Cache-Control": "no-cache"})


# ===================================================================
#  Styles (same layout as ./dashboard)
# ===================================================================

SIDEBAR_BG = f"url('/assets/{SIDEBAR_IMAGE.name}')" if SIDEBAR_IMAGE.exists() else "linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)"

ui.add_head_html(shared=True, code=f"""
<style>
    html, body {{
        margin: 0; height: 100%; overflow: hidden;
        background: #f0f0f0;
        font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont,
                     'Segoe UI', sans-serif;
    }}
    #app, .nicegui-content {{ height: 100%; margin: 0; padding: 0; }}

    .layout {{ display: flex; height: 100vh; width: 100%; }}

    .presentation-panel {{
        flex: 0 0 20%; height: 100%;
        background-image: {SIDEBAR_BG};
        background-size: cover; background-position: top;
        background-repeat: no-repeat;
        display: flex; flex-direction: column;
        justify-content: flex-end; align-items: center;
        padding-bottom: 32px;
    }}
    .sidebar-title {{
        color: rgba(255,255,255,0.92); font-size: 1.1rem; font-weight: 700;
        text-shadow: 0 2px 12px rgba(0,0,0,0.5);
        letter-spacing: 0.05em;
    }}
    .sidebar-subtitle {{
        color: rgba(255,255,255,0.65); font-size: 0.78rem;
        text-shadow: 0 1px 8px rgba(0,0,0,0.4);
        margin-top: 4px;
    }}

    .content-panel {{
        flex: 1; background: #ffffff; color: #222;
        display: flex; flex-direction: column;
        min-height: 0; overflow: hidden; box-sizing: border-box;
    }}

    .custom-tabs .q-tab__label {{
        font-size: clamp(0.72rem, 0.5vw + 0.45rem, 0.95rem);
        font-weight: 600;
    }}
    .custom-tabs .q-tab {{ min-height: 42px; }}
    .custom-tab-panels {{ flex: 1; min-height: 0; overflow-y: auto; }}
    .custom-tab-panels .q-tab-panel {{ padding: 0; }}

    .tab-content {{
        width: 100%; max-width: 1200px; margin: 0 auto;
        display: flex; flex-direction: column;
        gap: 20px; padding: 20px 24px 32px;
    }}

    .stat-card {{
        background: #f8f8f8; border-radius: 12px; padding: 16px 20px;
        border: 1px solid #e0e0e0;
    }}
    .stat-value {{ font-size: 1.6rem; font-weight: 700; color: #d62728; }}
    .stat-label {{ font-size: 0.82rem; color: #666; margin-top: 2px; }}
    .gate-open {{ font-size: 1.6rem; color: #1f8f4c; font-weight: 700; }}
    .gate-closed {{ font-size: 1.6rem; color: #c0392b; font-weight: 700; }}

    .regime-badge {{
        display: inline-block; padding: 6px 18px; border-radius: 20px;
        font-weight: 700; font-size: 0.95rem;
    }}
    .regime-heuristic {{ background: #2ca02c18; color: #2ca02c; border: 2px solid #2ca02c; }}
    .regime-smart_money {{ background: #ff7f0e18; color: #ff7f0e; border: 2px solid #ff7f0e; }}
    .regime-follow_leads {{ background: #1f77b418; color: #1f77b4; border: 2px solid #1f77b4; }}
    .regime-cash {{ background: #d6272818; color: #d62728; border: 2px solid #d62728; }}

    .section-title {{
        font-size: 1.15rem; font-weight: 700; color: #222;
        border-bottom: 2px solid #111; padding-bottom: 4px;
        margin-top: 8px;
    }}

    @media (max-width: 768px) {{
        .layout {{ flex-direction: column; }}
        .presentation-panel {{ display: none; }}
        .tab-content {{ padding: 12px 10px 20px; gap: 14px; }}
        .custom-tabs .q-tab__label {{ font-size: 0.78rem; }}
    }}
</style>
""")


# ===================================================================
#  Page
# ===================================================================

@ui.page("/")
def main_page():

    with ui.element("div").classes("layout"):
        # ---- Sidebar (same as ./dashboard) ----
        with ui.element("div").classes("presentation-panel"):
            ui.label("MyQTM-ETF").classes("sidebar-title")
            ui.label("Quantitative ETF Strategy").classes("sidebar-subtitle")

        # ---- Content ----
        with ui.column().classes("content-panel"):

            with ui.tabs().classes(
                    "w-full custom-tabs").props("dense active-color=red"
                    ) as tabs:
                tab_robot = ui.tab("Robot")
                tab_equity = ui.tab("Equity")
                tab_robust = ui.tab("Robustness")
                tab_universe = ui.tab("Universe")

            with ui.tab_panels(tabs, value=tab_robot).classes(
                    "w-full flex-1 custom-tab-panels"):

                # ======== TAB 1: Robot ========
                with ui.tab_panel(tab_robot):
                    with ui.column().classes("tab-content"):
                        _build_robot_tab()

                # ======== TAB 2: Equity ========
                with ui.tab_panel(tab_equity):
                    with ui.column().classes("tab-content"):
                        _build_equity_tab()

                # ======== TAB 3: Robustness ========
                with ui.tab_panel(tab_robust):
                    with ui.column().classes("tab-content"):
                        _build_robustness_tab()

                # ======== TAB 4: Universe ========
                with ui.tab_panel(tab_universe):
                    with ui.column().classes("tab-content"):
                        _build_universe_tab()


# ===================================================================
#  Tab builders
# ===================================================================

def _build_robot_tab():
    ui.label("Robot State").classes("section-title")

    alloc = load_allocation()
    ic_gate = load_ic_gate()
    step_info = load_step_info()

    if not alloc or "error" in alloc:
        err = alloc.get("error", "No data") if alloc else "No allocation data"
        ui.label(f"Error: {err}").classes("text-red-600")
        return

    regime = alloc.get("regime", "unknown")
    regime_labels = {
        "heuristic": "Heuristic (top-5 Sharpe 2y)",
        "smart_money": "Smart Money Model",
        "follow_leads": "Follow Leads Model",
        "cash": "Cash (no positions)",
    }
    with ui.row().classes("items-center gap-4"):
        ui.html(f'<span class="regime-badge regime-{regime}">'
                f'{regime_labels.get(regime, regime)}</span>')
        ui.label(f"Data: {alloc.get('date', '?')}").classes(
            "text-sm text-gray-500")

    # Stats row
    with ui.row().classes("gap-3 flex-wrap"):
        _stat_card("VIX EMA100", f"{alloc.get('vix_ema100', 0):.1f}",
                   f"slope: {alloc.get('vix_slope', 0):+.3f}")

        if ic_gate:
            sm = ic_gate["smart_money"]
            _stat_card("SM Gate",
                       f"{'OPEN' if sm['gate_open'] else 'CLOSED'}",
                       f"EMA={sm['ic_ema']:.4f}  thr={sm['threshold']}",
                       value_class="gate-open" if sm["gate_open"] else "gate-closed")

            fl = ic_gate["follow_leads"]
            _stat_card("FL Gate",
                       f"{'OPEN' if fl['gate_open'] else 'CLOSED'}",
                       f"EMA={fl['ic_ema']:.4f}  thr={fl['threshold']}",
                       value_class="gate-open" if fl["gate_open"] else "gate-closed")

        if step_info:
            _stat_card("Last Train",
                       f"Step {step_info['last_step']}",
                       f"{step_info['test_start']} -> {step_info['test_end']}")

    # Allocation table
    if alloc.get("weights"):
        ui.element("div").classes("w-full h-0.5 bg-black mt-2")
        model_label = f" [{alloc.get('model_used', 'heuristic')}]" if alloc.get("model_used") else ""
        ui.label(f"Current Allocation{model_label}").classes("section-title")
        from etf import UNIVERSE as _U, TRADING_MAP as _TM
        alloc_rows = []
        for etf_id in alloc.get("top_etfs", []):
            w = alloc["weights"].get(etf_id, 0)
            sh = alloc["sharpe_all"].get(etf_id, 0)
            name = next((e.name for e in _U if e.bourso == etf_id), etf_id)
            ucits = _TM.get(etf_id, (etf_id,))[0]
            exch = _TM.get(etf_id, ("", "", "", "", ""))[1]
            alloc_rows.append({
                "proxy": etf_id, "ucits": ucits, "exchange": exch,
                "name": name, "weight": f"{w:.1%}", "sharpe": f"{sh:.3f}",
            })
        ui.table(
            columns=[
                {"name": "proxy", "label": "Proxy", "field": "proxy"},
                {"name": "ucits", "label": "UCITS", "field": "ucits"},
                {"name": "exchange", "label": "Exchange", "field": "exchange"},
                {"name": "name", "label": "Name", "field": "name"},
                {"name": "weight", "label": "Weight", "field": "weight",
                 "align": "right"},
                {"name": "sharpe", "label": "Sharpe 2y", "field": "sharpe",
                 "align": "right"},
            ],
            rows=alloc_rows, row_key="proxy",
        ).classes("w-full")
    elif regime == "cash":
        ui.label("No positions (cash regime)").classes(
            "text-lg text-red-600 font-semibold mt-2")

    # Trade history
    logs = load_trade_logs()
    if logs:
        ui.element("div").classes("w-full h-0.5 bg-black mt-4")
        ui.label("Trade History").classes("section-title")
        for log in logs[:5]:
            with ui.expansion(log.get("datetime", "?")).classes("w-full"):
                trades = log.get("trades", [])
                if trades:
                    ui.table(
                        columns=[
                            {"name": "etf", "label": "ETF", "field": "etf"},
                            {"name": "action", "label": "Action", "field": "action"},
                            {"name": "qty", "label": "Qty", "field": "qty",
                             "align": "right"},
                            {"name": "fill", "label": "Fill", "field": "fill",
                             "align": "right"},
                        ],
                        rows=trades, row_key="etf",
                    ).classes("w-full")
                else:
                    ui.label("No trades executed")


def _build_equity_tab():
    stats = load_backtest_stats()

    # Last 12 months zoom (150k EUR starting capital)
    ui.label("Last 12 Months (150 000 EUR)").classes("section-title")
    ui.image("/chart/equity-12m").classes("w-full rounded-lg shadow-lg")

    # Full backtest
    ui.element("div").classes("w-full h-0.5 bg-black mt-4")
    ui.label("Full Backtest (2006-2026, 150 000 EUR)").classes("section-title")

    if stats:
        with ui.row().classes("gap-3 flex-wrap"):
            _stat_card("Sharpe", stats["sharpe"])
            _stat_card("Ann. Return", stats["ann_return"])
            _stat_card("Max DD", stats["max_dd"])
            _stat_card("Total Return", stats["total_return"])
        ui.label(f"Period: {stats['period']}").classes("text-sm text-gray-500")

    if (OUTPUTS / "backtest_equity.jpg").exists():
        ui.image("/chart/equity").classes("w-full rounded-lg shadow-lg")
    else:
        ui.label("No chart. Run: python backtest.py").classes("text-red-600")


def _build_robustness_tab():
    ui.label("Robustness (Monte Carlo)").classes("section-title")

    rob = load_robustness_stats()
    if rob:
        with ui.row().classes("gap-3 flex-wrap"):
            _stat_card("Runs", str(rob["n_runs"]))
            _stat_card("Median Sharpe", rob["median_sharpe"])
            _stat_card("Range", f"{rob['min_sharpe']} - {rob['max_sharpe']}")
            _stat_card("Median Ann.", rob["median_ann"])
            _stat_card("Median DD", rob["median_dd"])

    ui.element("div").classes("w-full h-0.5 bg-black")
    if (OUTPUTS / "backtest_robustness.jpg").exists():
        ui.image("/chart/robustness").classes("w-full rounded-lg shadow-lg")
    else:
        ui.label("No chart. Run: python backtest.py --robustness").classes(
            "text-red-600")


def _build_universe_tab():
    ui.label("ETF Universe (26 ETFs)").classes("section-title")
    ui.label("US proxy tickers for backtesting, UCITS EUR for live trading"
             ).classes("text-sm text-gray-500")

    from etf import UNIVERSE as _UNIV, TRADING_MAP as _TMAP
    rows = []
    for etf in _UNIV:
        tm = _TMAP.get(etf.bourso, ("?", "?", "?", "?", "?"))
        rows.append({
            "proxy": etf.bourso, "name": etf.name, "section": etf.section,
            "ucits": tm[0], "exchange": tm[1], "currency": tm[2],
            "issuer": tm[3], "fee_model": tm[4],
        })

    ui.table(
        columns=[
            {"name": "proxy", "label": "Proxy", "field": "proxy", "sortable": True},
            {"name": "name", "label": "Name", "field": "name", "sortable": True},
            {"name": "section", "label": "Section", "field": "section",
             "sortable": True},
            {"name": "ucits", "label": "UCITS", "field": "ucits", "sortable": True},
            {"name": "exchange", "label": "Exchange", "field": "exchange",
             "sortable": True},
            {"name": "currency", "label": "Curr", "field": "currency"},
            {"name": "issuer", "label": "Issuer", "field": "issuer",
             "sortable": True},
            {"name": "fee_model", "label": "Fee", "field": "fee_model"},
        ],
        rows=rows, row_key="proxy",
        pagination={"rowsPerPage": 30},
    ).classes("w-full")


def _stat_card(label: str, value: str, subtitle: str = "",
               value_class: str = "stat-value"):
    with ui.column().classes("stat-card min-w-[130px]"):
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
