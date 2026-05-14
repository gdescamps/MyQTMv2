"""
Backtest analysis: capital utilisation, ETF frequency, equity curve vs S&P 500.

Output: outputs/backtest_analysis.png
        outputs/etf_stats.csv
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

DATA    = Path(__file__).parent / "data"
OUTPUTS = Path(__file__).parent / "outputs"
OUTPUTS.mkdir(exist_ok=True)

EQUITY_SECTIONS    = {"geo", "sector_us", "thematic"}
DEFENSIVE_SECTIONS = {"bond", "commodity", "crypto"}

PARAM_BOUNDS = [
    (-0.20, 0.00), (-0.80, 0.00), (-3.00, 3.00),
    ( 0.50, 0.95), ( 0.00, 0.20), ( 0.50, 4.00),
    ( 0.05, 0.50), ( 0.01, 0.10), ( 0.05, 0.40),
]

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

def softmax_masked(s, mask, T):
    """Softmax over masked columns. NaN scores → ETF excluded from softmax that day."""
    out = np.zeros_like(s)
    if not mask.any():
        return out
    s_m = s[:, mask].copy() / max(T, 1e-6)
    # For each row, only include non-NaN entries
    for i in range(s_m.shape[0]):
        row = s_m[i]
        valid = ~np.isnan(row)
        if not valid.any():
            continue
        row_v = row[valid] - row[valid].max()
        e = np.exp(row_v)
        probs = e / e.sum()
        result = np.zeros(len(row))
        result[valid] = probs
        idx = np.where(mask)[0]
        out[i, idx] = result
    return out

def clip_params(params):
    return [float(np.clip(v, lo, hi)) for v, (lo, hi) in zip(params, PARAM_BOUNDS)]

def compute_weights(scores_wide, vix_s, hy_z60_s, is_equity, is_defensive, params):
    params = clip_params(params)
    p1, p2, p3, equity_max, cash_min, temp, max_w, min_change, def_min = params

    vix    = vix_s.reindex(scores_wide.index, method="ffill").fillna(20.0).values
    hy_z60 = hy_z60_s.reindex(scores_wide.index, method="ffill").fillna(0.0).values
    danger        = sigmoid(p1 * vix + p2 * hy_z60 + p3)
    equity_budget = equity_max * (1.0 - danger)
    def_budget    = np.where(
        danger > 0.5,
        def_min + (1.0 - equity_budget - cash_min) * (1 - danger),
        (1.0 - equity_budget - cash_min) * (1 - danger),
    ).clip(0, None)

    scores = scores_wide.values
    w_eq   = softmax_masked(scores, is_equity,    temp) * equity_budget[:, None]
    w_def  = softmax_masked(scores, is_defensive, temp) * def_budget[:, None]
    weights = np.minimum(w_eq + w_def, max_w)

    # Anti-churn
    prev = np.zeros(weights.shape[1])
    final = np.empty_like(weights)
    for t in range(len(weights)):
        delta = weights[t] - prev
        applied = np.where(np.abs(delta) > min_change, weights[t], prev)
        final[t] = applied
        prev = applied

    return pd.DataFrame(final, index=scores_wide.index, columns=scores_wide.columns)


def main():
    from etf import UNIVERSE
    sections = {e.bourso: e.section for e in UNIVERSE}

    print("Loading data...")
    oos = pd.read_parquet(DATA / "oos_predictions.parquet")
    oos = oos.reset_index()
    oos["date"] = pd.to_datetime(oos["date"])
    oos = oos.set_index(["date", "etf_id"])
    if oos.index.duplicated().any():
        oos = oos.sort_values("step").groupby(level=["date","etf_id"]).last()

    best_params = np.load(OUTPUTS / "best_params.npy").tolist()
    port_ret    = pd.read_parquet(DATA / "backtest_results.parquet")["port_return"]

    scores_wide = oos["score"].unstack("etf_id").sort_index()
    labels_wide = oos["label"].unstack("etf_id").reindex(columns=scores_wide.columns)
    all_etfs    = scores_wide.columns.tolist()

    is_equity    = np.array([sections.get(e) in EQUITY_SECTIONS    for e in all_etfs])
    is_defensive = np.array([sections.get(e) in DEFENSIVE_SECTIONS for e in all_etfs])

    # Macro
    vix_s    = pd.read_parquet(DATA/"fred_vix.parquet").iloc[:,0]
    hy_raw   = pd.read_parquet(DATA/"fred_hy_spread.parquet").iloc[:,0]
    hy_z60_s = (hy_raw - hy_raw.rolling(60).mean()) / hy_raw.rolling(60).std()

    print("Reconstructing weights...")
    weights = compute_weights(scores_wide, vix_s, hy_z60_s, is_equity, is_defensive, best_params)

    # -----------------------------------------------------------------------
    # Stats
    # -----------------------------------------------------------------------
    total_invested  = weights.sum(axis=1)          # capital deployed each day
    cash_pct        = (1 - total_invested)
    active_positions = (weights > 0.001).sum(axis=1)

    print(f"\n{'='*55}")
    print(f"{'CAPITAL UTILISATION':^55}")
    print(f"{'='*55}")
    print(f"  Capital investi moyen  : {total_invested.mean():.1%}")
    print(f"  Capital investi médian : {total_invested.median():.1%}")
    print(f"  Cash moyen             : {cash_pct.mean():.1%}")
    print(f"  Positions actives moy  : {active_positions.mean():.1f}")
    print(f"  Positions actives max  : {active_positions.max()}")

    # ETF frequency stats
    n_days = len(weights)
    etf_stats = pd.DataFrame({
        "freq_%":       (weights > 0.001).sum() / n_days * 100,
        "avg_weight_%": weights[weights > 0.001].mean() * 100,
        "max_weight_%": weights.max() * 100,
        "section":      pd.Series({e: sections.get(e,"?") for e in all_etfs}),
    }).sort_values("freq_%", ascending=False)

    print(f"\n{'='*55}")
    print(f"{'ETF FREQUENCY (top 20)':^55}")
    print(f"{'='*55}")
    print(f"  {'ETF':<14} {'Freq%':>6}  {'AvgW%':>6}  {'MaxW%':>6}  Section")
    print(f"  {'-'*52}")
    for etf, row in etf_stats.head(20).iterrows():
        print(f"  {etf:<14} {row['freq_%']:6.1f}  {row['avg_weight_%']:6.1f}  {row['max_weight_%']:6.1f}  {row['section']}")

    etf_stats.to_csv(OUTPUTS / "etf_stats.csv")

    # -----------------------------------------------------------------------
    # S&P 500 benchmark
    # -----------------------------------------------------------------------
    spx = None
    for ticker in ["IVV", "CSPX_AS", "QQQ"]:
        p = DATA / f"{ticker}.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            c  = df["close"] if "close" in df.columns else df["adj_close"]
            c.index = pd.to_datetime(c.index).tz_localize(None)
            spx = c.pct_change().rename("spx")
            break

    # Align to backtest period
    start = port_ret.index[0]
    end   = port_ret.index[-1]
    port_aligned = port_ret.loc[start:end]
    spx_aligned  = spx.loc[start:end] if spx is not None else None

    equity_port = (1 + port_aligned).cumprod()
    equity_spx  = (1 + spx_aligned.reindex(port_aligned.index).fillna(0)).cumprod() if spx_aligned is not None else None

    # -----------------------------------------------------------------------
    # Figure: 3 panels
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(14, 14),
                             gridspec_kw={"height_ratios": [3, 1.2, 1.2]})
    fig.suptitle("MyQTM-ETF — Backtest OOS 2002–2026", fontsize=14, fontweight="bold")

    # Panel 1: equity curves
    ax1 = axes[0]
    ax1.plot(equity_port.index, equity_port.values, color="#1f77b4", lw=1.8,
             label=f"MyQTM-ETF  (Sharpe={port_aligned.mean()/port_aligned.std()*252**0.5:.2f})")
    if equity_spx is not None:
        ax1.plot(equity_spx.index, equity_spx.values, color="#d62728", lw=1.2,
                 alpha=0.7, label="S&P 500 (IVV)")
    ax1.set_ylabel("Valeur du portefeuille (base 1)")
    ax1.set_title("Courbe d'équité (OOS uniquement)")
    ax1.legend(fontsize=10)
    ax1.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:.1f}x"))
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor("#f8f8f8")

    # Drawdown shading
    dd = equity_port / equity_port.cummax() - 1
    ax1_twin = ax1.twinx()
    ax1_twin.fill_between(dd.index, dd.values, 0, alpha=0.15, color="red", label="Drawdown")
    ax1_twin.set_ylabel("Drawdown", color="red")
    ax1_twin.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax1_twin.tick_params(axis="y", colors="red")

    # Panel 2: capital invested + cash
    ax2 = axes[1]
    invest_daily = total_invested.reindex(port_aligned.index).ffill()
    ax2.fill_between(invest_daily.index, invest_daily.values,
                     alpha=0.7, color="#1f77b4", label="Capital investi")
    ax2.fill_between(invest_daily.index, invest_daily.values, 1,
                     alpha=0.4, color="#aec7e8", label="Cash")
    ax2.set_ylim(0, 1.05)
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax2.set_ylabel("Allocation")
    ax2.set_title(f"Capital investi (moy={invest_daily.mean():.1%}) vs Cash")
    ax2.legend(fontsize=9, loc="lower left")
    ax2.grid(True, alpha=0.3)
    ax2.set_facecolor("#f8f8f8")

    # Panel 3: top ETF usage heatmap (bar chart of freq)
    ax3 = axes[2]
    top_etfs = etf_stats.head(15)
    colors_map = {
        "geo": "#1f77b4", "sector_us": "#ff7f0e", "thematic": "#2ca02c",
        "bond": "#9467bd", "commodity": "#8c564b", "crypto": "#e377c2",
    }
    bar_colors = [colors_map.get(etf_stats.loc[e, "section"], "#7f7f7f") for e in top_etfs.index]
    bars = ax3.bar(top_etfs.index, top_etfs["freq_%"], color=bar_colors, alpha=0.8)
    ax3.set_ylabel("Fréquence (%)")
    ax3.set_title("Fréquence d'utilisation des ETFs (top 15)")
    ax3.tick_params(axis="x", rotation=30)
    ax3.grid(True, alpha=0.3, axis="y")
    ax3.set_facecolor("#f8f8f8")

    # Legend for sections
    from matplotlib.patches import Patch
    legend_elems = [Patch(facecolor=c, label=s) for s, c in colors_map.items()]
    ax3.legend(handles=legend_elems, fontsize=8, loc="upper right", ncol=3)

    plt.tight_layout()
    out = OUTPUTS / "backtest_analysis.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {out}")
    print(f"Saved → {OUTPUTS/'etf_stats.csv'}")


if __name__ == "__main__":
    main()
