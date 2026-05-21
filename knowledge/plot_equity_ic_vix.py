"""
Diagnostic chart: equity (brut, red, log) + per-step Test IC + VIX coloured by
regime + universe-wide institutional $ flow (Δshares_outstanding × NAV
aggregated across iShares ETFs).
Output: outputs/equity_ic_vix.jpg
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import os
if "--long" in sys.argv:
    os.environ["QTM_MODE"] = "long"
    sys.argv = [a for a in sys.argv if a != "--long"]
from etf import data_subdir, outputs_subdir
from backtest import VIX_SPIKE_MIN, VIX_CALM_THRESHOLD, VIX_CALM_COND_EMA100_SUP_EMA300
from feature_engineering import ISHARES_MAP

DATA     = ROOT / "data"
DATA_OUT = DATA / data_subdir()
OUT      = ROOT / "outputs" / outputs_subdir()

# --- 1) Equity (brut, red) ---
eq = pd.read_csv(OUT / "backtest_equity.csv", parse_dates=["date"]).set_index("date")["equity"]

# --- 2) Per-step Test IC ---
oos = pd.read_parquet(DATA_OUT / "oos_predictions.parquet").reset_index()
oos["date"] = pd.to_datetime(oos["date"])
test = oos[oos["split"] == "test"].dropna(subset=["score", "label"])


def _step_ic(g):
    return g.groupby("date").apply(
        lambda x: x["score"].corr(x["label"]) if len(x) > 1 else np.nan,
        include_groups=False,
    ).mean()


ic_per_step = test.groupby("step").apply(_step_ic, include_groups=False)
step_start = test.groupby("step")["date"].min()
ic_df = pd.DataFrame({"date": step_start, "ic": ic_per_step}).dropna()
ic_df = ic_df.sort_values("date").reset_index(drop=True)

# --- 3) VIX + regime ---
vix_raw = pd.read_parquet(DATA / "fred_vix.parquet").iloc[:, 0]
vix_raw.index = pd.to_datetime(vix_raw.index)
vix_raw = vix_raw.reindex(eq.index, method="ffill").dropna()
ema100 = vix_raw.ewm(span=100).mean()
ema300 = vix_raw.ewm(span=300).mean()
v5 = vix_raw.diff(5)

spike_mask = pd.Series(False, index=vix_raw.index)
for sd in v5[v5 > VIX_SPIKE_MIN].index:
    pos = vix_raw.index.get_loc(sd)
    for off in range(3):
        if pos + off < len(vix_raw):
            spike_mask.iloc[pos + off] = True

calm_cond = ema100 < VIX_CALM_THRESHOLD
if VIX_CALM_COND_EMA100_SUP_EMA300:
    calm_cond = calm_cond & (ema100 < ema300)
calm_mask = calm_cond.fillna(False)

colors = pd.Series("#ff7f0e", index=vix_raw.index)   # orange = model
colors[calm_mask] = "#2ca02c"                          # green = calm
colors[spike_mask] = "#d62728"                         # red = spike

# --- 4) Institutional $ flow per ETF (Δshares_outstanding × NAV)
per_etf_flow = {}
for bourso, ishares_tk in ISHARES_MAP.items():
    p = DATA / f"ishares_{ishares_tk}_hist.parquet"
    if not p.exists():
        continue
    df = pd.read_parquet(p)
    df.index = pd.to_datetime(df.index)
    so = df["shares_outstanding"].astype(float)
    nav = df["nav_per_share"].astype(float)
    per_etf_flow[bourso] = (so.diff() * nav)

flows_df = pd.DataFrame(per_etf_flow).fillna(0)
flows_df = flows_df.reindex(eq.index, method="ffill").fillna(0) / 1e9   # G$

flow_net   = flows_df.sum(axis=1)              # signed net = directional + rotation residue
flow_gross = flows_df.abs().sum(axis=1)        # total churn = directional + rotation
flow_rot   = flow_gross - flow_net.abs()       # pure rotation component
flow_smooth = flow_net.rolling(20, min_periods=5).mean()
flow_rot_smooth = flow_rot.rolling(20, min_periods=5).mean()
flow_per_date = flow_net   # keep name for downstream

# --- Plot ---
fig, axes = plt.subplots(4, 1, figsize=(22, 16), sharex=True,
                         gridspec_kw={"height_ratios": [2.3, 1.1, 1.5, 1.5]})

# Top: equity (red brut, log scale)
ax = axes[0]
ax.plot(eq.index, eq.values, color="#d62728", lw=1.7, label="Equity (brut, IB fees)")
ax.set_yscale("log")
ax.set_ylabel("Equity (× initial capital, log scale)")
ax.set_title(f"Backtest equity (log) vs per-step Test IC vs VIX regime "
             f"— {eq.index[0].date()} → {eq.index[-1].date()}", fontsize=13)
ax.legend(loc="upper left")
ax.grid(alpha=0.3, which="both")
ax.set_facecolor("#f8f8f8")

# Middle: Test IC per step (bars + zero line)
ax = axes[1]
bar_colors = np.where(ic_df["ic"] >= 0, "#1f77b4", "#9467bd")
ax.bar(ic_df["date"], ic_df["ic"], width=15, color=bar_colors, alpha=0.7,
       label="Test IC per WF step")
ax.axhline(0, color="black", lw=0.5)
ax.axhline(ic_df["ic"].mean(), color="#1f77b4", lw=1.2, ls=":",
           label=f"mean = {ic_df['ic'].mean():+.4f}")
ax.set_ylabel("Test IC")
ax.legend(loc="upper left")
ax.grid(alpha=0.3)

# Bottom: VIX coloured by regime
ax = axes[2]
prev_c, seg_start = colors.iloc[0], 0
for i in range(1, len(vix_raw)):
    if colors.iloc[i] != prev_c or i == len(vix_raw) - 1:
        end = i + 1
        seg = slice(seg_start, end)
        ax.fill_between(vix_raw.index[seg], vix_raw.values[seg],
                        alpha=0.18, color=prev_c, linewidth=0)
        ax.plot(vix_raw.index[seg], vix_raw.values[seg],
                color=prev_c, lw=1.1)
        seg_start = i
        prev_c = colors.iloc[i]
ax.plot(ema100.index, ema100.values, color="#444", lw=1.2, alpha=0.7, label="VIX EMA100")
ax.plot(ema300.index, ema300.values, color="#444", lw=1.2, alpha=0.4, ls="--", label="VIX EMA300")
ax.axhline(VIX_CALM_THRESHOLD, color="#2ca02c", lw=0.9, ls=":",
           label=f"calm threshold = {VIX_CALM_THRESHOLD}")
ax.set_xlabel("Date")
ax.set_ylabel("VIX")
# Manual legend for regime colors
from matplotlib.patches import Patch
handles, labels = ax.get_legend_handles_labels()
handles += [Patch(color="#d62728", alpha=0.6, label="Spike (3d cash)"),
            Patch(color="#2ca02c", alpha=0.6, label="Calm (top-3 Sharpe)"),
            Patch(color="#ff7f0e", alpha=0.6, label="Model (XGB)")]
ax.legend(handles=handles, loc="upper left", ncol=2, fontsize=9)
ax.grid(alpha=0.3)

# Panel 4: institutional flow (daily only, no cumulative)
ax = axes[3]
bar_colors_flow = np.where(flow_per_date.values >= 0, "#2ca02c", "#d62728")
ax.bar(flow_per_date.index, flow_per_date.values, width=2.0,
       color=bar_colors_flow, alpha=0.45, label="Daily net flow (G$)")
ax.plot(flow_smooth.index, flow_smooth.values, color="black", lw=1.3,
        label="20d smoothed net (G$)")
ax.plot(flow_rot_smooth.index, flow_rot_smooth.values, color="#1f77b4",
        lw=1.2, ls="--", label="20d smoothed rotation = gross − |net| (G$)")
ax.axhline(0, color="black", lw=0.5)
ax.set_ylabel("Flow (G$/day)")
ax.set_xlabel("Date")
ax.legend(loc="upper left", fontsize=9)
n_etfs_in_flow = (flows_df.abs().sum(axis=0) > 0).sum()
ax.set_title(f"Institutional net $ flow per day (Δshares × NAV aggregated over {n_etfs_in_flow} iShares ETFs) — "
             f"rotation = gross flow − |net flow| isolates inter-ETF churn from directional conviction",
             fontsize=11)
ax.grid(alpha=0.3)

axes[3].xaxis.set_major_locator(mdates.YearLocator())
axes[3].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

plt.tight_layout()
out = OUT / "equity_ic_vix.jpg"
plt.savefig(out, dpi=110, bbox_inches="tight")
print(f"steps={len(ic_df)}, mean test IC = {ic_df['ic'].mean():+.4f}, "
      f"% pos steps = {100*(ic_df['ic']>0).mean():.1f}%")
print(f"Daily net flow range: [{flow_per_date.min():+.2f}, {flow_per_date.max():+.2f}] G$/day")
print(f"Saved → {out}")
