"""
XGBoost Walk-Forward vs QQQ Buy & Hold only.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ── Load data ──────────────────────────────────────────────
qqq = pd.read_parquet("/home/greg/data_local/code/MyQTMv2/data/QQQ.parquet")["close"]
vix = pd.read_parquet("/home/greg/data_local/code/MyQTMv2/data/vix_ohlc.parquet")["close"]
spread = pd.read_parquet("/home/greg/data_local/code/MyQTMv2/data/fred_baa_spread.parquet")["baa_spread"]

qqq = qqq.loc["2006-01-01":"2026-05-31"].dropna()
vix = vix.loc["2006-01-01":"2026-05-31"].dropna()
spread = spread.loc["2006-01-01":"2026-05-31"].dropna()

common = qqq.index.intersection(vix.index).intersection(spread.index)
qqq = qqq.loc[common]
vix = vix.loc[common]
spread = spread.loc[common]

prices = qqq.values
dates = qqq.index
n = len(prices)

# ── Realtime target ────────────────────────────────────────
DD_EXIT = -0.10
DD_REENTER = -0.05
running_max = np.maximum.accumulate(prices)
drawdown_arr = (prices - running_max) / running_max

realtime_target = np.ones(n, dtype=int)
invested = True
for i in range(n):
    if invested and drawdown_arr[i] < DD_EXIT:
        invested = False
    elif not invested and drawdown_arr[i] > DD_REENTER:
        invested = True
    realtime_target[i] = 1 if invested else 0

# ── Features ───────────────────────────────────────────────
df = pd.DataFrame(index=dates)
df["qqq_close"] = prices
df["vix"] = vix.values
df["spread"] = spread.values

for w in [5, 10, 20, 50, 100, 200]:
    df[f"qqq_sma{w}"] = df["qqq_close"].rolling(w).mean()
    df[f"qqq_ret{w}"] = df["qqq_close"].pct_change(w)
    df[f"qqq_vol{w}"] = df["qqq_close"].pct_change().rolling(w).std()
for w in [20, 50, 100, 200]:
    df[f"qqq_vs_sma{w}"] = df["qqq_close"] / df[f"qqq_sma{w}"] - 1
df["qqq_dd"] = drawdown_arr
delta = df["qqq_close"].diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = (-delta.clip(upper=0)).rolling(14).mean()
df["qqq_rsi"] = 100 - 100 / (1 + gain / loss)
for w in [5, 10, 20, 50]:
    df[f"vix_sma{w}"] = df["vix"].rolling(w).mean()
    df[f"vix_ret{w}"] = df["vix"].pct_change(w)
df["vix_vs_sma20"] = df["vix"] / df["vix_sma20"] - 1
df["vix_vs_sma50"] = df["vix"] / df["vix_sma50"] - 1
for w in [5, 10, 20, 50]:
    df[f"spread_sma{w}"] = df["spread"].rolling(w).mean()
    df[f"spread_ret{w}"] = df["spread"].pct_change(w)
df["spread_vs_sma20"] = df["spread"] / df["spread_sma20"] - 1
df["spread_vs_sma50"] = df["spread"] / df["spread_sma50"] - 1

df["target"] = realtime_target
df = df.dropna()

feature_cols = [c for c in df.columns if c != "target"]
X = df[feature_cols].values
y = df["target"].values
plot_dates = df.index
N = len(df)

# ── Walk-forward ──────────────────────────────────────────
MIN_TRAIN = 504
STEP = 21

xgb_params = dict(
    n_estimators=300, max_depth=4, learning_rate=0.03,
    subsample=0.7, colsample_bytree=0.7, min_child_weight=20,
    reg_alpha=1.0, reg_lambda=5.0, gamma=1.0,
    random_state=42, eval_metric="logloss",
)

wf_pred = np.full(N, -1, dtype=int)
t = MIN_TRAIN
while t < N:
    test_end = min(t + STEP, N)
    train_idx = np.arange(0, t)
    test_idx = np.arange(t, test_end)
    model = xgb.XGBClassifier(**xgb_params)
    model.fit(X[train_idx], y[train_idx], verbose=False)
    wf_pred[test_idx] = model.predict(X[test_idx])
    t = test_end

pred_mask = wf_pred >= 0
wf_dates = plot_dates[pred_mask]
wf_p = wf_pred[pred_mask]
N_wf = len(wf_dates)

# ── Equity ────────────────────────────────────────────────
LEVERAGE = 1.5
qqq_ret_df = qqq.pct_change().fillna(0).loc[df.index].values[pred_mask]

bh_eq = np.cumprod(1 + qqq_ret_df)

model_eq = np.ones(N_wf)
model_lev_eq = np.ones(N_wf)
for i in range(1, N_wf):
    if wf_p[i]:
        model_eq[i] = model_eq[i - 1] * (1 + qqq_ret_df[i])
        model_lev_eq[i] = model_lev_eq[i - 1] * (1 + qqq_ret_df[i] * LEVERAGE)
    else:
        model_eq[i] = model_eq[i - 1]
        model_lev_eq[i] = model_lev_eq[i - 1]

years = (wf_dates[-1] - wf_dates[0]).days / 365.25
bh_cagr = bh_eq[-1] ** (1 / years) - 1
model_cagr = model_eq[-1] ** (1 / years) - 1
model_lev_cagr = model_lev_eq[-1] ** (1 / years) - 1
bh_dd = ((bh_eq - np.maximum.accumulate(bh_eq)) / np.maximum.accumulate(bh_eq)).min()
model_dd = ((model_eq - np.maximum.accumulate(model_eq)) / np.maximum.accumulate(model_eq)).min()
model_lev_dd = ((model_lev_eq - np.maximum.accumulate(model_lev_eq)) / np.maximum.accumulate(model_lev_eq)).min()
n_cash = (wf_p == 0).sum()

# ── Plot ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 7))

ax.semilogy(wf_dates, bh_eq,
            label=f"QQQ Buy & Hold (CAGR {bh_cagr*100:.1f}%, DD {bh_dd*100:.1f}%)",
            color="tab:blue", linewidth=1.5)

# Model x1
ax.semilogy(wf_dates, model_eq, color="tab:green", linewidth=1.2, alpha=0.5)

# Model x1.5 colored by state
for i in range(1, N_wf):
    color = "tab:green" if wf_p[i] else "tab:red"
    ax.semilogy([wf_dates[i - 1], wf_dates[i]],
                [model_lev_eq[i - 1], model_lev_eq[i]],
                color=color, linewidth=1.8)

# Shade cash
in_cash = False
for i in range(N_wf):
    if wf_p[i] == 0 and not in_cash:
        in_cash = True
        start = i
    elif wf_p[i] == 1 and in_cash:
        in_cash = False
        ax.axvspan(wf_dates[start], wf_dates[i], alpha=0.10, color="red")
if in_cash:
    ax.axvspan(wf_dates[start], wf_dates[-1], alpha=0.10, color="red")

from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color="tab:blue", linewidth=2,
           label=f"QQQ Buy & Hold (CAGR {bh_cagr*100:.1f}%, DD {bh_dd*100:.1f}%)"),
    Line2D([0], [0], color="tab:green", linewidth=2, alpha=0.5,
           label=f"XGBoost WF x1 (CAGR {model_cagr*100:.1f}%, DD {model_dd*100:.1f}%)"),
    Line2D([0], [0], color="tab:green", linewidth=2,
           label=f"XGBoost WF x{LEVERAGE} (CAGR {model_lev_cagr*100:.1f}%, DD {model_lev_dd*100:.1f}%)"),
    Line2D([0], [0], color="tab:red", linewidth=2, label="Cash"),
    plt.Rectangle((0, 0), 1, 1, fc="red", alpha=0.12, label=f"Periodes cash ({n_cash}j, {n_cash/N_wf*100:.0f}%)"),
]

ax.set_ylabel("Equity (log scale)")
ax.set_title("XGBoost Walk-Forward vs QQQ Buy & Hold")
ax.legend(handles=legend_elements, loc="upper left", fontsize=11)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.set_xlabel("Date")

plt.tight_layout()
plt.savefig("/home/greg/data_local/code/MyQTMv2/myfiles/xgb_wf_vs_qqq.png", dpi=150)
plt.show()
print(f"Saved: myfiles/xgb_wf_vs_qqq.png")
