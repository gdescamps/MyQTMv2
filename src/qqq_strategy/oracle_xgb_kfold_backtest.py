"""
XGBoost 10-fold OOS backtest: equity curve with invest/cash periods highlighted.
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

# ── Oracle target ──────────────────────────────────────────
DRAWDOWN_THRESHOLD = -0.10
running_max = np.maximum.accumulate(prices)
drawdown_arr = (prices - running_max) / running_max

crises = []
i = 0
while i < n:
    peak_i = i
    while i < n and drawdown_arr[i] >= 0:
        peak_i = i
        i += 1
    if i >= n:
        break
    trough_i = i
    min_dd = drawdown_arr[i]
    significant = min_dd < DRAWDOWN_THRESHOLD
    while i < n and drawdown_arr[i] < 0:
        if drawdown_arr[i] < min_dd:
            min_dd = drawdown_arr[i]
            trough_i = i
        if drawdown_arr[i] < DRAWDOWN_THRESHOLD:
            significant = True
        i += 1
    if significant:
        crises.append((peak_i, trough_i))

invested_mask = np.ones(n, dtype=int)
for peak_i, trough_i in crises:
    invested_mask[peak_i + 1: trough_i + 1] = 0

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
df["target"] = invested_mask
df = df.dropna()

feature_cols = [c for c in df.columns if c != "target"]
X = df[feature_cols].values
y = df["target"].values
plot_dates = df.index
N = len(df)

# ── 10-fold OOS predictions ───────────────────────────────
K = 10
fold_size = N // K
folds = [np.arange(k * fold_size, (k + 1) * fold_size if k < K - 1 else N) for k in range(K)]

xgb_params = dict(
    n_estimators=300, max_depth=4, learning_rate=0.03,
    subsample=0.7, colsample_bytree=0.7, min_child_weight=20,
    reg_alpha=1.0, reg_lambda=5.0, gamma=1.0,
    random_state=42, eval_metric="logloss",
)

oos_pred = np.zeros(N, dtype=int)
for k in range(K):
    test_idx = folds[k]
    train_idx = np.concatenate([folds[j] for j in range(K) if j != k])
    model = xgb.XGBClassifier(**xgb_params)
    model.fit(X[train_idx], y[train_idx], verbose=False)
    oos_pred[test_idx] = model.predict(X[test_idx])

# ── Equity curve ──────────────────────────────────────────
qqq_ret_df = qqq.pct_change().fillna(0).loc[df.index].values

model_eq = np.ones(N)
for i in range(1, N):
    if oos_pred[i]:
        model_eq[i] = model_eq[i - 1] * (1 + qqq_ret_df[i])
    else:
        model_eq[i] = model_eq[i - 1]

years = (plot_dates[-1] - plot_dates[0]).days / 365.25
model_cagr = model_eq[-1] ** (1 / years) - 1
model_dd = ((model_eq - np.maximum.accumulate(model_eq)) / np.maximum.accumulate(model_eq)).min()
n_cash = (oos_pred == 0).sum()

# ── Detect cash-out periods ──────────────────────────────
cash_periods = []
in_cash = False
for i in range(N):
    if oos_pred[i] == 0 and not in_cash:
        in_cash = True
        start = i
    elif oos_pred[i] == 1 and in_cash:
        in_cash = False
        cash_periods.append((start, i - 1))
if in_cash:
    cash_periods.append((start, N - 1))

# ── Plot ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 7))

# Equity curve colored by state
for i in range(1, N):
    color = "tab:green" if oos_pred[i] else "tab:red"
    ax.semilogy([plot_dates[i - 1], plot_dates[i]],
                [model_eq[i - 1], model_eq[i]],
                color=color, linewidth=1.5)

# Shade cash periods
for s, e in cash_periods:
    ax.axvspan(plot_dates[s], plot_dates[e], alpha=0.12, color="red")

# Legend proxies
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color="tab:green", linewidth=2, label="Investi"),
    Line2D([0], [0], color="tab:red", linewidth=2, label="Cash"),
    plt.Rectangle((0, 0), 1, 1, fc="red", alpha=0.12, label="Periodes cash"),
]

ax.set_ylabel("Equity (log scale)")
ax.set_title(f"XGBoost 10-fold OOS  |  CAGR {model_cagr*100:.1f}%  |  "
             f"DD {model_dd*100:.1f}%  |  Cash {n_cash}j ({n_cash/N*100:.0f}%)")
ax.legend(handles=legend_elements, loc="upper left", fontsize=11)
ax.grid(True, alpha=0.3)

ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.set_xlabel("Date")

plt.tight_layout()
plt.savefig("/home/greg/data_local/code/MyQTMv2/myfiles/xgb_kfold_backtest.png", dpi=150)
plt.show()

# Print cash periods
print(f"\n{len(cash_periods)} periodes cash:")
for s, e in cash_periods:
    days = e - s + 1
    print(f"  {plot_dates[s].date()} -> {plot_dates[e].date()}  ({days}j)")

print(f"\nSaved: myfiles/xgb_kfold_backtest.png")
