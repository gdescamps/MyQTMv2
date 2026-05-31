"""
XGBoost Walk-Forward (realistic): predict invest/cash using only past data.

- Minimum 2 years of training data before first prediction
- Retrain every 21 days (1 month) on all available past data
- Predict next 21 days, then retrain with new data
- No future leakage
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

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

print(f"Oracle: {len(crises)} crises, {(y == 0).sum()} cash days ({(y == 0).sum()/N*100:.0f}%)")
print(f"Features: {len(feature_cols)} cols, {N} rows")

# ── Walk-forward parameters ────────────────────────────────
MIN_TRAIN = 504  # ~2 years of trading days
STEP = 21        # retrain every 21 days

xgb_params = dict(
    n_estimators=300, max_depth=4, learning_rate=0.03,
    subsample=0.7, colsample_bytree=0.7, min_child_weight=20,
    reg_alpha=1.0, reg_lambda=5.0, gamma=1.0,
    random_state=42, eval_metric="logloss",
)

# ── Walk-forward loop ─────────────────────────────────────
wf_pred = np.full(N, -1, dtype=int)  # -1 = no prediction (warmup)
wf_proba = np.full(N, np.nan)

n_models = 0
t = MIN_TRAIN
while t < N:
    test_end = min(t + STEP, N)
    train_idx = np.arange(0, t)
    test_idx = np.arange(t, test_end)

    model = xgb.XGBClassifier(**xgb_params)
    model.fit(X[train_idx], y[train_idx], verbose=False)

    wf_pred[test_idx] = model.predict(X[test_idx])
    wf_proba[test_idx] = model.predict_proba(X[test_idx])[:, 1]

    n_models += 1
    if n_models % 20 == 0 or test_end == N:
        print(f"  Step {n_models}: train[0:{t}] -> test[{t}:{test_end}]  "
              f"({plot_dates[t].date()} -> {plot_dates[test_end-1].date()})")

    t = test_end

# Filter to predicted region only
pred_mask = wf_pred >= 0
wf_dates = plot_dates[pred_mask]
wf_y = y[pred_mask]
wf_p = wf_pred[pred_mask]
wf_prob = wf_proba[pred_mask]

print(f"\nWalk-forward: {n_models} models trained")
print(f"Prediction period: {wf_dates[0].date()} -> {wf_dates[-1].date()}")

# ── Metrics ────────────────────────────────────────────────
acc = accuracy_score(wf_y, wf_p)
f1 = f1_score(wf_y, wf_p, pos_label=0, zero_division=0)
cm = confusion_matrix(wf_y, wf_p)

print(f"\nOOS Walk-Forward Accuracy: {acc*100:.1f}%")
print(f"OOS F1 (cash class):      {f1*100:.1f}%")
print(f"Confusion matrix:")
print(f"             pred_inv  pred_cash")
print(f"  actual_inv   {cm[1,1]:5d}     {cm[1,0]:5d}")
print(f"  actual_cash  {cm[0,1]:5d}     {cm[0,0]:5d}")

# ── Equity curves ─────────────────────────────────────────
qqq_ret_all = qqq.pct_change().fillna(0).loc[df.index].values
# Only on prediction period
pred_indices = np.where(pred_mask)[0]
qqq_ret_wf = qqq_ret_all[pred_indices]

bh_eq = np.cumprod(1 + qqq_ret_wf)

oracle_eq = np.ones(len(pred_indices))
for i in range(1, len(pred_indices)):
    if wf_y[i]:
        oracle_eq[i] = oracle_eq[i - 1] * (1 + qqq_ret_wf[i])
    else:
        oracle_eq[i] = oracle_eq[i - 1]

model_eq = np.ones(len(pred_indices))
for i in range(1, len(pred_indices)):
    if wf_p[i]:
        model_eq[i] = model_eq[i - 1] * (1 + qqq_ret_wf[i])
    else:
        model_eq[i] = model_eq[i - 1]

years = (wf_dates[-1] - wf_dates[0]).days / 365.25
bh_cagr = bh_eq[-1] ** (1 / years) - 1
oracle_cagr = oracle_eq[-1] ** (1 / years) - 1
model_cagr = model_eq[-1] ** (1 / years) - 1

bh_dd = ((bh_eq - np.maximum.accumulate(bh_eq)) / np.maximum.accumulate(bh_eq)).min()
oracle_dd = ((oracle_eq - np.maximum.accumulate(oracle_eq)) / np.maximum.accumulate(oracle_eq)).min()
model_dd = ((model_eq - np.maximum.accumulate(model_eq)) / np.maximum.accumulate(model_eq)).min()

n_cash = (wf_p == 0).sum()

print(f"\n{'='*60}")
print(f"Periode: {wf_dates[0].date()} -> {wf_dates[-1].date()} ({years:.1f} ans)")
print(f"QQQ Buy & Hold:       CAGR {bh_cagr*100:.1f}%  x{bh_eq[-1]:.1f}  DD {bh_dd*100:.1f}%")
print(f"Oracle parfait:       CAGR {oracle_cagr*100:.1f}%  x{oracle_eq[-1]:.1f}  DD {oracle_dd*100:.1f}%")
print(f"XGBoost walk-forward: CAGR {model_cagr*100:.1f}%  x{model_eq[-1]:.1f}  DD {model_dd*100:.1f}%  "
      f"cash={n_cash}j ({n_cash/len(wf_p)*100:.0f}%)")

# ── Plot ──────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), height_ratios=[3, 1],
                                sharex=True, gridspec_kw={"hspace": 0.08})

ax1.semilogy(wf_dates, bh_eq,
             label=f"QQQ Buy & Hold (CAGR {bh_cagr*100:.1f}%)",
             color="tab:blue", linewidth=1.2, alpha=0.6)
ax1.semilogy(wf_dates, oracle_eq,
             label=f"Oracle parfait (CAGR {oracle_cagr*100:.1f}%)",
             color="tab:green", linewidth=1.5, alpha=0.6)

# Model equity colored by state
for i in range(1, len(wf_dates)):
    color = "tab:red" if wf_p[i] else "gray"
    lw = 1.5 if wf_p[i] else 2.5
    ax1.semilogy([wf_dates[i - 1], wf_dates[i]],
                [model_eq[i - 1], model_eq[i]],
                color=color, linewidth=lw)

# Shade model cash periods
in_cash = False
for i in range(len(wf_p)):
    if wf_p[i] == 0 and not in_cash:
        in_cash = True
        start = i
    elif wf_p[i] == 1 and in_cash:
        in_cash = False
        ax1.axvspan(wf_dates[start], wf_dates[i], alpha=0.10, color="orange")
if in_cash:
    ax1.axvspan(wf_dates[start], wf_dates[-1], alpha=0.10, color="orange")

from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color="tab:blue", linewidth=1.5, alpha=0.6, label=f"QQQ B&H (CAGR {bh_cagr*100:.1f}%)"),
    Line2D([0], [0], color="tab:green", linewidth=1.5, alpha=0.6, label=f"Oracle (CAGR {oracle_cagr*100:.1f}%)"),
    Line2D([0], [0], color="tab:red", linewidth=2, label=f"WF investi (CAGR {model_cagr*100:.1f}%)"),
    Line2D([0], [0], color="gray", linewidth=2.5, label="WF cash"),
    plt.Rectangle((0, 0), 1, 1, fc="orange", alpha=0.15, label="Periodes cash"),
]

ax1.set_ylabel("Equity (log scale)")
ax1.set_title(f"XGBoost Walk-Forward (realiste) | retrain tous les {STEP}j | min train {MIN_TRAIN}j")
ax1.legend(handles=legend_elements, loc="upper left", fontsize=10)
ax1.grid(True, alpha=0.3)

# Panel 2: probability
ax2.plot(wf_dates, wf_prob, color="tab:purple", linewidth=0.6, alpha=0.8, label="P(invested)")
ax2.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
ax2.fill_between(wf_dates, 0, 1, where=(wf_y == 0), alpha=0.15, color="red", label="Oracle cash")
ax2.fill_between(wf_dates, 0, 1, where=(wf_p == 0), alpha=0.15, color="blue", label="Model cash")
ax2.set_ylabel("P(invested)")
ax2.set_xlabel("Date")
ax2.legend(loc="upper right", fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(-0.05, 1.05)

ax2.xaxis.set_major_locator(mdates.YearLocator(2))
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

plt.tight_layout()
plt.savefig("/home/greg/data_local/code/MyQTMv2/myfiles/oracle_vs_xgb_walkforward.png", dpi=150)
plt.show()
print(f"\nSaved: myfiles/oracle_vs_xgb_walkforward.png")
