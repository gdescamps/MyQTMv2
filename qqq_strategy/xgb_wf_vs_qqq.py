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
qqq_daily_ret = pd.Series(prices, index=dates).pct_change()

# ── Linear features ───────────────────────────────────────
for w in [5, 10, 20, 50, 100, 200]:
    df[f"qqq_sma{w}"] = df["qqq_close"].rolling(w).mean()
    df[f"qqq_ret{w}"] = df["qqq_close"].pct_change(w)
    df[f"qqq_vol{w}"] = qqq_daily_ret.rolling(w).std()
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

# ── Non-linear features ──────────────────────────────────

# 1. Asymmetric volatility (downside vs upside)
for w in [20, 50]:
    down_ret = qqq_daily_ret.clip(upper=0)
    up_ret = qqq_daily_ret.clip(lower=0)
    df[f"qqq_downvol{w}"] = down_ret.rolling(w).std()
    df[f"qqq_upvol{w}"] = up_ret.rolling(w).std()
    df[f"qqq_vol_asym{w}"] = df[f"qqq_downvol{w}"] / (df[f"qqq_upvol{w}"] + 1e-8)

# 2. Skewness & kurtosis of returns
for w in [20, 50]:
    df[f"qqq_skew{w}"] = qqq_daily_ret.rolling(w).skew()
    df[f"qqq_kurt{w}"] = qqq_daily_ret.rolling(w).kurt()

# 3. Acceleration (rate of change of returns)
for w in [5, 10, 20]:
    ret_w = df[f"qqq_ret{w}"]
    df[f"qqq_accel{w}"] = ret_w.diff(w)

# 4. Volatility of volatility (vol clustering)
vol20 = df["qqq_vol20"]
df["volvol_20"] = vol20.rolling(20).std()
df["vol_accel"] = vol20.diff(5)
df["vol_regime"] = vol20 / vol20.rolling(100).mean()

# 5. Drawdown dynamics
df["dd_speed"] = pd.Series(drawdown_arr, index=dates).diff(5)  # how fast DD deepens
df["dd_duration"] = 0.0  # days since last high
dd_dur = np.zeros(n)
cnt = 0
for i in range(n):
    if drawdown_arr[i] >= 0:
        cnt = 0
    else:
        cnt += 1
    dd_dur[i] = cnt
df["dd_duration"] = dd_dur
df["dd_depth_x_duration"] = drawdown_arr * dd_dur  # interaction

# 6. Cross-asset interactions
df["vix_x_spread"] = df["vix"] * df["spread"]
df["vix_x_dd"] = df["vix"] * drawdown_arr
df["vix_x_vol20"] = df["vix"] * df["qqq_vol20"]
df["spread_x_vol20"] = df["spread"] * df["qqq_vol20"]
df["vix_x_ret20"] = df["vix"] * df["qqq_ret20"]

# 7. VIX term structure proxy (short vs long MA = contango/backwardation)
df["vix_term"] = df["vix_sma5"] / df["vix_sma50"]  # >1 = backwardation (fear)
df["spread_term"] = df["spread_sma5"] / df["spread_sma50"]

# 8. Bollinger band position
for w in [20, 50]:
    sma = df[f"qqq_sma{w}"]
    std = df["qqq_close"].rolling(w).std()
    df[f"qqq_bband{w}"] = (df["qqq_close"] - sma) / (std + 1e-8)  # z-score

# 9. Mean reversion signals
df["qqq_ret5_vs_ret50"] = df["qqq_ret5"] - df["qqq_ret50"]  # short vs long momentum
df["qqq_ret10_vs_ret100"] = df["qqq_ret10"] - df["qqq_ret100"]

# 10. Consecutive down days
neg_days = (qqq_daily_ret < 0).astype(int)
consec = np.zeros(n)
cnt = 0
for i in range(n):
    if neg_days.iloc[i]:
        cnt += 1
    else:
        cnt = 0
    consec[i] = cnt
df["consec_down"] = consec

# 11. Max drawdown over rolling windows (worst case recently)
for w in [20, 50]:
    rm = df["qqq_close"].rolling(w).max()
    df[f"qqq_maxdd{w}"] = (df["qqq_close"] - rm) / rm

# 12. VIX spike (single-day jump)
df["vix_spike1d"] = df["vix"].pct_change()
df["vix_spike5d"] = df["vix"].pct_change(5)

# 13. Spread acceleration
df["spread_accel"] = df["spread"].diff(5).diff(5)

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

print(f"Features: {len(feature_cols)} cols, {N} rows")

TEMPERATURE = 3.0  # >1 = smoother probabilities, 1 = original

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

wf_pred = np.full(N, -1, dtype=int)
wf_proba = np.full(N, np.nan)
last_model = None
t = MIN_TRAIN
while t < N:
    test_end = min(t + STEP, N)
    train_idx = np.arange(0, t)
    test_idx = np.arange(t, test_end)
    model = xgb.XGBClassifier(**xgb_params)
    model.fit(X[train_idx], y[train_idx], verbose=False)
    # Raw logits -> temperature scaling -> smooth probability
    import xgboost as xgb_mod
    dtest = xgb_mod.DMatrix(X[test_idx])
    raw_logits = model.get_booster().predict(dtest, output_margin=True)
    smooth_proba = sigmoid(raw_logits / TEMPERATURE)
    wf_proba[test_idx] = smooth_proba
    wf_pred[test_idx] = (smooth_proba >= 0.5).astype(int)
    last_model = model
    t = test_end

print(f"Temperature: {TEMPERATURE}")

# Top features from last model
imp = last_model.feature_importances_
top_idx = np.argsort(imp)[-20:][::-1]
print(f"\nTop 20 features (last model):")
for idx in top_idx:
    print(f"  {feature_cols[idx]:30s} {imp[idx]:.4f}")

pred_mask = wf_pred >= 0
wf_dates = plot_dates[pred_mask]
wf_p = wf_pred[pred_mask]
wf_prob = wf_proba[pred_mask]
N_wf = len(wf_dates)

# ── Equity ────────────────────────────────────────────────
MAX_LEVERAGE = 1.5
qqq_ret_df = qqq.pct_change().fillna(0).loc[df.index].values[pred_mask]

bh_eq = np.cumprod(1 + qqq_ret_df)

# Binary x1
model_eq = np.ones(N_wf)
for i in range(1, N_wf):
    if wf_p[i]:
        model_eq[i] = model_eq[i - 1] * (1 + qqq_ret_df[i])
    else:
        model_eq[i] = model_eq[i - 1]

# Continuous allocation: P(invested) mapped to [0%, 150%]
# P=1.0 -> 150%, P=0.5 -> 0%, P=0.0 -> 0%
# Linear ramp from threshold to 1.0
PROB_CASH = 0.5    # below this -> 0% allocation
PROB_FULL = 0.85   # above this -> 150% allocation
alloc = np.clip((wf_prob - PROB_CASH) / (PROB_FULL - PROB_CASH), 0, 1) * MAX_LEVERAGE

cont_eq = np.ones(N_wf)
for i in range(1, N_wf):
    cont_eq[i] = cont_eq[i - 1] * (1 + qqq_ret_df[i] * alloc[i])

years = (wf_dates[-1] - wf_dates[0]).days / 365.25
bh_cagr = bh_eq[-1] ** (1 / years) - 1
model_cagr = model_eq[-1] ** (1 / years) - 1
cont_cagr = cont_eq[-1] ** (1 / years) - 1
bh_dd = ((bh_eq - np.maximum.accumulate(bh_eq)) / np.maximum.accumulate(bh_eq)).min()
model_dd = ((model_eq - np.maximum.accumulate(model_eq)) / np.maximum.accumulate(model_eq)).min()
cont_dd = ((cont_eq - np.maximum.accumulate(cont_eq)) / np.maximum.accumulate(cont_eq)).min()
n_cash = (wf_p == 0).sum()

print(f"\n{'='*70}")
print(f"Periode: {wf_dates[0].date()} -> {wf_dates[-1].date()} ({years:.1f} ans)")
print(f"{'':35s} {'CAGR':>8s} {'Total':>8s} {'MaxDD':>8s}")
print(f"{'QQQ Buy & Hold':35s} {bh_cagr*100:7.1f}% {bh_eq[-1]:7.1f}x {bh_dd*100:7.1f}%")
print(f"{'XGBoost WF binary x1':35s} {model_cagr*100:7.1f}% {model_eq[-1]:7.1f}x {model_dd*100:7.1f}%")
print(f"{'XGBoost WF continuous 0-150%':35s} {cont_cagr*100:7.1f}% {cont_eq[-1]:7.1f}x {cont_dd*100:7.1f}%")
print(f"Alloc mean: {alloc.mean()*100:.0f}%  median: {np.median(alloc)*100:.0f}%  "
      f"at 0%: {(alloc==0).sum()}j  at 150%: {(alloc>=MAX_LEVERAGE-0.01).sum()}j")

# ── Plot ──────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), height_ratios=[3, 1],
                                sharex=True, gridspec_kw={"hspace": 0.08})

ax1.semilogy(wf_dates, bh_eq,
             label=f"QQQ Buy & Hold (CAGR {bh_cagr*100:.1f}%, DD {bh_dd*100:.1f}%)",
             color="tab:blue", linewidth=1.5, alpha=0.7)
ax1.semilogy(wf_dates, model_eq,
             label=f"XGBoost binary x1 (CAGR {model_cagr*100:.1f}%, DD {model_dd*100:.1f}%)",
             color="tab:green", linewidth=1.2, alpha=0.5)
ax1.semilogy(wf_dates, cont_eq,
             label=f"XGBoost continu 0-{MAX_LEVERAGE*100:.0f}% (CAGR {cont_cagr*100:.1f}%, DD {cont_dd*100:.1f}%)",
             color="tab:red", linewidth=2)

ax1.set_ylabel("Equity (log scale)")
ax1.set_title("XGBoost Walk-Forward: allocation continue 0-150% selon P(invested)")
ax1.legend(loc="upper left", fontsize=10)
ax1.grid(True, alpha=0.3)

# Panel 2: allocation level
ax2.fill_between(wf_dates, 0, alloc * 100, alpha=0.4, color="tab:green", label="Allocation %")
ax2.axhline(100, color="gray", linestyle="--", alpha=0.5, linewidth=0.8, label="100% (no leverage)")
ax2.axhline(150, color="red", linestyle="--", alpha=0.5, linewidth=0.8, label="150% (max leverage)")
ax2.set_ylabel("Allocation (%)")
ax2.set_xlabel("Date")
ax2.set_ylim(-5, 165)
ax2.legend(loc="lower right", fontsize=9)
ax2.grid(True, alpha=0.3)

ax2.xaxis.set_major_locator(mdates.YearLocator(2))
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

plt.tight_layout()
plt.savefig("/home/greg/data_local/code/MyQTMv2/myfiles/xgb_wf_vs_qqq.png", dpi=150)
plt.show()
print(f"\nSaved: myfiles/xgb_wf_vs_qqq.png")
