"""
XGBoost Walk-Forward with REALTIME target (no future leakage).

Target: cash when QQQ is >10% below its running high (observable in real-time).
Re-invest when QQQ recovers to <5% below running high.
This is a pure momentum/trend-following signal that requires NO future knowledge.
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

# ── Realtime target ────────────────────────────────────────
# Cash when drawdown from running high > 10%, re-enter when < 5%
DD_EXIT = -0.10
DD_REENTER = -0.05

running_max = np.maximum.accumulate(prices)
drawdown_arr = (prices - running_max) / running_max

# State machine: invest until DD < EXIT, then cash until DD > REENTER
realtime_target = np.ones(n, dtype=int)
invested = True
n_exits = 0
for i in range(n):
    if invested and drawdown_arr[i] < DD_EXIT:
        invested = False
        n_exits += 1
    elif not invested and drawdown_arr[i] > DD_REENTER:
        invested = True
    realtime_target[i] = 1 if invested else 0

n_cash_rt = (realtime_target == 0).sum()
print(f"Realtime target: exit when DD<{DD_EXIT*100:.0f}%, re-enter when DD>{DD_REENTER*100:.0f}%")
print(f"  {n_exits} exits, {n_cash_rt} cash days ({n_cash_rt/n*100:.0f}%)")

# ── Also build oracle target for comparison ────────────────
DRAWDOWN_THRESHOLD = -0.10
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

oracle_target = np.ones(n, dtype=int)
for peak_i, trough_i in crises:
    oracle_target[peak_i + 1: trough_i + 1] = 0

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

df["target_realtime"] = realtime_target
df["target_oracle"] = oracle_target
df = df.dropna()

feature_cols = [c for c in df.columns if not c.startswith("target")]
X = df[feature_cols].values
y_rt = df["target_realtime"].values
y_oracle = df["target_oracle"].values
plot_dates = df.index
N = len(df)

print(f"Features: {len(feature_cols)} cols, {N} rows")
print(f"Realtime cash in filtered data: {(y_rt == 0).sum()} ({(y_rt == 0).sum()/N*100:.0f}%)")

# ── Baseline: pure rule (no ML) ───────────────────────────
# The realtime target IS the rule output — let's compute its equity
qqq_ret_df = qqq.pct_change().fillna(0).loc[df.index].values

rule_eq = np.ones(N)
for i in range(1, N):
    if y_rt[i]:
        rule_eq[i] = rule_eq[i - 1] * (1 + qqq_ret_df[i])
    else:
        rule_eq[i] = rule_eq[i - 1]

# ── Walk-forward XGBoost ──────────────────────────────────
MIN_TRAIN = 504
STEP = 21

xgb_params = dict(
    n_estimators=300, max_depth=4, learning_rate=0.03,
    subsample=0.7, colsample_bytree=0.7, min_child_weight=20,
    reg_alpha=1.0, reg_lambda=5.0, gamma=1.0,
    random_state=42, eval_metric="logloss",
)

wf_pred = np.full(N, -1, dtype=int)
wf_proba = np.full(N, np.nan)
n_models = 0

t = MIN_TRAIN
while t < N:
    test_end = min(t + STEP, N)
    train_idx = np.arange(0, t)
    test_idx = np.arange(t, test_end)

    model = xgb.XGBClassifier(**xgb_params)
    model.fit(X[train_idx], y_rt[train_idx], verbose=False)

    wf_pred[test_idx] = model.predict(X[test_idx])
    wf_proba[test_idx] = model.predict_proba(X[test_idx])[:, 1]

    n_models += 1
    if n_models % 50 == 0:
        print(f"  Step {n_models}: {plot_dates[t].date()} -> {plot_dates[test_end-1].date()}")

    t = test_end

print(f"Walk-forward: {n_models} models trained")

# Filter to prediction period
pred_mask = wf_pred >= 0
wf_dates = plot_dates[pred_mask]
wf_rt = y_rt[pred_mask]
wf_oracle = y_oracle[pred_mask]
wf_p = wf_pred[pred_mask]
wf_prob = wf_proba[pred_mask]
wf_qqq_ret = qqq_ret_df[pred_mask]
N_wf = len(wf_dates)

# ── Metrics ────────────────────────────────────────────────
acc = accuracy_score(wf_rt, wf_p)
f1 = f1_score(wf_rt, wf_p, pos_label=0, zero_division=0)
print(f"\nRealtime target accuracy: {acc*100:.1f}%  F1_cash: {f1*100:.1f}%")

# ── Equity curves (prediction period only) ────────────────
bh_eq = np.cumprod(1 + wf_qqq_ret)

oracle_eq = np.ones(N_wf)
for i in range(1, N_wf):
    if wf_oracle[i]:
        oracle_eq[i] = oracle_eq[i - 1] * (1 + wf_qqq_ret[i])
    else:
        oracle_eq[i] = oracle_eq[i - 1]

rule_eq_wf = np.ones(N_wf)
for i in range(1, N_wf):
    if wf_rt[i]:
        rule_eq_wf[i] = rule_eq_wf[i - 1] * (1 + wf_qqq_ret[i])
    else:
        rule_eq_wf[i] = rule_eq_wf[i - 1]

model_eq = np.ones(N_wf)
for i in range(1, N_wf):
    if wf_p[i]:
        model_eq[i] = model_eq[i - 1] * (1 + wf_qqq_ret[i])
    else:
        model_eq[i] = model_eq[i - 1]

years = (wf_dates[-1] - wf_dates[0]).days / 365.25

def metrics(eq):
    cagr = eq[-1] ** (1 / years) - 1
    dd = ((eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq)).min()
    cash = 0  # placeholder
    return cagr, dd

bh_cagr, bh_dd = metrics(bh_eq)
oracle_cagr, oracle_dd = metrics(oracle_eq)
rule_cagr, rule_dd = metrics(rule_eq_wf)
model_cagr, model_dd = metrics(model_eq)

n_cash_rule = (wf_rt == 0).sum()
n_cash_model = (wf_p == 0).sum()

print(f"\n{'='*70}")
print(f"Periode: {wf_dates[0].date()} -> {wf_dates[-1].date()} ({years:.1f} ans)")
print(f"{'':30s} {'CAGR':>8s} {'Total':>8s} {'MaxDD':>8s} {'Cash':>10s}")
print(f"{'QQQ Buy & Hold':30s} {bh_cagr*100:7.1f}% {bh_eq[-1]:7.1f}x {bh_dd*100:7.1f}%")
print(f"{'Oracle parfait (hindsight)':30s} {oracle_cagr*100:7.1f}% {oracle_eq[-1]:7.1f}x {oracle_dd*100:7.1f}%")
print(f"{'Regle DD pure (no ML)':30s} {rule_cagr*100:7.1f}% {rule_eq_wf[-1]:7.1f}x {rule_dd*100:7.1f}% {n_cash_rule:5d}j ({n_cash_rule/N_wf*100:.0f}%)")
print(f"{'XGBoost walk-forward':30s} {model_cagr*100:7.1f}% {model_eq[-1]:7.1f}x {model_dd*100:7.1f}% {n_cash_model:5d}j ({n_cash_model/N_wf*100:.0f}%)")

# ── Plot ──────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), height_ratios=[3, 1],
                                sharex=True, gridspec_kw={"hspace": 0.08})

ax1.semilogy(wf_dates, bh_eq,
             label=f"QQQ Buy & Hold ({bh_cagr*100:.1f}%)",
             color="tab:blue", linewidth=1.2, alpha=0.5)
ax1.semilogy(wf_dates, oracle_eq,
             label=f"Oracle hindsight ({oracle_cagr*100:.1f}%)",
             color="tab:green", linewidth=1.2, alpha=0.5)
ax1.semilogy(wf_dates, rule_eq_wf,
             label=f"Regle DD pure ({rule_cagr*100:.1f}%)",
             color="tab:orange", linewidth=1.5, linestyle="--")
ax1.semilogy(wf_dates, model_eq,
             label=f"XGBoost WF ({model_cagr*100:.1f}%)",
             color="tab:red", linewidth=2)

# Shade model cash periods
in_cash = False
for i in range(N_wf):
    if wf_p[i] == 0 and not in_cash:
        in_cash = True
        start = i
    elif wf_p[i] == 1 and in_cash:
        in_cash = False
        ax1.axvspan(wf_dates[start], wf_dates[i], alpha=0.10, color="red")
if in_cash:
    ax1.axvspan(wf_dates[start], wf_dates[-1], alpha=0.10, color="red")

ax1.set_ylabel("Equity (log scale)")
ax1.set_title(f"Target realiste (cash quand DD>{abs(DD_EXIT)*100:.0f}%, re-enter quand DD<{abs(DD_REENTER)*100:.0f}%) | Walk-Forward")
ax1.legend(loc="upper left", fontsize=10)
ax1.grid(True, alpha=0.3)

# Panel 2: drawdown + model proba
wf_dd = df["qqq_dd"].values[pred_mask]
ax2.plot(wf_dates, wf_dd * 100, color="tab:blue", linewidth=0.6, alpha=0.6, label="QQQ Drawdown %")
ax2.axhline(DD_EXIT * 100, color="red", linestyle="--", alpha=0.7, linewidth=0.8, label=f"Exit ({DD_EXIT*100:.0f}%)")
ax2.axhline(DD_REENTER * 100, color="green", linestyle="--", alpha=0.7, linewidth=0.8, label=f"Re-enter ({DD_REENTER*100:.0f}%)")
ax2.fill_between(wf_dates, -60, 0, where=(wf_p == 0), alpha=0.15, color="red", label="Model cash")
ax2.set_ylabel("Drawdown (%)")
ax2.set_xlabel("Date")
ax2.legend(loc="lower left", fontsize=8)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(-60, 5)

ax2.xaxis.set_major_locator(mdates.YearLocator(2))
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

plt.tight_layout()
plt.savefig("/home/greg/data_local/code/MyQTMv2/myfiles/oracle_vs_xgb_realtime.png", dpi=150)
plt.show()
print(f"\nSaved: myfiles/oracle_vs_xgb_realtime.png")
