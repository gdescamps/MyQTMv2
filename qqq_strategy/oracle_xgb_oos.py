"""
Oracle vs XGBoost Out-of-Sample (dual A/B split).

Split dataset in 2 halves:
  - Model A: train on 1st half, predict on 2nd half
  - Model B: train on 2nd half, predict on 1st half
Combined OOS prediction covers 100% of the data.
XGBoost is regularized to avoid overfitting.
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

# ── Build Oracle target ────────────────────────────────────
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

# ── Feature engineering ────────────────────────────────────
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

print(f"Oracle: {len(crises)} crises, {(y == 0).sum()} cash days ({(y == 0).sum()/len(y)*100:.0f}%)")
print(f"Features: {len(feature_cols)} cols, {len(df)} rows")

# ── Split A/B ──────────────────────────────────────────────
mid = len(df) // 2
idx_A = np.arange(0, mid)       # 1st half
idx_B = np.arange(mid, len(df)) # 2nd half

mid_date = plot_dates[mid]
print(f"\nSplit: {plot_dates[0].date()} -- [{mid_date.date()}] -- {plot_dates[-1].date()}")
print(f"  Half A: {len(idx_A)} rows  (cash: {(y[idx_A]==0).sum()})")
print(f"  Half B: {len(idx_B)} rows  (cash: {(y[idx_B]==0).sum()})")

# ── XGBoost regularized ───────────────────────────────────
xgb_params = dict(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.7,
    colsample_bytree=0.7,
    min_child_weight=20,
    reg_alpha=1.0,
    reg_lambda=5.0,
    scale_pos_weight=1.0,
    gamma=1.0,
    random_state=42,
    eval_metric="logloss",
)

print(f"\nXGBoost params (regularized):")
for k, v in xgb_params.items():
    print(f"  {k:25s} = {v}")

# Model A: train 1st half, test 2nd half
model_A = xgb.XGBClassifier(**xgb_params)
model_A.fit(X[idx_A], y[idx_A], verbose=False)
pred_B = model_A.predict(X[idx_B])
proba_B = model_A.predict_proba(X[idx_B])[:, 1]

# Model B: train 2nd half, test 1st half
model_B = xgb.XGBClassifier(**xgb_params)
model_B.fit(X[idx_B], y[idx_B], verbose=False)
pred_A = model_B.predict(X[idx_A])
proba_A = model_B.predict_proba(X[idx_A])[:, 1]

# Combine OOS predictions
oos_pred = np.zeros(len(df), dtype=int)
oos_proba = np.zeros(len(df))
oos_pred[idx_A] = pred_A
oos_pred[idx_B] = pred_B
oos_proba[idx_A] = proba_A
oos_proba[idx_B] = proba_B

# ── Metrics ────────────────────────────────────────────────
acc = accuracy_score(y, oos_pred)
f1 = f1_score(y, oos_pred, pos_label=0)  # F1 for cash class (minority)
cm = confusion_matrix(y, oos_pred)

print(f"\n{'='*60}")
print(f"OOS Combined Accuracy: {acc*100:.1f}%")
print(f"OOS F1 (cash class):   {f1*100:.1f}%")
print(f"Confusion matrix (rows=actual, cols=pred):")
print(f"             pred_inv  pred_cash")
print(f"  actual_inv   {cm[1,1]:5d}     {cm[1,0]:5d}")
print(f"  actual_cash  {cm[0,1]:5d}     {cm[0,0]:5d}")

# Per-half metrics
for name, idx, pred in [("A (train B, test A)", idx_A, pred_A),
                          ("B (train A, test B)", idx_B, pred_B)]:
    a = accuracy_score(y[idx], pred)
    f = f1_score(y[idx], pred, pos_label=0)
    print(f"  {name}: acc={a*100:.1f}%  F1_cash={f*100:.1f}%")

# ── Build equity curves ───────────────────────────────────
qqq_ret_df = qqq.pct_change().fillna(0).loc[df.index].values

bh_equity = np.cumprod(1 + qqq_ret_df)

oracle_eq = np.ones(len(df))
for i in range(1, len(df)):
    if y[i]:
        oracle_eq[i] = oracle_eq[i - 1] * (1 + qqq_ret_df[i])
    else:
        oracle_eq[i] = oracle_eq[i - 1]

model_eq = np.ones(len(df))
for i in range(1, len(df)):
    if oos_pred[i]:
        model_eq[i] = model_eq[i - 1] * (1 + qqq_ret_df[i])
    else:
        model_eq[i] = model_eq[i - 1]

years = (plot_dates[-1] - plot_dates[0]).days / 365.25
bh_cagr = (bh_equity[-1]) ** (1 / years) - 1
oracle_cagr = (oracle_eq[-1]) ** (1 / years) - 1
model_cagr = (model_eq[-1]) ** (1 / years) - 1

bh_dd = ((bh_equity - np.maximum.accumulate(bh_equity)) / np.maximum.accumulate(bh_equity)).min()
oracle_dd = ((oracle_eq - np.maximum.accumulate(oracle_eq)) / np.maximum.accumulate(oracle_eq)).min()
model_dd = ((model_eq - np.maximum.accumulate(model_eq)) / np.maximum.accumulate(model_eq)).min()

n_cash_model = (oos_pred == 0).sum()

print(f"\nPerformance:")
print(f"QQQ Buy & Hold:     CAGR {bh_cagr*100:.1f}%  x{bh_equity[-1]:.1f}  DD {bh_dd*100:.1f}%")
print(f"Oracle parfait:     CAGR {oracle_cagr*100:.1f}%  x{oracle_eq[-1]:.1f}  DD {oracle_dd*100:.1f}%")
print(f"XGBoost OOS (A/B):  CAGR {model_cagr*100:.1f}%  x{model_eq[-1]:.1f}  DD {model_dd*100:.1f}%  "
      f"cash={n_cash_model}j ({n_cash_model/len(df)*100:.0f}%)")

# ── Plot ───────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), height_ratios=[3, 1],
                                sharex=True, gridspec_kw={"hspace": 0.08})

ax1.semilogy(plot_dates, bh_equity,
             label=f"QQQ Buy & Hold (CAGR {bh_cagr*100:.1f}%)",
             color="tab:blue", linewidth=1.2, alpha=0.7)
ax1.semilogy(plot_dates, oracle_eq,
             label=f"Oracle parfait (CAGR {oracle_cagr*100:.1f}%)",
             color="tab:green", linewidth=2)
ax1.semilogy(plot_dates, model_eq,
             label=f"XGBoost OOS A/B (CAGR {model_cagr*100:.1f}%)",
             color="tab:red", linewidth=1.5)

# Split line
ax1.axvline(mid_date, color="gray", linestyle=":", alpha=0.6, linewidth=1)
ax2.axvline(mid_date, color="gray", linestyle=":", alpha=0.6, linewidth=1)
ax1.text(mid_date, bh_equity.max() * 0.5, "  A|B split", fontsize=9, color="gray")

# Shade oracle cash periods
for peak_i_orig, trough_i_orig in crises:
    if peak_i_orig < len(dates) and trough_i_orig < len(dates):
        ax1.axvspan(dates[peak_i_orig], dates[trough_i_orig], alpha=0.08, color="gray")

ax1.set_ylabel("Equity (log scale)")
ax1.set_title("Oracle parfait vs XGBoost OOS (dual A/B, regularized)")
ax1.legend(loc="upper left", fontsize=11)
ax1.grid(True, alpha=0.3)

# Panel 2: OOS probability + oracle cash zones
ax2.plot(plot_dates, oos_proba, color="tab:purple", linewidth=0.6, alpha=0.8, label="P(invested) OOS")
ax2.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
ax2.fill_between(plot_dates, 0, 1, where=(y == 0), alpha=0.15, color="red", label="Oracle cash")
ax2.fill_between(plot_dates, 0, 1, where=(oos_pred == 0), alpha=0.15, color="blue", label="Model cash")
ax2.set_ylabel("P(invested)")
ax2.set_xlabel("Date")
ax2.legend(loc="upper right", fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(-0.05, 1.05)

ax2.xaxis.set_major_locator(mdates.YearLocator(2))
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

plt.tight_layout()
plt.savefig("/home/greg/data_local/code/MyQTMv2/myfiles/oracle_vs_xgb_oos.png", dpi=150)
plt.show()
print("\nSaved: myfiles/oracle_vs_xgb_oos.png")
