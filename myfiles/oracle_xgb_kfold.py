"""
Oracle vs XGBoost Out-of-Sample (K-fold temporal).

Split dataset into K blocks. For each block, train on all OTHER blocks,
predict on the held-out block. Every sample gets an OOS prediction.
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
N = len(df)

print(f"Oracle: {len(crises)} crises, {(y == 0).sum()} cash days ({(y == 0).sum()/N*100:.0f}%)")
print(f"Features: {len(feature_cols)} cols, {N} rows")

# ── K-Fold temporal ────────────────────────────────────────
K = 10
fold_size = N // K
folds = []
for k in range(K):
    start = k * fold_size
    end = (k + 1) * fold_size if k < K - 1 else N
    folds.append(np.arange(start, end))

print(f"\n{K}-Fold temporal split:")
for k, idx in enumerate(folds):
    d0, d1 = plot_dates[idx[0]].date(), plot_dates[idx[-1]].date()
    n_cash = (y[idx] == 0).sum()
    print(f"  Fold {k}: {d0} -> {d1}  ({len(idx)} rows, {n_cash} cash)")

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
    gamma=1.0,
    random_state=42,
    eval_metric="logloss",
)

# Train K models, each trained on K-1 folds, predict on held-out fold
oos_pred = np.zeros(N, dtype=int)
oos_proba = np.zeros(N)

for k in range(K):
    test_idx = folds[k]
    train_idx = np.concatenate([folds[j] for j in range(K) if j != k])

    model = xgb.XGBClassifier(**xgb_params)
    model.fit(X[train_idx], y[train_idx], verbose=False)

    oos_pred[test_idx] = model.predict(X[test_idx])
    oos_proba[test_idx] = model.predict_proba(X[test_idx])[:, 1]

    acc_k = accuracy_score(y[test_idx], oos_pred[test_idx])
    f1_k = f1_score(y[test_idx], oos_pred[test_idx], pos_label=0, zero_division=0)
    print(f"  Fold {k} OOS: acc={acc_k*100:.1f}%  F1_cash={f1_k*100:.1f}%")

# ── Metrics ────────────────────────────────────────────────
acc = accuracy_score(y, oos_pred)
f1 = f1_score(y, oos_pred, pos_label=0)
cm = confusion_matrix(y, oos_pred)

print(f"\n{'='*60}")
print(f"OOS Combined ({K}-fold) Accuracy: {acc*100:.1f}%")
print(f"OOS F1 (cash class):              {f1*100:.1f}%")
print(f"Confusion matrix (rows=actual, cols=pred):")
print(f"             pred_inv  pred_cash")
print(f"  actual_inv   {cm[1,1]:5d}     {cm[1,0]:5d}")
print(f"  actual_cash  {cm[0,1]:5d}     {cm[0,0]:5d}")

# ── Build equity curves ───────────────────────────────────
qqq_ret_df = qqq.pct_change().fillna(0).loc[df.index].values

bh_equity = np.cumprod(1 + qqq_ret_df)

oracle_eq = np.ones(N)
for i in range(1, N):
    if y[i]:
        oracle_eq[i] = oracle_eq[i - 1] * (1 + qqq_ret_df[i])
    else:
        oracle_eq[i] = oracle_eq[i - 1]

model_eq = np.ones(N)
for i in range(1, N):
    if oos_pred[i]:
        model_eq[i] = model_eq[i - 1] * (1 + qqq_ret_df[i])
    else:
        model_eq[i] = model_eq[i - 1]

years = (plot_dates[-1] - plot_dates[0]).days / 365.25
bh_cagr = bh_equity[-1] ** (1 / years) - 1
oracle_cagr = oracle_eq[-1] ** (1 / years) - 1
model_cagr = model_eq[-1] ** (1 / years) - 1

bh_dd = ((bh_equity - np.maximum.accumulate(bh_equity)) / np.maximum.accumulate(bh_equity)).min()
oracle_dd = ((oracle_eq - np.maximum.accumulate(oracle_eq)) / np.maximum.accumulate(oracle_eq)).min()
model_dd = ((model_eq - np.maximum.accumulate(model_eq)) / np.maximum.accumulate(model_eq)).min()

n_cash_model = (oos_pred == 0).sum()

print(f"\nPerformance:")
print(f"QQQ Buy & Hold:      CAGR {bh_cagr*100:.1f}%  x{bh_equity[-1]:.1f}  DD {bh_dd*100:.1f}%")
print(f"Oracle parfait:      CAGR {oracle_cagr*100:.1f}%  x{oracle_eq[-1]:.1f}  DD {oracle_dd*100:.1f}%")
print(f"XGBoost {K}-fold OOS:  CAGR {model_cagr*100:.1f}%  x{model_eq[-1]:.1f}  DD {model_dd*100:.1f}%  "
      f"cash={n_cash_model}j ({n_cash_model/N*100:.0f}%)")

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
             label=f"XGBoost {K}-fold OOS (CAGR {model_cagr*100:.1f}%)",
             color="tab:red", linewidth=1.5)

# Fold boundaries
for k in range(1, K):
    ax1.axvline(plot_dates[folds[k][0]], color="gray", linestyle=":", alpha=0.3, linewidth=0.5)

# Shade oracle cash periods
for peak_i_orig, trough_i_orig in crises:
    if peak_i_orig < len(dates) and trough_i_orig < len(dates):
        ax1.axvspan(dates[peak_i_orig], dates[trough_i_orig], alpha=0.08, color="gray")

ax1.set_ylabel("Equity (log scale)")
ax1.set_title(f"Oracle parfait vs XGBoost OOS ({K}-fold temporal, regularized)")
ax1.legend(loc="upper left", fontsize=11)
ax1.grid(True, alpha=0.3)

# Panel 2: OOS probability
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

# ── Feature importance (average across folds) ─────────────
importance_matrix = np.zeros((K, len(feature_cols)))
for k in range(K):
    train_idx = np.concatenate([folds[j] for j in range(K) if j != k])
    m = xgb.XGBClassifier(**xgb_params)
    m.fit(X[train_idx], y[train_idx], verbose=False)
    importance_matrix[k] = m.feature_importances_

avg_imp = importance_matrix.mean(axis=0)
std_imp = importance_matrix.std(axis=0)
top_idx = np.argsort(avg_imp)[-20:][::-1]
print(f"\nTop 20 features (avg importance across {K} folds):")
for idx in top_idx:
    print(f"  {feature_cols[idx]:25s}  mean={avg_imp[idx]:.4f}  std={std_imp[idx]:.4f}")

plt.tight_layout()
plt.savefig("/home/greg/data_local/code/MyQTMv2/myfiles/oracle_vs_xgb_kfold.png", dpi=150)
plt.show()
print(f"\nSaved: myfiles/oracle_vs_xgb_kfold.png")
