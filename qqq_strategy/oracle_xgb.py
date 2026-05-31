"""
Train an XGBoost model to replicate the Oracle's invest/cash decisions.
Intentionally overfit (no validation split) to see how well the features
can capture the oracle signal in-sample.
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

# Align dates
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
drawdown = (prices - running_max) / running_max

crises = []
i = 0
while i < n:
    peak_i = i
    while i < n and drawdown[i] >= 0:
        peak_i = i
        i += 1
    if i >= n:
        break
    trough_i = i
    min_dd = drawdown[i]
    significant = min_dd < DRAWDOWN_THRESHOLD
    while i < n and drawdown[i] < 0:
        if drawdown[i] < min_dd:
            min_dd = drawdown[i]
            trough_i = i
        if drawdown[i] < DRAWDOWN_THRESHOLD:
            significant = True
        i += 1
    if significant:
        crises.append((peak_i, trough_i))

# Target: 1 = invested, 0 = cash
invested_mask = np.ones(n, dtype=int)
for peak_i, trough_i in crises:
    invested_mask[peak_i + 1: trough_i + 1] = 0

print(f"Oracle: {len(crises)} crises, {(invested_mask == 0).sum()} cash days "
      f"({(invested_mask == 0).sum()/n*100:.0f}%)")

# ── Feature engineering ────────────────────────────────────
df = pd.DataFrame(index=dates)
df["qqq_close"] = prices
df["vix"] = vix.values
df["spread"] = spread.values

# QQQ features
for w in [5, 10, 20, 50, 100, 200]:
    df[f"qqq_sma{w}"] = df["qqq_close"].rolling(w).mean()
    df[f"qqq_ret{w}"] = df["qqq_close"].pct_change(w)
    df[f"qqq_vol{w}"] = df["qqq_close"].pct_change().rolling(w).std()

# QQQ vs SMAs
for w in [20, 50, 100, 200]:
    df[f"qqq_vs_sma{w}"] = df["qqq_close"] / df[f"qqq_sma{w}"] - 1

# Drawdown from running max
df["qqq_dd"] = drawdown

# RSI
delta = df["qqq_close"].diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = (-delta.clip(upper=0)).rolling(14).mean()
df["qqq_rsi"] = 100 - 100 / (1 + gain / loss)

# VIX features
for w in [5, 10, 20, 50]:
    df[f"vix_sma{w}"] = df["vix"].rolling(w).mean()
    df[f"vix_ret{w}"] = df["vix"].pct_change(w)
df["vix_vs_sma20"] = df["vix"] / df["vix_sma20"] - 1
df["vix_vs_sma50"] = df["vix"] / df["vix_sma50"] - 1

# Credit spread features
for w in [5, 10, 20, 50]:
    df[f"spread_sma{w}"] = df["spread"].rolling(w).mean()
    df[f"spread_ret{w}"] = df["spread"].pct_change(w)
df["spread_vs_sma20"] = df["spread"] / df["spread_sma20"] - 1
df["spread_vs_sma50"] = df["spread"] / df["spread_sma50"] - 1

# Target
df["target"] = invested_mask

# Drop warmup rows
df = df.dropna()
print(f"Features: {len([c for c in df.columns if c != 'target'])} cols, {len(df)} rows")

# ── Train XGBoost (overfit) ────────────────────────────────
feature_cols = [c for c in df.columns if c != "target"]
X = df[feature_cols].values
y = df["target"].values

model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    subsample=1.0,
    colsample_bytree=1.0,
    min_child_weight=1,
    reg_alpha=0,
    reg_lambda=0,
    random_state=42,
    use_label_encoder=False,
    eval_metric="logloss",
)
model.fit(X, y, verbose=False)

# Predictions
y_pred = model.predict(X)
y_proba = model.predict_proba(X)[:, 1]

accuracy = (y_pred == y).mean()
print(f"In-sample accuracy: {accuracy*100:.1f}%")
print(f"  Invested correct: {((y_pred == 1) & (y == 1)).sum()}/{(y == 1).sum()}")
print(f"  Cash correct:     {((y_pred == 0) & (y == 0)).sum()}/{(y == 0).sum()}")

# ── Build equity curves ────────────────────────────────────
qqq_ret_all = qqq.pct_change().fillna(0)
qqq_ret_df = qqq_ret_all.loc[df.index].values

# Buy & Hold
bh_equity = np.cumprod(1 + qqq_ret_df)

# Oracle equity
oracle_target = df["target"].values
oracle_eq = np.ones(len(df))
for i in range(1, len(df)):
    if oracle_target[i]:
        oracle_eq[i] = oracle_eq[i - 1] * (1 + qqq_ret_df[i])
    else:
        oracle_eq[i] = oracle_eq[i - 1]

# XGBoost model equity
model_eq = np.ones(len(df))
for i in range(1, len(df)):
    if y_pred[i]:
        model_eq[i] = model_eq[i - 1] * (1 + qqq_ret_df[i])
    else:
        model_eq[i] = model_eq[i - 1]

# Metrics
plot_dates = df.index
years = (plot_dates[-1] - plot_dates[0]).days / 365.25

bh_cagr = (bh_equity[-1] ** (1 / years)) - 1
oracle_cagr = (oracle_eq[-1] ** (1 / years)) - 1
model_cagr = (model_eq[-1] ** (1 / years)) - 1

bh_dd = ((bh_equity - np.maximum.accumulate(bh_equity)) / np.maximum.accumulate(bh_equity)).min()
oracle_dd = ((oracle_eq - np.maximum.accumulate(oracle_eq)) / np.maximum.accumulate(oracle_eq)).min()
model_dd = ((model_eq - np.maximum.accumulate(model_eq)) / np.maximum.accumulate(model_eq)).min()

print(f"\n{'='*60}")
print(f"Periode: {plot_dates[0].date()} -> {plot_dates[-1].date()} ({years:.1f} ans)")
print(f"")
print(f"QQQ Buy & Hold:    CAGR {bh_cagr*100:.1f}%  x{bh_equity[-1]:.1f}  DD {bh_dd*100:.1f}%")
print(f"Oracle parfait:    CAGR {oracle_cagr*100:.1f}%  x{oracle_eq[-1]:.1f}  DD {oracle_dd*100:.1f}%")
print(f"XGBoost overfit:   CAGR {model_cagr*100:.1f}%  x{model_eq[-1]:.1f}  DD {model_dd*100:.1f}%")

# ── Feature importance ─────────────────────────────────────
importances = model.feature_importances_
top_idx = np.argsort(importances)[-15:][::-1]
print(f"\nTop 15 features:")
for idx in top_idx:
    print(f"  {feature_cols[idx]:25s} {importances[idx]:.4f}")

# ── Plot ───────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), height_ratios=[3, 1],
                                sharex=True, gridspec_kw={"hspace": 0.08})

ax1.semilogy(plot_dates, bh_equity, label=f"QQQ Buy & Hold (CAGR {bh_cagr*100:.1f}%)",
             color="tab:blue", linewidth=1.2, alpha=0.7)
ax1.semilogy(plot_dates, oracle_eq,
             label=f"Oracle parfait (CAGR {oracle_cagr*100:.1f}%)",
             color="tab:green", linewidth=2)
ax1.semilogy(plot_dates, model_eq,
             label=f"XGBoost overfit (CAGR {model_cagr*100:.1f}%)",
             color="tab:red", linewidth=1.5, linestyle="--")

# Shade oracle cash periods
for peak_i_orig, trough_i_orig in crises:
    if peak_i_orig < len(dates) and trough_i_orig < len(dates):
        ax1.axvspan(dates[peak_i_orig], dates[trough_i_orig], alpha=0.1, color="gray")

ax1.set_ylabel("Equity (log scale)")
ax1.set_title("Oracle parfait vs XGBoost overfit (QQQ + VIX + Credit Spread)")
ax1.legend(loc="upper left", fontsize=11)
ax1.grid(True, alpha=0.3)

# Panel 2: model probability of being invested
ax2.plot(plot_dates, y_proba, color="tab:purple", linewidth=0.6, alpha=0.8, label="P(invested)")
ax2.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
ax2.fill_between(plot_dates, 0, 1, where=(y == 0), alpha=0.15, color="red", label="Oracle cash")
ax2.set_ylabel("P(invested)")
ax2.set_xlabel("Date")
ax2.legend(loc="upper right", fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(-0.05, 1.05)

ax2.xaxis.set_major_locator(mdates.YearLocator(2))
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

plt.tight_layout()
plt.savefig("/home/greg/data_local/code/MyQTMv2/myfiles/oracle_vs_xgb.png", dpi=150)
plt.show()
print("\nSaved: myfiles/oracle_vs_xgb.png")
