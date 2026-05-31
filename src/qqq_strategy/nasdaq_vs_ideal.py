"""
NASDAQ QQQ vs Perfect Oracle Simulation (2006-2026)

Oracle strategy: with perfect hindsight on QQQ price alone.
- Detect all significant drawdowns (>10% peak-to-trough)
- EXIT at the exact peak before each drawdown
- RE-ENTER at the exact trough (bottom) of each drawdown
- No VIX, no heuristic — pure price oracle
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# --- Load data ---
qqq = pd.read_parquet("/home/greg/data_local/code/MyQTMv2/data/QQQ.parquet")
qqq = qqq.loc["2006-01-01":"2026-05-31", "close"].dropna()

DRAWDOWN_THRESHOLD = -0.10  # only avoid drawdowns > 10%

# --- Detect significant drawdowns ---
prices = qqq.values
dates = qqq.index
n = len(prices)

# Running max and drawdown at each point
running_max = np.maximum.accumulate(prices)
drawdown = (prices - running_max) / running_max

# Find crisis periods: contiguous regions where drawdown < threshold
# A crisis = from the peak before a >10% drawdown to the trough of that drawdown
crises = []  # list of (peak_idx, trough_idx)
in_crisis = False
peak_idx = 0
trough_idx = 0
trough_val = prices[0]

for i in range(n):
    if drawdown[i] >= 0:
        # New high — potential peak
        peak_idx = i
        trough_val = prices[i]
        trough_idx = i
        if in_crisis:
            in_crisis = False
    else:
        if prices[i] < trough_val:
            trough_val = prices[i]
            trough_idx = i
        if drawdown[i] < DRAWDOWN_THRESHOLD and not in_crisis:
            in_crisis = True

# After the loop, handle trailing crisis
# Now re-scan to collect complete crises (peak -> trough -> recovery)
crises = []
i = 0
while i < n:
    # Find next significant drawdown
    peak_i = i
    # Advance to running max
    while i < n and drawdown[i] >= 0:
        peak_i = i
        i += 1
    if i >= n:
        break
    # We're in a drawdown. Track the trough.
    trough_i = i
    min_dd = drawdown[i]
    significant = min_dd < DRAWDOWN_THRESHOLD
    # Continue until recovery (new high) or end
    while i < n and drawdown[i] < 0:
        if drawdown[i] < min_dd:
            min_dd = drawdown[i]
            trough_i = i
        if drawdown[i] < DRAWDOWN_THRESHOLD:
            significant = True
        i += 1
    if significant:
        crises.append((peak_i, trough_i))

print(f"Detected {len(crises)} significant drawdowns (>{abs(DRAWDOWN_THRESHOLD)*100:.0f}%):\n")
for peak_i, trough_i in crises:
    dd = (prices[trough_i] - prices[peak_i]) / prices[peak_i] * 100
    duration = (dates[trough_i] - dates[peak_i]).days
    print(f"  {dates[peak_i].date()} -> {dates[trough_i].date()}  "
          f"({duration}j)  DD: {dd:.1f}%  "
          f"(${prices[peak_i]:.0f} -> ${prices[trough_i]:.0f})")

# --- Build oracle equity ---
# Oracle is invested except between peak and trough of each crisis
qqq_ret = qqq.pct_change().fillna(0).values
LEVERAGE = 1.5

# Build a mask: True = invested, False = cash
invested_mask = np.ones(n, dtype=bool)
for peak_i, trough_i in crises:
    # Exit at close of peak day, re-enter at close of trough day
    # So days peak+1 to trough are in cash
    invested_mask[peak_i + 1: trough_i + 1] = False

oracle_equity = np.ones(n)
oracle_lev_equity = np.ones(n)
for i in range(1, n):
    if invested_mask[i]:
        oracle_equity[i] = oracle_equity[i - 1] * (1 + qqq_ret[i])
        oracle_lev_equity[i] = oracle_lev_equity[i - 1] * (1 + qqq_ret[i] * LEVERAGE)
    else:
        oracle_equity[i] = oracle_equity[i - 1]
        oracle_lev_equity[i] = oracle_lev_equity[i - 1]

# --- Buy & Hold ---
qqq_equity = np.cumprod(1 + qqq_ret)

# --- Metrics ---
years = (dates[-1] - dates[0]).days / 365.25
qqq_cagr = (qqq_equity[-1] / qqq_equity[0]) ** (1 / years) - 1
oracle_cagr = (oracle_equity[-1] / oracle_equity[0]) ** (1 / years) - 1

qqq_cummax = np.maximum.accumulate(qqq_equity)
qqq_dd = ((qqq_equity - qqq_cummax) / qqq_cummax).min()

oracle_cummax = np.maximum.accumulate(oracle_equity)
oracle_dd = ((oracle_equity - oracle_cummax) / oracle_cummax).min()

oracle_lev_cagr = (oracle_lev_equity[-1] / oracle_lev_equity[0]) ** (1 / years) - 1
oracle_lev_cummax = np.maximum.accumulate(oracle_lev_equity)
oracle_lev_dd = ((oracle_lev_equity - oracle_lev_cummax) / oracle_lev_cummax).min()

n_cash_days = int((~invested_mask).sum())

print(f"\n{'='*60}")
print(f"Periode: {dates[0].date()} -> {dates[-1].date()} ({years:.1f} ans)")
print(f"")
print(f"QQQ Buy & Hold:")
print(f"  CAGR:         {qqq_cagr*100:.1f}%")
print(f"  Total:        x{qqq_equity[-1]:.1f}")
print(f"  Max Drawdown: {qqq_dd*100:.1f}%")
print(f"")
print(f"Oracle x1 (exit peak, re-enter trough, DD>{abs(DRAWDOWN_THRESHOLD)*100:.0f}%):")
print(f"  CAGR:         {oracle_cagr*100:.1f}%")
print(f"  Total:        x{oracle_equity[-1]:.1f}")
print(f"  Max Drawdown: {oracle_dd*100:.1f}%")
print(f"  Jours en cash: {n_cash_days} ({n_cash_days/n*100:.0f}%)")
print(f"  Crises evitees: {len(crises)}")
print(f"")
print(f"Oracle x{LEVERAGE} (levier {LEVERAGE}):")
print(f"  CAGR:         {oracle_lev_cagr*100:.1f}%")
print(f"  Total:        x{oracle_lev_equity[-1]:.1f}")
print(f"  Max Drawdown: {oracle_lev_dd*100:.1f}%")
print(f"")
print(f"Gain oracle vs B&H: +{(oracle_cagr - qqq_cagr)*100:.1f}% CAGR annuel")

# --- Plot ---
fig, ax1 = plt.subplots(1, 1, figsize=(14, 7))

ax1.semilogy(dates, qqq_equity, label=f"QQQ Buy & Hold (CAGR {qqq_cagr*100:.1f}%)",
             color="tab:blue", linewidth=1.5)
ax1.semilogy(dates, oracle_equity,
             label=f"Oracle parfait x1 (CAGR {oracle_cagr*100:.1f}%)",
             color="tab:green", linewidth=2)
ax1.semilogy(dates, oracle_lev_equity,
             label=f"Oracle parfait x{LEVERAGE} (CAGR {oracle_lev_cagr*100:.1f}%)",
             color="tab:red", linewidth=2, linestyle="--")

# Shade crisis periods (cash)
for peak_i, trough_i in crises:
    dd = (prices[trough_i] - prices[peak_i]) / prices[peak_i] * 100
    ax1.axvspan(dates[peak_i], dates[trough_i], alpha=0.15, color="red")
    # Label the biggest crises
    if dd < -25:
        mid = dates[peak_i + (trough_i - peak_i) // 2]
        ax1.annotate(f"{dd:.0f}%", xy=(mid, oracle_equity[peak_i] * 0.7),
                     fontsize=9, color="red", ha="center", fontweight="bold")

ax1.set_ylabel("Equity (log scale)")
ax1.set_title(f"QQQ Buy & Hold vs Oracle Parfait (evite les DD >{abs(DRAWDOWN_THRESHOLD)*100:.0f}%)")
ax1.legend(loc="upper left", fontsize=12)
ax1.grid(True, alpha=0.3)

ax1.xaxis.set_major_locator(mdates.YearLocator(2))
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax1.set_xlabel("Date")

plt.tight_layout()
plt.savefig("/home/greg/data_local/code/MyQTMv2/myfiles/nasdaq_vs_oracle.png", dpi=150)
plt.show()
print("\nSaved: myfiles/nasdaq_vs_oracle.png")
