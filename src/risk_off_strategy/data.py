"""
Data loading, feature engineering, and target construction for crisis-avoidance strategy.
Supports QQQ (Nasdaq-100) and SPY (S&P 500).
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


def load_data(ticker="QQQ", start="2006-01-01", end="2026-05-31", spread_lag=2):
    price = pd.read_parquet(DATA_DIR / f"{ticker}.parquet")["close"]
    vix = pd.read_parquet(DATA_DIR / "vix_ohlc.parquet")["close"]
    spread = pd.read_parquet(DATA_DIR / "fred_baa_spread.parquet")["baa_spread"]
    tlt = pd.read_parquet(DATA_DIR / "TLT.parquet")["close"]

    # Shift spread by J+2 to reflect FRED publication delay
    if spread_lag > 0:
        spread = spread.shift(spread_lag)

    price = price.loc[start:end].dropna()
    vix = vix.loc[start:end].dropna()
    spread = spread.loc[start:end].dropna()
    tlt = tlt.loc[start:end].dropna()

    common = price.index.intersection(vix.index).intersection(spread.index).intersection(tlt.index)
    return price.loc[common], vix.loc[common], spread.loc[common], tlt.loc[common]


def build_realtime_target(prices, dd_exit=-0.10, dd_reenter=-0.05, lookahead=0,
                          reenter_from_bottom=None):
    """Build binary target from drawdown state machine.

    lookahead: number of future days used to compute the state at time t.
    Label at t = state machine result at t+lookahead.
    With embargo >= lookahead in walk-forward, there is no data leakage.

    reenter_from_bottom: if set (e.g. 0.10), re-enter when price recovers
    this fraction from the lowest point since exit, instead of using dd_reenter.
    """
    n = len(prices)
    running_max = np.maximum.accumulate(prices)
    drawdown_arr = (prices - running_max) / running_max

    # Run state machine on full price series
    full_target = np.ones(n, dtype=int)
    invested = True
    crisis_low = 0.0
    for i in range(n):
        if invested and drawdown_arr[i] < dd_exit:
            invested = False
            crisis_low = prices[i]
        elif not invested:
            crisis_low = min(crisis_low, prices[i])
            if reenter_from_bottom is not None:
                if prices[i] >= crisis_low * (1 + reenter_from_bottom):
                    invested = True
            else:
                if drawdown_arr[i] > dd_reenter:
                    invested = True
        full_target[i] = 1 if invested else 0

    if lookahead > 0:
        # Label at t = state at t+lookahead (clip at end)
        target = np.ones(n, dtype=int)
        for i in range(n):
            target[i] = full_target[min(i + lookahead, n - 1)]
    else:
        target = full_target

    return target, drawdown_arr


def build_features(price, vix, spread, tlt=None, prefix="px"):
    """Build features from price series. prefix is used for column names."""
    prices = price.values
    dates = price.index
    n = len(prices)

    running_max = np.maximum.accumulate(prices)
    drawdown_arr = (prices - running_max) / running_max

    df = pd.DataFrame(index=dates)
    p = prefix
    df[f"{p}_close"] = prices
    df["vix"] = vix.values
    df["spread"] = spread.values
    daily_ret = pd.Series(prices, index=dates).pct_change()

    # ── Linear features ───────────────────────────────────
    for w in [5, 10, 20, 50, 100, 200]:
        df[f"{p}_sma{w}"] = df[f"{p}_close"].rolling(w).mean()
        df[f"{p}_ret{w}"] = df[f"{p}_close"].pct_change(w)
        df[f"{p}_vol{w}"] = daily_ret.rolling(w).std()
    for w in [20, 50, 100, 200]:
        df[f"{p}_vs_sma{w}"] = df[f"{p}_close"] / df[f"{p}_sma{w}"] - 1
    df[f"{p}_dd"] = drawdown_arr
    delta = df[f"{p}_close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df[f"{p}_rsi"] = 100 - 100 / (1 + gain / loss)
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

    # ── Non-linear features ───────────────────────────────

    # 1. Asymmetric volatility
    for w in [20, 50]:
        down_ret = daily_ret.clip(upper=0)
        up_ret = daily_ret.clip(lower=0)
        df[f"{p}_downvol{w}"] = down_ret.rolling(w).std()
        df[f"{p}_upvol{w}"] = up_ret.rolling(w).std()
        df[f"{p}_vol_asym{w}"] = df[f"{p}_downvol{w}"] / (df[f"{p}_upvol{w}"] + 1e-8)

    # 2. Skewness & kurtosis
    for w in [20, 50]:
        df[f"{p}_skew{w}"] = daily_ret.rolling(w).skew()
        df[f"{p}_kurt{w}"] = daily_ret.rolling(w).kurt()

    # 3. Acceleration
    for w in [5, 10, 20]:
        df[f"{p}_accel{w}"] = df[f"{p}_ret{w}"].diff(w)

    # 4. Volatility of volatility
    vol20 = df[f"{p}_vol20"]
    df["volvol_20"] = vol20.rolling(20).std()
    df["vol_accel"] = vol20.diff(5)
    df["vol_regime"] = vol20 / vol20.rolling(100).mean()

    # 5. Drawdown dynamics
    df["dd_speed"] = pd.Series(drawdown_arr, index=dates).diff(5)
    dd_dur = np.zeros(n)
    cnt = 0
    for i in range(n):
        if drawdown_arr[i] >= 0:
            cnt = 0
        else:
            cnt += 1
        dd_dur[i] = cnt
    df["dd_duration"] = dd_dur
    df["dd_depth_x_duration"] = drawdown_arr * dd_dur

    # 6. Cross-asset interactions
    df["vix_x_spread"] = df["vix"] * df["spread"]
    df["vix_x_dd"] = df["vix"] * drawdown_arr
    df[f"vix_x_vol20"] = df["vix"] * df[f"{p}_vol20"]
    df[f"spread_x_vol20"] = df["spread"] * df[f"{p}_vol20"]
    df[f"vix_x_ret20"] = df["vix"] * df[f"{p}_ret20"]

    # 7. VIX term structure proxy
    df["vix_term"] = df["vix_sma5"] / df["vix_sma50"]
    df["spread_term"] = df["spread_sma5"] / df["spread_sma50"]

    # 8. Bollinger band position
    for w in [20, 50]:
        sma = df[f"{p}_sma{w}"]
        std = df[f"{p}_close"].rolling(w).std()
        df[f"{p}_bband{w}"] = (df[f"{p}_close"] - sma) / (std + 1e-8)

    # 9. Mean reversion signals
    df[f"{p}_ret5_vs_ret50"] = df[f"{p}_ret5"] - df[f"{p}_ret50"]
    df[f"{p}_ret10_vs_ret100"] = df[f"{p}_ret10"] - df[f"{p}_ret100"]

    # 10. Consecutive down days
    neg_days = (daily_ret < 0).astype(int)
    consec = np.zeros(n)
    cnt = 0
    for i in range(n):
        if neg_days.iloc[i]:
            cnt += 1
        else:
            cnt = 0
        consec[i] = cnt
    df["consec_down"] = consec

    # 11. Rolling max drawdown
    for w in [20, 50]:
        rm = df[f"{p}_close"].rolling(w).max()
        df[f"{p}_maxdd{w}"] = (df[f"{p}_close"] - rm) / rm

    # 12. VIX spike
    df["vix_spike1d"] = df["vix"].pct_change()
    df["vix_spike5d"] = df["vix"].pct_change(5)

    # 13. Spread acceleration
    df["spread_accel"] = df["spread"].diff(5).diff(5)

    # 14. Price/TLT ratio (risk-on vs risk-off)
    if tlt is not None:
        tlt_s = pd.Series(tlt.values, index=dates)
        df["tlt"] = tlt_s
        df[f"{p}_tlt_ratio"] = df[f"{p}_close"] / tlt_s
        for w in [5, 10, 20, 50]:
            df[f"{p}_tlt_ratio_ret{w}"] = df[f"{p}_tlt_ratio"].pct_change(w)
        df[f"{p}_tlt_ratio_vs_sma20"] = df[f"{p}_tlt_ratio"] / df[f"{p}_tlt_ratio"].rolling(20).mean() - 1
        df[f"{p}_tlt_ratio_vs_sma50"] = df[f"{p}_tlt_ratio"] / df[f"{p}_tlt_ratio"].rolling(50).mean() - 1
        # TLT momentum (flight to safety)
        for w in [5, 10, 20, 50]:
            df[f"tlt_ret{w}"] = tlt_s.pct_change(w)
        df["tlt_vs_sma20"] = tlt_s / tlt_s.rolling(20).mean() - 1
        df["tlt_vs_sma50"] = tlt_s / tlt_s.rolling(50).mean() - 1
        # Cross interactions
        df["tlt_x_vix"] = df["tlt_ret5"] * df["vix"]
        df["tlt_x_dd"] = df["tlt_ret5"] * drawdown_arr
    return df
