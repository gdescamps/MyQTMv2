"""
Data loading, feature engineering, and target construction for QQQ crisis-avoidance strategy.
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


def load_data(start="2006-01-01", end="2026-05-31", spread_lag=2):
    qqq = pd.read_parquet(DATA_DIR / "QQQ.parquet")["close"]
    vix = pd.read_parquet(DATA_DIR / "vix_ohlc.parquet")["close"]
    spread = pd.read_parquet(DATA_DIR / "fred_baa_spread.parquet")["baa_spread"]
    tlt = pd.read_parquet(DATA_DIR / "TLT.parquet")["close"]

    # Shift spread by J+2 to reflect FRED publication delay
    if spread_lag > 0:
        spread = spread.shift(spread_lag)

    qqq = qqq.loc[start:end].dropna()
    vix = vix.loc[start:end].dropna()
    spread = spread.loc[start:end].dropna()
    tlt = tlt.loc[start:end].dropna()

    common = qqq.index.intersection(vix.index).intersection(spread.index).intersection(tlt.index)
    return qqq.loc[common], vix.loc[common], spread.loc[common], tlt.loc[common]


def build_realtime_target(prices, dd_exit=-0.10, dd_reenter=-0.05):
    n = len(prices)
    running_max = np.maximum.accumulate(prices)
    drawdown_arr = (prices - running_max) / running_max

    target = np.ones(n, dtype=int)
    invested = True
    for i in range(n):
        if invested and drawdown_arr[i] < dd_exit:
            invested = False
        elif not invested and drawdown_arr[i] > dd_reenter:
            invested = True
        target[i] = 1 if invested else 0

    return target, drawdown_arr


def build_features(qqq, vix, spread, tlt=None):
    prices = qqq.values
    dates = qqq.index
    n = len(prices)

    running_max = np.maximum.accumulate(prices)
    drawdown_arr = (prices - running_max) / running_max

    df = pd.DataFrame(index=dates)
    df["qqq_close"] = prices
    df["vix"] = vix.values
    df["spread"] = spread.values
    qqq_daily_ret = pd.Series(prices, index=dates).pct_change()

    # ── Linear features ───────────────────────────────────
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

    # ── Non-linear features ───────────────────────────────

    # 1. Asymmetric volatility
    for w in [20, 50]:
        down_ret = qqq_daily_ret.clip(upper=0)
        up_ret = qqq_daily_ret.clip(lower=0)
        df[f"qqq_downvol{w}"] = down_ret.rolling(w).std()
        df[f"qqq_upvol{w}"] = up_ret.rolling(w).std()
        df[f"qqq_vol_asym{w}"] = df[f"qqq_downvol{w}"] / (df[f"qqq_upvol{w}"] + 1e-8)

    # 2. Skewness & kurtosis
    for w in [20, 50]:
        df[f"qqq_skew{w}"] = qqq_daily_ret.rolling(w).skew()
        df[f"qqq_kurt{w}"] = qqq_daily_ret.rolling(w).kurt()

    # 3. Acceleration
    for w in [5, 10, 20]:
        df[f"qqq_accel{w}"] = df[f"qqq_ret{w}"].diff(w)

    # 4. Volatility of volatility
    vol20 = df["qqq_vol20"]
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
    df["vix_x_vol20"] = df["vix"] * df["qqq_vol20"]
    df["spread_x_vol20"] = df["spread"] * df["qqq_vol20"]
    df["vix_x_ret20"] = df["vix"] * df["qqq_ret20"]

    # 7. VIX term structure proxy
    df["vix_term"] = df["vix_sma5"] / df["vix_sma50"]
    df["spread_term"] = df["spread_sma5"] / df["spread_sma50"]

    # 8. Bollinger band position
    for w in [20, 50]:
        sma = df[f"qqq_sma{w}"]
        std = df["qqq_close"].rolling(w).std()
        df[f"qqq_bband{w}"] = (df["qqq_close"] - sma) / (std + 1e-8)

    # 9. Mean reversion signals
    df["qqq_ret5_vs_ret50"] = df["qqq_ret5"] - df["qqq_ret50"]
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

    # 11. Rolling max drawdown
    for w in [20, 50]:
        rm = df["qqq_close"].rolling(w).max()
        df[f"qqq_maxdd{w}"] = (df["qqq_close"] - rm) / rm

    # 12. VIX spike
    df["vix_spike1d"] = df["vix"].pct_change()
    df["vix_spike5d"] = df["vix"].pct_change(5)

    # 13. Spread acceleration
    df["spread_accel"] = df["spread"].diff(5).diff(5)

    # 14. QQQ/TLT ratio (risk-on vs risk-off)
    if tlt is not None:
        tlt_s = pd.Series(tlt.values, index=dates)
        df["tlt"] = tlt_s
        df["qqq_tlt_ratio"] = df["qqq_close"] / tlt_s
        for w in [5, 10, 20, 50]:
            df[f"qqq_tlt_ratio_ret{w}"] = df["qqq_tlt_ratio"].pct_change(w)
        df["qqq_tlt_ratio_vs_sma20"] = df["qqq_tlt_ratio"] / df["qqq_tlt_ratio"].rolling(20).mean() - 1
        df["qqq_tlt_ratio_vs_sma50"] = df["qqq_tlt_ratio"] / df["qqq_tlt_ratio"].rolling(50).mean() - 1
        # TLT momentum (flight to safety)
        for w in [5, 10, 20, 50]:
            df[f"tlt_ret{w}"] = tlt_s.pct_change(w)
        df["tlt_vs_sma20"] = tlt_s / tlt_s.rolling(20).mean() - 1
        df["tlt_vs_sma50"] = tlt_s / tlt_s.rolling(50).mean() - 1
        # Cross interactions
        df["tlt_x_vix"] = df["tlt_ret5"] * df["vix"]
        df["tlt_x_dd"] = df["tlt_ret5"] * drawdown_arr
    return df
