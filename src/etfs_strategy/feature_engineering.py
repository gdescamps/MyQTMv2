"""
Feature engineering for MyQTM-ETF portfolio model.

Builds a panel DataFrame with MultiIndex (date, etf_id) containing:
  - Technical features per ETF (momentum, volatility, RSI, MA, ATR, volume)
  - Macro/regime features from FRED (VIX, HY spread, yield curve, DXY)
  - Smart money: shares_outstanding_z20 from iShares XLS
  - Cross-sectional: ret_20d/ret_5d z-score within section block
  - Label: forward ret_10d (absolute, z-scored cross-sectionally in train.py)

Output: data/features.parquet
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
ROOT = Path(__file__).resolve().parent.parent.parent
from etf import UNIVERSE

DATA = ROOT / "data"
DATA.mkdir(parents=True, exist_ok=True)

# Map each ETF bourso ticker → iShares XLS ticker (for shares_outstanding)
# 26-ETF universe — smart money via iShares US or UCITS XLS
ISHARES_MAP = {
    # --- Geo equity ---
    "IVV":      "IVV",     # S&P 500 — direct since 2000
    "QQQ":      "CNDX",    # Nasdaq 100 — CNDX UCITS proxy
    "ACWI":     "ACWI",    # MSCI World — direct
    "EEM":      "EEM",     # Emerging Markets — since 2003
    "IEMG":     "EIMI",    # Core MSCI EM IMI — EIMI UCITS proxy
    "EMXC":     "EMXC",    # MSCI EM ex-China — direct
    "ILF":      "LTAM",    # Latin America 40 — LTAM UCITS proxy
    "EWY":      "EWY",     # Korea — since 2000
    "EWT":      "EWT",     # Taiwan — since 2000
    "EWZ":      "EWZ",     # Brazil — since 2000
    "EWW":      "EWW",     # Mexico — since 1996
    "EWC":      "EWC",     # Canada — since 1996
    "EWJ":      "EWJ",     # Japan — since 1996
    "TUR":      "TUR",     # Turkey — since 2008
    "FXI":      "FXI",     # China — since 2004
    "ISF.L":    "ISF",     # FTSE 100 — direct
    "IEUR":     "IMEU",    # Core MSCI Europe — IMEU UCITS proxy
    "EZU":      "CEU1",    # MSCI Eurozone — CEU1 UCITS proxy
    # EPP removed — no UCITS EUR equivalent on IB
    "SUSA":     "SUAS",    # MSCI USA SRI — SUAS UCITS proxy
    # --- Thematic ---
    "SOXX":     "SOXX",    # Semiconductors — since 2001
    "ROBO":     "RBOT",    # Automation & Robotics — RBOT UCITS proxy
    "ICLN":     "ICLN",    # Global Clean Energy — since 2008
    "EXX1.DE":  "EXX1",    # Euro Banks — since 2002
    # --- Commodity ---
    "RING":     "RING",    # Gold Miners — since 2012
    "IEO":      "IEO",     # Oil & Gas — since 2006
    "SXRS.DE":  "GSG",     # Diversified Commodity — GSG proxy
}


# ---------------------------------------------------------------------------
# Technical helpers
# ---------------------------------------------------------------------------

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).rolling(period, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _atr_norm(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean() / close.replace(0, np.nan)


def _z(series: pd.Series, window: int) -> pd.Series:
    m = series.rolling(window, min_periods=window // 2).mean()
    s = series.rolling(window, min_periods=window // 2).std()
    return (series - m) / s.replace(0, np.nan)


# ---------------------------------------------------------------------------
# Per-ETF technical features
# ---------------------------------------------------------------------------

def compute_etf_features(ohlcv: pd.DataFrame, shares_df: pd.DataFrame | None) -> pd.DataFrame:
    # Use 'close' if available; some files have 'adj_close' instead
    if "close" in ohlcv.columns:
        c = ohlcv["close"]
    else:
        c = ohlcv["adj_close"]

    h = ohlcv.get("high", c)
    l = ohlcv.get("low",  c)
    v = ohlcv.get("volume", pd.Series(np.nan, index=c.index))

    f = pd.DataFrame(index=c.index)
    r1 = c.pct_change(1)

    # ---- Momentum (multiple horizons) ----
    for d in [1, 5, 10, 20, 40, 60, 120, 250]:
        f[f"ret_{d}d"] = c.pct_change(d)

    # Momentum ratios (short/long)
    f["mom_ratio_5v20"]   = f["ret_5d"]  / f["ret_20d"].replace(0, np.nan)
    f["mom_ratio_5v60"]   = f["ret_5d"]  / f["ret_60d"].replace(0, np.nan)
    f["mom_ratio_20v60"]  = f["ret_20d"] / f["ret_60d"].replace(0, np.nan)
    f["mom_ratio_20v120"] = f["ret_20d"] / f["ret_120d"].replace(0, np.nan)
    f["mom_ratio_60v120"] = f["ret_60d"] / f["ret_120d"].replace(0, np.nan)

    # ---- Volatility (multiple horizons, annualised) ----
    for d in [5, 10, 20, 60, 120]:
        f[f"vol_{d}d"] = r1.rolling(d, min_periods=max(d // 2, 3)).std() * np.sqrt(252)

    # Volatility ratios (short/long)
    f["vol_ratio_5v20"]   = f["vol_5d"]  / f["vol_20d"].replace(0, np.nan)
    f["vol_ratio_5v60"]   = f["vol_5d"]  / f["vol_60d"].replace(0, np.nan)
    f["vol_ratio_20v60"]  = f["vol_20d"] / f["vol_60d"].replace(0, np.nan)
    f["vol_ratio_10v120"] = f["vol_10d"] / f["vol_120d"].replace(0, np.nan)
    f["vol_ratio_20v120"] = f["vol_20d"] / f["vol_120d"].replace(0, np.nan)

    # ---- RSI (multiple periods) ----
    for p in [5, 7, 14, 21, 60]:
        f[f"rsi_{p}"] = _rsi(c, p)

    # RSI crossovers
    f["rsi_7v21"] = f["rsi_7"] - f["rsi_21"]
    f["rsi_14v60"] = f["rsi_14"] - f["rsi_60"]

    # ---- Moving averages ----
    ma10  = c.rolling(10,  min_periods=7).mean()
    ma20  = c.rolling(20,  min_periods=15).mean()
    ma50  = c.rolling(50,  min_periods=40).mean()
    ma100 = c.rolling(100, min_periods=75).mean()
    ma200 = c.rolling(200, min_periods=150).mean()

    # Price vs MA
    for ma, name in [(ma10, "10"), (ma20, "20"), (ma50, "50"), (ma100, "100"), (ma200, "200")]:
        f[f"price_vs_ma{name}"] = c / ma.replace(0, np.nan) - 1

    # MA slopes
    f["ma_10_slope"] = ma10.pct_change(5)
    f["ma_20_slope"] = ma20.pct_change(5)
    f["ma_50_slope"] = ma50.pct_change(20)
    f["ma_100_slope"] = ma100.pct_change(20)
    f["ma_200_slope"] = ma200.pct_change(60)

    # MA crossovers (A/B ratios)
    f["ma10_vs_ma20"]  = ma10 / ma20.replace(0, np.nan) - 1
    f["ma10_vs_ma50"]  = ma10 / ma50.replace(0, np.nan) - 1
    f["ma20_vs_ma50"]  = ma20 / ma50.replace(0, np.nan) - 1
    f["ma20_vs_ma100"] = ma20 / ma100.replace(0, np.nan) - 1
    f["ma50_vs_ma100"] = ma50 / ma100.replace(0, np.nan) - 1
    f["ma50_vs_ma200"] = ma50 / ma200.replace(0, np.nan) - 1
    f["ma100_vs_ma200"] = ma100 / ma200.replace(0, np.nan) - 1

    # ---- ATR (multiple periods) ----
    for p in [7, 14, 21]:
        f[f"atr_{p}"] = _atr_norm(h, l, c, p)

    # ATR ratios
    f["atr_ratio_7v14"] = f["atr_7"] / f["atr_14"].replace(0, np.nan)
    f["atr_ratio_7v21"] = f["atr_7"] / f["atr_21"].replace(0, np.nan)
    f["atr_ratio_14v21"] = f["atr_14"] / f["atr_21"].replace(0, np.nan)

    # ---- Volume dollar z-scores (multiple horizons) ----
    dvol = c * v
    for d in [5, 10, 20, 60, 120]:
        f[f"volume_z{d}"] = _z(dvol, d)

    # Volume crossovers
    f["vol_cross_5v20"]  = f["volume_z5"]  - f["volume_z20"]
    f["vol_cross_5v60"]  = f["volume_z5"]  - f["volume_z60"]
    f["vol_cross_20v60"] = f["volume_z20"] - f["volume_z60"]
    f["vol_cross_20v120"] = f["volume_z20"] - f["volume_z120"]

    # ---- Price position (Bollinger-like) ----
    for d in [20, 60, 120]:
        ma = c.rolling(d, min_periods=d // 2).mean()
        std = c.rolling(d, min_periods=d // 2).std()
        f[f"bb_position_{d}"] = (c - ma) / std.replace(0, np.nan)

    # ---- Drawdown from rolling max ----
    for d in [20, 60, 120, 250]:
        roll_max = c.rolling(d, min_periods=d // 2).max()
        f[f"drawdown_{d}"] = c / roll_max.replace(0, np.nan) - 1

    # ---- High/Low range position ----
    for d in [20, 60]:
        roll_high = h.rolling(d, min_periods=d // 2).max()
        roll_low  = l.rolling(d, min_periods=d // 2).min()
        rng = (roll_high - roll_low).replace(0, np.nan)
        f[f"hl_position_{d}"] = (c - roll_low) / rng

    # ---- Skewness & Kurtosis of returns ----
    for d in [20, 60]:
        f[f"skew_{d}"]     = r1.rolling(d, min_periods=d // 2).skew()
        f[f"kurtosis_{d}"] = r1.rolling(d, min_periods=d // 2).kurt()

    # ---- Shares outstanding z-scores (smart money flows) ----
    # Smart-money is enabled in both short and long mode. When XLS data is
    # missing for a given ETF the columns are still emitted (as NaN) to keep
    # the panel schema consistent across ETFs.
    if shares_df is not None and "shares_outstanding" in shares_df.columns:
        so = shares_df["shares_outstanding"].reindex(c.index, method="ffill")
        so_chg = so.diff(1)
        for d in [5, 10, 20, 60, 120]:
            f[f"shares_outstanding_z{d}"] = _z(so_chg, d)
        # Smart money crossovers
        f["so_cross_5v20"]   = f["shares_outstanding_z5"]  - f["shares_outstanding_z20"]
        f["so_cross_5v60"]   = f["shares_outstanding_z5"]  - f["shares_outstanding_z60"]
        f["so_cross_20v60"]  = f["shares_outstanding_z20"] - f["shares_outstanding_z60"]
        f["so_cross_20v120"] = f["shares_outstanding_z20"] - f["shares_outstanding_z120"]
        f["so_cross_60v120"] = f["shares_outstanding_z60"] - f["shares_outstanding_z120"]
        # Smart money momentum (pct change of SO level)
        f["so_ret_20d"] = so.pct_change(20)
        f["so_ret_60d"] = so.pct_change(60)
    else:
        for d in [5, 10, 20, 60, 120]:
            f[f"shares_outstanding_z{d}"] = np.nan
        f["so_cross_5v20"]   = np.nan
        f["so_cross_5v60"]   = np.nan
        f["so_cross_20v60"]  = np.nan
        f["so_cross_20v120"] = np.nan
        f["so_cross_60v120"] = np.nan
        f["so_ret_20d"]      = np.nan
        f["so_ret_60d"]      = np.nan

    # ---- Expanding (since inception) statistics ----
    cum_ret = c / c.iloc[0]
    days_since_start = pd.Series(np.arange(1, len(c) + 1), index=c.index, dtype=float)
    f["ann_ret_since_start"] = cum_ret ** (252 / days_since_start.clip(lower=1)) - 1

    cum_max = c.expanding().max()
    drawdown_series = c / cum_max - 1
    f["max_dd_since_start"] = drawdown_series.expanding().min()

    f["vol_since_start"] = r1.expanding(min_periods=60).std() * np.sqrt(252)

    expanding_mean = r1.expanding(min_periods=60).mean() * 252
    expanding_std = r1.expanding(min_periods=60).std() * np.sqrt(252)
    f["sharpe_since_start"] = expanding_mean / expanding_std.replace(0, np.nan)

    f["current_dd"] = drawdown_series

    # Forward returns (label candidates) — shifted BACK N days (no lookahead)
    f["ret_5d_fwd"]  = c.pct_change(5).shift(-5)
    f["ret_10d_fwd"] = c.pct_change(10).shift(-10)
    f["ret_15d_fwd"] = c.pct_change(15).shift(-15)
    f["ret_20d_fwd"] = c.pct_change(20).shift(-20)
    f["ret_90d_fwd"] = c.pct_change(90).shift(-90)

    # Forward Sharpe label for Follow Leads — risk-adjusted forward return
    vol_10d_fwd = r1.rolling(10, min_periods=5).std().shift(-10) * np.sqrt(252)
    f["sharpe_10d_fwd"] = f["ret_10d_fwd"] / vol_10d_fwd.replace(0, np.nan)

    return f


# ---------------------------------------------------------------------------
# Macro features (same for all ETFs on each date)
# ---------------------------------------------------------------------------

def load_macro() -> pd.DataFrame:
    mac = pd.DataFrame()

    def _fred(fname: str, col: str) -> pd.Series:
        path = DATA / f"fred_{fname}.parquet"
        if not path.exists():
            return pd.Series(dtype=float, name=col)
        df = pd.read_parquet(path)
        return df.iloc[:, 0].rename(col)

    vix = _fred("vix", "vix")
    mac["vix_level"]           = vix
    mac["vix_velocity"]        = vix.diff(5)
    mac["vix_velocity_20"]     = vix.diff(20)
    mac["vix_reversion_force"] = _z(vix, 60) * -1
    for d in [5, 20, 60]:
        mac[f"vix_z{d}"] = _z(vix, d)
    mac["vix_cross_5v20"]  = _z(vix, 5) - _z(vix, 20)
    mac["vix_cross_5v60"]  = _z(vix, 5) - _z(vix, 60)
    mac["vix_cross_20v60"] = _z(vix, 20) - _z(vix, 60)

    hy = _fred("hy_spread", "hy")
    mac["hy_spread"] = hy
    for d in [5, 20, 60]:
        mac[f"hy_spread_z{d}"] = _z(hy, d)
    mac["hy_spread_velocity"]    = hy.diff(5)
    mac["hy_spread_velocity_20"] = hy.diff(20)
    mac["hy_cross_5v20"]  = _z(hy, 5) - _z(hy, 20)
    mac["hy_cross_5v60"]  = _z(hy, 5) - _z(hy, 60)
    mac["hy_cross_20v60"] = _z(hy, 20) - _z(hy, 60)

    # BAA-10Y credit spread (long history since 2000)
    baa = _fred("baa_spread", "baa")
    mac["baa_spread"] = baa
    for d in [5, 20, 60]:
        mac[f"baa_spread_z{d}"] = _z(baa, d)
    mac["baa_spread_velocity"]    = baa.diff(5)
    mac["baa_spread_velocity_20"] = baa.diff(20)
    mac["baa_cross_5v20"]  = _z(baa, 5) - _z(baa, 20)
    mac["baa_cross_5v60"]  = _z(baa, 5) - _z(baa, 60)
    mac["baa_cross_20v60"] = _z(baa, 20) - _z(baa, 60)
    # EMA ratio (used in cash-out logic)
    baa_ema50 = baa.ewm(span=50).mean()
    baa_ema200 = baa.ewm(span=200).mean()
    mac["baa_ema50_200_ratio"] = baa_ema50 / baa_ema200

    yc = _fred("yield_curve", "yc")
    mac["yield_curve"] = yc
    for d in [5, 20, 60]:
        mac[f"yield_curve_z{d}"] = _z(yc, d)
    mac["yield_curve_velocity"]    = yc.diff(5)
    mac["yield_curve_velocity_20"] = yc.diff(20)
    mac["yc_cross_5v20"]  = _z(yc, 5) - _z(yc, 20)
    mac["yc_cross_20v60"] = _z(yc, 20) - _z(yc, 60)

    dxy = _fred("dxy", "dxy")
    for d in [5, 20, 60]:
        mac[f"dxy_ret_{d}d"] = dxy.pct_change(d)
        mac[f"dxy_z{d}"] = _z(dxy, d)
    mac["dxy_cross_5v20"]  = _z(dxy, 5) - _z(dxy, 20)
    mac["dxy_cross_20v60"] = _z(dxy, 20) - _z(dxy, 60)

    # SPX proxy (IVV or CSPX_AS)
    for ticker in ["IVV", "CSPX_AS", "QQQ"]:
        spx_path = DATA / f"{ticker}.parquet"
        if spx_path.exists():
            spx_close = pd.read_parquet(spx_path)
            spx_close = spx_close["close"] if "close" in spx_close.columns else spx_close["adj_close"]
            for d in [5, 20, 60]:
                mac[f"ret_spx_{d}d"] = spx_close.pct_change(d)
            mac["spx_vol_20d"] = spx_close.pct_change(1).rolling(20).std() * np.sqrt(252)
            mac["spx_drawdown_60"] = spx_close / spx_close.rolling(60).max().replace(0, np.nan) - 1
            break

    # ---- Non-linearities: interactions and squared terms ----
    # VIX squared (extreme VIX matters more)
    mac["vix_squared"] = mac.get("vix_level", pd.Series(dtype=float)) ** 2
    # VIX × HY spread interaction
    if "vix_level" in mac.columns and "hy_spread" in mac.columns:
        mac["vix_x_hy"] = mac["vix_level"] * mac["hy_spread"]
    if "vix_level" in mac.columns and "baa_spread" in mac.columns:
        mac["vix_x_baa"] = mac["vix_level"] * mac["baa_spread"]
    # Yield curve × VIX (inverted curve + high VIX = crisis)
    if "yield_curve" in mac.columns and "vix_level" in mac.columns:
        mac["yc_x_vix"] = mac["yield_curve"] * mac["vix_level"]

    return mac


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Building features for {len(UNIVERSE)} ETFs...")

    macro = load_macro()
    flow_path = DATA / "flow_proxies.parquet"
    flow = pd.read_parquet(flow_path) if flow_path.exists() else pd.DataFrame()

    panels = []

    for etf in UNIVERSE:
        etf_id = etf.bourso

        # OHLCV: prefer proxy proxy (longer history), fallback to bourso UCITS
        proxy_file   = DATA / f"{etf.proxy.replace('.', '_')}.parquet"
        bourso_file = DATA / f"{etf.bourso.replace('.', '_')}.parquet"
        if proxy_file.exists():
            ohlcv = pd.read_parquet(proxy_file)
        elif bourso_file.exists():
            ohlcv = pd.read_parquet(bourso_file)
        else:
            print(f"  SKIP {etf_id}: no OHLCV file")
            continue

        ohlcv.index = pd.to_datetime(ohlcv.index).tz_localize(None)

        # iShares shares outstanding (smart-money) — enabled in both modes
        shares_df = None
        ishares_ticker = ISHARES_MAP.get(etf_id)
        if ishares_ticker:
            ish_path = DATA / f"ishares_{ishares_ticker}_hist.parquet"
            if ish_path.exists():
                shares_df = pd.read_parquet(ish_path)
                shares_df.index = pd.to_datetime(shares_df.index).tz_localize(None)

        # Compute per-ETF features
        feat = compute_etf_features(ohlcv, shares_df)

        # Join macro (forward-fill FRED on OHLCV trading days)
        mac_aligned = macro.reindex(feat.index, method="ffill")
        feat = feat.join(mac_aligned, how="left")

        # VIX level at max drawdown dates (expanding + rolling windows)
        if "vix_level" in feat.columns:
            c_etf = ohlcv["close"] if "close" in ohlcv.columns else ohlcv["adj_close"]
            c_etf = c_etf.reindex(feat.index)

            # Expanding (since inception)
            cum_max_etf = c_etf.expanding().max()
            dd_etf = c_etf / cum_max_etf - 1
            expanding_min_dd = dd_etf.expanding().min()
            is_new_max_dd = dd_etf == expanding_min_dd
            feat["vix_at_max_dd"] = feat["vix_level"].where(is_new_max_dd).ffill()

        # Join flow proxies
        if not flow.empty:
            flow_aligned = flow.reindex(feat.index, method="ffill")
            feat = feat.join(flow_aligned, how="left")

        feat["section"] = etf.section
        feat["etf_id"]  = etf_id
        feat.index.name = "date"

        panels.append(feat)
        print(f"  {etf_id:<14}  {len(feat)} rows  shares_z20={'✓' if shares_df is not None else '✗'}")

    if not panels:
        print("ERROR: no panels built")
        return

    # Concatenate into panel
    panel = pd.concat(panels)
    panel = panel.reset_index()  # columns: date, etf_id, section, features...

    # Cross-sectional z-scores within section
    for col in ["ret_5d", "ret_20d", "ret_60d"]:
        if col in panel.columns:
            panel[f"{col}_z_within_block"] = (
                panel.groupby(["date", "section"])[col]
                .transform(lambda x: (x - x.mean()) / max(x.std(), 1e-8))
            )

    # Cross-sectional z-scores vs full universe
    for col in ["ret_1d", "ret_5d", "ret_10d", "ret_20d", "ret_40d", "ret_60d", "ret_120d", "ret_250d",
                "vol_5d", "vol_20d", "vol_60d", "vol_120d",
                "rsi_14", "atr_14", "bb_position_20", "drawdown_60"]:
        if col in panel.columns:
            panel[f"{col}_z_xs"] = (
                panel.groupby("date")[col]
                .transform(lambda x: (x - x.mean()) / max(x.std(), 1e-8))
            )

    # Cross-sectional ranks
    for col in ["ret_5d", "ret_20d", "ret_60d", "ret_120d", "ret_250d",
                "vol_20d", "shares_outstanding_z20"]:
        if col in panel.columns:
            panel[f"{col}_rank"] = panel.groupby("date")[col].rank(pct=True)

    # Momentum acceleration cross-sectional
    for short, long in [("ret_5d", "ret_20d"), ("ret_20d", "ret_60d"),
                         ("ret_60d", "ret_120d"), ("ret_5d", "ret_60d")]:
        col = f"mom_accel_{short.split('_')[1]}v{long.split('_')[1]}"
        if short in panel.columns and long in panel.columns:
            panel[col] = panel[short] - panel[long]
            panel[f"{col}_z_xs"] = (
                panel.groupby("date")[col]
                .transform(lambda x: (x - x.mean()) / max(x.std(), 1e-8))
            )

    # Non-linearities: squared cross-sectional features
    for col in ["ret_20d_z_xs", "ret_60d_z_xs", "ret_120d_z_xs"]:
        if col in panel.columns:
            panel[f"{col}_sq"] = panel[col] ** 2 * np.sign(panel[col])

    # Interaction: momentum × volume (high momentum + high volume = stronger signal)
    if "ret_20d_z_xs" in panel.columns and "vol_20d_z_xs" in panel.columns:
        panel["mom_x_vol_20d"] = panel["ret_20d_z_xs"] * panel["vol_20d_z_xs"]
    if "ret_60d_z_xs" in panel.columns and "vol_60d_z_xs" in panel.columns:
        panel["mom_x_vol_60d"] = panel["ret_60d_z_xs"] * panel["vol_60d_z_xs"]

    # Interaction: smart money × momentum
    if "shares_outstanding_z20" in panel.columns and "ret_20d_z_xs" in panel.columns:
        so_z_xs = panel.groupby("date")["shares_outstanding_z20"].transform(
            lambda x: (x - x.mean()) / max(x.std(), 1e-8)
        )
        panel["so_x_mom_20d"] = so_z_xs * panel["ret_20d_z_xs"]

    # ---------------------------------------------------------------------------
    # Cross-sectorial complex & non-linear features (ETF-by-row model)
    # ---------------------------------------------------------------------------

    # --- Dispersion / breadth features ---
    # Universe momentum dispersion (high = regime uncertainty)
    for d in [20, 60]:
        col = f"ret_{d}d"
        if col in panel.columns:
            panel[f"universe_disp_{d}d"] = panel.groupby("date")[col].transform("std")
            panel[f"universe_mean_{d}d"] = panel.groupby("date")[col].transform("mean")
            # ETF vs universe mean (relative strength)
            panel[f"ret_{d}d_vs_univ"] = panel[col] - panel[f"universe_mean_{d}d"]

    # Breadth: fraction of universe with positive momentum
    for d in [20, 60]:
        col = f"ret_{d}d"
        if col in panel.columns:
            panel[f"breadth_pos_{d}d"] = panel.groupby("date")[col].transform(
                lambda x: (x > 0).mean()
            )

    # --- Relative value features (rank-based non-linearities) ---
    # Rank momentum reversal: extreme rank → mean reversion signal
    for col in ["ret_20d_rank", "ret_60d_rank", "ret_120d_rank"]:
        if col in panel.columns:
            # Squared distance from median rank (captures extremes)
            panel[f"{col}_extreme"] = (panel[col] - 0.5) ** 2
            # Cubic: signed extreme (captures asymmetry)
            panel[f"{col}_cubic"] = (panel[col] - 0.5) ** 3

    # --- Multi-factor interactions (non-linear) ---
    # Momentum × volatility × smart money (triple interaction)
    if all(c in panel.columns for c in ["ret_20d_z_xs", "vol_20d_z_xs"]):
        if "shares_outstanding_z20" in panel.columns:
            so_z = panel.groupby("date")["shares_outstanding_z20"].transform(
                lambda x: (x - x.mean()) / max(x.std(), 1e-8)
            )
            panel["mom_x_vol_x_so_20d"] = panel["ret_20d_z_xs"] * panel["vol_20d_z_xs"] * so_z

    # Momentum × drawdown (buying dips with momentum)
    if "ret_20d_z_xs" in panel.columns and "drawdown_60" in panel.columns:
        dd_z = panel.groupby("date")["drawdown_60"].transform(
            lambda x: (x - x.mean()) / max(x.std(), 1e-8)
        )
        panel["mom_x_dd_60"] = panel["ret_20d_z_xs"] * dd_z

    # RSI × momentum (overbought/oversold confirmation)
    if "rsi_14" in panel.columns and "ret_20d_z_xs" in panel.columns:
        rsi_z = panel.groupby("date")["rsi_14"].transform(
            lambda x: (x - x.mean()) / max(x.std(), 1e-8)
        )
        panel["rsi_x_mom_20d"] = rsi_z * panel["ret_20d_z_xs"]

    # --- Regime-conditional features ---
    # VIX regime × momentum (momentum less reliable in high VIX)
    if "vix_level" in panel.columns and "ret_20d_z_xs" in panel.columns:
        panel["mom_x_vix"] = panel["ret_20d_z_xs"] * panel["vix_level"]
    if "vix_level" in panel.columns and "ret_60d_z_xs" in panel.columns:
        panel["mom60_x_vix"] = panel["ret_60d_z_xs"] * panel["vix_level"]

    # Yield curve regime × momentum (inverted curve = defensive)
    if "yield_curve" in panel.columns and "ret_20d_z_xs" in panel.columns:
        panel["mom_x_yc"] = panel["ret_20d_z_xs"] * panel["yield_curve"]

    # DXY × EM momentum (strong dollar hurts EM)
    if "dxy_ret_20d" in panel.columns and "ret_20d_z_xs" in panel.columns:
        panel["mom_x_dxy"] = panel["ret_20d_z_xs"] * panel["dxy_ret_20d"]

    # --- Cross-sectoral flow divergence ---
    # Section mean momentum vs universe mean (sector rotation signal)
    for d in [20, 60]:
        col = f"ret_{d}d"
        if col in panel.columns:
            section_mean = panel.groupby(["date", "section"])[col].transform("mean")
            univ_mean = panel.groupby("date")[col].transform("mean")
            panel[f"section_vs_univ_{d}d"] = section_mean - univ_mean
            # ETF vs its own section
            panel[f"ret_{d}d_vs_section"] = panel[col] - section_mean

    # --- Volatility regime interaction ---
    # Vol z-score × dispersion (concentrated bets when dispersion high + vol low)
    if "vol_20d_z_xs" in panel.columns and "universe_disp_20d" in panel.columns:
        panel["vol_x_disp_20d"] = panel["vol_20d_z_xs"] * panel["universe_disp_20d"]

    # --- Smart money flow divergence vs price ---
    # SO increasing but price falling (accumulation) or SO decreasing + price up (distribution)
    if "shares_outstanding_z20" in panel.columns and "ret_20d" in panel.columns:
        so_z2 = panel.groupby("date")["shares_outstanding_z20"].transform(
            lambda x: (x - x.mean()) / max(x.std(), 1e-8)
        )
        ret_z2 = panel.groupby("date")["ret_20d"].transform(
            lambda x: (x - x.mean()) / max(x.std(), 1e-8)
        )
        panel["so_price_divergence"] = so_z2 - ret_z2  # positive = accumulation

    # --- Mean reversion signals ---
    # Distance from 52-week high × momentum (reversal from extremes)
    if "drawdown_250" in panel.columns and "ret_5d_z_xs" in panel.columns:
        panel["dd250_x_shortmom"] = panel["drawdown_250"] * panel["ret_5d_z_xs"]

    # Bollinger extreme × volume surge
    if "bb_position_20" in panel.columns and "volume_z5" in panel.columns:
        bb_z = panel.groupby("date")["bb_position_20"].transform(
            lambda x: (x - x.mean()) / max(x.std(), 1e-8)
        )
        vol_z = panel.groupby("date")["volume_z5"].transform(
            lambda x: (x - x.mean()) / max(x.std(), 1e-8)
        )
        panel["bb_x_volsurge"] = bb_z * vol_z

    # ---------------------------------------------------------------------------
    # Iteration 2: Temporal, correlation, higher-order, regime features
    # ---------------------------------------------------------------------------

    # --- Momentum consistency (sign persistence) ---
    # How many of last N periods had positive returns (trend consistency)
    for d, step in [(20, 5), (60, 20)]:
        col = f"ret_{step}d"
        if col in panel.columns:
            # Compute rolling sign consistency per ETF (done per-ETF already)
            pass  # done below after groupby

    # --- Volatility-adjusted momentum (Sharpe-like per ETF) ---
    for d in [20, 60, 120]:
        ret_col = f"ret_{d}d"
        vol_col = f"vol_{d}d"
        if ret_col in panel.columns and vol_col in panel.columns:
            panel[f"sharpe_{d}d"] = panel[ret_col] / panel[vol_col].replace(0, np.nan)
            # Cross-sectional z-score of sharpe
            panel[f"sharpe_{d}d_z_xs"] = panel.groupby("date")[f"sharpe_{d}d"].transform(
                lambda x: (x - x.mean()) / max(x.std(), 1e-8)
            )

    # --- Momentum acceleration (2nd derivative) ---
    # ret_20d change over 20d = momentum of momentum
    for d in [20, 60]:
        col = f"ret_{d}d"
        if col in panel.columns:
            panel[f"mom2_{d}d"] = panel.groupby("etf_id")[col].diff(d)
            panel[f"mom2_{d}d_z_xs"] = panel.groupby("date")[f"mom2_{d}d"].transform(
                lambda x: (x - x.mean()) / max(x.std(), 1e-8)
            )

    # --- Correlation regime features ---
    # Rolling correlation between each ETF and market (IVV/universe mean)
    if "ret_5d" in panel.columns and "universe_mean_20d" in panel.columns:
        # ETF beta proxy: ret vs universe mean (rolling estimated at panel level)
        panel["ret_vs_mkt"] = panel["ret_5d"] - panel.groupby("date")["ret_5d"].transform("mean")

    # --- Relative vol regime ---
    # ETF vol vs universe vol (low vol = defensive, high vol = aggressive)
    if "vol_20d" in panel.columns:
        panel["vol_20d_vs_univ"] = panel["vol_20d"] - panel.groupby("date")["vol_20d"].transform("mean")
        panel["vol_ratio_vs_univ"] = panel["vol_20d"] / panel.groupby("date")["vol_20d"].transform("mean").replace(0, np.nan)

    # --- Time-in-drawdown features ---
    if "drawdown_60" in panel.columns:
        # How deep in drawdown (already have drawdown_60, drawdown_250)
        # Drawdown severity: drawdown^2 (penalize deep drawdowns more)
        panel["dd_60_severity"] = panel["drawdown_60"] ** 2 * np.sign(panel["drawdown_60"])
    if "drawdown_250" in panel.columns:
        panel["dd_250_severity"] = panel["drawdown_250"] ** 2 * np.sign(panel["drawdown_250"])

    # --- Cross-sectional momentum spread ---
    # Spread between top and bottom quartile returns (market dispersion signal)
    for d in [20, 60]:
        col = f"ret_{d}d"
        if col in panel.columns:
            q75 = panel.groupby("date")[col].transform(lambda x: x.quantile(0.75))
            q25 = panel.groupby("date")[col].transform(lambda x: x.quantile(0.25))
            panel[f"mom_spread_{d}d"] = q75 - q25
            # ETF position within the spread
            panel[f"pos_in_spread_{d}d"] = (panel[col] - q25) / (q75 - q25).replace(0, np.nan)

    # --- Trend following signals ---
    # Price vs exponential MA (faster reaction than SMA)
    for d in [10, 20, 50]:
        ema_col = f"ema_{d}"
        if ema_col not in panel.columns:
            # EMA computed per ETF earlier, but we don't have it in panel
            # Use price_vs_ma as proxy — already computed
            pass

    # --- Regime change detection ---
    # VIX acceleration (2nd derivative)
    if "vix_level" in panel.columns:
        panel["vix_accel"] = panel.groupby("date")["vix_level"].transform(
            lambda x: x  # same value for all ETFs on a date, just keep it
        )
        # Already have vix_velocity, add acceleration
    if "vix_velocity" in panel.columns:
        vix_vel = panel.groupby("etf_id")["vix_velocity"].diff(5)
        panel["vix_accel_5d"] = vix_vel

    # --- Smart money consensus ---
    # Fraction of universe with positive SO change
    if "shares_outstanding_z20" in panel.columns:
        panel["so_breadth_pos"] = panel.groupby("date")["shares_outstanding_z20"].transform(
            lambda x: (x > 0).mean()
        )
        # ETF SO vs universe mean SO (relative flow)
        panel["so_vs_univ"] = panel["shares_outstanding_z20"] - panel.groupby("date")["shares_outstanding_z20"].transform("mean")

    # --- Interaction: Sharpe × Smart money ---
    if "sharpe_20d_z_xs" in panel.columns and "shares_outstanding_z20" in panel.columns:
        so_z3 = panel.groupby("date")["shares_outstanding_z20"].transform(
            lambda x: (x - x.mean()) / max(x.std(), 1e-8)
        )
        panel["sharpe_x_so_20d"] = panel["sharpe_20d_z_xs"] * so_z3

    # --- Interaction: momentum acceleration × vol ---
    if "mom2_20d_z_xs" in panel.columns and "vol_20d_z_xs" in panel.columns:
        panel["mom2_x_vol_20d"] = panel["mom2_20d_z_xs"] * panel["vol_20d_z_xs"]

    # --- Relative strength persistence ---
    # Rank autocorrelation: is ETF consistently ranked high/low?
    for d in [20, 60]:
        rank_col = f"ret_{d}d_rank"
        if rank_col in panel.columns:
            # Lagged rank (20d ago)
            panel[f"rank_lag_{d}d"] = panel.groupby("etf_id")[rank_col].shift(d)
            # Rank persistence = current rank - lagged rank (positive = improving)
            panel[f"rank_change_{d}d"] = panel[rank_col] - panel[f"rank_lag_{d}d"]

    # Label: absolute forward 10d return (brut) — used by the Smart Money model
    panel["label"] = panel["ret_10d_fwd"]

    # Soft smart-money activation: keep all OHLCV rows but report when each ETF
    # gains its first valid shares_outstanding_z20 value. XGBoost handles
    # missing smart-money columns via `missing=NaN` — an ETF with no SM yet
    # still participates in the cross-section through its technical/macro
    # features. The IC gate in backtest.py prevents the model from being
    # deployed before enough ETFs are smart-money-active.
    if "shares_outstanding_z20" in panel.columns:
        first_sm_date = (
            panel.dropna(subset=["shares_outstanding_z20"])
                 .groupby("etf_id")["date"].min()
        )
        print("\nSmart-money activation schedule (first 5 / last 5):")
        ordered = first_sm_date.sort_values()
        for etf, d in pd.concat([ordered.head(5), ordered.tail(5)]).items():
            print(f"  {etf:<10s} {d.date()}")

    # Set MultiIndex
    panel = panel.set_index(["date", "etf_id"]).sort_index()

    out = DATA / "features.parquet"
    panel.to_parquet(out)
    valid = panel["label"].notna().sum()
    print(f"\nSaved {out.name}: {panel.shape[0]} rows × {panel.shape[1]} cols  ({valid} labeled)")
    print(f"Date range: {panel.index.get_level_values('date').min().date()} → "
          f"{panel.index.get_level_values('date').max().date()}")


if __name__ == "__main__":
    main()
