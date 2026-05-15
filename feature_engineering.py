"""
Feature engineering for MyQTM-ETF portfolio model.

Builds a panel DataFrame with MultiIndex (date, etf_id) containing:
  - Technical features per ETF (momentum, volatility, RSI, MA, ATR, volume)
  - Macro/regime features from FRED (VIX, HY spread, yield curve, DXY)
  - Smart money: shares_outstanding_z20 from iShares XLS
  - Cross-sectional: ret_20d/ret_5d z-score within section block
  - Label: forward ret_20d[etf_i] - mean(ret_20d[universe])  (no lookahead)

Output: data/features.parquet
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from etf import UNIVERSE

DATA = Path(__file__).parent / "data"

# Map each ETF bourso ticker → iShares XLS ticker (for shares_outstanding)
ISHARES_MAP = {
    "CSP1.PA":  "IVV",   "CNX1.PA":  "IUIT",  "WPEA.PA":  "ACWI",
    "IEMA.AS":  "EEM",   "CSKR.PA":  "EWY",   "ITWN.PA":  "EWT",
    "IFFI.AS":  "EEMA",  "EXCH.AS":  "EMXC",  "SJPE.AS":  "EWJ",
    "IBZL.AS":  "EWZ",   "IMEX.AS":  "EWW",   "ICAU.AS":  "EWC",
    "ITKY.AS":  "TUR",   "ISF.L":    "ISF",   "FXC.AS":   "FXI",
    "IUIT.AS":  "IUIT",  "IUES.AS":  "IUES",  "IUII.AS":  "IUII",
    "IUCD.AS":  "IUCD",  "IUHC.AS":  "IUHC",  "IUCS.AS":  "IUCS",
    "IUFS.AS":  "IUFS",  "EXX1.DE":  "EXX1",  "EXV1.DE":  "EXV1",
    "SEMI.AS":  "SOXX",  "AINF.PA":  "AINF",  "IART.PA":  "IART",
    "ECAR.AS":  "ECAR",  "INRA.AS":  "ICLN",  "CITY.AS":  "CITY",
    "IQQQ.DE":  "IQQQ",  "IGLN.AS":  "IAU",   "SXRS.DE":  "GSG",
    "RING":     "RING",  "IOGP.AS":  "IEO",   "DTLA.AS":  "TLT",
    "IBTA.AS":  "IEF",   "IHYU.AS":  "HYG",   "ITPS.AS":  "TIP",
    "IBTC.AS":  "IBIT",
    "QQQ":      "IUIT",   # US Nasdaq proxy → IUIT smart money
    "GLD":      "IAU",    # US Gold proxy → IAU smart money
    "IVV":      "IVV",    # S&P 500 — direct smart money since 2000
    "SOXX":     "SOXX",   # Semiconductors — smart money since 2001
    "EEM":      "EEM",    # Emerging Markets — smart money since 2003
    "TLT":      "TLT",    # Treasury 20y+ — smart money since 2002
    "IEO":      "IEO",    # Oil & Gas — smart money since 2006
    "EXX1.DE":  "EXX1",   # Euro Banks — smart money since 2002
    "TIP":      "TIP",    # TIPS inflation — smart money since 2003
    "EWJ":      "EWJ",    # Japan — smart money since 1996
    "EXV1.DE":  "EXV1",   # Europe Tech — smart money since 2002
    "EWC":      "EWC",    # Canada — smart money since 1996
    "EWZ":      "EWZ",    # Brazil — smart money since 2000
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
    # Use 'close' if available; some FMP files have 'adj_close' instead
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

    # Forward returns (label candidates) — shifted BACK N days (no lookahead)
    f["ret_10d_fwd"] = c.pct_change(10).shift(-10)
    f["ret_20d_fwd"] = c.pct_change(20).shift(-20)
    f["ret_90d_fwd"] = c.pct_change(90).shift(-90)

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

        # OHLCV: prefer fmp proxy (longer history), fallback to bourso UCITS
        fmp_file   = DATA / f"{etf.fmp.replace('.', '_')}.parquet"
        bourso_file = DATA / f"{etf.bourso.replace('.', '_')}.parquet"
        if fmp_file.exists():
            ohlcv = pd.read_parquet(fmp_file)
        elif bourso_file.exists():
            ohlcv = pd.read_parquet(bourso_file)
        else:
            print(f"  SKIP {etf_id}: no OHLCV file")
            continue

        ohlcv.index = pd.to_datetime(ohlcv.index).tz_localize(None)

        # iShares shares outstanding
        ishares_ticker = ISHARES_MAP.get(etf_id)
        shares_df = None
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

    # Label: forward ret_20d vs universe mean on same date
    # Label: absolute forward return (no demeaning for single-ETF mode)
    if panel["etf_id"].nunique() == 1:
        panel["label"] = panel["ret_90d_fwd"]
    else:
        panel["label"] = panel.groupby("date")["ret_90d_fwd"].transform(
            lambda x: x - x.mean()
        )

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
