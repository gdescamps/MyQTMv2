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
    "CSP1.PA":  "IVV",   "CNX1.PA":  "CNX1",  "WPEA.PA":  "ACWI",
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

    # Momentum
    f["ret_1d"]  = c.pct_change(1)
    f["ret_5d"]  = c.pct_change(5)
    f["ret_20d"] = c.pct_change(20)
    f["ret_60d"] = c.pct_change(60)

    # Volatility (annualised)
    r1 = c.pct_change(1)
    f["vol_20d"] = r1.rolling(20, min_periods=15).std() * np.sqrt(252)
    f["vol_60d"] = r1.rolling(60, min_periods=40).std() * np.sqrt(252)

    # RSI
    f["rsi_14"] = _rsi(c, 14)

    # Moving averages
    ma20  = c.rolling(20,  min_periods=15).mean()
    ma50  = c.rolling(50,  min_periods=40).mean()
    ma200 = c.rolling(200, min_periods=150).mean()
    f["price_vs_ma50"]  = c / ma50.replace(0, np.nan) - 1
    f["price_vs_ma200"] = c / ma200.replace(0, np.nan) - 1
    f["ma_20_slope"]    = ma20.pct_change(5)
    f["ma_50_slope"]    = ma50.pct_change(20)

    # ATR normalised
    f["atr_14"] = _atr_norm(h, l, c, 14)

    # Volume dollar z-scores
    dvol = c * v
    f["volume_z5"]  = _z(dvol, 5)
    f["volume_z20"] = _z(dvol, 20)
    f["volume_z60"] = _z(dvol, 60)

    # Shares outstanding z-scores from iShares XLS (smart money flows)
    if shares_df is not None and "shares_outstanding" in shares_df.columns:
        so = shares_df["shares_outstanding"].reindex(c.index, method="ffill")
        so_chg = so.diff(1)
        f["shares_outstanding_z5"]  = _z(so_chg, 5)
        f["shares_outstanding_z20"] = _z(so_chg, 20)
        f["shares_outstanding_z60"] = _z(so_chg, 60)
    else:
        f["shares_outstanding_z5"]  = np.nan
        f["shares_outstanding_z20"] = np.nan
        f["shares_outstanding_z60"] = np.nan

    # Forward return (label component) — shifted BACK 20 days (no lookahead)
    f["ret_20d_fwd"] = c.pct_change(20).shift(-20)

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
    mac["vix_reversion_force"] = _z(vix, 60) * -1   # positive = vix above norm (dangerous)
    mac["vix_z5"]              = _z(vix, 5)
    mac["vix_z20"]             = _z(vix, 20)
    mac["vix_z60"]             = _z(vix, 60)

    hy = _fred("hy_spread", "hy")
    mac["hy_spread"]          = hy
    mac["hy_spread_z5"]       = _z(hy, 5)
    mac["hy_spread_z20"]      = _z(hy, 20)
    mac["hy_spread_z60"]      = _z(hy, 60)
    mac["hy_spread_velocity"] = hy.diff(5)

    yc = _fred("yield_curve", "yc")
    mac["yield_curve"]          = yc
    mac["yield_curve_velocity"] = yc.diff(20)

    dxy = _fred("dxy", "dxy")
    mac["dxy_ret_20d"] = dxy.pct_change(20)
    mac["dxy_z60"]     = _z(dxy, 60)

    # SPX proxy (IVV or CSPX_AS)
    for ticker in ["IVV", "CSPX_AS", "QQQ"]:
        spx_path = DATA / f"{ticker}.parquet"
        if spx_path.exists():
            spx_close = pd.read_parquet(spx_path)
            spx_close = spx_close["close"] if "close" in spx_close.columns else spx_close["adj_close"]
            mac["ret_spx_20d"] = spx_close.pct_change(20)
            break

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

    # Cross-sectional z-scores within section (same date, same section)
    for col in ["ret_20d", "ret_5d"]:
        panel[f"{col}_z_within_block"] = (
            panel.groupby(["date", "section"])[col]
            .transform(lambda x: (x - x.mean()) / max(x.std(), 1e-8))
        )

    # Label: forward ret_20d vs universe mean on same date
    panel["label"] = panel.groupby("date")["ret_20d_fwd"].transform(
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
