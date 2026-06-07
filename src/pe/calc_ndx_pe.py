"""
Calculate historical cap-weighted PE for the top-20 NASDAQ-100 constituents.

Downloads quarterly key-metrics (PE + market cap) from FMP API,
caches everything under data/pe/, and produces a time-series chart.

Usage:  python src/pe/calc_ndx_pe.py
"""

import json, os, time
from pathlib import Path

import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ── Config ────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
PE_DIR = ROOT / "data" / "pe"
PE_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR = ROOT / "outputs" / "pe"
OUT_DIR.mkdir(parents=True, exist_ok=True)

API_KEY = os.environ.get("FMPAPI") or os.environ.get("FMP_APIKEY")
if not API_KEY:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    API_KEY = os.environ.get("FMPAPI") or os.environ.get("FMP_APIKEY")
assert API_KEY, "Set FMPAPI or FMP_APIKEY in .env"

BASE = "https://financialmodelingprep.com/api/v3"
TOP_N = 20
RATE_LIMIT_SLEEP = 0.25  # seconds between API calls


# ── 1. Get current NASDAQ-100 constituents sorted by market cap ───
def fetch_top_constituents(n=TOP_N):
    cache = PE_DIR / "ndx_constituents.json"

    # Get constituent list
    url = f"{BASE}/nasdaq_constituent?apikey={API_KEY}"
    r = requests.get(url)
    r.raise_for_status()
    constituents = r.json()
    symbols = [c["symbol"] for c in constituents]

    # Get current profiles (for market cap ranking)
    profiles = []
    for i in range(0, len(symbols), 50):
        batch = ",".join(symbols[i:i+50])
        url = f"{BASE}/profile/{batch}?apikey={API_KEY}"
        resp = requests.get(url)
        resp.raise_for_status()
        profiles.extend(resp.json())
        time.sleep(RATE_LIMIT_SLEEP)

    # Sort by market cap, take top N
    profiles.sort(key=lambda x: x.get("mktCap", 0) or 0, reverse=True)

    # Deduplicate (GOOGL/GOOG → keep GOOGL)
    seen_names = set()
    top = []
    for p in profiles:
        name = p.get("companyName", "")
        if name in seen_names:
            continue
        seen_names.add(name)
        top.append({
            "symbol": p["symbol"],
            "name": name[:40],
            "mktCap": p.get("mktCap", 0),
            "sector": p.get("sector", ""),
        })
        if len(top) >= n:
            break

    with open(cache, "w") as f:
        json.dump(top, f, indent=2)
    print(f"Top {len(top)} constituents saved → {cache.name}")
    return top


# ── 2. Download quarterly key-metrics per ticker ─────────
def fetch_quarterly_metrics(symbol):
    cache = PE_DIR / f"{symbol}_quarterly.json"
    if cache.exists():
        with open(cache) as f:
            data = json.load(f)
        print(f"  {symbol}: loaded {len(data)} quarters from cache")
        return data

    url = f"{BASE}/key-metrics/{symbol}?period=quarter&limit=200&apikey={API_KEY}"
    r = requests.get(url)
    r.raise_for_status()
    data = r.json()

    with open(cache, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  {symbol}: downloaded {len(data)} quarters → {cache.name}")
    time.sleep(RATE_LIMIT_SLEEP)
    return data


# ── 3. Build quarterly PE + market cap DataFrame per ticker ───
def build_ticker_df(symbol, raw_data):
    rows = []
    for q in raw_data:
        date = q.get("date")
        pe = q.get("peRatio")
        mcap = q.get("marketCap")
        if date and mcap and mcap > 0:
            rows.append({
                "date": pd.Timestamp(date),
                "symbol": symbol,
                "pe": pe,
                "marketCap": mcap,
            })
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df
    df = df.sort_values("date").set_index("date")
    return df


# ── 4. Calculate cap-weighted PE per quarter ──────────────
def calc_weighted_pe(all_dfs):
    # Merge all tickers into a single DataFrame
    combined = pd.concat(all_dfs.values(), axis=0).reset_index()
    combined = combined.sort_values("date")

    # Group by quarter
    combined["quarter"] = combined["date"].dt.to_period("Q")
    quarters = sorted(combined["quarter"].unique())

    results = []
    for q in quarters:
        qdata = combined[combined["quarter"] == q].copy()

        # Drop tickers with missing or negative PE for weighted calc
        # but keep them for "with outliers" version
        total_mcap = qdata["marketCap"].sum()
        n_tickers = len(qdata)

        # Cap-weighted PE (all tickers with valid PE)
        valid = qdata[qdata["pe"].notna() & (qdata["pe"] > 0)]
        if len(valid) == 0:
            continue

        weights = valid["marketCap"] / valid["marketCap"].sum()
        weighted_pe = (valid["pe"] * weights).sum()

        # Median PE (unweighted)
        median_pe = valid["pe"].median()

        # Blended PE = total market cap / total earnings
        # earnings_i = marketCap_i / pe_i
        valid_earnings = valid.copy()
        valid_earnings["earnings"] = valid_earnings["marketCap"] / valid_earnings["pe"]
        blended_pe = valid_earnings["marketCap"].sum() / valid_earnings["earnings"].sum()

        results.append({
            "date": q.start_time,
            "quarter": str(q),
            "weighted_pe": weighted_pe,
            "blended_pe": blended_pe,
            "median_pe": median_pe,
            "n_tickers": n_tickers,
            "n_valid_pe": len(valid),
            "total_mcap_B": total_mcap / 1e9,
        })

    df = pd.DataFrame(results).set_index("date")
    return df


# ── 5. Plot ───────────────────────────────────────────────
def plot_pe_history(pe_df, tickers_info):
    # Load NDX price
    ndx = pd.read_parquet(ROOT / "data" / "^NDX.parquet")
    ndx_price = ndx["close"] if "close" in ndx.columns else ndx.iloc[:, 0]

    fig, (ax_price, ax_pe) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[2, 3],
                                           gridspec_kw={"hspace": 0.08}, sharex=True)

    # Top: NASDAQ-100 price (log)
    ax_price.semilogy(ndx_price.index, ndx_price.values, "k-", lw=1.5, label="NASDAQ-100")
    ax_price.set_ylabel("NDX Price (log)")
    ax_price.legend(loc="upper left", fontsize=9)
    ax_price.grid(True, alpha=0.3)

    tickers_str = ", ".join([t["symbol"] for t in tickers_info[:10]])
    ax_price.set_title(f"NASDAQ-100 Price vs Top-{len(tickers_info)} PE Ratios (quarterly, log)\n"
                       f"{tickers_str}...", fontsize=13)

    # Bottom: PE ratios — log scale
    ax_pe.semilogy(pe_df.index, pe_df["weighted_pe"], "b-", lw=2, label="Cap-weighted PE")
    ax_pe.semilogy(pe_df.index, pe_df["blended_pe"], "r-", lw=1.5, alpha=0.5, label="Blended PE")
    ax_pe.semilogy(pe_df.index, pe_df["median_pe"], "g--", lw=1, alpha=0.4, label="Median PE")

    # Reference lines
    for val, col, lbl in [(20, "green", "PE=20"), (35, "orange", "PE=35"),
                           (50, "red", "PE=50"), (100, "darkred", "PE=100")]:
        ax_pe.axhline(val, color=col, ls=":", alpha=0.4, lw=1)
        ax_pe.text(pe_df.index[0], val * 1.05, lbl, fontsize=7, color=col, alpha=0.7)

    ymax = pe_df["weighted_pe"].max() * 1.5
    ax_pe.set_ylim(5, ymax)
    ax_pe.set_ylabel("PE Ratio (log)")
    ax_pe.legend(loc="upper left", fontsize=9)
    ax_pe.grid(True, alpha=0.3)

    # Crisis shading on both axes
    crises = [
        ("2000-03-01", "2002-10-01", "Dot-com"),
        ("2007-10-01", "2009-03-01", "GFC"),
        ("2020-02-01", "2020-04-01", "Covid"),
        ("2022-01-01", "2022-10-01", "Rate hike"),
    ]
    for start, end, label in crises:
        for ax in [ax_price, ax_pe]:
            ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color="red", alpha=0.08)
        if label:
            ax_price.text(pd.Timestamp(start), ax_price.get_ylim()[1] * 0.7, label,
                          fontsize=8, color="red", alpha=0.7)

    ax_pe.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    out = OUT_DIR / "ndx_top20_pe_history.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nChart saved → {out}")
    plt.close()


# ── 6. Plot QQQ + PE + CAPE combined ─────────────────────
def plot_qqq_pe_cape(pe_df):
    qqq = pd.read_parquet(ROOT / "data" / "QQQ.parquet")
    qqq_price = qqq["close"] if "close" in qqq.columns else qqq.iloc[:, 0]

    cape = pd.read_parquet(ROOT / "data" / "shiller_cape.parquet")
    cape_col = cape.columns[0]
    cape_val = cape[cape_col].replace(0, np.nan).dropna()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[2, 3],
                                    gridspec_kw={"hspace": 0.12}, sharex=True)

    # Top: QQQ price (log scale)
    ax1.semilogy(qqq_price.index, qqq_price.values, "k-", lw=1.5, label="QQQ")
    ax1.set_ylabel("QQQ Price (log)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.set_title("QQQ Price vs NASDAQ-100 PE (FMP top-20) vs Shiller CAPE (S&P 500)", fontsize=13)

    # Bottom: PE ratios
    ax2.plot(pe_df.index, pe_df["blended_pe"], "b-", lw=2, label="NDX Top-20 Blended PE (FMP)")
    ax2.plot(pe_df.index, pe_df["weighted_pe"], "b--", lw=1, alpha=0.4, label="NDX Top-20 Weighted PE")
    ax2.plot(pe_df.index, pe_df["median_pe"], "c-", lw=1, alpha=0.5, label="NDX Top-20 Median PE")
    ax2.plot(cape_val.index, cape_val.values, "r-", lw=2, label="Shiller CAPE (S&P 500)")

    ax2.axhline(20, color="green", ls=":", alpha=0.5, lw=1)
    ax2.axhline(25, color="gray", ls=":", alpha=0.5, lw=1)
    ax2.axhline(35, color="orange", ls=":", alpha=0.5, lw=1)

    crises = [
        ("2000-03-01", "2002-10-01", "Dot-com"),
        ("2007-10-01", "2009-03-01", "GFC"),
        ("2020-02-01", "2020-04-01", "Covid"),
        ("2022-01-01", "2022-10-01", "Rate hike"),
    ]
    for start, end, label in crises:
        for ax in [ax1, ax2]:
            ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color="red", alpha=0.08)
        ax1.text(pd.Timestamp(start), ax1.get_ylim()[1] * 0.85, label,
                 fontsize=8, color="red", alpha=0.7)

    ax2.set_ylabel("PE Ratio")
    ax2.set_ylim(0, 80)
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.set_xlim(pd.Timestamp("1998-01-01"), pd.Timestamp("2026-07-01"))

    out = OUT_DIR / "qqq_vs_pe_vs_cape.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Chart saved → {out}")
    plt.close()


# ── Main ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== NASDAQ-100 Top-20 Historical Cap-Weighted PE ===\n")

    # 1. Get top constituents
    top = fetch_top_constituents(TOP_N)
    for i, t in enumerate(top, 1):
        print(f"  {i:2d}. {t['symbol']:<7s} {t['name']:<35s} ${t['mktCap']/1e9:.0f}B")

    # 2. Download quarterly data
    print(f"\nDownloading quarterly key-metrics...")
    all_dfs = {}
    for t in top:
        sym = t["symbol"]
        raw = fetch_quarterly_metrics(sym)
        df = build_ticker_df(sym, raw)
        if len(df) > 0:
            all_dfs[sym] = df
            print(f"    → {sym}: {df.index.min().date()} to {df.index.max().date()} ({len(df)} quarters)")

    # 3. Calculate weighted PE
    print(f"\nCalculating cap-weighted PE...")
    pe_df = calc_weighted_pe(all_dfs)
    pe_df.to_csv(PE_DIR / "ndx_top20_pe_quarterly.csv")
    print(f"Saved → ndx_top20_pe_quarterly.csv ({len(pe_df)} quarters)")
    print(f"Period: {pe_df.index.min().date()} → {pe_df.index.max().date()}")

    # Summary
    print(f"\n{'Quarter':<10s} {'Weighted':>9s} {'Blended':>9s} {'Median':>8s} {'#':>3s}")
    print("-" * 45)
    for _, row in pe_df.tail(12).iterrows():
        print(f"{row['quarter']:<10s} {row['weighted_pe']:8.1f}  {row['blended_pe']:8.1f}  "
              f"{row['median_pe']:7.1f}  {row['n_valid_pe']:3.0f}")

    # Current
    latest = pe_df.iloc[-1]
    print(f"\nActuel ({latest['quarter']}):")
    print(f"  Cap-weighted PE : {latest['weighted_pe']:.1f}")
    print(f"  Blended PE      : {latest['blended_pe']:.1f}")
    print(f"  Median PE       : {latest['median_pe']:.1f}")

    # 4. Plot
    plot_pe_history(pe_df, top)
    plot_qqq_pe_cape(pe_df)
