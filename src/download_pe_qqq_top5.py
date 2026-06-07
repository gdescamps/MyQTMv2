"""
Download & interpolate daily cap-weighted PE for the top-5 NASDAQ-100 constituents.

Downloads quarterly key-metrics (PE + market cap) from FMP API,
interpolates daily using price, saves to data/pe/.

Usage:  python src/download_pe_qqq_top5.py
"""

import json, os, time
from pathlib import Path

import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ── Config ────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
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
TOP_PE = 5
RATE_LIMIT_SLEEP = 0.25


# ── 1. Get current NASDAQ-100 constituents sorted by market cap ───
def fetch_top_constituents(n=TOP_N):
    cache = PE_DIR / "ndx_constituents.json"
    url = f"{BASE}/nasdaq_constituent?apikey={API_KEY}"
    r = requests.get(url)
    r.raise_for_status()
    symbols = [c["symbol"] for c in r.json()]

    profiles = []
    for i in range(0, len(symbols), 50):
        batch = ",".join(symbols[i:i+50])
        resp = requests.get(f"{BASE}/profile/{batch}?apikey={API_KEY}")
        resp.raise_for_status()
        profiles.extend(resp.json())
        time.sleep(RATE_LIMIT_SLEEP)

    profiles.sort(key=lambda x: x.get("mktCap", 0) or 0, reverse=True)
    seen_names = set()
    top = []
    for p in profiles:
        name = p.get("companyName", "")
        if name in seen_names:
            continue
        seen_names.add(name)
        top.append({"symbol": p["symbol"], "name": name[:40],
                     "mktCap": p.get("mktCap", 0), "sector": p.get("sector", "")})
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
            rows.append({"date": pd.Timestamp(date), "symbol": symbol,
                         "pe": pe, "marketCap": mcap})
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df
    return df.sort_values("date").set_index("date")


# ── 4. Calculate quarterly cap-weighted PE (top-N) ────────
def calc_quarterly_pe(all_dfs, top_n=TOP_PE):
    combined = pd.concat(all_dfs.values(), axis=0).reset_index()
    combined = combined.sort_values("date")
    combined["quarter"] = combined["date"].dt.to_period("Q")

    results = []
    for q in sorted(combined["quarter"].unique()):
        qdata = combined[combined["quarter"] == q]
        valid = qdata[qdata["pe"].notna() & (qdata["pe"] > 0)]
        if len(valid) == 0:
            continue
        valid = valid.sort_values("marketCap", ascending=False).head(top_n)
        w = valid["marketCap"] / valid["marketCap"].sum()
        results.append({"date": q.start_time, "quarter": str(q),
                        "pe_top5": (valid["pe"] * w).sum(),
                        "n_valid": len(valid)})

    return pd.DataFrame(results).set_index("date")


# ── 5. Interpolate daily PE using price ───────────────────
def interpolate_daily_pe(pe_quarterly):
    """Interpolate quarterly PE to daily using NDX price.

    PE_daily(t) = PE_quarter * (price(t) / price_at_quarter_date)
    Between two quarterly points, EPS is assumed constant.
    """
    ndx = pd.read_parquet(ROOT / "data" / "^NDX.parquet")
    ndx_price = ndx["close"] if "close" in ndx.columns else ndx.iloc[:, 0]

    pe_q = pe_quarterly["pe_top5"].dropna()
    daily_dates = ndx_price.index
    pe_daily = pd.Series(np.nan, index=daily_dates, name="pe_top5_daily")

    # For each quarter, compute EPS_implied = price / PE, then extend daily
    q_dates = pe_q.index.sort_values()

    for i in range(len(q_dates)):
        q_start = q_dates[i]
        q_end = q_dates[i + 1] if i + 1 < len(q_dates) else daily_dates[-1]

        # Find nearest trading day to quarter date
        mask = (daily_dates >= q_start) & (daily_dates < q_end)
        if i + 1 == len(q_dates):
            mask = daily_dates >= q_start

        if mask.sum() == 0:
            continue

        # Price at quarter report date (nearest available)
        near_idx = ndx_price.index.searchsorted(q_start)
        near_idx = min(near_idx, len(ndx_price) - 1)
        price_at_q = ndx_price.iloc[near_idx]
        pe_at_q = pe_q.iloc[i]

        if price_at_q > 0 and pe_at_q > 0:
            # EPS = price / PE (constant for the quarter)
            # PE_daily = price_daily / EPS = price_daily / (price_at_q / PE_at_q)
            #          = PE_at_q * (price_daily / price_at_q)
            pe_daily[mask] = pe_at_q * (ndx_price[mask] / price_at_q)

    pe_daily = pe_daily.dropna()
    return pe_daily


# ── 6. Plot NDX + PE ──────────────────────────────────────
def plot_pe_history(pe_quarterly, pe_daily, tickers_info):
    ndx = pd.read_parquet(ROOT / "data" / "^NDX.parquet")
    ndx_price = ndx["close"] if "close" in ndx.columns else ndx.iloc[:, 0]

    fig, (ax_price, ax_pe) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[2, 3],
                                           gridspec_kw={"hspace": 0.08}, sharex=True)

    ax_price.semilogy(ndx_price.index, ndx_price.values, "k-", lw=1.5, label="NASDAQ-100")
    ax_price.set_ylabel("NDX Price (log)")
    ax_price.legend(loc="upper left", fontsize=9)
    ax_price.grid(True, alpha=0.3)

    tickers_str = ", ".join([t["symbol"] for t in tickers_info[:TOP_PE]])
    ax_price.set_title(f"NASDAQ-100 Price vs Top-{TOP_PE} Cap-Weighted PE (daily interpolated)\n"
                       f"{tickers_str}", fontsize=13)

    # Daily interpolated PE
    ax_pe.semilogy(pe_daily.index, pe_daily.values, color="royalblue", lw=1, alpha=0.7,
                   label="PE Top 5 (daily interpolated)")
    # Quarterly markers
    ax_pe.semilogy(pe_quarterly.index, pe_quarterly["pe_top5"], "ko", ms=3, alpha=0.5,
                   label="PE Top 5 (quarterly)")

    for val, col, lbl in [(20, "green", "PE=20"), (35, "orange", "PE=35"),
                           (50, "red", "PE=50"), (100, "darkred", "PE=100")]:
        ax_pe.axhline(val, color=col, ls=":", alpha=0.4, lw=1)
        ax_pe.text(pe_daily.index[0], val * 1.05, lbl, fontsize=7, color=col, alpha=0.7)

    ymax = max(pe_daily.max(), pe_quarterly["pe_top5"].max()) * 1.5
    ax_pe.set_ylim(5, ymax)
    ax_pe.set_ylabel("PE Ratio (log)")
    ax_pe.legend(loc="upper left", fontsize=9)
    ax_pe.grid(True, alpha=0.3)

    crises = [("2000-03-01", "2002-10-01", "Dot-com"), ("2007-10-01", "2009-03-01", "GFC"),
              ("2020-02-01", "2020-04-01", "Covid"), ("2022-01-01", "2022-10-01", "Rate hike")]
    for start, end, label in crises:
        for ax in [ax_price, ax_pe]:
            ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color="red", alpha=0.08)
        ax_price.text(pd.Timestamp(start), ax_price.get_ylim()[1] * 0.7, label,
                      fontsize=8, color="red", alpha=0.7)

    ax_pe.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    out = OUT_DIR / "ndx_top5_pe_daily.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nChart saved → {out}")
    plt.close()


# ── Main ──────────────────────────────────────────────────
def load_pe_daily():
    """Load cached daily PE. Call from backtest."""
    cache = PE_DIR / "pe_top5_daily.parquet"
    if cache.exists():
        return pd.read_parquet(cache)["pe_top5_daily"]
    return None


if __name__ == "__main__":
    print(f"=== NASDAQ-100 Top-{TOP_PE} Historical Cap-Weighted PE ===\n")

    # 1. Get top constituents
    top = fetch_top_constituents(TOP_N)
    for i, t in enumerate(top[:TOP_PE], 1):
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

    # 3. Calculate quarterly PE
    print(f"\nCalculating cap-weighted PE (top {TOP_PE})...")
    pe_q = calc_quarterly_pe(all_dfs, TOP_PE)
    pe_q.to_csv(PE_DIR / "pe_top5_quarterly.csv")
    print(f"Saved quarterly PE → {len(pe_q)} quarters")

    # 4. Interpolate daily
    print("Interpolating daily PE...")
    pe_daily = interpolate_daily_pe(pe_q)
    pe_daily.to_frame().to_parquet(PE_DIR / "pe_top5_daily.parquet")
    print(f"Saved daily PE → {len(pe_daily)} days ({pe_daily.index.min().date()} → {pe_daily.index.max().date()})")

    # Summary
    print(f"\nLast 8 quarters:")
    for _, row in pe_q.tail(8).iterrows():
        print(f"  {row['quarter']:<10s}  PE={row['pe_top5']:6.1f}  ({row['n_valid']:.0f} tickers)")
    print(f"\nCurrent daily PE: {pe_daily.iloc[-1]:.1f}")

    # 5. Plot
    plot_pe_history(pe_q, pe_daily, top)
