"""
Download iShares fund XLS files (SpreadsheetML) containing full historical
NAV + Shares Outstanding data from fund inception.

These are the US-listed iShares ETF proxies used for institutional flow signals
(shares_outstanding_z20) in the ETF quantitative strategy.

Method:
  1. GET the product page to acquire session cookies (anti-bot protection)
  2. GET the XLS download URL with those cookies

Output: ./data/ishares/{TICKER}_fund.xls  (one file per ticker)

Then run:  python parse_ishares_xls.py
"""

import time
import requests
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data" / "ishares"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DELAY = 2.0  # seconds between requests (be polite)

# iShares US product IDs and slugs for each ticker
# URL: https://www.ishares.com/us/products/{pid}/{slug}
#
# Coverage map (UCITS ticker → US iShares proxy used for shares_outstanding):
#   CSP1/CSPX  → IVV   (iShares Core S&P 500)
#   WPEA       → ACWI  (iShares MSCI ACWI — World+EM)
#   IEMA       → EEM   (iShares MSCI Emerging Markets)
#   EXCH       → EMXC  (iShares MSCI EM ex China)
#   IFFI       → EEMA  (iShares MSCI EM Asia — closest to Far East ex Japan)
#   SJPE       → EWJ   (iShares MSCI Japan — unhedged; same flows)
#   CSKR/EWY   → EWY   ✓ direct
#   ITWN/EWT   → EWT   ✓ direct
#   LTAM       → n/a   (ILF closed fund, no shares outstanding available)
#   IBZL/EWZ   → EWZ   ✓ direct
#   IMEX/EWW   → EWW   ✓ direct
#   ICAU/EWC   → EWC   ✓ direct
#   ITKY/TUR   → TUR   ✓ direct
#   FXC/FXC.L  → FXI   (iShares China Large-Cap)
#   SEMI       → SOXX  (iShares Semiconductor ETF)
#   INRA/ICLN  → ICLN  (iShares Global Clean Energy)
#   IGLN/GLD   → IAU   (iShares Gold Trust — iShares native vs SPDR)
#   SXRS/GSG   → GSG   (iShares S&P GSCI Commodity)
#   IOGP/IEO   → IEO   ✓ direct
#   DTLA/TLT   → TLT   ✓ direct
#   IBTA/IEF   → IEF   ✓ direct
#   IHYU/HYG   → HYG   ✓ direct
#   ITPS/TIP   → TIP   ✓ direct
#   IBTC/IBIT  → IBIT  ✓ direct
#
# No US iShares equivalent for:
#   EXX1.DE  (EU Banks), EXV1.DE (EU Tech), AINF.PA (AI Infra, proxy=CHAT),
#   IART.PA (AI Innov, proxy=WTAI), ECAR.AS (EV, proxy=DRIV),
#   CITY.AS (Smart City), IQQQ.DE (Water)

ISHARES_PRODUCTS = {
    # --- Direct proxies (US-listed iShares ETFs already in universe) ---
    "EWY":  (239659, "ishares-msci-south-korea-etf"),
    "EWT":  (239691, "ishares-msci-taiwan-etf"),
    "EWZ":  (239513, "ishares-msci-brazil-etf"),
    "EWW":  (239690, "ishares-msci-mexico-capped-etf"),
    "EWC":  (239615, "ishares-msci-canada-etf"),
    "TUR":  (239689, "ishares-msci-turkey-etf"),
    "TLT":  (239454, "ishares-20-plus-year-treasury-bond-etf"),
    "IEF":  (239456, "ishares-7-10-year-treasury-bond-etf"),
    "HYG":  (239565, "ishares-iboxx-high-yield-corporate-bond-etf"),
    "TIP":  (239467, "ishares-tips-bond-etf"),
    "IEO":  (239519, "ishares-us-oil-gas-exploration-production-etf"),
    "RING": (239654, "ishares-msci-global-gold-miners-etf"),
    "IBIT": (333016, "ishares-bitcoin-trust-etf"),
    # --- US iShares equivalents for UCITS ETFs not directly covered ---
    "IVV":  (239726, "ishares-core-s-p-500-etf"),          # CSP1/CSPX
    "ACWI": (239600, "ishares-msci-acwi-etf"),             # WPEA (World)
    "EEM":  (239637, "ishares-msci-emerging-markets-etf"), # IEMA
    "EMXC": (288504, "ishares-msci-emerging-markets-ex-china-etf"),  # EXCH
    "EEMA": (239629, "ishares-msci-em-asia-etf"),          # IFFI (Far East)
    "EWJ":  (239665, "ishares-msci-japan-etf"),            # SJPE
    "FXI":  (239536, "ishares-china-large-cap-etf"),       # FXC
    "SOXX": (239705, "ishares-semiconductor-etf"),         # SEMI
    "ICLN": (239738, "ishares-global-clean-energy-etf"),   # INRA
    "IAU":  (239561, "ishares-gold-trust"),                # IGLN (Gold)
    "GSG":  (239757, "ishares-sp-gsci-commodity-indexed-trust"),  # SXRS
    # Note: ILF (Latin America 40) is a closed/delisted fund — no download available
}

BASE_PRODUCT_URL = "https://www.ishares.com/us/products"
DOWNLOAD_URL = "{base}/{pid}/{slug}/fund/1521942788811.ajax?fileType=xls&fileName={ticker}_fund&dataType=fund"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": "https://www.ishares.com/",
}


def download_fund_xls(ticker: str, pid: int, slug: str, session: requests.Session) -> bool:
    """Download iShares fund XLS for a given ticker. Returns True on success."""
    out_path = DATA_DIR / f"{ticker}_fund.xls"
    if out_path.exists():
        print(f"  SKIP  {ticker:<6}  (already downloaded)")
        return True

    product_url = f"{BASE_PRODUCT_URL}/{pid}/{slug}"
    xls_url = DOWNLOAD_URL.format(base=BASE_PRODUCT_URL, pid=pid, slug=slug, ticker=ticker)

    # Step 1: GET product page to acquire session cookies
    try:
        r = session.get(product_url, headers=HEADERS, timeout=30)
        r.raise_for_status()
    except requests.RequestException as e:
        print(f"  [ERROR] {ticker}: product page failed — {e}")
        return False

    time.sleep(0.5)

    # Step 2: GET XLS with session cookies
    try:
        r = session.get(xls_url, headers={**HEADERS, "Referer": product_url}, timeout=60)
        r.raise_for_status()
    except requests.RequestException as e:
        print(f"  [ERROR] {ticker}: XLS download failed — {e}")
        return False

    # Verify it's XML/SpreadsheetML, not HTML error page
    content = r.content
    if b"<?xml" not in content[:200] and b"ss:Workbook" not in content[:500]:
        print(f"  [ERROR] {ticker}: response is not SpreadsheetML (got HTML?) — {len(content)} bytes")
        return False

    out_path.write_bytes(content)
    size_kb = len(content) / 1024
    print(f"  OK    {ticker:<6}  {size_kb:.0f} KB  → {out_path.name}")
    return True


def main():
    print(f"Downloading {len(ISHARES_PRODUCTS)} iShares fund XLS files\n")

    session = requests.Session()
    ok, skipped, failed = 0, 0, 0

    for i, (ticker, (pid, slug)) in enumerate(ISHARES_PRODUCTS.items(), 1):
        print(f"[{i:2d}/{len(ISHARES_PRODUCTS)}] {ticker}", end="  ", flush=True)

        out_path = DATA_DIR / f"{ticker}_fund.xls"
        if out_path.exists():
            print(f"SKIP (already {out_path.stat().st_size // 1024} KB)")
            skipped += 1
            continue

        print("fetching...", end=" ", flush=True)
        success = download_fund_xls(ticker, pid, slug, session)
        if success:
            ok += 1
        else:
            failed += 1

        if i < len(ISHARES_PRODUCTS):
            time.sleep(DELAY)

    print(f"\nDone: {ok} downloaded, {skipped} skipped, {failed} failed")
    print(f"Data directory: {DATA_DIR}")
    print("\nNext step: python parse_ishares_xls.py")


if __name__ == "__main__":
    main()
