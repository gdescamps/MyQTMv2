"""
ETF universe for Boursorama quantitative trading strategy.

Fields:
  bourso     : Boursorama ticker (e.g. "CSP1.PA")
  name       : Full Boursorama display name
  section    : "geo" | "sector_us" | "thematic" | "commodity" | "bond" | "crypto"
  theme      : sub-theme for thematic/commodity (e.g. "semi", "ia", "gold", "btc")
  pea        : True if PEA-eligible (French tax wrapper)
  zero_fees  : True if 0% commission on Boursorama (purchase >= 500 EUR)
  proxy        : proxy ticker used for OHLCV data (US/LSE proxy or direct)
  is_proxy  : True if proxy ticker is a US/LSE proxy, False if direct UCITS match
  perf_1y    : 1-year performance (float, e.g. 0.21 for +21%), None if unavailable
  perf_5y    : 5-year performance (float), None if unavailable

Boursorama symbol: "1rT" + base ticker without exchange suffix
  e.g. CSP1.PA -> "1rTCSP1", SEMI.AS -> "1rTSEMI", EXX1.DE -> "1rTEXX1"
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ETF:
    bourso: str
    name: str
    section: str
    theme: str
    pea: bool
    zero_fees: bool
    proxy: str
    is_proxy: bool
    perf_1y: Optional[float]
    perf_5y: Optional[float]

    @property
    def bourso_symbol(self) -> str:
        """Boursorama internal symbol (1rT prefix + base ticker)."""
        base = self.bourso.split(".")[0]
        return f"1rT{base}"


# ---------------------------------------------------------------------------
# 1. Equity géographique
# ---------------------------------------------------------------------------
GEO = [
    ETF("CSP1.PA", "iShares Core S&P 500 ETF USD Acc",              "geo", "us",         pea=True,  zero_fees=True,  proxy="CSPX.AS", is_proxy=False, perf_1y=0.21,  perf_5y=0.96),
    ETF("CNX1.PA", "iShares NASDAQ 100 ETF USD Acc",                 "geo", "us",         pea=True,  zero_fees=True,  proxy="QQQ",     is_proxy=True,  perf_1y=0.39,  perf_5y=1.23),
    ETF("WPEA.PA", "iShares MSCI World Swap PEA ETF",                "geo", "world",      pea=True,  zero_fees=True,  proxy="WPEA.PA", is_proxy=False, perf_1y=0.20,  perf_5y=0.81),  # 5Y from IWDA.AS proxy
    ETF("IEMA.AS", "iShares MSCI Emerging Markets UCITS ETF USD Acc", "geo", "em",         pea=False, zero_fees=True,  proxy="IEMA.AS", is_proxy=False, perf_1y=0.39,  perf_5y=0.49),
    ETF("CSKR.PA", "iShares MSCI Korea ETF USD Dist",                "geo", "korea",      pea=False, zero_fees=True,  proxy="EWY",     is_proxy=True,  perf_1y=2.08,  perf_5y=1.02),
    # ETF("ITWN.PA", "iShares MSCI Taiwan ETF USD Dist",               "geo", "taiwan",     pea=False, zero_fees=True,  proxy="EWT",     is_proxy=True,  perf_1y=0.77,  perf_5y=0.65),  # IC=-0.118, systematiquement mal predit

    ETF("IFFI.AS", "iShares MSCI AC Far East ex-Japan ETF",          "geo", "far_east",   pea=False, zero_fees=True,  proxy="IFFI.AS", is_proxy=False, perf_1y=0.63,  perf_5y=0.46),
    ETF("EXCH.AS", "iShares MSCI EM ex-China ETF USD Acc",           "geo", "em_exch",    pea=False, zero_fees=True,  proxy="EXCH.AS", is_proxy=False, perf_1y=0.62,  perf_5y=0.72),
    ETF("SJPE.AS", "iShares Core MSCI Japan IMI ETF EUR Hedged",     "geo", "japan",      pea=False, zero_fees=True,  proxy="SJPE.AS", is_proxy=False, perf_1y=0.43,  perf_5y=1.40),
    ETF("IBZL.AS", "iShares MSCI Brazil ETF USD Dist",               "geo", "brazil",     pea=False, zero_fees=True,  proxy="EWZ",     is_proxy=True,  perf_1y=0.39,  perf_5y=0.04),
    ETF("IMEX.AS", "iShares MSCI Mexico Capped ETF USD Acc",         "geo", "mexico",     pea=False, zero_fees=True,  proxy="EWW",     is_proxy=True,  perf_1y=0.38,  perf_5y=0.74),
    ETF("ICAU.AS", "iShares MSCI Canada ETF USD Acc",                "geo", "canada",     pea=False, zero_fees=True,  proxy="EWC",     is_proxy=True,  perf_1y=0.35,  perf_5y=0.61),
    ETF("ITKY.AS", "iShares MSCI Turkey ETF USD Dist",               "geo", "turkey",     pea=False, zero_fees=True,  proxy="TUR",     is_proxy=True,  perf_1y=0.31,  perf_5y=0.87),
    ETF("ISF.L",   "iShares Core FTSE 100 ETF GBP Dist",             "geo", "uk",         pea=False, zero_fees=True,  proxy="ISF.L",   is_proxy=False, perf_1y=0.19,  perf_5y=0.46),
    # ETF("FXC.AS",  "iShares China Large Cap ETF USD Dist",           "geo", "china",      pea=False, zero_fees=True,  proxy="FXC.L",   is_proxy=True,  perf_1y=0.02,  perf_5y=-0.12),  # IC=-0.043, politique chinoise imprevisible
]

# ---------------------------------------------------------------------------
# 2. Secteurs US S&P 500
# ---------------------------------------------------------------------------
SECTOR_US = [
    ETF("IUIT.AS", "iShares S&P 500 Info Technology UCITS ETF USD Acc",  "sector_us", "tech",      pea=False, zero_fees=False, proxy="XLK", is_proxy=True, perf_1y=0.54, perf_5y=1.67),
    ETF("IUES.AS", "iShares S&P 500 Energy Sector UCITS ETF USD Acc",    "sector_us", "energy",    pea=False, zero_fees=False, proxy="XLE", is_proxy=True, perf_1y=0.36, perf_5y=1.19),
    ETF("IUII.AS", "iShares S&P 500 Industrials Sector UCITS ETF Acc",   "sector_us", "indus",     pea=False, zero_fees=False, proxy="XLI", is_proxy=True, perf_1y=0.24, perf_5y=0.72),
    ETF("IUCD.AS", "iShares S&P 500 Consumer Discret UCITS ETF USD Acc", "sector_us", "cons_disc", pea=False, zero_fees=False, proxy="XLY", is_proxy=True, perf_1y=0.11, perf_5y=0.42),
    ETF("IUHC.AS", "iShares S&P 500 Health Care Sector UCITS ETF Acc",   "sector_us", "health",    pea=False, zero_fees=False, proxy="XLV", is_proxy=True, perf_1y=0.07, perf_5y=0.20),
    ETF("IUCS.AS", "iShares S&P 500 Consumer Staples UCITS ETF USD Acc", "sector_us", "cons_stpl", pea=False, zero_fees=False, proxy="XLP", is_proxy=True, perf_1y=0.04, perf_5y=0.21),
    ETF("IUFS.AS", "iShares S&P 500 Financials Sector UCITS ETF Acc",    "sector_us", "finance",   pea=False, zero_fees=False, proxy="XLF", is_proxy=True, perf_1y=0.01, perf_5y=0.41),
]

# ---------------------------------------------------------------------------
# 3. Secteurs thématiques & IA
# ---------------------------------------------------------------------------
THEMATIC = [
    ETF("EXX1.DE", "iShares EURO STOXX Banks 30-15 ETF DE acc",       "thematic", "banks_eu", pea=True,  zero_fees=True,  proxy="EXX1.DE", is_proxy=False, perf_1y=0.31,  perf_5y=1.74),
    ETF("EXV1.DE", "iShares STOXX Europe 600 Tech (DE) acc",          "thematic", "tech_eu",  pea=True,  zero_fees=True,  proxy="EXV1.DE", is_proxy=False, perf_1y=0.30,  perf_5y=1.64),
    ETF("SEMI.AS", "iShares MSCI Global Semiconductors ETF USD Acc",  "thematic", "semi",     pea=False, zero_fees=True,  proxy="SEMI.AS", is_proxy=False, perf_1y=1.67,  perf_5y=2.64),
    ETF("AINF.PA", "iShares AI Infrastructure ETF USD Acc",           "thematic", "ia",       pea=False, zero_fees=True,  proxy="CHAT",    is_proxy=True,  perf_1y=1.12,  perf_5y=2.25),
    ETF("IART.PA", "iShares AI Innovation Active ETF USD Acc",        "thematic", "ia",       pea=False, zero_fees=True,  proxy="WTAI",    is_proxy=True,  perf_1y=0.86,  perf_5y=0.62),
    ETF("ECAR.AS", "iShares Electric Vehicle & Driving Tech ETF USD", "thematic", "ev",       pea=False, zero_fees=True,  proxy="DRIV",    is_proxy=True,  perf_1y=0.77,  perf_5y=0.56),
    ETF("INRA.AS", "iShares Global Clean Energy Transition ETF USD",  "thematic", "clean_nrg",pea=False, zero_fees=True,  proxy="INRA.AS", is_proxy=False, perf_1y=0.75,  perf_5y=0.42),
    ETF("CITY.AS", "iShares Smart City Infra ETF USD Acc",            "thematic", "infra",    pea=False, zero_fees=True,  proxy="CITY.AS", is_proxy=False, perf_1y=0.44,  perf_5y=0.60),
    # ETF("IQQQ.DE", "iShares Global Water ETF USD Acc",                "thematic", "water",    pea=False, zero_fees=True,  proxy="IQQQ.DE", is_proxy=False, perf_1y=0.01,  perf_5y=0.26),  # IC=-0.041, thematique niche
]

# ---------------------------------------------------------------------------
# 4. Commodités & Alternatif
# ---------------------------------------------------------------------------
COMMODITY = [
    ETF("IGLN.AS", "iShares Physical Gold ETC",                 "commodity", "gold",       pea=False, zero_fees=False, proxy="GLD",     is_proxy=True,  perf_1y=0.45,  perf_5y=1.54),
    ETF("SXRS.DE", "iShares Diversified Commodity Swap ETF DE", "commodity", "commodity",  pea=True,  zero_fees=True,  proxy="SXRS.DE", is_proxy=False, perf_1y=0.37,  perf_5y=0.82),
    ETF("RING",    "iShares Gold Producers ETF USD Acc",        "commodity", "gold_miners",pea=False, zero_fees=True,  proxy="RING",    is_proxy=True,  perf_1y=1.17,  perf_5y=1.74),
    ETF("IOGP.AS", "iShares Oil & Gas Explr&Prod ETF USD Acc",  "commodity", "oil",        pea=False, zero_fees=True,  proxy="IEO",     is_proxy=True,  perf_1y=0.33,  perf_5y=1.22),
]

# ---------------------------------------------------------------------------
# 5. Obligations / Taux (CTO uniquement)
# ---------------------------------------------------------------------------
BOND = [
    # ETF("DTLA.AS", "iShares USD Treasury Bond 20+yr UCITS ETF",     "bond", "us_lt",    pea=False, zero_fees=False, proxy="TLT", is_proxy=True, perf_1y=-0.01, perf_5y=-0.37),  # IC=-0.022
    # ETF("IBTA.AS", "iShares USD Treasury Bond 7-10yr UCITS ETF",    "bond", "us_mt",    pea=False, zero_fees=False, proxy="IEF", is_proxy=True, perf_1y=0.01,  perf_5y=-0.17),  # IC=+0.011
    # ETF("IHYU.AS", "iShares USD High Yield Corp Bond UCITS ETF",    "bond", "hy",       pea=False, zero_fees=False, proxy="HYG", is_proxy=True, perf_1y=0.01,  perf_5y=-0.08),  # IC=-0.046
    # ETF("ITPS.AS", "iShares USD TIPS UCITS ETF",                    "bond", "tips",     pea=False, zero_fees=False, proxy="TIP", is_proxy=True, perf_1y=0.03,  perf_5y=-0.13),  # IC=-0.001
]

# ---------------------------------------------------------------------------
# 6. ETN Crypto (disponibles depuis mars 2025)
# ---------------------------------------------------------------------------
CRYPTO = [
    # ETF("IBTC.AS", "iShares Physical Bitcoin ETP USD Acc", "crypto", "btc", pea=False, zero_fees=False, proxy="IBIT", is_proxy=True, perf_1y=0.55, perf_5y=None),  # IC=-0.20, pas de smart money, trop récent
]

# ---------------------------------------------------------------------------
# Full universe
# ---------------------------------------------------------------------------
UNIVERSE_FULL: list[ETF] = GEO + SECTOR_US + THEMATIC + COMMODITY + BOND + CRYPTO
# 27 ETFs — filtre smart money disponible depuis ≤2017 (proxy iShares/UCITS)
# bourso/proxy = ticker OHLCV (yfinance) ; smart money via ISHARES_MAP
UNIVERSE: list[ETF] = [
    # --- Geo equity ---
    ETF("IVV",     "S&P 500",            "geo", "us",      pea=False, zero_fees=False, proxy="IVV",     is_proxy=False, perf_1y=None, perf_5y=None),
    ETF("QQQ",     "Nasdaq 100",         "geo", "nasdaq",  pea=False, zero_fees=False, proxy="QQQ",     is_proxy=False, perf_1y=None, perf_5y=None),
    ETF("ACWI",    "MSCI World",         "geo", "world",   pea=False, zero_fees=False, proxy="ACWI",    is_proxy=False, perf_1y=None, perf_5y=None),
    ETF("EEM",     "Emerging Markets",   "geo", "em",      pea=False, zero_fees=False, proxy="EEM",     is_proxy=False, perf_1y=None, perf_5y=None),
    ETF("IEMG",    "Core EM IMI",        "geo", "em_imi",  pea=False, zero_fees=False, proxy="IEMG",    is_proxy=False, perf_1y=None, perf_5y=None),
    ETF("EMXC",    "EM ex-China",        "geo", "em_exch", pea=False, zero_fees=False, proxy="EMXC",    is_proxy=False, perf_1y=None, perf_5y=None),
    ETF("ILF",     "Latin America 40",   "geo", "latam",   pea=False, zero_fees=False, proxy="ILF",     is_proxy=False, perf_1y=None, perf_5y=None),
    ETF("EWY",     "South Korea",        "geo", "korea",   pea=False, zero_fees=False, proxy="EWY",     is_proxy=False, perf_1y=None, perf_5y=None),
    ETF("EWT",     "Taiwan",             "geo", "taiwan",  pea=False, zero_fees=False, proxy="EWT",     is_proxy=False, perf_1y=None, perf_5y=None),
    ETF("EWZ",     "Brazil",             "geo", "brazil",  pea=False, zero_fees=False, proxy="EWZ",     is_proxy=False, perf_1y=None, perf_5y=None),
    ETF("EWW",     "Mexico",             "geo", "mexico",  pea=False, zero_fees=False, proxy="EWW",     is_proxy=False, perf_1y=None, perf_5y=None),
    ETF("EWC",     "Canada",             "geo", "canada",  pea=False, zero_fees=False, proxy="EWC",     is_proxy=False, perf_1y=None, perf_5y=None),
    ETF("EWJ",     "Japan",              "geo", "japan",   pea=False, zero_fees=False, proxy="EWJ",     is_proxy=False, perf_1y=None, perf_5y=None),
    ETF("TUR",     "Turkey",             "geo", "turkey",  pea=False, zero_fees=False, proxy="TUR",     is_proxy=False, perf_1y=None, perf_5y=None),
    ETF("FXI",     "China Large-Cap",    "geo", "china",   pea=False, zero_fees=False, proxy="FXI",     is_proxy=False, perf_1y=None, perf_5y=None),
    ETF("ISF.L",   "FTSE 100",           "geo", "uk",      pea=False, zero_fees=False, proxy="ISF.L",   is_proxy=False, perf_1y=None, perf_5y=None),
    ETF("IEUR",    "Core Europe",        "geo", "europe",  pea=False, zero_fees=False, proxy="IEUR",    is_proxy=False, perf_1y=None, perf_5y=None),
    ETF("EZU",     "Eurozone",           "geo", "emu",     pea=False, zero_fees=False, proxy="EZU",     is_proxy=False, perf_1y=None, perf_5y=None),
    ETF("EPP",     "Pacific ex-Japan",   "geo", "pacific", pea=False, zero_fees=False, proxy="EPP",     is_proxy=False, perf_1y=None, perf_5y=None),
    ETF("SUSA",    "USA SRI",            "geo", "usa_sri", pea=False, zero_fees=False, proxy="SUSA",    is_proxy=False, perf_1y=None, perf_5y=None),
    # --- Thematic ---
    ETF("SOXX",    "Semiconductors",       "thematic", "semi",      pea=False, zero_fees=False, proxy="SOXX",    is_proxy=False, perf_1y=None, perf_5y=None),
    ETF("ROBO",    "Automation & Robotics","thematic", "robotics",  pea=False, zero_fees=False, proxy="ROBO",    is_proxy=False, perf_1y=None, perf_5y=None),
    ETF("ICLN",    "Global Clean Energy",  "thematic", "clean_nrg", pea=False, zero_fees=False, proxy="ICLN",    is_proxy=False, perf_1y=None, perf_5y=None),
    ETF("EXX1.DE", "EURO STOXX Banks",     "thematic", "banks_eu",  pea=False, zero_fees=False, proxy="EXX1.DE", is_proxy=False, perf_1y=None, perf_5y=None),
    # --- Commodity ---
    ETF("RING",    "Gold Miners",          "commodity", "gold_miners", pea=False, zero_fees=False, proxy="RING",    is_proxy=False, perf_1y=None, perf_5y=None),
    ETF("IEO",     "Oil & Gas E&P",        "commodity", "oil",         pea=False, zero_fees=False, proxy="IEO",     is_proxy=False, perf_1y=None, perf_5y=None),
    ETF("SXRS.DE", "Diversified Commodity","commodity", "commodity",   pea=False, zero_fees=False, proxy="SXRS.DE", is_proxy=False, perf_1y=None, perf_5y=None),
]

# Quick lookup dicts
BY_BOURSO: dict[str, ETF] = {e.bourso: e for e in UNIVERSE}
BY_PROXY: dict[str, list[ETF]] = {}
for _e in UNIVERSE:
    BY_PROXY.setdefault(_e.proxy, []).append(_e)


def pea_only() -> list[ETF]:
    """Return PEA-eligible ETFs."""
    return [e for e in UNIVERSE if e.pea]


def zero_fees_only() -> list[ETF]:
    """Return zero-fee ETFs (Boursorama purchase >= 500 EUR)."""
    return [e for e in UNIVERSE if e.zero_fees]


def by_section(section: str) -> list[ETF]:
    """Return ETFs filtered by section."""
    return [e for e in UNIVERSE if e.section == section]


def proxy_tickers() -> list[str]:
    """Return deduplicated list of proxy tickers needed to cover the universe."""
    return sorted(set(e.proxy for e in UNIVERSE))


if __name__ == "__main__":
    print(f"Universe: {len(UNIVERSE)} ETFs")
    print(f"  PEA-eligible : {len(pea_only())}")
    print(f"  Zero-fees    : {len(zero_fees_only())}")
    print(f"  Proxy tickers: {len(proxy_tickers())}")
    print()
    for section in ("geo", "sector_us", "thematic", "commodity", "bond", "crypto"):
        etfs = by_section(section)
        print(f"  [{section:<10}] {len(etfs):2d} ETFs")
    print()
    print("Proxy tickers:", proxy_tickers())
