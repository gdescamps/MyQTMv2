"""
ETF universe for Boursorama quantitative trading strategy.

Fields:
  bourso     : Boursorama ticker (e.g. "CSP1.PA")
  name       : Full Boursorama display name
  section    : "geo" | "sector_us" | "thematic" | "commodity" | "bond" | "crypto"
  theme      : sub-theme for thematic/commodity (e.g. "semi", "ia", "gold", "btc")
  pea        : True if PEA-eligible (French tax wrapper)
  zero_fees  : True if 0% commission on Boursorama (purchase >= 500 EUR)
  fmp        : FMP ticker used for signal data (direct or proxy)
  fmp_proxy  : True if fmp ticker is a US/LSE proxy, False if direct UCITS match
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
    fmp: str
    fmp_proxy: bool
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
    ETF("CSP1.PA", "iShares Core S&P 500 ETF USD Acc",              "geo", "us",         pea=True,  zero_fees=True,  fmp="CSPX.AS", fmp_proxy=False, perf_1y=0.21,  perf_5y=0.96),
    ETF("CNX1.PA", "iShares NASDAQ 100 ETF USD Acc",                 "geo", "us",         pea=True,  zero_fees=True,  fmp="QQQ",     fmp_proxy=True,  perf_1y=0.39,  perf_5y=1.23),
    ETF("WPEA.PA", "iShares MSCI World Swap PEA ETF",                "geo", "world",      pea=True,  zero_fees=True,  fmp="WPEA.PA", fmp_proxy=False, perf_1y=0.20,  perf_5y=0.81),  # 5Y from IWDA.AS proxy
    ETF("IEMA.AS", "iShares MSCI Emerging Markets UCITS ETF USD Acc", "geo", "em",         pea=False, zero_fees=True,  fmp="IEMA.AS", fmp_proxy=False, perf_1y=0.39,  perf_5y=0.49),
    ETF("CSKR.PA", "iShares MSCI Korea ETF USD Dist",                "geo", "korea",      pea=False, zero_fees=True,  fmp="EWY",     fmp_proxy=True,  perf_1y=2.08,  perf_5y=1.02),
    # ETF("ITWN.PA", "iShares MSCI Taiwan ETF USD Dist",               "geo", "taiwan",     pea=False, zero_fees=True,  fmp="EWT",     fmp_proxy=True,  perf_1y=0.77,  perf_5y=0.65),  # IC=-0.118, systematiquement mal predit

    ETF("IFFI.AS", "iShares MSCI AC Far East ex-Japan ETF",          "geo", "far_east",   pea=False, zero_fees=True,  fmp="IFFI.AS", fmp_proxy=False, perf_1y=0.63,  perf_5y=0.46),
    ETF("EXCH.AS", "iShares MSCI EM ex-China ETF USD Acc",           "geo", "em_exch",    pea=False, zero_fees=True,  fmp="EXCH.AS", fmp_proxy=False, perf_1y=0.62,  perf_5y=0.72),
    ETF("SJPE.AS", "iShares Core MSCI Japan IMI ETF EUR Hedged",     "geo", "japan",      pea=False, zero_fees=True,  fmp="SJPE.AS", fmp_proxy=False, perf_1y=0.43,  perf_5y=1.40),
    ETF("IBZL.AS", "iShares MSCI Brazil ETF USD Dist",               "geo", "brazil",     pea=False, zero_fees=True,  fmp="EWZ",     fmp_proxy=True,  perf_1y=0.39,  perf_5y=0.04),
    ETF("IMEX.AS", "iShares MSCI Mexico Capped ETF USD Acc",         "geo", "mexico",     pea=False, zero_fees=True,  fmp="EWW",     fmp_proxy=True,  perf_1y=0.38,  perf_5y=0.74),
    ETF("ICAU.AS", "iShares MSCI Canada ETF USD Acc",                "geo", "canada",     pea=False, zero_fees=True,  fmp="EWC",     fmp_proxy=True,  perf_1y=0.35,  perf_5y=0.61),
    ETF("ITKY.AS", "iShares MSCI Turkey ETF USD Dist",               "geo", "turkey",     pea=False, zero_fees=True,  fmp="TUR",     fmp_proxy=True,  perf_1y=0.31,  perf_5y=0.87),
    ETF("ISF.L",   "iShares Core FTSE 100 ETF GBP Dist",             "geo", "uk",         pea=False, zero_fees=True,  fmp="ISF.L",   fmp_proxy=False, perf_1y=0.19,  perf_5y=0.46),
    # ETF("FXC.AS",  "iShares China Large Cap ETF USD Dist",           "geo", "china",      pea=False, zero_fees=True,  fmp="FXC.L",   fmp_proxy=True,  perf_1y=0.02,  perf_5y=-0.12),  # IC=-0.043, politique chinoise imprevisible
]

# ---------------------------------------------------------------------------
# 2. Secteurs US S&P 500
# ---------------------------------------------------------------------------
SECTOR_US = [
    ETF("IUIT.AS", "iShares S&P 500 Info Technology UCITS ETF USD Acc",  "sector_us", "tech",      pea=False, zero_fees=False, fmp="XLK", fmp_proxy=True, perf_1y=0.54, perf_5y=1.67),
    ETF("IUES.AS", "iShares S&P 500 Energy Sector UCITS ETF USD Acc",    "sector_us", "energy",    pea=False, zero_fees=False, fmp="XLE", fmp_proxy=True, perf_1y=0.36, perf_5y=1.19),
    ETF("IUII.AS", "iShares S&P 500 Industrials Sector UCITS ETF Acc",   "sector_us", "indus",     pea=False, zero_fees=False, fmp="XLI", fmp_proxy=True, perf_1y=0.24, perf_5y=0.72),
    ETF("IUCD.AS", "iShares S&P 500 Consumer Discret UCITS ETF USD Acc", "sector_us", "cons_disc", pea=False, zero_fees=False, fmp="XLY", fmp_proxy=True, perf_1y=0.11, perf_5y=0.42),
    ETF("IUHC.AS", "iShares S&P 500 Health Care Sector UCITS ETF Acc",   "sector_us", "health",    pea=False, zero_fees=False, fmp="XLV", fmp_proxy=True, perf_1y=0.07, perf_5y=0.20),
    ETF("IUCS.AS", "iShares S&P 500 Consumer Staples UCITS ETF USD Acc", "sector_us", "cons_stpl", pea=False, zero_fees=False, fmp="XLP", fmp_proxy=True, perf_1y=0.04, perf_5y=0.21),
    ETF("IUFS.AS", "iShares S&P 500 Financials Sector UCITS ETF Acc",    "sector_us", "finance",   pea=False, zero_fees=False, fmp="XLF", fmp_proxy=True, perf_1y=0.01, perf_5y=0.41),
]

# ---------------------------------------------------------------------------
# 3. Secteurs thématiques & IA
# ---------------------------------------------------------------------------
THEMATIC = [
    ETF("EXX1.DE", "iShares EURO STOXX Banks 30-15 ETF DE acc",       "thematic", "banks_eu", pea=True,  zero_fees=True,  fmp="EXX1.DE", fmp_proxy=False, perf_1y=0.31,  perf_5y=1.74),
    ETF("EXV1.DE", "iShares STOXX Europe 600 Tech (DE) acc",          "thematic", "tech_eu",  pea=True,  zero_fees=True,  fmp="EXV1.DE", fmp_proxy=False, perf_1y=0.30,  perf_5y=1.64),
    ETF("SEMI.AS", "iShares MSCI Global Semiconductors ETF USD Acc",  "thematic", "semi",     pea=False, zero_fees=True,  fmp="SEMI.AS", fmp_proxy=False, perf_1y=1.67,  perf_5y=2.64),
    ETF("AINF.PA", "iShares AI Infrastructure ETF USD Acc",           "thematic", "ia",       pea=False, zero_fees=True,  fmp="CHAT",    fmp_proxy=True,  perf_1y=1.12,  perf_5y=2.25),
    ETF("IART.PA", "iShares AI Innovation Active ETF USD Acc",        "thematic", "ia",       pea=False, zero_fees=True,  fmp="WTAI",    fmp_proxy=True,  perf_1y=0.86,  perf_5y=0.62),
    ETF("ECAR.AS", "iShares Electric Vehicle & Driving Tech ETF USD", "thematic", "ev",       pea=False, zero_fees=True,  fmp="DRIV",    fmp_proxy=True,  perf_1y=0.77,  perf_5y=0.56),
    ETF("INRA.AS", "iShares Global Clean Energy Transition ETF USD",  "thematic", "clean_nrg",pea=False, zero_fees=True,  fmp="INRA.AS", fmp_proxy=False, perf_1y=0.75,  perf_5y=0.42),
    ETF("CITY.AS", "iShares Smart City Infra ETF USD Acc",            "thematic", "infra",    pea=False, zero_fees=True,  fmp="CITY.AS", fmp_proxy=False, perf_1y=0.44,  perf_5y=0.60),
    # ETF("IQQQ.DE", "iShares Global Water ETF USD Acc",                "thematic", "water",    pea=False, zero_fees=True,  fmp="IQQQ.DE", fmp_proxy=False, perf_1y=0.01,  perf_5y=0.26),  # IC=-0.041, thematique niche
]

# ---------------------------------------------------------------------------
# 4. Commodités & Alternatif
# ---------------------------------------------------------------------------
COMMODITY = [
    ETF("IGLN.AS", "iShares Physical Gold ETC",                 "commodity", "gold",       pea=False, zero_fees=False, fmp="GLD",     fmp_proxy=True,  perf_1y=0.45,  perf_5y=1.54),
    ETF("SXRS.DE", "iShares Diversified Commodity Swap ETF DE", "commodity", "commodity",  pea=True,  zero_fees=True,  fmp="SXRS.DE", fmp_proxy=False, perf_1y=0.37,  perf_5y=0.82),
    ETF("RING",    "iShares Gold Producers ETF USD Acc",        "commodity", "gold_miners",pea=False, zero_fees=True,  fmp="RING",    fmp_proxy=True,  perf_1y=1.17,  perf_5y=1.74),
    ETF("IOGP.AS", "iShares Oil & Gas Explr&Prod ETF USD Acc",  "commodity", "oil",        pea=False, zero_fees=True,  fmp="IEO",     fmp_proxy=True,  perf_1y=0.33,  perf_5y=1.22),
]

# ---------------------------------------------------------------------------
# 5. Obligations / Taux (CTO uniquement)
# ---------------------------------------------------------------------------
BOND = [
    # ETF("DTLA.AS", "iShares USD Treasury Bond 20+yr UCITS ETF",     "bond", "us_lt",    pea=False, zero_fees=False, fmp="TLT", fmp_proxy=True, perf_1y=-0.01, perf_5y=-0.37),  # IC=-0.022
    # ETF("IBTA.AS", "iShares USD Treasury Bond 7-10yr UCITS ETF",    "bond", "us_mt",    pea=False, zero_fees=False, fmp="IEF", fmp_proxy=True, perf_1y=0.01,  perf_5y=-0.17),  # IC=+0.011
    # ETF("IHYU.AS", "iShares USD High Yield Corp Bond UCITS ETF",    "bond", "hy",       pea=False, zero_fees=False, fmp="HYG", fmp_proxy=True, perf_1y=0.01,  perf_5y=-0.08),  # IC=-0.046
    # ETF("ITPS.AS", "iShares USD TIPS UCITS ETF",                    "bond", "tips",     pea=False, zero_fees=False, fmp="TIP", fmp_proxy=True, perf_1y=0.03,  perf_5y=-0.13),  # IC=-0.001
]

# ---------------------------------------------------------------------------
# 6. ETN Crypto (disponibles depuis mars 2025)
# ---------------------------------------------------------------------------
CRYPTO = [
    # ETF("IBTC.AS", "iShares Physical Bitcoin ETP USD Acc", "crypto", "btc", pea=False, zero_fees=False, fmp="IBIT", fmp_proxy=True, perf_1y=0.55, perf_5y=None),  # IC=-0.20, pas de smart money, trop récent
]

# ---------------------------------------------------------------------------
# Full universe
# ---------------------------------------------------------------------------
UNIVERSE_FULL: list[ETF] = GEO + SECTOR_US + THEMATIC + COMMODITY + BOND + CRYPTO
# Extended US/EU ETFs with smart money
UNIVERSE: list[ETF] = [
    ETF("IVV", "iShares Core S&P 500 ETF", "geo", "us", pea=False, zero_fees=False, fmp="IVV", fmp_proxy=False, perf_1y=None, perf_5y=None),
    ETF("SOXX", "iShares Semiconductor ETF", "thematic", "semi", pea=False, zero_fees=False, fmp="SOXX", fmp_proxy=False, perf_1y=None, perf_5y=None),
    ETF("EEM", "iShares MSCI Emerging Markets ETF", "geo", "em", pea=False, zero_fees=False, fmp="EEM", fmp_proxy=False, perf_1y=None, perf_5y=None),
    ETF("GLD", "SPDR Gold Shares", "commodity", "gold", pea=False, zero_fees=False, fmp="GLD", fmp_proxy=False, perf_1y=None, perf_5y=None),
    ETF("TLT", "iShares 20+ Year Treasury Bond ETF", "bond", "us_lt", pea=False, zero_fees=False, fmp="TLT", fmp_proxy=False, perf_1y=None, perf_5y=None),
    ETF("IEO", "iShares U.S. Oil & Gas Exploration ETF", "commodity", "oil", pea=False, zero_fees=False, fmp="IEO", fmp_proxy=False, perf_1y=None, perf_5y=None),
    ETF("EXX1.DE", "iShares EURO STOXX Banks 30-15 ETF", "thematic", "banks_eu", pea=False, zero_fees=False, fmp="EXX1.DE", fmp_proxy=False, perf_1y=None, perf_5y=None),
    ETF("TIP", "iShares TIPS Bond ETF", "bond", "tips", pea=False, zero_fees=False, fmp="TIP", fmp_proxy=False, perf_1y=None, perf_5y=None),
    ETF("EWJ", "iShares MSCI Japan ETF", "geo", "japan", pea=False, zero_fees=False, fmp="EWJ", fmp_proxy=False, perf_1y=None, perf_5y=None),
    ETF("EXV1.DE", "iShares STOXX Europe 600 Tech ETF", "thematic", "tech_eu", pea=False, zero_fees=False, fmp="EXV1.DE", fmp_proxy=False, perf_1y=None, perf_5y=None),
    ETF("EWC", "iShares MSCI Canada ETF", "geo", "canada", pea=False, zero_fees=False, fmp="EWC", fmp_proxy=False, perf_1y=None, perf_5y=None),
    ETF("EWZ", "iShares MSCI Brazil ETF", "geo", "brazil", pea=False, zero_fees=False, fmp="EWZ", fmp_proxy=False, perf_1y=None, perf_5y=None),
]

# Quick lookup dicts
BY_BOURSO: dict[str, ETF] = {e.bourso: e for e in UNIVERSE}
BY_FMP: dict[str, list[ETF]] = {}
for _e in UNIVERSE:
    BY_FMP.setdefault(_e.fmp, []).append(_e)


def pea_only() -> list[ETF]:
    """Return PEA-eligible ETFs."""
    return [e for e in UNIVERSE if e.pea]


def zero_fees_only() -> list[ETF]:
    """Return zero-fee ETFs (Boursorama purchase >= 500 EUR)."""
    return [e for e in UNIVERSE if e.zero_fees]


def by_section(section: str) -> list[ETF]:
    """Return ETFs filtered by section."""
    return [e for e in UNIVERSE if e.section == section]


def fmp_tickers() -> list[str]:
    """Return deduplicated list of FMP tickers needed to cover the universe."""
    return sorted(set(e.fmp for e in UNIVERSE))


if __name__ == "__main__":
    print(f"Universe: {len(UNIVERSE)} ETFs")
    print(f"  PEA-eligible : {len(pea_only())}")
    print(f"  Zero-fees    : {len(zero_fees_only())}")
    print(f"  FMP tickers  : {len(fmp_tickers())}")
    print()
    for section in ("geo", "sector_us", "thematic", "commodity", "bond", "crypto"):
        etfs = by_section(section)
        print(f"  [{section:<10}] {len(etfs):2d} ETFs")
    print()
    print("FMP tickers:", fmp_tickers())
