"""Fetch ETF quotes from Boursorama (no authentication required)."""

import re
import requests

BASE_URL = "https://www.boursorama.com"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/143.0.0.0 Safari/537.36",
}

# Symboles Boursorama pour les ETFs courants
SYMBOLS = {
    "PANX": "1rTPANX",   # Amundi PEA Nasdaq-100 (FR0013412269)
    "LQQ":  "1rTLQQ",    # Amundi Nasdaq-100 2x (FR0010342592)
    "CW8":  "1rTCW8",    # Amundi MSCI World (LU1681043599)
    "PE500": "1rTPE500",  # Amundi PEA S&P 500 (FR0013412285)
}


def search_symbol(query):
    """Search Boursorama for a tracker symbol.

    Returns list of dicts with keys: symbol, name, url.
    """
    resp = requests.get(
        f"{BASE_URL}/recherche/ajax?query={query}",
        headers=HEADERS, timeout=10,
    )
    results = []
    for m in re.finditer(
        r'href="/bourse/trackers/cours/(?P<sym>[^/"]+)/"[^>]*>.*?'
        r'class="search__list-link-label[^"]*"[^>]*>(?P<name>[^<]+)',
        resp.text, re.DOTALL,
    ):
        results.append({
            "symbol": m.group("sym"),
            "name": m.group("name").strip(),
            "url": f"{BASE_URL}/bourse/trackers/cours/{m.group('sym')}/",
        })
    return results


def get_quote(symbol):
    """Get current quote for a Boursorama symbol (e.g. '1rTPANX' or alias 'PANX').

    Returns dict with keys: symbol, name, last, change, change_pct, currency, date.
    """
    sym = SYMBOLS.get(symbol.upper(), symbol)
    url = f"{BASE_URL}/bourse/trackers/cours/{sym}/"
    resp = requests.get(url, headers=HEADERS, timeout=10)
    resp.raise_for_status()
    html = resp.text

    result = {"symbol": sym, "url": url}

    # Name from <title>
    m = re.search(r'<title>([^,]+)', html)
    result["name"] = m.group(1).strip() if m else None

    # Last price (first data-ist-last on page = the tracker itself)
    m = re.search(r'data-ist-last>([0-9\s]+[,.]?\d*)<', html)
    if m:
        result["last"] = float(m.group(1).replace("\xa0", "").replace(" ", "").replace(",", "."))
    else:
        result["last"] = None

    # Change
    m = re.search(r'data-ist-variation>([^<]+)<', html)
    if m:
        val = m.group(1).strip().replace(",", ".").replace("+", "").replace("%", "").replace("\xa0", "")
        try:
            result["change_pct"] = float(val)
        except ValueError:
            result["change_pct"] = None
    else:
        result["change_pct"] = None

    # Currency
    m = re.search(r'c-faceplate__price-currency[^>]*>([^<]+)', html)
    result["currency"] = m.group(1).strip() if m else "EUR"

    # Date/time
    m = re.search(r'c-faceplate__real-time[^>]*>([^<]+)', html)
    result["date"] = m.group(1).strip() if m else None

    return result


def get_quotes(symbols=None):
    """Get quotes for multiple symbols. Defaults to all known symbols."""
    if symbols is None:
        symbols = list(SYMBOLS.keys())
    return {s: get_quote(s) for s in symbols}


def print_quotes(symbols=None):
    """Print quotes in a readable format."""
    quotes = get_quotes(symbols)
    print(f"{'ETF':<8} {'Nom':<45} {'Cours':>10} {'Var%':>8}")
    print("-" * 75)
    for key, q in quotes.items():
        name = (q["name"] or "")[:45]
        last = f"{q['last']:.2f}" if q["last"] else "N/A"
        var = f"{q['change_pct']:+.2f}%" if q["change_pct"] is not None else "N/A"
        print(f"{key:<8} {name:<45} {last:>10} {var:>8}")


if __name__ == "__main__":
    print_quotes()
