"""Frais Bourso réduits (0.1 % par côté = 0.2 % A/R) : base vs base + garde-fou MM20 sentiment,
avec re-balayage de la bande de non-transaction (calibrée à l'origine pour 0.5 %/côté).
    python -m myfiles.pust17_sentiment_lowfee
"""
import numpy as np
import pandas as pd

from myfiles.pust17_sentiment_guard import EVAL_START
import myfiles.pust17_tweet_overlay as ov
from myfiles.pust17_tweet_overlay import metrics, simulate
from src.risk_off_strategy.data import load_funding, load_macro, load_ohlc
from src.risk_off_strategy.strategy import compute_allocation

ov.BUY_FEE = ov.SELL_FEE = 0.001
ohlc = load_ohlc("QQQ", "2000-01-01", None)
price = ohlc["close"]; idx = price.index
nfci, cpi = load_macro(idx)
fund = np.asarray(load_funding(idx), float)
ret = price.pct_change().fillna(0).values
alloc = compute_allocation(price.values, nfci, cpi, high=ohlc["high"].values, low=ohlc["low"].values, open_=ohlc["open"].values)
s = pd.read_parquet("data/text/cnbc/sentiment/sentiment.parquet").set_index("date").sort_index()["score"]
mm = s.rolling(20, min_periods=10).mean()
mm = mm.reindex(idx.union(mm.index)).ffill().reindex(idx).shift(1).fillna(0).values
rows = []
for b, sl in [(0.25, 0.50), (0.15, 0.30), (0.10, 0.20), (0.10, 0.30), (0.05, 0.10)]:
    ov.BUY_THR_ALLOC, ov.SELL_THR_ALLOC = b, sl
    for name, thr, capv in [("base", None, None), ("MM20≤−0.625 cap1", -0.625, 1.0), ("MM20≤−1 cap1", -1.0, 1.0),
                            ("MM20≤−0.625 cap0", -0.625, 0.0), ("MM20≤−0.875 cap0", -0.875, 0.0)]:
        cap = None if thr is None else np.where(mm <= thr, capv, np.nan)
        eq, fees = simulate(ret, alloc, fund, cap)
        m = metrics(eq, fees, idx, EVAL_START)
        mf = metrics(eq, fees, idx, "2000-01-01") if thr is None else {}
        rows.append({"bande achat/vente (expo)": f"{2*b:.2f}/{2*sl:.2f}", "variante": name, **m,
                     **({"base 2000+ %/an": mf.get("%/an"), "base 2000+ maxDD": mf.get("maxDD %"), "base 2000+ Calmar": mf.get("Calmar")} if mf else {})})
d = pd.DataFrame(rows).round(2)
pd.set_option("display.width", 250)
print(d.to_string(index=False))
d.to_csv("myfiles/pust17_sentiment_lowfee.csv", index=False)
