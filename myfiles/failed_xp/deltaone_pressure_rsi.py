"""Le score de pression news cumulé (cf. deltaone_news_pressure.py) est-il nul parce que les news
« dans la tendance » marchent et les news « à contre-courant du RSI daily » échouent, et s'annulent ?

Pour chaque bougie horaire NDX : S (Σ sens·conf/100 amorti, tweets avant le close de la bougie),
RSI14 daily QQQ au dernier close US passé, momentum prix (EWM même demi-vie).
IC = Spearman(S, rendement futur h bougies), par régime RSI et selon que S va dans le sens du régime
ou à contre-courant ; puis IC résiduel (S résidualisé sur momentum, sur RSI, sur les deux) et IC
d'un signal « retourné » (S × −1 quand S est à contre-courant du RSI).
"""
import numpy as np
import pandas as pd

from src.risk_off_strategy.data import load_ohlc
from src.risk_off_strategy.strategy import rsi_wilder

CUT = pd.Timestamp("2026-07-01", tz="UTC")
px = pd.read_parquet("data/stocks/USATECHIDXUSD_1h.parquet").loc["2023-01-01":]
idx = px.index
ret = np.log(px.close).diff().fillna(0).values
tb = (idx + pd.Timedelta(hours=1)).values.astype("datetime64[ns]").astype(np.int64)

q = load_ohlc("QQQ", "2021-01-01", None)
rsi_d = pd.Series(rsi_wilder(q["close"].values), index=pd.to_datetime(q.index).tz_localize(None).normalize())
ny = idx.tz_convert("America/New_York")
day = ny.tz_localize(None).normalize()
known = pd.DatetimeIndex(np.where(ny.hour >= 16, day, day - pd.Timedelta(days=1)))
rsi = rsi_d.reindex(known, method="ffill").values
trend = np.where(rsi > 60, 1, np.where(rsi < 40, -1, 0))

df = pd.read_parquet("data/text/x/deltaone/classified.parquet")
df["created_at"] = pd.to_datetime(df.created_at, utc=True)
df = df[~df.categorie.isin(["reaction_marche", "hors_marche"]) & (df.sens != 0)].dropna(subset=["sens"])


def pressure(sub, hl):
    sub = sub.sort_values("created_at")
    ts = sub.created_at.values.astype("datetime64[ns]").astype(np.int64)
    w = (sub.sens * sub.confiance / 100).values
    lam = np.log(2) / (hl * 3600e9)
    S, j, acc, last = np.zeros(len(tb)), 0, 0.0, tb[0]
    for i, t in enumerate(tb):
        acc *= np.exp(-lam * (t - last))
        while j < len(ts) and ts[j] < t:
            acc += w[j] * np.exp(-lam * (t - ts[j]))
            j += 1
        S[i], last = acc, t
    return S


def resid(y, *xs):
    X = np.column_stack([np.ones(len(y))] + [pd.Series(x).rank().values for x in xs])
    yr = pd.Series(y).rank().values
    b = np.linalg.lstsq(X, yr, rcond=None)[0]
    return yr - X @ b


def ic(a, b):
    return pd.Series(a).corr(pd.Series(b), method="spearman")


rows = []
for name, sub in (("movers", df[df.mover == True]), ("tous conf", df)):  # noqa: E712
    for hl in (24, 72):
        S = pressure(sub, hl)
        mom = pd.Series(ret).ewm(alpha=1 - 0.5 ** (1 / hl)).mean().values
        for h in (23, 46, 115):
            f = pd.Series(ret[::-1]).rolling(h).sum().values[::-1]
            f = np.concatenate([f[1:], [np.nan]])
            for per, pm in (("biaisé", idx < CUT), ("OOS", idx >= CUT)):
                d = pd.DataFrame({"S": S, "f": f, "mom": mom, "rsi": rsi, "tr": trend})[pm].dropna()
                d = d[d.S.abs() > 1e-3]
                al = np.sign(d.S) == d.tr       # S dans le sens du régime
                ct = np.sign(d.S) == -d.tr      # S à contre-courant
                flip = np.where(ct, -d.S.values, d.S.values)
                rows.append({
                    "score": f"{name} HL{hl}", "h": h, "période": per, "n": len(d),
                    "IC brut": ic(d.S, d.f),
                    "IC dans le sens": ic(d.S[al], d.f[al]), "n sens": int(al.sum()),
                    "IC contre-courant": ic(d.S[ct], d.f[ct]), "n contre": int(ct.sum()),
                    "IC RSI 40-60": ic(d.S[d.tr == 0], d.f[d.tr == 0]),
                    "IC hors mom": ic(resid(d.S.values, d.mom.values), d.f.values),
                    "IC hors RSI": ic(resid(d.S.values, d.rsi.values), d.f.values),
                    "IC hors mom+RSI": ic(resid(d.S.values, d.mom.values, d.rsi.values), d.f.values),
                    "IC retourné": ic(flip, d.f.values),
                    "corr S/RSI": ic(d.S, d.rsi),
                })
res = pd.DataFrame(rows).round(3)
pd.set_option("display.width", 250)
for per in ("biaisé", "OOS"):
    print(f"\n#### {per}")
    print(res[res["période"] == per].drop(columns="période").to_string(index=False))
res.to_csv("myfiles/failed_xp/deltaone_pressure_rsi.csv", index=False)
