"""« Pression news » cumulée et amortie : S(t) = Σ sens·w·exp(-(t - t_i)·ln2/HL) sur les tweets publiés avant t.

Test : est-ce que S au close d'une bougie horaire NDX prédit le rendement des h bougies suivantes ?
IC = corrélation de Spearman (S, rendement futur). Contrôle : momentum prix sur la même demi-vie
(la news cumulée ne fait-elle que refléter le prix passé ?). Hors reaction_marche/hors_marche.
"""
import numpy as np
import pandas as pd

CUTOFF = pd.Timestamp("2026-07-01", tz="UTC")
df = pd.read_parquet("data/text/x/deltaone/classified.parquet")
df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
df = df[~df.categorie.isin(["reaction_marche", "hors_marche"]) & (df.sens != 0)].dropna(subset=["sens"])
px = pd.read_parquet("data/stocks/USATECHIDXUSD_1h.parquet").loc["2023-01-01":]
idx = px.index
t_bar = (idx + pd.Timedelta(hours=1)).values.astype("datetime64[ns]").astype(np.int64)  # close de la bougie
ret = np.log(px.close).diff().fillna(0).values


def pressure(sub, hl_h, weight):
    ts = sub.created_at.values.astype("datetime64[ns]").astype(np.int64)
    w = sub.sens.values * weight(sub)
    lam = np.log(2) / (hl_h * 3600e9)
    S, j, acc, last = np.zeros(len(t_bar)), 0, 0.0, t_bar[0]
    order = np.argsort(ts)
    ts, w = ts[order], w[order]
    for i, t in enumerate(t_bar):
        acc *= np.exp(-lam * (t - last))
        while j < len(ts) and ts[j] < t:
            acc += w[j] * np.exp(-lam * (t - ts[j]))
            j += 1
        S[i], last = acc, t
    return S


def momentum(hl_h):
    a = 1 - 0.5 ** (1 / hl_h)
    return pd.Series(ret).ewm(alpha=a).mean().values


VARIANTS = {
    "tous, poids conf": (lambda d: d, lambda s: s.confiance.values / 100),
    "tous, poids 1": (lambda d: d, lambda s: np.ones(len(s))),
    "movers, poids conf": (lambda d: d[d.mover == True], lambda s: s.confiance.values / 100),  # noqa: E712
}
rows = []
for name, (filt, wf) in VARIANTS.items():
    for hl in [2, 8, 24, 72]:
        S = pressure(filt(df), hl, wf)
        mom = momentum(hl)
        for h in [1, 4, 23]:
            fwd = pd.Series(ret[::-1]).rolling(h).sum().values[::-1]  # Σ ret[i..i+h-1]
            fwd = np.concatenate([fwd[1:], [np.nan]])                 # rendement après le close i
            for per, m in (("biaisé", idx < CUTOFF), ("OOS", idx >= CUTOFF)):
                d = pd.DataFrame({"S": S, "f": fwd, "mom": mom})[m].dropna()
                d = d[d.S != 0]
                ic = d.S.corr(d.f, method="spearman")
                # IC partiel : S résiduel après momentum prix
                b = np.polyfit(d.mom.rank(), d.S.rank(), 1)
                res = d.S.rank() - np.polyval(b, d.mom.rank())
                icp = pd.Series(res.values).corr(d.f.rank().reset_index(drop=True))
                hit = (np.sign(d.S) == np.sign(d.f)).mean()
                rows.append({"variante": name, "demi-vie h": hl, "horizon h": h, "période": per,
                             "IC": round(ic, 3), "IC hors momentum": round(icp, 3),
                             "corr(S, mom passé)": round(d.S.corr(d.mom, method="spearman"), 2),
                             "% bon sens": round(100 * hit, 1), "n": len(d)})
res = pd.DataFrame(rows)
pd.set_option("display.width", 220)
for per in ("biaisé", "OOS"):
    print(f"\n#### {per}")
    print(res[res["période"] == per].drop(columns="période").to_string(index=False))
res.to_csv("myfiles/deltaone_news_pressure.csv", index=False)
