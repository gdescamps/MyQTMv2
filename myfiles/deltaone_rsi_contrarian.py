"""Tweets « à contre-courant » du RSI daily : baissiers quand RSI14 > 60, haussiers quand RSI14 < 40.

RSI de Wilder (14) sur le close daily QQQ (strategy.rsi_wilder), connu au dernier close US passé
(pas de look-ahead). Signal = tweets DeltaOne classés par Haiku (classified.parquet).
Mesure : rendement NDX CFD 1h dans le sens du tweet, de l'ouverture de la 1re bougie complète après
le tweet à +2 / +23 / +46 / +115 bougies (≈ 2 h, 1, 2, 5 séances), comparé :
  - aux mêmes tweets dans les autres régimes de RSI ;
  - au drift « aléatoire » (toutes les bougies du même régime, même sens) → colonnes exc.
"""
import numpy as np
import pandas as pd

from src.risk_off_strategy.data import load_ohlc
from src.risk_off_strategy.strategy import rsi_wilder

HS = [2, 23, 46, 115]
CUT = pd.Timestamp("2026-07-01", tz="UTC")

q = load_ohlc("QQQ", "2021-01-01", None)
rsi_d = pd.Series(rsi_wilder(q["close"].values), index=pd.to_datetime(q.index).tz_localize(None).normalize())
px = pd.read_parquet("data/stocks/USATECHIDXUSD_1h.parquet").loc["2023-01-01":]
idx = px.index
o, c = px.open.values, px.close.values
ny = idx.tz_convert("America/New_York")
day = ny.tz_localize(None).normalize()
known_day = np.where(ny.hour >= 16, day, day - pd.Timedelta(days=1))  # dernier close US passé
rsi_h = rsi_d.reindex(pd.DatetimeIndex(known_day), method="ffill").values


def fwd(j, h):
    ok = j + h - 1 < len(idx)
    out = np.full(len(j), np.nan)
    out[ok] = (c[j[ok] + h - 1] / o[j[ok]] - 1) * 100
    return out


def regime(r):
    return np.where(r > 60, "RSI>60", np.where(r < 40, "RSI<40", "40-60"))


allj = np.arange(len(idx))
ref = pd.DataFrame({"reg": regime(rsi_h)} | {f"h{h}": fwd(allj, h) for h in HS})
drift = ref.groupby("reg").mean()

df = pd.read_parquet("data/text/x/deltaone/classified.parquet")
df["created_at"] = pd.to_datetime(df.created_at, utc=True)
df = df[(df.sens != 0) & ~df.categorie.isin(["reaction_marche", "hors_marche"])].dropna(subset=["sens"])
j = idx.searchsorted(df.created_at.dt.ceil("h"))
ok = j < len(idx)
df, j = df[ok].copy(), j[ok]
df["rsi"] = rsi_h[j]
df["reg"] = regime(df.rsi.values)
for h in HS:
    df[f"h{h}"] = df.sens.values * fwd(j, h)

pd.set_option("display.width", 220)
for sel_name, sel in (("MOVERS (Haiku)", df.mover == True),  # noqa: E712
                      ("movers conf≥60", (df.mover == True) & (df.confiance >= 60)),  # noqa: E712
                      ("tous tweets orientés conf≥30", df.confiance >= 30)):
    d = df[sel]
    rows = []
    for sens, sl in ((-1, "baissier"), (1, "haussier")):
        for reg in ("RSI>60", "40-60", "RSI<40"):
            g = d[(d.sens == sens) & (d.reg == reg)]
            tag = " ◀" if (sens == -1 and reg == "RSI>60") or (sens == 1 and reg == "RSI<40") else ""
            row = {"tweet": sl, "régime": reg + tag, "n": len(g), "OOS": int((g.created_at >= CUT).sum())}
            for h in HS:
                x = g[f"h{h}"].dropna()
                row[f"+{h}b"] = round(x.mean(), 2) if len(x) else None
                row[f"exc{h}"] = round(x.mean() - sens * drift.loc[reg, f"h{h}"], 2) if len(x) else None
            x = g["h23"].dropna()
            row["bon sens 1s %"] = round(100 * (x > 0).mean()) if len(x) else None
            rows.append(row)
    print(f"\n######## {sel_name}  (rendement % dans le sens du tweet ; exc = moins le drift du régime)")
    print(pd.DataFrame(rows).to_string(index=False))
print("\nDrift moyen long par régime (%) :")
print(drift.round(3).to_string())
print("Part du temps par régime :", (ref.reg.value_counts(normalize=True) * 100).round(0).to_dict())

# Robustesse du cas « baissier mover quand RSI>60 » : jours distincts, par année, rachat contrariant net de frais
g = df[(df.mover == True) & (df.sens == -1) & (df.reg == "RSI>60")].copy()  # noqa: E712
g["jour"] = g.created_at.dt.tz_convert("America/New_York").dt.date
print(f"\nBaissiers movers RSI>60 : {len(g)} tweets sur {g.jour.nunique()} jours distincts")
first = g.sort_values("created_at").groupby("jour").first()  # 1 signal par jour
for h in HS[1:]:
    x = -first[f"h{h}"].dropna()  # rendement d'un ACHAT contrariant
    print(f"  achat contrariant 1/jour, tenue {h:3d} b : n {len(x)}  brut {x.mean():+.2f} %  net 1 % A/R {x.mean() - 1:+.2f} %  "
          f"gagnants {100 * (x > 0).mean():.0f} %  (drift long régime {drift.loc['RSI>60', f'h{h}']:+.2f} %)")
first["an"] = pd.to_datetime(first.index).year
print("  par an (tenue 46 b, achat) :", (-first.groupby("an")["h46"].mean()).round(2).to_dict(),
      " n :", first.groupby("an").size().to_dict())
