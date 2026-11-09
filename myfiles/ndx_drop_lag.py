"""Délai entre la news déclencheuse et la baisse, sur les plus gros épisodes baissiers NDX (2023 →).

Épisodes : myfiles/ndx_vol_episodes_news.jsonl (déclencheurs = tweets DeItaone choisis par Haiku).
t0 = heure du 1er tweet déclencheur. Mesures sur le CFD NDX 1h :
- part de la baisse totale (pic avant t0 → plus bas de l'épisode) faite à t0+1h, +4h, +24h ;
- délai t0 → 50 % de la baisse ; t0 pendant la séance US ou hors séance.
"""
import json

import numpy as np
import pandas as pd

px = pd.read_parquet("data/stocks/USATECHIDXUSD_1h.parquet").loc["2022-12-25":]
idx, c = px.index, px.close
rows = []
for l in open("myfiles/ndx_vol_episodes_news.jsonl"):
    e = json.loads(l)
    if e["side"] != "down":
        continue
    tr = [t for t in e.get("news_declencheur") or [] if t.get("time")]
    if not tr:
        continue
    s, en = pd.Timestamp(e["start"]), pd.Timestamp(e["end"])
    first = min(tr, key=lambda t: t["time"])
    t0 = pd.Timestamp(first["time"], tz="UTC")
    a = min(t0.floor("h"), s)
    ref = c.loc[:t0.floor("h") - pd.Timedelta(hours=1)].iloc[-1]  # close de la dernière bougie finie avant la news
    low_t = c.loc[a:en].idxmin()
    low = c.loc[low_t]
    drop = low / ref - 1
    if drop > -0.01:
        continue

    def done(h):
        seg = c.loc[t0.floor("h"):t0.floor("h") + pd.Timedelta(hours=h - 1)]  # bougie du tweet incluse
        return (seg.min() / ref - 1) / drop if len(seg) else np.nan

    path = c.loc[t0.floor("h"):low_t]
    half = path[(path / ref - 1) <= drop / 2]
    lag50 = (half.index[0] + pd.Timedelta(hours=1) - t0) / pd.Timedelta(hours=1) if len(half) else np.nan
    ny = t0.tz_convert("America/New_York")
    sess = ny.weekday() < 5 and (9.5 <= ny.hour + ny.minute / 60 < 16)
    rows.append({"news (UTC)": f"{t0:%Y-%m-%d %H:%M}", "séance US": "oui" if sess else "non",
                 "baisse %": round(100 * drop, 1), "1h %": round(100 * done(1), 0), "4h %": round(100 * done(4), 0),
                 "24h %": round(100 * done(24), 0), "délai 50 % (h)": round(lag50, 1),
                 "catalyseur": e.get("catalyseur"), "news": first["text"][:70].replace("\n", " ")})
d = pd.DataFrame(rows).sort_values("baisse %").drop_duplicates("news (UTC)")
d["type"] = np.where(d["délai 50 % (h)"] <= 2, "instantané", np.where(d["délai 50 % (h)"] <= 12, "quelques heures", "différé (>12h)"))
pd.set_option("display.width", 260)
pd.set_option("display.max_colwidth", 70)
print(d.head(25).to_string(index=False))
print("\nRépartition (épisodes ≥ 1 % de baisse depuis la news) :")
for lab, g in (("tous", d), ("top 25", d.head(25)), ("tarifs", d[d.catalyseur.fillna("").str.contains("tarif|douan|commerce", case=False)])):
    print(f"  {lab:7s} n={len(g):3d}  " + "  ".join(f"{k} {100 * (g.type == k).mean():.0f}%" for k in ("instantané", "quelques heures", "différé (>12h)"))
          + f"  | délai médian {g['délai 50 % (h)'].median():.1f} h ; hors séance {100 * (g['séance US'] == 'non').mean():.0f} %")
d.to_csv("myfiles/ndx_drop_lag.csv", index=False)
