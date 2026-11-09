"""Comparaison Haiku 5.5 vs Sonnet 5.5 sur les tweets DeltaOne hors échantillon (2026-07 →), mêmes prompts.

1. accord des classements ; 2. calibration (mouvement NDX réel sur 2 bougies selon la confiance) ;
3. F1 vs vérité terrain (tweets déclencheurs des épisodes, myfiles/ndx_vol_episodes_news.jsonl) ;
4. pouvoir prédictif (IC) du score cumulé amorti des movers ; 5. trade naïf mover, 1 % A/R.
"""
import json

import numpy as np
import pandas as pd

CUT = pd.Timestamp("2026-07-01", tz="UTC")
H = pd.read_parquet("data/text/x/deltaone/classified.parquet")
S = pd.read_parquet("data/text/x/deltaone/classified_sonnet.parquet")
for d in (H, S):
    d["created_at"] = pd.to_datetime(d.created_at, utc=True)
H = H[H.created_at >= CUT]
m = H.merge(S[["id", "categorie", "sens", "mover", "confiance"]], on="id", suffixes=("_h", "_s"))
print(f"Tweets communs : {len(m)}")

# 1. accord
for k in ("h", "s"):
    mv = m[f"mover_{k}"] == True  # noqa: E712
    print(f"  {'Haiku' if k == 'h' else 'Sonnet'} : movers {mv.sum()} ({100 * mv.mean():.1f} %), "
          f"haussiers {(mv & (m[f'sens_{k}'] == 1)).sum()}, baissiers {(mv & (m[f'sens_{k}'] == -1)).sum()}")
mh, ms = m.mover_h == True, m.mover_s == True  # noqa: E712
print(f"  movers communs {(mh & ms).sum()} | Haiku seul {(mh & ~ms).sum()} | Sonnet seul {(~mh & ms).sum()}")
print(f"  même sens (tous tweets) {100 * (m.sens_h == m.sens_s).mean():.0f} %, même catégorie {100 * (m.categorie_h == m.categorie_s).mean():.0f} %, "
      f"corr. confiance {m.confiance_h.corr(m.confiance_s, method='spearman'):.2f}")

# 2. calibration
px = pd.read_parquet("data/stocks/USATECHIDXUSD_1h.parquet")
r = np.log(px.close).diff()
i = px.index.searchsorted(m.created_at.dt.floor("h"))
ok = i + 1 < len(px)
m, i = m[ok].copy(), i[ok]
m["r"] = r.values[i] + r.values[i + 1]
print("\n## Calibration (movers, mouvement NDX bougie du tweet + suivante)")
rows = []
for k, name in (("h", "Haiku"), ("s", "Sonnet")):
    d = m[(m[f"mover_{k}"] == True) & (m[f"sens_{k}"] != 0)]  # noqa: E712
    for lo, hi in ((0, 50), (50, 70), (70, 101)):
        g = d[(d[f"confiance_{k}"] >= lo) & (d[f"confiance_{k}"] < hi)]
        rows.append({"modèle": name, "confiance": f"{lo}-{min(hi, 100)}", "n": len(g),
                     "|mvt| moyen %": round(100 * g.r.abs().mean(), 2) if len(g) else None,
                     "bon sens %": round(100 * (np.sign(g.r) == g[f"sens_{k}"]).mean()) if len(g) else None})
    rows.append({"modèle": name, "confiance": "tous", "n": len(d), "|mvt| moyen %": round(100 * d.r.abs().mean(), 2),
                 "bon sens %": round(100 * (np.sign(d.r) == d[f"sens_{k}"]).mean())})
base = m[(m.mover_h != True) & (m.mover_s != True)]  # noqa: E712
rows.append({"modèle": "non-movers (les 2)", "confiance": "-", "n": len(base), "|mvt| moyen %": round(100 * base.r.abs().mean(), 2)})
print(pd.DataFrame(rows).to_string(index=False))

# 3. F1 vs vérité terrain
gt = {}
for l in open("myfiles/ndx_vol_episodes_news.jsonl"):
    e = json.loads(l)
    s = 1 if e["side"] == "up" else -1
    for kind in ("news_declencheur", "news_prolongation"):
        for t in e.get(kind) or []:
            if t.get("url"):
                gt.setdefault(kind, {})[t["url"].split("/")[-1]] = s
m["sid"] = m.url.str.split("/").str[-1]
print("\n## F1 vs vérité terrain (épisodes ≥ 1.5 %)")
for label, keys in (("déclencheurs", ["news_declencheur"]), ("décl. + prolongations", ["news_declencheur", "news_prolongation"])):
    g = {}
    for kk in keys:
        g.update(gt.get(kk, {}))
    m["vt"] = m.sid.map(g).fillna(0)
    for k, name in (("h", "Haiku"), ("s", "Sonnet")):
        for side, sl in ((1, "hausse"), (-1, "baisse"), (0, "les deux")):
            pred = (m[f"mover_{k}"] == True) & ((m[f"sens_{k}"] == side) if side else (m[f"sens_{k}"] != 0))  # noqa: E712
            true = (m.vt == side) if side else (m.vt != 0)
            tp = (pred & true & ((m[f"sens_{k}"] == m.vt) if side == 0 else True)).sum()
            fp, fn = pred.sum() - tp, true.sum() - tp
            p = tp / pred.sum() if pred.sum() else np.nan
            rc = tp / true.sum() if true.sum() else np.nan
            f1 = 2 * p * rc / (p + rc) if p + rc > 0 else 0
            print(f"  {label:22s} {name:6s} {sl:8s} GT {true.sum():3d} prédits {pred.sum():3d} TP {tp:3d} FP {fp:3d} "
                  f"P {p:.2f} R {rc:.2f} F1 {f1:.2f}")

# 4. IC du score cumulé (movers, poids confiance)
pxo = px.loc["2026-07-01":]
tb = (pxo.index + pd.Timedelta(hours=1)).values.astype("datetime64[ns]").astype(np.int64)
ro = np.log(pxo.close).diff().fillna(0).values
print("\n## IC score cumulé movers (Spearman score vs rendement futur)")
for k, name in (("h", "Haiku"), ("s", "Sonnet")):
    d = m[(m[f"mover_{k}"] == True) & (m[f"sens_{k}"] != 0)].sort_values("created_at")  # noqa: E712
    ts = d.created_at.values.astype("datetime64[ns]").astype(np.int64)
    w = (d[f"sens_{k}"] * d[f"confiance_{k}"] / 100).values
    out = []
    for hl in (2, 8, 24):
        lam = np.log(2) / (hl * 3600e9)
        Sc = np.array([(w[ts < t] * np.exp(-lam * (t - ts[ts < t]))).sum() for t in tb])
        for h in (1, 4, 23):
            f = pd.Series(ro[::-1]).rolling(h).sum().values[::-1]
            f = np.concatenate([f[1:], [np.nan]])
            x = pd.DataFrame({"S": Sc, "f": f}).dropna()
            x = x[x.S.abs() > 1e-3]
            out.append(f"HL{hl}/h{h} {x.S.corr(x.f, method='spearman'):+.3f}")
    print(f"  {name:6s} " + "  ".join(out))

# 5. trade naïf : entrée open bougie suivante, sortie après 23 bougies, 1 % A/R
print("\n## Trade mover (entrée bougie suivante, tenue 23 bougies, net 1 % A/R, chevauchements permis)")
for k, name in (("h", "Haiku"), ("s", "Sonnet")):
    for th in (0, 60):
        d = m[(m[f"mover_{k}"] == True) & (m[f"sens_{k}"] != 0) & (m[f"confiance_{k}"] >= th)]  # noqa: E712
        j = px.index.searchsorted(d.created_at.dt.ceil("h"))
        okj = j + 23 < len(px)
        d, j = d[okj], j[okj]
        g = d[f"sens_{k}"].values * (px.close.values[j + 22] / px.open.values[j] - 1) * 100
        print(f"  {name:6s} conf≥{th:2d} : n {len(g):3d}  brut moyen {g.mean():+.2f} %  net {g.mean() - 1:+.2f} %  gagnants bruts {100 * (g > 0).mean():.0f} %")
