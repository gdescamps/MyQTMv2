"""Seuil de fermeture des positions longues myqtm : P(haussier) < c signale-t-il bien la fin des hausses ?

A. Niveau jour, sur les jours de tendance haussière seulement. Palier X (5 … 25 %) : positifs = les X % derniers
   jours de chaque hausse (phase >= 1 − X), négatifs = le reste de la hausse (fermer y serait prématuré).
   Signal = P(haussier) < c. Précision = part des signaux tombant dans la fin de la hausse ; rappel = part des
   fins de hausse signalées ; base = X. On mesure aussi le délai de réaction après la fin de la hausse.
B. Niveau position (comme le moteur, long seulement) : entrée le premier jour où P(haussier) franchit
   OPEN = 0,6, sortie le premier jour suivant où P(haussier) < c. Où tombe la sortie (prématurée, dans les
   20 % de fin, après la fin, entrée hors hausse), durée, rendement ouverture J+1 → ouverture du lendemain
   de la sortie, brut et relatif à la moyenne des membres.

    PYTHONPATH=. python myfiles/myqtm_close_threshold.py
"""
import numpy as np
import pandas as pd

from src.portefolio_strategy.myqtm import config as C
from src.portefolio_strategy.myqtm import features as F
from src.portefolio_strategy.myqtm import train as T
from src.portefolio_strategy.myqtm.trends import detect_trends

TAILS = (0.05, 0.10, 0.15, 0.20, 0.25)
CLOSES = (0.1, 0.2, 0.3, 0.37, 0.4, 0.5, 0.6)
OPEN = 0.6
TAIL_B = 0.20


def segments(df):
    """→ colonnes : bull (jour dans une hausse définitive), phase (position dans la hausse), seg (id de hausse),
    seg_end (date de fin de la hausse en cours ou de la dernière hausse terminée), labelled."""
    out = []
    for sym, g in df.groupby("symbol", sort=False):
        close = g.set_index("date")["c"]
        segs, final_until = detect_trends(close, **C.TREND)
        d = pd.DataFrame(index=close.index)
        d["symbol"], d["bull"], d["phase"], d["seg"] = sym, False, np.nan, -1
        d["labelled"] = d.index < final_until
        d["last_end"] = pd.NaT
        k = 0
        for s in segs:
            if s["trend"] != "bullish":
                continue
            m = (d.index >= s["start"]) & (d.index <= s["end"])
            d.loc[m, "bull"], d.loc[m, "phase"], d.loc[m, "seg"] = True, np.arange(m.sum()) / m.sum(), k
            d.loc[d.index > s["end"], "last_end"] = s["end"]        # écrasé par la hausse suivante
            k += 1
        out.append(d.reset_index(names="date"))
    return pd.concat(out, ignore_index=True)


def load():
    df = pd.read_parquet(F.FEATURES, columns=["date", "symbol", "member", "o", "c"])
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    df = df.merge(segments(df), on=["date", "symbol"])
    g = df.groupby("symbol")["o"]
    df["o_next"] = g.shift(-1)
    p = pd.read_parquet(T.PREDICTIONS)[["date", "symbol", "ybull"]].merge(df, on=["date", "symbol"])
    p = p[p["labelled"]].sort_values(["symbol", "date"]).reset_index(drop=True)
    p["mkt"] = p.groupby("date")["o_next"].transform("mean")
    return p


def part_a(p):
    bull = p[p["bull"]]
    rows = []
    for x in TAILS:
        pos = bull["phase"] >= 1 - x
        for c in CLOSES:
            s = bull["ybull"] < c
            rows.append(dict(palier=f"{int(x * 100)} %", seuil=c, base=pos.mean(), precision=pos[s].mean(),
                             rappel=(s & pos).sum() / pos.sum(), lift=pos[s].mean() / pos.mean(),
                             premat=(s & ~pos).sum() / (~pos).sum()))
    return pd.DataFrame(rows)


def reaction(p):
    """Délai (jours de bourse) entre la fin d'une hausse et le premier jour où P(haussier) < c."""
    after = p[~p["bull"] & p["last_end"].notna()].copy()
    after["k"] = after.groupby(["symbol", "last_end"]).cumcount() + 1
    out = {}
    for c in CLOSES:
        first = after[after["ybull"] < c].groupby(["symbol", "last_end"])["k"].min()
        n_end = after.groupby(["symbol", "last_end"]).ngroups
        # déjà sous le seuil avant la fin (dans les 20 % de fin) → délai 0
        tail = p[p["bull"] & (p["phase"] >= 1 - TAIL_B) & (p["ybull"] < c)]
        pre = set(zip(tail["symbol"], tail.groupby(["symbol", "seg"])["date"].transform("max")))
        out[c] = dict(fins=n_end, median_jours=float(first.median()), p75_jours=float(first.quantile(0.75)),
                      sous_seuil_en_5j=float((first <= 5).sum() / n_end), avant_fin=len(pre) / n_end)
    return pd.DataFrame(out).T


def part_b(p):
    rows = []
    p = p.copy()
    above = p["ybull"] >= OPEN
    p["entry"] = above & ~above.groupby(p["symbol"]).shift(1, fill_value=False)
    for c in CLOSES:
        res = []
        for sym, g in p.groupby("symbol", sort=False):
            yb, ent = g["ybull"].to_numpy(), g["entry"].to_numpy()
            i, n = 0, len(g)
            while i < n:
                if not ent[i]:
                    i += 1
                    continue
                j = i + 1
                while j < n and yb[j] >= c:
                    j += 1
                if j >= n:
                    break
                e, x = g.iloc[i], g.iloc[j]
                if x["phase"] >= 1 - TAIL_B and x["bull"]:
                    where = "fin (20 % derniers)"
                elif x["bull"] and e["bull"] and x["seg"] == e["seg"]:
                    where = "prématurée"
                elif not e["bull"] and not (x["bull"]):
                    where = "entrée hors hausse"
                else:
                    where = "après la fin"
                r = x["o_next"] / e["o_next"] - 1
                rm = x["mkt"] / e["mkt"] - 1
                res.append(dict(where=where, days=j - i, ret=r, rel=r - rm,
                                entry_phase=e["phase"] if e["bull"] else np.nan))
                i = j + 1
        r = pd.DataFrame(res).dropna(subset=["ret"])
        share = r["where"].value_counts(normalize=True)
        rows.append(dict(seuil=c, positions=len(r), duree_med=r["days"].median(),
                         **{f"% {k}": 100 * share.get(k, 0) for k in
                            ("prématurée", "fin (20 % derniers)", "après la fin", "entrée hors hausse")},
                         gain_moy_pct=100 * r["ret"].mean(), rel_moy_pct=100 * r["rel"].mean(),
                         rel_par_jour_pb=1e4 * r["rel"].sum() / r["days"].sum(),
                         gagnantes_pct=100 * (r["ret"] > 0).mean()))
    return pd.DataFrame(rows)


def main():
    p = load()
    pd.set_option("display.width", 220, "display.float_format", "{:.3f}".format)
    a = part_a(p)
    a.to_csv("myfiles/myqtm_close_threshold_days.csv", index=False)
    print(f"{len(p)} prédictions labellisées, dont {p['bull'].sum()} jours de hausse\n")
    print("=== A. Jours de hausse : précision (rappel) du signal P(haussier) < c pour la FIN de hausse ===")
    order = [f"{int(x * 100)} %" for x in TAILS]
    tab = a.assign(cell=a.apply(lambda r: f"{r.precision:.1%} ({r.rappel:.0%})", axis=1)).pivot(
        index="seuil", columns="palier", values="cell")[order]
    print("base :", "  ".join(f"{k} {v:.1%}" for k, v in a.groupby("palier", sort=False)["base"].first().items()))
    print(tab.to_string())
    print("lift :\n" + a.pivot(index="seuil", columns="palier", values="lift")[order].round(2).to_string())
    print("part des jours de hausse « non finale » (hors 20 % de fin) signalés à tort :\n"
          + a[a["palier"] == "20 %"].set_index("seuil")["premat"].map("{:.1%}".format).to_string())
    print("\n=== Délai de réaction après la fin d'une hausse ===")
    print(reaction(p).to_string())
    b = part_b(p)
    b.to_csv("myfiles/myqtm_close_threshold_positions.csv", index=False)
    print(f"\n=== B. Positions longues : entrée P(haussier) franchit {OPEN}, sortie P(haussier) < c ===")
    print(b.to_string(index=False))


if __name__ == "__main__":
    main()
