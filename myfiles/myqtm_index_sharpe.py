"""Indice concentré sur le S&P 500 : chercher le meilleur Sharpe, contre celui de l'univers équipondéré.

Prédictions de src/portefolio_strategy/myqtm/monthly_rank.py (hors échantillon depuis 2019). Grille :
- classement : modèles (Sharpe du mois suivant ; futurs leaders 6 mois ; faible volatilité prédite ; mélanges de rangs ;
  futurs leaders ou momentum restreints aux 50 / 100 plus grosses capitalisations)
  et références simples (momentum 6 mois ; faible volatilité 63 j ; momentum / volatilité ; grosses capitalisations) ;
- taille : top 10, 20, 30, 40 ;
- allocation : parts égales ; inverse de la volatilité 63 j ; selon le score (poids linéaire au rang dans la sélection) ;
  score × inverse volatilité ; capitalisation ;
- fréquence : mise à jour mensuelle ou hebdomadaire (mêmes modèles, notes recalculées chaque fin de semaine).
Achat à l'ouverture qui suit la décision, frais 0,05 % par sens sur la rotation, valeur quotidienne.
Protocole : on choisit sur 2019-2023, on vérifie sur 2024 → fin. Une configuration « bat l'univers » si son Sharpe
dépasse celui de l'univers équipondéré sur les DEUX périodes.

    PYTHONPATH=.:myfiles python myfiles/myqtm_index_sharpe.py
"""
import numpy as np
import pandas as pd

from src.portefolio_strategy.myqtm import config as C
from src.portefolio_strategy.myqtm import monthly_rank as M

NS = (10, 20, 30, 40)
WEIGHTS = ("égal", "inverse volatilité", "score", "score × inverse volatilité", "capitalisation")
SPLIT = pd.Timestamp("2024-01-01")
MODELS = ("leaders parmi les 50 plus grosses", "leaders parmi les 100 plus grosses", "momentum parmi les 100 plus grosses",
          "modèle Sharpe", "modèle futurs leaders", "modèle faible volatilité", "modèle leaders + faible volatilité",
          "modèle Sharpe + leaders + faible volatilité")
BASES = ("momentum 6 mois", "faible volatilité 63 j", "momentum / volatilité", "grosses capitalisations")


def scores(p):
    r = lambda col, sign=1: (sign * p[col]).groupby(p["date"]).rank(pct=True)        # noqa: E731
    big = p["cap"].groupby(p["date"]).rank(ascending=False)

    def among(k, col):                                      # classement restreint aux k plus grosses capitalisations
        return p[col].where(big <= k)
    return {"leaders parmi les 50 plus grosses": among(50, "pred_mom6n"),
            "leaders parmi les 100 plus grosses": among(100, "pred_mom6n"),
            "momentum parmi les 100 plus grosses": among(100, "mom6"),
            "modèle Sharpe": p["pred_sharpe"], "modèle futurs leaders": p["pred_mom6n"],
            "modèle faible volatilité": -p["pred_vol"],
            "modèle leaders + faible volatilité": r("pred_mom6n") + r("pred_vol", -1),
            "modèle Sharpe + leaders + faible volatilité": r("pred_sharpe") + r("pred_mom6n") + r("pred_vol", -1),
            "momentum 6 mois": p["mom6"], "faible volatilité 63 j": -p["vol63_d"],
            "momentum / volatilité": p["mom6"] / p["vol63_d"], "grosses capitalisations": p["cap"]}


def prepare(p, opens):
    """→ périodes [(dates, rendements relatifs depuis l'entrée (jours × titres))] et lignes de chaque décision."""
    col = {s: j for j, s in enumerate(opens.columns)}
    px = opens.to_numpy(float)
    pos = {d: i for i, d in enumerate(opens.index)}
    periods, groups = [], []
    for _, g in p.groupby("date", sort=True):
        e, x = pos[g["entry"].iloc[0]], pos[g["exit"].iloc[0]]
        periods.append((opens.index[e:x + 1], px[e:x + 1] / px[e]))
        groups.append(g.assign(j=g["symbol"].map(col).to_numpy()))
    return periods, groups


def weights(top, scheme):
    n = len(top)
    iv = 1 / top["vol63_d"].fillna(top["vol63_d"].median()).clip(lower=1e-4).to_numpy()
    lin = np.arange(n, 0, -1, dtype=float)
    w = {"égal": np.ones(n), "inverse volatilité": iv, "score": lin, "score × inverse volatilité": lin * iv,
         "capitalisation": top["cap"].fillna(top["cap"].median()).to_numpy()}[scheme]
    return w / w.sum()


def simulate(periods, picks, n_sym, fee=C.FEE):
    v, prev, vals, dates, turn = 1.0, np.zeros(n_sym), [], [], []
    for (d, rel), (j, w) in zip(periods, picks):
        full = np.zeros(n_sym)
        full[j] = w
        tr = np.abs(full - prev).sum()
        v *= 1 - fee * tr
        turn.append(tr / 2)
        path = rel[:, j] @ w
        vals.append(v * path[:-1])
        dates.append(d[:-1])
        prev = np.zeros(n_sym)
        prev[j] = w * rel[-1, j] / path[-1]
        v *= path[-1]
    return pd.Series(np.concatenate(vals), index=np.concatenate(dates)), float(np.mean(turn[1:]))


def stats(v):
    r = v.pct_change().dropna()

    def sh(x):
        return x.mean() / x.std() * np.sqrt(252) if len(x) > 20 and x.std() > 0 else np.nan
    years = (v.index[-1] - v.index[0]).days / 365.25
    return dict(sharpe_2019_23=sh(r[r.index < SPLIT]), sharpe_2024_26=sh(r[r.index >= SPLIT]), sharpe=sh(r),
                rendement_an_pct=100 * ((v.iloc[-1] / v.iloc[0]) ** (1 / years) - 1),
                max_dd_pct=100 * (v / v.cummax() - 1).min(), volatilité_pct=100 * r.std() * np.sqrt(252))


def main():
    opens = pd.read_parquet(M.OPENS)
    bench = pd.read_parquet(M.BENCH)
    rows, uni = [], None
    for freq, path, per_year in (("mensuelle", M.PREDS, 12), ("hebdomadaire", M.PREDS_W, 52)):
        p = pd.read_parquet(path)
        p = p[p["exit"].notna()].reset_index(drop=True)
        sc = scores(p)
        periods, groups = prepare(p, opens)
        if uni is None:
            ref = bench["ro"].loc[periods[0][0][0]:periods[-1][0][-1]]
            uni = stats((1 + ref.fillna(0)).cumprod().shift(1, fill_value=1.0))
        for name, s in sc.items():
            p["_s"] = s
            tops = [g.assign(_s=p.loc[g.index, "_s"]).nlargest(max(NS), "_s") for g in groups]
            for n in NS:
                for scheme in WEIGHTS:
                    picks = [(t["j"].to_numpy()[:n], weights(t.iloc[:n], scheme)) for t in tops]
                    v, turn = simulate(periods, picks, opens.shape[1])
                    rows.append(dict(classement=name, top=n, allocation=scheme, fréquence=freq,
                                     rotation_an_x=turn * per_year, **stats(v)))
        print(f"fréquence {freq} : {len(groups)} décisions", flush=True)
    out = pd.DataFrame(rows)
    out["bat_univers"] = (out["sharpe_2019_23"] > uni["sharpe_2019_23"]) & (out["sharpe_2024_26"] > uni["sharpe_2024_26"])
    out.to_csv("myfiles/myqtm_index_sharpe.csv", index=False)
    pd.set_option("display.width", 250, "display.float_format", "{:.2f}".format, "display.max_rows", 400)
    print(f"\nunivers équipondéré : Sharpe 2019-23 {uni['sharpe_2019_23']:.2f}, 2024-26 {uni['sharpe_2024_26']:.2f}, "
          f"ensemble {uni['sharpe']:.2f} ; rendement {uni['rendement_an_pct']:.1f} %/an, DD {uni['max_dd_pct']:.1f} %")
    eq = out[out["allocation"] == "égal"]
    for col in ("sharpe_2019_23", "sharpe_2024_26", "sharpe"):
        print(f"\n{col} — parts égales, par classement, taille et fréquence")
        print(eq.pivot_table(index="classement", columns=["fréquence", "top"], values=col).reindex(MODELS + BASES).to_string())
    print("\nSharpe (ensemble) moyen par allocation et classement (moyenne des tailles et des fréquences)")
    print(out.pivot_table(index="classement", columns="allocation", values="sharpe").reindex(MODELS + BASES)[list(WEIGHTS)].to_string())
    print("\nSharpe moyen par fréquence et classement (moyenne des tailles et allocations) ; rotation annuelle moyenne")
    print(out.pivot_table(index="classement", columns="fréquence", values=["sharpe", "rotation_an_x"]).reindex(MODELS + BASES).to_string())
    cols = ["classement", "top", "allocation", "fréquence", "sharpe_2019_23", "sharpe_2024_26", "sharpe", "rendement_an_pct",
            "max_dd_pct", "volatilité_pct", "rotation_an_x"]
    print(f"\n15 meilleures configurations sur 2019-2023 (choix), avec leur vérification 2024-26 — {len(out)} configurations")
    print(out.sort_values("sharpe_2019_23", ascending=False).head(15)[cols].to_string(index=False))
    win = out[out["bat_univers"]].copy()
    win["marge_min"] = np.minimum(win["sharpe_2019_23"] - uni["sharpe_2019_23"], win["sharpe_2024_26"] - uni["sharpe_2024_26"])
    print(f"\nconfigurations qui battent l'univers sur les deux périodes : {len(win)} / {len(out)} "
          f"(par classement : {win['classement'].value_counts().to_dict()})")
    print(win.sort_values("marge_min", ascending=False).head(20)[cols + ["marge_min"]].to_string(index=False))


if __name__ == "__main__":
    main()
