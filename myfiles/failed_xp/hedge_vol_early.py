"""Les donnees de couverture / positionnement institutionnel detectent-elles la volatilite plus tot,
ou separent-elles la vol "importante" de la vol "benigne" ? (QQQ, 2006-2026)

Donnees (myfiles/failed_xp/hedge_data_download.py + data/cboe_putcall.parquet), toutes decalees d'1 seance ;
CFTC (positions du mardi) utilisable a partir du vendredi de publication, puis +1 seance :
  CBOE    : VIX, VIX/VIX3M, VIX9D/VIX, VVIX, VRP = VIX − YZ10, SKEW (+ z 1 an)
  put/call: index, equity, total (MM5 + z 1 an)
  DIX/GEX : SqueezeMetrics (z 1 an, depuis 2011)
  CFTC TFF: E-mini NQ / ES, VIX futures — net % open interest des asset managers, leveraged funds, dealers
            (+ variation 4 semaines)
Cibles :
  surp  : log(vol realisee J+1..J+10 / YZ10 en J)          -> "la vol va-t-elle monter plus que ce qu'on voit ?"
  spike : vol J+1..J+10 >= 1.5·YZ10 ET >= 25 %              (binaire)
  bad   : sur les jours "stress" (YZ10 >= ABOVE_CAP = 15 %, le cap coupe deja) : min(prix J+1..J+21)/prix J − 1
          < −8 %  -> vol "importante" (suivie d'une vraie baisse) vs "benigne"
Valeur AJOUTEE uniquement : base = ce que la strategie sait deja (log YZ10, log vol60, ecart SMA250, r5, r21).
Validation croisee 2 moities (H1 2006-2015 / H2 2016-2026) : modele base vs base + variable appris sur une
moitie, score sur l'autre (Δ R² hors echantillon pour surp, Δ AUC pour spike / bad). Une variable n'est
retenue que si le gain est > 0 dans les DEUX sens.
Backtest (strategie deployee x1.7, frais 0.5 %) : la YZ10 de la formule est remplacee par une PREVISION
YZ10·exp(pred_surp − moyenne train), cross-fit 2 moities ; controle = meme chose avec le modele base seul.
Variante "YZ10 x delta" : YZ10·exp(pred_base+variable − pred_base) = seul l'apport propre de la variable,
sans le retour a la moyenne du modele base (les seuils ABOVE_CAP / VOL_TARGET sont calibres sur YZ10 brute).
    python -W ignore -m myfiles.failed_xp.hedge_vol_early      (log : myfiles/failed_xp/hedge_vol_early.log)

Resultats (2026-11-09) — NEGATIF :
  - Base prix seule : R² surprise 0.05/0.15, AUC pic 0.69/0.67, AUC importante/benigne 0.58/0.60 (les prix
    separent mal la vol importante de la benigne... et les donnees de couverture n'y changent rien).
  - Meilleurs gains robustes : +0.01 R² (CFTC asset managers ES), +0.01 AUC pic (CFTC NQ), +0.01/+0.03 AUC
    importante/benigne (put/call index z). 7-8 variables "robustes" sur 34 par cible = ce que le hasard donne
    (P(gain > 0 dans les 2 sens | aucune info) ~ 25 %). VIX, VVIX, SKEW, GEX, DIX : rien au-dela de YZ10.
  - Backtest 2006-2026 : ref YZ10 Calmar 0.70 ; aucune variante ne fait mieux (meilleure : YZ10 x delta
    put/call total = 0.67 ; blocs CBOE / couverture 0.58-0.64). Remplacer YZ10 par une prevision, meme prix
    seuls, degrade (0.52) : les seuils de la formule sont calibres sur la vol realisee brute.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import src.risk_off_strategy.strategy as st
from src.risk_off_strategy.data import load_funding, load_macro, load_ohlc
from src.risk_off_strategy.strategy import CPI_OFF, E_MAX, NFCI_OFF, simulate_net

D = "data/hedge_data"
START = "2006-07-01"
SPLIT = pd.Timestamp("2016-01-01")
BASE = ["b_lyz", "b_lrv60", "b_gap", "b_r5", "b_r21"]


def z(s, n=252):
    return (s - s.rolling(n, min_periods=n // 2).mean()) / s.rolling(n, min_periods=n // 2).std()


def daily(s, idx, lag=1):
    s = s.copy()
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s.reindex(s.index.union(idx)).ffill().reindex(idx).shift(lag)


def features(o, idx):
    p = o["close"]
    r = np.log(p).diff()
    yz = pd.Series(st.yang_zhang_vol(o["open"].values, o["high"].values, o["low"].values, p.values), idx)
    F = pd.DataFrame(index=idx)
    F["b_lyz"] = np.log(yz)
    F["b_lrv60"] = np.log(r.rolling(60).std() * np.sqrt(252))
    F["b_gap"] = p / p.rolling(250).mean() - 1
    F["b_r5"] = p.pct_change(5)
    F["b_r21"] = p.pct_change(21)

    c = pd.read_parquet(f"{D}/cboe_idx.parquet")
    vix, vix3m, vix9d = (daily(c[k], idx) for k in ("vix", "vix3m", "vix9d"))
    F["vix"] = np.log(vix)
    F["vix_vix3m"] = vix / vix3m
    F["vix9d_vix"] = vix9d / vix
    F["vvix"] = daily(c["vvix"], idx)
    F["vrp"] = vix / 100 - yz.shift(1)
    F["skew"] = daily(c["skew"], idx)
    F["skew_z"] = z(F["skew"])

    pc = pd.read_parquet("data/cboe_putcall.parquet")
    for k in ("pc_index", "pc_equity", "pc_total"):
        s = daily(pc[k], idx).rolling(5, min_periods=3).mean()
        F[k] = s
        F[k + "_z"] = z(s)

    dx = pd.read_parquet(f"{D}/dix.parquet")
    F["dix_z"] = z(daily(dx["dix"], idx).rolling(5, min_periods=3).mean())
    gex = daily(dx["gex"], idx)
    F["gex_z"] = z(gex)
    F["gex_neg"] = (gex < 0).astype(float).where(gex.notna())

    for k in ("nq", "es", "vix"):
        d = pd.read_parquet(f"{D}/cftc_{k}.parquet")
        d.index = d.index + pd.Timedelta(days=3)          # mardi -> publication vendredi
        oi = d["open_interest_all"]
        for who, lo, sh in (("am", "asset_mgr_positions_long", "asset_mgr_positions_short"),
                            ("lev", "lev_money_positions_long", "lev_money_positions_short"),
                            ("dlr", "dealer_positions_long_all", "dealer_positions_short_all")):
            net = (d[lo] - d[sh]) / oi
            F[f"cot_{k}_{who}"] = daily(net, idx)
            F[f"cot_{k}_{who}_d4"] = daily(net - net.shift(4), idx)
    return F, yz, p


def targets(p, yz, idx):
    r = np.log(p).diff()
    fwd_vol = r[::-1].rolling(10).std()[::-1].shift(-1) * np.sqrt(252)        # vol J+1..J+10
    fwd_min = p[::-1].rolling(21).min()[::-1].shift(-1) / p - 1                # pire point J+1..J+21
    T = pd.DataFrame(index=idx)
    T["surp"] = np.log(fwd_vol / yz)
    T["spike"] = ((fwd_vol >= 1.5 * yz) & (fwd_vol >= 0.25)).astype(float).where(fwd_vol.notna())
    stress = yz >= st.ABOVE_CAP
    T["bad"] = (fwd_min < -0.08).astype(float).where(fwd_min.notna() & stress)
    return T


def fit_score(Xtr, ytr, Xte, yte, binary):
    sc = StandardScaler().fit(Xtr)
    if binary:
        if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
            return np.nan
        m = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(Xtr), ytr)
        return roc_auc_score(yte, m.predict_proba(sc.transform(Xte))[:, 1])
    m = LinearRegression().fit(sc.transform(Xtr), ytr)
    pred = m.predict(sc.transform(Xte))
    return 1 - np.mean((yte - pred) ** 2) / np.mean((yte - yte.mean()) ** 2)


def incremental(F, T, feats):
    rows = []
    for f in feats:
        for tgt in ("surp", "spike", "bad"):
            d = pd.concat([F[BASE + [f]], T[tgt]], axis=1).dropna()
            d = d[d.index >= START]
            h1, h2 = d[d.index < SPLIT], d[d.index >= SPLIT]
            if len(h1) < 150 or len(h2) < 150:
                continue
            binary = tgt != "surp"
            r = {"variable": f, "cible": tgt, "n": len(d), "debut": d.index[0].year}
            for name, tr, te in (("H1->H2", h1, h2), ("H2->H1", h2, h1)):
                b = fit_score(tr[BASE].values, tr[tgt].values, te[BASE].values, te[tgt].values, binary)
                bf = fit_score(tr[BASE + [f]].values, tr[tgt].values, te[BASE + [f]].values, te[tgt].values, binary)
                r[f"base {name}"] = round(b, 3)
                r[f"gain {name}"] = round(bf - b, 3)
            # signe de la correlation de rang avec le residu de la base (lisibilite : + = annonce plus de risque)
            res = d[tgt] - LinearRegression().fit(d[BASE], d[tgt]).predict(d[BASE])
            r["IC residu"] = round(pd.Series(d[f]).rank().corr(res.rank()), 3)
            r["robuste"] = bool(r["gain H1->H2"] > 0 and r["gain H2->H1"] > 0)
            rows.append(r)
    return pd.DataFrame(rows)


def crossfit_surp(F, T, cols):
    """Prevision de surprise de vol cross-fit : chaque moitie predite par le modele appris sur l'autre,
    centree sur la moyenne des predictions de la moitie d'apprentissage. NaN hors couverture."""
    d = pd.concat([F[cols], T["surp"]], axis=1)
    d = d[d.index >= START]
    X = d[cols].dropna()
    out = pd.Series(np.nan, index=F.index)
    for tr_mask, te_mask in ((X.index < SPLIT, X.index >= SPLIT), (X.index >= SPLIT, X.index < SPLIT)):
        tr = d.loc[X.index[tr_mask]].dropna()
        sc = StandardScaler().fit(tr[cols])
        m = LinearRegression().fit(sc.transform(tr[cols]), tr["surp"])
        mu = m.predict(sc.transform(tr[cols])).mean()
        te_idx = X.index[te_mask]
        out.loc[te_idx] = m.predict(sc.transform(X.loc[te_idx, cols])) - mu
    return out


def win(eq, idx, a, b):
    m = (idx >= pd.Timestamp(a)) & (idx < pd.Timestamp(b))
    e = eq[m] / eq[m][0]
    cagr = e[-1] ** (252 / m.sum()) - 1
    dd = (e / np.maximum.accumulate(e) - 1).min()
    return 100 * cagr, 100 * dd, cagr / abs(dd)


def backtest(o, idx, yz, preds):
    p = o["close"].values.astype(float)
    nfci, cpi = load_macro(idx)
    fund = load_funding(idx)
    macro = (np.nan_to_num(nfci, nan=-9) > NFCI_OFF) | (np.nan_to_num(cpi, nan=-9) > CPI_OFF)
    rows = []
    for name, pr in preds.items():
        mult = np.ones(len(idx)) if pr is None else np.exp(np.clip(np.nan_to_num(pr.values, nan=0.0), -0.5, 0.6))
        c = st.allocation_components(p, yz.values * mult)
        a = np.where(macro, 0.0, np.clip(np.where(c["arm_above"], c["above"], c["below"]), 0, 1))
        eq, fees, rev = simulate_net(o["close"], a, leverage=2, funding=fund, e_max=E_MAX)
        r = {"vol utilisee": name, "frais %/an": round(fees, 2), "revis/an": round(rev, 1)}
        for h, (x, y) in {"H1 06-15": (START, SPLIT), "H2 16-26": (SPLIT, "2100"), "2006-26": (START, "2100")}.items():
            cg, dd, cal = win(eq, idx, x, y)
            r.update({f"{h} %/an": round(cg, 1), f"{h} maxDD": round(dd, 1), f"{h} Calmar": round(cal, 2)})
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    o = load_ohlc("QQQ", "2000-01-01", None)
    idx = o.index
    F, yz, p = features(o, idx)
    T = targets(p, yz, idx)
    feats = [c for c in F.columns if not c.startswith("b_")]
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 300)

    print(f"Jours stress (YZ10 >= {st.ABOVE_CAP:.0%}) depuis {START} : {int(T['bad'].loc[START:].notna().sum())}, "
          f"dont 'importants' (baisse > 8 % sous 21 j) : {int(T['bad'].loc[START:].sum())}")
    print(f"Pics de vol : {int(T['spike'].loc[START:].sum())} jours sur {int(T['spike'].loc[START:].notna().sum())}\n")

    inc = incremental(F, T, feats)
    inc.to_csv("myfiles/failed_xp/hedge_vol_early.csv", index=False)
    base_rows = inc.groupby("cible")[["base H1->H2", "base H2->H1"]].first()
    print("=== Score du modele BASE seul (prix : YZ10, vol60, ecart SMA, r5, r21) ===")
    print(base_rows.to_string())
    for tgt, lab in (("surp", "SURPRISE DE VOL (Δ R² hors echantillon)"), ("spike", "PIC DE VOL (Δ AUC)"),
                     ("bad", "VOL IMPORTANTE vs BENIGNE sur jours stress (Δ AUC)")):
        g = inc[inc.cible == tgt].copy()
        g["gain min"] = g[["gain H1->H2", "gain H2->H1"]].min(axis=1)
        print(f"\n=== {lab} — trie par le gain du pire sens ===")
        print(g.sort_values("gain min", ascending=False)[["variable", "debut", "n", "gain H1->H2", "gain H2->H1",
                                                         "gain min", "IC residu", "robuste"]].to_string(index=False))

    # backtest : top variables robustes sur 'surp' (+ blocs thematiques)
    g = inc[(inc.cible == "surp") & inc.robuste].copy()
    g["gain min"] = g[["gain H1->H2", "gain H2->H1"]].min(axis=1)
    top = g.sort_values("gain min", ascending=False).variable.head(4).tolist()
    blocks = {
        "CBOE vol (VIX, terme, VVIX, VRP)": ["vix", "vix_vix3m", "vvix", "vrp"],
        "couverture pure (SKEW, put/call, CFTC)": ["skew_z", "pc_index_z", "pc_equity_z", "cot_es_am", "cot_nq_am",
                                                   "cot_vix_lev", "cot_es_lev_d4", "cot_vix_lev_d4"],
    }
    pb = crossfit_surp(F, T, BASE)
    preds = {"YZ10 (ref deployee)": None, "prevision base (prix seuls)": pb}
    extra = {f: [f] for f in top}
    extra.update(blocks)
    for name, cols in extra.items():
        pf = crossfit_surp(F, T, BASE + cols)
        preds[f"prevision base + {name}"] = pf
        # apport PROPRE de la variable : YZ10 · exp(pred_avec − pred_base) — garde l'echelle YZ10 calibree
        preds[f"YZ10 x delta {name}"] = pf - pb
    bt = backtest(o, idx, yz, preds)
    print("\n=== Backtest x1.7 net (frais 0.5 %) : YZ10 remplacee par une prevision de vol cross-fit ===")
    print(bt.to_string(index=False))
    bt.to_csv("myfiles/failed_xp/hedge_vol_early_backtest.csv", index=False)


if __name__ == "__main__":
    main()
