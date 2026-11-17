"""Indice concentré + limitation de l'allocation globale lors des chocs de volatilité (comme la stratégie risk-off).

Signal = univers équipondéré du S&P 500 (rendements d'ouverture à ouverture, depuis 2015) :
- « plafond de vol » : exposition = min(1, plafond / volatilité réalisée w séances annualisée), le reste en cash ;
- « risk-off » : la formule de src/risk_off_strategy/strategy.py appliquée à l'univers (allocation_components) —
  au-dessus de sa moyenne 250 j montante : min(0,15 / vol 10 j, 1) ; en dessous : 0,8 · min(0,08 / vol, 1) · decay —
  avec ou sans les garde-fous macro (NFCI > 0,5 ou inflation > 7 % → cash).
Exécution : le signal calculé avec les rendements connus à l'ouverture de J−1 est appliqué à partir de l'ouverture
de J (un jour plein de délai) ; bande de non-transaction de BAND sur l'exposition ; frais 0,05 % sur |Δ exposition| ;
cash rémunéré à 0. Appliqué à l'univers lui-même et à quelques indices de myqtm_index_sharpe.py.
Protocole : paramètres choisis sur 2019-2023, vérifiés sur 2024 → fin.

    PYTHONPATH=.:myfiles python myfiles/myqtm_index_vol_overlay.py
"""
import numpy as np
import pandas as pd

import myqtm_index_sharpe as IS
from src.portefolio_strategy.myqtm import config as C
from src.portefolio_strategy.myqtm import monthly_rank as M
from src.risk_off_strategy import data as RD
from src.risk_off_strategy import strategy as RS

BAND = 0.10
CAPS = (0.12, 0.15, 0.18, 0.20, 0.25)
WINDOWS = (10, 20)
BASES = [("univers équipondéré", None, None, None),
         ("20 plus grosses, parts égales", "grosses capitalisations", 20, "égal"),
         ("20 plus grosses, inverse volatilité", "grosses capitalisations", 20, "inverse volatilité"),
         ("30 plus grosses, parts égales", "grosses capitalisations", 30, "égal"),
         ("futurs leaders top 10, selon le score", "modèle futurs leaders", 10, "score"),
         ("futurs leaders top 20, parts égales", "modèle futurs leaders", 20, "égal"),
         ("leaders parmi les 50 plus grosses, top 20", "leaders parmi les 50 plus grosses", 20, "égal"),
         ("momentum 6 mois top 20, parts égales", "momentum 6 mois", 20, "égal")]


def signals(ro):
    """ro : rendements de l'univers (ouverture t → t+1), historique complet → {nom: exposition cible alignée sur ro}."""
    r = ro.fillna(0)
    level = (1 + r).cumprod()
    out = {"sans limitation": pd.Series(1.0, index=ro.index)}
    vol = {w: pd.Series(RS.realized_vol(r.to_numpy(), w), index=ro.index) for w in WINDOWS}
    for w in WINDOWS:
        for cap in CAPS:
            out[f"plafond de vol {cap:.0%} ({w} j)"] = (cap / vol[w]).clip(upper=1.0)
    comp = RS.allocation_components(level.to_numpy(), vol[10].to_numpy())
    ro_alloc = pd.Series(np.where(comp["arm_above"], comp["above"], comp["below"]), index=ro.index)
    out["risk-off (formule complète)"] = ro_alloc
    nfci, cpi = RD.load_macro(ro.index)
    off = np.zeros(len(ro), dtype=bool)
    if nfci is not None:
        off |= np.nan_to_num(np.asarray(nfci, float), nan=-9) > RS.NFCI_OFF
    if cpi is not None:
        off |= np.nan_to_num(np.asarray(cpi, float), nan=-9) > RS.CPI_OFF
    out["risk-off + garde-fous macro"] = ro_alloc.where(~off, 0.0)
    return {k: v.shift(2).fillna(1.0) for k, v in out.items()}     # connu à l'ouverture J−1, appliqué dès l'ouverture J


def overlay(v, target, band=BAND, fee=C.FEE):
    """v : valeur quotidienne de l'indice ; target : exposition cible par date → (valeur, exposition moyenne)."""
    r = (v.shift(-1) / v - 1).fillna(0).to_numpy()
    tg = target.reindex(v.index).fillna(1.0).to_numpy()
    out, e, val, expo = np.ones(len(v)), 1.0, 1.0, []
    for t in range(len(v)):
        if abs(tg[t] - e) >= band or (tg[t] in (0.0, 1.0) and tg[t] != e):
            val *= 1 - fee * abs(tg[t] - e)
            e = tg[t]
        out[t] = val
        expo.append(e)
        val *= 1 + e * r[t]
    return pd.Series(out, index=v.index), float(np.mean(expo))


def base_values(opens, bench):
    p = pd.read_parquet(M.PREDS)
    p = p[p["exit"].notna()].reset_index(drop=True)
    sc = IS.scores(p)
    periods, groups = IS.prepare(p, opens)
    ref = bench["ro"].loc[periods[0][0][0]:periods[-1][0][-1]]
    vals = {}
    for name, ranking, n, scheme in BASES:
        if ranking is None:
            vals[name] = (1 + ref.fillna(0)).cumprod().shift(1, fill_value=1.0).iloc[:-1]
            continue
        p["_s"] = sc[ranking]
        picks = []
        for g in groups:
            t = g.assign(_s=p.loc[g.index, "_s"]).nlargest(n, "_s")
            picks.append((t["j"].to_numpy(), IS.weights(t, scheme)))
        vals[name], _ = IS.simulate(periods, picks, opens.shape[1])
    return vals


def main():
    opens, bench = pd.read_parquet(M.OPENS), pd.read_parquet(M.BENCH)
    sig = signals(bench["ro"])
    vals = base_values(opens, bench)
    rows = []
    for base, v in vals.items():
        for name, tg in sig.items():
            ov, expo = overlay(v, tg)
            rows.append(dict(indice=base, limitation=name, exposition_moy_pct=100 * expo, **IS.stats(ov)))
    out = pd.DataFrame(rows)
    out.to_csv("myfiles/myqtm_index_vol_overlay.csv", index=False)
    pd.set_option("display.width", 250, "display.float_format", "{:.2f}".format, "display.max_rows", 400)
    uni = out[(out["indice"] == "univers équipondéré") & (out["limitation"] == "sans limitation")].iloc[0]
    print(f"univers équipondéré sans limitation : Sharpe 2019-23 {uni['sharpe_2019_23']:.2f}, 2024-26 {uni['sharpe_2024_26']:.2f}, "
          f"ensemble {uni['sharpe']:.2f}, rendement {uni['rendement_an_pct']:.1f} %/an, DD {uni['max_dd_pct']:.1f} %")
    order = list(sig)
    for col in ("sharpe_2019_23", "sharpe_2024_26", "sharpe", "max_dd_pct", "rendement_an_pct"):
        print(f"\n{col}")
        print(out.pivot(index="limitation", columns="indice", values=col).reindex(order)[[b[0] for b in BASES]].to_string())
    print("\nexposition moyenne (%) par limitation")
    print(out[out["indice"] == "univers équipondéré"].set_index("limitation")["exposition_moy_pct"].reindex(order).round(0).to_string())
    out["bat_univers"] = (out["sharpe_2019_23"] > uni["sharpe_2019_23"]) & (out["sharpe_2024_26"] > uni["sharpe_2024_26"])
    cols = ["indice", "limitation", "sharpe_2019_23", "sharpe_2024_26", "sharpe", "rendement_an_pct", "max_dd_pct",
            "volatilité_pct", "exposition_moy_pct"]
    print(f"\nbattent l'univers (sans limitation) sur les deux périodes : {int(out['bat_univers'].sum())} / {len(out)}")
    print(out[out["bat_univers"]].sort_values("sharpe", ascending=False).head(25)[cols].to_string(index=False))
    print("\nchoix sur 2019-2023 (meilleure limitation par indice) et vérification")
    best = out.loc[out.groupby("indice")["sharpe_2019_23"].idxmax()]
    print(best[cols].to_string(index=False))


if __name__ == "__main__":
    main()
