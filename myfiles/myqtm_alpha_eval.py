"""Mesure de l'alpha (et non du Sharpe brut) du classifieur myqtm, sur l'univers Nasdaq ou S&P 500 (--sp).

Règles évaluées sur les prédictions hors échantillon (2024 → fin) :
- moteur ref/MyQTM avec les seuils par défaut (trade.simulate, 12 positions max) ;
- top k du jour en lots dynamiques 1/n (myqtm_topk_dynamic.simulate), sortie EMA 3 < EMA 10, k = 1 … 20 ;
- les mêmes avec le filtre « résultats ≤ 20 j » (earn_days_to_next au jour du signal).
Référence = univers équipondéré, d'ouverture à ouverture (moyenne des membres du jour) : même grille de cours que
les portefeuilles, et la bonne comparaison pour une sélection de titres à parts égales. Par règle :
- bêta, alpha annualisé (r_p − bêta·r_ref) et son t de Student (rendements quotidiens) ;
- ratio d'information (r_p − r_ref) ; alpha par sous-période (2024-25 / 2026) ;
- rappel : CAGR, drawdown, Sharpe, et l'indice pondéré par la capitalisation.
Par trade (compté une fois, à son entrée) : gain net moins le rendement de la référence sur la même durée, moyenne,
médiane et t de Student sur les moyennes hebdomadaires. (Une mesure « par jour » sur les k premiers compterait le
même trade chaque jour où le titre reste en tête et gonflerait l'écart avec quelques très longs trades.)

    PYTHONPATH=.:myfiles python myfiles/myqtm_alpha_eval.py [--sp] [--pred=ts0] [--sfx=_x4m100l20p100] [--avg]
"""
import sys

import numpy as np
import pandas as pd

from myqtm_topk_dynamic import simulate
from myqtm_topk_slots import signals
from src.portefolio_strategy.myqtm import config as C
from src.portefolio_strategy.myqtm import features as F
from src.portefolio_strategy.myqtm import trade as T
from src.portefolio_strategy.myqtm import train_rank as R

KS = (1, 2, 3, 5, 10, 20)
PROBS = ["ybear", "yrange", "ybull"]
SPLIT = pd.Timestamp("2026-01-01")


def load(pred, sfx, avg):
    names = [f"{pred}{s}{sfx}" for s in ("", "s7", "s123")] if avg else [pred + sfx]
    p = pd.concat([pd.read_parquet(C.DATA / f"predictions_{n}_from2024.parquet", columns=["date", "symbol"] + PROBS)
                   for n in names]).groupby(["date", "symbol"], as_index=False)[PROBS].mean()
    f = pd.read_parquet(F.FEATURES, columns=["date", "symbol", "member", "o", "c", "earn_days_to_next"])
    f = f.sort_values(["symbol", "date"]).reset_index(drop=True)
    f["x_ema"] = f.groupby("symbol")["c"].transform(T.ema_exit).astype(bool)
    f["gain"], _ = R.trade_gains(f)
    return p, f


def rel_stats(v, ref):
    """v : valeur du portefeuille par date (cours d'ouverture) ; ref : rendement de la référence de t à t+1."""
    r = (v.shift(-1) / v - 1).dropna()
    b = ref.reindex(r.index).fillna(0.0)
    beta = float(np.cov(r, b)[0, 1] / b.var())
    a, act = r - beta * b, r - b
    years = (v.index[-1] - v.index[0]).days / 365.25
    dd = v / v.cummax() - 1
    out = dict(cagr_pct=100 * ((v.iloc[-1] / v.iloc[0]) ** (1 / years) - 1), max_dd_pct=100 * dd.min(),
               sharpe=r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0.0, beta=beta,
               alpha_an_pct=100 * 252 * a.mean(), alpha_t=a.mean() / a.std() * np.sqrt(len(a)) if a.std() > 0 else 0.0,
               ratio_info=act.mean() / act.std() * np.sqrt(252) if act.std() > 0 else 0.0)
    for name, m in (("alpha_2024_25_pct", a.index < SPLIT), ("alpha_2026_pct", a.index >= SPLIT)):
        out[name] = 100 * 252 * a[m].mean()
    return out


def trade_stats(name, c, opens, level):
    """Gain relatif de chaque trade (une fois par trade) : gain net − rendement de la référence sur la même durée ;
    t de Student sur les moyennes hebdomadaires (semaine d'entrée)."""
    g = np.array([opens.at[x, s] / opens.at[e, s] * (1 - C.FEE) ** 2 - 1
                  for e, x, s in zip(c["entry"], c["exit"], c["symbol"])])
    rel = pd.Series(g - (level.reindex(c["exit"]).to_numpy() / level.reindex(c["entry"]).to_numpy() - 1),
                    index=pd.DatetimeIndex(c["entry"])).dropna()
    out = []
    for per, msk in (("2024-25", rel.index < SPLIT), ("2026", rel.index >= SPLIT), ("2024-26", rel.index >= rel.index.min())):
        y = rel[msk]
        w = y.groupby(y.index.to_period("W")).mean()
        out.append(dict(règle=name, période=per, trades=len(y), rel_moy_pct=100 * y.mean(), rel_méd_pct=100 * y.median(),
                        gagnants_rel_pct=100 * (y > 0).mean(),
                        t_hebdo=w.mean() / w.std() * np.sqrt(len(w)) if len(w) > 2 and w.std() > 0 else np.nan))
    return out


def main():
    if "--sp" in sys.argv:
        from src.portefolio_strategy.myqtm.sp import use_sp
        use_sp()
    pred = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--pred=")), "ts0")
    sfx = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--sfx=")), "")
    avg = "--avg" in sys.argv
    tag = ("sp_" if "--sp" in sys.argv else "ndx_") + pred + sfx + ("_avg" if avg else "")
    p, f = load(pred, sfx, avg)
    start, end = p["date"].min(), p["date"].max()
    opens = f.pivot(index="date", columns="symbol", values="o").sort_index()
    nd = pd.Series(opens.index[1:].append(pd.DatetimeIndex([pd.NaT])), index=opens.index)
    opens = opens.loc[start:end]
    # référence : univers équipondéré, ouverture t → ouverture t+1
    m = f[f["member"]].assign(o_next=f.groupby("symbol")["o"].shift(-1))
    m = m[(m["date"] >= start) & (m["date"] <= end)]
    ref = (m["o_next"] / m["o"] - 1).groupby(m["date"]).mean()
    p = p.merge(f[["date", "symbol", "o", "c", "x_ema", "earn_days_to_next", "gain"]], on=["date", "symbol"])
    p["next_date"] = p["date"].map(nd)
    p = p.sort_values(["symbol", "date"]).reset_index(drop=True)
    rows, per_trade, opens_f = [], [], opens.ffill()
    ew = (1 + ref.fillna(0)).cumprod().shift(1, fill_value=1.0)
    rows.append(dict(règle="univers équipondéré", trades=np.nan,
                     **{**rel_stats(ew, ref), "alpha_t": 0.0, "ratio_info": 0.0}))
    idx = pd.read_parquet(F.INDEX)["level"].loc[start:end]
    rows.append(dict(règle="indice pondéré capitalisation", trades=np.nan, **rel_stats(idx, ref)))
    td = T.build_trade_data(p[["date", "symbol", "o", "c"] + PROBS].assign(ema_exit=p["x_ema"]))
    res = T.simulate(td, lambda d: dict(C.DEFAULT_PARAMS))
    rows.append(dict(règle="moteur ref/MyQTM, seuils par défaut", trades=len(res["history"]) + len(res["open_positions"]),
                     **rel_stats(pd.Series(res["values"], index=res["dates"]), ref)))
    for k in KS:
        cand = signals(p, k)
        cand = cand[cand["exit"] <= end].merge(p[["date", "symbol", "earn_days_to_next"]],
                                               left_on=["signal", "symbol"], right_on=["date", "symbol"])
        for name, c in ((f"top {k}", cand), (f"top {k} + résultats ≤ 20 j", cand[cand["earn_days_to_next"] <= 20])):
            if len(c) < 5:
                continue
            v, _, _ = simulate(c, opens)
            rows.append(dict(règle=name, trades=len(c), **rel_stats(v, ref)))
            per_trade += trade_stats(name, c, opens_f, ew)
        print(f"top {k} ok", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(f"myfiles/myqtm_alpha_eval_{tag}.csv", index=False)
    dd = pd.DataFrame(per_trade)
    dd.to_csv(f"myfiles/myqtm_alpha_eval_trades_{tag}.csv", index=False)
    pd.set_option("display.width", 250, "display.float_format", "{:.2f}".format, "display.max_rows", 200)
    print(f"\n{tag} : {start:%Y-%m-%d} → {end:%Y-%m-%d}, membres par jour {m.groupby('date').size().median():.0f}")
    print(out.to_string(index=False))
    print("\ngain relatif par trade (gain du trade, frais compris, moins l'univers équipondéré sur la même durée)")
    print(dd.pivot(index="règle", columns="période", values=["trades", "rel_moy_pct", "rel_méd_pct", "t_hebdo"])
          .reindex([r for k in KS for r in (f"top {k}", f"top {k} + résultats ≤ 20 j")]).to_string())


if __name__ == "__main__":
    main()
