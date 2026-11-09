"""Remplacer la règle experte par un modèle APPRIS à très peu de paramètres (moins que la formule).
Entrées = mêmes variables que la formule (calculées au close t, exécution J+1 par simulate_net) :
  g   = prix/SMA250 − 1           (écart à la MA)
  sl  = SMA250 / SMA250[t−40] − 1 (pente de la MA, continue)
  lv  = log(vol Yang-Zhang 10j)
  nf  = NFCI (lag 5),  cpi = IPC YoY (lag 15)
Formule experte = 8 constantes (SMA 250, pente 40, cap 0.15, cible vol 0.08, échelle 0.8, coupure 0.15,
NFCI 0.5, IPC 7) + fenêtre vol 10. Les modèles réutilisent SMA250 / pente 40 / YZ10 comme entrées (fixes).
Modèles (paramètres appris) :
  L4  : σ(w0 + w1·g + w2·lv + w3·nf)                              4 param.
  L6  : σ(w0 + w1·g + w2·sl + w3·lv + w4·nf + w5·cpi)             6 param.
  VT5 : min(1, c/vol) · σ(w0 + w1·g + w2·nf + w3·cpi)              5 param.
  VT6 : min(1, c/vol) · σ(w0 + w1·g + w2·sl + w3·nf + w4·cpi)      6 param.
Exécution = déploiement étudié : E_MAX 1.5 + saturation, bande 0.50/1.00, frais 0.5 %/côté.
Objectifs : (a) Calmar net 2000-2026 ; (b) CAGR net avec maxDD ≤ 30 %.
Apprentissage 2000-2026 = backtest 2000-2026 (in-sample assumé). En plus, pour info : validation croisée
2 plis (apprendre H1 -> tester H2, et l'inverse).
    python -m myfiles.pust15_learned_model
"""
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

import src.risk_off_strategy.strategy as st
from myfiles.pust17_emax_saturate import metr
from src.risk_off_strategy.data import load_funding, load_macro, load_ohlc

EM = 1.5
st.BUY_FEE = st.SELL_FEE = 0.005

o = load_ohlc("QQQ", "2000-01-01", None)
P, IDX = o.close, o.index
C = P.values
NF, CP = load_macro(IDX)
FUND = np.nan_to_num(np.asarray(load_funding(IDX), float))
VOL = st.yang_zhang_vol(o.open.values, o.high.values, o.low.values, C)
S = pd.Series(C).rolling(250, min_periods=1).mean().values
G = C / S - 1
SL = np.zeros_like(S)
SL[40:] = S[40:] / S[:-40] - 1
LV = np.log(VOL)
nf = pd.Series(NF).ffill().fillna(-0.5).values
cpi = pd.Series(CP).ffill().fillna(2.5).values
Z = {k: (v - np.nanmean(v)) / np.nanstd(v) for k, v in dict(g=G, sl=SL, lv=LV, nf=nf, cpi=cpi).items()}
BASE = st.compute_allocation(C, NF, CP, high=o.high.values, low=o.low.values, open_=o.open.values)


def sig(x):
    return 1 / (1 + np.exp(-np.clip(x, -50, 50)))


MODELS = {
    "L4": (4, lambda w: sig(w[0] + w[1] * Z["g"] + w[2] * Z["lv"] + w[3] * Z["nf"])),
    "L6": (6, lambda w: sig(w[0] + w[1] * Z["g"] + w[2] * Z["sl"] + w[3] * Z["lv"] + w[4] * Z["nf"] + w[5] * Z["cpi"])),
    "VT5": (5, lambda w: np.minimum(1, w[0] / VOL) * sig(w[1] + w[2] * Z["g"] + w[3] * Z["nf"] + w[4] * Z["cpi"])),
    "VT6": (6, lambda w: np.minimum(1, w[0] / VOL) * sig(w[1] + w[2] * Z["g"] + w[3] * Z["sl"] + w[4] * Z["nf"] + w[5] * Z["cpi"])),
}


def bounds(name, k):
    b = [(-10, 10)] * k
    if name.startswith("VT"):
        b[0] = (0.03, 0.60)
    return b


def run(al):
    return st.simulate_net(P, al, leverage=2, funding=FUND, e_max=EM, sat_gap=st.SAT_GAP_E)


def score(al, obj, a, b):
    eq = run(al)[0]
    cg, dd, sh, cal = metr(eq, IDX, a, b)
    if obj == "calmar":
        return -cal
    return -(cg - 3.0 * max(0.0, -dd - 30.0))       # CAGR, pénalité au-delà de −30 %


def fit(name, obj, a="2000-01-01", b=None, seed=0):
    k, f = MODELS[name]
    r = differential_evolution(lambda w: score(f(w), obj, a, b), bounds(name, k), seed=seed,
                               maxiter=80, popsize=15, tol=1e-6, polish=False)
    return r.x


def row(label, al, k):
    eq, fz, rv = run(al)
    r = {"modèle": label, "param.": k, "frais/an": round(fz, 2), "ordres/an": round(rv, 1)}
    for h, (a, b) in {"tout": ("2000-01-01", None), "H1": ("2000-01-01", "2012-12-31"), "H2": ("2013-01-01", None),
                      "10ans": (str(IDX[-2520].date()), None), "5ans": (str(IDX[-1260].date()), None)}.items():
        cg, dd, sh, cal = metr(eq, IDX, a, b)
        r.update({f"{h} %/an": round(cg, 1), f"{h} DD": round(dd, 1), f"{h} Sh": round(sh, 2), f"{h} Cal": round(cal, 2)})
    for k_, (a, b) in {"2008": ("2007-10-01", "2009-03-31"), "2020": ("2020-02-01", "2020-04-30"),
                       "2022": ("2021-11-01", "2022-12-31")}.items():
        m = (IDX >= a) & (IDX <= b)
        r[k_] = round(100 * (eq[m][-1] / eq[m][0] - 1))
    return r


def main():
    rows = [row("FORMULE experte", BASE, 8)]
    params = {}
    for obj in ("calmar", "cagr_dd30"):
        for name, (k, f) in MODELS.items():
            w = fit(name, obj)
            params[f"{name} {obj}"] = np.round(w, 3).tolist()
            rows.append(row(f"{name} [{obj}]", f(w), k))
            # validation croisée 2 plis (info)
            w1 = fit(name, obj, "2000-01-01", "2012-12-31")
            w2 = fit(name, obj, "2013-01-01", None)
            e1, e2 = run(f(w1))[0], run(f(w2))[0]
            rows[-1]["CV H1→H2 Cal"] = round(metr(e1, IDX, "2013-01-01", None)[3], 2)
            rows[-1]["CV H2→H1 Cal"] = round(metr(e2, IDX, "2000-01-01", "2012-12-31")[3], 2)
            print(f"done {name} {obj}: {params[f'{name} {obj}']}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv("myfiles/pust15_learned_model.csv", index=False)
    pd.Series(params).to_json("myfiles/pust15_learned_model_params.json")
    pd.set_option("display.width", 340)
    cols = ["modèle", "param.", "tout %/an", "tout DD", "tout Sh", "tout Cal", "H1 Cal", "H2 Cal", "10ans %/an", "10ans Cal",
            "5ans %/an", "5ans DD", "frais/an", "ordres/an", "2008", "2020", "2022", "CV H1→H2 Cal", "CV H2→H1 Cal"]
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
