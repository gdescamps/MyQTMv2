"""Modèles appris (≤ 10 paramètres) : apprentissage sur les ANNÉES PAIRES, validation sur les ANNÉES IMPAIRES.
Backtest net complet (E_MAX 1.5 + saturation, bande 0.50/1.00, frais 0.5 %) ; le score d'un ensemble d'années =
Calmar de la courbe recousue à partir des rendements journaliers de ces années seulement (CAGR / maxDD).
Entrées = variables de la formule : g = prix/SMA250 − 1, sl = pente SMA250 sur 40 j, vol = YZ10 (lv = log), NFCI, IPC.
Modèles :
  FORM8  : la formule experte, ses 8 constantes ré-apprises (SMA, pente, cap, cible vol, échelle, coupure, NFCI, IPC)
  VT5    : c/vol · σ(w0 + w1 g + w2 nf + w3 cpi)
  VT6    : c/vol · σ(w0 + w1 g + w2 sl + w3 nf + w4 cpi)
  L8     : σ(w0 + w1 g + w2 sl + w3 lv + w4 nf + w5 cpi + w6 g·lv + w7 g·sl)
  ARM9   : 2 bras (comme la formule) — au-dessus (g>0 & sl>0) : c1/vol·σ(a0 + a1 g) ; sinon c2/vol·σ(b0 + b1 g) ;
           × σ(m0 − m1·nf − m2·cpi)  -> 9 param.
  VT10   : c/vol · σ(w0 + w1 g + w2 sl + w3 lv + w4 nf + w5 cpi + w6 g·lv + w7 g·sl + w8 g²)  -> 10 param.
    python -m myfiles.pust15_learned_evenodd
"""
import json

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
nf = pd.Series(NF).ffill().fillna(-0.5).values
cpi = pd.Series(CP).ffill().fillna(2.5).values
Z = {k: (v - v.mean()) / v.std() for k, v in dict(g=G, sl=SL, lv=np.log(VOL), nf=nf, cpi=cpi).items()}
EVEN = (IDX.year % 2 == 0)
ODD = ~EVEN
NFm = np.nan_to_num(np.asarray(NF, float), nan=-9)
CPm = np.nan_to_num(np.asarray(CP, float), nan=-9)
CS = np.cumsum(np.concatenate([[0.0], C]))


def sig(x):
    return 1 / (1 + np.exp(-np.clip(x, -50, 50)))


def formula(w):
    sma, k, cap, vt, scale, cut, n0, c0 = w
    sma, k = int(round(sma)), int(round(k))
    i = np.arange(len(C))
    lo = np.maximum(0, i - sma + 1)
    s = (CS[i + 1] - CS[lo]) / (i + 1 - lo)
    up = np.ones(len(C), bool)
    up[k:] = s[k:] > s[:-k]
    gap = C / s - 1
    above = np.minimum(cap / VOL, 1)
    below = scale * np.minimum(vt / VOL, 1) * np.clip(1 + gap / cut, 0, 1)
    a = np.clip(np.where((C > s) & up, above, below), 0, 1)
    return np.where((NFm > n0) | (CPm > c0), 0.0, a)


def arm9(w):
    c1, a0, a1, c2, b0, b1, m0, m1, m2 = w
    up = (G > 0) & (SL > 0)
    a = np.where(up, np.minimum(1, c1 / VOL) * sig(a0 + a1 * Z["g"]), np.minimum(1, c2 / VOL) * sig(b0 + b1 * Z["g"]))
    return a * sig(m0 - m1 * Z["nf"] - m2 * Z["cpi"])


MODELS = {
    "FORM8": (formula, [(150, 300), (10, 80), (0.08, 0.30), (0.04, 0.20), (0.4, 1.0), (0.05, 0.40), (0.0, 1.5), (4.0, 10.0)]),
    "VT5": (lambda w: np.minimum(1, w[0] / VOL) * sig(w[1] + w[2] * Z["g"] + w[3] * Z["nf"] + w[4] * Z["cpi"]),
            [(0.03, 0.6)] + [(-10, 10)] * 4),
    "VT6": (lambda w: np.minimum(1, w[0] / VOL) * sig(w[1] + w[2] * Z["g"] + w[3] * Z["sl"] + w[4] * Z["nf"] + w[5] * Z["cpi"]),
            [(0.03, 0.6)] + [(-10, 10)] * 5),
    "L8": (lambda w: sig(w[0] + w[1] * Z["g"] + w[2] * Z["sl"] + w[3] * Z["lv"] + w[4] * Z["nf"] + w[5] * Z["cpi"]
                         + w[6] * Z["g"] * Z["lv"] + w[7] * Z["g"] * Z["sl"]), [(-10, 10)] * 8),
    "ARM9": (arm9, [(0.03, 0.6), (-10, 10), (-10, 10), (0.03, 0.6), (-10, 10), (-10, 10), (-10, 10), (-10, 10), (-10, 10)]),
    "VT10": (lambda w: np.minimum(1, w[0] / VOL) * sig(w[1] + w[2] * Z["g"] + w[3] * Z["sl"] + w[4] * Z["lv"] + w[5] * Z["nf"]
                                                       + w[6] * Z["cpi"] + w[7] * Z["g"] * Z["lv"] + w[8] * Z["g"] * Z["sl"]
                                                       + w[9] * Z["g"] ** 2), [(0.03, 0.6)] + [(-10, 10)] * 9),
}
FORM_DEFAULT = [250, 40, 0.15, 0.08, 0.8, 0.15, 0.5, 7.0]


def run(al):
    return st.simulate_net(P, al, leverage=2, funding=FUND, e_max=EM, sat_gap=st.SAT_GAP_E)


def stitched(eq, mask):
    r = np.diff(eq, prepend=eq[0]) / np.concatenate([[eq[0]], eq[:-1]])
    e = np.cumprod(1 + r[mask])
    yrs = mask.sum() / 252
    cg = (e[-1] ** (1 / yrs) - 1) * 100
    dd = (e / np.maximum.accumulate(e) - 1).min() * 100
    rr = r[mask]
    return cg, dd, rr.mean() / rr.std() * np.sqrt(252), cg / -dd


def obj(w, f):
    return -stitched(run(f(w))[0], EVEN)[3]


def row(label, k, al):
    eq, fz, rv = run(al)
    r = {"modèle": label, "param.": k, "frais/an": round(fz, 2), "ordres/an": round(rv, 1)}
    for h, m in (("PAIRES", EVEN), ("IMPAIRES", ODD)):
        cg, dd, sh, cal = stitched(eq, m)
        r.update({f"{h} %/an": round(cg, 1), f"{h} DD": round(dd, 1), f"{h} Sh": round(sh, 2), f"{h} Cal": round(cal, 2)})
    for h, (a, b) in {"tout": ("2000-01-01", None), "10ans": (str(IDX[-2520].date()), None),
                      "5ans": (str(IDX[-1260].date()), None)}.items():
        cg, dd, sh, cal = metr(eq, IDX, a, b)
        r.update({f"{h} %/an": round(cg, 1), f"{h} DD": round(dd, 1), f"{h} Cal": round(cal, 2)})
    return r


def main():
    rows = [row("FORMULE experte (constantes actuelles)", 8, formula(FORM_DEFAULT))]
    assert np.allclose(formula(FORM_DEFAULT), st.compute_allocation(C, NF, CP, high=o.high.values, low=o.low.values,
                                                                    open_=o.open.values))
    params = {}
    for name, (f, b) in MODELS.items():
        res = differential_evolution(obj, b, args=(f,), seed=0, maxiter=100, popsize=15, tol=1e-7, polish=False)
        params[name] = np.round(res.x, 3).tolist()
        rows.append(row(f"{name} appris (paires)", len(b), f(res.x)))
        print(f"done {name}: {params[name]}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv("myfiles/pust15_learned_evenodd.csv", index=False)
    json.dump(params, open("myfiles/pust15_learned_evenodd_params.json", "w"), indent=1)
    pd.set_option("display.width", 340)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
