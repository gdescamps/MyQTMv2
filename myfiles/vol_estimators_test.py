"""
Test estimateurs de vol range-based (Parkinson, Garman-Klass, Yang-Zhang) vs
close-to-close, et une variante risk-off plus precoce.

Non-fuite : tous les estimateurs sont causaux (fenetre glissante se terminant au
jour t, O/H/L/C de t connus au close de t). Le backtest applique exec_lag=1
(pos[t]=alloc[t-1]) -> la decision du soir s'execute le lendemain.

Usage: python myfiles/vol_estimators_test.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.risk_off_strategy.data import load_macro
from src.risk_off_strategy.strategy import (
    simulate, SMA_LONG, VOL_TARGET, BELOW_SCALE, GAP_CUTOFF, NFCI_OFF, CPI_OFF, ANN,
)

# ── Estimateurs de vol (annualises, causaux) ──────────────
def _ann(x):
    return np.sqrt(x * ANN)


def vol_close(c, w=20):
    ret = np.log(c / c.shift(1))
    return _ann((ret ** 2).rolling(w, min_periods=5).mean())


def vol_parkinson(h, l, w=20):
    # sigma^2 = 1/(4 ln2) * mean( ln(H/L)^2 )   (H/L uniquement)
    hl = np.log(h / l) ** 2
    return _ann(hl.rolling(w, min_periods=5).mean() / (4 * np.log(2)))


def vol_garman_klass(o, h, l, c, w=20):
    # 0.5 ln(H/L)^2 - (2ln2-1) ln(C/O)^2
    hl = np.log(h / l) ** 2
    co = np.log(c / o) ** 2
    var = 0.5 * hl - (2 * np.log(2) - 1) * co
    return _ann(var.rolling(w, min_periods=5).mean())


def vol_yang_zhang(o, h, l, c, w=20):
    # overnight + k*open-close + (1-k)*Rogers-Satchell : gere les gaps overnight
    lo = np.log(o / c.shift(1))          # overnight (O_t vs C_{t-1})
    lc = np.log(c / o)                   # open->close
    ho, lo_ = np.log(h / o), np.log(l / o)
    hc, lc_ = np.log(h / c), np.log(l / c)
    rs = ho * (ho - lc) + lo_ * (lo_ - lc)   # Rogers-Satchell (drift-independent)
    # variances glissantes (ddof=1)
    vo = lo.rolling(w, min_periods=5).var()
    vc = lc.rolling(w, min_periods=5).var()
    vrs = rs.rolling(w, min_periods=5).mean()
    k = 0.34 / (1.34 + (w + 1) / (w - 1))
    return _ann(vo + k * vc + (1 - k) * vrs)


# ── Allocation parametrable ───────────────────────────────
def compute_alloc(p, rv, nfci, cpi, above_target=None):
    """Reprend la formule deployee mais avec une vol `rv` fournie.
    above_target : si non-None, applique aussi un cap vol AU-DESSUS de la MA
                   (risk-off precoce) -> clip(above_target/rv,0,1)."""
    p = np.asarray(p, float)
    rv = np.maximum(np.asarray(rv, float), 1e-6)
    s = pd.Series(p).rolling(SMA_LONG, min_periods=1).mean().values
    decay = np.clip(1.0 + (p / s - 1.0) / GAP_CUTOFF, 0, 1)
    below = BELOW_SCALE * np.clip(VOL_TARGET / rv, 0, 1) * decay
    if above_target is None:
        above = np.ones_like(p)
    else:
        above = np.clip(above_target / rv, 0, 1)
    alloc = np.clip(np.where(p > s, above, below), 0, 1)
    if nfci is not None:
        alloc = np.where(np.nan_to_num(nfci, nan=-9) > NFCI_OFF, 0.0, alloc)
    if cpi is not None:
        alloc = np.where(np.nan_to_num(cpi, nan=-9) > CPI_OFF, 0.0, alloc)
    return alloc


def fmt(name, m):
    return (f"{name:<26s} CAGR {m['cagr']*100:5.1f}%  maxDD {m['maxdd']*100:6.1f}%  "
            f"Sharpe {m['sharpe']:.2f}  Calmar {m['calmar']:5.2f}  "
            f"TiM {m['time_in_market']*100:4.0f}%  turn {m['turnover']:.1f}")


def main():
    df = pd.read_parquet(ROOT / "data" / "QQQ.parquet").sort_index()
    df = df.loc["2000-01-01":]
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    dates = df.index
    nfci, cpi = load_macro(dates)

    ests = {
        "cc-20 (baseline)": vol_close(c, 20),
        "cc-10":            vol_close(c, 10),
        "parkinson-10":     vol_parkinson(h, l, 10),
        "garman_klass-10":  vol_garman_klass(o, h, l, c, 10),
        "yang_zhang-10":    vol_yang_zhang(o, h, l, c, 10),
        "yang_zhang-20":    vol_yang_zhang(o, h, l, c, 20),
    }
    # bfill pour la fenetre initiale (comme realized_vol), reste causal ensuite
    ests = {k: v.bfill().values for k, v in ests.items()}

    price = c.values
    print("=" * 100)
    print("A. Drop-in dans la formule deployee (vol utilisee SOUS la MA uniquement)")
    print("=" * 100)
    base_alloc = compute_alloc(price, ests["cc-20 (baseline)"], nfci, cpi)
    print(fmt("baseline (cc-20)", simulate(price, base_alloc)))
    print("-" * 100)
    for name, rv in ests.items():
        if name == "cc-20 (baseline)":
            continue
        a = compute_alloc(price, rv, nfci, cpi)
        print(fmt(name, simulate(price, a)))

    # ── Precocite : nb de jours d'avance pour franchir un seuil de vol ──
    print()
    print("=" * 100)
    print("B. Precocite au demarrage des crises : 1er jour ou vol >= seuil (0.30 annualise)")
    print("   avance (j) = combien de jours AVANT le baseline cc-20 chaque estimateur alerte")
    print("=" * 100)
    THR = 0.30
    crises = {
        "Dotcom 2000":  ("2000-07-01", "2001-04-01"),
        "GFC 2008":     ("2007-10-01", "2008-10-01"),
        "Covid 2020":   ("2020-01-15", "2020-03-15"),
        "Rate 2022":    ("2021-11-01", "2022-05-01"),
        "2018Q4":       ("2018-09-01", "2018-12-15"),
    }
    di = pd.Series(np.arange(len(dates)), index=dates)
    for cname, (a0, a1) in crises.items():
        win = (dates >= a0) & (dates <= a1)
        idx = np.where(win)[0]
        def first_cross(rv):
            hits = idx[rv[idx] >= THR]
            return hits[0] if len(hits) else None
        base_i = first_cross(ests["cc-20 (baseline)"])
        print(f"\n{cname}  [{a0} -> {a1}]")
        for name, rv in ests.items():
            fi = first_cross(rv)
            if fi is None:
                print(f"   {name:<22s} jamais >= {THR}")
            elif base_i is None:
                print(f"   {name:<22s} {dates[fi].date()}  (baseline jamais)")
            else:
                lead = base_i - fi
                tag = f"+{lead}j plus tot" if lead > 0 else (f"{lead}j" if lead < 0 else "= baseline")
                print(f"   {name:<22s} {dates[fi].date()}  ({tag})")

    # ── Variante risk-off precoce : cap vol AUSSI au-dessus de la MA ──
    print()
    print("=" * 100)
    print("C. Risk-off precoce : vol-cap actif AUSSI au-dessus de la MA (Yang-Zhang-10)")
    print("   above_target = vol annualisee au-dela de laquelle on commence a couper meme en tendance")
    print("=" * 100)
    ryz = ests["yang_zhang-10"]
    print(fmt("baseline (cc-20, pas de cap)", simulate(price, base_alloc)))
    print("-" * 100)
    for at in [0.25, 0.30, 0.35, 0.40, 0.50]:
        a = compute_alloc(price, ryz, nfci, cpi, above_target=at)
        print(fmt(f"YZ-10 + above_cap={at:.2f}", simulate(price, a)))


if __name__ == "__main__":
    main()
