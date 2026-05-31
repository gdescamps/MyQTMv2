"""
Universe diagnostic — model-free test of whether the ETF universe is
decorrelated and suitable for cross-sectional rotation.

Computes, from return data only (no model, no retrain):
  1. Effective number of independent bets  N_eff = (Σλ)² / Σλ²
     of the return-correlation matrix — for the 27 vs 38 ETF universes.
  2. Correlated blocks (hierarchical clustering of 1 - corr).
  3. Per-ETF redundancy: nearest-neighbour correlation.
  4. Per-ETF rotation usefulness: volatility of its cross-sectional rank.

Correlations use WEEKLY returns (robust to non-synchronous US / .DE / .L closes).

Run:  python universe_diagnostic.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from etf import UNIVERSE

DATA    = Path(__file__).resolve().parent.parent / "data"
MYFILES = Path(__file__).resolve().parent.parent / "knowledge"

def load_returns() -> pd.DataFrame:
    rets = {}
    for e in UNIVERSE:
        f = DATA / f"{e.proxy.replace('.', '_')}.parquet"
        if not f.exists():
            continue
        df = pd.read_parquet(f)
        col = "close" if "close" in df.columns else "adj_close"
        r = df[col].pct_change()
        r.index = pd.to_datetime(r.index).tz_localize(None)
        rets[e.bourso] = r
    return pd.DataFrame(rets)


def n_eff(corr: pd.DataFrame) -> float:
    """Effective number of independent bets (eigenvalue participation ratio)."""
    lam = np.linalg.eigvalsh(corr.values)
    lam = lam[lam > 1e-10]
    return float((lam.sum() ** 2) / (lam ** 2).sum())


def blocks(corr: pd.DataFrame, thr: float = 0.80) -> list[list[str]]:
    """Group ETFs whose average-linkage correlation exceeds `thr`."""
    try:
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform
        d = 1.0 - corr.values
        np.fill_diagonal(d, 0.0)
        d = (d + d.T) / 2
        Z = linkage(squareform(d, checks=False), method="average")
        labels = fcluster(Z, t=1.0 - thr, criterion="distance")
    except Exception:
        # Greedy union-find fallback if scipy is unavailable
        cols = list(corr.columns)
        parent = {c: c for c in cols}
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x
        for i, a in enumerate(cols):
            for b in cols[i + 1:]:
                if corr.loc[a, b] >= thr:
                    parent[find(a)] = find(b)
        labels = np.array([hash(find(c)) for c in cols])
    out = {}
    for tkr, lab in zip(corr.columns, labels):
        out.setdefault(lab, []).append(tkr)
    return sorted(out.values(), key=len, reverse=True)


def main():
    rets = load_returns()
    rets = rets.dropna(how="any")          # common window: all ETFs present
    wk = (1 + rets).resample("W-FRI").prod() - 1   # weekly returns
    corr = wk.corr()
    tickers = list(corr.columns)
    n = len(tickers)

    print(f"Univers : {n} ETF  |  fenêtre commune "
          f"{rets.index[0].date()} → {rets.index[-1].date()}  "
          f"({len(wk)} semaines)")

    # --- 1. Effective number of independent bets ---
    ne = n_eff(corr)
    print("\n=== 1. Nombre effectif de paris indépendants (N_eff) ===")
    print(f"  N_eff = {ne:.1f} / {n}   ({ne/n:.0%} d'indépendance)")
    print(f"  (univers orthogonal → N_eff = N ; tout corrélé → N_eff = 1)")

    # --- 2. Correlated blocks ---
    print("\n=== 2. Blocs corrélés (corr. hebdo moyenne ≥ 0.80) ===")
    for blk in blocks(corr, 0.80):
        if len(blk) < 2:
            continue
        sub = corr.loc[blk, blk]
        avg = (sub.values.sum() - len(blk)) / (len(blk) ** 2 - len(blk))
        print(f"  bloc de {len(blk)} (corr moy {avg:.2f}) : {', '.join(blk)}")

    # --- 3. Per-ETF redundancy ---
    print("\n=== 3. Redondance — corrélation au plus proche voisin ===")
    rows = []
    for t in tickers:
        o = corr[t].drop(t)
        rows.append((t, o.abs().mean(), o.max(), o.idxmax()))
    for t, mc, mx, nn in sorted(rows, key=lambda x: -x[2])[:14]:
        flag = "⚠ QUASI-DOUBLON" if mx > 0.90 else ("redondant" if mx > 0.80 else "")
        print(f"  {t:9s} voisin={nn:9s} corr={mx:.2f}  (corr moy {mc:.2f})  {flag}")

    # --- 4. Rotation usefulness: volatility of cross-sectional rank ---
    eq = (1 + rets).cumprod()
    mom = eq / eq.shift(63) - 1                     # 63-day momentum
    rank = mom.rank(axis=1, pct=True)               # cross-sectional percentile
    rank_std = rank.std()
    print("\n=== 4. Utilité en rotation (volatilité du rang cross-sectionnel) ===")
    print("  élevé = visite le haut ET le bas du classement → utile en rotation")
    lo = rank_std.sort_values()
    for t in lo.index[:6]:
        print(f"  faible : {t:9s} rang_std={lo[t]:.3f}")
    hi = rank_std.sort_values(ascending=False)
    for t in hi.index[:6]:
        print(f"  fort   : {t:9s} rang_std={hi[t]:.3f}")

    MYFILES.mkdir(exist_ok=True)
    corr.to_csv(MYFILES / "universe_corr.csv")
    print(f"\nMatrice de corrélation → knowledge/universe_corr.csv")


if __name__ == "__main__":
    main()
