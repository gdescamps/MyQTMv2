"""Épisodes de volatilité les plus brutaux sur le CFD Nasdaq-100 (USATECHIDXUSD, 1h), 2023 → fin.

Méthode
-------
- Un épisode = un mouvement directionnel d'une barre de départ i vers une barre de fin j > i,
  avec une durée ≤ 5 séances de trading (5 × 23 barres horaires = 115 barres).
- Baissier : amplitude = low[j] / high[i] − 1 (du plus haut au plus bas).
  Haussier : amplitude = high[j] / low[i] − 1 (du plus bas au plus haut).
- Pour chaque barre de départ i on retient la meilleure fin j dans la fenêtre ; sélection
  gloutonne par |amplitude| décroissante d'épisodes qui ne se chevauchent pas (par sens), avec
  recalcul du candidat sur l'espace libre s'il chevauche ; on garde tous les épisodes ≥ 1.5 %.
- Heure de début = horodatage de la barre portant l'extrême de départ, heure de fin = celle
  portant l'extrême de fin (résolution 1h, horodatage = début de la barre, UTC).
- Durée = heures calendaires (fin − début) et nombre de barres (heures de cotation).
"""
import numpy as np
import pandas as pd

SRC = "data/stocks/USATECHIDXUSD_1h.parquet"
START = "2023-01-01"
MAX_BARS = 5 * 23
MIN_AMP = 0.015  # seuil d'amplitude |mouvement| ≥ 1.5 %
OUT = "myfiles/ndx_vol_episodes.csv"


def _best_end(i, limit, hi, lo, side):
    """Meilleure barre de fin j ∈ [i+1, limit] → (amplitude, j)."""
    if limit <= i:
        return None
    if side == "down":
        k = int(np.argmin(lo[i + 1:limit + 1]))
        return lo[i + 1 + k] / hi[i] - 1, i + 1 + k
    k = int(np.argmax(hi[i + 1:limit + 1]))
    return hi[i + 1 + k] / lo[i] - 1, i + 1 + k


def find_episodes(df, side):
    """Sélection gloutonne sans chevauchement, avec recalcul paresseux : un candidat dont
    l'intervalle chevauche un épisode déjà retenu est recalculé sur l'espace libre restant
    (fin bornée avant le prochain épisode retenu) puis remis dans la file."""
    import heapq
    hi, lo = df["high"].to_numpy(), df["low"].to_numpy()
    n = len(df)
    sgn = -1 if side == "down" else 1
    occupied = np.zeros(n, bool)
    heap = []
    for i in range(n - 1):
        r = _best_end(i, min(n - 1, i + MAX_BARS), hi, lo, side)
        if r and sgn * r[0] > 0:
            heap.append((-abs(r[0]), i, r[1]))
    heapq.heapify(heap)
    rows = []
    while heap:
        neg, i, e = heapq.heappop(heap)
        if -neg < MIN_AMP:
            break  # file triée : tous les candidats restants sont sous le seuil
        if occupied[i]:
            continue
        if occupied[i:e + 1].any():
            nxt = i + int(np.argmax(occupied[i:e + 1]))  # 1re barre occupée
            r = _best_end(i, nxt - 1, hi, lo, side)
            if r and sgn * r[0] > 0:
                heapq.heappush(heap, (-abs(r[0]), i, r[1]))
            continue
        occupied[i:e + 1] = True
        t0, t1 = df.index[i], df.index[e]
        rows.append({
            "side": side, "start": t0, "end": t1,
            "start_price": hi[i] if side == "down" else lo[i],
            "end_price": lo[e] if side == "down" else hi[e],
            "amplitude_pct": -100 * neg * sgn,
            "duration_h": (t1 - t0).total_seconds() / 3600,
            "duration_bars": e - i,
        })
    return pd.DataFrame(rows)


def stats(ep):
    s = ep[["duration_h", "duration_bars", "amplitude_pct"]].agg(["mean", "min", "max", "std"]).T
    return s.round(2)


def main():
    df = pd.read_parquet(SRC).loc[START:].dropna()
    print(f"{SRC}: {df.index[0]} → {df.index[-1]}, {len(df)} barres\n")
    out = []
    for side, label in (("down", "BAISSIERS"), ("up", "HAUSSIERS")):
        ep = find_episodes(df, side)
        out.append(ep)
        print(f"=== {len(ep)} épisodes {label} (≤ {MAX_BARS} barres) ===")
        print(stats(ep).to_string())
        print("\nTop 10 :")
        print(ep.head(10)[["start", "end", "amplitude_pct", "duration_h", "duration_bars"]]
              .round(2).to_string(index=False))
        print()
        print("Heure UTC de début (distribution) :")
        print(ep["start"].dt.hour.value_counts().sort_index().to_string())
        print()
    pd.concat(out).to_csv(OUT, index=False)
    print(f"→ {OUT}")


if __name__ == "__main__":
    main()
