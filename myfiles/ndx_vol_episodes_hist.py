"""Histogramme des amplitudes des épisodes de volatilité NDX (≥ 1.5 %, ≤ 5 séances), 2023 → fin."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ep = pd.read_csv("myfiles/ndx_vol_episodes.csv")
ep["amp"] = ep.amplitude_pct.abs()
CAP = 8.0
edges = np.append(np.arange(1.5, CAP + 0.5, 0.5), CAP + 0.5)  # dernier bac [8, ∞) = « ≥ 8 % »
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, sharey=True, facecolor=SURF)
for ax, (side, lab, col) in zip(axes, [("down", "Baissiers", "#eb6834"), ("up", "Haussiers", "#2a78d6")]):
    a = ep.loc[ep.side == side, "amp"]
    h, _ = np.histogram(a.clip(upper=CAP + 0.25), bins=edges)
    x = edges[:-1]
    ax.bar(x, h, width=0.5, align="edge", color=col, edgecolor=SURF, linewidth=2)
    for xi, hi in zip(x, h):
        if hi:
            ax.text(xi + 0.25, hi + 1, str(hi), ha="center", va="bottom", fontsize=8, color=INK2)
    med = a.median()
    ax.axvline(med, color=INK2, lw=1, ls="--")
    ax.text(med + 0.08, ax.get_ylim()[1] * 0.92 if ax.get_ylim()[1] > 1 else 1, f"médiane {med:.2f} %",
            fontsize=9, color=INK2, va="top")
    ax.set_title(f"{lab} — {len(a)} épisodes   (moy. {a.mean():.2f} %, max {a.max():.1f} %)",
                 loc="left", fontsize=11, color=INK)
    ax.set_facecolor(SURF)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.set_ylabel("nb d'épisodes", color=INK2, fontsize=9)
for ax in axes:
    ax.set_ylim(0, max(p.get_height() for a_ in axes for p in a_.patches) * 1.15)
ticks = list(np.arange(1.5, CAP, 0.5)) + [CAP + 0.25]
axes[1].set_xticks(ticks)
axes[1].set_xticklabels([f"{t:g}" for t in ticks[:-1]] + [f"≥{CAP:g}"])
axes[1].set_xlabel("amplitude |%| (extrême → extrême, ≤ 5 séances)", color=INK2, fontsize=9)
fig.suptitle("Nasdaq-100 (CFD 1h) — amplitude des épisodes de volatilité ≥ 1,5 %, 2023-01 → 2026-11",
             x=0.01, ha="left", fontsize=12, color=INK)
fig.tight_layout()
fig.savefig("myfiles/ndx_vol_episodes_hist.png", dpi=130, facecolor=SURF)
for side in ("down", "up"):
    a = ep.loc[ep.side == side, "amp"]
    print(side, pd.cut(a, [1.5, 2, 3, 5, 8, 20], include_lowest=True).value_counts().sort_index().to_dict())
