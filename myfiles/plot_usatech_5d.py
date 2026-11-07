"""Chandeliers 1h USATECHIDXUSD sur les 5 derniers jours ouvrés, axe en UTC.

Fond gris clair = séance Euronext Paris 09:00–17:30 Europe/Paris ; gris foncé = séance cash Wall Street 09:30–16:00 America/New_York, convertie en UTC jour par jour
(gère les changements d'heure US). Bougie « HH:00 » = heure d'ouverture, couvre HH:00–HH+1:00.
"""
import datetime as dt

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

TZ = "UTC"
df = pd.read_parquet("data/stocks/USATECHIDXUSD_1h.parquet").tz_convert(TZ)
days = sorted({d for d in df.index.date if d.weekday() < 5})[-5:]
d = df[pd.Index(df.index.date).isin(days)]
n = len(d)
UP, DOWN, INK, MUTED = "#2a78b5", "#d2642a", "#222", "#888"


def pos(t: pd.Timestamp) -> float:
    """Position x d'un instant : bougie i couvre [i-0.5, i+0.5] = [heure, heure+1)."""
    i = d.index.searchsorted(t.floor("h"))
    return i - 0.5 + (t - t.floor("h")).total_seconds() / 3600


fig, ax = plt.subplots(figsize=(14, 6), dpi=130)
SESSIONS = [  # (fuseau, ouverture, durée, alpha, libellé)
    ("Europe/Paris", dt.time(9, 0), pd.Timedelta(hours=8, minutes=30), 0.035, "Séance Paris 09:00–17:30 (Paris)"),
    ("America/New_York", dt.time(9, 30), pd.Timedelta(hours=6, minutes=30), 0.10, "Séance Wall Street 09:30–16:00 NY"),
]
for tz, t0, dur, alpha, _ in SESSIONS:
    for day in days:
        o = pd.Timestamp(dt.datetime.combine(day, t0), tz=tz)
        ax.axvspan(pos(o.tz_convert(TZ)), pos((o + dur).tz_convert(TZ)), color="#000", alpha=alpha, lw=0)
for i, r in enumerate(d.itertuples()):
    up = r.close >= r.open
    c = UP if up else DOWN
    ax.vlines(i, r.low, r.high, color=c, lw=1)
    lo, hi = sorted((r.open, r.close))
    ax.bar(i, max(hi - lo, 1e-6), bottom=lo, width=0.7, color="white" if up else c, edgecolor=c, lw=1)

dates = pd.Index(d.index.date)
starts = [i for i in range(n) if i == 0 or dates[i] != dates[i - 1]]
for s in starts[1:]:
    ax.axvline(s - 0.5, color=MUTED, lw=0.6, ls=":")
# ticks : début de jour (date) + toutes les 3h (heure)
minor = [i for i in range(n) if d.index[i].hour % 3 == 0 and i not in starts]
ax.set_xticks(starts)
ax.set_xticklabels([d.index[s].strftime("%a %d/%m\n%H:%M") for s in starts], color=INK, fontsize=9)
ax.set_xticks(minor, minor=True)
ax.set_xticklabels([d.index[i].strftime("%H") for i in minor], minor=True, fontsize=7, color=MUTED)
ax.set_xlim(-1, n)
ax.set_xlabel("Heure UTC (ouverture de bougie)", color=INK)
ax.grid(axis="y", color="#ddd", lw=0.5)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.set_ylabel("Points", color=INK)
first, last = d.open.iloc[0], d.close.iloc[-1]
ax.set_title(f"USATECHIDXUSD (CFD Nasdaq-100, Dukascopy) – bougies 1h, 5 derniers jours ouvrés\n"
             f"{d.index[0]:%d/%m %H:%M} → {d.index[-1]:%d/%m %H:%M} UTC · "
             f"{first:,.0f} → {last:,.0f} ({last / first - 1:+.1%})",
             loc="left", fontsize=10, color=INK)
ax.legend(handles=[Patch(facecolor="white", edgecolor=UP, label="Hausse (creuse)"),
                   Patch(facecolor=DOWN, edgecolor=DOWN, label="Baisse (pleine)"),
                   *[Patch(facecolor="#000", alpha=a, label=lab) for *_, a, lab in SESSIONS]],
          loc="upper right", frameon=False)
fig.tight_layout()
fig.savefig("myfiles/usatech_1h_5d.png")
for tz, t0, dur, *_ in SESSIONS:
    o = pd.Timestamp(dt.datetime.combine(days[-1], t0), tz=tz)
    print(tz, "UTC :", o.tz_convert(TZ).strftime("%H:%M"), "→", (o + dur).tz_convert(TZ).strftime("%H:%M"))
