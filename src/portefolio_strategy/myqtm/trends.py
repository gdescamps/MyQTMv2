"""Label « tendance » (0 baissier, 1 range, 2 haussier) — portage de ref/MyQTM/src/trends.py.

Le label d'un jour dépend de la suite des cours (un segment n'est connu qu'à son croisement de sortie).
ref/MyQTM l'utilisait tel quel. Ici `trend_labels` rend aussi `final_until` : la date à partir de laquelle
les labels ne sont pas encore définitifs avec les cours fournis. L'entraînement à la date T n'utilise que
des cours < T et écarte les jours >= final_until → aucun label ne contient d'information postérieure à T.
"""
import numpy as np
import pandas as pd


def _crosses(a, b):
    sign = np.sign(a - b).replace(0, np.nan).ffill()
    return sign, a.index[sign.ne(sign.shift())]


def detect_trends(close, ma_short=4, ma_mid=8, ma_long=30, min_days=8, avg_daily_percent_thr=0.15, bars=False):
    """close : Series indexée par date croissante → (segments, final_until).
    bars=True (version horaire) : durées comptées en nombre de lignes (barres) au lieu de jours calendaires ;
    min_days = barres minimum, avg_daily_percent_thr = % moyen par barre."""
    close = close.sort_index()
    pos_of = pd.Series(np.arange(len(close)), index=close.index)

    def span(a, b):
        return int(pos_of[b] - pos_of[a]) if bars else (b - a).days
    e_s = close.ewm(span=ma_short, adjust=False).mean()
    e_m = close.ewm(span=ma_mid, adjust=False).mean()
    e_l = close.ewm(span=ma_long, adjust=False).mean()
    sign_ml, cross_ml = _crosses(e_m, e_l)
    _, cross_sm = _crosses(e_s, e_m)
    trends, last_end = [], None
    for i in range(1, len(cross_ml)):
        start_ml, end_ml = cross_ml[i - 1], cross_ml[i]
        if span(start_ml, end_ml) < min_days:
            continue
        before = cross_sm[cross_sm < start_ml]
        adj_start = before[-1] if len(before) else start_ml
        before = cross_sm[cross_sm < end_ml]
        adj_end = before[-1] if len(before) else end_ml
        if last_end is not None and adj_start <= last_end:
            pos = close.index.get_loc(last_end)
            if pos + 1 >= len(close.index):
                continue
            adj_start = close.index[pos + 1]
            if adj_start > adj_end:
                continue
        days = int(span(adj_start, adj_end) or 1)
        ret = float(close.loc[adj_end] / close.loc[adj_start] - 1)
        kind = "bullish" if sign_ml.loc[start_ml] > 0 else "bearish"
        if (kind == "bullish" and ret / days >= avg_daily_percent_thr / 100) or \
           (kind == "bearish" and ret / days <= -avg_daily_percent_thr / 100):
            trends.append({"trend": kind, "start": adj_start, "end": adj_end, "total_return": 100 * ret, "days": days})
            last_end = adj_end
    # Le prochain segment commencera au plus tôt au dernier croisement court/moyen précédant le dernier
    # croisement moyen/long : avant cette date, les labels ne changeront plus.
    if len(cross_ml):
        before = cross_sm[cross_sm < cross_ml[-1]]
        final_until = before[-1] if len(before) else cross_ml[-1]
    else:
        final_until = close.index[0]
    return trends, final_until


def trend_labels(close, with_phase=False, **kw):
    """→ (Series de classes 0/1/2 par date, final_until).

    with_phase=True → (y, phase, final_until) : phase = position du jour dans sa tendance haussière ou
    baissière, i / n pour le i-ème jour de bourse (0 = premier jour) d'un segment de n jours ; NaN en range."""
    segs, final_until = detect_trends(close, **kw)
    y = pd.Series(1, index=close.index, dtype="int8")
    phase = pd.Series(np.nan, index=close.index)
    for s in segs:
        m = (y.index >= s["start"]) & (y.index <= s["end"])
        y[m] = 2 if s["trend"] == "bullish" else 0
        phase[m] = np.arange(m.sum()) / m.sum()
    return (y, phase, final_until) if with_phase else (y, final_until)
