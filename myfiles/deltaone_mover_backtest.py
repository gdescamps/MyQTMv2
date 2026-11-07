"""Backtest des tweets DeltaOne « mover » classés par Haiku 5.5 (data/text/x/deltaone/classified.parquet).

Règles (NDX CFD 1h, data/stocks/USATECHIDXUSD_1h.parquet) :
- signal  : tweet mover=True, sens=+1 (long) [ou -1 pour la variante short, hors PEA],
            confiance >= seuil ;
- confirmation (1 bougie) : la première bougie horaire COMPLÈTE après le tweet doit clôturer
            dans le sens du signal (close > open pour un long) ; sinon pas de trade ;
- entrée  : ouverture de la bougie suivante (pas de look-ahead) ;
- sortie  : clôture après HOLD bougies ; un seul trade à la fois (signaux ignorés en position) ;
- frais   : 1 % par aller-retour (0,5 % à l'entrée + 0,5 % à la sortie, courtage Bourso PEA).
Sharpe = moyenne / écart-type des rendements journaliers (jours de cotation, 0 hors position) × √252.
Limites : ETF PEA cotés seulement 9h-17h30 Paris (non modélisé), spread/change ignorés.
"""
import numpy as np
import pandas as pd

FEE_SIDE = 0.005
TH = [0, 40, 50, 60, 70]
CUTOFF = pd.Timestamp("2026-07-01", tz="UTC")  # coupure des connaissances Haiku 5.5 = juin 2026
HOLDS = [4, 23, 46]  # bougies : ~4 h, ~1 séance, ~2 séances


def run(px, sig, th, hold, sides):
    ret = px.close.pct_change().fillna(0.0)
    idx = px.index
    pos = pd.Series(0.0, index=idx)
    fee = pd.Series(0.0, index=idx)
    trades, grosses, busy_until = [], [], -1
    for _, m in sig[(sig.confiance >= th) & sig.sens.isin(sides)].sort_values("created_at").iterrows():
        c = idx.searchsorted(m.created_at.ceil("h"))  # 1re bougie complète après le tweet
        e = c + 1                                     # entrée à l'ouverture de la suivante
        if e + hold > len(idx) or c <= busy_until:
            continue
        conf_ok = np.sign(px.close.iloc[c] - px.open.iloc[c]) == m.sens
        if not conf_ok:
            continue
        x = e + hold - 1                              # sortie à la clôture de la bougie x
        # rendement de la bougie e mesuré depuis son ouverture (entrée), puis close-to-close
        pos.iloc[e:x + 1] = m.sens
        first = px.close.iloc[e] / px.open.iloc[e] - 1
        ret.iloc[e] = first  # entrée à l'open : 1re bougie comptée open→close
        fee.iloc[e] += FEE_SIDE
        fee.iloc[x] += FEE_SIDE
        gross = px.close.iloc[x] / px.open.iloc[e] - 1
        trades.append(m.sens * gross - 2 * FEE_SIDE)
        grosses.append(m.sens * gross)
        busy_until = x
    hourly = pos * ret - fee
    daily = (1 + hourly).groupby(idx.tz_convert("America/New_York").date).prod() - 1
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else np.nan
    tot = (1 + daily).prod() - 1
    yrs = len(daily) / 252
    cagr = (1 + tot) ** (1 / yrs) - 1 if yrs > 0 else np.nan
    return (len(trades), np.mean(grosses) if trades else np.nan, np.mean(trades) if trades else np.nan,
            (np.array(trades) > 0).mean() if trades else np.nan, cagr, sharpe)


def main():
    df = pd.read_parquet("data/text/x/deltaone/classified.parquet")
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    allpx = pd.read_parquet("data/stocks/USATECHIDXUSD_1h.parquet")
    periods = [("BIAISÉE 2023 → 2026-06 (Haiku connaît la suite)", df.created_at < CUTOFF),
               ("HORS ÉCHANTILLON 2026-07 → fin", df.created_at >= CUTOFF)]
    for plabel, mask in periods:
        d = df[mask]
        sig = d[d.mover == True].copy()  # noqa: E712
        t0, t1 = d.created_at.min().floor("D"), d.created_at.max().ceil("D")
        px = allpx.loc[t0:t1]
        bh = (1 + px.close.pct_change().fillna(0)).groupby(px.index.tz_convert("America/New_York").date).prod() - 1
        bh_cagr = (1 + bh).prod() ** (252 / len(bh)) - 1
        print(f"\n################ {plabel}")
        print(f"{t0:%Y-%m-%d} → {t1:%Y-%m-%d} ({bh.size} jours) — tweets {len(d)}, movers {len(sig)} "
              f"({100 * len(sig) / len(d):.1f} %) : long {int((sig.sens == 1).sum())}, short {int((sig.sens == -1).sum())}")
        print(f"Buy & hold NDX : {100 * bh_cagr:+.1f} %/an, Sharpe {bh.mean() / bh.std() * np.sqrt(252):.2f}\n")
        for label, sides in (("LONG seul (PEA)", [1]), ("LONG + SHORT (hors PEA)", [1, -1])):
            rows = []
            for hold in HOLDS:
                for th in TH:
                    n, g, avg, win, cagr, sh = run(px, sig, th, hold, sides)
                    rows.append({"sortie": hold, "seuil": th, "trades": n,
                                 "brut/trade %": round(100 * g, 2) if n else None,
                                 "net/trade %": round(100 * avg, 2) if n else None,
                                 "gagnants %": round(100 * win) if n else None,
                                 "%/an": round(100 * cagr, 1), "Sharpe": round(sh, 2) if n else None})
            print(f"=== {label} — confirmation 1 bougie, frais 1 % A/R ===")
            print(pd.DataFrame(rows).to_string(index=False), "\n")


if __name__ == "__main__":
    main()
