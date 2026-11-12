"""Évaluation de l'apport de chaque famille de sentiment : même XGBoost walk-forward, mêmes règles de trading,
seules les features de sentiment changent (features.VARIANTS : base, vendor, pit, all).

Pour chaque variante : entraînement (si les prédictions manquent ou avec --force), backtest, puis un tableau
commun → outputs/portefolio_strategy/comparison.md / .csv / .png. Mesure aussi le signal brut de chaque note
(corrélation de rang avec le rendement futur), indépendamment du modèle.

    python -m src.portefolio_strategy.compare                 # les quatre variantes
    python -m src.portefolio_strategy.compare base vendor     # seulement celles-ci
    python -m src.portefolio_strategy.compare --force         # ré-entraîne tout
"""
import json
import sys

import matplotlib
import pandas as pd

from . import benchmark as B
from . import config as C
from . import features as F
from . import train as TR

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HORIZONS = (1, 5, 20)


def forward_returns(df):
    """Rendement réalisable : ouverture de J+1 → ouverture de J+1+h (le signal de J est connu à la clôture de J)."""
    d = df.sort_values(["symbol", "date"])
    g = d.groupby("symbol")["o"]
    return pd.DataFrame({f"f{h}": g.shift(-h - 1) / g.shift(-1) - 1 for h in HORIZONS}, index=d.index)


def rank_ic(signal, fwd):
    ok = signal.notna()
    return {h: round(float(signal[ok].corr(fwd.loc[ok, f"f{h}"], method="spearman")), 4) for h in HORIZONS}


def raw_signal_ic(df):
    """Signal brut des notes de sentiment, le jour de la news ou le lendemain (note fraîche), membres seulement."""
    d = df[df["member"] & (df["date"] >= C.FIRST_OOS_DATE)]
    fwd = forward_returns(df).loc[d.index]
    out = {}
    specs = {"fournisseur : polarité du jour": ("ts_vs_polarity_0", "vs_days"),
             "fournisseur : polarité 5 jours de news": ("vs_pol5", "vs_days"),
             "PIT : entreprise": ("ts_news_sentiment_company_0", "news_days"),
             "PIT : concurrents": ("ts_news_sentiment_competitors_0", "news_days"),
             "PIT : global": ("ts_news_sentiment_global_0", "news_days")}
    for name, (col, days) in specs.items():
        if col in d and d[col].notna().sum() > 1000:
            fresh = d[days].between(0, 1)
            out[name] = {"n": int((fresh & d[col].notna()).sum()), **rank_ic(d.loc[fresh, col], fwd.loc[fresh])}
    return out


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    force = "--force" in argv
    variants = [a for a in argv if not a.startswith("--")] or list(F.VARIANTS)
    df = pd.read_parquet(F.FEATURES)
    rows, curves = [], {}
    for v in variants:
        if force or not TR.predictions_path(v).exists():
            TR.main(df=df, variant=v)
        m = B.main(v)
        rep = json.loads((TR.out_dir(v) / "train_report.json").read_text())
        p = pd.read_parquet(TR.predictions_path(v))
        ic = rank_ic(p["ybull"] - p["ybear"], forward_returns(p))
        eq = pd.read_csv(TR.out_dir(v) / "equity.csv", parse_dates=["date"]).set_index("date")
        curves[v], curves["indice"] = eq["strategy"], eq["index"]
        s, pos = m["strategy"], m["positions"]
        rows.append({"variante": v, "description": F.VARIANTS[v], "features": len(F.feature_columns(df, v)),
                     "F1 macro": rep["oos_macro_f1"], **{f"IC {h} j": ic[h] for h in HORIZONS},
                     "rendement %": round(s["return_pct"], 1), "CAGR %": round(s["cagr_pct"], 1),
                     "maxDD %": round(s["max_drawdown_pct"], 1), "Sharpe": round(s["sharpe"], 2),
                     "positions": sum(x["n"] for x in pos.values()),
                     "gain moyen long %": pos.get("long", {}).get("mean_gain_pct")})
        idx = m["index"]
    rows.append({"variante": "indice", "description": "indice de l'univers (achat-conservation)",
                 "rendement %": round(idx["return_pct"], 1), "CAGR %": round(idx["cagr_pct"], 1),
                 "maxDD %": round(idx["max_drawdown_pct"], 1), "Sharpe": round(idx["sharpe"], 2)})
    table = pd.DataFrame(rows)
    C.OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(C.OUT / "comparison.csv", index=False)
    raw = raw_signal_ic(df)
    md = ["# Apport du sentiment : comparaison des variantes", "",
          f"Période hors échantillon : {m['period'][0]} → {m['period'][1]}. Même XGBoost walk-forward et mêmes règles "
          "de trading ; seules les features de sentiment changent. « IC » = corrélation de rang entre le signal du "
          "modèle (P haussier − P baissier) et le rendement futur réalisable (ouverture J+1 → J+1+h).", "",
          table.to_markdown(index=False), "", "## Signal brut des notes de sentiment (hors modèle)", "",
          "Corrélation de rang entre la note fraîche (jour de la news ou lendemain) et le rendement futur.", "",
          pd.DataFrame(raw).T.rename(columns={h: f"IC {h} j" for h in HORIZONS}).to_markdown() if raw else "(aucune note)",
          ""]
    (C.OUT / "comparison.md").write_text("\n".join(md))
    fig, ax = plt.subplots(figsize=(13, 6))
    for name, s in curves.items():
        ax.plot(s.index, s.values, lw=2 if name == "indice" else 1.3, ls="--" if name == "indice" else "-",
                label=name if name == "indice" else f"{name} — {F.VARIANTS[name]}")
    ax.set_yscale("log")
    ax.set_ylabel("Valeur ($, log)")
    ax.set_title("portefolio_strategy — apport du sentiment (hors échantillon, net de frais)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(C.OUT / "comparison.png", dpi=110)
    plt.close(fig)
    print("\n".join(md))
    return table


if __name__ == "__main__":
    main()
