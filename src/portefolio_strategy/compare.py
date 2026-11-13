"""Apport de chaque famille de données : même modèle de classement, même portefeuille, seules les features changent.

Variantes (features.VARIANTS) : base (données structurées seules), pit, pit_types, all, et sur demande vendor,
qsent (sentiment Qwen, NON point-in-time). Pour chacune : entraînement si les scores manquent (ou --force),
backtest, puis un tableau commun → outputs/portefolio_strategy/comparison.md / .csv / .png.

    python -m src.portefolio_strategy.compare                    # base pit pit_types all
    python -m src.portefolio_strategy.compare base pit --force
"""
import json
import sys

import matplotlib
import pandas as pd

from . import benchmark as B
from . import config as C
from . import features as F
from . import train as TR
from .labels import add_labels

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DEFAULT = ["base", "pit", "pit_types", "all"]
RAW = {"PIT P(bonne) entreprise, 5 j de news": "pit_pos_company_5", "PIT P(mauvaise) entreprise, 5 j de news": "pit_neg_company_5",
       "PIT P(bonne) entreprise, dernier jour": "pit_pos_company", "PIT P(mauvaise) entreprise, dernier jour": "pit_neg_company",
       "Qwen type « résultats », 5 j de news": "qt_earnings_5", "Qwen type « analystes », 5 j de news": "qt_analyst_5",
       "fournisseur : polarité 5 j de news": "vs_pol5", "momentum 12-1 mois": "mom_12_1", "rendement 21 j": "ret_21"}


def raw_signal_ic(df):
    """Signal brut de quelques variables, hors modèle : moyenne des corrélations de rang quotidiennes avec le
    rendement futur, parmi les membres, sur la période hors échantillon."""
    d = df[df["member"] & (df["date"] >= C.FIRST_OOS_DATE)]
    out = {}
    for name, col in RAW.items():
        if col in d and d[col].notna().sum() > 1000:
            out[name] = {"n": int(d[col].notna().sum()),
                         **{f"IC {h} j": round(float(d.groupby("date")[[col, f"fwd_{h}"]].apply(
                             lambda g: g[col].corr(g[f"fwd_{h}"], method="spearman")).mean()), 4) for h in C.HORIZONS}}
    return out


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    force = "--force" in argv
    variants = [a for a in argv if not a.startswith("--")] or DEFAULT
    df = pd.read_parquet(F.FEATURES)
    rows, curves = [], {}
    for v in variants:
        if force or not TR.predictions_path(v).exists():
            TR.main(df=df, variant=v)
        m = B.main(v)
        rep = json.loads((TR.out_dir(v) / "train_report.json").read_text())
        eq = pd.read_csv(TR.out_dir(v) / "equity.csv", parse_dates=["date"]).set_index("date")
        curves[v], curves["indice"], curves["équipondéré"] = eq["strategy"], eq["index"], eq["equal_weight"]
        s, d = m["strategy"], m["default_config"]
        rows.append({"variante": v, "features": rep["n_features"],
                     **{f"IC {h} j": rep["diagnostics"][f"h{h}"]["ic_mean"] for h in C.HORIZONS},
                     "t IC 21 j": rep["diagnostics"]["h21"]["ic_t"],
                     "CAGR % (walk-forward)": round(s["cagr_pct"], 1), "écart indice (pts)": round(s["excess_cagr_pct"], 1),
                     "RI": round(s["info_ratio"], 2), "maxDD %": round(s["max_drawdown_pct"], 1),
                     "écart config. par défaut (pts)": round(d["excess_cagr_pct"], 1),
                     "grille > indice": f"{m['grid']['beating_index']}/{m['grid']['configurations']}",
                     "écart médian grille (pts)": round(m["grid"]["median_excess_cagr_pct"], 1)})
    for name, key in (("univers équipondéré", "equal_weight"), ("indice (capitalisation)", "index")):
        x = m[key]
        rows.append({"variante": name, "CAGR % (walk-forward)": round(x["cagr_pct"], 1), "maxDD %": round(x["max_drawdown_pct"], 1),
                     "écart indice (pts)": round(x.get("excess_cagr_pct", 0.0), 1)})
    table = pd.DataFrame(rows)
    C.OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(C.OUT / "comparison.csv", index=False)
    raw = raw_signal_ic(add_labels(df, C.HORIZONS))
    md = ["# Sélection longue dans l'univers Nasdaq : apport de chaque famille de données", "",
          f"Période hors échantillon : {m['period'][0]} → {m['period'][1]}, frais {100 * C.FEE:.2f} % par sens. "
          "« IC » = moyenne des corrélations de rang quotidiennes entre le score et le rendement futur réalisable "
          "(ouverture J+1 → J+1+h). « walk-forward » = configuration (horizon, fréquence, nombre de titres, buffer, pondération) "
          "choisie chaque année sur le passé seulement ; « grille » = toutes les configurations testées.", "",
          "\n".join(f"- **{v}** : {F.VARIANTS[v][1]}" for v in variants), "",
          table.to_markdown(index=False), "", "## Signal brut de quelques variables (hors modèle)", "",
          pd.DataFrame(raw).T.to_markdown() if raw else "(aucune)", ""]
    (C.OUT / "comparison.md").write_text("\n".join(md))
    fig, ax = plt.subplots(figsize=(13, 6))
    for name, s in curves.items():
        ref = name in ("indice", "équipondéré")
        ax.plot(s.index, s.values, lw=2 if ref else 1.3, ls="--" if ref else "-", color={"indice": "tab:blue", "équipondéré": "tab:gray"}.get(name),
                label=name if ref else f"{name} — {F.VARIANTS[name][1]}")
    ax.set_yscale("log")
    ax.set_ylabel("Valeur ($, log)")
    ax.set_title("portefolio_strategy — sélection longue, configuration choisie en walk-forward (hors échantillon, nette de frais)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(C.OUT / "comparison.png", dpi=110)
    plt.close(fig)
    print("\n".join(md))
    return table


if __name__ == "__main__":
    main()
