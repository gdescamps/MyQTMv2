"""Modèle « ensemble » : il voit les ~100 titres de l'univers en même temps et choisit les 20 de la semaine suivante.

Contrairement à train.py (XGBoost, un titre à la fois), l'entrée d'un jour est le tableau complet
(membres × features) et la sortie un score par titre :
- chaque feature est remplacée par son rang du jour parmi les membres (−0,5 … +0,5 ; manquant = 0) : le modèle ne
  voit que des positions relatives, les variables identiques pour tous les titres disparaissent ;
- un petit transformer sans position (l'ordre des titres n'a pas de sens) laisse chaque titre « regarder » les
  autres avant de donner son score ;
- cible (TARGET) sur le rendement de la période suivante (ouverture J+1 → J+1+HORIZON) :
  « rank » (défaut) = rang du jour centré, perte quadratique — pénalise autant de choisir un futur perdant que de
  rater un gagnant ;
  « top » = appartenir aux N_TOP meilleurs, entropie croisée sur la liste entière. Essayé en premier : le modèle
  apprend surtout à choisir les titres volatils (souvent dans les meilleurs ET dans les pires), d'où un
  portefeuille perdant malgré 25 % de bons choix contre 20 % au hasard.
Même walk-forward purgé que train.py (labels connus avant la date de coupe), nombre d'époques fixe, moyenne de
SEEDS initialisations. Évalué avec le même moteur : N_TOP lignes à poids égaux, renouvelées tous les HORIZON jours.

Sorties : data/portefolio_strategy/predictions_set_<variante>.parquet, outputs/portefolio_strategy/set_model/.

    python -m src.portefolio_strategy.set_model [variante]
"""
import json
import time

import matplotlib
import numpy as np
import pandas as pd

from . import benchmark as B
from . import config as C
from . import features as F
from . import portfolio as PF
from . import train as TR
from .labels import add_labels, label_known

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = C.OUT / "set_model"
HORIZON = 5                    # jours de bourse : sélection pour la semaine suivante
N_TOP = 20
DIM, HEADS, LAYERS, DROPOUT = 32, 4, 2, 0.3
EPOCHS, BATCH, LR, WEIGHT_DECAY = 25, 32, 1e-3, 1e-2
SEEDS = 3
TARGET = "rank"                # "rank" ou "top"


def predictions_path(variant):
    return C.DATA / f"predictions_set_{variant}.parquet"


def build_tensors(df, cols):
    """df (membres seulement, avec y) → X (jours, titres max, features) en rangs du jour, masque, cible top-N,
    rendement futur, dates, symboles."""
    d = df.sort_values(["date", "symbol"])
    ranks = d.groupby("date")[cols].rank(pct=True).sub(0.5).fillna(0.0).to_numpy(np.float32)
    dates = np.sort(d["date"].unique())
    di = pd.Series(np.arange(len(dates)), index=dates)[d["date"]].to_numpy()
    slot = d.groupby("date").cumcount().to_numpy()
    n = int(slot.max()) + 1
    X = np.zeros((len(dates), n, len(cols)), np.float32)
    M = np.zeros((len(dates), n), bool)
    Y = np.full((len(dates), n), np.nan, np.float32)
    X[di, slot], M[di, slot], Y[di, slot] = ranks, True, d[f"fwd_{HORIZON}"].to_numpy(np.float32)
    sym = np.full((len(dates), n), None, object)
    sym[di, slot] = d["symbol"].to_numpy()
    return X, M, Y, dates, sym


def top_target(Y, M, n_top=N_TOP):
    """Répartition cible : 1/n_top sur les n_top meilleurs rendements futurs du jour, 0 ailleurs."""
    y = np.where(M & np.isfinite(Y), Y, -np.inf)
    order = np.argsort(-y, axis=1)[:, :n_top]
    T = np.zeros_like(Y)
    np.put_along_axis(T, order, 1.0 / n_top, axis=1)
    return T * (M & np.isfinite(Y))


def rank_target(Y, M):
    """Rang du jour du rendement futur parmi les membres, centré (−0,5 … +0,5) ; 0 hors univers ou inconnu."""
    ok = M & np.isfinite(Y)
    r = pd.DataFrame(np.where(ok, Y, np.nan)).rank(axis=1, pct=True).to_numpy(np.float32) - 0.5
    return np.where(ok, r, 0.0).astype(np.float32)


def make_model(n_features, seed):
    import torch
    from torch import nn
    torch.manual_seed(seed)

    class SetRanker(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Sequential(nn.Dropout(0.1), nn.Linear(n_features, DIM))
            layer = nn.TransformerEncoderLayer(DIM, HEADS, 2 * DIM, DROPOUT, batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(layer, LAYERS, enable_nested_tensor=False)
            self.out = nn.Sequential(nn.LayerNorm(DIM), nn.Linear(DIM, 1))

        def forward(self, x, mask):
            s = self.out(self.enc(self.inp(x), src_key_padding_mask=~mask)).squeeze(-1)
            return s.masked_fill(~mask, -1e9)

    return SetRanker()


def fit_predict(Xtr, Mtr, Ttr, Xte, Mte, seed):
    import torch
    torch.set_num_threads(C.WORKERS)
    dev = "cuda" if torch.cuda.is_available() else "cpu"      # petit modèle, mais ~30× plus rapide sur le GPU du gb10
    model = make_model(Xtr.shape[2], seed).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    X, M, T = (torch.from_numpy(a).to(dev) for a in (Xtr, Mtr, Ttr))
    g = torch.Generator().manual_seed(seed)
    for _ in range(EPOCHS):
        model.train()
        for idx in torch.randperm(len(X), generator=g).split(BATCH):
            idx = idx.to(dev)
            s = model(X[idx], M[idx])
            if TARGET == "top":
                loss = -(T[idx] * torch.log_softmax(s, dim=1)).sum(1).mean()
            else:
                loss = (((s - T[idx]) ** 2) * M[idx]).sum() / M[idx].sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(Xte).to(dev), torch.from_numpy(Mte).to(dev)).cpu().numpy()


def train(variant="pit"):
    df = pd.read_parquet(F.FEATURES)
    cols = [c for c in F.feature_columns(df, variant) if not c.startswith("xs_")]     # déjà mis en rangs ici
    df = add_labels(df.sort_values(["symbol", "date"]).reset_index(drop=True), [HORIZON])
    pos = df.drop_duplicates("date").set_index("date")["pos"]
    X, M, Y, dates, sym = build_tensors(df[df["member"]], cols)
    dpos = pos.reindex(dates).to_numpy()
    T = top_target(Y, M) if TARGET == "top" else rank_target(Y, M)
    known = (M & np.isfinite(Y)).sum(1) > N_TOP
    last = pd.Timestamp(dates[-1])
    cuts = TR.cutoffs(C.FIRST_OOS_DATE, last)
    print(f"modèle ensemble, cible « {TARGET} », variante « {variant} » : {len(cols)} features, {X.shape[1]} titres par jour, "
          f"{len(dates)} jours", flush=True)
    rows, t0 = [], time.time()
    for i, cut in enumerate(cuts):
        nxt = cuts[i + 1] if i + 1 < len(cuts) else last + pd.Timedelta(days=1)
        cutoff_pos = int(pos[pos.index < cut].max()) + 1
        tr = label_known(dpos, HORIZON, cutoff_pos) & known
        te = (dates >= np.datetime64(cut)) & (dates < np.datetime64(nxt))
        if not te.any():
            continue
        s = np.mean([fit_predict(X[tr], M[tr], T[tr], X[te], M[te], seed) for seed in range(SEEDS)], axis=0)
        for k, d in enumerate(dates[te]):
            m = M[te][k]
            rows.append(pd.DataFrame({"date": d, "symbol": sym[te][k][m], f"s_{HORIZON}": s[k][m],
                                      f"fwd_{HORIZON}": Y[te][k][m]}))
        print(f"  {cut:%Y-%m-%d} : {int(tr.sum())} jours d'apprentissage jusqu'au {pd.Timestamp(dates[tr][-1]):%Y-%m-%d} "
              f"[{time.time() - t0:.0f} s]", flush=True)
    p = pd.concat(rows, ignore_index=True)
    p.to_parquet(predictions_path(variant), index=False)
    return p


def evaluate(p, variant="pit"):
    OUT.mkdir(parents=True, exist_ok=True)
    scores = p.pivot(index="date", columns="symbol", values=f"s_{HORIZON}").sort_index()
    _, opens, _ = B.load(variant)                     # ouvertures sur les mêmes dates que le backtest XGBoost
    xgb = pd.read_parquet(TR.predictions_path(variant))
    xs = xgb.pivot(index="date", columns="symbol", values=f"s_{HORIZON}").sort_index()
    opens = opens.reindex(index=scores.index, columns=scores.columns)
    bench = B.index_curve(scores.index)
    elig = scores.notna()
    ew = PF.simulate(elig.astype(float).where(elig), opens, None, 21)["values"]
    curves = {"modèle ensemble (transformer)": PF.simulate_offsets(scores, opens, N_TOP, HORIZON)[0],
              "XGBoost titre par titre (même horizon)": PF.simulate_offsets(xs.reindex_like(scores), opens, N_TOP, HORIZON)[0],
              "univers équipondéré": ew, "indice (capitalisation)": bench}
    gross = PF.simulate_offsets(scores, opens, N_TOP, HORIZON, fee=0.0)[0]
    rows = []
    for name, v in curves.items():
        m, a = PF.metrics(v), PF.active(v, bench)
        rows.append({"portefeuille": name, "CAGR net %": round(m["cagr_pct"], 1), "maxDD %": round(m["max_drawdown_pct"], 1),
                     "Sharpe": round(m["sharpe"], 2), "écart indice (pts/an)": round(a["excess_cagr_pct"], 1),
                     "années > indice": a["years_beating_index"] if name != "indice (capitalisation)" else ""})
    table = pd.DataFrame(rows)
    diag = {"modèle ensemble": TR.diagnostics(p, HORIZON),
            "XGBoost": TR.diagnostics(xgb[["date", "symbol", f"s_{HORIZON}", f"fwd_{HORIZON}"]], HORIZON)}
    hit = p.dropna(subset=[f"fwd_{HORIZON}"]).copy()
    hit["sel"] = hit.groupby("date")[f"s_{HORIZON}"].rank(ascending=False) <= N_TOP
    hit["top"] = hit.groupby("date")[f"fwd_{HORIZON}"].rank(ascending=False) <= N_TOP
    precision = float(hit.loc[hit["sel"], "top"].mean())
    rep = {"variant": variant, "horizon": HORIZON, "n_top": N_TOP, "fee_per_side": C.FEE, "diagnostics": diag,
           "top_precision": round(precision, 4), "top_precision_random": round(float(hit["top"].mean()), 4),
           "gross_cagr_pct": round(PF.metrics(gross)["cagr_pct"], 1), "table": rows}
    (OUT / "report.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    pd.DataFrame(curves).to_csv(OUT / "equity.csv")
    md = [f"# Modèle ensemble : {N_TOP} lignes à poids égaux, renouvelées tous les {HORIZON} jours de bourse", "",
          f"Cible « {TARGET} », variante de features « {variant} ». Période hors échantillon {scores.index[0]:%Y-%m-%d} → "
          f"{scores.index[-1]:%Y-%m-%d}, frais {100 * C.FEE:.1f} % par sens.", "", table.to_markdown(index=False), "",
          f"- Corrélation de rang score / rendement à {HORIZON} j : modèle ensemble {diag['modèle ensemble']['ic_mean']:+.4f} "
          f"(t {diag['modèle ensemble']['ic_t']}), XGBoost {diag['XGBoost']['ic_mean']:+.4f} (t {diag['XGBoost']['ic_t']}).",
          f"- Part des {N_TOP} titres choisis qui sont réellement dans les {N_TOP} meilleurs de la semaine : "
          f"{100 * precision:.1f} % (hasard : {100 * rep['top_precision_random']:.1f} %).",
          f"- CAGR brut (sans frais) du modèle ensemble : {rep['gross_cagr_pct']:.1f} %.", ""]
    (OUT / "report.md").write_text("\n".join(md))
    fig, ax = plt.subplots(figsize=(13, 6))
    for (name, v), c in zip(curves.items(), ("tab:orange", "tab:green", "tab:gray", "tab:blue")):
        ax.plot(v.index, v, lw=1.8 if "ensemble" in name or "indice" in name else 1.1, color=c,
                label=f"{name} (CAGR {PF.metrics(v)['cagr_pct']:.1f} %)")
    ax.set_yscale("log")
    ax.set_ylabel("Valeur (base 1, log)")
    ax.set_title(f"{N_TOP} lignes à poids égaux renouvelées chaque semaine — hors échantillon, net de frais")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "equity.png", dpi=110)
    plt.close(fig)
    print("\n".join(md))
    return rep


def main(variant="pit"):
    return evaluate(train(variant), variant)


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "pit")
