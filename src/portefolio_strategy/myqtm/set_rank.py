"""Modèle « ensemble » quotidien sur le S&P 500 : il voit les ~500 membres en même temps et les classe.

Contrairement à monthly_rank.py (XGBoost, un titre à la fois, une ligne par mois), l'entrée d'un jour est le tableau
complet (membres × features) et la sortie un score par titre ; un tableau par séance depuis fin 2015.
- chaque feature est remplacée par son rang du jour parmi les membres (−0,5 … +0,5 ; manquant = 0) : les variables
  identiques pour tous les titres (macro, marché) disparaissent, le modèle ne voit que des positions relatives ;
- petit transformer sans position (set_model.py : chaque titre « regarde » les autres avant de recevoir son score) ;
- cible : rang du jour du rendement des HORIZON séances suivantes (ouverture J+1 → ouverture J+1+HORIZON), perte
  quadratique. Deux jours consécutifs ont des cibles qui se recouvrent à 95 % : beaucoup de tableaux, peu de mois
  indépendants.
Walk-forward calé sur monthly_rank.py : modèle réappris tous les RETRAIN mois sur les jours dont la cible est connue
à la décision, SEEDS initialisations moyennées, première décision fin décembre 2018. Mêmes features (186).

Sorties (data/portefolio_strategy/myqtm_sp) : set_x.npy (rangs, float16), set_meta.npz, set_preds.parquet (score
quotidien hors échantillon), monthly_preds_set.parquet (monthly_preds + pred_set aux fins de mois).

    python -m src.portefolio_strategy.myqtm.set_rank [--build] [--train] [--epochs=N] [--dim=N] [--tag=nom]
"""
import sys
import time

import numpy as np
import pandas as pd

from . import config as C
from . import monthly_rank as M
from .. import set_model as SM

DATA = M.DATA
X_PATH, META = DATA / "set_x.npy", DATA / "set_meta.npz"
HORIZON = 21
EPOCHS, BATCH, LR, WEIGHT_DECAY = 10, 32, 1e-3, 1e-2
SEEDS = (42, 7, 123)
TAG = ""


def preds_path():
    return DATA / f"set_preds{TAG}.parquet"


def monthly_path():
    return DATA / f"monthly_preds_set{TAG}.parquet"


def build():
    cols = M.feature_columns(pd.read_parquet(M.FEATS))
    df = M.daily_frame()
    of = df.pivot(index="date", columns="symbol", values="o").sort_index().ffill()
    fwd = (of.shift(-(HORIZON + 1)) / of.shift(-1) - 1).stack(future_stack=True).rename("fwd").reset_index()
    d = df.loc[df["member"], ["date", "symbol"] + cols].merge(fwd, on=["date", "symbol"], how="left")
    d = d.sort_values(["date", "symbol"]).reset_index(drop=True)
    ranks = d.groupby("date")[cols].rank(pct=True).sub(0.5).fillna(0.0).to_numpy(np.float16)
    dates = np.sort(d["date"].unique())
    di = pd.Series(np.arange(len(dates)), index=dates)[d["date"]].to_numpy()
    slot = d.groupby("date").cumcount().to_numpy()
    n = int(slot.max()) + 1
    X = np.zeros((len(dates), n, len(cols)), np.float16)
    mask = np.zeros((len(dates), n), bool)
    Y = np.full((len(dates), n), np.nan, np.float32)
    sym = np.full((len(dates), n), "", "U12")
    X[di, slot], mask[di, slot], Y[di, slot], sym[di, slot] = ranks, True, d["fwd"].to_numpy(np.float32), d["symbol"].to_numpy()
    np.save(X_PATH, X)
    np.savez(META, mask=mask, y=Y, dates=dates, sym=sym, cols=np.array(cols))
    print(f"tableaux : {X.shape[0]} jours × {X.shape[1]} titres × {X.shape[2]} features, "
          f"{pd.Timestamp(dates[0]):%Y-%m-%d} → {pd.Timestamp(dates[-1]):%Y-%m-%d}", flush=True)


def fit_predict(X, Mk, T, tr, te, seed, dev):
    import torch
    model = SM.make_model(X.shape[2], seed).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    tr = torch.from_numpy(tr).to(dev)
    g = torch.Generator().manual_seed(seed)
    for _ in range(EPOCHS):
        model.train()
        for idx in torch.randperm(len(tr), generator=g).split(BATCH):
            idx = tr[idx.to(dev)]
            m = Mk[idx]
            loss = (((model(X[idx], m) - T[idx]) ** 2) * m).sum() / m.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    out = []
    with torch.no_grad():
        for idx in torch.from_numpy(te).to(dev).split(128):
            out.append(model(X[idx], Mk[idx]).float().cpu().numpy())
    return np.concatenate(out)


def train():
    import torch
    torch.set_num_threads(C.WORKERS)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    meta = np.load(META)
    mask, Y, dates, sym = meta["mask"], meta["y"], meta["dates"], meta["sym"]
    T = SM.rank_target(Y, mask)
    known = (mask & np.isfinite(Y)).sum(1) > 100
    X = torch.from_numpy(np.load(X_PATH)).to(dev).float()
    Mk, Tt = torch.from_numpy(mask).to(dev), torch.from_numpy(T).to(dev)
    mp = pd.read_parquet(M.PREDS)
    months = np.sort(mp["date"].unique())
    months = months[months >= np.datetime64(M.FIRST_DECISION)]
    pos = np.searchsorted(dates, months)                    # position de chaque fin de mois dans les jours
    print(f"modèle ensemble S&P 500 : {X.shape[0]} jours × {X.shape[1]} titres × {X.shape[2]} features, horizon {HORIZON} j, "
          f"{EPOCHS} époques, dim {SM.DIM}, {len(months)} fins de mois, {dev}", flush=True)
    rows, t0 = [], time.time()
    for k in range(0, len(months), M.RETRAIN):
        end = pos[k + M.RETRAIN] if k + M.RETRAIN < len(months) else len(dates)
        tr = np.flatnonzero(known & (np.arange(len(dates)) + HORIZON + 1 <= pos[k]))     # cible connue à la décision
        te = np.arange(pos[k], end)
        s = np.mean([fit_predict(X, Mk, Tt, tr, te, seed, dev) for seed in SEEDS], axis=0)
        for i, t in enumerate(te):
            m = mask[t]
            rows.append(pd.DataFrame({"date": dates[t], "symbol": sym[t][m], "pred_set": s[i][m], "fwd": Y[t][m]}))
        print(f"  {pd.Timestamp(months[k]):%Y-%m-%d} : {len(tr)} jours d'apprentissage jusqu'au "
              f"{pd.Timestamp(dates[tr[-1]]):%Y-%m-%d} [{time.time() - t0:.0f} s]", flush=True)
    p = pd.concat(rows, ignore_index=True)
    p.to_parquet(preds_path(), index=False)
    out = mp.merge(p[["date", "symbol", "pred_set"]], on=["date", "symbol"], how="left")
    out.to_parquet(monthly_path(), index=False)
    ic = p.dropna(subset=["fwd"]).groupby("date")[["pred_set", "fwd"]].apply(lambda g: g["pred_set"].corr(g["fwd"], method="spearman"))
    print(f"corrélation de rang quotidienne score / rendement {HORIZON} j : {ic.mean():+.4f} "
          f"(2019-23 {ic[ic.index < '2024-01-01'].mean():+.4f}, 2024-26 {ic[ic.index >= '2024-01-01'].mean():+.4f}) ; "
          f"fins de mois sans score : {int(out['pred_set'].isna().sum())} / {len(out)}", flush=True)


if __name__ == "__main__":
    for a in sys.argv[1:]:
        if a.startswith("--epochs="):
            EPOCHS = int(a.split("=")[1])
        if a.startswith("--dim="):
            SM.DIM = int(a.split("=")[1])
        if a.startswith("--tag="):
            TAG = "_" + a.split("=")[1]
    if "--build" in sys.argv:
        build()
    if "--train" in sys.argv:
        train()
