"""Entraînement XGBoost en walk-forward — portage de ref/MyQTM/src/train.py sans fuite temporelle.

ref/MyQTM entraînait deux modèles A/B sur des blocs de 90 j entrelacés couvrant toute la période, chacun
prédisant les blocs de l'autre : un bloc prédit est encadré par des blocs d'apprentissage postérieurs, et
le label (tendance ex post) déborde d'un bloc sur l'autre. Ici :
- à chaque date de coupe T (tous les RETRAIN_MONTHS mois), on n'utilise que les lignes < T dont le label
  est définitif avec les cours < T (trends.trend_labels → final_until) ;
- la mécanique de ref/MyQTM est conservée À L'INTÉRIEUR de cette fenêtre : blocs A/B entrelacés de 90 j,
  6 modèles pour la stabilité de l'importance (moyenne / écart-type^puissance), sélection des meilleures
  features, modèle A arrêté sur le F1 macro des blocs B et inversement ;
- ce F1 (et le diagnostic hors échantillon) ne compte que le début des tendances : un jour haussier ou
  baissier n'entre dans le F1 que s'il est dans les C.F1_TREND_HEAD (20 %) premiers jours de son segment,
  les jours de range comptent tous ; l'apprentissage, lui, utilise toutes les lignes ;
- les dates T ≤ date < T suivant sont prédites par la moyenne des modèles A et B → prédictions 100 % hors
  échantillon, écrites dans data/portefolio_strategy/myqtm/predictions.parquet.

    python -m src.portefolio_strategy.myqtm.train
    python -m src.portefolio_strategy.myqtm.train all --no-ab   # un modèle par coupe, sans blocs A/B
    python -m src.portefolio_strategy.myqtm.train all --severe  # filtrage A/B sévère : 20 ou 30 features
    python -m src.portefolio_strategy.myqtm.train all --no-ab --first=2024-01-01   # walk-forward dès 2024
    python -m src.portefolio_strategy.myqtm.train all --top=40 --first=2024-01-01  # filtrage A/B : 40 features
    python -m src.portefolio_strategy.myqtm.train all --head-only --first=2024-01-01  # sans les fins de tendance
    python -m src.portefolio_strategy.myqtm.train all --head-only=0.15 --first=2024-01-01  # début = 15 % premiers
    python -m src.portefolio_strategy.myqtm.train all --select-head=0.2 --first=2024-01-01  # features choisies sur
                                                    # les débuts de tendance, modèle final appris sur tout
    python -m src.portefolio_strategy.myqtm.train all --tail-weight=0.1 --first=2024-01-01  # fins de tendance
                                                    # pondérées : 1 jusqu'à 30 %, puis linéaire jusqu'à 0,1
    python -m src.portefolio_strategy.myqtm.train all --tail-sample=0.1 --first=2024-01-01  # fins de tendance
                            # sous-échantillonnées : P(garder) = 1 jusqu'à 30 %, puis linéaire jusqu'à 0,1
"""
import json
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score

from . import config as C
from . import features as F
from .trends import trend_labels

PREDICTIONS = C.DATA / "predictions.parquet"
# Prédictions de validation de chaque coupe (blocs B prédits par le modèle A, blocs A par le modèle B, toutes
# antérieures à la coupe) : distribution de sortie propre à chaque modèle, pour des seuils relatifs causaux.
VAL_PREDICTIONS = C.DATA / "val_predictions.parquet"


def _first_sfx(first):
    return "" if first in (None, C.FIRST_OOS_DATE) else f"_from{first[:4]}"


def _head_sfx(head):
    """head : False, True (= C.F1_TREND_HEAD par défaut, 0,2) ou fraction de début de tendance gardée."""
    if not head:
        return ""
    return "_head" if head is True or abs(head - 0.2) < 1e-9 else f"_head{round(100 * head)}"


def _top_sfx(tops):
    return f"_top{'-'.join(map(str, tops))}" if tops else ""


def _tw_sfx(tw):
    return "" if tw is None else f"_tw{round(100 * tw)}"


def tail_weight(phase, tw, start=None):
    """Poids d'apprentissage selon la position dans la tendance : 1 en range et avant `start` (C.TAIL_RAMP_START),
    puis décroissance linéaire jusqu'à `tw` au dernier jour de la tendance."""
    start = C.TAIL_RAMP_START if start is None else start
    ph = np.nan_to_num(np.asarray(phase, dtype=float), nan=0.0)
    return np.where(ph < start, 1.0, 1.0 - (1.0 - tw) * (ph - start) / (1.0 - start))


def _ts_sfx(ts):
    if ts is None:
        return ""
    return f"_ts{round(100 * ts)}" + ("" if C.TAIL_SAMPLE_SEED == 42 else f"s{C.TAIL_SAMPLE_SEED}")


def tail_sample(phase, ts, seed=None):
    """Sous-échantillonnage progressif des fins de tendance : chaque ligne est gardée avec la probabilité
    tail_weight(phase, ts) (1 en range et avant C.TAIL_RAMP_START). Tirage fixe (graine) → même tirage pour une
    ligne donnée à chaque coupe, tant que df garde le même ordre."""
    u = np.random.default_rng(C.TAIL_SAMPLE_SEED if seed is None else seed).random(len(phase))
    return u < tail_weight(phase, ts)


def _sel_sfx(sel):
    return f"_sel{round(100 * sel)}" if sel else ""


def predictions_path(variant="all", ab=True, severe=False, first=None, tops=None, head=False, sel=None, tw=None,
                     ts=None):
    """Variante « all » (tout le sentiment disponible) = fichiers historiques ; les autres ont un suffixe.
    ab=False (sans blocs A/B) → suffixe _noab ; severe=True (filtrage A/B sévère) → _severe ;
    tops=[n, …] (filtrage A/B à n features, choix par F1 entre les valeurs) → _top{n-…} ;
    head=True (apprentissage sans les jours de tendance évidents, après les C.F1_TREND_HEAD premiers) → _head ;
    sel=x (classement des features sur les x premiers % des tendances + range, modèle final sur tout) → _sel{x} ;
    tw=p (fins de tendance pondérées, poids 1 → p après C.TAIL_RAMP_START) → _tw{p} ;
    ts=p (fins de tendance sous-échantillonnées, P(garder) 1 → p après C.TAIL_RAMP_START) → _ts{p}."""
    sfx = ("" if ab else "_noab") + ("_severe" if severe else "") + _top_sfx(tops) + _head_sfx(head) + _sel_sfx(sel) + _tw_sfx(tw) + _ts_sfx(ts) \
        + _first_sfx(first)
    return C.DATA / f"predictions{sfx}.parquet" if variant == "all" else C.DATA / f"predictions_{variant}{sfx}.parquet"


def out_dir(variant="all", ab=True, severe=False, first=None, tops=None, head=False, sel=None, tw=None, ts=None):
    base = C.OUT if variant == "all" else C.OUT / variant
    base = base if ab else base / "noab"
    base = base / "severe" if severe else base
    base = base / _top_sfx(tops).lstrip("_") if tops else base
    base = base / _head_sfx(head).lstrip("_") if head else base
    base = base / _sel_sfx(sel).lstrip("_") if sel else base
    base = base / _tw_sfx(tw).lstrip("_") if tw is not None else base
    base = base / _ts_sfx(ts).lstrip("_") if ts is not None else base
    return base / _first_sfx(first).lstrip("_") if _first_sfx(first) else base


def cutoffs(first, last):
    out, t = [], pd.Timestamp(first)
    while t <= last:
        out.append(t)
        t += pd.DateOffset(months=C.RETRAIN_MONTHS)
    return out


def labels_before(df, cutoff, with_phase=False):
    """Labels définitifs connus avec les cours < cutoff → (y, f1_keep) alignés sur df.

    y : classe 0/1/2 (NaN = pas de label) ; f1_keep : la ligne compte dans le F1 (range, ou début de tendance :
    phase < C.F1_TREND_HEAD). with_phase=True → (y, f1_keep, phase), phase NaN en range ou sans label."""
    y = pd.Series(np.nan, index=df.index)
    keep = pd.Series(False, index=df.index)
    phs = pd.Series(np.nan, index=df.index)
    for _, g in df[df["date"] < cutoff].groupby("symbol", sort=False):
        lab, phase, final_until = trend_labels(g.set_index("date")["c"], with_phase=True, **C.TREND)
        ok = g["date"].to_numpy() < np.datetime64(final_until)
        y.loc[g.index[ok]] = lab.to_numpy()[ok]
        ph = phase.to_numpy()[ok]
        keep.loc[g.index[ok]] = np.isnan(ph) | (ph < C.F1_TREND_HEAD)
        phs.loc[g.index[ok]] = ph
    return (y, keep, phs) if with_phase else (y, keep)


def interval_index(dates, start):
    return ((dates - pd.Timestamp(start)).dt.days // C.INTERVAL_DAYS).to_numpy()


def _f1(predt, dmat):
    return "f1", float(f1_score(dmat.get_label(), np.argmax(predt, axis=1), average="macro"))


def _params():
    p = dict(C.XGB_PARAMS, disable_default_eval_metric=1)
    try:
        import torch
        if torch.cuda.is_available():
            p["device"] = "cuda"
    except ImportError:
        pass
    return p


def fit(X, y, Xv, yv, rounds=C.XGB_ROUNDS, w=None):
    dtr, dva = xgb.DMatrix(X, label=y, weight=w), xgb.DMatrix(Xv, label=yv)
    m = xgb.train(_params(), dtr, num_boost_round=rounds, evals=[(dva, "eval")], custom_metric=_f1,
                  maximize=True, early_stopping_rounds=C.XGB_PATIENCE, verbose_eval=False)
    return m, m.best_iteration + 1, m.best_score


def select_features(X, y, idx, cols, keep, w=None):
    """6 modèles (3 sous-blocs A, 3 sous-blocs B) → importance stable = moyenne / écart-type^puissance.
    keep : lignes comptées dans le F1 d'arrêt (début de tendance ou range)."""
    a, b = (idx % 2 == 0) & keep, (idx % 2 == 1) & keep
    imp = {}
    for k in range(6):
        part = idx % 6 == k
        if part.sum() < 200 or len(np.unique(y[part])) < 3:
            continue
        other = b if k % 2 == 0 else a
        m, _, _ = fit(X[part], y[part], X[other], y[other], rounds=200, w=None if w is None else w[part])
        s = m.get_score(importance_type="weight")
        imp[k] = [s.get(f"f{j}", 0) for j in range(len(cols))]
    d = pd.DataFrame(imp, index=cols)
    d = d / d.max().replace(0, 1)
    d = d[(d != 0).all(axis=1)]
    d["mean"], d["std"] = d.mean(axis=1), d.std(axis=1)
    d["score"] = d["mean"] / (d["std"] ** C.MEAN_STD_POWER + 1e-9)
    return d.sort_values("score", ascending=False)


def train_cutoff(df, cols, y, cutoff, nxt, keep, tops=None, head=False, sel=False, w=None, sample=None):
    """head=True : les jours de tendance au-delà des C.F1_TREND_HEAD premiers (zones évidentes) sont retirés de
    l'apprentissage, de la sélection des features et de l'arrêt anticipé ; restent le range et les débuts.
    sel=True : seuls les 6 modèles de classement des features sont appris sur le range et les débuts ; les modèles
    A/B finaux apprennent sur toutes les lignes (arrêt anticipé toujours sur le range et les débuts).
    w : poids d'apprentissage alignés sur df (fins de tendance pondérées, tail_weight) ; le F1 d'arrêt n'est pas pondéré.
    sample : masque booléen aligné sur df (tail_sample) ; les lignes écartées sortent de l'apprentissage, du
    classement des features et des jeux d'arrêt anticipé."""
    tr = (df["member"] & (df["date"] >= C.START_DATE) & (df["date"] < cutoff) & y.notna()).to_numpy()
    if head:
        tr &= keep.to_numpy()
    if sample is not None:
        tr &= np.asarray(sample)
    te = (df["member"] & (df["date"] >= cutoff) & (df["date"] < nxt)).to_numpy()
    if not te.any():
        return None
    X, yy = df.loc[tr, cols].to_numpy(np.float32), y[tr].to_numpy(int)
    idx = interval_index(df.loc[tr, "date"], C.START_DATE)
    a, b = idx % 2 == 0, idx % 2 == 1
    kk = keep[tr].to_numpy()
    va, vb = a & kk, b & kk                      # jeux d'évaluation du F1 : début de tendance ou range
    ww = None if w is None else np.asarray(w)[tr]
    rank = select_features(X[kk], yy[kk], idx[kk], cols, kk[kk]) if sel else select_features(X, yy, idx, cols, kk, ww)
    best = None
    for top in (tops or C.TOP_FEATURES):
        sel = [cols.index(c) for c in rank.index[:top]]
        ma, na, f1a = fit(X[a][:, sel], yy[a], X[vb][:, sel], yy[vb], w=None if ww is None else ww[a])
        mb, nb, f1b = fit(X[b][:, sel], yy[b], X[va][:, sel], yy[va], w=None if ww is None else ww[b])
        f1 = (f1a + f1b) / 2
        if best is None or f1 > best["f1"]:
            best = dict(f1=f1, f1a=f1a, f1b=f1b, sel=sel, ma=ma, mb=mb, na=na, nb=nb, top=len(sel))
    Xt = xgb.DMatrix(df.loc[te, cols].to_numpy(np.float32)[:, best["sel"]])
    p = (best["ma"].predict(Xt, iteration_range=(0, best["na"])) + best["mb"].predict(Xt, iteration_range=(0, best["nb"]))) / 2
    out = df.loc[te, ["date", "symbol", "o", "c"]].copy()
    out["ybear"], out["yrange"], out["ybull"] = p[:, 0], p[:, 1], p[:, 2]
    sel = best["sel"]
    pv = np.empty((len(yy), 3), dtype=np.float32)
    pv[b] = best["ma"].predict(xgb.DMatrix(X[b][:, sel]), iteration_range=(0, best["na"]))
    pv[a] = best["mb"].predict(xgb.DMatrix(X[a][:, sel]), iteration_range=(0, best["nb"]))
    val = df.loc[tr, ["date", "symbol"]].copy()
    val["cutoff"] = cutoff
    val["ybear"], val["yrange"], val["ybull"], val["y"] = pv[:, 0], pv[:, 1], pv[:, 2], yy
    info = dict(cutoff=f"{cutoff:%Y-%m-%d}", train_rows=int(tr.sum()), test_rows=int(te.sum()),
                train_until=f"{df.loc[tr, 'date'].max():%Y-%m-%d}", f1_cv=round(best["f1"], 4),
                top_features=best["top"], rounds=[best["na"], best["nb"]],
                class_share=[round(float((yy == k).mean()), 3) for k in range(3)])
    return out, info, rank, val


def train_cutoff_noab(df, cols, y, cutoff, nxt):
    """Un seul modèle sur toute la fenêtre < cutoff, toutes les features, NOAB_ROUNDS arbres, sans validation."""
    tr = (df["member"] & (df["date"] >= C.START_DATE) & (df["date"] < cutoff) & y.notna()).to_numpy()
    te = (df["member"] & (df["date"] >= cutoff) & (df["date"] < nxt)).to_numpy()
    if not te.any():
        return None
    X, yy = df.loc[tr, cols].to_numpy(np.float32), y[tr].to_numpy(int)
    m = xgb.train(_params(), xgb.DMatrix(X, label=yy), num_boost_round=C.NOAB_ROUNDS)
    p = m.predict(xgb.DMatrix(df.loc[te, cols].to_numpy(np.float32)))
    out = df.loc[te, ["date", "symbol", "o", "c"]].copy()
    out["ybear"], out["yrange"], out["ybull"] = p[:, 0], p[:, 1], p[:, 2]
    s = m.get_score(importance_type="weight")
    rank = pd.DataFrame({"score": [s.get(f"f{j}", 0) for j in range(len(cols))]}, index=cols)
    info = dict(cutoff=f"{cutoff:%Y-%m-%d}", train_rows=int(tr.sum()), test_rows=int(te.sum()),
                train_until=f"{df.loc[tr, 'date'].max():%Y-%m-%d}", top_features=len(cols), rounds=[C.NOAB_ROUNDS],
                class_share=[round(float((yy == k).mean()), 3) for k in range(3)])
    return out, info, rank, None


def main(df=None, variant="all", ab=True, severe=False, first=None, tops=None, head=False, sel=None, tw=None,
         ts=None):
    df = pd.read_parquet(F.FEATURES) if df is None else df
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    cols = F.feature_columns(df, variant)
    out = out_dir(variant, ab, severe, first, tops, head, sel, tw, ts)
    path = predictions_path(variant, ab, severe, first, tops, head, sel, tw, ts)
    custom = bool(tops)
    tops = tops or (C.TOP_FEATURES_SEVERE if severe else C.TOP_FEATURES)
    print(f"variante « {variant} » ({F.VARIANTS[variant]}) : {len(cols)} features"
          + ("" if ab else " — sans blocs A/B") + (f" — filtrage A/B {tops} features" if severe or custom else "")
          + (f" — sans les fins de tendance (début = {C.F1_TREND_HEAD:.0%} premiers)" if head else "")
          + (f" — features choisies sur les débuts de tendance ({C.F1_TREND_HEAD:.0%} premiers)" if sel else "")
          + (f" — fins de tendance pondérées (1 → {tw} après {C.TAIL_RAMP_START:.0%})" if tw is not None else "")
          + (f" — fins de tendance sous-échantillonnées (1 → {ts} après {C.TAIL_RAMP_START:.0%})" if ts is not None else ""),
          flush=True)
    last = df["date"].max()
    cuts = cutoffs(first or C.FIRST_OOS_DATE, last)
    preds, infos, ranks, vals, t0 = [], [], [], [], time.time()
    for i, cut in enumerate(cuts):
        nxt = cuts[i + 1] if i + 1 < len(cuts) else last + pd.Timedelta(days=1)
        y, keep, phase = labels_before(df, cut, with_phase=True)
        w = None if tw is None else tail_weight(phase, tw)
        smp = None if ts is None else tail_sample(phase.to_numpy(), ts)
        r = train_cutoff(df, cols, y, cut, nxt, keep, tops, head, bool(sel), w, smp) if ab else train_cutoff_noab(df, cols, y, cut, nxt)
        if r is None:
            continue
        preds.append(r[0])
        infos.append(r[1])
        vals.append(r[3])
        ranks.append(r[2]["score"].rename(r[1]["cutoff"]))
        print(f"  {r[1]}  [{time.time() - t0:.0f} s]", flush=True)
    p = pd.concat(preds, ignore_index=True)
    # Diagnostic seulement (jamais utilisé pour décider) : F1 hors échantillon contre les labels définitifs.
    y_full, keep_full = labels_before(df, last + pd.Timedelta(days=1))
    p = p.merge(pd.DataFrame({"date": df["date"], "symbol": df["symbol"], "y": y_full, "f1_keep": keep_full}),
                on=["date", "symbol"])
    p["cls"] = p[["ybear", "yrange", "ybull"]].to_numpy().argmax(axis=1)
    lab = p["y"].notna()
    ok = lab & p["f1_keep"]                      # F1 de référence : début de tendance ou range
    f1 = f1_score(p.loc[ok, "y"].astype(int), p.loc[ok, "cls"], average="macro")
    acc = float((p.loc[ok, "y"].astype(int) == p.loc[ok, "cls"]).mean())
    f1_all = f1_score(p.loc[lab, "y"].astype(int), p.loc[lab, "cls"], average="macro")
    p.to_parquet(path, index=False)
    if variant == "all" and ab and not severe and not custom and not head and not sel and tw is None and ts is None and not _first_sfx(first):
        pd.concat(vals, ignore_index=True).to_parquet(VAL_PREDICTIONS, index=False)
    out.mkdir(parents=True, exist_ok=True)
    imp = pd.concat(ranks, axis=1)
    imp["mean_rank_score"] = imp.mean(axis=1)
    imp.sort_values("mean_rank_score", ascending=False).to_csv(out / "feature_importance.csv")
    sent_cov = float(df.loc[df["member"] & (df["date"] >= C.START_DATE), "ts_news_sentiment_company_0"].notna().mean())
    vend_cov = float(df.loc[df["member"] & (df["date"] >= C.START_DATE), "vs_n5"].notna().mean()) \
        if "vs_n5" in df else 0.0
    rep = dict(variant=variant, vendor_sentiment_coverage=round(vend_cov, 4), oos_macro_f1=round(float(f1), 4), oos_accuracy=round(acc, 4),
               oos_macro_f1_all_days=round(float(f1_all), 4), f1_trend_head=C.F1_TREND_HEAD, n_predictions=len(p),
               first=f"{p['date'].min():%Y-%m-%d}", last=f"{p['date'].max():%Y-%m-%d}", n_features=len(cols),
               sentiment_coverage=round(sent_cov, 4), cutoffs=infos)
    (out / "train_report.json").write_text(json.dumps(rep, indent=2))
    print(f"prédictions hors échantillon : {len(p)} lignes {rep['first']} → {rep['last']} ; "
          f"F1 macro début de tendance {f1:.3f} (tous les jours {f1_all:.3f}), exactitude {acc:.3f} ; "
          f"couverture sentiment {sent_cov:.1%}")
    return p


if __name__ == "__main__":
    import sys
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    first = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--first=")), None)
    top = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--top=")), None)
    head = next((float(a.split("=", 1)[1]) for a in sys.argv if a.startswith("--head-only=")), "--head-only" in sys.argv)
    if head and head is not True:
        C.F1_TREND_HEAD = head                   # la même fraction sert à l'apprentissage et au F1 d'arrêt
    sel = next((float(a.split("=", 1)[1]) for a in sys.argv if a.startswith("--select-head=")), None)
    if sel:
        C.F1_TREND_HEAD = sel                    # même fraction pour le classement des features et le F1 d'arrêt
    tw = next((float(a.split("=", 1)[1]) for a in sys.argv if a.startswith("--tail-weight=")), None)
    ts = next((float(a.split("=", 1)[1]) for a in sys.argv if a.startswith("--tail-sample=")), None)
    C.TAIL_SAMPLE_SEED = next((int(a.split("=", 1)[1]) for a in sys.argv if a.startswith("--sample-seed=")),
                              C.TAIL_SAMPLE_SEED)
    main(variant=args[0] if args else "all", ab="--no-ab" not in sys.argv, severe="--severe" in sys.argv, first=first,
         tops=[int(t) for t in top.split(",")] if top else None, head=head, sel=sel, tw=tw, ts=ts)
