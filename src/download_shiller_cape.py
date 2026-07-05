"""
CAPE (Shiller P/E10) + Excess CAPE Yield (ECY) du S&P 500, historique COMPLET
1871 -> aujourd'hui, avec mise a jour quotidienne.

Sources :
  - CAPE  : table multpl.com/shiller-pe (historique mensuel complet + point du
            jour, vrais benefices E10 a jour). Fallback = ie_data.xls de Shiller
            (Yale, fige a ~2024-09) si multpl echoue. Les deux sont croises pour
            coherence. NB : "geler le E10 de Shiller et scaler par le prix" est
            faux (~+6% sur 2 ans car les benefices montent) -> on prend le CAPE
            deja calcule par multpl.
  - ECY   : colonne 'Excess CAPE Yield' du fichier Shiller pour l'historique
            (1881 -> derniere date Shiller), puis extension du tail par
            1/CAPE - taux reel 10 ans (FRED DFII10, TIPS), RECALEE sur le niveau
            Shiller. Les TIPS n'existent que depuis 2003 : pas de calcul maison
            sur toute l'histoire, d'ou l'usage de la colonne Shiller pour le passe.

Variable de CONTEXTE de valorisation (webapp + panneau backtest), ajustee des
taux et comparable entre epoques. PAS dans la strategie (garde-fou candidat, pas
signal -- cf. l'analyse : la valorisation explique a posteriori mais ne time pas).

Sorties : data/shiller/cape_ecy.parquet (mensuel : cape, ecy) + outputs/shiller/cape_ecy.png
Usage   : python -m src.download_shiller_cape       (a lancer quotidiennement, cf. cron)
"""

import io
import os
import json
import ssl
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "shiller"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR = ROOT / "outputs" / "shiller"
OUT_DIR.mkdir(parents=True, exist_ok=True)

XLS_URLS = [
    "https://img1.wsimg.com/blobby/go/e5e77e0b-59d1-44d9-ab25-4763ac982e53/downloads/ie_data.xls",
    "http://www.econ.yale.edu/~shiller/data/ie_data.xls",
]
MULTPL_URL = "https://www.multpl.com/shiller-pe/table/by-month"

_CTX = ssl.create_default_context()
_CTX.check_hostname = False
_CTX.verify_mode = ssl.CERT_NONE


def _http(url, timeout=45):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return urllib.request.urlopen(req, timeout=timeout, context=_CTX).read()


# ── CAPE : multpl (primaire) ──────────────────────────────
def fetch_multpl_cape():
    """Serie mensuelle du CAPE (Shiller PE) depuis multpl : historique complet
    + point du jour. Index = debut de mois. None si le scrape echoue."""
    try:
        html = _http(MULTPL_URL).decode("utf-8", "ignore")
        t = pd.read_html(io.StringIO(html))[0]
        t.columns = ["date", "cape"]
        t["date"] = pd.to_datetime(t["date"], errors="coerce")
        t["cape"] = pd.to_numeric(t["cape"].astype(str).str.extract(r"([0-9.]+)")[0], errors="coerce")
        t = t.dropna()
        # le dernier point est date du jour (valeur live du mois courant) -> normalise au mois
        t["date"] = t["date"].dt.to_period("M").dt.to_timestamp()
        s = t.set_index("date")["cape"].sort_index()
        s = s[~s.index.duplicated(keep="last")]
        return s
    except Exception as e:  # noqa: BLE001
        print(f"  WARN multpl indisponible : {repr(e)[:90]}")
        return None


# ── Fichier Shiller (ECY historique + fallback CAPE + cross-check) ──
def _shiller_date(v):
    y = int(v)
    m = min(max(int(round((v - y) * 100)), 1), 12)
    return pd.Timestamp(year=y, month=m, day=1)


def _locate_columns(df):
    header = df.iloc[0:8].astype(str).apply(lambda c: " ".join(c.tolist()).upper(), axis=0)
    cape_col = ecy_col = None
    for col, h in header.items():
        if "CAPE" in h and "TR" not in h and "EXCESS" not in h and cape_col is None:
            cape_col = col
        if "EXCESS" in h and "CAPE" in h:
            ecy_col = col
    return (cape_col if cape_col is not None else 12), (ecy_col if ecy_col is not None else 16)


def fetch_shiller():
    """(cape, ecy) mensuels du fichier ie_data.xls de Shiller. None si echec."""
    raw = None
    for u in XLS_URLS:
        try:
            raw = _http(u)
            print(f"  Shiller xls : {len(raw)} octets <- {u.split('/')[2]}")
            break
        except Exception as e:  # noqa: BLE001
            print(f"  Shiller xls echec {u.split('/')[2]} : {repr(e)[:70]}")
    if raw is None:
        return None
    df = pd.read_excel(io.BytesIO(raw), sheet_name="Data", header=None)
    cape_col, ecy_col = _locate_columns(df)
    out = df[[0, cape_col, ecy_col]].copy()
    out.columns = ["date", "cape", "ecy"]
    out["date"] = pd.to_numeric(out["date"], errors="coerce")
    out = out[out["date"].between(1871, 2100)]
    out["date"] = out["date"].map(_shiller_date)
    for c in ("cape", "ecy"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.set_index("date").sort_index()


# ── Taux reel 10 ans (FRED DFII10, TIPS) pour le tail ECY ──
def fetch_dfii10_monthly():
    """Taux reel 10 ans (TIPS) mensuel en fraction (0.0225 = 2.25%). None si echec.
    Ne remonte qu'a 2003 (les TIPS n'existent pas avant)."""
    key = os.environ.get("FRED")
    if not key:
        load_dotenv(ROOT / ".env")
        key = os.environ.get("FRED")
    if not key:
        print("  WARN FRED absent -> pas d'extension ECY")
        return None
    try:
        u = (f"https://api.stlouisfed.org/fred/series/observations?series_id=DFII10"
             f"&api_key={key}&file_type=json")
        obs = json.loads(_http(u))["observations"]
        s = pd.Series({o["date"]: float(o["value"]) for o in obs if o["value"] not in (".", "")})
        s.index = pd.to_datetime(s.index)
        return s.resample("MS").mean() / 100.0
    except Exception as e:  # noqa: BLE001
        print(f"  WARN DFII10 indisponible : {repr(e)[:80]}")
        return None


# ── Assemblage ────────────────────────────────────────────
def build():
    multpl = fetch_multpl_cape()
    shiller = fetch_shiller()

    if multpl is not None:
        cape = multpl
        src = "multpl"
    elif shiller is not None:
        cape = shiller["cape"].dropna()
        src = "Shiller xls (FALLBACK, fige)"
    else:
        raise RuntimeError("aucune source CAPE disponible (multpl + Shiller ont echoue)")
    print(f"  CAPE source = {src} : {len(cape)} mois, {cape.index.min():%Y-%m} -> {cape.index.max():%Y-%m}")

    # cross-check multpl vs Shiller sur l'historique commun
    if multpl is not None and shiller is not None:
        sc = shiller["cape"].dropna()
        common = multpl.index.intersection(sc.index)
        if len(common) > 12:
            diff = (multpl.loc[common] - sc.loc[common]).abs() / sc.loc[common]
            print(f"  cross-check CAPE ({len(common)} mois communs) : "
                  f"ecart median {diff.median()*100:.2f}%, max {diff.max()*100:.2f}%")

    # ECY = colonne Shiller (historique) + tail (1/CAPE - DFII10) recale
    if shiller is not None:
        ecy_hist = shiller["ecy"].dropna()
    else:
        ecy_hist = pd.Series(dtype=float)
    ecy = ecy_hist
    dfii = fetch_dfii10_monthly()
    if dfii is not None and not ecy_hist.empty:
        recon = (1.0 / cape) - dfii                 # aligne sur les mois communs
        recon = recon.dropna()
        common = ecy_hist.index.intersection(recon.index)
        if len(common) >= 12:
            win = common[common >= (common.max() - pd.DateOffset(months=36))]
            offset = float((ecy_hist.loc[win] - recon.loc[win]).mean())
            last_sh = ecy_hist.index.max()
            tail = (recon + offset)[recon.index > last_sh]
            ecy = pd.concat([ecy_hist, tail]).sort_index()
            ecy = ecy[~ecy.index.duplicated(keep="first")]
            print(f"  ECY : historique Shiller -> {last_sh:%Y-%m} ; tail DFII10 recale "
                  f"(offset {offset*100:+.2f} pt) -> {ecy.index.max():%Y-%m} "
                  f"({len(tail)} mois ajoutes)")
        else:
            print("  ECY : recouvrement Shiller/DFII10 insuffisant -> pas d'extension")
    else:
        print("  ECY : historique Shiller seul (pas d'extension)")

    out = pd.DataFrame({"cape": cape, "ecy": ecy}).sort_index()
    return out


# ── Graphe ────────────────────────────────────────────────
def plot(df):
    cape = df["cape"].dropna()
    ecy = df["ecy"].dropna()
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.semilogy(cape.index, cape.values, color="teal", lw=1.1, label="CAPE (Shiller P/E10)")
    med = float(cape.median())
    for lvl, lbl, col in [(med, f"mediane {med:.0f}", "grey"), (30, "CAPE=30", "orange"),
                          (44, "pic 2000 (44)", "red")]:
        ax.axhline(lvl, color=col, ls=":", lw=1, alpha=0.5)
        ax.text(cape.index[0], lvl * 1.02, lbl, fontsize=7, color=col, alpha=0.8)
    ax.set_ylabel("CAPE (log)", color="teal")
    ax.tick_params(axis="y", labelcolor="teal")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title(f"S&P 500 — CAPE & Excess CAPE Yield  (1871 -> {cape.index.max():%Y-%m-%d})",
                 fontsize=13)

    ae = ax.twinx()
    ae.plot(ecy.index, ecy.values * 100, color="darkorange", lw=1.0, alpha=0.85,
            label="Excess CAPE Yield (%)")
    ae.axhline(0, color="darkorange", ls="--", lw=0.8, alpha=0.5)
    ae.set_ylabel("ECY (%)", color="darkorange")
    ae.tick_params(axis="y", labelcolor="darkorange")
    ae.legend(loc="lower left", fontsize=9)

    for s, e in [("2000-03", "2002-10"), ("2007-10", "2009-03"),
                 ("2020-02", "2020-04"), ("2021-11", "2022-10")]:
        ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), color="red", alpha=0.08)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlim(pd.Timestamp("1900-01-01"), cape.index.max())

    out = OUT_DIR / "cape_ecy.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  graphe -> {out}")


def main():
    print("=== S&P CAPE / ECY (multpl + Shiller + FRED) ===")
    df = build()
    df.to_parquet(DATA_DIR / "cape_ecy.parquet")
    cape = df["cape"].dropna()
    ecy = df["ecy"].dropna()
    pct = float((cape <= cape.iloc[-1]).mean() * 100)
    print(f"  CAPE {cape.index.max():%Y-%m-%d} = {cape.iloc[-1]:.1f} "
          f"(mediane {cape.median():.1f}, percentile {pct:.0f})")
    if not ecy.empty:
        print(f"  ECY  {ecy.index.max():%Y-%m} = {ecy.iloc[-1]*100:+.2f}%")
    plot(df)
    print(f"  parquet -> {DATA_DIR / 'cape_ecy.parquet'}  ({len(df)} mois)")
    return df


if __name__ == "__main__":
    main()
