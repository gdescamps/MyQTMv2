"""Version « S&P 500 reconstitué » (sp500.py : 500 titres, point-in-time) du classifieur myqtm.

Même méthode que sur l'univers Nasdaq dynamique (features.py, train.py), avec ~5 fois plus de titres pour gagner
en puissance statistique. Différences :
- univers et capitalisations = sp500.UNIVERSE / sp500.CAPS (financières incluses) ; indice = moyenne pondérée par
  la capitalisation des 500 membres ;
- pas de features issues des LLM (sentiment PIT, news structurées Llama) : elles n'existent que pour l'univers
  Nasdaq ; les colonnes PIT sont retirées et les événements ne sont pas ajoutés ;
- volume de news EODHD par catégorie recalculé pour cet univers (vendor_sentiment.parquet propre) ;
- features fondamentales (comptes trimestriels) et de groupe GICS ajoutées (fundamentals.py).
Sorties dans data/portefolio_strategy/myqtm_sp et outputs/portefolio_strategy/myqtm_sp.

    python -m src.portefolio_strategy.myqtm.sp [--vendor] [--features [--base]]    # défaut : les deux
    python -m src.portefolio_strategy.myqtm.sp --augment    # (re)calcule fondamentaux + groupe sur le fichier existant
    python -m src.portefolio_strategy.myqtm.train all --sp --tail-sample=0 --first=2024-01-01
"""
import sys

import pandas as pd

from . import config as C
from . import features as F
from . import fundamentals as FD
from .. import config as PC
from .. import sp500 as SP
from .. import universe as U
from .. import vendor_sentiment as VS

DATA = PC.DATA / "myqtm_sp"
OUT = PC.OUT / "myqtm_sp"


def use_sp():
    """Bascule univers, features et dossiers sur le S&P 500 (appelé par train.py --sp et les scripts d'évaluation)."""
    U.UNIVERSE, U.CAPS = SP.UNIVERSE, SP.CAPS
    C.DATA, C.OUT = DATA, OUT
    F.FEATURES, F.INDEX = DATA / "features.parquet", DATA / "index.parquet"
    F.SENTIMENTS = DATA / "pas_de_sentiment_pit.parquet"          # absent → pas de sentiment PIT
    F.EVENTS_FILE = DATA / "pas_de_news_structurees.parquet"      # absent → pas de features ev_*
    VS.VENDOR = DATA / "vendor_sentiment.parquet"


def augment(df):
    """Ajoute les features fondamentales et de groupe GICS (fundamentals.py) ; remplace celles déjà présentes."""
    df = df.drop(columns=FD.added_columns(df))
    groups = pd.read_parquet(SP.CAPS, columns=["symbol", "group"]).drop_duplicates("symbol", keep="last")
    return FD.add_group_features(FD.add_fundamentals(df), groups.set_index("symbol")["group"])


def build_features(base_only=False):
    df = F.main()
    df = df.drop(columns=[c for c in df.columns if F.is_pit_col(c)])
    report(df if base_only else augment(df))


def report(df):
    df.to_parquet(F.FEATURES, index=False)
    m = df[df["member"]]
    cov = m[[c for c in df.columns if c.startswith(("fq_", "fv_"))]].notna().mean()
    print(f"S&P 500 : {len(df)} lignes ({len(m)} membres), {df['symbol'].nunique()} titres, "
          f"{len(F.feature_columns(df))} features, membres par jour {m.groupby('date').size().median():.0f}", flush=True)
    if len(cov):
        print(f"fondamentaux : {len(cov)} features, couverture médiane {cov.median():.1%}, min {cov.min():.1%} "
              f"({cov.idxmin()})", flush=True)


if __name__ == "__main__":
    use_sp()
    DATA.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)
    if "--vendor" in sys.argv or not {"--features", "--augment"} & set(sys.argv):
        VS.main()
    if "--augment" in sys.argv:                    # ajoute fondamentaux + groupe au fichier existant
        report(augment(pd.read_parquet(F.FEATURES)))
    elif "--features" in sys.argv or "--vendor" not in sys.argv:
        build_features(base_only="--base" in sys.argv)
