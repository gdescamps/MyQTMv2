"""Configuration du portage de ref/MyQTM sur data/eodhd (recherche, hors chaîne live).

Différences voulues avec ref/MyQTM/src/config.py :
- univers = liste Nasdaq dynamique point-in-time (universe.py), pas TRADE_STOCKS ;
- sentiment = Diamegs/PIT-4B-FT-201912 (coupure 2019-12), donc tout commence au 2020-01-01 ;
- modèle de classement des titres (long seulement) en walk-forward, au lieu de seuils d'ouverture / fermeture.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EODHD = ROOT / "data" / "eodhd"                  # LECTURE SEULE
DATA = ROOT / "data" / "portefolio_strategy"     # structures dérivées
OUT = ROOT / "outputs" / "portefolio_strategy"   # sorties

# Traitements LLM du corpus de news : un dossier par modèle, mêmes noms de fichiers et mêmes clés (day, symbol).
#   llm/haiku/   digests.jsonl (condensés company / competitors / global), timing.json
#   llm/pit/     sentiment.parquet, sentiment_emb/part_NNNNN.{npy,parquet}, timing.json
#   llm/qwen25/  sentiment.parquet, sentiment_emb/…, categories.parquet (types de news), timing.json
LLM = DATA / "llm"
HAIKU_DIR, PIT_DIR, QWEN_DIR = LLM / "haiku", LLM / "pit", LLM / "qwen25"

WORKERS = 5                    # cœurs max pour les traitements par lot (gb10)

# --- Dates -----------------------------------------------------------------
PRICE_START = "2015-01-01"     # historique de cours pour les min/max glissants et l'EMA 200
START_DATE = "2020-01-01"      # début des news notées (coupure du LLM PIT = 2019-12)
FIRST_OOS_DATE = "2021-01-01"  # première date prédite hors échantillon (1 an d'apprentissage)
END_DATE = None                # None = dernière date de cours disponible
RETRAIN_MONTHS = 3             # ré-entraînement walk-forward tous les trimestres
# Pas d'embargo fixe : train.py écarte les jours dont le label de tendance n'est pas définitif à la coupe.

# --- Univers Nasdaq dynamique ---------------------------------------------
UNIVERSE_FIRST_YEAR = 2019     # première reconstitution : classement de fin novembre 2018
UNIVERSE_SIZE = 100
UNIVERSE_ENTER_RANK = 75       # règle NDX : entre d'office si rang <= 75
UNIVERSE_KEEP_RANK = 100       # un membre reste s'il est encore dans les 100
UNIVERSE_MIN_DOLLAR_VOL = 5e6  # médiane 60 j du volume en dollars (écarte les capitalisations aberrantes)
UNIVERSE_MIN_TURNOVER = 0.08  # rotation annuelle minimale (volume $ × 252 / capitalisation)
SHARES_LAG_DAYS = 45           # délai de publication du nombre d'actions après la fin de trimestre
EXCLUDED_SECTORS = {"Financials", "Financial Services"}   # le Nasdaq-100 exclut les financières

# --- News / LLM ------------------------------------------------------------
NEWS_CUTOFF_HOUR_UTC = 20      # un article publié après 20:00 UTC compte pour le lendemain
NEWS_MAX_SYMBOLS = 8           # au-delà : article de marché généraliste, écarté
NEWS_MIN_CHARS = 200
NEWS_MAX_ARTICLES = 6          # par (titre, jour), les plus ciblés d'abord
NEWS_MAX_CHARS = 3000          # par article envoyé à Haiku
NEWS_WARMUP_DAYS = 45          # news collectées avant l'entrée dans l'univers (série de 10 jours de news)
HAIKU_MODEL = "claude-haiku-5-5"
HAIKU_MAX_TOKENS = 600
HAIKU_PRICE_IN = 0.10          # $/MTok, tarif de l'API Anthropic (clé ANTHROPIC_API)
HAIKU_PRICE_OUT = 0.50
PIT_REPO = "Diamegs/PIT-4B-FT-201912"
PIT_REVISION = "dd482b6a647f86b963626355a8e78d5588f63803"
PIT_CTX = 2048
QWEN_REPO = "Qwen/Qwen2.5-7B-Instruct"
QWEN_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
# Types de news classés par Qwen (une question fermée par type, sur la rubrique « company » du condensé).
NEWS_TYPES = {"earnings": "the financial results or earnings of the company",
              "analyst": "analyst ratings, price targets or estimates",
              "legal": "a lawsuit, an investigation or a legal issue",
              "mna": "a merger or an acquisition",
              "capital": "a dividend, a share buyback or a stock offering",
              "product": "a new product, a contract or a partnership",
              "management": "a change of executives or layoffs",
              "macro": "the economy, interest rates or government policy"}
SENT_CATEGORIES = ("company", "competitors", "global")
NEWS_TS_SIZE = 10

# --- Modèle de classement -------------------------------------------------
# Cible : rang du jour, parmi les membres, du rendement futur (ouverture J+1 → ouverture J+1+h).
HORIZONS = (5, 21, 63)         # jours de bourse ; un modèle par horizon
XGB_PARAMS = dict(objective="reg:squarederror", tree_method="hist", max_depth=4, learning_rate=0.03,
                  subsample=0.7, colsample_bytree=0.5, min_child_weight=100, reg_lambda=10,
                  max_bin=64, nthread=WORKERS, seed=42, verbosity=0, device="cpu")
XGB_ROUNDS = 300               # fixe : pas d'arrêt anticipé (il faudrait une validation dans le futur)

# --- Portefeuille (positions longues seulement) ---------------------------
INITIAL_CAPITAL = 10000.0
FEE = 0.001                    # frais par sens, sur le montant échangé à chaque rééquilibrage (0,1 %)
FEE_SENSITIVITY = (0.0, 0.0005, 0.001, 0.0025, 0.005)
GRID_EVERY = (5, 10, 21, 42, 63)   # fréquence de rééquilibrage (jours de bourse)
GRID_TOP = (5, 10, 15, 20)         # nombre de lignes du portefeuille
GRID_BUFFER = (1.0, 2.0)           # un titre détenu est gardé tant que son rang <= n_top × buffer
GRID_WEIGHTING = ("equal",)        # poids égaux ("cap" = proportionnels à la capitalisation, disponible)
MAX_OFFSETS = 4                    # portefeuilles décalés (jour de départ) moyennés par configuration
DEFAULT_CONFIG = dict(horizon=21, every=21, n_top=10, buffer=1.0, weighting="equal")
PARAM_MIN_MONTHS = 12          # historique hors échantillon exigé avant de choisir la configuration
