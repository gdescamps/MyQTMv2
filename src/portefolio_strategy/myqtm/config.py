"""Configuration de la méthode d'origine de ref/MyQTM (portage sans fuite sur data/eodhd).

Restaurée depuis le commit 73c6534 (avant le passage au modèle de classement long seulement, 5c5d4a8).
Méthode : label de tendance haussière / range / baissière (trends.py), classifieur XGBoost 3 classes
entraîné en walk-forward, seuils d'ouverture / fermeture long et short optimisés par CMA-ES en walk-forward,
moteur de trading à 12 positions (trade.py).

Différences voulues avec ref/MyQTM/src/config.py :
- univers = liste Nasdaq dynamique point-in-time (universe.py), pas TRADE_STOCKS ;
- sentiment = Diamegs/PIT-4B-FT-201912 (coupure 2019-12), donc tout commence au 2020-01-01 ;
- entraînement et seuils en walk-forward (pas d'intervalles A/B entrelacés sur toute la période).

Les étapes communes (univers, news, LLM) et leurs fichiers restent ceux de src/portefolio_strategy ;
seules les features, prédictions et sorties de cette méthode vont dans des sous-dossiers myqtm/.
"""
from ..config import *  # noqa: F401,F403
from ..config import DATA as _DATA, OUT as _OUT, WORKERS

DATA = _DATA / "myqtm"         # features.parquet, index.parquet, predictions*.parquet de cette méthode
OUT = _OUT / "myqtm"           # backtest.png, metrics.json, equity.csv, positions.csv, …

# Pas d'embargo fixe : train.py écarte les jours dont le label de tendance n'est pas définitif à la coupe.

# --- Features (comme ref/MyQTM) -------------------------------------------
TS_SIZE = 6
TREND = dict(ma_short=1, ma_mid=3, ma_long=10, min_days=14, avg_daily_percent_thr=0.15)

# --- XGBoost ---------------------------------------------------------------
XGB_PARAMS = dict(objective="multi:softprob", num_class=3, tree_method="hist", max_depth=7,
                  learning_rate=0.05, subsample=0.6, colsample_bytree=0.7, gamma=4, min_child_weight=5,
                  reg_alpha=0.4, reg_lambda=4, nthread=WORKERS, seed=42, verbosity=0)
XGB_ROUNDS = 600
XGB_PATIENCE = 40
MEAN_STD_POWER = 1.71
TOP_FEATURES = [60, 80]
INTERVAL_DAYS = 90             # blocs entrelacés A/B à l'intérieur de la fenêtre d'apprentissage
# F1 mesuré seulement sur le début des tendances : un jour haussier / baissier ne compte que s'il est dans
# les F1_TREND_HEAD premiers de son segment (les 80 % de fin sont exclus) ; les jours de range comptent tous.
# S'applique au F1 qui décide (arrêt anticipé, choix 60/80 features) et au diagnostic hors échantillon ;
# l'apprentissage utilise toujours toutes les lignes.
F1_TREND_HEAD = 0.2

# --- Trading (comme ref/MyQTM) --------------------------------------------
INITIAL_CAPITAL = 10000.0
MAX_POSITIONS = 12
FEE = 0.0005                   # par sens (ref/MyQTM : 0)
ALLOW_SHORT = False             # positions longues seulement (ref/MyQTM : long + short)
# Fermeture des positions longues : "ema" = règle experte causale, EMA(ma_mid) < EMA(ma_long) de TREND
# (EMA 3 < EMA 10) constaté à la clôture, exécuté à l'ouverture suivante (myfiles/myqtm_close_rule.py) ;
# "proba" = P(haussier) < long_close (ref/MyQTM).
LONG_CLOSE_RULE = "ema"
DEFAULT_PARAMS = dict(long_open=0.60, long_close=0.37, short_open=0.60, short_close=0.37, increase=0.70)
CMA_ITERS = 30
CMA_POP = 10
CMA_STD = 0.2
CMA_DROPOUT_ROUNDS = 3
CMA_DROPOUT = 5
MC_RUNS = 20                   # Monte-Carlo final (retrait aléatoire de CMA_DROPOUT titres)
PARAM_MIN_MONTHS = 12          # historique hors échantillon exigé avant d'optimiser les seuils
