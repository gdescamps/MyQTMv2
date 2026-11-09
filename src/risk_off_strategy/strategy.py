"""
Strategie deployee : trend-following long + vol-managed + garde-fous anticrise.

Cœur (x1, close-to-close), gap = prix/s - 1 :
    s          = SMA(prix, 250)
    vol        = Yang-Zhang(OHLC, 10j) annualisee  (integre les gaps overnight -> plus
                 precoce que le close-to-close ; fallback rvol20 close si pas d'OHLC)
    decay_down = clip(1 + gap / GAP_CUTOFF, 0, 1)          # SOUS la MA : coupe l'expo
                                                            # en s'enfoncant sous la MA
    above      = clip(ABOVE_CAP / vol, 0, 1)               # RISK-OFF PRECOCE : coupe deja
                                            # au-dessus de la MA quand la vol monte (=1 si OHLC absent)
    slope_up   = s > s(il y a SLOPE_K jours)               # FILTRE DE PENTE : le bras "above"
                                            # exige une MA montante (coupe le chop sur MA plate)
    alloc      = where(prix > s ET slope_up, above, BELOW_SCALE * clip(VOL_TARGET/vol, 0, 1) * decay_down)

Garde-fous macro (cash total, ignores si la donnee est absente) :
    NFCI    > NFCI_OFF   -> stress credit (capte 2008)
    IPC YoY > CPI_OFF %  -> inflation extreme (capte 2022)

exec_lag=1 est gere en aval : la valeur alloc[-1] calculee au close du soir
s'execute a l'ouverture du lendemain, donc pas de look-ahead. Yang-Zhang au jour t
n'utilise que O/H/L/C de t (et C_{t-1}) sur une fenetre glissante finissant a t :
causal, aucune fuite temporelle.

Historique/backtest depuis 2000 (le QQQ demarre le 2000-01-03).
"""
import numpy as np
import pandas as pd

ANN = 252

# ── Parametres de la strategie ────────────────────────────
SMA_LONG = 250       # MA de tendance
VOL_TARGET = 0.08    # V : cible de vol du bras vol-managed (sous la MA)
BELOW_SCALE = 0.8    # L2 : facteur d'expo sous la MA
GAP_CUTOFF = 0.15    # G : alloc -> 0 lorsque le prix est G en dessous de la MA
NFCI_OFF = 0.5       # coupe-circuit conditions financieres (NFCI)
CPI_OFF = 7.0        # coupe-circuit inflation (IPC YoY, %)
VOL_WINDOW = 10      # fenetre Yang-Zhang (10j ~= 20j close-to-close en bruit, 2x plus reactif)
ABOVE_CAP = 0.15     # risk-off precoce : vol annualisee au-dela de laquelle on coupe deja
                     #  au-dessus de la MA. Genou de la courbe etendue : sous 0.15 le Calmar
                     #  plafonne (0.62) et le maxDD ne s'ameliore plus (-17% -> -16%), on ne
                     #  fait plus que de-lever. Protection quasi-maximale.
# decay_up (trim de la sur-extension au-dessus de la MA : gap > +15% -> expo x0.4 a +27%) a ete
# RETIRE en 2026-11 : dans la strategie deployee x1.7 nette (bande 0.50/1.00 en expo, frais 0.5%),
# il est redondant avec le cap de vol et masque par la bande -- le retirer fait egal ou mieux sur
# les deux moities (Calmar H1/H2 0.23/0.85 -> 0.23/0.88, CAGR 15.9 -> 16.3%, maxDD -29.5% inchange ;
# idem a 0.1%/cote : 0.27/0.90 -> 0.27/0.94) et le laisser declencher des ordres sous la bande
# degrade (myfiles/pust17_simplify.py, pust17_selective_band.py). 3 parametres en moins.
SLOPE_K = 40         # filtre de pente : le bras "above" exige SMA250 > sa valeur d'il y a
                     #  SLOPE_K jours (sinon bras below). Coupe les regimes de chop -- prix
                     #  oscillant au-dessus d'une MA plate (2015-16, sommet dot-com) -- la ou
                     #  se logent les pires maxDD ; complementaire du risk-off precoce (crashs
                     #  rapides sous MA montante : 2018/2025). Plateau robuste k=30-50, valide
                     #  sur 2 moities + net d'execution (cf. myfiles/x2_calmar_finalists_test.py).
                     #  x1 : Calmar 0.62->0.70, maxDD -17%->-15% ; x2 : 0.66->0.72, -31.5%->-29% ;
                     #  CAGR inchange dans les deux cas.


def realized_vol(ret, w=20):
    """Vol realisee close-to-close annualisee (plancher a 1e-6). Fallback si pas d'OHLC."""
    return np.maximum(
        pd.Series(ret).rolling(w, min_periods=5).std().bfill().values * np.sqrt(ANN),
        1e-6,
    )


def yang_zhang_vol(open_, high, low, close, w=VOL_WINDOW):
    """Vol Yang-Zhang annualisee (causale) = overnight + k*open-close + (1-k)*Rogers-Satchell.

    Combine le gap overnight (O_t vs C_{t-1}), la variance open->close et le terme
    Rogers-Satchell (drift-independent, H/L/O/C). ~5x plus efficace que le
    close-to-close et surtout capte le risque de gap nocturne -- la ou naissent les
    krachs -- donc alerte plus tot. Plancher a 1e-6.
    """
    o, h, l, c = (pd.Series(np.asarray(x, float)) for x in (open_, high, low, close))
    lo = np.log(o / c.shift(1))               # overnight (O_t / C_{t-1})
    lc = np.log(c / o)                         # open -> close
    ho, ll = np.log(h / o), np.log(l / o)
    rs = ho * (ho - lc) + ll * (ll - lc)       # Rogers-Satchell
    k = 0.34 / (1.34 + (w + 1) / (w - 1))
    var = (lo.rolling(w, min_periods=5).var()
           + k * lc.rolling(w, min_periods=5).var()
           + (1 - k) * rs.rolling(w, min_periods=5).mean())
    vol = np.sqrt(np.clip(var, 0, None) * ANN)
    return np.maximum(pd.Series(vol).bfill().values, 1e-6)


def allocation_components(price, rv, above_cap=ABOVE_CAP, vol_cap=True):
    """Variables intermediaires de la formule (avant garde-fous macro), source unique
    pour compute_allocation ET le chart du backtest.

    rv : vol annualisee alignee (Yang-Zhang si OHLC, sinon close-to-close).
    vol_cap=False : pas d'OHLC -> comportement historique (bras haut = 1, sans cap de vol).
    Renvoie un dict d'arrays : s (SMA), gap, slope_up, arm_above (bras « au-dessus »
    actif), cap_vol = min(above_cap/vol, 1), above (= cap_vol, bras haut),
    vt = min(VOL_TARGET/vol, 1), decay_down, below (= BELOW_SCALE . vt . decay_down)."""
    p = np.asarray(price, dtype=float)
    rv = np.asarray(rv, dtype=float)
    s = pd.Series(p).rolling(SMA_LONG, min_periods=1).mean().values
    slope_up = np.ones(len(p), dtype=bool)                       # filtre de pente : MA montante
    if len(p) > SLOPE_K:
        slope_up[SLOPE_K:] = s[SLOPE_K:] > s[:-SLOPE_K]
    gap = p / s - 1.0
    decay_down = np.clip(1.0 + gap / GAP_CUTOFF, 0, 1)          # sous la MA : coupe en s'enfoncant
    if vol_cap:
        cap_vol = np.clip(above_cap / rv, 0, 1)
        above = cap_vol                                          # cap de vol (risk-off precoce)
    else:
        cap_vol = np.ones_like(p)
        above = np.ones_like(p)                                  # comportement historique
    vt = np.clip(VOL_TARGET / rv, 0, 1)
    below = BELOW_SCALE * vt * decay_down
    return dict(s=s, gap=gap, slope_up=slope_up, arm_above=(p > s) & slope_up, cap_vol=cap_vol,
                above=above, vt=vt, decay_down=decay_down, below=below)


def compute_allocation(price, nfci=None, cpi=None, high=None, low=None, open_=None,
                       above_cap=ABOVE_CAP):
    """Allocation continue dans [0, 1].

    price : array-like des cloture.
    high/low/open_ : OHLC alignes -> active la vol Yang-Zhang + le risk-off precoce
                     (cap vol au-dessus de la MA). Absents -> fallback close-to-close
                     et comportement historique (expo=1 au-dessus de la MA).
    nfci  : array aligne (deja decale du lag de publication) ou None -> arm ignore.
    cpi   : idem pour l'IPC YoY (%).
    """
    p = np.asarray(price, dtype=float)
    ohlc = high is not None and low is not None and open_ is not None
    if ohlc:
        rv = yang_zhang_vol(open_, high, low, p)                # vol precoce (gaps overnight)
    else:
        rv = realized_vol(pd.Series(p).pct_change().fillna(0).values)   # fallback close-to-close
    c = allocation_components(p, rv, above_cap=above_cap, vol_cap=ohlc)
    alloc = np.clip(np.where(c["arm_above"], c["above"], c["below"]), 0, 1)
    if nfci is not None:
        alloc = np.where(np.nan_to_num(np.asarray(nfci, float), nan=-9) > NFCI_OFF, 0.0, alloc)
    if cpi is not None:
        alloc = np.where(np.nan_to_num(np.asarray(cpi, float), nan=-9) > CPI_OFF, 0.0, alloc)
    return alloc


# ── Frais reels (backtest net) ────────────────────────────
TER_PUST = 0.0030     # frais courants PUST (Amundi PEA Nasdaq-100, x1)
TER_LQQ = 0.0060      # frais courants LQQ (Amundi Nasdaq-100 Daily 2x)
SWAP_SPREAD = 0.0040  # spread de financement du swap LQQ au-dela du taux court
SELL_FEE = 0.005      # frais de vente Bourso (0.5%)
BUY_FEE = 0.005       # frais d'achat Bourso (0.5%, constates le 2026-09-28 : 89.35 EUR sur 17 869 EUR de PUST)
# Bande de non-action ASYMETRIQUE (en expo x1 = alloc, multipliee par le levier) :
# on ne re-monte l'expo que si la cible s'ecarte de BUY_THR (achats 0.5%) et on
# ne la baisse que si elle s'ecarte de SELL_THR (vente 0.5% -> on ne DE-lève que par
# grands pas). Calibre NET DE FRAIS sur le backtest QQQ 2000-2026 (grille complete +
# validation 2 moities dans myfiles/asym_band_optimize.py) : (0.25 / 0.50) MAXIMISE le
# Sharpe ET le Calmar net et DOMINE l'ancien symetrique 0.20 sur tous les axes
#   x1 : CAGR 8.8->9.9%, maxDD -18.6->-17.6%, Sharpe 0.82->0.85, Calmar 0.48->0.57,
#        frais 0.72->0.32%/an, revisions 11->4/an, Sharpe pire-moitie 0.42->0.48.
#   x2 : CAGR 15.1->17.9%, maxDD -34.4->-33.3%, Calmar 0.44->0.54 (memes seuils, .levier).
# Deux enseignements : (1) elargir FORTEMENT les bandes coupe le whipsaw du bras vol-
# managed -> moins de ventes perdantes, donc + de rendement ET - de drawdown ; (2)
# l'asymetrie utile est MODEREE (vente = 2x l'achat), pas extreme -- suivre la montee
# trop finement sur-investit et se paye en ventes au repli. Au-dela (sell >= expo 1.0)
# la bande annule le risk-off (maxDD x2 -> -52%) : 0.50 reste loin de cette falaise.
BUY_THR_ALLOC = 0.25   # seuil d'ACHAT (montee d'expo) — gratuit
SELL_THR_ALLOC = 0.50  # seuil de VENTE (baisse d'expo) — limite les frais, laisse courir

# ── Strategie DEPLOYEE : PUST + LQQ, exposition plafonnee ──────────────────────
# Exposition cible E = min(LEVERAGE_MAX . alloc, E_MAX), realisee "drag-minimale" :
#   E <= 1 : PUST = E,     LQQ = 0,     cash = 1 - E
#   E >  1 : PUST = 2 - E, LQQ = E - 1, cash = 0     (LQQ ne porte que la part > 100%)
# vs l'ancien live "LQQ + cash" (poids w = E/2 en LQQ, reste en cash) : a exposition
# egale, la derive de l'expo entre deux rebalancements est ~2.7x plus lente (dE ~ 0.24 r
# contre 0.64 r a E = 1.6), le portage du levier est paye au minimum, et tout le cash
# travaille. Plafond E_MAX = 1.7 : sweep net 2000-2026 (myfiles/x2_lqq_levers_*.py,
# PUST+LQQ, bande 0.50/1.00 en expo) — 2.0 : CAGR 17.9% / maxDD -33% / Calmar 0.54 ;
# 1.8 : 17.5% / -30% / 0.58 ; 1.7 : 17.3% / -29% / Calmar 0.60, Sharpe 0.82, frais
# 0.42%/an, 3.8 revis/an ; 1.6 : 16.2% / -31% / 0.53. Moities H1/H2 : 0.30 / 0.93
# (2.0 : 0.26 / 0.81). Le plafond est un curseur risque/rendement (pas un gain gratuit) :
# ~-0.5 pt de CAGR contre ~4 pts de maxDD en moins et 15% de trades en moins.
LEVERAGE_MAX = 2.0     # levier de l'instrument x2 (LQQ) = expo max theorique 2 . alloc
E_MAX = 1.7            # plafond d'exposition deployee (PUST + LQQ)
# Seuils de bande en ESPACE EXPOSITION (= seuils alloc . levier, comme simulate_net)
BUY_THR_E = BUY_THR_ALLOC * LEVERAGE_MAX     # 0.50 : on remonte l'expo (achats gratuits)
SELL_THR_E = SELL_THR_ALLOC * LEVERAGE_MAX   # 1.00 : on la baisse (vente 0.5%)
# Saturation au plafond : le jour ou la cible PASSE a E_MAX (veille < E_MAX), on remonte a
# E_MAX des que l'ecart reel >= SAT_GAP_E, meme sous BUY_THR_E (bande normale ensuite).
# Backtest 2000-2026 x1.7 frais 0.5% (myfiles/pust17_saturate_cap.py) : 15.9%/an, maxDD
# -32.1%, Calmar 0.50, H1/H2 0.22/0.79 vs sans 16.3% / -29.5% / 0.55 / 0.23/0.88 —
# Testee puis ECARTEE (2026-11) : la strategie selectionnee reste x1.7 SANS saturation
# (la zone 1.2-1.7 sans achat amortit les chocs). Mecanisme conserve, desactive :
# SATURATE_CAP=False -> simulate_net(sat_gap=None) par defaut, signal saturate_cap=false.
# Variante etudiee : E_MAX 1.5 + saturation (myfiles/pust15_*.py) = 15.6% / -29.8% / 0.52.
SATURATE_CAP = False
SAT_GAP_E = 0.10       # seuil de la variante (utilise seulement si SATURATE_CAP / sat_gap passe)


def saturates_cap(prev_alloc, alloc, e_max=E_MAX, leverage=LEVERAGE_MAX):
    """True le jour ou la cible PASSE au plafond (veille sous e_max, jour = e_max) : le
    matin suivant, l'execution remonte a e_max si l'ecart >= SAT_GAP_E (cf. simulate_net)."""
    if not SATURATE_CAP:
        return False
    cap = min(e_max, 2.0) - 1e-9
    return bool(target_exposure(alloc, e_max, leverage) >= cap
                and target_exposure(prev_alloc, e_max, leverage) < cap)


def target_exposure(alloc, e_max=E_MAX, leverage=LEVERAGE_MAX):
    """Exposition cible deployee : min(levier . alloc, e_max), bornee a [0, 2]."""
    return float(np.clip(leverage * float(alloc), 0.0, min(e_max, 2.0)))


def composition(e):
    """Realisation drag-minimale d'une exposition e in [0, 2] -> poids (PUST, LQQ, cash)."""
    e = float(np.clip(e, 0.0, 2.0))
    if e <= 1.0:
        return e, 0.0, 1.0 - e
    return 2.0 - e, e - 1.0, 0.0


# ── Deploiement progressif d'un APPORT de capital (DCA pilote par le RSI) ───────
# Un apport detecte sur le PEA n'est pas investi d'un coup : un budget B (part de
# l'apport liberee, 0 au depart) grossit chaque semaine SI le RSI(14) du QQQ est sous
# DCA_RSI_GATE, d'une tranche DCA_BASE + DCA_SLOPE . (gate - RSI) (plus le RSI est
# bas, plus la tranche est grosse) ; la part non liberee est RESERVEE (exclue de
# l'equity que la strategie alloue). Filet : tout est libere apres DCA_MAX_WEEKS.
# Calibre sur 1300 apports simules 2001-2025 (myfiles/rsi_dca_entry_test.py, 1 an
# apres l'apport, x1 net) : vs entree immediate, pire 5% des drawdowns de 1ere annee
# -17.1% -> ~-14/-15%, aucune annee < -10%, pour ~-1.5 pt de rendement moyen de 1ere
# annee (CAGR 3 ans 11.6% -> 11.0%) : c'est une assurance contre une tres mauvaise
# premiere annee, pas un gain.
DCA_RSI_GATE = 50.0    # tranche uniquement si RSI(14) < gate
DCA_BASE = 0.20        # tranche de base (part de l'apport) par semaine eligible
DCA_SLOPE = 0.02       # + DCA_SLOPE par point de RSI sous la gate (RSI 30 -> 0.60)
DCA_MAX_WEEKS = 26     # filet : tout l'apport est libere au plus tard apres 26 semaines
DCA_WEEK_DAYS = 7      # cadence des tranches (jours calendaires)


def rsi_wilder(close, n=14):
    """RSI de Wilder (lissage exponentiel alpha=1/n), causal ; 50 avant warmup."""
    d = pd.Series(np.asarray(close, float)).diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rsi = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    rsi = rsi.where(~((dn == 0) & (up > 0)), 100.0)     # que des hausses -> 100
    return rsi.fillna(50).values


def dca_tranche(rsi, gate=DCA_RSI_GATE, base=DCA_BASE, slope=DCA_SLOPE):
    """Part de l'apport liberee cette semaine : 0 si RSI >= gate, sinon
    clip(base + slope . (gate - rsi), 0, 1). rsi=None (inconnu) -> tranche de base."""
    if rsi is None or not np.isfinite(rsi):
        return base
    if rsi >= gate:
        return 0.0
    return float(np.clip(base + slope * (gate - rsi), 0.0, 1.0))


def simulate_net(price, alloc, leverage=1, funding=None, cash_rate=0.0,
                 sell_thr=None, buy_thr=None, exec_lag=1, e_max=None,
                 return_expo=False, sat_gap=None):
    """Backtest NET DE FRAIS, execution discretisee a la Bourso.

    Modele des instruments (espace index QQQ, cf. convention du repo) :
      PUST : r = ret − TER_PUST/252
      LQQ  : r = 2·ret − (funding + SWAP_SPREAD + TER_LQQ)/252   (levier finance au taux court)
      cash : r = cash_rate/252   (0 par defaut = cash PEA non remunere ; €STR ≈ funding)

    Realisation drag-minimale de l'exposition cible E = leverage·alloc ∈ [0, 2] :
      E ≤ 1 : PUST=E,   LQQ=0,   cash=1−E     (aucun drag de levier sous 100%)
      E > 1 : PUST=2−E, LQQ=E−1, cash=0       (LQQ ne porte que la part >100%)

    Execution : bande de non-action ASYMETRIQUE. On rebalance si l'expo cible s'ecarte
    de l'effective d'au moins buy_thr (montee -> seuil fin) ou sell_thr (baisse -> seuil
    grossier), ou passage a/depuis le cash total. Frais Bourso de SELL_FEE (0.5%) sur le
    notionnel vendu et BUY_FEE (0.5%) sur le notionnel achete (tarif Decouverte constate
    en live le 2026-09-28 — les achats ne sont pas gratuits). exec_lag=1 (close J -> J+1).

    e_max : plafond d'exposition (None = 2.0) ; la strategie deployee = leverage=2,
    e_max=E_MAX (PUST + LQQ plafonne a 1.7).
    sat_gap : saturation au plafond (seulement si e_max est donne) — le jour ou la cible
    passe a e_max (veille < e_max), achat jusqu'a e_max si l'ecart >= sat_gap, meme sous
    buy_thr. None (defaut, strategie selectionnee) = desactive ; SAT_GAP_E = variante.

    funding requis si leverage>1 (financement LQQ) ; sinon renvoie None (le backtest
    retombe sur le brut). Renvoie (equity_full, fees_yr_%, revis_yr).
    return_expo=True : ajoute en 4e element l'exposition REELLE detenue chaque jour apres
    execution et seuillage par la bande (derive comprise entre deux rebalancements)."""
    p = np.asarray(price, float)
    ret = pd.Series(p).pct_change().fillna(0).values
    a = np.asarray(alloc, float)
    n = len(p)
    if leverage > 1 and funding is None:
        return None
    fund = np.zeros(n) if funding is None else np.asarray(funding, float)
    if sell_thr is None:
        sell_thr = SELL_THR_ALLOC * leverage       # 0.50 en alloc -> 1.00 en expo x2
    if buy_thr is None:
        buy_thr = BUY_THR_ALLOC * leverage         # 0.25 en alloc -> 0.50 en expo x2

    r_pust = ret - TER_PUST / ANN
    r_lqq = 2 * ret - (fund + SWAP_SPREAD + TER_LQQ) / ANN
    r_cash = cash_rate / ANN

    E = np.clip(leverage * a, 0.0, 2.0 if e_max is None else min(float(e_max), 2.0))
    Etgt = np.concatenate([np.zeros(exec_lag), E[:-exec_lag]]) if exec_lag else E
    cap = None if (e_max is None or sat_gap is None) else min(float(e_max), 2.0) - 1e-9
    vp = vl = 0.0
    vc = 1.0
    eq = np.empty(n)
    expo = np.empty(n)
    fees_frac = 0.0
    revis = 0
    for t in range(n):
        vp *= (1 + r_pust[t]); vl *= (1 + r_lqq[t]); vc *= (1 + r_cash)
        V = vp + vl + vc
        e_eff = (vp + 2 * vl) / V if V > 0 else 0.0
        et = Etgt[t]
        force = (et == 0.0 and e_eff > 1e-9) or (e_eff == 0.0 and et > 1e-9)
        thr = buy_thr if et > e_eff else sell_thr   # asymetrie : monte fin, descend grossier
        sat = (cap is not None and t > 0 and et >= cap and Etgt[t - 1] < cap
               and et - e_eff >= sat_gap)
        if abs(et - e_eff) >= thr or force or sat:
            wl = max(0.0, et - 1.0)
            wp = et if et <= 1.0 else 2.0 - et
            tp, tl = wp * V, wl * V
            fee = (SELL_FEE * (max(0.0, vp - tp) + max(0.0, vl - tl))    # ventes
                   + BUY_FEE * (max(0.0, tp - vp) + max(0.0, tl - vl)))  # achats
            fees_frac += fee / V
            V -= fee
            vp, vl, vc = wp * V, wl * V, max(0.0, 1.0 - et) * V
            revis += 1
        eq[t] = V
        expo[t] = (vp + 2 * vl) / V if V > 0 else 0.0
    yrs = n / ANN
    if return_expo:
        return eq, fees_frac / yrs * 100.0, revis / yrs, expo
    return eq, fees_frac / yrs * 100.0, revis / yrs


def simulate(price, alloc, exec_lag=1):
    """Backtest x1, close-to-close, sans frais. Renvoie un dict de metriques + equity.

    exec_lag=1 : la position du jour t est alloc[t-1] (decision au close t-1,
    executee au close t) -> aucun look-ahead.
    """
    p = np.asarray(price, dtype=float)
    ret = pd.Series(p).pct_change().fillna(0).values
    a = np.asarray(alloc, dtype=float)
    pos = np.zeros_like(a)
    pos[exec_lag:] = a[:-exec_lag] if exec_lag else a
    r = ret * pos
    eq = np.cumprod(1 + r)
    years = len(p) / ANN
    peak = np.maximum.accumulate(eq)
    dd = ((eq - peak) / peak).min()
    return {
        "equity": eq,
        "cagr": eq[-1] ** (1 / years) - 1,
        "sharpe": r.mean() / r.std() * np.sqrt(ANN) if r.std() > 0 else 0.0,
        "maxdd": float(dd),
        "calmar": (eq[-1] ** (1 / years) - 1) / abs(dd) if dd < 0 else np.inf,
        "time_in_market": float((pos > 0).mean()),
        "turnover": float(np.abs(np.diff(pos)).sum() / years),
    }
