"""
Strategie deployee : trend-following long + vol-managed + garde-fous anticrise.

Cœur (x1, close-to-close), gap = prix/s - 1 :
    s          = SMA(prix, 250)
    vol        = Yang-Zhang(OHLC, 10j) annualisee  (integre les gaps overnight -> plus
                 precoce que le close-to-close ; fallback rvol20 close si pas d'OHLC)
    decay_down = clip(1 + gap / GAP_CUTOFF, 0, 1)          # SOUS la MA : coupe l'expo
                                                            # en s'enfoncant sous la MA
    decay_up   = clip(1 - max(0, gap - GAP2_START)/GAP2_SPAN, DECAY2_FLOOR, 1)  # AU-DESSUS :
                                            # trim la sur-extension (reversion des extremes)
    above      = clip(ABOVE_CAP / vol, 0, 1) * decay_up    # RISK-OFF PRECOCE : coupe deja
                                            # au-dessus de la MA (vol + sur-extension ; =1 si OHLC absent)
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
                     #  fait plus que de-lever. Protection quasi-maximale. Avec decay_up :
                     #  CAGR 10.6% / Sharpe 0.95 / Calmar 0.62 / maxDD -17% vs -36% baseline.
GAP2_START = 0.15    # decay_up : gap (prix/s-1) au-dela duquel on trim l'expo AU-DESSUS de la MA
                     #  (sur-extension = 75e percentile du gap). Symetrique du decay_down.
GAP2_SPAN = 0.20     # decay_up : plage de rampe du trim (de 1 au plancher)
DECAY2_FLOOR = 0.4   # decay_up : plancher (on ne descend pas sous 40% de l'expo cappee)
                     #  Gain robuste 2 moities : Sharpe 0.88->0.94, Calmar x2 0.61->0.64, maxDD ~stable.
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
    s = pd.Series(p).rolling(SMA_LONG, min_periods=1).mean().values
    slope_up = np.ones(len(p), dtype=bool)                       # filtre de pente : MA montante
    if len(p) > SLOPE_K:
        slope_up[SLOPE_K:] = s[SLOPE_K:] > s[:-SLOPE_K]
    gap = p / s - 1.0
    decay_down = np.clip(1.0 + gap / GAP_CUTOFF, 0, 1)          # sous la MA : coupe en s'enfoncant
    decay_up = np.clip(1.0 - np.maximum(0.0, gap - GAP2_START) / GAP2_SPAN,
                       DECAY2_FLOOR, 1.0)                        # au-dessus : trim la sur-extension
    if high is not None and low is not None and open_ is not None:
        rv = yang_zhang_vol(open_, high, low, p)                # vol precoce (gaps overnight)
        above = np.clip(above_cap / rv, 0, 1) * decay_up        # cap-vol + trim sur-extension
    else:
        ret = pd.Series(p).pct_change().fillna(0).values
        rv = realized_vol(ret)                                  # fallback close-to-close
        above = np.ones_like(p)                                 # comportement historique
    below = BELOW_SCALE * np.clip(VOL_TARGET / rv, 0, 1) * decay_down
    alloc = np.clip(np.where((p > s) & slope_up, above, below), 0, 1)
    if nfci is not None:
        alloc = np.where(np.nan_to_num(np.asarray(nfci, float), nan=-9) > NFCI_OFF, 0.0, alloc)
    if cpi is not None:
        alloc = np.where(np.nan_to_num(np.asarray(cpi, float), nan=-9) > CPI_OFF, 0.0, alloc)
    return alloc


# ── Frais reels (backtest net) ────────────────────────────
TER_PUST = 0.0030     # frais courants PUST (Amundi PEA Nasdaq-100, x1)
TER_LQQ = 0.0060      # frais courants LQQ (Amundi Nasdaq-100 Daily 2x)
SWAP_SPREAD = 0.0040  # spread de financement du swap LQQ au-dela du taux court
SELL_FEE = 0.005      # frais de vente Bourso (0.5%) ; achats gratuits
SELL_THR_ALLOC = 0.20  # seuil de revente en alloc x1 (= expo/levier) -> limite les revisions


def simulate_net(price, alloc, leverage=1, funding=None, cash_rate=0.0,
                 sell_thr=None, exec_lag=1):
    """Backtest NET DE FRAIS, execution discretisee a la Bourso.

    Modele des instruments (espace index QQQ, cf. convention du repo) :
      PUST : r = ret − TER_PUST/252
      LQQ  : r = 2·ret − (funding + SWAP_SPREAD + TER_LQQ)/252   (levier finance au taux court)
      cash : r = cash_rate/252   (0 par defaut = cash PEA non remunere ; €STR ≈ funding)

    Realisation drag-minimale de l'exposition cible E = leverage·alloc ∈ [0, 2] :
      E ≤ 1 : PUST=E,   LQQ=0,   cash=1−E     (aucun drag de levier sous 100%)
      E > 1 : PUST=2−E, LQQ=E−1, cash=0       (LQQ ne porte que la part >100%)

    Execution : on ne rebalance que si |E_cible − E_effective| ≥ sell_thr (bande de
    non-action -> limite le nombre de revisions) ou passage a/depuis le cash total.
    Frais de 0.5% sur le notionnel VENDU seulement. exec_lag=1 (close J -> J+1).

    funding requis si leverage>1 (financement LQQ) ; sinon renvoie None (le backtest
    retombe sur le brut). Renvoie (equity_full, fees_yr_%, revis_yr)."""
    p = np.asarray(price, float)
    ret = pd.Series(p).pct_change().fillna(0).values
    a = np.asarray(alloc, float)
    n = len(p)
    if leverage > 1 and funding is None:
        return None
    fund = np.zeros(n) if funding is None else np.asarray(funding, float)
    if sell_thr is None:
        sell_thr = SELL_THR_ALLOC * leverage       # 0.20 en alloc -> 0.40 en expo x2

    r_pust = ret - TER_PUST / ANN
    r_lqq = 2 * ret - (fund + SWAP_SPREAD + TER_LQQ) / ANN
    r_cash = cash_rate / ANN

    E = np.clip(leverage * a, 0.0, 2.0)
    Etgt = np.concatenate([np.zeros(exec_lag), E[:-exec_lag]]) if exec_lag else E
    vp = vl = 0.0
    vc = 1.0
    eq = np.empty(n)
    fees_frac = 0.0
    revis = 0
    for t in range(n):
        vp *= (1 + r_pust[t]); vl *= (1 + r_lqq[t]); vc *= (1 + r_cash)
        V = vp + vl + vc
        e_eff = (vp + 2 * vl) / V if V > 0 else 0.0
        et = Etgt[t]
        force = (et == 0.0 and e_eff > 1e-9) or (e_eff == 0.0 and et > 1e-9)
        if abs(et - e_eff) >= sell_thr or force:
            wl = max(0.0, et - 1.0)
            wp = et if et <= 1.0 else 2.0 - et
            tp, tl = wp * V, wl * V
            fee = SELL_FEE * (max(0.0, vp - tp) + max(0.0, vl - tl))   # ventes seulement
            fees_frac += fee / V
            V -= fee
            vp, vl, vc = wp * V, wl * V, max(0.0, 1.0 - et) * V
            revis += 1
        eq[t] = V
    yrs = n / ANN
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
