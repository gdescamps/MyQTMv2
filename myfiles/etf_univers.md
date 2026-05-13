## 2. Univers ETF — Équivalents UCITS sur Euronext

Les ETF US (SPY, QQQ…) sont bloqués pour les particuliers européens (règlement PRIIPs/MiFID II).
Il faut utiliser les équivalents UCITS cotés en EUR sur Euronext.

### Equity géographique Boursorama & FMP

| Ticker  | Nom exact Boursorama                         | Env.   | 0%frais | FMP     | Perf 1 an | Perf 5 ans |
|---------|----------------------------------------------|--------|:-------:|:-------:|:---------:|:----------:|
| CSP1.PA | iShares Core S&P 500 ETF USD Acc             | PEA ✅ | ✅      | CSPX.AS | +21%      | +96%       |
| CNX1.PA | iShares NASDAQ 100 ETF USD Acc               | PEA ✅ | ✅      | QQQ ⚠️  | +39%      | +123%      |
| WPEA.PA | iShares MSCI World Swap PEA ETF              | PEA ✅ | ✅      | WPEA.PA | +20%      | +81% ⁽¹⁾  |
| IEMA.AS | iShares MSCI Emerging Markets UCITS ETF USD Acc | CTO | ✅      | IEMA.AS | +39%      | +49%       |
| CSKR.PA | iShares MSCI Korea ETF USD Dist              | CTO    | ✅      | EWY ⚠️  | +208%     | +102%      |
| ITWN.PA | iShares MSCI Taiwan ETF USD Dist             | CTO    | ✅      | EWT ⚠️  | +77%      | +65%       |
| IFFI.AS | iShares MSCI AC Far East ex-Japan ETF        | CTO    | ✅      | IFFI.AS | +63%      | +46%       |
| EXCH.AS | iShares MSCI EM ex-China ETF USD Acc         | CTO    | ✅      | EXCH.AS | +62%      | +72%       |
| SJPE.AS | iShares Core MSCI Japan IMI ETF EUR Hedged   | CTO    | ✅      | SJPE.AS | +43%      | +140%      |
| LTAM.AS | iShares MSCI EM Latin America ETF USD Dist   | CTO    | ✅      | ILF ⚠️  | +41%      | +23%       |
| IBZL.AS | iShares MSCI Brazil ETF USD Dist             | CTO    | ✅      | EWZ ⚠️  | +39%      | +4%        |
| IMEX.AS | iShares MSCI Mexico Capped ETF USD Acc       | CTO    | ✅      | EWW ⚠️  | +38%      | +74%       |
| ICAU.AS | iShares MSCI Canada ETF USD Acc              | CTO    | ✅      | EWC ⚠️  | +35%      | +61%       |
| ITKY.AS | iShares MSCI Turkey ETF USD Dist             | CTO    | ✅      | TUR ⚠️  | +31%      | +87%       |
| ISF.L   | iShares Core FTSE 100 ETF GBP Dist           | CTO    | ✅      | ISF.L   | +19%      | +46%       |
| FXC.AS  | iShares China Large Cap ETF USD Dist         | CTO    | ✅      | FXC.L ⚠️| +2%       | -12%       |

> ⁽¹⁾ 5 ans WPEA.PA incomplet (lancé ~2020) → valeur reprise de l'équivalent CTO IWDA.AS (même indice MSCI World).
> **CNX1** : ticker CNX1.PA non couvert dans FMP → proxy QQQ (même indice Nasdaq-100, corrélation >0.99).
> **IEMA.AS** : iShares MSCI EM, zéro frais, couvert directement dans FMP. Remplace AEEM.PA (Amundi).
> **FXC** : couvert dans FMP via FXC.L (version LSE, même fonds). Perf faible due à la Chine en 2020-2024.
> **WPEA.PA** : historique FMP limité (5Y=+31% car lancé ~2020) — données partielles.

### Secteurs US (iShares S&P 500 sectoriels) Boursorama & FMP

| Ticker  | Nom exact Boursorama                              | Env. | 0%frais | FMP    | Perf 1 an | Perf 5 ans |
|---------|---------------------------------------------------|------|:-------:|:------:|:---------:|:----------:|
| IUIT.AS | iShares S&P 500 Info Technology UCITS ETF USD Acc | CTO  | ❌ frais| XLK ⚠️ | +54%      | +167%      |
| IUES.AS | iShares S&P 500 Energy Sector UCITS ETF USD Acc   | CTO  | ❌ frais| XLE ⚠️ | +36%      | +119%      |
| IUII.AS | iShares S&P 500 Industrials Sector UCITS ETF Acc  | CTO  | ❌ frais| XLI ⚠️ | +24%      | +72%       |
| IUCD.AS | iShares S&P 500 Consumer Discret UCITS ETF USD Acc| CTO  | ❌ frais| XLY ⚠️ | +11%      | +42%       |
| IUHC.AS | iShares S&P 500 Health Care Sector UCITS ETF Acc  | CTO  | ❌ frais| XLV ⚠️ | +7%       | +20%       |
| IUCS.AS | iShares S&P 500 Consumer Staples UCITS ETF USD Acc| CTO  | ❌ frais| XLP ⚠️ | +4%       | +21%       |
| IUFS.AS | iShares S&P 500 Financials Sector UCITS ETF Acc   | CTO  | ❌ frais| XLF ⚠️ | +1%       | +41%       |

> Aucun ticker UCITS sectoriel iShares (.AS) n'est couvert dans FMP → proxies SPDR XL* (même indice S&P 500 sectoriel, corrélation >0.98).
> Aucun ETF sectoriel en zéro frais sur Boursorama — frais standards (~0.22% achat + 0.22% vente).

### Secteurs thématiques & IA — Boursorama & FMP

| Ticker  | Nom exact Boursorama                              | Thème          | Env.   | 0%frais | FMP      | Perf 1 an | Perf 5 ans |
|---------|---------------------------------------------------|----------------|--------|:-------:|:--------:|:---------:|:----------:|
| EXX1.DE | iShares EURO STOXX Banks 30-15ETF DE acc          | Banques EU     | PEA ✅ | ✅      | EXX1.DE  | +31%      | +174%      |
| EXV1.DE | iShares STOXX Europe 600 Tech (DE) acc            | Tech EU        | PEA ✅ | ✅      | EXV1.DE  | +30%      | +164%      |
| SEMI.AS | iShares MSCI Global Semiconductors ETF USD Acc    | Semi-conducteurs| CTO   | ✅      | SEMI.AS  | +167%     | +264%      |
| AINF.PA | iShares AI Infrastructure ETF USD Acc             | IA             | CTO    | ✅      | CHAT ⚠️  | +112%     | +225%      |
| IART.PA | iShares AI Innovation Active ETF USD Acc          | IA             | CTO    | ✅      | WTAI ⚠️  | +86%      | +62%       |
| ECAR.AS | iShares Electric Vehicle & Driving Tech ETF USD   | Véhicules élec.| CTO    | ✅      | DRIV ⚠️  | +77%      | +56%       |
| INRA.AS | iShares Global Clean Energy Transition ETF USD    | Énergie propre | CTO    | ✅      | INRA.AS  | +75%      | +42%       |
| CITY.AS | iShares Smart City Infra ETF USD Acc              | Infra / Smart  | CTO    | ✅      | CITY.AS  | +44%      | +60%       |
| IQQQ.DE | iShares Global Water ETF USD Acc                  | Eau            | CTO    | ✅      | IQQQ.DE  | +1%       | +26%       |

> ⚠️ = proxy US ou LSE (UCITS Euronext non couvert dans FMP).
> ETF IA iShares (AINF, IART) trop récents pour FMP → proxies : CHAT (Gen AI), WTAI (AI Innovation).

### Commodités & Alternatif — CTO uniquement

| Ticker  | Nom exact Boursorama                              | 0%frais | FMP      | Signal               | Perf 1 an | Perf 5 ans |
|---------|---------------------------------------------------|:-------:|:--------:|----------------------|:---------:|:----------:|
| IGLN.AS | iShares Physical Gold ETC                         | ❌ frais| GLD ⚠️   | Risk-off / inflation | +45%      | +154%      |
| SXRS.DE | iShares Diversified Commodity Swap (DE)           | ✅ PEA  | SXRS.DE  | Commodités larges    | +37%      | +82%       |
| RING    | iShares Gold Producers ETF USD Acc                | ✅      | RING ⚠️  | Or minier / levier or| +117%     | +174%      |
| IOGP.AS | iShares Oil & Gas Explr&Prod ETF USD Acc          | ✅      | IEO ⚠️   | Pétrole E&P          | +33%      | +122%      |

> **Or physique (IGLN)** : non couvert dans FMP → proxy GLD (SPDR Gold, corrélation >0.99).
> **SXRS.DE** : seule commodité zéro frais + éligible PEA, couverte directement dans FMP.
> **RING** : mines d'or = levier sur l'or (~1.5-2x), plus volatil que l'or physique.

### Obligations / Taux — CTO uniquement

| Ticker  | Nom exact Boursorama                              | 0%frais | FMP     | Signal               | Perf 1 an | Perf 5 ans |
|---------|---------------------------------------------------|:-------:|:-------:|----------------------|:---------:|:----------:|
| DTLA.AS | iShares USD Treasury Bond 20+yr UCITS ETF         | ❌ frais| TLT ⚠️  | Risk-off refuge taux | -1%       | -37%       |
| IBTA.AS | iShares USD Treasury Bond 7-10yr UCITS ETF        | ❌ frais| IEF ⚠️  | Taux moyen terme     | +1%       | -17%       |
| IHYU.AS | iShares USD High Yield Corp Bd UCITS ETF          | ❌ frais| HYG ⚠️  | Appétit au risque    | +1%       | -8%        |
| ITPS.AS | iShares USD TIPS UCITS ETF                        | ❌ frais| TIP ⚠️  | Hedge inflation      | +3%       | -13%       |

> **Tickers UCITS non couverts dans FMP** → proxies US (corrélation >0.98) : TLT, IEF, HYG, TIP.
> **Aucun ETF obligataire dans la liste zéro frais Boursorama** — tous à tarif standard.
> **Cycle Fed 2022-2023** : toutes en forte baisse. Rebond limité depuis; signal utile en regime risk-off.

### ETN Crypto (disponibles depuis mars 2025 sur Boursobank)

| Ticker   | Nom exact Boursorama                         | 0%frais | FMP      | Actif    | Perf 1 an | Perf 5 ans |
|----------|----------------------------------------------|:-------:|:--------:|----------|:---------:|:----------:|
| IBTC.AS  | iShares Physical Bitcoin ETP USD Acc         | ❌ frais| IBIT ⚠️  | Bitcoin  | +55%      | N/D        |

> **FMP couvre IBIT / ETHA** (US Bitcoin/ETH ETFs Jan 2024+) comme proxies de signal.
> **Tickers Boursorama** : préfixe `1rT` + ticker (ex : IBTC.AS → `1rTIBTC`).

**Note Boursomarkets :** achat >= 500EUR -> 0EUR de courtage. Vente -> ~0.22% (tarif standard).
Symbole Boursorama = préfixe `1rT` + ticker (ex : CSP1 -> `1rTCSP1`).
