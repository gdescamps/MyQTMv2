# Filtrage ETF — smart money disponible depuis 2015-2017

Critère : conserver les ETF tradables (Boursorama UCITS) pour lesquels il existe
un proxy iShares/SPDR avec historique de shares outstanding depuis ≤ 2017.

Le smart money vient des fichiers XLS iShares (US ou EU). Les iShares US publient
les shares outstanding depuis l'inception ; les iShares EU (.DE/.L/.AS) aussi.

## ✅ À GARDER (proxy avec smart money ≤ 2017)

| ETF tradable (Boursorama)               | Proxy   | Inception SO | Section    |
|-----------------------------------------|---------|--------------|------------|
| iShares MSCI Korea                      | EWY     | 2000         | geo        |
| iShares MSCI Global Semiconductors      | SOXX    | 2001         | thematic   |
| iShares MSCI Taiwan                     | EWT     | 2000         | geo        |
| iShares Gold Producers                  | RING    | 2012         | commodity  |
| iShares Global Clean Energy Transition  | ICLN    | 2008         | thematic   |
| iShares MSCI AC Far East ex-Japan       | AAXJ    | 2008         | geo        |
| iShares MSCI EM ex-China                | EMXC    | 2017         | geo        |
| iShares Oil & Gas Explr & Prod          | IEO     | 2006         | commodity  |
| iShares MSCI Brazil                     | EWZ     | 2000         | geo        |
| iShares EURO STOXX Banks 30-15          | EXX1.DE | ~2001 direct | thematic   |
| iShares MSCI EM Latin America           | ILF     | 2001         | geo        |
| iShares Core MSCI Japan IMI (EURH/USD)  | EWJ     | 1996         | geo        |
| iShares MSCI Japan (Hedged/Acc/Dist)    | EWJ     | 1996         | geo        |
| iShares MSCI EM (Acc/Dist)              | EEM     | 2003         | geo        |
| iShares Core MSCI EM IMI                | IEMG    | 2012         | geo        |
| iShares MSCI Turkey                     | TUR     | 2008         | geo        |
| iShares Automation & Robotics           | ROBO    | 2013         | thematic   |
| iShares Diversified Commodity Swap      | DBC     | 2006         | commodity  |
| iShares NASDAQ 100                      | QQQ     | 1999         | geo        |
| iShares MSCI Mexico Capped              | EWW     | 2000         | geo        |
| iShares MSCI Canada                     | EWC     | 2000         | geo        |
| iShares MSCI China A                    | ASHR    | 2013         | geo        |
| iShares Global Aerospace & Defence      | ITA     | 2006         | thematic   |
| iShares Core S&P 500 (Acc/Dist/EURH)    | IVV     | 2000         | geo        |
| iShares MSCI USA / USA Swap             | IVV     | 2000         | geo        |
| iShares MSCI North America              | IVV     | 2000         | geo        |
| iShares MSCI World (Core/Swap/Hedged/PEA)| URTH   | 2012         | geo        |
| iShares Core FTSE 100                   | ISF.L   | ~2000 direct | geo        |
| iShares MSCI USA SRI                    | SUSA    | 2005         | geo        |
| iShares Core MSCI Pacific ex-Japan      | EPP     | 2001         | geo        |
| iShares Core MSCI EMU                   | EZU     | 2000         | geo        |
| iShares Core MSCI Europe                | IEUR    | 2014         | geo        |
| iShares S&P 500 Equal Weight            | RSP     | 2003         | geo        |
| iShares EURO STOXX Select Div 30        | EXSG.DE | ~2009 direct | thematic   |
| iShares EURO STOXX Small                | DJSC.DE | ~2009 direct | geo        |
| iShares EURO STOXX Mid                  | DJMC.DE | ~2009 direct | geo        |

## ❌ À EXCLURE (pas de smart money ≤ 2017)

| ETF tradable                            | Raison                                  |
|-----------------------------------------|-----------------------------------------|
| iShares AI Infrastructure               | proxy CHAT trop récent (2023)           |
| iShares AI Innovation Active            | proxy WTAI trop récent (2020)           |
| iShares Electric Vehicle & Driving Tech | proxy DRIV inception 2018 (> 2017)      |
| iShares Nasdaq 100 Top 30               | pas de proxy (sous-ensemble)            |
| iShares S&P 500 Top 20                  | pas de proxy (sous-ensemble)            |
| iShares Smart City Infra                | pas de proxy, thème récent              |
| iShares Digital Entertainment & Education | pas de proxy, thème récent            |
| iShares Dow Jones Global Leaders Screen | pas de proxy clair                      |
| iShares Ageing Population               | pas de proxy, thème récent              |
| iShares MSCI EM SRI                     | pas de proxy SRI avec SO long           |
| iShares MSCI World SRI                  | pas de proxy SRI avec SO long           |
| iShares Euro Dividend                   | pas de proxy avec SO long fiable        |

## Résumé

- **36 ETF gardés** (doublons Acc/Dist comptés une fois ≈ 30 expositions distinctes)
- **12 ETF exclus** — thèmes IA/EV/niche trop récents ou sous-ensembles sans proxy
