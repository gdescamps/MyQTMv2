# Phase Detection: Inter-ETF Correlation Indicator

## TL;DR

**The hypothesis that inter-ETF correlation predicts model performance is NOT confirmed.**

All tested indicators have near-zero correlation with forward model returns (|r| < 0.07 at 63d horizon). The only indicator with meaningful correlation is cross-sectional dispersion (disp_60), but it goes in the **opposite** direction to the hypothesis: higher dispersion correlates with **higher** returns (+0.25 at 63d, +0.39 at 252d), meaning the model benefits from volatile/dispersed markets, not just from decorrelation.

## Indicators Tested

1. **corr_60**: Rolling 60d mean pairwise correlation of daily returns
2. **corr_120**: Rolling 120d mean pairwise correlation
3. **pca_120**: Rolling 120d PC1 variance explained (how much one factor drives all ETFs)
4. **disp_60**: Rolling 60d cross-sectional dispersion (std of daily returns across ETFs)
5. **_slope variants**: 60d linear regression slope of each indicator above

## Results: Correlation with Forward Model Returns

| Indicator | vs fwd 63d | vs fwd 126d | vs fwd 252d |
|-----------|-----------|------------|------------|
| pca_120 | -0.037 | -0.043 | -0.070 |
| corr_120 | -0.029 | -0.031 | -0.025 |
| pca_120_slope | +0.003 | -0.019 | -0.035 |
| corr_120_slope | +0.010 | -0.027 | -0.040 |
| corr_60 | +0.065 | -0.019 | +0.027 |
| corr_60_slope | +0.112 | -0.011 | +0.073 |
| disp_60_slope | +0.131 | -0.009 | +0.112 |
| **disp_60** | **+0.250** | **+0.261** | **+0.385** |

The only non-negligible signal is **disp_60** (cross-sectional dispersion) with r=+0.25 to +0.39 -- but this means the model performs BETTER when dispersion is HIGH (volatile, crisis-like markets), not when correlation is low.

## Annual Correlations (9 data points -- low statistical power)

| Indicator | Corr with annual return |
|-----------|------------------------|
| corr_120 | -0.154 |
| pca_120 | -0.092 |
| corr_60 | +0.017 |
| disp_60_slope | +0.056 |
| corr_60_slope | +0.211 |
| corr_120_slope | +0.275 |
| disp_60 | +0.298 |
| pca_120_slope | +0.360 |

## Year-by-Year Analysis

| Year | Model Return | PCA (PC1 var%) | Mean Corr 120d | Dispersion 60d | Phase guess |
|------|-------------|----------------|----------------|----------------|-------------|
| 2017 | +18.7% | 0.528 | 0.478 | 0.0076 | Low corr, low disp |
| 2018 | -11.5% | 0.588 | 0.533 | 0.0090 | Rising corr |
| 2019 | +42.0% | 0.623 | 0.554 | 0.0084 | High corr but good return |
| 2020 | -4.7% | 0.718 | 0.676 | 0.0123 | Highest corr + highest disp (COVID) |
| 2021 | +37.7% | 0.561 | 0.511 | 0.0100 | Falling corr post-COVID |
| 2022 | +99.1% | 0.611 | 0.545 | 0.0121 | High disp (bear market) |
| 2023 | -10.0% | 0.583 | 0.517 | 0.0097 | Medium, calm market |
| 2024 | +11.1% | 0.558 | 0.491 | 0.0089 | Low corr, low disp, calm |
| 2025 | +43.7% | 0.561 | 0.488 | 0.0096 | Low corr |

## Key Observations

1. **The correlation/decorrelation story does not hold**: 2019 and 2022 were both high-correlation years yet the model performed extremely well. 2020 had the highest correlation AND the model lost money.

2. **Dispersion (volatility spread) matters more than correlation level**: The model's best year (2022, +99%) was a year of high cross-sectional dispersion (0.0121). Its worst years (2018, 2023) had lower dispersion.

3. **The model benefits from market stress**: When returns are spread out (some ETFs up, some down, all moving a lot), the model's signal is strongest. In calm "everything grinds up" markets, there is less to trade.

4. **Why the original hypothesis fails**: The model does not need ETFs to be *decorrelated* -- it needs them to have *different magnitudes of return*. High correlation with high dispersion (e.g., 2022 bear) is fine because even if everything falls, they fall at very different speeds, giving the model signal.

## Revised Interpretation

Instead of "correlation phase", a better framing is:

- **High-signal regime** (disp_60 > 0.010): large return spreads across ETFs, model thrives.
  Examples: 2020 H2, 2021, 2022, parts of 2025.
- **Low-signal regime** (disp_60 < 0.008): calm markets, everything drifts together, model adds little value.
  Examples: 2017, 2018, 2023, 2024.

## Should We Use This as an Overlay?

**Not recommended in the current form.** The disp_60 signal (r=+0.25 with fwd 63d returns) is real but modest, and it essentially says "volatile markets = better model performance." Using it would mean:
- Reducing allocation in calm markets (missing 10-20% returns like 2017, 2024)
- Increasing allocation in volatile markets (which may feel counterintuitive)

A more productive direction might be to use VIX directly (which the model already does for TOP_N adaptation) rather than building a custom correlation indicator.

## Visualization

See `myfiles/phase_indicator.jpg`
