# Temporal Shift Robustness Test

Testing sensitivity to `MIN_TRAIN_ROWS` offset.
Each shift moves step boundaries by ~2-5 trading days.
All runs: 5 seeds, XGB+LGB ensemble, optimised LGB params.

## Results

| MIN_TRAIN | Sharpe | DD | Ann brut | Ann net | Rob Sharpe | Rob DD | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4931 | 1.46 | -30.0% | 34.8% | 26.1% | 1.33 | -30.8% | +8.0% | +26.8% | +27.8% | +84.1% | +11.7% | +7.6% | +105.0% | +33.0% |
| 4981 | 1.27 | -23.1% | 29.2% | 22.1% | 1.25 | -25.2% | +2.9% | +38.1% | +3.5% | +71.4% | +15.0% | +12.4% | +75.6% | +31.7% |
| 5031 | 1.73 | -22.4% | 43.4% | 33.6% | 1.65 | -22.3% | +14.0% | +32.1% | -3.7% | +107.4% | +18.8% | +22.9% | +76.9% | +90.8% |
| 5081 | 1.63 | -31.9% | 38.6% | 28.8% | 1.23 | -34.6% | +8.4% | +54.6% | +27.8% | +90.2% | +20.5% | +11.0% | +40.0% | +40.4% |
| 5131 | 1.12 | -24.8% | 25.1% | 18.7% | 1.18 | -24.6% | nan | +40.3% | -0.7% | +90.9% | -2.1% | +31.6% | +20.9% | +26.6% |

## Statistics

- Sharpe: mean=1.44, std=0.25, min=1.12, max=1.73
- Robustness Sharpe: mean=1.33, std=0.19

## Interpretation

The model shows **moderate sensitivity** to temporal shifts.

Check annual returns for year-level stability.
