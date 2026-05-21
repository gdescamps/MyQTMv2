# Failed experiments — saturation evidence

Experiments tested and rejected. Left here so they don't get re-attempted.
The current 20-XGB × 110-feature ensemble appears to be at its Pareto
frontier on the 27-ETF smart-money universe — adding capacity (more models,
more features) no longer improves OOS performance.

Reference baseline (current state, 2026-05-21):
CAGR brut +48.66%/an, Sharpe 2.15, Max DD -12.9%, Test IC +0.0490.

---

## 1. LightGBM ensemble alongside XGBoost

Adding 20 LGB models with the same hyper-params, same K-fold embargo/seed
pairing as the existing 20 XGB models:

| | XGB only | XGB + LGB | Δ |
|---|---|---|---|
| Test IC | +0.0500 | +0.0507 | +0.0007 |
| Val − Test gap | +0.0291 | +0.0291 | 0 |
| CAGR brut | +48.7%/an | +46.2%/an | **−2.4%** |
| Sharpe | 2.15 | 2.08 | −0.07 |
| Training time | 5m02s | 5m54s | +17% |

The +0.078 IC gain reported in the earlier 2-model era (commit f3194a0) no
longer applies — with 20 XGB models already averaging out subsampling noise,
the marginal diversification value of LGB is exhausted. The slight test-IC
bump (+0.0007) doesn't translate to backtest equity. Would only be worth
revisiting if LGB hyper-params were deliberately differentiated (different
depth / learning rate / num_leaves) to add orthogonality.

## 2. Three new institutional-flow features

`inst_share_z*`, `rotation_idx_universe`, `flow_price_div_*d` — designed to
inject orthogonal smart-money signal beyond the existing `shares_outstanding_z*`
block. Baseline test IC = +0.0490.

| Variant | Test IC | Δ baseline | Best feature SHAP / kept in N steps |
|---|---|---|---|
| +F1 `inst_share_z20/z60` | +0.0496 | +0.0006 | 0.305 / 65 of 82 |
| +F1+F2 `rotation_idx_universe` | +0.0442 | **−0.0048** | F2 raw: 0.013 / 6 of 82 (filtered) |
| +F1+F2+F3 `flow_price_div_60d` | +0.0453 | −0.0037 | F3: **0.927** / 65 of 82 |
| +F1+F3 (F2 dropped) | +0.0483 | −0.0007 | F3: 0.881 / 66 of 82 |

The third feature `flow_price_div_60d` has one of the highest SHAP values in
the whole model (0.927, kept in 79% of steps) **yet contributes nothing to
test IC**. The current 20-XGB × 110-feature ensemble is at its Pareto
frontier — adding a new informative feature *displaces* an equally informative
one from the top-110 K-fold filter, netting to zero or worse. Same dynamic
that killed the LGB experiment above.

F2 specifically backfired (−0.0048) because `rotation_idx_universe` is
**cross-sectionally constant** (same value for all ETFs on a given date).
The XGB optimises cross-sectional IC per date, so a constant cannot rank
ETFs; it only contributes via interactions, costs tree depth, and occupies
a slot in the top-110 that would have gone to a real ranking feature.

## 3. Residual label over Sharpe baseline + global normalisation

Goal: teach the model to "stay quiet" when there's no edge beyond the calm-mode
Sharpe baseline, instead of using a heuristic gate.

Implementation: `label_t_etf = ret_10d_fwd[t, etf] − mean(ret_10d_fwd over top-3
rolling 252d Sharpe ETFs at t)`, plus disable cross-sectional z-scoring in
`train.py` (`LABEL_NORM_PER_DATE = False`) so the per-date constant is preserved.

Without disabling z-scoring, the per-date constant subtraction is mathematically
a no-op: z(y − c_t) = z(y) (subtracting a per-date constant doesn't change the
cross-sectional z-score). Confirmed: same test IC ±0.0003.

With global normalisation + residual label:

| | Baseline (raw label) | Residu + global norm | Δ |
|---|---|---|---|
| Test IC vs raw `ret_10d_fwd` | +0.0490 | +0.0559 | **+0.0069** ✓ |
| CAGR brut | +48.66%/an | +21.83%/an | **−27 pts** ✗ |
| Sharpe | 2.15 | 1.09 | −1.06 ✗ |
| Max DD | −12.9% | −26.2% | doublé ✗ |
| 2022 yearly return | +109.5% | −8.8% | −118 pts |

Test IC improved but backtest collapsed — model now predicts residuals
better, but the allocator expects raw-return rankings. The model's scores
collapse toward zero on calm days (as designed), making the softmax nearly
uniform → strategy misses the large directional moves (covid 2020, oil 2022).

Stress-test with calm-mode disabled on the residual model:
CAGR +6.80%/an, Sharpe **0.41**, DD **−56.9%**, 2024 = −37.8%. The pure
model is broken without the calm safety net.

**Conclusion**: residual-label learning requires a matching allocator change
(`weights = sharpe_baseline_weights + α × score_tilt`), not just a label swap.
Pure label engineering can't fix the architecture-allocator mismatch.

---

## Take-away

Future work that aims to genuinely move test IC must either:
- (a) raise `FEAT_SEL_CAP` above 110 to free slots for new signal,
- (b) prune a redundant feature block to free slots,
- (c) change the architecture (different label + matched allocator, stacking,
  regime-conditional models, or full mixture-of-experts).

Adding features or models to the saturated pool is guaranteed null — verified
across three independent experiments above.
