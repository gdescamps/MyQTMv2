"""
Walk-forward XGBoost backtest with continuous allocation 0-150%.
"""

import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def select_features_stable(X, y, feature_cols, block_size=21, embargo_rows=5,
                           n_periods=3, mean_std_power=1.7, top_n=None,
                           xgb_params=None, verbose=True):
    """Select features with stable importance across interlaced periods.

    Splits data into n_periods interlaced blocks, trains on (n-1) periods,
    measures feature importance, keeps features with high mean/std^power.
    """
    if xgb_params is None:
        xgb_params = dict(
            n_estimators=300, max_depth=4, learning_rate=0.03,
            subsample=0.7, colsample_bytree=0.7, min_child_weight=20,
            reg_alpha=1.0, reg_lambda=5.0, gamma=1.0,
            random_state=42, eval_metric="logloss",
        )

    N = len(X)
    row_pos = np.arange(N)
    block_idx = row_pos // block_size
    block_id = block_idx % n_periods
    pos_in_block = row_pos % block_size
    not_embargoed = (pos_in_block >= embargo_rows) & (pos_in_block < block_size - embargo_rows)

    importances = {}
    for period in range(n_periods):
        train_mask = (block_id != period) & not_embargoed
        train_idx = np.where(train_mask)[0]

        model = xgb.XGBClassifier(**xgb_params)
        model.fit(X[train_idx], y[train_idx], verbose=False)
        importances[period] = model.feature_importances_

        top_idx = np.argsort(importances[period])[-5:][::-1]
        if verbose:
            print(f"  Period {period}: train={len(train_idx)} rows  "
                  f"top5={[feature_cols[i] for i in top_idx]}")

    # Stability: mean / std^power
    imp_arr = np.array([importances[p] for p in range(n_periods)])  # (n_periods, n_features)
    mean_imp = imp_arr.mean(axis=0)
    std_imp = imp_arr.std(axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        stability = np.where(std_imp > 0, mean_imp / (std_imp ** mean_std_power), 0)
    stability = np.where(mean_imp > 0, stability, 0)

    # Rank and select
    ranked_idx = np.argsort(stability)[::-1]
    n_nonzero = np.sum(stability > 0)
    if top_n is None:
        top_n = n_nonzero
    n_selected = min(top_n, n_nonzero)
    selected_idx = ranked_idx[:n_selected]

    if verbose:
        print(f"\n  Feature selection: {n_selected}/{len(feature_cols)} features kept "
              f"(mean/std^{mean_std_power})")
        for i, idx in enumerate(selected_idx[:20]):
            print(f"    {feature_cols[idx]:30s}  mean={mean_imp[idx]:.4f}  "
                  f"std={std_imp[idx]:.4f}  stab={stability[idx]:.1f}")
        if n_selected > 20:
            print(f"    ... ({n_selected - 20} more)")

    selected_names = [feature_cols[i] for i in selected_idx]
    return selected_idx, selected_names


def walk_forward(X, y, feature_cols, min_train=504, step=21, embargo=21,
                 temperature=3.0, xgb_params=None, feat_select=True,
                 feat_top_n=None, feat_power=1.7):
    if xgb_params is None:
        xgb_params = dict(
            n_estimators=300, max_depth=4, learning_rate=0.03,
            subsample=0.7, colsample_bytree=0.7, min_child_weight=20,
            reg_alpha=1.0, reg_lambda=5.0, gamma=1.0,
            random_state=42, eval_metric="logloss",
        )

    N = len(X)
    wf_pred = np.full(N, -1, dtype=int)
    wf_proba = np.full(N, np.nan)
    last_model = None
    last_feat_names = list(feature_cols)
    n_steps = 0

    t = min_train
    while t < N:
        test_end = min(t + step, N)
        train_end = max(t - embargo, 0)
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(t, test_end)

        if len(train_idx) < min_train:
            t = test_end
            continue

        # Per-step feature selection on train data only
        if feat_select and len(train_idx) >= min_train * 1.5:
            sel_idx, sel_names = select_features_stable(
                X[train_idx], y[train_idx], feature_cols,
                mean_std_power=feat_power, top_n=feat_top_n,
                xgb_params=xgb_params, verbose=False,
            )
            if len(sel_idx) >= 5:
                X_train = X[train_idx][:, sel_idx]
                X_test = X[test_idx][:, sel_idx]
                last_feat_names = sel_names
            else:
                X_train = X[train_idx]
                X_test = X[test_idx]
        else:
            X_train = X[train_idx]
            X_test = X[test_idx]

        model = xgb.XGBClassifier(**xgb_params)
        model.fit(X_train, y[train_idx], verbose=False)

        dtest = xgb.DMatrix(X_test)
        raw_logits = model.get_booster().predict(dtest, output_margin=True)
        smooth_proba = sigmoid(raw_logits / temperature)

        wf_proba[test_idx] = smooth_proba
        wf_pred[test_idx] = (smooth_proba >= 0.5).astype(int)
        last_model = model
        n_steps += 1

        if feat_select and n_steps % 50 == 1:
            n_feat = len(last_feat_names)
            print(f"  Step {n_steps}: t={t} train={len(train_idx)} "
                  f"features={n_feat}/{len(feature_cols)}")

        t = test_end

    if feat_select:
        print(f"\nWalk-forward: {n_steps} steps, last feature set: "
              f"{len(last_feat_names)}/{len(feature_cols)}")

    # Top features (last model)
    imp = last_model.feature_importances_
    top_idx = np.argsort(imp)[-20:][::-1]
    print(f"\nTop 20 features (last model):")
    for idx in top_idx:
        print(f"  {last_feat_names[idx]:30s} {imp[idx]:.4f}")

    return wf_pred, wf_proba, last_model


def compute_equity(qqq_ret, proba, max_leverage=1.5,
                   prob_cash=0.5, prob_full=0.85):
    N = len(qqq_ret)

    # Buy & Hold
    bh_eq = np.cumprod(1 + qqq_ret)

    # Continuous allocation
    alloc = np.clip((proba - prob_cash) / (prob_full - prob_cash), 0, 1) * max_leverage
    cont_eq = np.ones(N)
    for i in range(1, N):
        cont_eq[i] = cont_eq[i - 1] * (1 + qqq_ret[i] * alloc[i])

    return bh_eq, cont_eq, alloc


def compute_metrics(eq, years):
    cagr = eq[-1] ** (1 / years) - 1
    dd = ((eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq)).min()
    return cagr, dd


def plot_results(wf_dates, bh_eq, cont_eq, alloc, max_leverage,
                 save_path=None):
    years = (wf_dates[-1] - wf_dates[0]).days / 365.25
    bh_cagr, bh_dd = compute_metrics(bh_eq, years)
    cont_cagr, cont_dd = compute_metrics(cont_eq, years)

    print(f"\n{'='*70}")
    print(f"Periode: {wf_dates[0].date()} -> {wf_dates[-1].date()} ({years:.1f} ans)")
    print(f"{'':35s} {'CAGR':>8s} {'Total':>8s} {'MaxDD':>8s}")
    print(f"{'QQQ Buy & Hold':35s} {bh_cagr*100:7.1f}% {bh_eq[-1]:7.1f}x {bh_dd*100:7.1f}%")
    print(f"{'XGBoost WF continuous 0-150%':35s} {cont_cagr*100:7.1f}% {cont_eq[-1]:7.1f}x {cont_dd*100:7.1f}%")
    print(f"Alloc mean: {alloc.mean()*100:.0f}%  median: {np.median(alloc)*100:.0f}%  "
          f"at 0%: {(alloc==0).sum()}j  at 150%: {(alloc>=max_leverage-0.01).sum()}j")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), height_ratios=[3, 1],
                                    sharex=True, gridspec_kw={"hspace": 0.08})

    # Last 10 years metrics
    import pandas as pd
    cutoff_10y = wf_dates[-1] - pd.DateOffset(years=10)
    mask_10y = wf_dates >= cutoff_10y
    idx_10y = np.where(mask_10y)[0][0]
    bh_10y = bh_eq[idx_10y:] / bh_eq[idx_10y]
    cont_10y = cont_eq[idx_10y:] / cont_eq[idx_10y]
    years_10 = (wf_dates[-1] - wf_dates[idx_10y]).days / 365.25
    bh_cagr_10, bh_dd_10 = compute_metrics(bh_10y, years_10)
    cont_cagr_10, cont_dd_10 = compute_metrics(cont_10y, years_10)

    print(f"\nDernieres 10 ans ({wf_dates[idx_10y].date()} -> {wf_dates[-1].date()}):")
    print(f"{'QQQ Buy & Hold':35s} {bh_cagr_10*100:7.1f}% {bh_10y[-1]:7.1f}x {bh_dd_10*100:7.1f}%")
    print(f"{'XGBoost WF continuous 0-150%':35s} {cont_cagr_10*100:7.1f}% {cont_10y[-1]:7.1f}x {cont_dd_10*100:7.1f}%")

    ax1.semilogy(wf_dates, bh_eq,
                 label=f"QQQ Buy & Hold (CAGR {bh_cagr*100:.1f}%, DD {bh_dd*100:.1f}%)",
                 color="tab:blue", linewidth=1.5, alpha=0.7)
    ax1.semilogy(wf_dates, cont_eq,
                 label=f"XGBoost continu 0-{max_leverage*100:.0f}% (CAGR {cont_cagr*100:.1f}%, DD {cont_dd*100:.1f}%)",
                 color="tab:red", linewidth=2)

    # 10y vertical line + annotation
    ax1.axvline(wf_dates[idx_10y], color="gray", linestyle=":", alpha=0.5)
    y_mid = np.sqrt(cont_eq.max() * cont_eq.min())
    ax1.annotate(
        f"10 ans\nB&H {bh_cagr_10*100:.1f}%/an  DD {bh_dd_10*100:.0f}%\nXGB {cont_cagr_10*100:.1f}%/an  DD {cont_dd_10*100:.0f}%",
        xy=(wf_dates[idx_10y], y_mid), fontsize=9, color="gray",
        ha="right", va="center",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
    )

    ax1.set_ylabel("Equity (log scale)")
    ax1.set_title("XGBoost Walk-Forward: allocation continue 0-150% selon P(invested)")
    ax1.legend(loc="upper left", fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2.fill_between(wf_dates, 0, alloc * 100, alpha=0.4, color="tab:green", label="Allocation %")
    ax2.axhline(100, color="gray", linestyle="--", alpha=0.5, linewidth=0.8, label="100% (no leverage)")
    ax2.axhline(150, color="red", linestyle="--", alpha=0.5, linewidth=0.8, label="150% (max leverage)")
    ax2.set_ylabel("Allocation (%)")
    ax2.set_xlabel("Date")
    ax2.set_ylim(-5, 165)
    ax2.legend(loc="lower right", fontsize=9)
    ax2.grid(True, alpha=0.3)

    ax2.xaxis.set_major_locator(mdates.YearLocator(2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"\nSaved: {save_path}")
    plt.show()
