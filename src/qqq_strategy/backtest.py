"""
Walk-forward XGBoost backtest with continuous allocation 0-150%.
"""

import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def walk_forward(X, y, feature_cols, min_train=504, step=21, temperature=3.0,
                 xgb_params=None):
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

    t = min_train
    while t < N:
        test_end = min(t + step, N)
        train_idx = np.arange(0, t)
        test_idx = np.arange(t, test_end)

        model = xgb.XGBClassifier(**xgb_params)
        model.fit(X[train_idx], y[train_idx], verbose=False)

        dtest = xgb.DMatrix(X[test_idx])
        raw_logits = model.get_booster().predict(dtest, output_margin=True)
        smooth_proba = sigmoid(raw_logits / temperature)

        wf_proba[test_idx] = smooth_proba
        wf_pred[test_idx] = (smooth_proba >= 0.5).astype(int)
        last_model = model
        t = test_end

    # Top features
    imp = last_model.feature_importances_
    top_idx = np.argsort(imp)[-20:][::-1]
    print(f"\nTop 20 features (last model):")
    for idx in top_idx:
        print(f"  {feature_cols[idx]:30s} {imp[idx]:.4f}")

    return wf_pred, wf_proba, last_model


def compute_equity(qqq_ret, pred, proba, max_leverage=1.5,
                   prob_cash=0.5, prob_full=0.85):
    N = len(qqq_ret)

    # Buy & Hold
    bh_eq = np.cumprod(1 + qqq_ret)

    # Binary x1
    bin_eq = np.ones(N)
    for i in range(1, N):
        if pred[i]:
            bin_eq[i] = bin_eq[i - 1] * (1 + qqq_ret[i])
        else:
            bin_eq[i] = bin_eq[i - 1]

    # Continuous allocation
    alloc = np.clip((proba - prob_cash) / (prob_full - prob_cash), 0, 1) * max_leverage
    cont_eq = np.ones(N)
    for i in range(1, N):
        cont_eq[i] = cont_eq[i - 1] * (1 + qqq_ret[i] * alloc[i])

    return bh_eq, bin_eq, cont_eq, alloc


def compute_metrics(eq, years):
    cagr = eq[-1] ** (1 / years) - 1
    dd = ((eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq)).min()
    return cagr, dd


def plot_results(wf_dates, bh_eq, bin_eq, cont_eq, alloc, max_leverage,
                 save_path=None):
    years = (wf_dates[-1] - wf_dates[0]).days / 365.25
    bh_cagr, bh_dd = compute_metrics(bh_eq, years)
    bin_cagr, bin_dd = compute_metrics(bin_eq, years)
    cont_cagr, cont_dd = compute_metrics(cont_eq, years)

    print(f"\n{'='*70}")
    print(f"Periode: {wf_dates[0].date()} -> {wf_dates[-1].date()} ({years:.1f} ans)")
    print(f"{'':35s} {'CAGR':>8s} {'Total':>8s} {'MaxDD':>8s}")
    print(f"{'QQQ Buy & Hold':35s} {bh_cagr*100:7.1f}% {bh_eq[-1]:7.1f}x {bh_dd*100:7.1f}%")
    print(f"{'XGBoost WF binary x1':35s} {bin_cagr*100:7.1f}% {bin_eq[-1]:7.1f}x {bin_dd*100:7.1f}%")
    print(f"{'XGBoost WF continuous 0-150%':35s} {cont_cagr*100:7.1f}% {cont_eq[-1]:7.1f}x {cont_dd*100:7.1f}%")
    print(f"Alloc mean: {alloc.mean()*100:.0f}%  median: {np.median(alloc)*100:.0f}%  "
          f"at 0%: {(alloc==0).sum()}j  at 150%: {(alloc>=max_leverage-0.01).sum()}j")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), height_ratios=[3, 1],
                                    sharex=True, gridspec_kw={"hspace": 0.08})

    ax1.semilogy(wf_dates, bh_eq,
                 label=f"QQQ Buy & Hold (CAGR {bh_cagr*100:.1f}%, DD {bh_dd*100:.1f}%)",
                 color="tab:blue", linewidth=1.5, alpha=0.7)
    ax1.semilogy(wf_dates, bin_eq,
                 label=f"XGBoost binary x1 (CAGR {bin_cagr*100:.1f}%, DD {bin_dd*100:.1f}%)",
                 color="tab:green", linewidth=1.2, alpha=0.5)
    ax1.semilogy(wf_dates, cont_eq,
                 label=f"XGBoost continu 0-{max_leverage*100:.0f}% (CAGR {cont_cagr*100:.1f}%, DD {cont_dd*100:.1f}%)",
                 color="tab:red", linewidth=2)

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
