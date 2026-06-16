"""
Walk-forward XGBoost backtest with continuous allocation 0-150%.
"""

import hashlib
import json
import os
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "outputs" / "risk_off_strategy"


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

        train_classes = np.unique(y[train_idx])
        if len(train_classes) < 2:
            importances[period] = np.zeros(X.shape[1])
            if verbose:
                print(f"  Period {period}: train={len(train_idx)} rows  SKIPPED (single class)")
            continue

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


def _config_hash(feature_cols, xgb_params, feat_select, feat_power, feat_top_n,
                 min_train, step, embargo, temperature, X=None):
    """Hash of walk-forward config + data fingerprint. Stable across data updates."""
    h = hashlib.sha256()
    h.update(json.dumps(sorted(feature_cols)).encode())
    h.update(json.dumps(xgb_params, sort_keys=True).encode())
    h.update(f"{feat_select}_{feat_power}_{feat_top_n}".encode())
    h.update(f"{min_train}_{step}_{embargo}_{temperature}".encode())
    if X is not None:
        # Fingerprint: first row + last row + shape to distinguish data sources
        h.update(f"{X.shape}".encode())
        h.update(X[0].tobytes())
        h.update(X[min(100, len(X)-1)].tobytes())
    return h.hexdigest()[:16]


def _detect_device():
    """Try GPU, fallback to CPU. Respects XGBOOST_DEVICE env var."""
    override = os.environ.get("XGBOOST_DEVICE")
    if override:
        return override
    try:
        m = xgb.XGBClassifier(device="cuda", n_estimators=1, verbosity=0)
        m.fit(np.zeros((10, 2)), np.zeros(10, dtype=int), verbose=False)
        return "cuda"
    except Exception:
        return "cpu"


def walk_forward(X, y, feature_cols, min_train=504, step=21, embargo=21,
                 temperature=3.0, xgb_params=None, feat_select=True,
                 feat_top_n=None, feat_power=1.7, use_cache=True):
    if xgb_params is None:
        device = _detect_device()
        print(f"XGBoost device: {device}")
        xgb_params = dict(
            n_estimators=300, max_depth=4, learning_rate=0.03,
            subsample=0.7, colsample_bytree=0.7, min_child_weight=20,
            reg_alpha=1.0, reg_lambda=5.0, gamma=1.0,
            random_state=42, eval_metric="logloss",
            device=device,
        )

    N = len(X)
    wf_pred = np.full(N, -1, dtype=int)
    wf_proba = np.full(N, np.nan)
    last_model = None
    last_feat_names = list(feature_cols)
    resume_t = min_train  # where to start computing

    # ── Incremental cache: load previous results and resume ──
    cache_path = None
    if use_cache:
        cfg_hash = _config_hash(feature_cols, xgb_params, feat_select,
                                feat_power, feat_top_n, min_train, step,
                                embargo, temperature, X=X)
        cache_path = CACHE_DIR / f"wf_incr_{cfg_hash}.npz"
        if cache_path.exists():
            cached = np.load(cache_path, allow_pickle=True)
            cached_N = int(cached["N"])
            cached_pred = cached["wf_pred"]
            cached_proba = cached["wf_proba"]
            last_feat_names = list(cached["feat_names"])

            if cached_N == N:
                # Data unchanged → full cache hit
                print(f"Walk-forward loaded from cache ({cache_path.name}, {N} rows)")
                imp = cached["importances"]
                top_idx = np.argsort(imp)[-20:][::-1]
                print(f"\nTop 20 features (cached):")
                for idx in top_idx:
                    print(f"  {last_feat_names[idx]:30s} {imp[idx]:.4f}")
                return cached_pred, cached_proba, None

            elif cached_N < N:
                # Data grew → reuse old predictions, resume from last computed step
                wf_pred[:cached_N] = cached_pred[:cached_N]
                wf_proba[:cached_N] = cached_proba[:cached_N]
                # Find resume point: last step boundary that was computed
                resume_t = cached_N  # start computing from where old data ended
                # Align to step boundary
                t = min_train
                while t + step <= cached_N:
                    t += step
                resume_t = t
                n_old = (cached_N - min_train) // step
                n_new = (N - cached_N + step - 1) // step
                print(f"Incremental cache: {cached_N}→{N} rows (+{N - cached_N}), "
                      f"reusing {n_old} steps, computing ~{n_new} new steps")
            else:
                # Data shrank (shouldn't happen) → recompute all
                print(f"Cache data size mismatch ({cached_N}>{N}), recomputing...")

    n_steps = 0
    n_cached = 0

    t = min_train
    while t < N:
        test_end = min(t + step, N)
        train_end = max(t - embargo, 0)
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(t, test_end)

        if len(train_idx) < min_train:
            t = test_end
            continue

        # Skip steps already in cache
        if t < resume_t and not np.isnan(wf_proba[t]):
            n_cached += 1
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

        # Skip if only one class in training data
        train_classes = np.unique(y[train_idx])
        if len(train_classes) < 2:
            wf_proba[test_idx] = 1.0 if train_classes[0] == 1 else 0.0
            wf_pred[test_idx] = train_classes[0]
            n_steps += 1
            t = test_end
            continue

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

    total_steps = n_cached + n_steps
    if feat_select:
        print(f"\nWalk-forward: {total_steps} steps total "
              f"({n_cached} cached, {n_steps} computed), "
              f"last feature set: {len(last_feat_names)}/{len(feature_cols)}")

    # Top features (last model or cached)
    if last_model is not None:
        imp = last_model.feature_importances_
    elif cache_path and cache_path.exists():
        imp = np.load(cache_path, allow_pickle=True)["importances"]
    else:
        imp = np.zeros(len(last_feat_names))

    top_idx = np.argsort(imp)[-20:][::-1]
    print(f"\nTop 20 features:")
    for idx in top_idx:
        print(f"  {last_feat_names[idx]:30s} {imp[idx]:.4f}")

    # Save incremental cache
    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, wf_pred=wf_pred, wf_proba=wf_proba,
                 feat_names=np.array(last_feat_names),
                 importances=imp, N=N)
        print(f"Cache saved: {cache_path.name} ({N} rows)")

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


def simulate_with_fees(qqq_ret, wf_prob, max_lev, prob_cash=0.5, prob_full=0.85,
                       fee_sell=0.005, sell_step=0.20,
                       ter_1x=0.0023/252, ter_2x=0.0060/252):
    """Simulate equity with Boursorama PEA fees (0% buy, 0.5% sell) and PUST+LQQ TER."""
    alloc_target = np.clip((wf_prob - prob_cash) / (prob_full - prob_cash), 0, 1) * max_lev
    N = len(qqq_ret)
    INIT = 100_000
    equity, actual_alloc, sell_count = INIT, 0.0, 0
    eq_curve, alloc_curve = np.zeros(N), np.zeros(N)

    for i in range(N):
        tgt = alloc_target[i]
        delta = tgt - actual_alloc
        if delta > 0.001:
            actual_alloc = tgt
        elif delta < -0.001:
            if tgt < 0.01:
                equity -= actual_alloc * equity * fee_sell
                sell_count += 1
                actual_alloc = 0.0
            elif actual_alloc - tgt >= sell_step:
                equity -= (actual_alloc - tgt) * equity * fee_sell
                sell_count += 1
                actual_alloc = tgt

        if actual_alloc <= 1.0:
            dr = qqq_ret[i] * actual_alloc
            dt = ter_1x * actual_alloc
        else:
            lf = actual_alloc - 1.0
            pf = 1.0 - lf
            dr = qqq_ret[i] * pf + 2 * qqq_ret[i] * lf
            dt = ter_1x * pf + ter_2x * lf

        equity *= (1 + dr - dt)
        eq_curve[i] = equity
        alloc_curve[i] = actual_alloc

    return eq_curve / INIT, alloc_curve, sell_count


def plot_results(wf_dates, qqq_ret, wf_prob, prob_cash=0.5, prob_full=0.85,
                 save_path=None, ticker="QQQ", leverages=None, oracle_labels=None,
                 panx_ret=None):
    import pandas as pd

    if leverages is None:
        leverages = [1.0, 1.5, 1.75, 2.0]

    years = (wf_dates[-1] - wf_dates[0]).days / 365.25
    N = len(qqq_ret)
    bh_eq = np.cumprod(1 + qqq_ret)
    bh_cagr, bh_dd = compute_metrics(bh_eq, years)
    colors = {1.0: "tab:orange", 1.5: "tab:red", 1.75: "crimson", 2.0: "darkred"}

    idx_10y = np.where(wf_dates >= wf_dates[-1] - pd.DateOffset(years=10))[0][0]
    y10 = (wf_dates[-1] - wf_dates[idx_10y]).days / 365.25
    bh_c10, bh_d10 = compute_metrics(bh_eq[idx_10y:] / bh_eq[idx_10y], y10)

    results = {}
    for lev in leverages:
        eq_n, al_c, sc = simulate_with_fees(qqq_ret, wf_prob, lev, prob_cash, prob_full)
        n_cagr, n_dd = compute_metrics(eq_n, years)
        n_c10, n_d10 = compute_metrics(eq_n[idx_10y:] / eq_n[idx_10y], y10)
        results[lev] = dict(eq_n=eq_n, al_c=al_c, sc=sc,
                            n_cagr=n_cagr, n_dd=n_dd, n_c10=n_c10, n_d10=n_d10)

    # Oracle (perfect label) equity
    oracle = None
    if oracle_labels is not None:
        oracle_eq = np.ones(N)
        for i in range(1, N):
            oracle_eq[i] = oracle_eq[i - 1] * (1 + qqq_ret[i] * oracle_labels[i])
        oracle_cagr, oracle_dd = compute_metrics(oracle_eq, years)
        oracle_c10, oracle_d10 = compute_metrics(
            oracle_eq[idx_10y:] / oracle_eq[idx_10y], y10)
        oracle = dict(eq=oracle_eq, cagr=oracle_cagr, dd=oracle_dd,
                      c10=oracle_c10, d10=oracle_d10)

    # Print summary
    print(f"\n{'='*70}")
    print(f"Periode: {wf_dates[0].date()} -> {wf_dates[-1].date()} ({years:.1f} ans)")
    print(f"{'':35s} {'CAGR':>8s} {'Total':>8s} {'MaxDD':>8s}")
    print(f"{ticker + ' Buy & Hold':35s} {bh_cagr*100:7.1f}% {bh_eq[-1]:7.1f}x {bh_dd*100:7.1f}%")
    if oracle:
        print(f"{'Oracle (perfect label)':35s} {oracle['cagr']*100:7.1f}% "
              f"{oracle['eq'][-1]:7.1f}x {oracle['dd']*100:7.1f}%")
    for lev in leverages:
        r = results[lev]
        lbl = f"XGB x{lev:.1f} net Bourso"
        print(f"{lbl:35s} {r['n_cagr']*100:7.1f}% "
              f"{r['eq_n'][-1]:7.1f}x {r['n_dd']*100:7.1f}%")
    print(f"\nDernieres 10 ans ({wf_dates[idx_10y].date()} -> {wf_dates[-1].date()}):")
    print(f"{ticker + ' Buy & Hold':35s} {bh_c10*100:7.1f}%         {bh_d10*100:7.1f}%")
    if oracle:
        print(f"{'Oracle (perfect label)':35s} {oracle['c10']*100:7.1f}%         {oracle['d10']*100:7.1f}%")
    for lev in leverages:
        r = results[lev]
        lbl = f"XGB x{lev:.1f} net Bourso"
        print(f"{lbl:35s} {r['n_c10']*100:7.1f}%         {r['n_d10']*100:7.1f}%")

    # ── PANX execution (CC signal → PANX open) ──
    panx_results = None
    if panx_ret is not None:
        # Find first date with real PANX data (non-zero return after first few days)
        nonzero = np.where(panx_ret != 0)[0]
        if len(nonzero) > 0:
            panx_start_idx = max(0, nonzero[0] - 1)
            # Run simulation only on the PANX-available slice
            panx_ret_slice = panx_ret[panx_start_idx:]
            prob_slice = wf_prob[panx_start_idx:]
            panx_eq_s, panx_alloc_s, _ = simulate_with_fees(
                panx_ret_slice, prob_slice, 1.0, prob_cash, prob_full)
            panx_dates_s = wf_dates[panx_start_idx:]
            panx_years = (panx_dates_s[-1] - panx_dates_s[0]).days / 365.25
            panx_cagr, panx_dd = compute_metrics(panx_eq_s, panx_years)
            panx_results = dict(eq=panx_eq_s, start_idx=panx_start_idx,
                                dates=panx_dates_s, cagr=panx_cagr, dd=panx_dd)
            print(f"\n{'PEA: CC signal -> PANX open x1.0':35s} {panx_cagr*100:7.1f}%         {panx_dd*100:7.1f}%"
                  f"  ({panx_dates_s[0].date()} -> {panx_dates_s[-1].date()})")

    # ── Load PE daily ──
    try:
        from src.download_pe_qqq_top5 import load_pe_daily
        pe_daily = load_pe_daily()
    except Exception:
        pe_daily = None

    # ── Plot ──
    has_pe = pe_daily is not None and len(pe_daily) > 0
    n_rows = 3 if has_pe else 2
    h_ratios = [3, 1, 1] if has_pe else [3, 1]
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 11 if has_pe else 9),
                             height_ratios=h_ratios, sharex=True)
    ax1 = axes[0]
    ax2 = axes[1]
    ax_pe = axes[2] if has_pe else None

    # Equity curves
    ax1.semilogy(wf_dates, bh_eq,
                 label=f"{ticker} Buy & Hold ({bh_cagr*100:.1f}%, DD {bh_dd*100:.1f}%)",
                 color="tab:blue", linewidth=1.5, alpha=0.6)
    # Shade crisis periods (label=0) on equity and allocation charts
    if oracle_labels is not None:
        crisis = oracle_labels == 0
        crisis_axes = [ax1, ax2] + ([ax_pe] if ax_pe else [])
        for ax in crisis_axes:
            ax.fill_between(wf_dates, 0, 1, where=crisis,
                            color="red", alpha=0.08, transform=ax.get_xaxis_transform())
    for lev in leverages:
        r = results[lev]
        ax1.semilogy(wf_dates, r["eq_n"],
                     label=f"XGB x{lev:.1f} net Bourso ({r['n_cagr']*100:.1f}%, "
                           f"DD {r['n_dd']*100:.1f}%)",
                     color=colors[lev], linewidth=2)

    # PANX execution curve (normalized to QQQ B&H at PANX start)
    if panx_results is not None:
        pr = panx_results
        si = pr["start_idx"]
        scale = bh_eq[si]
        panx_eq_norm = pr["eq"] / pr["eq"][0] * scale
        ax1.semilogy(pr["dates"], panx_eq_norm,
                     label=f"PEA PANX open x1.0 ({pr['cagr']*100:.1f}%, DD {pr['dd']*100:.1f}%)",
                     color="tab:green", linewidth=1.5, linestyle="--")

    ax1.axvline(wf_dates[idx_10y], color="gray", linestyle=":", alpha=0.5)
    annot = f"10 ans\nB&H {bh_c10*100:.1f}%/an DD {bh_d10*100:.0f}%\n"
    for lev in leverages:
        r = results[lev]
        annot += f"x{lev:.1f} {r['n_c10']*100:.1f}%/an DD {r['n_d10']*100:.0f}%\n"
    mid_lev = leverages[len(leverages) // 2]
    y_mid = np.sqrt(results[mid_lev]["eq_n"].max() * results[mid_lev]["eq_n"].min())
    ax1.annotate(annot.strip(), xy=(wf_dates[idx_10y], y_mid), fontsize=8,
                 color="gray", ha="right", va="center",
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    ax1.set_ylabel("Equity (log scale)")
    ax1.set_title(f"{ticker} — XGBoost WF Strict Feature Selection — net frais Boursorama PEA")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)
    plt.setp(ax1.get_xticklabels(), visible=False)

    # Allocation
    for lev in sorted(leverages, reverse=True):
        ax2.fill_between(wf_dates, 0, results[lev]["al_c"] * 100,
                         alpha=0.2, color=colors[lev], label=f"x{lev:.1f}")
    max_alloc = max(leverages) * 100
    ax2.axhline(100, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    if max(leverages) >= 1.5:
        ax2.axhline(150, color="tab:red", linestyle="--", alpha=0.3, linewidth=0.8)
    if max(leverages) >= 2.0:
        ax2.axhline(200, color="darkred", linestyle="--", alpha=0.3, linewidth=0.8)
    ax2.set_ylabel("Allocation (%)")
    ax2.set_ylim(-5, max_alloc + 15)
    ax2.legend(loc="lower right", fontsize=8)
    ax2.grid(True, alpha=0.3)
    if ax_pe:
        plt.setp(ax2.get_xticklabels(), visible=False)
    else:
        ax2.set_xlabel("Date")
        ax2.xaxis.set_major_locator(mdates.YearLocator(2))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # PE Top 5 panel
    if ax_pe is not None and pe_daily is not None:
        pe_masked = pe_daily.reindex(wf_dates)
        pe_masked = pe_masked.ffill()
        ax_pe.plot(wf_dates, pe_masked.values, color="darkblue", lw=1.2,
                   label="PE Top 5 (daily)")
        ax_pe.axhline(20, color="green", ls=":", alpha=0.5, lw=1)
        ax_pe.axhline(35, color="orange", ls=":", alpha=0.5, lw=1)
        ax_pe.axhline(50, color="red", ls=":", alpha=0.5, lw=1)
        ax_pe.set_ylabel("PE Top 5")
        ax_pe.set_xlabel("Date")
        ax_pe.legend(loc="upper left", fontsize=8)
        ax_pe.grid(True, alpha=0.3)
        ax_pe.xaxis.set_major_locator(mdates.YearLocator(2))
        ax_pe.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.subplots_adjust(hspace=0.08)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"\nSaved: {save_path}")
    plt.show()
    return results


def plot_projection(results, leverages, pea_init=120_000, cto_init=50_000,
                    pea_inject=180_000, cto_inject=140_000,
                    inject_year=1.0, proj_years=5, save_path=None):
    """PEA+CTO projection: with and without cash injection, side by side."""
    TAX_PEA, TAX_CTO = 0.172, 0.30

    def _compute(lev, pea_inj, cto_inj):
        c = results[lev]["n_c10"]
        # PEA
        pea_g = pea_init * (1 + c) ** proj_years
        if pea_inj > 0:
            pea_g += pea_inj * (1 + c) ** (proj_years - inject_year)
        pea_cost = pea_init + pea_inj
        pea_t = (pea_g - pea_cost) * TAX_PEA
        pea_n = pea_g - pea_t
        # CTO
        inject_y = int(inject_year) + 1
        cto_val, cto_t, cto_cost = cto_init, 0, cto_init
        for y in range(proj_years):
            if cto_inj > 0 and y == inject_y:
                cto_val += cto_inj
                cto_cost += cto_inj
            gain = cto_val * c
            tax = gain * TAX_CTO
            cto_t += tax
            cto_val += gain - tax
        return dict(cagr=c, pea_cost=pea_cost, pea_n=pea_n, pea_t=pea_t,
                    cto_cost=cto_cost, cto_n=cto_val, cto_t=cto_t,
                    total_n=pea_n + cto_val, total_t=pea_t + cto_t,
                    total_g=pea_g + cto_val + cto_t)

    scenarios = [
        ("Sans injection", 0, 0),
        (f"+{pea_inject/1000:.0f}k PEA +{cto_inject/1000:.0f}k CTO @ {inject_year}a",
         pea_inject, cto_inject),
    ]

    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(leverages))
    bw = 0.35

    for si, (label, p_inj, c_inj) in enumerate(scenarios):
        for i, lev in enumerate(leverages):
            d = _compute(lev, p_inj, c_inj)
            pea_gn = d["pea_n"] - d["pea_cost"]
            cto_gn = d["cto_n"] - d["cto_cost"]
            pos = i + (si - 0.5) * bw

            ax.bar(pos, d["pea_cost"] / 1000, bw, color="tab:blue", alpha=0.3,
                   label="PEA capital" if i == 0 and si == 0 else "")
            ax.bar(pos, pea_gn / 1000, bw, bottom=d["pea_cost"] / 1000,
                   color="tab:blue", alpha=0.7,
                   label="PEA gains nets" if i == 0 and si == 0 else "")

            cto_bot = (d["pea_cost"] + pea_gn) / 1000
            ax.bar(pos, d["cto_cost"] / 1000, bw, bottom=cto_bot,
                   color="tab:orange", alpha=0.3,
                   label="CTO capital" if i == 0 and si == 0 else "")
            ax.bar(pos, cto_gn / 1000, bw, bottom=cto_bot + d["cto_cost"] / 1000,
                   color="tab:orange", alpha=0.7,
                   label="CTO gains nets" if i == 0 and si == 0 else "")

            net_top = d["total_n"] / 1000
            ax.bar(pos, d["total_t"] / 1000, bw, bottom=net_top,
                   color="red", alpha=0.3, hatch="///",
                   label="Impots" if i == 0 and si == 0 else "")
            ax.text(pos, net_top + d["total_t"] / 1000 + 15,
                    f"{d['total_n']/1000:.0f}k\u20ac",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Print projection summary
    for lev in leverages:
        d = _compute(lev, pea_inject, cto_inject)
        print(f"  x{lev:.1f} proj {proj_years}y: PEA {d['pea_n']/1000:.0f}k + "
              f"CTO {d['cto_n']/1000:.0f}k = {d['total_n']/1000:.0f}k net  "
              f"(impots {d['total_t']/1000:.0f}k)")

    ax.set_xticks(x)
    ax.set_xticklabels([f"x{lev:.1f}\nCAGR {results[lev]['n_c10']*100:.1f}%"
                         for lev in leverages], fontsize=10)
    ax.set_ylabel("Montant (k\u20ac)")
    ax.set_title(f"Projection {proj_years} ans net d'impot — PEA 17.2% sortie, CTO 30%/an\n"
                 f"PEA {pea_init/1000:.0f}k\u20ac + CTO {cto_init/1000:.0f}k\u20ac  |  "
                 f"Injection +{pea_inject/1000:.0f}k PEA +{cto_inject/1000:.0f}k CTO @ {inject_year}a",
                 fontsize=11)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    for i in range(len(leverages)):
        ax.text(i - 0.5 * bw, -0.08, "sans", ha="center", fontsize=7, color="gray",
                transform=ax.get_xaxis_transform())
        ax.text(i + 0.5 * bw, -0.08, "avec", ha="center", fontsize=7, color="gray",
                transform=ax.get_xaxis_transform())

    total_with = pea_init + pea_inject + cto_init + cto_inject
    ax.axhline(total_with / 1000, color="gray", linestyle="--", alpha=0.5)
    ax.text(-0.4, total_with / 1000, f" Capital avec\n {total_with/1000:.0f}k\u20ac",
            fontsize=8, color="gray", va="center")

    ax.set_ylim(0, max(_compute(lev, pea_inject, cto_inject)["total_g"]
                       for lev in leverages) / 1000 * 1.25)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.show()


def plot_recent(wf_dates, qqq_ret, wf_prob, prob_cash=0.5, prob_full=0.85,
                days=21, save_path=None, ticker="QQQ", leverages=None):
    """Plot last N trading days with leverages, fees, and allocation."""

    if leverages is None:
        leverages = [1.0, 1.5, 2.0]
    colors = {1.0: "tab:orange", 1.5: "tab:red", 1.75: "crimson", 2.0: "darkred"}

    # Slice last N days
    n = min(days, len(wf_dates))
    dates = wf_dates[-n:]
    ret = qqq_ret[-n:]
    prob = wf_prob[-n:]

    # Buy & hold
    cum_bh = np.cumprod(1 + ret)

    # Use integer x-axis (no weekend gaps)
    x = np.arange(n)
    date_labels = [d.strftime("%d %b") for d in dates]

    # Load PE daily
    try:
        from src.download_pe_qqq_top5 import load_pe_daily
        pe_daily = load_pe_daily()
    except Exception:
        pe_daily = None
    has_pe = pe_daily is not None and len(pe_daily) > 0

    n_rows = 4 if has_pe else 3
    h_ratios = [2, 1, 1, 1] if has_pe else [2, 1, 1]
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 11 if has_pe else 9),
                             height_ratios=h_ratios, sharex=True,
                             gridspec_kw={"hspace": 0.08})
    ax1, ax2, ax3 = axes[0], axes[1], axes[2]
    ax_pe = axes[3] if has_pe else None

    # B&H curve
    ax1.plot(x, (cum_bh - 1) * 100, label=f"{ticker} B&H", color="tab:blue", linewidth=2)

    title_parts = [f"{ticker} {(cum_bh[-1]-1)*100:+.1f}%"]

    for lev in leverages:
        # Simulate with fees on this slice
        eq_n, al_c, _ = simulate_with_fees(ret, prob, lev, prob_cash, prob_full)
        period_ret = (eq_n[-1] - 1) * 100

        ax1.plot(x, (eq_n - 1) * 100,
                 label=f"XGB x{lev:.1f} net ({period_ret:+.1f}%)",
                 color=colors[lev], linewidth=2)
        title_parts.append(f"x{lev:.1f} {period_ret:+.1f}%")

        # Allocation on ax2 (stacked fill, most visible = x2 behind)
        if lev == 2.0:
            ax2.fill_between(x, 0, al_c * 100, alpha=0.15, color=colors[lev], label=f"x{lev:.1f}")
        elif lev == 1.5:
            ax2.fill_between(x, 0, al_c * 100, alpha=0.25, color=colors[lev], label=f"x{lev:.1f}")
        else:
            ax2.fill_between(x, 0, al_c * 100, alpha=0.35, color=colors[lev], label=f"x{lev:.1f}")

    ax1.axhline(0, color="gray", linestyle="-", alpha=0.3)
    ax1.set_ylabel("Rendement cumulé (%)")
    ax1.set_title(f"Derniers {n}j ({dates[0].date()} \u2192 {dates[-1].date()})  "
                  + "  ".join(title_parts))
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)

    max_alloc = max(leverages) * 100
    ax2.axhline(100, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    if max(leverages) >= 1.5:
        ax2.axhline(150, color="tab:red", linestyle="--", alpha=0.3, linewidth=0.8)
    if max(leverages) >= 2.0:
        ax2.axhline(200, color="darkred", linestyle="--", alpha=0.3, linewidth=0.8)
    ax2.set_ylabel("Allocation (%)")
    ax2.set_ylim(-5, max_alloc + 15)
    ax2.legend(loc="lower right", fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Probability
    ax3.plot(x, prob, color="tab:purple", linewidth=2, marker="o", markersize=3)
    ax3.axhline(prob_cash, color="gray", linestyle="--", alpha=0.5, label=f"Cash ({prob_cash})")
    ax3.axhline(0.85, color="red", linestyle="--", alpha=0.5, label="Full (0.85)")
    ax3.fill_between(x, prob_cash, prob, where=prob >= prob_cash,
                     alpha=0.2, color="tab:green")
    ax3.fill_between(x, prob_cash, prob, where=prob < prob_cash,
                     alpha=0.2, color="tab:red")
    ax3.set_ylabel("P(invested)")
    ax3.set_ylim(0, 1)
    ax3.legend(loc="lower right", fontsize=9)
    ax3.grid(True, alpha=0.3)

    # PE Top 5 panel
    if ax_pe is not None and pe_daily is not None:
        import pandas as pd
        pe_slice = pe_daily.reindex(dates).ffill()
        ax_pe.plot(x, pe_slice.values, color="darkblue", lw=1.5, label="PE Top 5")
        ax_pe.axhline(20, color="green", ls=":", alpha=0.5, lw=1)
        ax_pe.axhline(35, color="orange", ls=":", alpha=0.5, lw=1)
        ax_pe.axhline(50, color="red", ls=":", alpha=0.5, lw=1)
        ax_pe.set_ylabel("PE Top 5")
        ax_pe.legend(loc="upper left", fontsize=8)
        ax_pe.grid(True, alpha=0.3)

    last_ax = ax_pe if ax_pe is not None else ax3
    last_ax.set_xlabel("Date")
    tick_step = max(1, n // 15)
    last_ax.set_xticks(x[::tick_step])
    last_ax.set_xticklabels([date_labels[i] for i in range(0, n, tick_step)], rotation=45)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.show()


def plot_comparison(results, prob_cash=0.5, prob_full=0.85, save_path=None):
    """Compare multiple tickers on same chart with x1.5 leverage."""
    import pandas as pd
    import matplotlib.cm as cm

    tickers = list(results.keys())
    n_tickers = len(tickers)
    cmap = cm.get_cmap("tab10")
    styles = ["-", "--", ":"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), height_ratios=[3, 1],
                                    sharex=True, gridspec_kw={"hspace": 0.08})

    for i, ticker in enumerate(tickers):
        res = results[ticker]
        wf_dates, price_ret, wf_prob = res[0], res[1], res[2]
        ticker_levs = res[3] if len(res) > 3 else [1.0, 1.5, 2.0]
        lev = max(ticker_levs)  # best available leverage
        years = (wf_dates[-1] - wf_dates[0]).days / 365.25
        color = cmap(i)
        ls = styles[i % len(styles)]

        # Best leverage with fees
        eq_n, al_c, _ = simulate_with_fees(price_ret, wf_prob, lev, prob_cash, prob_full)
        n_cagr = eq_n[-1] ** (1 / years) - 1
        n_dd = ((eq_n - np.maximum.accumulate(eq_n)) / np.maximum.accumulate(eq_n)).min()
        ax1.semilogy(wf_dates, eq_n,
                     label=f"{ticker} x{lev:.1f} ({n_cagr*100:.1f}%, DD {n_dd*100:.1f}%)",
                     color=color, linewidth=2.5, linestyle=ls)

        ax2.fill_between(wf_dates, 0, al_c * 100, alpha=0.15, color=color,
                         label=f"{ticker} x{lev:.1f}")

    title_tickers = " vs ".join(tickers)
    ax1.set_ylabel("Equity (log scale)")
    ax1.set_title(f"{title_tickers} — XGBoost WF net frais Boursorama PEA")
    ax1.legend(loc="upper left", fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)

    ax2.axhline(100, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    ax2.axhline(150, color="gray", linestyle="--", alpha=0.3, linewidth=0.8)
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
