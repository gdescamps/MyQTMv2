"""
Walk-forward XGBoost backtest with continuous allocation 0-150%.
"""

import hashlib
import json
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


def _cache_key(X, y, feature_cols, xgb_params, feat_select, feat_power, feat_top_n,
               min_train, step, embargo, temperature):
    """Compute a hash of all inputs that affect walk-forward results."""
    h = hashlib.sha256()
    h.update(X.tobytes())
    h.update(y.tobytes())
    h.update(json.dumps(sorted(feature_cols)).encode())
    h.update(json.dumps(xgb_params, sort_keys=True).encode())
    h.update(f"{feat_select}_{feat_power}_{feat_top_n}".encode())
    h.update(f"{min_train}_{step}_{embargo}_{temperature}".encode())
    return h.hexdigest()[:16]


def _detect_device():
    """Try GPU, fallback to CPU."""
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

    # Check cache
    if use_cache:
        cache_hash = _cache_key(X, y, feature_cols, xgb_params, feat_select,
                                feat_power, feat_top_n, min_train, step,
                                embargo, temperature)
        cache_path = CACHE_DIR / f"wf_cache_{cache_hash}.npz"
        if cache_path.exists():
            cached = np.load(cache_path, allow_pickle=True)
            print(f"Walk-forward loaded from cache ({cache_path.name})")
            wf_pred = cached["wf_pred"]
            wf_proba = cached["wf_proba"]
            feat_names = list(cached["feat_names"])
            # Print top features from cached importances
            imp = cached["importances"]
            top_idx = np.argsort(imp)[-20:][::-1]
            print(f"\nTop 20 features (cached):")
            for idx in top_idx:
                print(f"  {feat_names[idx]:30s} {imp[idx]:.4f}")
            return wf_pred, wf_proba, None

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

    if feat_select:
        print(f"\nWalk-forward: {n_steps} steps, last feature set: "
              f"{len(last_feat_names)}/{len(feature_cols)}")

    # Top features (last model)
    imp = last_model.feature_importances_
    top_idx = np.argsort(imp)[-20:][::-1]
    print(f"\nTop 20 features (last model):")
    for idx in top_idx:
        print(f"  {last_feat_names[idx]:30s} {imp[idx]:.4f}")

    # Save cache
    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, wf_pred=wf_pred, wf_proba=wf_proba,
                 feat_names=np.array(last_feat_names),
                 importances=imp)
        print(f"Cache saved: {cache_path.name}")

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
                 pea_init=150_000, cto_init=100_000, proj_years=5,
                 save_path=None, ticker="QQQ", leverages=None, oracle_labels=None):
    import pandas as pd
    from matplotlib.gridspec import GridSpec

    if leverages is None:
        leverages = [1.0, 1.5, 2.0]

    years = (wf_dates[-1] - wf_dates[0]).days / 365.25
    N = len(qqq_ret)
    bh_eq = np.cumprod(1 + qqq_ret)
    bh_cagr, bh_dd = compute_metrics(bh_eq, years)
    colors = {1.0: "tab:orange", 1.5: "tab:red", 2.0: "darkred"}

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

    # 5y projection — PEA 17.2% at exit, CTO 30%/year
    TAX_PEA, TAX_CTO = 0.172, 0.30
    proj = {}
    for lev in leverages:
        c = results[lev]["n_c10"]
        pea_g = pea_init * (1 + c) ** proj_years
        pea_t = (pea_g - pea_init) * TAX_PEA
        pea_n = pea_g - pea_t
        cto_val, cto_t = cto_init, 0
        for _ in range(proj_years):
            gain = cto_val * c
            tax = gain * TAX_CTO
            cto_t += tax
            cto_val += gain - tax
        proj[lev] = dict(cagr=c, pea_g=pea_g, pea_t=pea_t, pea_n=pea_n,
                         cto_g=cto_init * (1 + c) ** proj_years,
                         cto_t=cto_t, cto_n=cto_val,
                         total_n=pea_n + cto_val, total_t=pea_t + cto_t,
                         total_g=pea_g + cto_init * (1 + c) ** proj_years)
        print(f"  x{lev:.1f} proj {proj_years}y: PEA {pea_n/1000:.0f}k + "
              f"CTO {cto_val/1000:.0f}k = {(pea_n+cto_val)/1000:.0f}k net  "
              f"(impots {(pea_t+cto_t)/1000:.0f}k)")

    # ── Plot ──
    fig = plt.figure(figsize=(18, 9))
    gs = GridSpec(2, 2, width_ratios=[3, 1.2], height_ratios=[3, 1],
                  hspace=0.08, wspace=0.15)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax3 = fig.add_subplot(gs[:, 1])

    # Equity curves
    ax1.semilogy(wf_dates, bh_eq,
                 label=f"{ticker} Buy & Hold ({bh_cagr*100:.1f}%, DD {bh_dd*100:.1f}%)",
                 color="tab:blue", linewidth=1.5, alpha=0.6)
    if oracle:
        ax1.semilogy(wf_dates, oracle["eq"],
                     label=f"Oracle label ({oracle['cagr']*100:.1f}%, DD {oracle['dd']*100:.1f}%)",
                     color="tab:green", linewidth=1.5, linestyle="--", alpha=0.7)
    for lev in leverages:
        r = results[lev]
        ax1.semilogy(wf_dates, r["eq_n"],
                     label=f"XGB x{lev:.1f} net Bourso ({r['n_cagr']*100:.1f}%, "
                           f"DD {r['n_dd']*100:.1f}%)",
                     color=colors[lev], linewidth=2)

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
    ax2.set_xlabel("Date")
    ax2.set_ylim(-5, max_alloc + 15)
    ax2.legend(loc="lower right", fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_locator(mdates.YearLocator(2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # 5y projection
    ax3.set_title(f"Projection {proj_years} ans net d'impot\n"
                  f"PEA 17.2% sortie - CTO 30%/an\n"
                  f"PEA {pea_init/1000:.0f}k\u20ac + CTO {cto_init/1000:.0f}k\u20ac",
                  fontsize=10)
    x = np.arange(len(leverages))
    bw = 0.5
    for i, lev in enumerate(leverages):
        d = proj[lev]
        pea_gn = d["pea_n"] - pea_init
        cto_gn = d["cto_n"] - cto_init

        ax3.bar(i, pea_init / 1000, bw, color="tab:blue", alpha=0.3,
                label="PEA capital" if i == 0 else "")
        ax3.bar(i, pea_gn / 1000, bw, bottom=pea_init / 1000,
                color="tab:blue", alpha=0.7, label="PEA gains nets" if i == 0 else "")

        cto_bot = (pea_init + pea_gn) / 1000
        ax3.bar(i, cto_init / 1000, bw, bottom=cto_bot,
                color="tab:orange", alpha=0.3, label="CTO capital" if i == 0 else "")
        ax3.bar(i, cto_gn / 1000, bw, bottom=cto_bot + cto_init / 1000,
                color="tab:orange", alpha=0.7, label="CTO gains nets" if i == 0 else "")

        net_top = d["total_n"] / 1000
        ax3.bar(i, d["total_t"] / 1000, bw, bottom=net_top,
                color="red", alpha=0.3, hatch="///", label="Impots" if i == 0 else "")
        ax3.text(i, net_top + d["total_t"] / 1000 + 15,
                 f"Net: {d['total_n']/1000:.0f}k\u20ac\nImpots: {d['total_t']/1000:.0f}k\u20ac",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax3.set_xticks(x)
    ax3.set_xticklabels([f"x{lev:.1f}\nCAGR {proj[lev]['cagr']*100:.1f}%"
                         for lev in leverages], fontsize=9)
    ax3.set_ylabel("Montant (k\u20ac)")
    ax3.legend(loc="upper left", fontsize=8)
    ax3.grid(True, alpha=0.3, axis="y")
    ax3.axhline((pea_init + cto_init) / 1000, color="gray", linestyle="--", alpha=0.5)
    ax3.text(-0.4, (pea_init + cto_init) / 1000, f" Capital\n {(pea_init+cto_init)/1000:.0f}k\u20ac",
             fontsize=8, color="gray", va="center")
    ax3.set_ylim(0, max(d["total_g"] for d in proj.values()) / 1000 * 1.25)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"\nSaved: {save_path}")
    plt.show()


def plot_recent(wf_dates, qqq_ret, wf_prob, prob_cash=0.5, prob_full=0.85,
                days=21, save_path=None, ticker="QQQ", leverages=None):
    """Plot last N trading days with leverages, fees, and allocation."""

    if leverages is None:
        leverages = [1.0, 1.5, 2.0]
    colors = {1.0: "tab:orange", 1.5: "tab:red", 2.0: "darkred"}

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

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 9), height_ratios=[2, 1, 1],
                                         sharex=True, gridspec_kw={"hspace": 0.08})

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
    ax3.set_xlabel("Date")
    ax3.set_ylim(0, 1)
    ax3.legend(loc="lower right", fontsize=9)
    ax3.grid(True, alpha=0.3)

    tick_step = max(1, n // 15)
    ax3.set_xticks(x[::tick_step])
    ax3.set_xticklabels([date_labels[i] for i in range(0, n, tick_step)], rotation=45)

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
