"""
Walk-forward XGBoost backtest — QQQ close-to-close, allocation continue 0-100%.

close-to-close : la proba de l'indice i est calculee ~5 min avant la cloture US[i]
(prix ~ close[i]) et l'ordre part a ce moment-la. La position prise n'encaisse
donc qu'a partir de close[i] -> le rendement ret[i+1]. C'est ce que reflete
`exec_lag=1` dans simulate (aucun look-ahead). Voir son docstring.

Backtest x1 uniquement, SANS frais (ni frais de transaction, ni TER).
"""

import os
import subprocess
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


def _gpu_busy(util_threshold=20):
    """Is another workload using the GPU right now?

    Returns True (busy), False (free), or None (can't tell — no nvidia-smi).
    Considers the GPU busy if any compute process is resident OR utilisation
    is above util_threshold. Memory is ignored (N/A on unified-memory GB10).
    """
    try:
        apps = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5)
        if apps.returncode == 0 and apps.stdout.strip():
            # Exclude our own PID: once XGBoost opens a CUDA context, this process
            # shows up here too — that's not "another workload".
            own = os.getpid()
            others = [int(x) for x in apps.stdout.split()
                      if x.strip().isdigit() and int(x) != own]
            if others:
                return True  # another process is resident on the GPU
        util = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        if util.returncode == 0 and util.stdout.strip():
            vals = [int(v) for v in util.stdout.split() if v.strip().isdigit()]
            if vals and max(vals) >= util_threshold:
                return True
        return False
    except Exception:
        return None  # nvidia-smi missing/unreadable — let caller decide


def _detect_device():
    """Pick XGBoost device. Respects XGBOOST_DEVICE override.

    - XGBOOST_DEVICE=cpu|cuda  → forced.
    - XGBOOST_DEVICE=auto or unset → use GPU only if available AND not busy,
      otherwise CPU. This lets the nightly job grab the GPU when it's free and
      gracefully share with other GPU workloads when they're running.
    """
    override = (os.environ.get("XGBOOST_DEVICE") or "").strip().lower()
    if override in ("cpu", "cuda"):
        return override

    # auto: skip GPU if another workload is using it
    if _gpu_busy() is True:
        print("GPU busy (autre workload détecté) → CPU")
        return "cpu"

    # GPU free (or undetectable) — use it if XGBoost can actually run on cuda
    try:
        m = xgb.XGBClassifier(device="cuda", n_estimators=1, verbosity=0)
        m.fit(np.zeros((10, 2)), np.zeros(10, dtype=int), verbose=False)
        return "cuda"
    except Exception:
        return "cpu"


def walk_forward(X, y, feature_cols, min_train=504, step=21, embargo=21,
                 temperature=3.0, xgb_params=None, feat_select=True,
                 feat_top_n=None, feat_power=1.7):
    """Rolling-window walk-forward. Recalcule tout a chaque appel (pas de cache)."""
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
            print(f"  Step {n_steps}: t={t} train={len(train_idx)} "
                  f"features={len(last_feat_names)}/{len(feature_cols)}")

        t = test_end

    if feat_select:
        print(f"\nWalk-forward: {n_steps} steps, "
              f"last feature set: {len(last_feat_names)}/{len(feature_cols)}")

    imp = last_model.feature_importances_ if last_model is not None \
        else np.zeros(len(last_feat_names))
    top_idx = np.argsort(imp)[-20:][::-1]
    print(f"\nTop 20 features:")
    for idx in top_idx:
        print(f"  {last_feat_names[idx]:30s} {imp[idx]:.4f}")

    return wf_pred, wf_proba, last_model


def compute_metrics(eq, years):
    cagr = eq[-1] ** (1 / years) - 1
    dd = ((eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq)).min()
    return cagr, dd


def simulate(qqq_ret, wf_prob, prob_cash=0.5, prob_full=0.85, exec_lag=1):
    """Backtest QQQ close-to-close, allocation continue 0-100%, SANS frais.

    L'allocation suit la cible chaque jour (proba -> alloc), sans frais de
    transaction ni TER : equity = cumprod(1 + ret * alloc).

    Alignement (exec_lag) — c'est ce qui distingue un backtest realiste d'un
    backtest avec look-ahead :

      La proba de l'indice i est calculee ~5 min avant la cloture US[i] (le prix
      a 15h55 ET ~ close[i]) et l'ordre part a ce moment-la. La position n'est
      donc en place qu'a partir de close[i] : elle encaisse le rendement
      close[i]->close[i+1] = ret[i+1], PAS ret[i] (close[i-1]->close[i], deja
      termine au moment ou on decide). exec_lag=1 reflete ce decalage = aucun
      look-ahead.

      exec_lag=0 apparie prob[i] x ret[i] : cela suppose d'etre deja positionne
      a close[i-1] avec une proba qui n'existe qu'a close[i] -> look-ahead d'un
      jour (gonfle fortement le CAGR). A n'utiliser que pour diagnostic.
    """
    prob = np.asarray(wf_prob, dtype=float)
    if exec_lag > 0:
        shifted = np.full_like(prob, prob_cash)  # avant le 1er signal exploitable -> cash
        shifted[exec_lag:] = prob[:-exec_lag]
        prob = shifted

    alloc = np.clip((prob - prob_cash) / (prob_full - prob_cash), 0, 1)
    eq = np.cumprod(1 + np.asarray(qqq_ret, dtype=float) * alloc)
    return eq, alloc


def _oracle_equity(qqq_ret, oracle_labels, exec_lag=1):
    """Equity d'un oracle parfait, meme decalage d'execution que la strategie."""
    lab = np.asarray(oracle_labels, dtype=float)
    if exec_lag > 0:
        shifted = np.zeros_like(lab)
        shifted[exec_lag:] = lab[:-exec_lag]
        lab = shifted
    N = len(qqq_ret)
    eq = np.ones(N)
    for i in range(1, N):
        eq[i] = eq[i - 1] * (1 + qqq_ret[i] * lab[i])
    return eq


def plot_results(wf_dates, qqq_ret, wf_prob, prob_cash=0.5, prob_full=0.85,
                 save_path=None, ticker="QQQ", oracle_labels=None):
    """Backtest QQQ close-to-close : equity (B&H / XGB net frais / oracle) + allocation + PE."""
    years = (wf_dates[-1] - wf_dates[0]).days / 365.25
    bh_eq = np.cumprod(1 + qqq_ret)
    bh_cagr, bh_dd = compute_metrics(bh_eq, years)

    eq_n, al_c = simulate(qqq_ret, wf_prob, prob_cash, prob_full)
    n_cagr, n_dd = compute_metrics(eq_n, years)

    oracle = None
    if oracle_labels is not None:
        oracle_eq = _oracle_equity(qqq_ret, oracle_labels)
        o_cagr, o_dd = compute_metrics(oracle_eq, years)
        oracle = dict(eq=oracle_eq, cagr=o_cagr, dd=o_dd)

    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"Periode: {wf_dates[0].date()} -> {wf_dates[-1].date()} ({years:.1f} ans)")
    print(f"{'':35s} {'CAGR':>8s} {'Total':>8s} {'MaxDD':>8s}")
    print(f"{ticker + ' Buy & Hold':35s} {bh_cagr*100:7.1f}% {bh_eq[-1]:7.1f}x {bh_dd*100:7.1f}%")
    if oracle:
        print(f"{'Oracle (label parfait)':35s} {oracle['cagr']*100:7.1f}% "
              f"{oracle['eq'][-1]:7.1f}x {oracle['dd']*100:7.1f}%")
    print(f"{'XGB strategy':35s} {n_cagr*100:7.1f}% {eq_n[-1]:7.1f}x {n_dd*100:7.1f}%")

    # ── Load PE daily (panneau optionnel) ──
    try:
        from src.download_pe_qqq_top5 import load_pe_daily
        pe_daily = load_pe_daily()
    except Exception:
        pe_daily = None
    has_pe = pe_daily is not None and len(pe_daily) > 0

    n_rows = 3 if has_pe else 2
    h_ratios = [3, 1, 1] if has_pe else [3, 1]
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 11 if has_pe else 9),
                             height_ratios=h_ratios, sharex=True)
    ax1, ax2 = axes[0], axes[1]
    ax_pe = axes[2] if has_pe else None

    # Equity curves
    ax1.semilogy(wf_dates, bh_eq,
                 label=f"{ticker} Buy & Hold ({bh_cagr*100:.1f}%, DD {bh_dd*100:.1f}%)",
                 color="tab:blue", linewidth=1.5, alpha=0.6)
    if oracle:
        ax1.semilogy(wf_dates, oracle["eq"],
                     label=f"Oracle ({oracle['cagr']*100:.1f}%, DD {oracle['dd']*100:.1f}%)",
                     color="gray", linewidth=1.0, alpha=0.5, linestyle=":")
    ax1.semilogy(wf_dates, eq_n,
                 label=f"XGB strategy ({n_cagr*100:.1f}%, DD {n_dd*100:.1f}%)",
                 color="tab:orange", linewidth=2)

    # Shade crisis periods (label=0)
    if oracle_labels is not None:
        crisis = np.asarray(oracle_labels) == 0
        for ax in [ax1, ax2] + ([ax_pe] if ax_pe else []):
            ax.fill_between(wf_dates, 0, 1, where=crisis,
                            color="red", alpha=0.08, transform=ax.get_xaxis_transform())

    ax1.set_ylabel("Equity (log scale)")
    ax1.set_title(f"{ticker} — XGBoost WF close-to-close (x1, sans frais)")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)
    plt.setp(ax1.get_xticklabels(), visible=False)

    # Allocation
    ax2.fill_between(wf_dates, 0, al_c * 100, alpha=0.3, color="tab:orange")
    ax2.axhline(100, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    ax2.set_ylabel("Allocation (%)")
    ax2.set_ylim(-5, 115)
    ax2.grid(True, alpha=0.3)
    if ax_pe:
        plt.setp(ax2.get_xticklabels(), visible=False)
    else:
        ax2.set_xlabel("Date")
        ax2.xaxis.set_major_locator(mdates.YearLocator(2))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # PE Top 5 panel
    if ax_pe is not None:
        pe_masked = pe_daily.reindex(wf_dates).ffill()
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

    return dict(eq_n=eq_n, al_c=al_c, cagr=n_cagr, dd=n_dd,
                bh_cagr=bh_cagr, bh_dd=bh_dd)


def plot_recent(wf_dates, qqq_ret, wf_prob, prob_cash=0.5, prob_full=0.85,
                days=21, save_path=None, ticker="QQQ"):
    """Plot last N trading days : rendement cumule, allocation, proba, PE."""
    n = min(days, len(wf_dates))
    dates = wf_dates[-n:]
    ret = qqq_ret[-n:]
    prob = wf_prob[-n:]

    cum_bh = np.cumprod(1 + ret)
    x = np.arange(n)
    date_labels = [d.strftime("%d %b") for d in dates]

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

    # Note: simulate sur la tranche recente (exec_lag interne = 1)
    eq_n, al_c = simulate(ret, prob, prob_cash, prob_full)
    period_ret = (eq_n[-1] - 1) * 100

    ax1.plot(x, (cum_bh - 1) * 100, label=f"{ticker} B&H ({(cum_bh[-1]-1)*100:+.1f}%)",
             color="tab:blue", linewidth=2)
    ax1.plot(x, (eq_n - 1) * 100, label=f"XGB ({period_ret:+.1f}%)",
             color="tab:orange", linewidth=2)
    ax1.axhline(0, color="gray", linestyle="-", alpha=0.3)
    ax1.set_ylabel("Rendement cumulé (%)")
    ax1.set_title(f"Derniers {n}j ({dates[0].date()} → {dates[-1].date()})  "
                  f"{ticker} {(cum_bh[-1]-1)*100:+.1f}%  XGB {period_ret:+.1f}%")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Allocation
    ax2.fill_between(x, 0, al_c * 100, alpha=0.3, color="tab:orange")
    ax2.axhline(100, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    ax2.set_ylabel("Allocation (%)")
    ax2.set_ylim(-5, 115)
    ax2.grid(True, alpha=0.3)

    # Probability
    ax3.plot(x, prob, color="tab:purple", linewidth=2, marker="o", markersize=3)
    ax3.axhline(prob_cash, color="gray", linestyle="--", alpha=0.5, label=f"Cash ({prob_cash})")
    ax3.axhline(prob_full, color="red", linestyle="--", alpha=0.5, label=f"Full ({prob_full})")
    ax3.fill_between(x, prob_cash, prob, where=prob >= prob_cash, alpha=0.2, color="tab:green")
    ax3.fill_between(x, prob_cash, prob, where=prob < prob_cash, alpha=0.2, color="tab:red")
    ax3.set_ylabel("P(invested)")
    ax3.set_ylim(0, 1)
    ax3.legend(loc="lower right", fontsize=9)
    ax3.grid(True, alpha=0.3)

    if ax_pe is not None:
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
    """Compare plusieurs tickers (equity x1 net frais + allocation)."""
    import matplotlib.cm as cm

    tickers = list(results.keys())
    cmap = cm.get_cmap("tab10")
    styles = ["-", "--", ":"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), height_ratios=[3, 1],
                                    sharex=True, gridspec_kw={"hspace": 0.08})

    for i, ticker in enumerate(tickers):
        res = results[ticker]
        wf_dates, price_ret, wf_prob = res[0], res[1], res[2]
        years = (wf_dates[-1] - wf_dates[0]).days / 365.25
        color = cmap(i)
        ls = styles[i % len(styles)]

        eq_n, al_c = simulate(price_ret, wf_prob, prob_cash, prob_full)
        n_cagr, n_dd = compute_metrics(eq_n, years)
        ax1.semilogy(wf_dates, eq_n,
                     label=f"{ticker} ({n_cagr*100:.1f}%, DD {n_dd*100:.1f}%)",
                     color=color, linewidth=2.5, linestyle=ls)
        ax2.fill_between(wf_dates, 0, al_c * 100, alpha=0.15, color=color, label=ticker)

    ax1.set_ylabel("Equity (log scale)")
    ax1.set_title(f"{' vs '.join(tickers)} — XGBoost WF close-to-close (x1, sans frais)")
    ax1.legend(loc="upper left", fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)

    ax2.axhline(100, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    ax2.set_ylabel("Allocation (%)")
    ax2.set_xlabel("Date")
    ax2.set_ylim(-5, 115)
    ax2.legend(loc="lower right", fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_locator(mdates.YearLocator(2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"\nSaved: {save_path}")
    plt.show()
