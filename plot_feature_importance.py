"""
Plot feature importance evolution over time from walk-forward training.

Input:  outputs/feature_importances.parquet
Output: outputs/feature_importance_evolution.jpg
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUTS = Path(__file__).parent / "outputs"


def main():
    path = OUTPUTS / "feature_importances.parquet"
    if not path.exists():
        print("ERROR: feature_importances.parquet not found — run train.py first")
        return

    df = pd.read_parquet(path)
    # Drop step column if present
    if "step" in df.columns:
        df = df.drop(columns=["step"])

    # Top N features: include up to first smart money feature
    mean_imp = df.mean().sort_values(ascending=False)
    TOP_N = 20
    for i, feat in enumerate(mean_imp.index):
        if "so_" in feat or "shares" in feat:
            TOP_N = max(TOP_N, i + 1)
            break
    top_features = mean_imp.head(TOP_N).index.tolist()

    print(f"Top {TOP_N} features by mean SHAP importance (up to smart money):")
    for i, feat in enumerate(top_features):
        is_sm = "so_" in feat or "shares" in feat
        tag = " ← SMART MONEY" if is_sm else ""
        print(f"  {i+1:2d}. {feat:<35s}  mean={mean_imp[feat]:.4f}{tag}")

    # Plot: 2-column layout (chart left, legend right)
    fig = plt.figure(figsize=(24, 14))
    gs = fig.add_gridspec(2, 2, width_ratios=[3, 1], height_ratios=[3, 1],
                          hspace=0.08, wspace=0.02,
                          top=0.95, bottom=0.05, left=0.05, right=0.98)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax_leg = fig.add_subplot(gs[:, 1])
    ax_leg.axis("off")

    # Panel 1: Stacked area of top features over time
    top_df = df[top_features]
    top_smooth = top_df.rolling(5, min_periods=1).mean()

    # Colors: use tab20, highlight smart money in red
    base_colors = plt.cm.tab20(np.linspace(0, 1, TOP_N))
    colors = []
    for i, feat in enumerate(top_features):
        if "so_" in feat or "shares" in feat:
            colors.append("#d62728")  # red for smart money
        else:
            colors.append(base_colors[i])

    ax1.stackplot(top_smooth.index, top_smooth.values.T,
                  labels=top_features, colors=colors, alpha=0.85)
    ax1.set_ylabel("SHAP Importance (stacked)")
    ax1.set_title(f"Top {TOP_N} Feature SHAP Importance — Walk-Forward", fontsize=14)
    ax1.grid(True, alpha=0.3)

    # Right column: feature list with rank and SHAP value
    y_start = 0.95
    y_step = min(0.035, 0.9 / max(TOP_N, 1))
    for i, feat in enumerate(top_features):
        y = y_start - i * y_step
        is_sm = "so_" in feat or "shares" in feat
        color = "#d62728" if is_sm else colors[i]
        fw = "bold" if is_sm else "normal"
        ax_leg.text(0.0, y, "■", fontsize=16, color=color, va="center",
                    transform=ax_leg.transAxes)
        ax_leg.text(0.08, y, f"{i+1:2d}. {feat}", fontsize=9, fontweight=fw,
                    va="center", transform=ax_leg.transAxes)
        ax_leg.text(0.85, y, f"{mean_imp[feat]:.4f}", fontsize=9, va="center",
                    ha="right", transform=ax_leg.transAxes, color="#333333")

    # Panel 2: Number of features with importance > 0
    n_active = (df > 0.001).sum(axis=1)
    ax2.plot(n_active.index, n_active.values, lw=2, color="#1f77b4")
    ax2.set_ylabel("Active features (SHAP > 0.001)")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)

    out = OUTPUTS / "feature_importance_evolution.jpg"
    fig.savefig(out, dpi=100, bbox_inches="tight", format="jpeg",
                pil_kwargs={"quality": 70, "optimize": True})
    plt.close(fig)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
