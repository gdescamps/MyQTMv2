"""
Compare Test IC, Val IC and gap (val - test) over time between:
  - XGB only (oos_predictions_xgb_only.parquet, backup before LGB)
  - XGB + LGB ensemble (oos_predictions.parquet, current)

Output: outputs/ic_xgb_vs_xgb_lgb.jpg
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = [
    ("XGB only",         ROOT / "data" / "oos_predictions_xgb_only.parquet", "#1f77b4"),
    ("XGB + LGB",        ROOT / "data" / "oos_predictions.parquet",          "#ff7f0e"),
]


def step_metrics(oos_path: Path) -> pd.DataFrame:
    oos = pd.read_parquet(oos_path).reset_index()
    oos["date"] = pd.to_datetime(oos["date"])

    val_per_step = (oos[oos["split"] == "val"]
                    .groupby("step")["val_ic"].first().rename("val_ic"))
    test_rows = oos[oos["split"] == "test"].dropna(subset=["score", "label"])

    def _step_test_ic(g):
        return g.groupby("date").apply(
            lambda x: x["score"].corr(x["label"]) if len(x) > 1 else np.nan,
            include_groups=False,
        ).mean()

    test_per_step = test_rows.groupby("step").apply(_step_test_ic).rename("test_ic")
    step_date = (oos[oos["split"] == "test"].groupby("step")["date"].min()
                 .rename("test_start"))

    df = pd.concat([step_date, val_per_step, test_per_step], axis=1).dropna()
    df["gap"] = df["val_ic"] - df["test_ic"]
    df = df.sort_values("test_start").set_index("test_start")
    df["val_ic_ema"]  = df["val_ic"].ewm(span=12).mean()
    df["test_ic_ema"] = df["test_ic"].ewm(span=12).mean()
    df["gap_ema"]     = df["gap"].ewm(span=12).mean()
    return df


frames = []
for label, path, color in CONFIGS:
    if not path.exists():
        sys.exit(f"Missing {path}")
    f = step_metrics(path)
    frames.append((label, color, f))
    print(f"{label:<14}  steps={len(f):3d}  val={f['val_ic'].mean():+.4f}  "
          f"test={f['test_ic'].mean():+.4f}  gap={f['gap'].mean():+.4f}")

# --- Plot: 3 panels ---
fig, axes = plt.subplots(3, 1, figsize=(18, 13), sharex=False)

# (1) Test IC EMA — the headline (true OOS performance)
ax = axes[0]
for label, color, df in frames:
    ax.plot(df.index, df["test_ic"],    color=color, alpha=0.18, lw=0.7)
    ax.plot(df.index, df["test_ic_ema"], color=color, lw=2.4,
            label=f"{label}  (mean {df['test_ic'].mean():+.4f})")
ax.axhline(0, color="black", lw=0.5, ls="--")
ax.set_ylabel("Test IC")
ax.set_title("Test IC per WF step (true OOS) — XGB only vs XGB + LGB ensemble",
             fontsize=13)
ax.legend(loc="upper left")
ax.grid(alpha=0.3)

# (2) Val IC EMA (honest: odd & not_embargoed)
ax = axes[1]
for label, color, df in frames:
    ax.plot(df.index, df["val_ic"],    color=color, alpha=0.18, lw=0.7)
    ax.plot(df.index, df["val_ic_ema"], color=color, lw=2.2,
            label=f"{label}  (mean {df['val_ic'].mean():+.4f})")
ax.axhline(0, color="black", lw=0.5, ls="--")
ax.set_ylabel("Val IC (odd & not_embargoed)")
ax.set_title("Val IC per WF step (honest reporting)", fontsize=13)
ax.legend(loc="upper left")
ax.grid(alpha=0.3)

# (3) Gap = val - test EMA
ax = axes[2]
for label, color, df in frames:
    ax.plot(df.index, df["gap"],     color=color, alpha=0.18, lw=0.7)
    ax.plot(df.index, df["gap_ema"], color=color, lw=2.4,
            label=f"{label}  (mean gap {df['gap'].mean():+.4f})")
ax.axhline(0, color="black", lw=0.5, ls="--")
ax.set_xlabel("Test window start")
ax.set_ylabel("Val IC − Test IC")
ax.set_title("Generalisation gap (positive = val higher than test = drift)",
             fontsize=12)
ax.legend(loc="upper left")
ax.grid(alpha=0.3)

plt.tight_layout()
out = ROOT / "outputs" / "ic_xgb_vs_xgb_lgb.jpg"
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=110, bbox_inches="tight")
print(f"\nSaved → {out}")
