"""
Backtest LONG mode but force calm-market always (no model).
Allocates top-3 by rolling 252d Sharpe at each step, VIX spike → 3d cash.
Outputs to outputs/long_calmonly/.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["QTM_MODE"] = "long"

import backtest  # noqa: E402
from etf import UNIVERSE  # noqa: E402

# Force calm regime always on
backtest.VIX_CALM_THRESHOLD = 1e9
backtest.VIX_CALM_COND_EMA100_SUP_EMA300 = False

# Redirect outputs
NEW_OUT = ROOT / "outputs" / "long_calmonly"
NEW_OUT.mkdir(parents=True, exist_ok=True)
backtest.OUTPUTS = NEW_OUT

# Patch _pivot_step so test_scores always exposes the full UNIVERSE columns,
# even for ETFs missing from the OOS predictions file (GLD/RING were added to
# UNIVERSE_LONG after training so they have no model scores). In calm mode the
# scores are overwritten with rolling 252d Sharpe anyway.
import numpy as np  # noqa: E402
_orig_pivot = backtest._pivot_step
_full_etfs = [e.bourso for e in UNIVERSE]


def _pivot_step_full(step_data):
    df = _orig_pivot(step_data)
    missing = [c for c in _full_etfs if c not in df.columns]
    for c in missing:
        df[c] = np.nan
    return df[_full_etfs]


backtest._pivot_step = _pivot_step_full

backtest.run_equity()
print("\n■ calm-only backtest done")
