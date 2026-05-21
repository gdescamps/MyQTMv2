"""
Backtest with the model deployed on every step where it has data — both the
VIX calm-mode override AND the IC gate are disabled. Only the VIX-spike
cash-out remains. Useful to see pure-model performance vs the gated version.
Output: outputs/no_calm/
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import backtest  # noqa: E402

# Force calm regime always OFF (threshold below any realistic VIX EMA100)
backtest.VIX_CALM_THRESHOLD = -1.0
backtest.VIX_CALM_COND_EMA100_SUP_EMA300 = False
# Disable IC gate (threshold below any realistic EMA value, min_steps=0)
backtest.IC_GATE_THRESHOLD = -1e9
backtest.IC_GATE_MIN_STEPS = 0

NEW_OUT = ROOT / "outputs" / "no_calm"
NEW_OUT.mkdir(parents=True, exist_ok=True)
backtest.OUTPUTS = NEW_OUT

backtest.run_equity()
print("\n■ no-calm backtest done")
