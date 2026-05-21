"""
Backtest with the model permanently disabled — calm mode (top-3 rolling
252d Sharpe) is forced everywhere except VIX spike cash-out days. Useful
as a baseline to measure how much the model adds on top of the rule-based
calm allocator over the full 2011-2026 walk-forward.

Method: keep the unified pipeline but force the IC gate closed (which
already drives the step into calm mode in the main backtest).

Outputs to outputs/calmonly/.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import backtest  # noqa: E402

# Force the IC gate to never open → every step uses calm mode.
backtest.IC_GATE_THRESHOLD = 1e9  # unreachable

# Redirect outputs
NEW_OUT = ROOT / "outputs" / "calmonly"
NEW_OUT.mkdir(parents=True, exist_ok=True)
backtest.OUTPUTS = NEW_OUT

backtest.run_equity()
print("\n■ calm-only backtest done")
