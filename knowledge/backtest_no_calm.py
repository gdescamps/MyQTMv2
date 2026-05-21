"""
Backtest with calm-mode DISABLED — VIX_CALM_THRESHOLD lowered so the calm
fallback never fires. The model drives every test day (except VIX spike cash
days, which remain unchanged). Useful to see pure-model performance vs the
calm-overridden version.
Output: outputs/no_calm/
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import backtest  # noqa: E402

# Force calm regime always OFF (threshold below any realistic VIX EMA100)
backtest.VIX_CALM_THRESHOLD = -1.0
backtest.VIX_CALM_COND_EMA100_SUP_EMA300 = False

NEW_OUT = ROOT / "outputs" / "no_calm"
NEW_OUT.mkdir(parents=True, exist_ok=True)
backtest.OUTPUTS = NEW_OUT

backtest.run_equity()
print("\n■ no-calm backtest done")
