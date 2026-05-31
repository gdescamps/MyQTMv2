#!/usr/bin/env python3
"""
Multi-ETF Strategy launcher (smart money + follow leads).

Usage:
    python run_multietfs_strategy.py train           # train smart money model
    python run_multietfs_strategy.py train_fl         # train follow leads model
    python run_multietfs_strategy.py features         # feature engineering
    python run_multietfs_strategy.py select           # feature selection
    python run_multietfs_strategy.py backtest         # run backtest (equity)
    python run_multietfs_strategy.py robustness       # run robustness analysis
    python run_multietfs_strategy.py search_features  # search best features
    python run_multietfs_strategy.py search_params    # search follow-leads hyperparams
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "multietfs_strategy"))

COMMANDS = {
    "train":           "train_smart_money",
    "train_fl":        "train_follow_leads",
    "features":        "feature_engineering",
    "select":          "select_features",
    "backtest":        "backtest",
    "robustness":      "backtest",
    "search_features": "search_features_xgb",
    "search_params":   "search_hyperparams_fl",
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        print("Available commands:", ", ".join(COMMANDS.keys()))
        sys.exit(0)

    cmd = sys.argv[1]
    if cmd not in COMMANDS:
        print(f"Unknown command: {cmd}")
        print("Available:", ", ".join(COMMANDS.keys()))
        sys.exit(1)

    module_name = COMMANDS[cmd]
    mod = __import__(module_name)

    if cmd == "robustness":
        sys.argv = [sys.argv[0], "--robustness"] + sys.argv[2:]
        mod.main()
    elif cmd == "features":
        mod.main()
    else:
        mod.main()


if __name__ == "__main__":
    main()
