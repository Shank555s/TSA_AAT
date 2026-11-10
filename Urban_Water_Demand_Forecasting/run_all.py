"""Run the full pipeline: python run_all.py"""

import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"

STEPS = [
    "preprocessing",
    "eda",
    "seasonal_analysis",
    "sarima_baseline",
    "xgboost_models",
    "peak_evaluation",
    "final_evaluation",
]

for i, step in enumerate(STEPS, 1):
    print(f"\n[{i}/{len(STEPS)}] {step}.py\n" + "-" * 60, flush=True)
    subprocess.run([sys.executable, str(SRC / f"{step}.py")], check=True)

print("\nDone. See the results/ folder.")
