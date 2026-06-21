import subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MASK = Path("stitched_mask.png")
CONFIG = Path("zone_config.json")   # change if yours is named differently,
                                      # e.g. "stitched_mask_zone_config.json"


def run_script(script_name):
    if not MASK.exists():
        raise FileNotFoundError(f"{MASK} not found — run the Map Parser step first.")
    if not CONFIG.exists():
        raise FileNotFoundError(f"{CONFIG} not found — run the Zone Editor step first.")

    return subprocess.run(
        [sys.executable, script_name, str(MASK), str(CONFIG)],
        cwd=ROOT, capture_output=True, text=True, timeout=300,
    )