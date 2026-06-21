import json
import subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MASK = Path("stitched_mask.png")
CONFIG = Path("zone_config.json")   # change if yours is named differently,
                                      # e.g. "stitched_mask_zone_config.json"

# Short-duration override so CI doesn't sit through each model's full
# default run (SFM/CA default to 120s, Continuum to 40s, RVO to 400s).
# Both casings are included since SFM/RVO/CA read "max_time" while
# continuum_evacuation_path.py reads "MAX_TIME" — each script just
# ignores keys it doesn't recognize.
PARAMS_PATH = Path(__file__).resolve().parent / "ci_params.json"
CI_PARAMS = {
    "max_time": 15,
    "MAX_TIME": 15,
    "fire_spread_speed": 1.0,
    "fire_intensity_factor": 1.0,
}


def run_script(script_name):
    if not MASK.exists():
        raise FileNotFoundError(f"{MASK} not found — run the Map Parser step first.")
    if not CONFIG.exists():
        raise FileNotFoundError(f"{CONFIG} not found — run the Zone Editor step first.")

    PARAMS_PATH.write_text(json.dumps(CI_PARAMS))

    return subprocess.run(
        [sys.executable, script_name, str(MASK), str(CONFIG), str(PARAMS_PATH)],
        cwd=ROOT, capture_output=True, text=True, timeout=120,
    )