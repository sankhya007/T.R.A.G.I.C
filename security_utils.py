"""Small security boundary helpers shared by the launcher and simulators."""

import json
import math
import os
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parent
MAX_JSON_BYTES = 5 * 1024 * 1024


def load_json_object(path: str | Path, *, label: str = "JSON file") -> dict[str, Any]:
    """Load a bounded UTF-8 JSON object, rejecting non-object JSON values."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"{label} is not a regular file: {source}")
    if source.stat().st_size > MAX_JSON_BYTES:
        raise ValueError(f"{label} exceeds the {MAX_JSON_BYTES // 1024 // 1024} MiB limit")
    with source.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _number(value: Any, name: str, *, minimum: float | None = None,
            maximum: float | None = None) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be at most {maximum}")


def load_zone_config(path: str | Path) -> dict[str, Any]:
    """Load and validate the structural data every evacuation model relies on."""
    config = load_json_object(path, label="Zone configuration")
    exits = config.get("exits")
    zones = config.get("zones")
    if not isinstance(exits, list) or not exits:
        raise ValueError("Zone configuration requires a non-empty 'exits' list")
    if not isinstance(zones, list):
        raise ValueError("Zone configuration requires a 'zones' list")
    if len(exits) > 256 or len(zones) > 10_000:
        raise ValueError("Zone configuration contains too many exits or zones")
    for index, exit_ in enumerate(exits):
        if not isinstance(exit_, dict):
            raise ValueError(f"exits[{index}] must be an object")
        _number(exit_.get("x"), f"exits[{index}].x", minimum=-1_000_000, maximum=1_000_000)
        _number(exit_.get("y"), f"exits[{index}].y", minimum=-1_000_000, maximum=1_000_000)
    for index, zone in enumerate(zones):
        if not isinstance(zone, dict):
            raise ValueError(f"zones[{index}] must be an object")
        _number(zone.get("zone_id"), f"zones[{index}].zone_id", minimum=0)
        _number(zone.get("agents", 0), f"zones[{index}].agents", minimum=0, maximum=100_000)
        if int(zone.get("agents", 0)) != zone.get("agents", 0):
            raise ValueError(f"zones[{index}].agents must be an integer")
        if "density_index" in zone:
            _number(zone["density_index"], f"zones[{index}].density_index", minimum=0)
    hazard = config.get("hazard")
    if hazard is not None:
        if not isinstance(hazard, dict):
            raise ValueError("hazard must be an object or null")
        _number(hazard.get("x"), "hazard.x", minimum=-1_000_000, maximum=1_000_000)
        _number(hazard.get("y"), "hazard.y", minimum=-1_000_000, maximum=1_000_000)
    return config


def load_runtime_params(path: str | Path, allowed_keys: Iterable[str]) -> dict[str, float | int]:
    """Load only expected finite numeric simulation parameters."""
    params = load_json_object(path, label="Simulation parameters")
    allowed = set(allowed_keys)
    unknown = set(params) - allowed
    if unknown:
        raise ValueError(f"Unsupported simulation parameter(s): {', '.join(sorted(unknown))}")
    for key, value in params.items():
        _number(value, f"Simulation parameter '{key}'", minimum=-1_000_000, maximum=1_000_000)
    return params


def output_path(filename: str) -> Path:
    """Return a controlled output path; filenames cannot escape the output directory."""
    if Path(filename).name != filename:
        raise ValueError("Output filename must not contain a directory component")
    root = Path(os.environ.get("TRAGIC_OUTPUT_DIR", PROJECT_ROOT / "output")).resolve()
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    target = (root / filename).resolve()
    if target.parent != root:
        raise ValueError("Output path escapes the configured output directory")
    return target
