import subprocess, sys, re, time
from pathlib import Path
from security_utils import load_zone_config, output_path

MASK   = sys.argv[1] if len(sys.argv) > 1 else "stitched_mask.png"
CONFIG = sys.argv[2] if len(sys.argv) > 2 else "zone_config.json"
if not Path(MASK).is_file():
    raise FileNotFoundError(f"Mask is not a regular file: {MASK}")
load_zone_config(CONFIG)

RUNS = [
    ("SFM",       "SFM_evacuation.py",            output_path("SFM_output_report.txt")),
    ("RVO",       "RVO_evacuation.py",             output_path("RVO_output_report.txt")),
    ("CA",        "CA_evacuation.py",              output_path("ca_report.txt")),
    ("Continuum", "continuum_evacuation_path.py",  output_path("continuum_report.txt")),
]

def parse_report(path):
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    def grab(pattern, cast=float):
        m = re.search(pattern, text)
        return cast(m.group(1)) if m else None
    return {
        "score":    grab(r"OVERALL SCORE\s*:\s*(\d+)", int),
        "evac_pct": grab(r"Evacuated\s*:\s*\d+\s*\(([\d.]+)%\)"),
        "mean_t":   grab(r"Mean evac time\s*:\s*([\d.]+)s"),
    }

results = {}
for name, script, report_path in RUNS:
    print(f"\n=== Running {name} ===")
    t0 = time.time()
    proc = subprocess.run([sys.executable, script, MASK, CONFIG], shell=False, cwd=Path(__file__).resolve().parent)
    elapsed = time.time() - t0
    if proc.returncode != 0 or not Path(report_path).exists():
        print(f"  {name} FAILED")
        results[name] = None
        continue
    m = parse_report(report_path)
    m["runtime_s"] = round(elapsed, 1)
    results[name] = m
    print(f"  done in {elapsed:.1f}s  score={m['score']}")

cols = ["Model", "Score", "Evac %", "Mean Evac (s)", "Wall-clock (s)"]
rows = [cols]
for name, _, _ in RUNS:
    m = results.get(name)
    rows.append([name, "FAIL", "-", "-", "-"] if m is None else [
        name, str(m["score"]), str(m["evac_pct"]), str(m["mean_t"]), str(m["runtime_s"])
    ])

widths = [max(len(r[i]) for r in rows) for i in range(len(cols))]
lines = ["  ".join(r[i].ljust(widths[i]) for i in range(len(cols))) for r in rows]
lines.insert(1, "-" * (sum(widths) + 2 * (len(widths) - 1)))
table = "\n".join(lines)

print("\n" + table)
comparison_path = output_path("model_comparison.txt")
comparison_path.write_text(table + "\n", encoding="utf-8")
print(f"\nSaved: {comparison_path}")
