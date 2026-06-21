# Contributing to T.R.A.G.I.C.

Thanks for taking a look at this project. T.R.A.G.I.C. (Tactical Risk Assessment & Grid-based Intelligent Crowd Simulation) parses architectural floorplans and simulates crowd evacuations using four different models. It's a young project, so contributions of any size — bug reports, doc fixes, new features — are welcome.

## Before you start

- Check open issues first to avoid duplicate work.
- For anything bigger than a small fix (new feature, big refactor), open an issue to discuss the approach before writing code. Saves you from rewriting a PR after the fact.

## Project layout

A quick map so you know where things live:

| Area | File(s) |
|---|---|
| Floorplan parsing (U-Net) | `model.py`, `predict_tiled.py` |
| Zone / exit / hazard editor | `zone_detector.py` |
| Simulation models | `SFM_evacuation.py`, `RVO_evacuation.py`, `continuum_evacuation_path.py`, `CA_evacuation.py` |
| GUI pipeline | `Tragic_launcher.py` |

**`SFM_evacuation.py` is the reference model.** Output format (PNG + TXT report), scoring logic, and CLI argument pattern (`script.py <mask> <zone_config> <params_json>`) all come from SFM. If you're adding a feature to one model, the other three should follow the same shape unless there's a good reason not to.

## Setting up locally

```bash
pip install -r requirements.txt
```

You'll need `unet.pth` (model weights) for the parsing step — see the README for where to get this. The simulation scripts run independently of the parser if you already have a `stitched_mask.png` and zone config.

## Making changes

- **Read the code before changing it.** Some things that look like bugs are intentional (e.g. soft repulsion near walls). Check git history / existing comments before "fixing" something.
- **Match existing patterns.** If you're adding a parameter, check how the other three simulation scripts expose it through `CONFIG`/`CFG` and `apply_runtime_args()` — new params should follow the same `argv[3]` JSON pickup convention so they work with the launcher GUI for free.
- **Output hygiene:** All simulation outputs go in `output/`. Don't write new files to the project root except `stitched_mask.png` and the zone config JSON — those are the two expected root-level artifacts.
- **`Tragic_launcher.py` is high-risk.** It's one big file wiring three GUI views together. Keep changes here minimal and targeted — don't refactor unrelated sections in the same PR.
- Keep diffs focused. One feature or fix per PR is easier to review than five at once.

## Testing your changes

There's no formal test suite yet. At minimum, before opening a PR:

- Run the script(s) you touched end-to-end with a sample mask + zone config.
- If you changed a simulation model, confirm it still produces a valid output PNG and report.
- If you touched the launcher, click through the actual flow (parse → zones → simulate) rather than just checking it imports.

## Style

- Plain, readable Python. No need for heavy abstraction — this codebase favors clarity over cleverness.
- Use `cv2`/`numpy` vectorized operations over per-pixel Python loops where it matters for performance (see the SFM refactor for an example).
- Comments explaining *why*, not just *what*, are appreciated — especially around anything physics/model-specific (SFM forces, BFS flow fields, etc).

## Submitting a PR

1. Fork the repo and create a branch from `main`.
2. Make your changes, test them.
3. Fill out the PR template — it's short on purpose.
4. Open the PR. Be ready to make small revisions if asked; that's normal, not a sign anything's wrong.

If you're unsure about anything, open an issue and ask before investing a lot of time. Happy to talk through ideas.