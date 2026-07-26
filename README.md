# T.R.A.G.I.C. — Crowd Evacuation Simulation & Floorplan Analysis
### Traffic Risk Assessment with Generative Intelligence and Crowd Simulation | Social Force Model · RVO · Continuum Crowds · Cellular Automata

> Feed it a floorplan. It tells you where people die.

![Hero — TRAGIC launcher showing a completed SFM simulation with agent trails, bottleneck markers, and a score overlay](docs/hero_launcher.png)

**TRAGIC** is an open-source **crowd evacuation simulation** tool that parses architectural **floorplan images** using a trained **U-Net** segmentation model, then runs **pedestrian dynamics simulations** using four algorithms — Social Force Model (SFM), Reciprocal Velocity Obstacles (RVO/ORCA), Continuum Crowds, and Cellular Automata (CA). It detects **evacuation bottlenecks**, scores **emergency exit placement**, and models **fire and hazard spread** — all from a single floorplan image with no manual wall tracing.

---

## The Why

Most building safety tools either cost a fortune or require you to hand-draw every wall and corridor into a proprietary format. TRAGIC takes a floorplan image — a photo, a scan, an architect's PNG — runs it through a trained U-Net to extract walkable space automatically, then simulates a crowd evacuation using four different algorithms. At the end you get a score, a heatmap, and a concrete recommendation like "the corridor at (420, 310) is your worst chokepoint, widen it."

Built for architects and students who want to understand whether a layout is actually safe, not just compliant on paper.

---

## How It's Structured

```
T.R.A.G.I.C/
├── Tragic_launcher.py           # Main GUI — 3-view pipeline
├── predict_tiled.py             # Standalone map parser (no GUI)
├── dxf_mask_maker.py            # DXF floorplan parser with PyQt6 GUI
├── model.py                     # U-Net architecture definition
├── unet.pth                     # Trained weights (downloaded on first run)
├── zone_detector.py             # Standalone zone editor (no GUI)
├── SFM_evacuation.py            # Social Force Model
├── RVO_evacuation.py            # Reciprocal Velocity Obstacles (ORCA)
├── continuum_evacuation_path.py # Continuum Crowds (Treuille et al. 2006)
├── CA_evacuation.py             # Cellular Automata
├── compare_models.py            # Run all four models and produce comparison table
└── output/                      # All runtime output written here
    ├── sfm_agent_paths.png
    ├── rvo_agent_paths.png
    ├── continuum_agent_paths.png
    ├── ca_paths.png
    ├── model_comparison.txt
    └── *.txt                    # Per-model text reports
```

The pipeline has three stages:

![Flow diagram — floorplan image → U-Net tiled inference → binary mask → watershed zone segmentation → agent spawn + BFS flow field → simulation loop → scored output image + text report](docs/pipeline_diagram.png)

**Stage 1 — Map Parser**: Tiles the input image with 50% overlap, runs each patch through the U-Net in batches, Gaussian-blends predictions back, binarizes to `stitched_mask.png` (black = walkable, white = wall).

**Stage 2 — Zone Editor**: Watershed segmentation splits walkable area into rooms and corridors. You click zones to set agent density, place exit markers, and optionally drop a hazard. Saves to a JSON config.

**Stage 3 — Simulation**: Launches one of the four simulation scripts as a subprocess, streams output to the UI, displays the result image and report.

---

## Quick Start

**Prerequisites:** Python 3.10+

```bash
git clone https://github.com/sankhya007/T.R.A.G.I.C
cd T.R.A.G.I.C
pip install torch>=2.0.0 opencv-python>=4.8.0 numpy>=1.24.0 scipy>=1.11.0 scikit-image>=0.21.0 PyQt6>=6.5.0
```

No GPU? Use the CPU-only torch wheel:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install opencv-python numpy scipy scikit-image PyQt6
```

```bash
python Tragic_launcher.py
```

No config files, no environment variables. Everything is driven through the UI.

> **Model weights** (`unet.pth`) download automatically the first time you run the parser. Manual download: [HuggingFace — sankhya007/Floorplan_parser_STITCH](https://huggingface.co/sankhya007/Floorplan_parser_STITCH/tree/main)

---

## Feature Walkthrough

### 1. Tiled Map Parsing (Batched Inference)

The U-Net was trained on 256×256 patches, but real floorplans can be 3000×2000px or larger. The parser computes a window size backwards from a patch budget — so you never exceed 40 tiles regardless of input resolution. Bigger image, bigger window, same patch count.

```python
aspect = pW / pH
n_cols_est = math.sqrt(MAX_PATCHES * aspect)
n_rows_est = math.sqrt(MAX_PATCHES / aspect)
WINDOW = max(WINDOW_MIN, int(math.ceil(max(pW / (n_cols_est * 0.5),
                                           pH / (n_rows_est * 0.5)))))
```

Patches are fed through the model in **batches of 8** rather than one at a time, cutting inference time to roughly 1/5 of the naive approach on CPU. Each patch still gets a Gaussian weight map so edges blend smoothly.

![Map Parser view — tiling parameters on the left, stitched binary mask on the right with white walls and black walkable space](docs/view1_parser.png)

The mask is also **directly editable** after parsing — draw over it with a brush to fix misclassified walls or corridors before moving to zone setup. Supports undo (Ctrl+Z) and variable brush size.

#### 1.1 DXF Floorplan Parsing (Experimental)

The U-Net model is a trained segmentation model and handles most clean architectural drawings well, but real-world cross-domain floorplans (hand-drawn, low-contrast, unusual scales) can produce inconsistent results. As an alternative input path, `dxf_mask_maker.py` accepts a raw DXF file — the native CAD format for architectural drawings — and lets you select which layers contain walls and obstacles directly, bypassing the image classification step entirely. This is not yet integrated into the main launcher UI; if it stabilises it will become a first-class alternative to the U-Net parser in Stage 1.

okay updating after trying some things out might not use this plan at all because figured out that combining a bunch of datasets to actually get a singular fix was not an option rather, would have used image manipulation at the 1st try. well so now that problem is solved to now you can use the parser directly on any floorplans which have curved walls and diagonal walls and it will parse pretty nicely, will be fixing the random noise for the objects in the map and the thin walls not getting predicted later on 

---

### 2. Zone Editor & Agent Configuration

Watershed segmentation automatically splits the walkable area into rooms and corridors. Click a zone, set its density index, and agent count is:

```python
agents = int(area_px * density_index * base_density / 1000)
```

A 5000px² room at density 1.0 with base 1.0 gets 5 agents. Set density to 0 to exclude a zone entirely (useful for storage rooms, staircases that aren't part of the evacuation path, etc.).

Exits are placed by clicking the map in Exit Mode, saved as `{"x": int, "y": int}` in the JSON config — the same format every simulation script reads. Multiple exits are supported; the scoring system measures how evenly they are used.

![Zone Editor — colored watershed regions, exit markers labeled E1/E2, zone list with density indices on the left](docs/view2_zones.png)

---

### 3. Hazard & Fire Spread

Hazard Mode drops a single point hazard on the map, saved as `{"x": int, "y": int}` under `"hazard"` in the zone config. All four models read it the same way and split the response into two layers:

- **Routing (permanent, hard)** — a fixed-radius circle (90px) around the hazard is carved out of the walkable mask *before* the flow field / BFS cost / potential field is built. Agents never path through it. Baked in once at setup, no per-tick rerouting cost.
- **Fire spread (growing, soft)** — a separate `fire_intensity` field grows and diffuses outward through the walkable area every simulation tick, producing a visual flame front and a soft repulsion force that pushes agents away from the fire.

Two parameters control fire behavior, exposed as sliders in the launcher:

```python
"fire_spread_speed":     1.0,   # diffusion rate multiplier
"fire_intensity_factor": 1.0,   # growth-to-saturation rate multiplier
```

CA's version of fire repulsion works as a tiebreak — among equally-good candidate cells, it picks the cooler one — since CA has no continuous velocity field to add a force to.

---

### 4. Four Simulation Algorithms

All four models take the same inputs (`stitched_mask.png` + `zone_config.json`) and produce the same output structure: a scored PNG with agent trails, exit utilization percentages, bottleneck markers, and a text report.

| Model | Core mechanic | Characteristic behavior |
|---|---|---|
| SFM | Attractive/repulsive force fields (Helbing 2000) | Smooth continuous flow, arch formation at exits |
| RVO | ORCA half-plane collision avoidance | Lane formation, minimal crossing paths |
| Continuum | Potential field + density-dependent speed (Treuille 2006) | Fluid crowd behavior, best for large numbers |
| CA | Moore neighbourhood BFS cost stepping | Discrete, natural bottleneck trees, fastest to run |

Score is 0–100 across four components, identical formula in all four scripts:

```python
score_rate    = (evacuated / total) * 50          # evacuation rate
score_time    = max(0, 20 * (1 - (mean_t - 20) / 60)) if mean_t > 20 else 20.0
score_balance = max(0, 15 * (1 - max_exit_deviation / ideal))  # exit load balance
score_bn      = max(0, 15 * (1 - (bn_fraction - 0.05) / 0.45)) # bottleneck severity
final_score   = int(score_rate + score_time + score_balance + score_bn)
```

![Simulation output — green/orange agent trails, red bottleneck circles labeled B1/B2, cyan exit circles with utilization percentages, score box top-left](docs/view3_output.png)

---

### 5. Model Comparison

Run all four models back-to-back on the same floorplan and get a comparison table:

```bash
python compare_models.py stitched_mask.png zone_config.json
```

Output (`output/model_comparison.txt`):

```
Model     Score  Evac %  Mean Evac (s)  Wall-clock (s)
--------  -----  ------  -------------  --------------
SFM       82     98.2    24.3           18.4
RVO       79     97.1    27.1           41.2
CA        85     99.0    21.8           6.3
Continuum 76     95.4    29.6           12.1
```

---

## Performance Notes

- **Parser**: batched inference (8 patches per forward pass) cuts wall-clock time ~5× vs. the naive one-at-a-time approach on CPU.
- **SFM**: agent-agent repulsion uses a spatial hash instead of an O(N²) distance matrix — O(N) in practice.
- **CA**: `DIRS_IDX` dict replaces `list.index()` in the inner movement loop; exit checks use scalar squared-distance math instead of numpy allocations per agent per tick.
- **Continuum**: `ContGrid` downsampling uses `cv2.resize` (C-level) instead of nested Python loops; `splat()` uses vectorized slice assignment instead of a triple nested loop.
- All four models thin trail storage to every 3–5 ticks — only used for drawing, zero impact on simulation accuracy.

---

## Gotchas

**"Output image not found" after simulation runs**
The launcher copies mask and zone config to hardcoded filenames in the project root before launching any subprocess. Always run `Tragic_launcher.py` from the project root, not a subdirectory.

**Continuum model is slow on large masks**
`grid_res` controls potential field resolution (default 4px per cell). If the sim is crawling, push it to 6 or 8 in the launcher config panel. You lose some path accuracy but it becomes usable on large floorplans.

**Agents spawn but zero evacuate**
Either (a) exit markers landed on a wall pixel — place them clearly inside a corridor, or (b) the zone config was saved before placing exits so `"exits"` is empty in the JSON. Open the JSON and check. Exit radius defaults to 22px; if exits are in tight spaces, increase it in the launcher.

**Hazard placed but agents don't route around it**
The carved-out radius may be swallowing the only corridor to an exit — if there's no walkable path left, agents hit maximum BFS cost and never evacuate. Move the hazard or reduce `hazard_block_radius` in the relevant script's config.

**Zone detector produces only 1–2 zones on a simple mask**
This is the known synthetic-mask plateau effect. Zone detection works correctly on real `stitched_mask.png` output from the parser. Don't run smoke tests with hand-drawn rectangles.

---

## Team

Aniket Sarkar · Shreyashi Malakar · Swapnil Roy · Tiyasa Nayak · Arinima Chakraborty

---

## Keywords

`crowd-evacuation-simulation` `floorplan-parsing` `pedestrian-dynamics` `social-force-model` `building-safety` `emergency-evacuation` `U-Net-segmentation` `fire-hazard-simulation` `occupancy-analysis` `egress-simulation` `RVO` `ORCA` `cellular-automata` `continuum-crowds` `PyQt6` `deep-learning` `computer-vision` `CubiCasa5K` `watershed-segmentation` `bottleneck-detection` `exit-scoring`

---

## Contributing

Bug reports, feature requests, and PRs welcome. SFM is the reference model for output format and report structure — new simulation backends should match its report schema. All output goes to `output/`, nothing written to the project root except `stitched_mask.png` and `zone_config.json`.

---

## License

MIT — do whatever you want with it.