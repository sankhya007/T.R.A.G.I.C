import cv2
import json
import sys
import numpy as np
from collections import deque
from pathlib import Path

#  CONFIG 
MASK_PATH   = Path("stitched_mask.png")
CONFIG_PATH = Path("zone_config.json")
OUT_PATHS   = Path("output/sfm_agent_paths.png")
OUT_REPORT  = Path("output/SFM_output_report.txt")

DT       = 0.05
MAX_TIME = 120

DESIRED_SPEED   = 55.0
RELAXATION_TIME = 0.3
AGENT_RADIUS    = 6
AA_STRENGTH     = 250.0
AA_RANGE        = 14.0
WA_STRENGTH     = 350.0
WA_RANGE        = 7.0
EXIT_RADIUS     = 22
STUCK_DIST      = 2
STUCK_TICKS     = 20

# for the fire spread 

FIRE_SPREAD_SPEED     = 1.0   # multiplier on diffusion rate
FIRE_INTENSITY_FACTOR = 1.0   # multiplier on growth-to-saturation rate
FIRE_BLOCK_THRESHOLD  = 0.45  # intensity above which a cell is a hard wall (impassable)
AGENT_VISION_RADIUS   = 60    # px — how far ahead an agent can "see" fire and react

if len(sys.argv) > 1: MASK_PATH = sys.argv[1]
if len(sys.argv) > 2: CONFIG_PATH = sys.argv[2]
if len(sys.argv) > 3 and Path(sys.argv[3]).exists():
    with open(sys.argv[3]) as f:
        _params = json.load(f)
    FIRE_SPREAD_SPEED     = _params.get("fire_spread_speed", FIRE_SPREAD_SPEED)
    FIRE_INTENSITY_FACTOR = _params.get("fire_intensity_factor", FIRE_INTENSITY_FACTOR)
# 


# 
# STEP 1 — Load mask
# 

img          = cv2.imread(MASK_PATH, cv2.IMREAD_GRAYSCALE)
H, W         = img.shape
walkable     = img < 128
walk_u8      = walkable.astype(np.uint8) * 255
dist_to_wall = cv2.distanceTransform(walk_u8, cv2.DIST_L2, 5)

wall_grad_y, wall_grad_x = np.gradient(dist_to_wall.astype(np.float32))
wall_grad_mag = np.sqrt(wall_grad_x**2 + wall_grad_y**2) + 1e-8
wall_grad_x  /= wall_grad_mag
wall_grad_y  /= wall_grad_mag


# 
# STEP 2 — BFS flow field
# 

def build_flow_field(walk_mask, exits_list, h, w):
    """BFS distance + flow direction toward nearest exit.
    walk_mask should already have fire-blocked cells removed by the caller."""
    cost   = np.full((h, w), -1, dtype=np.int32)
    flow_x = np.zeros((h, w), dtype=np.float32)
    flow_y = np.zeros((h, w), dtype=np.float32)
    queue = deque()
    for ex in exits_list:
        ex_x, ex_y = int(ex["x"]), int(ex["y"])
        for dy in range(-EXIT_RADIUS, EXIT_RADIUS + 1):
            for dx in range(-EXIT_RADIUS, EXIT_RADIUS + 1):
                nx, ny = ex_x + dx, ex_y + dy
                if 0 <= nx < w and 0 <= ny < h and walk_mask[ny, nx] and cost[ny, nx] == -1:
                    cost[ny, nx] = 0
                    queue.append((nx, ny))
    DIRS = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
    while queue:
        cx, cy = queue.popleft()
        for dx, dy in DIRS:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < w and 0 <= ny < h and walk_mask[ny, nx] and cost[ny, nx] == -1:
                cost[ny, nx] = cost[cy, cx] + 1
                queue.append((nx, ny))

    #  vectorized direction extraction (replaces the old per-pixel double loop) 
    # For each of the 8 neighbour offsets, shift the cost grid and keep whichever
    # neighbour gives the lowest cost so far (same "steepest descent toward exit"
    # rule as before, just done as 8 array ops instead of H*W*8 Python comparisons).
    src = cost.astype(np.float32)
    src[cost == -1] = np.inf
    best_cost = src.copy()
    bx = np.zeros((h, w), dtype=np.float32)
    by = np.zeros((h, w), dtype=np.float32)

    for dx, dy in DIRS:
        shifted = np.full((h, w), np.inf, dtype=np.float32)
        y0, y1 = max(0, -dy), h - max(0, dy)
        x0, x1 = max(0, -dx), w - max(0, dx)
        sy0, sy1 = max(0, dy), h - max(0, -dy)
        sx0, sx1 = max(0, dx), w - max(0, -dx)
        shifted[y0:y1, x0:x1] = src[sy0:sy1, sx0:sx1]

        better = shifted < best_cost
        best_cost = np.where(better, shifted, best_cost)
        bx = np.where(better, float(dx), bx)
        by = np.where(better, float(dy), by)

    mag = np.sqrt(bx**2 + by**2)
    valid = (cost >= 0) & (mag > 0)
    flow_x[valid] = bx[valid] / mag[valid]
    flow_y[valid] = by[valid] / mag[valid]

    return cost, flow_x, flow_y


def spread_fire(intensity, walk_mask, ticks_elapsed, speed_mult, growth_mult):
    """Vectorized fire growth + 4-neighbour diffusion (replaces the per-burning-pixel
    Python loop with shifted-array adds)."""
    burning = intensity > 0.02
    if not burning.any():
        return intensity
    growth_rate    = 0.15 * growth_mult
    diffusion_rate = 0.12 * speed_mult * ticks_elapsed

    intensity = intensity.copy()
    intensity[burning] = np.minimum(1.0, intensity[burning] + growth_rate * ticks_elapsed)

    push = intensity * diffusion_rate  # only meaningful where intensity > 0
    new_intensity = intensity.copy()

    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        shifted_push = np.zeros_like(push)
        y0, y1 = max(0, dy), intensity.shape[0] - max(0, -dy)
        x0, x1 = max(0, dx), intensity.shape[1] - max(0, -dx)
        sy0, sy1 = max(0, -dy), intensity.shape[0] - max(0, dy)
        sx0, sx1 = max(0, -dx), intensity.shape[1] - max(0, dx)
        shifted_push[y0:y1, x0:x1] = push[sy0:sy1, sx0:sx1]
        new_intensity = np.where(walk_mask, np.minimum(1.0, new_intensity + shifted_push), new_intensity)

    return new_intensity


with open(CONFIG_PATH) as f:
    cfg = json.load(f)


exits = cfg["exits"]
hazard_cfg = cfg.get("hazard")

HAZARD_BLOCK_RADIUS = 90  # px — permanently avoided zone, bump up if still not enough

routing_walkable = walkable
hazard_zone = np.zeros((H, W), dtype=bool)
fire_intensity = np.zeros((H, W), dtype=np.float32)
fire_grad_x = np.zeros((H, W), dtype=np.float32)
fire_grad_y = np.zeros((H, W), dtype=np.float32)

if hazard_cfg:
    fx, fy = int(np.clip(hazard_cfg["x"], 0, W-1)), int(np.clip(hazard_cfg["y"], 0, H-1))
    fire_intensity[fy, fx] = 1.0
    print(f"Hazard ignited at ({fx},{fy})")

    yy, xx = np.ogrid[:H, :W]
    hazard_zone = (xx - fx)**2 + (yy - fy)**2 <= HAZARD_BLOCK_RADIUS**2
    routing_walkable = walkable & ~hazard_zone

cost, flow_x, flow_y = build_flow_field(routing_walkable, exits, H, W)
print(f"Flow field built. Reachable cells: {(cost >= 0).sum()}")
base_cost, _, _ = build_flow_field(walkable, exits, H, W)

# 
# STEP 3 — Spawn agents
# 

from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

dist_norm  = cv2.normalize(dist_to_wall, None, 0, 1.0, cv2.NORM_MINMAX)
coords     = peak_local_max(dist_norm, min_distance=40, labels=walk_u8)
seed_mask  = np.zeros(dist_norm.shape, dtype=bool)
seed_mask[tuple(coords.T)] = True
markers, _ = ndi.label(seed_mask)
labels     = watershed(-dist_to_wall, markers, mask=walk_u8)


def zone_id_mask(labels_array, zid):
    try:
        target_id = int(zid) if np.issubdtype(labels_array.dtype, np.integer) else zid
    except (TypeError, ValueError):
        target_id = zid

    mask = labels_array == target_id
    if np.isscalar(mask) or getattr(mask, "ndim", 0) != labels_array.ndim:
        return np.zeros(labels_array.shape, dtype=bool)
    return mask


zone_pixels = {}
for zdata in cfg["zones"]:
    zid    = zdata["zone_id"]
    ys, xs = np.where(zone_id_mask(labels, zid))
    # valid  = (dist_to_wall[ys, xs] > AGENT_RADIUS) & (cost[ys, xs] >= 0)
    valid  = (dist_to_wall[ys, xs] > AGENT_RADIUS) & (base_cost[ys, xs] >= 0)
    if valid.any():
        zone_pixels[zid] = list(zip(xs[valid].tolist(), ys[valid].tolist()))

agents = []
for zdata in cfg["zones"]:
    n      = zdata["agents"]
    zid    = zdata["zone_id"]
    pixels = zone_pixels.get(zid, [])
    if n == 0 or not pixels:
        continue
    chosen = [pixels[i] for i in np.random.choice(len(pixels), min(n, len(pixels)), replace=False)]
    for px, py in chosen:
        agents.append({
            "x": float(px), "y": float(py),
            "vx": 0.0,      "vy": 0.0,
            "evacuated": False, "time": None,
            "exit_used": None,              # which exit index this agent used
            "trail": [(float(px), float(py))],
            "stuck_buf": [],
        })

print(f"Spawned {len(agents)} agents")


# 
# STEP 4 — Simulation
# 

exit_pts    = np.array([[e["x"], e["y"]] for e in exits], dtype=np.float32)
density_map = np.zeros((H, W), dtype=np.float32)

# Congestion map: accumulates agent-ticks per cell
# We use this to measure how long agents spent in each area
congestion_map = np.zeros((H, W), dtype=np.float32)

total_steps = int(MAX_TIME / DT)

for step in range(total_steps):
    sim_time = step * DT
    active   = [a for a in agents if not a["evacuated"]]
    if not active:
        break
    
    #  Fire growth (visual spread + soft repulsion only). Routing around
    #    the hazard is already permanently handled by hazard_zone, so no
    #    expensive reroute/BFS needed here anymore. 
    if hazard_cfg and step % 10 == 0:
        fire_intensity = spread_fire(fire_intensity, walkable, DT * 10,
                                      FIRE_SPREAD_SPEED, FIRE_INTENSITY_FACTOR)
        gy, gx = np.gradient(fire_intensity)
        fire_grad_x, fire_grad_y = -gx, -gy
        
    #  Vectorized agent state pull 
    n = len(active)
    pos = np.array([[a["x"], a["y"]] for a in active], dtype=np.float64)
    vel = np.array([[a["vx"], a["vy"]] for a in active], dtype=np.float64)
    ix  = np.clip(pos[:, 0], 0, W - 1).astype(np.int32)
    iy  = np.clip(pos[:, 1], 0, H - 1).astype(np.int32)
    
    # Force 1: Driving (toward exit via flow field). If an agent has no flow
    # vector, it's either inside the blocked hazard zone (push it straight
    # out, away from the hazard center) or in some other unreachable pocket
    # (random heading as before).
    ex_dir = np.stack([flow_x[iy, ix], flow_y[iy, ix]], axis=1)
    no_dir = (ex_dir[:, 0] == 0) & (ex_dir[:, 1] == 0)
    if no_dir.any():
        in_hazard = hazard_zone[iy, ix] & no_dir if hazard_cfg else np.zeros_like(no_dir)

        if in_hazard.any():
            away = pos[in_hazard] - np.array([fx, fy], dtype=np.float64)
            away_mag = np.linalg.norm(away, axis=1, keepdims=True)
            away_mag[away_mag < 1e-6] = 1.0
            ex_dir[in_hazard] = away / away_mag

        stuck_other = no_dir & ~in_hazard
        if stuck_other.any():
            rand_a = np.random.uniform(0, 2*np.pi, size=stuck_other.sum())
            ex_dir[stuck_other, 0] = np.cos(rand_a)
            ex_dir[stuck_other, 1] = np.sin(rand_a)
    f_drive = (DESIRED_SPEED * ex_dir - vel) / RELAXATION_TIME

    # Force 2: Agent-agent repulsion — vectorized pairwise computation
    # (replaces the old O(N^2) Python double loop with one NumPy pass).
    diff  = pos[:, None, :] - pos[None, :, :]                  # (n, n, 2)
    dist  = np.sqrt((diff**2).sum(axis=2))                     # (n, n)
    np.fill_diagonal(dist, np.inf)
    in_range = dist < AGENT_RADIUS * 6
    safe_dist = np.where(dist < 1e-3, 1e-3, dist)
    mag = AA_STRENGTH * np.exp((AGENT_RADIUS * 2 - safe_dist) / AA_RANGE)
    mag = np.where(in_range, mag, 0.0)
    f_agent = (mag[:, :, None] * diff / safe_dist[:, :, None]).sum(axis=1)

    # Force 3: Wall repulsion
    d = dist_to_wall[iy, ix]
    wall_active = d < WA_RANGE * 2
    wall_mag = WA_STRENGTH * np.exp(-d / WA_RANGE)
    f_wall = np.stack([wall_grad_x[iy, ix], wall_grad_y[iy, ix]], axis=1) * (wall_mag * wall_active)[:, None]

    # Force 4: Fire repulsion (soft push, complements the hard-wall reroute above)
    local_risk = fire_intensity[iy, ix]
    fire_active = local_risk > 0.05
    fire_mag = 400.0 * local_risk
    f_fire = np.stack([fire_grad_x[iy, ix], fire_grad_y[iy, ix]], axis=1) * (fire_mag * fire_active)[:, None]

    vel = vel + (f_drive + f_agent + f_wall + f_fire) * DT
    spd = np.linalg.norm(vel, axis=1)
    over = spd > DESIRED_SPEED * 1.5
    if over.any():
        vel[over] *= (DESIRED_SPEED * 1.5 / spd[over])[:, None]

    new_pos = np.clip(pos + vel * DT, [0, 0], [W - 1, H - 1])

    for i, agent in enumerate(active):
        x, y = pos[i]
        vx, vy = vel[i]
        nx_pos, ny_pos = new_pos[i]

        # Stuck check (kept per-agent: cheap, and mutates each agent's own buffer)
        buf = agent["stuck_buf"]
        buf.append((x, y))
        if len(buf) > STUCK_TICKS:
            buf.pop(0)
            if abs(buf[-1][0]-buf[0][0]) + abs(buf[-1][1]-buf[0][1]) < STUCK_DIST:
                a = np.random.uniform(0, 2 * np.pi)
                vx = np.cos(a) * DESIRED_SPEED * 0.5
                vy = np.sin(a) * DESIRED_SPEED * 0.5
                nx_pos = np.clip(x + vx * DT, 0, W - 1)
                ny_pos = np.clip(y + vy * DT, 0, H - 1)

        if   walkable[int(ny_pos), int(nx_pos)]: agent["x"], agent["y"] = nx_pos, ny_pos
        elif walkable[int(y),      int(nx_pos)]: agent["x"] = nx_pos;  vy *= 0.5
        elif walkable[int(ny_pos), int(x)     ]: agent["y"] = ny_pos;  vx *= 0.5
        else: vx, vy = 0.0, 0.0

        agent["vx"], agent["vy"] = vx, vy
        agent["trail"].append((agent["x"], agent["y"]))

        cell_x = int(agent["x"])
        cell_y = int(agent["y"])
        density_map[cell_y, cell_x]    += 1
        congestion_map[cell_y, cell_x] += DT   # seconds spent here

        # Exit check — record WHICH exit
        epos  = np.array([agent["x"], agent["y"]])
        dists = np.linalg.norm(exit_pts - epos, axis=1)
        nearest_exit = int(np.argmin(dists))
        if dists[nearest_exit] < EXIT_RADIUS:
            agent["evacuated"] = True
            agent["time"]      = sim_time
            agent["exit_used"] = nearest_exit

    if step % 100 == 0:
        evac = sum(a["evacuated"] for a in agents)
        print(f"  t={sim_time:.1f}s  active={len(active)}  evacuated={evac}")

evac_final = sum(a["evacuated"] for a in agents)
print(f"Done  t={sim_time:.1f}s  evacuated={evac_final}/{len(agents)}")


# 
# ANALYSIS 1 — Bottleneck scoring
# Find high-congestion blobs, rank by total agent-seconds lost there
# 

# Threshold: cells where congestion is in top 20%
walkable_congestion = congestion_map[walkable]
if walkable_congestion.max() > 0:
    threshold = np.percentile(walkable_congestion[walkable_congestion > 0], 80)
else:
    threshold = 1.0

bn_mask     = (congestion_map > threshold).astype(np.uint8) * 255
contours, _ = cv2.findContours(bn_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

bottlenecks = []
for cnt in contours:
    # bounding box of this blob
    bx, by, bw, bh = cv2.boundingRect(cnt)
    # total agent-seconds spent inside this blob
    region_congestion = congestion_map[by:by+bh, bx:bx+bw].sum()
    # narrowness: how tight is this corridor
    # use the minimum dimension of the bounding box as a proxy for width
    narrowness = min(bw, bh)
    # centroid
    M = cv2.moments(cnt)
    if M["m00"] > 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = bx + bw//2, by + bh//2
    bottlenecks.append({
        "cx": cx, "cy": cy,
        "agent_seconds": float(region_congestion),
        "width_px": narrowness,
    })

# Sort by agent-seconds lost (worst first)
bottlenecks.sort(key=lambda b: b["agent_seconds"], reverse=True)
top_bottlenecks = bottlenecks[:5]


# 
# ANALYSIS 2 — Exit utilization
# 

exit_counts = [0] * len(exits)
for a in agents:
    if a["evacuated"] and a["exit_used"] is not None:
        exit_counts[a["exit_used"]] += 1

total_evacuated = sum(exit_counts)


# 
# ANALYSIS 3 — Score 0-100
# Components:
#   - Evacuation rate         (0-50 pts)
#   - Mean evacuation time    (0-20 pts)  faster = better
#   - Exit balance            (0-15 pts)  even distribution = better
#   - Bottleneck severity     (0-15 pts)  less congestion = better
# 

total  = len(agents)
times  = [a["time"] for a in agents if a["evacuated"] and a["time"] is not None]
rate   = evac_final / total

# Component 1: evacuation rate
score_rate = rate * 50

# Component 2: mean time (cap at MAX_TIME for non-evacuated)
if times:
    mean_t = np.mean(times)
    # 20 pts if mean < 20s, 0 pts if mean > 80s
    score_time = max(0, 20 * (1 - (mean_t - 20) / 60)) if mean_t > 20 else 20.0
else:
    score_time = 0.0

# Component 3: exit balance (how evenly distributed is traffic)
if total_evacuated > 0:
    fractions = [c / total_evacuated for c in exit_counts]
    ideal     = 1.0 / len(exits)
    # max deviation from ideal
    max_dev   = max(abs(f - ideal) for f in fractions)
    # 0 dev = 15 pts, dev = 1.0 = 0 pts
    score_balance = max(0, 15 * (1 - max_dev / ideal))
else:
    score_balance = 0.0

# Component 4: bottleneck severity
# measure as fraction of total agent-time spent in bottlenecks
total_agent_time  = congestion_map.sum()
bn_agent_time     = sum(b["agent_seconds"] for b in top_bottlenecks)
bn_fraction       = bn_agent_time / (total_agent_time + 1e-8)
# 0.05 fraction = 15 pts, 0.5 fraction = 0 pts
score_bn          = max(0, 15 * (1 - (bn_fraction - 0.05) / 0.45))

final_score = int(score_rate + score_time + score_balance + score_bn)
final_score = min(100, max(0, final_score))

# Single recommendation based on worst component
worst = min(
    ("evacuation_rate", score_rate / 50),
    ("exit_balance",    score_balance / 15),
    ("bottlenecks",     score_bn / 15),
    ("evac_time",       score_time / 20),
    key=lambda x: x[1]
)

# Find worst bottleneck position for the recommendation
if top_bottlenecks:
    wb = top_bottlenecks[0]
    bn_pos = f"({wb['cx']}, {wb['cy']})"
else:
    bn_pos = "unknown"

# Find most underused exit
if total_evacuated > 0:
    min_exit_idx  = int(np.argmin(exit_counts))
    min_exit_pos  = f"({int(exits[min_exit_idx]['x'])}, {int(exits[min_exit_idx]['y'])})"
else:
    min_exit_idx  = 0
    min_exit_pos  = "N/A"

RECOMMENDATIONS = {
    "evacuation_rate": f"Too many agents failed to evacuate — check for isolated rooms with no path to any exit.",
    "exit_balance":    f"Exit {min_exit_idx} at {min_exit_pos} handled almost no traffic. Consider repositioning it closer to high-density zones or adding signage.",
    "bottlenecks":     f"The corridor at {bn_pos} is your biggest chokepoint. Widen it or add a parallel route.",
    "evac_time":       f"Evacuation is too slow. Add an exit closer to the center of the building.",
}
recommendation = RECOMMENDATIONS[worst[0]]


# 
# STEP 5 — Output image
# 

base = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# Density heatmap
if density_map.max() > 0:
    dn   = (density_map / density_map.max() * 255).astype(np.uint8)
    heat = cv2.applyColorMap(dn, cv2.COLORMAP_HOT)
    base = cv2.addWeighted(base, 0.45, heat * walkable.astype(np.uint8)[:,:,None], 0.55, 0)
  
# Hazard overlay  
if hazard_cfg and fire_intensity.max() > 0:
    fire_u8 = (np.clip(fire_intensity, 0, 1) * 255).astype(np.uint8)
    fire_color = cv2.applyColorMap(fire_u8, cv2.COLORMAP_HOT)
    fire_mask3 = (fire_intensity > 0.03).astype(np.uint8)[:, :, None]
    base = np.where(fire_mask3 > 0, cv2.addWeighted(base, 0.4, fire_color, 0.6, 0), base)

# Agent trails
for agent in agents:
    color = (0, 200, 0) if agent["evacuated"] else (0, 80, 200)
    trail = agent["trail"]
    for i in range(1, len(trail)):
        cv2.line(base,
                 (int(trail[i-1][0]), int(trail[i-1][1])),
                 (int(trail[i][0]),   int(trail[i][1])),
                 color, 1)

# Exits — labeled with utilization %
for idx, ex in enumerate(exits):
    pct   = (exit_counts[idx] / total_evacuated * 100) if total_evacuated > 0 else 0
    color = (0, 255, 255)
    cv2.circle(base, (int(ex["x"]), int(ex["y"])), 12, color, 2)
    cv2.putText(base, f"E{idx} {pct:.0f}%",
                (int(ex["x"]) - 18, int(ex["y"]) - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

# Top bottlenecks — numbered red circles
for rank, bn in enumerate(top_bottlenecks):
    cv2.circle(base, (bn["cx"], bn["cy"]), 14, (0, 0, 255), 2)
    cv2.putText(base, f"B{rank+1}",
                (bn["cx"] - 8, bn["cy"] + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

# Score box top-left
score_color = (0,200,0) if final_score >= 80 else (0,180,255) if final_score >= 60 else (0,0,255)
cv2.rectangle(base, (4, 4), (130, 30), (0, 0, 0), -1)
cv2.putText(base, f"SCORE: {final_score}/100",
            (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, score_color, 1)

import os as _os; _os.makedirs("output", exist_ok=True)
cv2.imwrite(OUT_PATHS, base)
print(f"Saved: {OUT_PATHS}")


# 
# STEP 6 — Report
# 

hazard_str = f"active @ ({fx},{fy}) r={HAZARD_BLOCK_RADIUS}px" if hazard_cfg else "none"

lines = [
    "=" * 55,
    "       EVACUATION ANALYSIS REPORT",
    "=" * 55,
    "",
    f"  OVERALL SCORE : {final_score} / 100",
    "",
    f"  Total agents    : {total}",
    f"  Evacuated       : {evac_final}  ({100*rate:.1f}%)",
    f"  Trapped/timeout : {total - evac_final}  ({100*(1-rate):.1f}%)",
    f"  Hazard          : {hazard_str}",
]

if times:
    lines += [
        f"  Fastest evac    : {min(times):.1f}s",
        f"  Mean evac time  : {np.mean(times):.1f}s",
        f"  Slowest evac    : {max(times):.1f}s",
    ]

lines += [
    "",
    "-" * 55,
    "  EXIT UTILIZATION",
    "-" * 55,
]
for idx, ex in enumerate(exits):
    pct    = (exit_counts[idx] / total_evacuated * 100) if total_evacuated > 0 else 0
    bar    = "" * int(pct / 5)
    status = " UNDERUSED" if pct < (100 / len(exits) * 0.4) else ""
    lines.append(f"  Exit {idx} ({int(ex['x'])},{int(ex['y'])}): {exit_counts[idx]:3d} agents  {pct:5.1f}%  {bar} {status}")

lines += [
    "",
    "-" * 55,
    "  TOP BOTTLENECKS  (ranked by agent-seconds lost)",
    "-" * 55,
]
for rank, bn in enumerate(top_bottlenecks):
    lines.append(
        f"  B{rank+1}  position ({bn['cx']:4d},{bn['cy']:4d})  "
        f"corridor width ~{bn['width_px']}px  "
        f"{bn['agent_seconds']:.0f} agent-seconds"
    )

lines += [
    "",
    "-" * 55,
    "  RECOMMENDATION",
    "-" * 55,
    f"  {recommendation}",
    "",
    "  Score breakdown:",
    f"    Evacuation rate  : {score_rate:.0f}/50",
    f"    Evacuation speed : {score_time:.0f}/20",
    f"    Exit balance     : {score_balance:.0f}/15",
    f"    Bottleneck sev.  : {score_bn:.0f}/15",
    "=" * 55,
]

report = "\n".join(lines)
#print("\n" + report) # get replaced bitch 
print("\n" + report.encode("ascii", errors="replace").decode("ascii"))
with open(OUT_REPORT, "w", encoding="utf-8") as f:
    f.write(report)
print(f"\nSaved: {OUT_REPORT}")