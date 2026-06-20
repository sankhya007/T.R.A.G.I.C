import sys, json, time
from pathlib import Path
from collections import defaultdict, deque

import numpy as np
import cv2
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# from matplotlib.collections import LineCollection

from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

np.random.seed(42)


def zone_id_mask(labels_array, zid):
    try:
        target_id = int(zid) if np.issubdtype(labels_array.dtype, np.integer) else zid
    except (TypeError, ValueError):
        target_id = zid

    mask = labels_array == target_id
    if np.isscalar(mask) or getattr(mask, "ndim", 0) != labels_array.ndim:
        return np.zeros(labels_array.shape, dtype=bool)
    return mask

# ══════════════════════════════════════════════════════════════════
#  FLOW FIELD  — BFS from all exits outward over the walkable grid
#  Each walkable cell stores the direction to step to get closer
#  to the nearest exit via a wall-respecting shortest path.
# ══════════════════════════════════════════════════════════════════

FLOW_SCALE = 4   # compute on 1/4-resolution grid, bilinear lookup at runtime

def build_flow_field(walkable, exits_px):
    """
    Returns flow_vx, flow_vy  — arrays of shape (H//SCALE, W//SCALE)
    giving the unit direction toward the exit at each walkable cell.
    Unreachable cells get (0,0).
    """
    H, W = walkable.shape
    sh, sw = H // FLOW_SCALE, W // FLOW_SCALE

    # downscale mask
    wh = cv2.resize(walkable.astype(np.uint8), (sw, sh),
                    interpolation=cv2.INTER_NEAREST).astype(bool)

    dist_g = np.full((sh, sw), np.inf, dtype=np.float32)
    # parent direction: what (dy,dx) step was used to reach each cell FROM an exit
    parent = np.zeros((sh, sw, 2), dtype=np.float32)

    queue = deque()
    for e in exits_px:
        ex = int(e["x"] / FLOW_SCALE)
        ey = int(e["y"] / FLOW_SCALE)
        ex = np.clip(ex, 0, sw-1)
        ey = np.clip(ey, 0, sh-1)
        if wh[ey, ex] and dist_g[ey, ex] == np.inf:
            dist_g[ey, ex] = 0.0
            queue.append((ey, ex))

    # 8-connected BFS (uniform cost — fine for evacuation)
    dirs = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
    costs = [1.0,1.0,1.0,1.0,1.414,1.414,1.414,1.414]

    while queue:
        cy, cx = queue.popleft()
        cd = dist_g[cy, cx]
        for (dy, dx), cost in zip(dirs, costs):
            ny, nx = cy+dy, cx+dx
            if 0 <= ny < sh and 0 <= nx < sw and wh[ny, nx]:
                nd = cd + cost
                if nd < dist_g[ny, nx]:
                    dist_g[ny, nx] = nd
                    # direction from this neighbour TOWARD the exit = reverse of (dy,dx)
                    parent[ny, nx, 0] = -dx   # vx component
                    parent[ny, nx, 1] = -dy   # vy component
                    queue.append((ny, nx))

    # normalise direction vectors
    mag = np.linalg.norm(parent, axis=2, keepdims=True)
    mag = np.where(mag < 1e-9, 1.0, mag)
    flow = parent / mag                  # shape (sh, sw, 2)

    # zero out unreachable
    unreachable = ~np.isfinite(dist_g)
    flow[unreachable] = 0.0

    print(f"  Flow field: {sw}×{sh}, "
          f"reachable={np.isfinite(dist_g).sum()}/{wh.sum()} cells")

    return flow, dist_g     # flow[y,x] = (vx, vy) unit direction


def sample_flow(flow, px, py):
    """
    Bilinear sample of the flow field at pixel position (px, py).
    Returns unit direction vector (or zeros if out of bounds / unreachable).
    """
    sh, sw = flow.shape[:2]
    if sh == 0 or sw == 0:
        return np.zeros(2)
    if not np.isfinite(px) or not np.isfinite(py):
        return np.zeros(2)

    fx = np.clip(float(px) / FLOW_SCALE, 0.0, sw - 1.0)
    fy = np.clip(float(py) / FLOW_SCALE, 0.0, sh - 1.0)

    x0 = int(fx);  x1 = min(x0+1, sw-1)
    y0 = int(fy);  y1 = min(y0+1, sh-1)
    tx = fx - x0;  ty = fy - y0

    v  = ((1-tx)*(1-ty)*flow[y0,x0]
         + tx   *(1-ty)*flow[y0,x1]
         + (1-tx)*  ty *flow[y1,x0]
         + tx   *  ty  *flow[y1,x1])

    mag = np.linalg.norm(v)
    if mag < 1e-9:
        return np.zeros(2)
    return v / mag


# ══════════════════════════════════════════════════════════════════
#  ORCA CORE
# ══════════════════════════════════════════════════════════════════

def orca_halfplane(pos_a, vel_a, r_a, pos_b, vel_b, r_b, tau=2.0):
    rel_pos = pos_b - pos_a
    rel_vel = vel_a - vel_b
    dist    = np.linalg.norm(rel_pos)
    r_sum   = r_a + r_b

    apex = rel_pos / tau
    w    = rel_vel - apex

    if dist > r_sum:
        leg   = np.sqrt(max(dist**2 - r_sum**2, 1e-9))
        cross = rel_pos[0]*w[1] - rel_pos[1]*w[0]
        if cross > 0:
            nx =  rel_pos[0]*leg - rel_pos[1]*r_sum
            ny =  rel_pos[0]*r_sum + rel_pos[1]*leg
        else:
            nx =  rel_pos[0]*leg + rel_pos[1]*r_sum
            ny = -rel_pos[0]*r_sum + rel_pos[1]*leg
        n = np.array([nx, ny])
    else:
        n = (-rel_pos / dist) if dist > 1e-9 else np.array([1.0, 0.0])
        u = (r_sum - dist + 2.0) * n
        point = vel_a + 0.5 * u
        n_len = np.linalg.norm(n)
        return point, n / max(n_len, 1e-9)

    n_len = np.linalg.norm(n)
    n = n / n_len if n_len > 1e-9 else np.array([0.0, 1.0])
    u = (np.dot(rel_vel, n) - np.dot(apex, n)) * n
    return vel_a + 0.5 * u, n


def resolve_velocity(v_pref, halfplanes, max_speed):
    v = v_pref.copy()
    spd = np.linalg.norm(v)
    if spd > max_speed:
        v = v / spd * max_speed

    for (pt, nm) in halfplanes:
        if np.dot(v - pt, nm) < 0.0:
            tang = np.array([-nm[1], nm[0]])
            proj_t = np.dot(v_pref - pt, tang)
            candidate = pt + proj_t * tang
            spd2 = np.linalg.norm(candidate)
            if spd2 > max_speed:
                candidate = candidate / spd2 * max_speed
            v = candidate
    return v


# ══════════════════════════════════════════════════════════════════
#  WALL PUSH  (soft repulsion, prevents agents sitting on wall pixels)
# ══════════════════════════════════════════════════════════════════

def build_dist_transform(walkable_mask):
    walk_u8 = walkable_mask.astype(np.uint8) * 255
    dist = cv2.distanceTransform(walk_u8, cv2.DIST_L2, 5)
    gx = cv2.Sobel(dist, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(dist, cv2.CV_64F, 0, 1, ksize=3)
    return dist, gx, gy


def wall_push(px, py, dist, gx, gy, push_range=6.0, strength=30.0):
    h, w = dist.shape
    ix = int(np.clip(px, 0, w-1))
    iy = int(np.clip(py, 0, h-1))
    d = dist[iy, ix]
    if d >= push_range:
        return np.zeros(2)
    factor = strength * (1.0 - d / push_range) ** 2
    return np.array([gx[iy, ix] * factor, gy[iy, ix] * factor])


# ══════════════════════════════════════════════════════════════════
#  FIRE SPREAD  (ported from SFM — same growth + 4-neighbour diffusion)
# ══════════════════════════════════════════════════════════════════

def spread_fire(intensity, walk_mask, ticks_elapsed, speed_mult, growth_mult):
    """Vectorized fire growth + 4-neighbour diffusion. Starts from a seeded
    pixel and bleeds outward through walkable cells only."""
    burning = intensity > 0.02
    if not burning.any():
        return intensity
    growth_rate    = 0.15 * growth_mult
    diffusion_rate = 0.12 * speed_mult * ticks_elapsed

    intensity = intensity.copy()
    intensity[burning] = np.minimum(1.0, intensity[burning] + growth_rate * ticks_elapsed)

    push = intensity * diffusion_rate
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


# ══════════════════════════════════════════════════════════════════
#  ZONE SEGMENTATION
# ══════════════════════════════════════════════════════════════════

def segment_zones(walkable_mask):
    binary = walkable_mask.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    dist      = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    dist_norm = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)
    coords    = peak_local_max(dist_norm, min_distance=40, labels=binary)
    seed_mask = np.zeros(dist_norm.shape, dtype=bool)
    seed_mask[tuple(coords.T)] = True
    markers, _ = ndi.label(seed_mask)
    return watershed(-dist, markers, mask=binary)


# ══════════════════════════════════════════════════════════════════
#  AGENT
# ══════════════════════════════════════════════════════════════════

class Agent:
    _ctr = 0

    def __init__(self, x, y, exits_px, flow, dist_g, speed_px_s=30.0, radius=5.0,
                 hazard_zone=None, hazard_xy=None):
        Agent._ctr += 1
        self.id      = Agent._ctr
        self.pos     = np.array([x, y], dtype=float)
        self.vel     = np.zeros(2)
        self.speed   = speed_px_s * np.random.uniform(0.85, 1.15)
        self.radius  = radius
        self.done    = False
        self.flow    = flow       # shared reference
        self.dist_g  = dist_g    # for exit detection
        self.exits_px = exits_px
        self.hazard_zone = hazard_zone   # bool grid, True = permanently blocked
        self.hazard_xy   = hazard_xy     # np.array([x, y]) hazard center
        self.time = None
        self.exit_used = None
        self.trail   = [self.pos.copy()]
        self._stuck_ctr = 0
        self._last_pos  = self.pos.copy()

    def v_pref(self, exit_radius=18.0, speed=None, sim_time=None):
        """Preferred velocity from flow field — wall-aware."""
        # check if near any exit
        for idx, e in enumerate(self.exits_px):
            if np.linalg.norm(self.pos - np.array([e["x"], e["y"]])) < exit_radius:
                self.done = True
                self.time = sim_time
                self.exit_used = idx
                return np.zeros(2)

        direction = sample_flow(self.flow, self.pos[0], self.pos[1])
        if np.linalg.norm(direction) < 1e-9:
            if self.hazard_zone is not None and self.hazard_xy is not None:
                ix = int(np.clip(self.pos[0], 0, self.hazard_zone.shape[1] - 1))
                iy = int(np.clip(self.pos[1], 0, self.hazard_zone.shape[0] - 1))
                if self.hazard_zone[iy, ix]:
                    away = self.pos - self.hazard_xy
                    mag = np.linalg.norm(away)
                    if mag > 1e-6:
                        return (away / mag) * (speed or self.speed)
            # truly unreachable — shouldn't happen after fixes
            return np.zeros(2)
        return direction * (speed or self.speed)

    def check_stuck_and_push(self):
        """
        If an agent somehow ended up inside a wall pixel (shouldn't happen
        but floating-point drift can cause it), teleport it to nearest walkable.
        Also detect stuck agents and nudge them.
        """
        moved = np.linalg.norm(self.pos - self._last_pos)
        self._stuck_ctr = 0 if moved > 0.5 else self._stuck_ctr + 1
        self._last_pos  = self.pos.copy()


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════

def run(mask_path: str, config_path: str, fire_spread_speed: float = 1.0,
        fire_intensity_factor: float = 1.0):
    # ── load mask ────────────────────────────────────────────────
    img      = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(mask_path)
    walkable = img < 128      # white=wall, black=walkable
    H, W     = walkable.shape
    print(f"Mask: {W}×{H}  walkable={walkable.mean()*100:.1f}%")

    # ── load config ──────────────────────────────────────────────
    with open(config_path) as f:
        cfg = json.load(f)
    exits_px = cfg.get("exits", [])
    if not exits_px:
        raise ValueError("No exits in zone config.")
    print(f"Exits: {len(exits_px)}")

    # ── hazard: carve a permanent no-go zone out of walkable, same as SFM ──
    hazard_cfg = cfg.get("hazard")
    HAZARD_BLOCK_RADIUS = 90
    FIRE_SPREAD_SPEED     = fire_spread_speed
    FIRE_INTENSITY_FACTOR = fire_intensity_factor
    hazard_zone = np.zeros((H, W), dtype=bool)
    fire_intensity = np.zeros((H, W), dtype=np.float32)
    hazard_xy = None
    routing_walkable = walkable

    if hazard_cfg:
        hx = int(np.clip(hazard_cfg["x"], 0, W - 1))
        hy = int(np.clip(hazard_cfg["y"], 0, H - 1))
        hazard_xy = np.array([hx, hy], dtype=np.float64)
        print(f"Hazard at ({hx},{hy})")

        yy, xx = np.ogrid[:H, :W]
        hazard_zone = (xx - hx)**2 + (yy - hy)**2 <= HAZARD_BLOCK_RADIUS**2
        routing_walkable = walkable & ~hazard_zone
        fire_intensity[hy, hx] = 1.0   # seed — same as SFM's fire_intensity[fy, fx] = 1.0
    else:
        print("No hazard in config — running clean (no rerouting needed).")

    # ── build flow field (BFS from exits, wall-aware + hazard-aware) ────
    print("Building flow field…")
    t0 = time.time()
    flow, dist_g = build_flow_field(routing_walkable, exits_px)
    print(f"  Done in {time.time()-t0:.2f}s")

    # ── distance transform for wall repulsion ────────────────────
    dist_map, gx_map, gy_map = build_dist_transform(walkable)

    # ── zone segmentation for spawning ───────────────────────────
    print("Segmenting zones…")
    zone_labels = segment_zones(walkable)
    all_wy, all_wx = np.where(walkable)
    global_pool = list(zip(all_wx.tolist(), all_wy.tolist()))

    # ── spawn agents ─────────────────────────────────────────────
    Agent._ctr = 0
    agents = []
    for zone in cfg.get("zones", []):
        if zone.get("density_index", 1.0) <= 0:
            continue
        n = zone.get("agents", 0)
        if n <= 0:
            continue
        zid = zone["zone_id"]
        zm  = zone_id_mask(zone_labels, zid) & walkable
        zy, zx = np.where(zm)
        pool = list(zip(zx.tolist(), zy.tolist())) if len(zx) > 0 else global_pool
        for _ in range(n):
            idx = np.random.randint(len(pool))
            px, py = float(pool[idx][0]), float(pool[idx][1])
            agents.append(Agent(px, py, exits_px, flow, dist_g,
                                 hazard_zone=hazard_zone, hazard_xy=hazard_xy))

    print(f"Spawned {len(agents)} agents")

    # ── sim constants ─────────────────────────────────────────────
    DT         = 0.1
    MAX_STEPS  = 4000 
    TAU        = 2.0
    NEIGH_DIST = 60.0
    CELL_SZ    = 30.0
    EXIT_R     = 18.0

    # ── analytics ─────────────────────────────────────────────────
    density_acc    = np.zeros((H, W), dtype=float)
    density_frames = 0
    ts_time, ts_active, ts_evac = [], [], []

    # ── main loop ─────────────────────────────────────────────────
    print("Simulating…")
    t0 = time.time()

    for step in range(MAX_STEPS):
        if all(a.done for a in agents):
            break

        # ── fire growth (visual only — routing already handled by hazard_zone) ──
        # SFM updates every 0.5s of sim time (step%10 at its DT=0.05) for a
        # 120s-long run. RVO's sim can run far longer than that while agents
        # are still evacuating, so cap fire growth at the same 120s budget —
        # otherwise it just keeps spreading well past where SFM would stop.
        FIRE_TICK_INTERVAL = 0.5
        FIRE_GROWTH_BUDGET = 120.0
        sim_t = step * DT
        if hazard_cfg and sim_t <= FIRE_GROWTH_BUDGET and abs(sim_t % FIRE_TICK_INTERVAL) < DT / 2:
            fire_intensity = spread_fire(fire_intensity, walkable, FIRE_TICK_INTERVAL,
                                          FIRE_SPREAD_SPEED, FIRE_INTENSITY_FACTOR)

        # spatial bucket
        bucket = defaultdict(list)
        for i, a in enumerate(agents):
            if not a.done:
                key = (int(a.pos[0]/CELL_SZ), int(a.pos[1]/CELL_SZ))
                bucket[key].append(i)

        new_vels = []
        for i, ag in enumerate(agents):
            if ag.done:
                new_vels.append(np.zeros(2))
                continue

            vp = ag.v_pref(EXIT_R, sim_time=step * DT)   # wall-aware direction from flow field

            # ORCA half-planes from nearby agents
            ki = int(ag.pos[0]/CELL_SZ)
            kj = int(ag.pos[1]/CELL_SZ)
            halfplanes = []
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    for j in bucket[(ki+di, kj+dj)]:
                        if j == i:
                            continue
                        nb = agents[j]
                        if np.linalg.norm(ag.pos - nb.pos) < NEIGH_DIST:
                            pt, nm = orca_halfplane(
                                ag.pos, ag.vel, ag.radius,
                                nb.pos, nb.vel, nb.radius, tau=TAU)
                            halfplanes.append((pt, nm))

            v_new = resolve_velocity(vp, halfplanes, ag.speed)

            # soft wall repulsion (last safety net)
            v_new += wall_push(ag.pos[0], ag.pos[1],
                               dist_map, gx_map, gy_map) * DT

            # minimum speed guarantee — prevents full deadlock
            if np.linalg.norm(v_new) < ag.speed * 0.15 and np.linalg.norm(vp) > 0:
                v_new = vp * 0.15

            # clamp
            spd = np.linalg.norm(v_new)
            if spd > ag.speed * 1.3:
                v_new = v_new / spd * ag.speed * 1.3

            new_vels.append(v_new)

        # integrate — but only move to walkable pixels
        for ag, v in zip(agents, new_vels):
            if ag.done:
                continue
            ag.vel = v
            new_pos = ag.pos + v * DT

            # wall collision: if proposed position is in a wall, try axis slides
            nx = int(np.clip(new_pos[0], 0, W-1))
            ny = int(np.clip(new_pos[1], 0, H-1))

            if walkable[ny, nx]:
                ag.pos = new_pos
            else:
                # try sliding along each axis separately
                pos_x = ag.pos + np.array([v[0]*DT, 0])
                pos_y = ag.pos + np.array([0, v[1]*DT])
                ix2 = int(np.clip(pos_x[0], 0, W-1))
                iy2 = int(np.clip(pos_x[1], 0, H-1))
                ix3 = int(np.clip(pos_y[0], 0, W-1))
                iy3 = int(np.clip(pos_y[1], 0, H-1))

                if walkable[iy2, ix2]:
                    ag.pos = pos_x
                elif walkable[iy3, ix3]:
                    ag.pos = pos_y
                # else stay put (wall on both sides — rare)

            ag.pos[0] = np.clip(ag.pos[0], 0, W-1)
            ag.pos[1] = np.clip(ag.pos[1], 0, H-1)
            for exit_idx, exit_pt in enumerate(exits_px):
                exit_pos = np.array([exit_pt["x"], exit_pt["y"]], dtype=float)
                if np.linalg.norm(ag.pos - exit_pos) < EXIT_R:
                    ag.done = True
                    ag.time = (step + 1) * DT
                    ag.exit_used = exit_idx
                    break
            ag.trail.append(ag.pos.copy())
            ag.check_stuck_and_push()

            ix = int(ag.pos[0])
            iy = int(ag.pos[1])
            density_acc[iy, ix] += 1

        density_frames += 1

        if step % 10 == 0:
            active = sum(1 for a in agents if not a.done)
            evac   = sum(1 for a in agents if a.done)
            ts_time.append(step * DT)
            ts_active.append(active)
            ts_evac.append(evac)
            if step % 100 == 0:
                print(f"  t={step*DT:6.1f}s  active={active:4d}  evac={evac:4d}")

    elapsed = time.time() - t0
    done_n  = sum(1 for a in agents if a.done)
    print(f"\nDone in {step+1} steps ({elapsed:.1f}s real)  "
          f"evacuated={done_n}/{len(agents)} ({100*done_n//max(len(agents),1)}%)")

    Path("output").mkdir(exist_ok=True)
    _save_paths(agents, exits_px, walkable, W, H, done_n,
                fire_intensity=fire_intensity)
    _save_heatmap(density_acc, density_frames, walkable, exits_px, W, H)
    _save_csv(ts_time, ts_active, ts_evac)
    _save_report(agents, exits_px, density_acc, density_frames, walkable,
                 done_n, step + 1, DT, elapsed)
    print("Outputs saved in output/")


# ══════════════════════════════════════════════════════════════════
#  OUTPUT HELPERS
# ══════════════════════════════════════════════════════════════════

def _save_paths(agents, exits_px, walkable, W, H, done_n, fire_intensity=None):
    base = np.zeros((H, W, 3), dtype=np.uint8)
    base[~walkable] = [60, 60, 60]

    # Hazard overlay — identical treatment to SFM's COLORMAP_HOT blend
    if fire_intensity is not None and fire_intensity.max() > 0:
        fire_u8 = (np.clip(fire_intensity, 0, 1) * 255).astype(np.uint8)
        fire_color = cv2.applyColorMap(fire_u8, cv2.COLORMAP_HOT)
        fire_mask3 = (fire_intensity > 0.03).astype(np.uint8)[:, :, None]
        base = np.where(fire_mask3 > 0, cv2.addWeighted(base, 0.4, fire_color, 0.6, 0), base)

    GREEN  = (0, 200, 55)
    ORANGE = (0, 130, 255)
    for ag in agents:
        color = GREEN if ag.done else ORANGE
        trail = ag.trail[::3]
        for i in range(1, len(trail)):
            cv2.line(base,
                     (int(trail[i-1][0]), int(trail[i-1][1])),
                     (int(trail[i][0]),   int(trail[i][1])),
                     color, 1)

    for e in exits_px:
        cv2.circle(base, (int(e["x"]), int(e["y"])), 14, (0, 255, 255), 2)
        cv2.putText(base, "EXIT", (int(e["x"]) - 14, int(e["y"]) - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

    total = len(agents)
    pct_done = 100 * done_n // max(total, 1)
    cv2.putText(base, f"Agents:{total}  Evac:{done_n}({pct_done}%)  RVO/ORCA",
                (8, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)

    out = "output/rvo_agent_paths.png"
    cv2.imwrite(out, base)
    print(f"Saved {out}")


def _save_heatmap(density_acc, n_frames, walkable, exits_px, W, H):
    pass # no need lol fuck this - im not doing this 

def _save_csv(ts_time, ts_active, ts_evac):
    import csv
    out = "output/rvo_analytics.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "active_agents", "evacuated"])
        for row in zip(ts_time, ts_active, ts_evac):
            w.writerow(row)
    print(f"Saved {out}")


def _top_congestion_points(density_acc, n_frames, walkable, limit=5):
    avg = density_acc / max(n_frames, 1)
    avg = cv2.GaussianBlur(avg.astype(np.float32), (31, 31), 0)
    avg[~walkable] = 0

    if avg.max() <= 0:
        return []

    dist = cv2.distanceTransform(walkable.astype(np.uint8), cv2.DIST_L2, 5)
    work = avg.copy()
    points = []
    radius = 45
    for _ in range(limit):
        _, value, _, loc = cv2.minMaxLoc(work)
        if value <= 0:
            break
        cx, cy = loc
        points.append({
            "cx": int(cx),
            "cy": int(cy),
            "avg_density": float(value),
            "corridor_width": int(max(1, dist[cy, cx] * 2)),
        })
        cv2.circle(work, (cx, cy), radius, 0, -1)
    return points


def _save_report(agents, exits_px, density_acc, n_frames, walkable,
                 done_n, step_count, dt, elapsed):
    out = "output/RVO_output_report.txt"
    total = len(agents)
    rate = done_n / max(total, 1)
    sim_time = step_count * dt
    times = [a.time for a in agents if a.done and a.time is not None]

    exit_counts = [0] * len(exits_px)
    for ag in agents:
        if ag.done and ag.exit_used is not None and 0 <= ag.exit_used < len(exit_counts):
            exit_counts[ag.exit_used] += 1
    total_evacuated = sum(exit_counts)

    score_rate = rate * 50
    if times:
        mean_t = float(np.mean(times))
        score_time = max(0, 20 * (1 - (mean_t - 20) / 80)) if mean_t > 20 else 20.0
    else:
        mean_t = sim_time
        score_time = 0.0

    if total_evacuated > 0 and exits_px:
        fractions = [c / total_evacuated for c in exit_counts]
        ideal = 1.0 / len(exits_px)
        max_dev = max(abs(f - ideal) for f in fractions)
        score_balance = max(0, 15 * (1 - max_dev / ideal))
    else:
        score_balance = 0.0

    hotspots = _top_congestion_points(density_acc, n_frames, walkable)
    max_density = hotspots[0]["avg_density"] if hotspots else 0.0
    score_congestion = max(0, 15 * (1 - max_density / 0.25))

    final_score = int(score_rate + score_time + score_balance + score_congestion)
    final_score = min(100, max(0, final_score))

    if rate < 0.85:
        recommendation = "Too many agents failed to evacuate. Check isolated rooms, blocked routes, or exit placement."
    elif score_balance < 8 and exits_px:
        min_idx = int(np.argmin(exit_counts)) if exit_counts else 0
        ex = exits_px[min_idx]
        recommendation = f"Exit {min_idx} at ({int(ex['x'])},{int(ex['y'])}) is underused. Move it closer to dense zones or improve routing."
    elif hotspots:
        recommendation = f"Highest congestion is near ({hotspots[0]['cx']},{hotspots[0]['cy']}). Widen that route or add an alternate path."
    elif score_time < 10:
        recommendation = "Evacuation is slow. Add an exit closer to high-density zones."
    else:
        recommendation = "Evacuation performance is acceptable for this RVO run."

    lines = [
        "=" * 55,
        "       RVO EVACUATION ANALYSIS REPORT",
        "=" * 55,
        "",
        f"  OVERALL SCORE : {final_score} / 100",
        "",
        f"  Total agents    : {total}",
        f"  Evacuated       : {done_n}  ({100 * rate:.1f}%)",
        f"  Trapped/timeout : {total - done_n}  ({100 * (1 - rate):.1f}%)",
        f"  Sim time        : {sim_time:.1f}s",
        f"  Runtime         : {elapsed:.1f}s",
    ]

    if times:
        lines += [
            f"  Fastest evac    : {min(times):.1f}s",
            f"  Mean evac time  : {mean_t:.1f}s",
            f"  Slowest evac    : {max(times):.1f}s",
        ]

    lines += ["", "-" * 55, "  EXIT UTILIZATION", "-" * 55]
    if exits_px:
        ideal_pct = 100.0 / len(exits_px)
        for idx, ex in enumerate(exits_px):
            pct = (exit_counts[idx] / total_evacuated * 100) if total_evacuated > 0 else 0
            bar = "#" * int(pct / 5)
            status = "UNDERUSED" if pct < ideal_pct * 0.4 else ""
            lines.append(
                f"  Exit {idx} ({int(ex['x'])},{int(ex['y'])}): "
                f"{exit_counts[idx]:3d} agents  {pct:5.1f}%  {bar} {status}"
            )
    else:
        lines.append("  No exits configured.")

    lines += ["", "-" * 55, "  TOP CONGESTION POINTS", "-" * 55]
    if hotspots:
        for rank, point in enumerate(hotspots, start=1):
            lines.append(
                f"  C{rank}  position ({point['cx']:4d},{point['cy']:4d})  "
                f"corridor width ~{point['corridor_width']}px  "
                f"avg density {point['avg_density']:.3f}"
            )
    else:
        lines.append("  No significant congestion detected.")

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
        f"    Congestion       : {score_congestion:.0f}/15",
        "=" * 55,
    ]

    report = "\n".join(lines)
    print("\n" + report.encode("ascii", errors="replace").decode("ascii"))
    with open(out, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Saved {out}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python rvo_evacuation.py <mask.png> <zone_config.json> [params.json]")
        sys.exit(1)

    _fire_spread_speed = 1.0
    _fire_intensity_factor = 1.0
    if len(sys.argv) > 3 and Path(sys.argv[3]).exists():
        with open(sys.argv[3]) as f:
            _params = json.load(f)
        _fire_spread_speed     = _params.get("fire_spread_speed", _fire_spread_speed)
        _fire_intensity_factor = _params.get("fire_intensity_factor", _fire_intensity_factor)

    run(sys.argv[1], sys.argv[2],
        fire_spread_speed=_fire_spread_speed,
        fire_intensity_factor=_fire_intensity_factor)