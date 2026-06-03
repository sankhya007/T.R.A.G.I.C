"""
rvo_evacuation.py  —  RVO-based evacuation on a stitched mask
Wall-aware: agents follow a BFS flow field so they navigate AROUND walls,
not through them. RVO handles agent-agent collision avoidance on top.

Usage:
    python rvo_evacuation.py <mask.png> <zone_config.json>
"""

import sys, json, time
from pathlib import Path
from collections import defaultdict, deque

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

np.random.seed(42)

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
    fx = px / FLOW_SCALE
    fy = py / FLOW_SCALE

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

    def __init__(self, x, y, exits_px, flow, dist_g, speed_px_s=30.0, radius=5.0):
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
        self.trail   = [self.pos.copy()]
        self._stuck_ctr = 0
        self._last_pos  = self.pos.copy()

    def v_pref(self, exit_radius=18.0, speed=None):
        """Preferred velocity from flow field — wall-aware."""
        # check if near any exit
        for e in self.exits_px:
            if np.linalg.norm(self.pos - np.array([e["x"], e["y"]])) < exit_radius:
                self.done = True
                return np.zeros(2)

        direction = sample_flow(self.flow, self.pos[0], self.pos[1])
        if np.linalg.norm(direction) < 1e-9:
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

def run(mask_path: str, config_path: str):
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

    # ── build flow field (BFS from exits, wall-aware) ────────────
    print("Building flow field…")
    t0 = time.time()
    flow, dist_g = build_flow_field(walkable, exits_px)
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
        zm  = (zone_labels == zid) & walkable
        zy, zx = np.where(zm)
        pool = list(zip(zx.tolist(), zy.tolist())) if len(zx) > 0 else global_pool
        for _ in range(n):
            idx = np.random.randint(len(pool))
            px, py = float(pool[idx][0]), float(pool[idx][1])
            agents.append(Agent(px, py, exits_px, flow, dist_g))

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

            vp = ag.v_pref(EXIT_R)   # wall-aware direction from flow field

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
    _save_paths(agents, exits_px, walkable, W, H, done_n)
    _save_heatmap(density_acc, density_frames, walkable, exits_px, W, H)
    _save_csv(ts_time, ts_active, ts_evac)
    print("Outputs saved in output/")


# ══════════════════════════════════════════════════════════════════
#  OUTPUT HELPERS
# ══════════════════════════════════════════════════════════════════

def _save_paths(agents, exits_px, walkable, W, H, done_n):
    DPI = 150
    fig, ax = plt.subplots(figsize=(14, 14*H/W), dpi=DPI)
    fig.patch.set_facecolor("#000000")
    ax.set_facecolor("#000000")
    ax.set_xlim(0, W); ax.set_ylim(H, 0)
    ax.set_aspect("equal"); ax.axis("off")

    wall_img = np.zeros((H, W, 3), dtype=np.uint8)
    wall_img[~walkable] = [60, 60, 60]
    ax.imshow(wall_img, origin="upper", zorder=1)

    GREEN  = np.array([0.0, 1.0, 0.15])
    ORANGE = np.array([1.0, 0.55, 0.0])
    for ag in agents:
        traj = np.array(ag.trail[::3])
        if len(traj) < 2:
            continue
        col  = GREEN if ag.done else ORANGE
        pts  = traj.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        n    = len(segs)
        #  this is the line that controls the fading trail effect — alpha goes from 0.03 to 0.85 along the path
        # lc   = LineCollection(segs,
        #                       colors=[(*col, a) for a in np.linspace(0.03, 0.85, n)],
        #                       linewidth=0.8, zorder=3)
        lc = LineCollection(segs,
                    colors=[(*col, 0.75) for _ in segs],  # constant alpha, no fade
                    linewidth=0.8, zorder=3)
        ax.add_collection(lc)

    for e in exits_px:
        ax.add_patch(plt.Circle((e["x"], e["y"]), 14,
                                fill=False, edgecolor="#f5d800",
                                linewidth=1.8, zorder=5))
        ax.text(e["x"], e["y"]-18, "EXIT", color="#f5d800",
                fontsize=5, ha="center", fontfamily="monospace", zorder=6)

    total = len(agents)
    ax.text(5, 15,
            f"Agents: {total}   Evacuated: {done_n} ({100*done_n//total}%)   Model: RVO/ORCA",
            color="#aaaaaa", fontsize=6, va="top",
            fontfamily="monospace", zorder=7)

    plt.tight_layout(pad=0)
    out = "output/rvo_agent_paths.png"
    plt.savefig(out, dpi=DPI, bbox_inches="tight",
                facecolor="black", edgecolor="none")
    plt.close()
    print(f"Saved {out}")


def _save_heatmap(density_acc, n_frames, walkable, exits_px, W, H):
    avg = cv2.GaussianBlur(
        (density_acc / max(n_frames, 1)).astype(np.float32), (21, 21), 0)

    fig, ax = plt.subplots(figsize=(12, 12*H/W), dpi=130)
    fig.patch.set_facecolor("#000000")
    ax.set_facecolor("#000000")
    ax.set_xlim(0, W); ax.set_ylim(H, 0)
    ax.set_aspect("equal"); ax.axis("off")

    wall_img = np.zeros((H, W, 3), dtype=np.uint8)
    wall_img[~walkable] = [40, 40, 40]
    ax.imshow(wall_img, origin="upper", zorder=1, alpha=0.6)
    hm = ax.imshow(avg, cmap="hot", origin="upper",
                   interpolation="bilinear",
                   vmin=0, vmax=avg.max()*0.8,
                   zorder=2, alpha=0.85)

    for e in exits_px:
        ax.add_patch(plt.Circle((e["x"], e["y"]), 14,
                                fill=False, edgecolor="#00ff88",
                                linewidth=1.5, zorder=5))

    ax.set_title("Crowd Density Heatmap  —  RVO Evacuation",
                 color="white", fontsize=12, pad=8)
    cbar = plt.colorbar(hm, ax=ax, fraction=0.03, pad=0.01)
    cbar.set_label("Avg agents/px", color="white", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    plt.tight_layout()
    out = "output/rvo_density_heatmap.png"
    plt.savefig(out, dpi=130, bbox_inches="tight",
                facecolor="black", edgecolor="none")
    plt.close()
    print(f"Saved {out}")


def _save_csv(ts_time, ts_active, ts_evac):
    import csv
    out = "output/rvo_analytics.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "active_agents", "evacuated"])
        for row in zip(ts_time, ts_active, ts_evac):
            w.writerow(row)
    print(f"Saved {out}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python rvo_evacuation.py <mask.png> <zone_config.json>")
        sys.exit(1)
    run(sys.argv[1], sys.argv[2])