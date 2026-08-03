import json, math, random, sys, warnings
import numpy as np
import cv2
from pathlib import Path
from collections import deque
import heapq
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from security_utils import (
    HAZARD_BLOCK_RADIUS, WATERSHED_MIN_DISTANCE, load_runtime_params,
    load_zone_config, output_path, validate_image_path,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)

# 
CFG = {
    "mask_path":    "",
    "zone_config": "zone_config.json",
    "output":      str(output_path("continuum_agent_paths.png")),

    "DT":          0.05,
    "MAX_TIME":    40,
    "speed_px_s":  150.0,
    "exit_radius": 22,

    "grid_res":    4,
    "alpha": 0.3,
    "beta":  0.7,

    "rho_min": 0.05,
    "rho_max": 0.40,
    "density_radius": 6,

    "agent_radius":  6,
    "repulse_str":   200.0,
    "repulse_range": 14.0,
    "relax_time":    0.3,

    "wall_color":   (85, 85, 85),
    "exit_color":   (0, 200, 220),
    "additive_scale": 8,

    # fire spread — same meaning as in SFM/RVO
    "fire_spread_speed":     1.0,   # multiplier on diffusion rate
    "fire_intensity_factor": 1.0,   # multiplier on growth-to-saturation rate
    "hazard_block_radius":   HAZARD_BLOCK_RADIUS,
}


def apply_runtime_args():
    """Allow the launcher to provide the active mask, config, and UI params."""
    if len(sys.argv) > 1:
        CFG["mask_path"] = sys.argv[1]
    if len(sys.argv) > 2:
        CFG["zone_config"] = sys.argv[2]
    if len(sys.argv) > 3:
        params_path = Path(sys.argv[3])
        if params_path.exists():
            params = load_runtime_params(params_path, set(CFG) - {"mask_path", "zone_config", "output", "wall_color", "exit_color"}, ignore_unknown=True)
            for key, value in params.items():
                if key in CFG:
                    CFG[key] = value


def zone_id_mask(labels_array, zid):
    try:
        target_id = int(zid) if np.issubdtype(labels_array.dtype, np.integer) else zid
    except (TypeError, ValueError):
        target_id = zid

    mask = labels_array == target_id
    if np.isscalar(mask) or getattr(mask, "ndim", 0) != labels_array.ndim:
        return np.zeros(labels_array.shape, dtype=bool)
    return mask


def spread_fire(intensity, walk_mask, ticks_elapsed, speed_mult, growth_mult):
    """Vectorized fire growth + 4-neighbour diffusion — identical to SFM/RVO.
    Visual + soft-push only; routing around the hazard is handled separately
    and permanently via the hazard_zone carved into ContGrid.walkable."""
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


class WalkMap:
    def __init__(self, path):
        image_path = validate_image_path(path)
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(path)
        self.walkable = img < 128
        self.h, self.w = img.shape
        self.raw = img

    def ok(self, x, y):
        ix, iy = int(round(x)), int(round(y))
        return 0 <= ix < self.w and 0 <= iy < self.h and self.walkable[iy, ix]


class ContGrid:
    def __init__(self, wm, exits_px, res, hazard_zone=None):
        self.res = res
        self.gw  = math.ceil(wm.w / res)
        self.gh  = math.ceil(wm.h / res)
        self.wm  = wm

        # Downsample walkable mask with a single cv2.resize call instead of
        # two nested Python loops — same nearest-neighbour logic, much faster.
        self.walkable = cv2.resize(
            wm.walkable.astype(np.uint8), (self.gw, self.gh),
            interpolation=cv2.INTER_NEAREST).astype(bool)

        if hazard_zone is not None:
            hz_down = cv2.resize(
                hazard_zone.astype(np.uint8), (self.gw, self.gh),
                interpolation=cv2.INTER_NEAREST).astype(bool)
            self.walkable &= ~hz_down

        self.exits = exits_px
        self.rho   = np.zeros((self.gh, self.gw), dtype=np.float32)
        self.vavg  = np.zeros((self.gh, self.gw, 2), dtype=np.float32)
        self.phi   = np.full((self.gh, self.gw), np.inf, dtype=np.float64)
        self.bfs   = np.full((self.gh, self.gw), np.inf, dtype=np.float64)
        self._bfs(exits_px)

    def p2g(self, px, py):
        return (int(np.clip(px / self.res, 0, self.gw - 1)),
                int(np.clip(py / self.res, 0, self.gh - 1)))

    def _bfs(self, exits_px):
        q = deque()
        for ex, ey in exits_px:
            gx, gy = self.p2g(ex, ey)
            for dy in range(-4, 5):
                for dx in range(-4, 5):
                    nx, ny = gx + dx, gy + dy
                    if (0 <= ny < self.gh and 0 <= nx < self.gw
                            and self.walkable[ny, nx]
                            and self.bfs[ny, nx] == np.inf):
                        self.bfs[ny, nx] = 0
                        q.append((nx, ny))
        while q:
            cx, cy = q.popleft()
            for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                nx, ny = cx + dx, cy + dy
                if (0 <= ny < self.gh and 0 <= nx < self.gw
                        and self.walkable[ny, nx]
                        and self.bfs[ny, nx] == np.inf):
                    self.bfs[ny, nx] = self.bfs[cy, cx] + 1
                    q.append((nx, ny))

    def splat(self, agents):
        self.rho[:]  = 0
        self.vavg[:] = 0
        r = max(1, CFG["density_radius"] // self.res)

        # Build weight kernel once (Manhattan-distance tent, same as before)
        ky, kx = np.mgrid[-r:r+1, -r:r+1]
        kernel = np.maximum(0, 1 - (np.abs(ky) + np.abs(kx)) / (r + 1)).astype(np.float32)

        active = [(a["x"], a["y"], a["vx"], a["vy"]) for a in agents if not a["done"]]
        if not active:
            return

        for ax, ay, avx, avy in active:
            gx, gy = self.p2g(ax, ay)
            # grid-cell range clipped to array bounds
            y0, y1 = max(0, gy - r), min(self.gh, gy + r + 1)
            x0, x1 = max(0, gx - r), min(self.gw, gx + r + 1)
            # corresponding kernel slice
            ky0, ky1 = y0 - (gy - r), y1 - (gy - r)
            kx0, kx1 = x0 - (gx - r), x1 - (gx - r)
            w = kernel[ky0:ky1, kx0:kx1]
            self.rho[y0:y1, x0:x1]       += w
            self.vavg[y0:y1, x0:x1, 0]   += w * avx
            self.vavg[y0:y1, x0:x1, 1]   += w * avy

        m = self.rho > 0
        self.vavg[m, 0] /= self.rho[m]
        self.vavg[m, 1] /= self.rho[m]

    def _speed(self, gx, gy, dx, dy):
        rho = self.rho[gy, gx]
        ft  = CFG["speed_px_s"]
        d   = math.hypot(dx, dy)
        if d < 1e-6:
            fv = CFG["speed_px_s"] * 0.2
        else:
            nx_, ny_ = dx / d, dy / d
            fv = (self.vavg[gy, gx, 0] * nx_ + self.vavg[gy, gx, 1] * ny_)
            fv = max(CFG["speed_px_s"] * 0.1, fv)
        rn, rx = CFG["rho_min"], CFG["rho_max"]
        if rho <= rn: return ft
        if rho >= rx: return fv
        t = (rho - rn) / (rx - rn)
        return ft + t * (fv - ft)

    def _cost(self, gx, gy, dx, dy):
        f = max(1.0, self._speed(gx, gy, dx, dy))
        return (CFG["alpha"] * f + CFG["beta"]) / f

    def build_phi(self):
        phi     = np.full((self.gh, self.gw), np.inf, dtype=np.float64)
        visited = np.zeros((self.gh, self.gw), dtype=bool)
        hp      = []
        for ex, ey in self.exits:
            gx, gy = self.p2g(ex, ey)
            for dy in range(-4, 5):
                for dx in range(-4, 5):
                    nx, ny = gx + dx, gy + dy
                    if (0 <= ny < self.gh and 0 <= nx < self.gw
                            and self.walkable[ny, nx]
                            and phi[ny, nx] == np.inf):
                        phi[ny, nx] = 0
                        heapq.heappush(hp, (0.0, nx, ny))
        DIRS = [(1,0),(-1,0),(0,1),(0,-1)]
        while hp:
            val, cx, cy = heapq.heappop(hp)
            if visited[cy, cx]:
                continue
            visited[cy, cx] = True
            phi[cy, cx] = val
            for dx, dy in DIRS:
                nx, ny = cx + dx, cy + dy
                if not (0 <= ny < self.gh and 0 <= nx < self.gw):
                    continue
                if visited[ny, nx] or not self.walkable[ny, nx]:
                    continue
                cand = val + self._cost(cx, cy, dx, dy)
                if cand < phi[ny, nx]:
                    phi[ny, nx] = cand
                    heapq.heappush(hp, (cand, nx, ny))
        self.phi = phi

    def grad_at(self, px, py):
        gx = int(np.clip(px / self.res, 1, self.gw - 2))
        gy = int(np.clip(py / self.res, 1, self.gh - 2))
        if not self.walkable[gy, gx]:
            return 0.0, 0.0

        phi_e = self.phi[gy, gx + 1]
        phi_w = self.phi[gy, gx - 1]
        phi_s = self.phi[gy + 1, gx]
        phi_n = self.phi[gy - 1, gx]

        dpx = (phi_e - phi_w) / 2 if (np.isfinite(phi_e) and np.isfinite(phi_w)) else np.nan
        dpy = (phi_s - phi_n) / 2 if (np.isfinite(phi_s) and np.isfinite(phi_n)) else np.nan

        if np.isfinite(dpx) and np.isfinite(dpy):
            mag = math.hypot(dpx, dpy)
            if mag > 1e-6:
                return -dpx / mag, -dpy / mag

        b_e = self.bfs[gy, min(gx + 1, self.gw - 1)]
        b_w = self.bfs[gy, max(gx - 1, 0)]
        b_s = self.bfs[min(gy + 1, self.gh - 1), gx]
        b_n = self.bfs[max(gy - 1, 0), gx]
        dpx = (b_e - b_w) / 2 if (np.isfinite(b_e) and np.isfinite(b_w)) else 0.0
        dpy = (b_s - b_n) / 2 if (np.isfinite(b_s) and np.isfinite(b_n)) else 0.0
        mag = math.hypot(dpx, dpy)
        if mag > 1e-6:
            return -dpx / mag, -dpy / mag
        return 0.0, 0.0


def rebuild_labels(mask_path):
    img      = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    walkable = cv2.bitwise_not(img)
    _, binary = cv2.threshold(walkable, 127, 255, cv2.THRESH_BINARY)
    k      = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)
    dist   = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    dn     = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)
    coords = peak_local_max(dn, min_distance=WATERSHED_MIN_DISTANCE, labels=binary)
    sm     = np.zeros(dn.shape, dtype=bool)
    sm[tuple(coords.T)] = True
    markers, _ = ndi.label(sm)
    return watershed(-dist, markers, mask=binary)


def main():
    apply_runtime_args()
    cfg_path = CFG["zone_config"]
    if not Path(cfg_path).exists():
        print(f"ERROR: {cfg_path} not found"); sys.exit(1)

    zcfg = load_zone_config(cfg_path)

    mask_path = CFG.get("mask_path") or zcfg.get("mask_path", "")
    if not Path(mask_path).exists():
        print(f"ERROR: mask not found at {mask_path}"); sys.exit(1)

    print("Loading mask...")
    wm = WalkMap(mask_path)

    exits_raw = zcfg.get("exits", [])
    if not exits_raw:
        print("ERROR: no exits in config"); sys.exit(1)
    exits_px = [(int(e["x"]), int(e["y"])) for e in exits_raw]
    print(f"  {len(exits_px)} exits")

    #  hazard: carve a permanent no-go zone 
    hazard_cfg = zcfg.get("hazard")
    hazard_zone = np.zeros((wm.h, wm.w), dtype=bool)
    fire_intensity = np.zeros((wm.h, wm.w), dtype=np.float32)
    hazard_xy = None

    if hazard_cfg:
        hx = int(np.clip(hazard_cfg["x"], 0, wm.w - 1))
        hy = int(np.clip(hazard_cfg["y"], 0, wm.h - 1))
        hazard_xy = np.array([hx, hy], dtype=np.float64)
        fire_intensity[hy, hx] = 1.0
        print(f"Hazard ignited at ({hx},{hy})")

        yy, xx = np.ogrid[:wm.h, :wm.w]
        hazard_zone = (xx - hx)**2 + (yy - hy)**2 <= CFG["hazard_block_radius"]**2

    print("Building potential field...")
    grid = ContGrid(wm, exits_px, CFG["grid_res"], hazard_zone=hazard_zone if hazard_cfg else None)
    grid.build_phi()
    print(f"  Reachable cells: {np.isfinite(grid.bfs).sum()}")

    print("Spawning agents...")
    zone_labels = None
    try:
        zone_labels = rebuild_labels(mask_path)
    except Exception as e:
        print(f"  label rebuild failed: {e}")

    ys, xs   = np.where(wm.walkable)
    pool_all = list(zip(xs.tolist(), ys.tolist()))

    agents = []
    for z in zcfg.get("zones", []):
        if z.get("density_index", 0) <= 0:
            continue
        count = z.get("agents", 0)
        if count <= 0:
            continue
        pool = pool_all
        if zone_labels is not None:
            zid  = z["zone_id"]
            zm   = zone_id_mask(zone_labels, zid) & wm.walkable
            zy, zx = np.where(zm)
            if len(zx) > 0:
                pool = list(zip(zx.tolist(), zy.tolist()))
        for _ in range(count):
            px, py = random.choice(pool)
            agents.append({
                "x": float(px), "y": float(py),
                "vx": 0.0, "vy": 0.0,
                "done": False, "time": None, "exit_used": None,
                "trail": [(float(px), float(py))],
            })

    print(f"Spawned {len(agents)} agents")

    exits_arr = np.array(exits_px, dtype=np.float32)

    walk_u8  = wm.walkable.astype(np.uint8) * 255
    dist_map = cv2.distanceTransform(walk_u8, cv2.DIST_L2, 5)
    gy_map   = cv2.Sobel(dist_map, cv2.CV_32F, 0, 1, ksize=3)
    gx_map   = cv2.Sobel(dist_map, cv2.CV_32F, 1, 0, ksize=3)

    DT        = CFG["DT"]
    MAX_STEPS = int(CFG["MAX_TIME"] / DT)
    EXIT_R    = CFG["exit_radius"]
    AR        = CFG["agent_radius"]
    REP_STR   = CFG["repulse_str"]
    REP_RANGE = CFG["repulse_range"]
    TAU       = CFG["relax_time"]
    CELL_SZ   = 30.0

    congestion_map = np.zeros((wm.h, wm.w), dtype=np.float32)
    exit_counts    = [0] * len(exits_px)
    last_print     = -5.0
    sim_time       = 0.0
    fire_grad_x    = np.zeros((wm.h, wm.w), dtype=np.float32)
    fire_grad_y    = np.zeros((wm.h, wm.w), dtype=np.float32)

    for step in range(MAX_STEPS):
        sim_time = step * DT
        active   = [a for a in agents if not a["done"]]
        if not active:
            break

        if step % 30 == 0:
            grid.splat(agents)
            grid.build_phi()

        #  fire growth (visual + soft push only — routing around the
        #    hazard is already permanently handled by hazard_zone in
        #    ContGrid.walkable, same split as SFM/RVO) 
        if hazard_cfg and step % 10 == 0:
            fire_intensity = spread_fire(fire_intensity, wm.walkable, DT * 10,
                                          CFG["fire_spread_speed"], CFG["fire_intensity_factor"])
            fgy, fgx = np.gradient(fire_intensity)
            fire_grad_x, fire_grad_y = -fgx, -fgy

        bucket = {}
        for i, a in enumerate(active):
            key = (int(a["x"] / CELL_SZ), int(a["y"] / CELL_SZ))
            bucket.setdefault(key, []).append(i)

        new_vels = []
        for i, ag in enumerate(active):
            ax, ay = ag["x"], ag["y"]

            gdx, gdy = grid.grad_at(ax, ay)
            gx_c, gy_c = grid.p2g(ax, ay)

            # stuck with no direction → if inside the hazard zone, push
            # straight out away from the hazard center (same fallback SFM
            # uses for agents with no flow vector)
            if gdx == 0.0 and gdy == 0.0 and hazard_xy is not None:
                ix_h = int(np.clip(ax, 0, wm.w - 1))
                iy_h = int(np.clip(ay, 0, wm.h - 1))
                if hazard_zone[iy_h, ix_h]:
                    away_x, away_y = ax - hazard_xy[0], ay - hazard_xy[1]
                    away_mag = math.hypot(away_x, away_y)
                    if away_mag > 1e-6:
                        gdx, gdy = away_x / away_mag, away_y / away_mag

            f_desired  = grid._speed(gx_c, gy_c, gdx, gdy)
            vd_x = gdx * f_desired
            vd_y = gdy * f_desired

            f_drive_x = (vd_x - ag["vx"]) / TAU
            f_drive_y = (vd_y - ag["vy"]) / TAU

            f_rep_x = f_rep_y = 0.0
            ki = int(ax / CELL_SZ)
            kj = int(ay / CELL_SZ)
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    for j in bucket.get((ki + di, kj + dj), []):
                        if j == i:
                            continue
                        nb = active[j]
                        dx = ax - nb["x"]
                        dy = ay - nb["y"]
                        dist = math.hypot(dx, dy)
                        if dist < 1e-3 or dist >= AR * 6:
                            continue
                        mag = REP_STR * math.exp((AR * 2 - dist) / REP_RANGE)
                        f_rep_x += mag * dx / dist
                        f_rep_y += mag * dy / dist

            ix_ = int(np.clip(ax, 0, wm.w - 1))
            iy_ = int(np.clip(ay, 0, wm.h - 1))
            d_wall = dist_map[iy_, ix_]
            f_wall_x = f_wall_y = 0.0
            if d_wall < 16.0:
                mag = 250.0 * math.exp(-d_wall / 8.0)
                f_wall_x = mag * gx_map[iy_, ix_]
                f_wall_y = mag * gy_map[iy_, ix_]

            # fire repulsion — soft push, complements the hard hazard_zone reroute
            local_risk = fire_intensity[iy_, ix_]
            f_fire_x = f_fire_y = 0.0
            if local_risk > 0.05:
                fire_mag = 400.0 * local_risk
                f_fire_x = fire_mag * fire_grad_x[iy_, ix_]
                f_fire_y = fire_mag * fire_grad_y[iy_, ix_]

            vx_new = ag["vx"] + (f_drive_x + f_rep_x + f_wall_x + f_fire_x) * DT
            vy_new = ag["vy"] + (f_drive_y + f_rep_y + f_wall_y + f_fire_y) * DT

            spd = math.hypot(vx_new, vy_new)
            if spd > CFG["speed_px_s"] * 1.4:
                vx_new = vx_new / spd * CFG["speed_px_s"] * 1.4
                vy_new = vy_new / spd * CFG["speed_px_s"] * 1.4

            new_vels.append((vx_new, vy_new))

        for i, ag in enumerate(active):
            vx, vy = new_vels[i]
            ag["vx"], ag["vy"] = vx, vy

            nx_ = np.clip(ag["x"] + vx * DT, 0, wm.w - 1)
            ny_ = np.clip(ag["y"] + vy * DT, 0, wm.h - 1)

            if   wm.walkable[int(ny_), int(nx_)]: ag["x"], ag["y"] = nx_, ny_
            elif wm.walkable[int(ag["y"]), int(nx_)]: ag["x"] = nx_; ag["vy"] *= 0.5
            elif wm.walkable[int(ny_), int(ag["x"])]: ag["y"] = ny_; ag["vx"] *= 0.5

            if step % 4 == 0:  # trail thinning — every 4th tick
                ag["trail"].append((ag["x"], ag["y"]))

            cx_ = int(ag["x"]); cy_ = int(ag["y"])
            congestion_map[cy_, cx_] += DT

            # exit check — scalar math, no per-agent numpy allocation
            ax2, ay2 = ag["x"], ag["y"]
            er2 = EXIT_R * EXIT_R
            nearest, best_d2 = 0, float("inf")
            for ei, (ex, ey) in enumerate(exits_px):
                d2 = (ax2 - ex)**2 + (ay2 - ey)**2
                if d2 < best_d2:
                    best_d2, nearest = d2, ei
            if best_d2 < er2:
                ag["done"]      = True
                ag["time"]      = sim_time
                ag["exit_used"] = nearest
                ag["vx"] = ag["vy"] = 0.0
                exit_counts[nearest] += 1

        if sim_time - last_print >= 5.0:
            evac = sum(1 for a in agents if a["done"])
            print(f"  t={sim_time:.1f}s  active={len(active)}  evacuated={evac}")
            last_print = sim_time

    evac_final = sum(1 for a in agents if a["done"])
    total      = len(agents)
    print(f"Done  t={sim_time:.1f}s  evacuated={evac_final}/{total}")

    #  Analysis report 
    times = [a["time"] for a in agents if a["done"] and a["time"] is not None]
    rate  = evac_final / total

    score_rate    = rate * 50
    if times:
        mean_t     = float(np.mean(times))
        score_time = max(0.0, 20 * (1 - (mean_t - 20) / 60)) if mean_t > 20 else 20.0
    else:
        mean_t = 0.0; score_time = 0.0

    total_evac = sum(exit_counts)
    if total_evac > 0:
        fractions = [c / total_evac for c in exit_counts]
        ideal     = 1.0 / len(exits_px)
        max_dev   = max(abs(f - ideal) for f in fractions)
        score_bal = max(0.0, 15 * (1 - max_dev / ideal))
    else:
        score_bal = 0.0

    walk_cong = congestion_map[wm.walkable]
    if walk_cong.max() > 0:
        thresh = np.percentile(walk_cong[walk_cong > 0], 80)
    else:
        thresh = 1.0
    bn_mask     = (congestion_map > thresh).astype(np.uint8) * 255
    contours, _ = cv2.findContours(bn_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bottlenecks = []
    for cnt in contours:
        bx, by, bw, bh = cv2.boundingRect(cnt)
        region_cong = congestion_map[by:by+bh, bx:bx+bw].sum()
        M = cv2.moments(cnt)
        if M["m00"] > 0:
            cx_ = int(M["m10"] / M["m00"]); cy_ = int(M["m01"] / M["m00"])
        else:
            cx_, cy_ = bx + bw//2, by + bh//2
        bottlenecks.append({"cx": cx_, "cy": cy_,
                             "agent_seconds": float(region_cong),
                             "width_px": min(bw, bh)})
    bottlenecks.sort(key=lambda b: b["agent_seconds"], reverse=True)
    top_bn = bottlenecks[:5]

    total_agent_time = congestion_map.sum()
    bn_time  = sum(b["agent_seconds"] for b in top_bn)
    bn_frac  = bn_time / (total_agent_time + 1e-8)
    score_bn = max(0.0, 15 * (1 - (bn_frac - 0.05) / 0.45))

    final_score = int(min(100, max(0, score_rate + score_time + score_bal + score_bn)))

    worst = min(
        ("evacuation_rate", score_rate / 50),
        ("exit_balance",    score_bal / 15),
        ("bottlenecks",     score_bn / 15),
        ("evac_time",       score_time / 20),
        key=lambda x: x[1]
    )
    bn_pos  = f"({top_bn[0]['cx']}, {top_bn[0]['cy']})" if top_bn else "unknown"
    min_ei  = int(np.argmin(exit_counts)) if total_evac > 0 else 0
    min_pos = f"({exits_px[min_ei][0]}, {exits_px[min_ei][1]})"
    RECS = {
        "evacuation_rate": "Too many agents failed to evacuate — check for isolated rooms with no path to any exit.",
        "exit_balance":    f"Exit {min_ei} at {min_pos} handled almost no traffic. Consider repositioning it or adding signage.",
        "bottlenecks":     f"The corridor at {bn_pos} is your biggest chokepoint. Widen it or add a parallel route.",
        "evac_time":       "Evacuation is too slow. Add an exit closer to the centre of the building.",
    }

    hazard_str = (f"active @ {tuple(int(v) for v in hazard_xy)} r={CFG['hazard_block_radius']}px"
                  if hazard_cfg else "none")

    sep = "=" * 55
    lines = [
        f"\n{sep}",
        "       CONTINUUM EVACUATION ANALYSIS REPORT",
        sep,
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
            f"  Mean evac time  : {mean_t:.1f}s",
            f"  Slowest evac    : {max(times):.1f}s",
        ]
    lines += ["", "-"*55, "  EXIT UTILIZATION", "-"*55]
    for idx, (ex, ey) in enumerate(exits_px):
        pct    = (exit_counts[idx] / total_evac * 100) if total_evac > 0 else 0
        bar    = "" * int(pct / 5)
        status = " UNDERUSED" if pct < (100 / len(exits_px) * 0.4) else ""
        lines.append(f"  Exit {idx} ({ex},{ey}): {exit_counts[idx]:3d} agents  {pct:5.1f}%  {bar} {status}")
    lines += ["", "-"*55, "  TOP BOTTLENECKS  (ranked by agent-seconds lost)", "-"*55]
    for rank, bn in enumerate(top_bn):
        lines.append(
            f"  B{rank+1}  position ({bn['cx']:4d},{bn['cy']:4d})  "
            f"corridor width ~{bn['width_px']}px  "
            f"{bn['agent_seconds']:.0f} agent-seconds"
        )
    lines += [
        "", "-"*55, "  RECOMMENDATION", "-"*55,
        f"  {RECS[worst[0]]}",
        "",
        "  Score breakdown:",
        f"    Evacuation rate  : {score_rate:.0f}/50",
        f"    Evacuation speed : {score_time:.0f}/20",
        f"    Exit balance     : {score_bal:.0f}/15",
        f"    Bottleneck sev.  : {score_bn:.0f}/15",
        sep,
    ]

    report = "\n".join(lines)
    print(report.encode("ascii", errors="replace").decode("ascii"))
    report_path = output_path("continuum_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Saved -> {report_path}")

    #  Render 
    print("\nRendering...")
    out = np.zeros((wm.h, wm.w, 3), dtype=np.uint8)
    out[wm.raw >= 128] = CFG["wall_color"]

    if congestion_map.max() > 0:
        dn   = (congestion_map / congestion_map.max() * 255).astype(np.uint8)
        heat = cv2.applyColorMap(dn, cv2.COLORMAP_HOT)
        mask_f = wm.walkable.astype(np.uint8)[:, :, np.newaxis]
        out  = cv2.addWeighted(out, 0.55, heat * mask_f, 0.45, 0)

    if hazard_cfg and fire_intensity.max() > 0:
        fire_u8    = (np.clip(fire_intensity, 0, 1) * 255).astype(np.uint8)
        fire_color = cv2.applyColorMap(fire_u8, cv2.COLORMAP_HOT)
        fire_mask3 = (fire_intensity > 0.03).astype(np.uint8)[:, :, None]
        out = np.where(fire_mask3 > 0, cv2.addWeighted(out, 0.4, fire_color, 0.6, 0), out)

    for a in agents:
        color = (0, 200, 55) if a["done"] else (0, 80, 200)
        trail = a["trail"]
        for i in range(1, len(trail)):
            cv2.line(out,
                     (int(trail[i-1][0]), int(trail[i-1][1])),
                     (int(trail[i][0]),   int(trail[i][1])),
                     color, 1)

    ec = CFG["exit_color"]
    for idx, (ex, ey) in enumerate(exits_px):
        pct = (exit_counts[idx] / total_evac * 100) if total_evac > 0 else 0
        cv2.circle(out, (ex, ey), 12, ec, 2)
        cv2.putText(out, f"E{idx} {pct:.0f}%", (ex - 18, ey - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, ec, 1)
    for rank, bn in enumerate(top_bn):
        cv2.circle(out, (bn["cx"], bn["cy"]), 14, (0, 0, 255), 2)
        cv2.putText(out, f"B{rank+1}", (bn["cx"] - 8, bn["cy"] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    sc_col = (0,200,0) if final_score >= 80 else (0,180,255) if final_score >= 60 else (0,0,255)
    cv2.rectangle(out, (4, 4), (140, 30), (0, 0, 0), -1)
    cv2.putText(out, f"SCORE: {final_score}/100", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, sc_col, 1)

    cv2.imwrite(CFG["output"], out)
    print(f"Saved -> {CFG['output']}")


if __name__ == "__main__":
    main()
