"""
continuum_paths.py  —  Continuum Crowds simulation + analysis report
Reads:  zone_config  +  stitched_mask.png
Writes: continuum_agent_paths.png  +  prints report to terminal
"""

import json, math, random, sys, warnings
import numpy as np
import cv2
from pathlib import Path
from collections import deque
import heapq
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ══════════════════════════════════════════════════════════════
CFG = {
    "zone_config": "stitched_mask_zone_config.json",
    "output":      "continuum_agent_paths.png",

    "DT":          0.05,
    "MAX_TIME":    40,
    "speed_px_s":  150.0,
    "exit_radius": 22,

    "grid_res":    2,
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
}


class WalkMap:
    def __init__(self, path):
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(path)
        self.walkable = img < 128
        self.h, self.w = img.shape
        self.raw = img

    def ok(self, x, y):
        ix, iy = int(round(x)), int(round(y))
        return 0 <= ix < self.w and 0 <= iy < self.h and self.walkable[iy, ix]


class ContGrid:
    def __init__(self, wm, exits_px, res):
        self.res = res
        self.gw  = math.ceil(wm.w / res)
        self.gh  = math.ceil(wm.h / res)
        self.wm  = wm

        self.walkable = np.zeros((self.gh, self.gw), dtype=bool)
        for gy in range(self.gh):
            for gx in range(self.gw):
                px = min(int((gx + 0.5) * res), wm.w - 1)
                py = min(int((gy + 0.5) * res), wm.h - 1)
                self.walkable[gy, gx] = wm.walkable[py, px]

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
        for a in agents:
            if a["done"]:
                continue
            gx, gy = self.p2g(a["x"], a["y"])
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    nx, ny = gx + dx, gy + dy
                    if 0 <= ny < self.gh and 0 <= nx < self.gw:
                        w = max(0, 1 - (abs(dx) + abs(dy)) / (r + 1))
                        self.rho[ny, nx] += w
                        self.vavg[ny, nx, 0] += w * a["vx"]
                        self.vavg[ny, nx, 1] += w * a["vy"]
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
    coords = peak_local_max(dn, min_distance=40, labels=binary)
    sm     = np.zeros(dn.shape, dtype=bool)
    sm[tuple(coords.T)] = True
    markers, _ = ndi.label(sm)
    return watershed(-dist, markers, mask=binary)


def main():
    cfg_path = CFG["zone_config"]
    if not Path(cfg_path).exists():
        print(f"ERROR: {cfg_path} not found"); sys.exit(1)

    with open(cfg_path) as f:
        zcfg = json.load(f)

    mask_path = zcfg.get("mask_path", "")
    if not Path(mask_path).exists():
        print(f"ERROR: mask not found at {mask_path}"); sys.exit(1)

    print("Loading mask...")
    wm = WalkMap(mask_path)

    exits_raw = zcfg.get("exits", [])
    if not exits_raw:
        print("ERROR: no exits in config"); sys.exit(1)
    exits_px = [(int(e["x"]), int(e["y"])) for e in exits_raw]
    print(f"  {len(exits_px)} exits")

    print("Building potential field...")
    grid = ContGrid(wm, exits_px, CFG["grid_res"])
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
            zm   = (zone_labels == zid) & wm.walkable
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

    for step in range(MAX_STEPS):
        sim_time = step * DT
        active   = [a for a in agents if not a["done"]]
        if not active:
            break

        if step % 10 == 0:
            grid.splat(agents)
            grid.build_phi()

        bucket = {}
        for i, a in enumerate(active):
            key = (int(a["x"] / CELL_SZ), int(a["y"] / CELL_SZ))
            bucket.setdefault(key, []).append(i)

        new_vels = []
        for i, ag in enumerate(active):
            ax, ay = ag["x"], ag["y"]

            gdx, gdy = grid.grad_at(ax, ay)
            gx_c, gy_c = grid.p2g(ax, ay)
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

            vx_new = ag["vx"] + (f_drive_x + f_rep_x + f_wall_x) * DT
            vy_new = ag["vy"] + (f_drive_y + f_rep_y + f_wall_y) * DT

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

            ag["trail"].append((ag["x"], ag["y"]))

            cx_ = int(ag["x"]); cy_ = int(ag["y"])
            congestion_map[cy_, cx_] += DT

            pos   = np.array([ag["x"], ag["y"]])
            dists = np.linalg.norm(exits_arr - pos, axis=1)
            nearest = int(np.argmin(dists))
            if dists[nearest] < EXIT_R:
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

    # ── Analysis report ────────────────────────────────────────
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

    sep = "=" * 55
    print(f"\n{sep}")
    print("       EVACUATION ANALYSIS REPORT")
    print(sep)
    print(f"  OVERALL SCORE : {final_score} / 100")
    print(f"  Total agents    : {total}")
    print(f"  Evacuated       : {evac_final}  ({100*rate:.1f}%)")
    print(f"  Trapped/timeout : {total - evac_final}  ({100*(1-rate):.1f}%)")
    if times:
        print(f"  Fastest evac    : {min(times):.1f}s")
        print(f"  Mean evac time  : {mean_t:.1f}s")
        print(f"  Slowest evac    : {max(times):.1f}s")
    print(f"\n{'-'*55}")
    print("  EXIT UTILIZATION")
    print(f"{'-'*55}")
    for idx, (ex, ey) in enumerate(exits_px):
        pct    = (exit_counts[idx] / total_evac * 100) if total_evac > 0 else 0
        bar    = "█" * int(pct / 5)
        status = "⚠ UNDERUSED" if pct < (100 / len(exits_px) * 0.4) else ""
        print(f"  Exit {idx} ({ex},{ey}): {exit_counts[idx]:3d} agents  {pct:5.1f}%  {bar} {status}")
    print(f"\n{'-'*55}")
    print("  TOP BOTTLENECKS  (ranked by agent-seconds lost)")
    print(f"{'-'*55}")
    for rank, bn in enumerate(top_bn):
        print(f"  B{rank+1}  position ({bn['cx']:4d},{bn['cy']:4d})  "
              f"corridor width ~{bn['width_px']}px  "
              f"{bn['agent_seconds']:.0f} agent-seconds")
    print(f"\n{'-'*55}")
    print("  RECOMMENDATION")
    print(f"{'-'*55}")
    print(f"  {RECS[worst[0]]}")
    print(f"\n  Score breakdown:")
    print(f"    Evacuation rate  : {score_rate:.0f}/50")
    print(f"    Evacuation speed : {score_time:.0f}/20")
    print(f"    Exit balance     : {score_bal:.0f}/15")
    print(f"    Bottleneck sev.  : {score_bn:.0f}/15")
    print(sep)

    # ── Render ────────────────────────────────────────────────
    print("\nRendering...")
    out = np.zeros((wm.h, wm.w, 3), dtype=np.uint8)
    out[wm.raw >= 128] = CFG["wall_color"]

    if congestion_map.max() > 0:
        dn   = (congestion_map / congestion_map.max() * 255).astype(np.uint8)
        heat = cv2.applyColorMap(dn, cv2.COLORMAP_HOT)
        mask_f = wm.walkable.astype(np.uint8)[:, :, np.newaxis]
        out  = cv2.addWeighted(out, 0.55, heat * mask_f, 0.45, 0)

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