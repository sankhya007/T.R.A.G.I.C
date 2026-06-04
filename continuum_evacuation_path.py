"""
continuum_paths.py  —  Continuum Crowds → static path image
Run: python continuum_paths.py

Reads:  zone_config.json  +  stitched_mask.png  (paths from JSON)
Writes: continuum_agent_paths.png
"""

import json, math, random, sys
import numpy as np
import cv2
from pathlib import Path
from collections import deque
import heapq
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

# ══════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════
CFG = {
    "zone_config":  "stitched_mask_zone_config.json",
    "output":       "continuum_agent_paths.png",

    # Potential field grid resolution (px per cell). 4 = fast, 2 = finer paths
    "grid_res": 4,

    # Path tracing: how many px each step moves along ∇φ
    "trace_step": 2,

    # Max steps per agent trace (safety limit)
    "max_steps": 4000,

    # Exit capture radius (px)
    "exit_radius": 20,

    # Cost weights  C = (alpha*f + beta + gamma*g) / f
    "alpha": 0.3,
    "beta":  0.7,
    "gamma": 0.0,

    # Base speed (only affects cost weighting, not rendering)
    "speed_base": 40,
    "speed_min":  8,

    # Density influence (agents slow each other → paths spread)
    "rho_min": 0.05,
    "rho_max": 0.40,
    "density_radius": 6,   # px splat radius

    # Visual
    "bg_color":         (0,   0,   0),    # pure black
    "wall_color":       (85, 85, 85),     # grey walls
    "exit_color":       (0, 200, 220),    # yellow-gold circles (BGR)
    "path_color_dim":   (0,  60,  0),     # single path
    "path_color_bright":(0, 220, 55),     # many overlapping paths
    "line_thickness": 1,
    "additive_scale": 10,                 # overlaps needed for full brightness
}
# ══════════════════════════════════════════════════════════════


# ── Walkability ───────────────────────────────────────────────
class WalkMap:
    def __init__(self, path):
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(path)
        self.walkable = img < 128      # white=wall, black=walkable
        self.h, self.w = img.shape
        self.raw = img

    def ok(self, x, y):
        if not (math.isfinite(x) and math.isfinite(y)):
            return False
        ix, iy = int(round(x)), int(round(y))
        return 0 <= ix < self.w and 0 <= iy < self.h and self.walkable[iy, ix]


# ── Coarse continuum grid ─────────────────────────────────────
class ContGrid:
    def __init__(self, wm: WalkMap, exits, res):
        self.res = res
        self.gw = math.ceil(wm.w / res)
        self.gh = math.ceil(wm.h / res)
        self.wm = wm

        # Walkable at grid resolution
        self.walkable = np.zeros((self.gh, self.gw), dtype=bool)
        for gy in range(self.gh):
            for gx in range(self.gw):
                px = min(int((gx+0.5)*res), wm.w-1)
                py = min(int((gy+0.5)*res), wm.h-1)
                self.walkable[gy, gx] = wm.walkable[py, px]

        self.exits = exits
        self.rho   = np.zeros((self.gh, self.gw), dtype=np.float32)
        self.vavg  = np.zeros((self.gh, self.gw, 2), dtype=np.float32)
        self.phi   = np.full((self.gh, self.gw), np.inf, dtype=np.float64)

        # BFS distance (fallback gradient + seed for fast-march)
        self.bfs = np.full((self.gh, self.gw), np.inf)
        self._bfs(exits)

    def p2g(self, px, py):
        return (int(np.clip(px/self.res, 0, self.gw-1)),
                int(np.clip(py/self.res, 0, self.gh-1)))

    def _bfs(self, exits):
        q = deque()
        for ex, ey in exits:
            gx, gy = self.p2g(ex, ey)
            for dy in range(-4, 5):
                for dx in range(-4, 5):
                    nx, ny = gx+dx, gy+dy
                    if 0<=ny<self.gh and 0<=nx<self.gw and self.walkable[ny,nx]:
                        if self.bfs[ny,nx] == np.inf:
                            self.bfs[ny,nx] = 0; q.append((nx,ny))
        while q:
            cx,cy = q.popleft()
            for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                nx,ny = cx+dx, cy+dy
                if 0<=ny<self.gh and 0<=nx<self.gw and self.walkable[ny,nx]:
                    if self.bfs[ny,nx]==np.inf:
                        self.bfs[ny,nx]=self.bfs[cy,cx]+1; q.append((nx,ny))

    def splat_density(self, agents):
        self.rho[:]=0; self.vavg[:]=0
        r = max(1, CFG["density_radius"]//self.res)
        for ax,ay,vx,vy in agents:
            gx,gy = self.p2g(ax,ay)
            for dy in range(-r,r+1):
                for dx in range(-r,r+1):
                    nx,ny=gx+dx,gy+dy
                    if 0<=ny<self.gh and 0<=nx<self.gw:
                        w=max(0,1-(abs(dx)+abs(dy))/(r+1))
                        self.rho[ny,nx]+=w
                        self.vavg[ny,nx,0]+=w*vx
                        self.vavg[ny,nx,1]+=w*vy
        m=self.rho>0
        self.vavg[m,0]/=self.rho[m]; self.vavg[m,1]/=self.rho[m]

    def _speed(self, gx, gy, dx, dy):
        rho = self.rho[gy,gx]
        ft = CFG["speed_base"]
        d=math.hypot(dx,dy)
        if d<1e-6: fv=CFG["speed_min"]
        else:
            nx_,ny_=dx/d,dy/d
            fv=self.vavg[gy,gx,0]*nx_+self.vavg[gy,gx,1]*ny_
            fv=max(CFG["speed_min"],fv)
        rn,rx=CFG["rho_min"],CFG["rho_max"]
        if rho<=rn: return ft
        if rho>=rx: return fv
        t=(rho-rn)/(rx-rn)
        return ft+t*(fv-ft)

    def _cost(self, gx, gy, dx, dy):
        f=max(1.0,self._speed(gx,gy,dx,dy))
        return (CFG["alpha"]*f+CFG["beta"])/f

    def build_phi(self):
        phi=np.full((self.gh,self.gw),np.inf)
        hp=[]
        visited=np.zeros((self.gh,self.gw),dtype=bool)
        for ex,ey in self.exits:
            gx,gy=self.p2g(ex,ey)
            for dy in range(-4,5):
                for dx in range(-4,5):
                    nx,ny=gx+dx,gy+dy
                    if 0<=ny<self.gh and 0<=nx<self.gw and self.walkable[ny,nx]:
                        if phi[ny,nx]==np.inf:
                            phi[ny,nx]=0; heapq.heappush(hp,(0.0,nx,ny))
        DIRS=[(1,0),(-1,0),(0,1),(0,-1)]
        while hp:
            val,cx,cy=heapq.heappop(hp)
            if visited[cy,cx]: continue
            visited[cy,cx]=True; phi[cy,cx]=val
            for dx,dy in DIRS:
                nx,ny=cx+dx,cy+dy
                if not(0<=ny<self.gh and 0<=nx<self.gw): continue
                if visited[ny,nx] or not self.walkable[ny,nx]: continue
                c=self._cost(cx,cy,dx,dy)
                cand=val+c
                if cand<phi[ny,nx]:
                    phi[ny,nx]=cand; heapq.heappush(hp,(cand,nx,ny))
        self.phi=phi

    def grad_at(self, px, py):
        """Return normalised descent direction at pixel position."""
        gx,gy=self.p2g(px,py)
        gx=int(np.clip(gx,1,self.gw-2)); gy=int(np.clip(gy,1,self.gh-2))
        if not self.walkable[gy,gx]: return 0.0,0.0
        dpx=(self.phi[gy,gx+1]-self.phi[gy,gx-1])/2
        dpy=(self.phi[gy+1,gx]-self.phi[gy-1,gx])/2
        if not (np.isfinite(dpx) and np.isfinite(dpy)):
            dpx=(self.bfs[gy,min(gx+1,self.gw-1)]-self.bfs[gy,max(gx-1,0)])/2
            dpy=(self.bfs[min(gy+1,self.gh-1),gx]-self.bfs[max(gy-1,0),gx])/2
        if not (np.isfinite(dpx) and np.isfinite(dpy)):
            return 0.0, 0.0
        mag=math.hypot(dpx,dpy)
        if mag<1e-6: return 0.0,0.0
        return -dpx/mag, -dpy/mag


# ── Rebuild zone labels (same as agent_sim.py) ────────────────
def rebuild_labels(mask_path):
    img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    walkable = cv2.bitwise_not(img)
    _, binary = cv2.threshold(walkable, 127, 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    dn = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)
    coords = peak_local_max(dn, min_distance=40, labels=binary)
    sm = np.zeros(dn.shape, dtype=bool); sm[tuple(coords.T)] = True
    markers, _ = ndi.label(sm)
    return watershed(-dist, markers, mask=binary)


# ── Trace one agent path ──────────────────────────────────────
def trace_path(start_x, start_y, grid: ContGrid, exits):
    step = CFG["trace_step"]
    er   = CFG["exit_radius"]
    path = [(start_x, start_y)]
    x, y = float(start_x), float(start_y)

    for _ in range(CFG["max_steps"]):
        # Check exit
        for ex, ey in exits:
            if math.hypot(x-ex, y-ey) < er:
                path.append((ex, ey))
                return path

        dx, dy = grid.grad_at(x, y)
        if not (math.isfinite(dx) and math.isfinite(dy)):
            break
        if math.hypot(dx,dy) < 1e-6:
            break   # stuck (unreachable from exits)

        nx, ny = x+dx*step, y+dy*step
        if grid.wm.ok(nx, ny):
            x, y = nx, ny
        elif grid.wm.ok(nx, y):
            x = nx
        elif grid.wm.ok(x, ny):
            y = ny
        else:
            break   # wall trap

        path.append((x, y))

    return path   # partial trace (agent didn't reach exit)


# ── Main ──────────────────────────────────────────────────────
def main():
    cfg_path = CFG["zone_config"]
    if not Path(cfg_path).exists():
        print(f"ERROR: {cfg_path} not found"); sys.exit(1)

    with open(cfg_path) as f:
        zcfg = json.load(f)

    mask_path = zcfg.get("mask_path","")
    if not Path(mask_path).exists():
        print(f"ERROR: mask not found at {mask_path}"); sys.exit(1)

    print("Loading mask…")
    wm = WalkMap(mask_path)

    exits_raw = zcfg.get("exits", [])
    if not exits_raw:
        print("ERROR: no exits in config"); sys.exit(1)
    exits = [(int(e["x"]), int(e["y"])) for e in exits_raw]
    print(f"  {len(exits)} exits")

    # Build grid and initial potential (no density yet)
    print("Building potential field…")
    grid = ContGrid(wm, exits, CFG["grid_res"])
    grid.build_phi()

    # Collect spawn points from zone config
    print("Collecting spawn points…")
    zones = zcfg.get("zones", [])
    zone_labels = None
    if Path(mask_path).exists():
        try:
            zone_labels = rebuild_labels(mask_path)
        except Exception as e:
            print(f"  label rebuild failed: {e}")

    ys, xs = np.where(wm.walkable)
    pool_all = list(zip(xs.tolist(), ys.tolist()))

    spawn_points = []
    for z in zones:
        if z.get("density_index",1) <= 0:
            continue
        count = z.get("agents", 0)
        if count <= 0:
            continue
        pool = pool_all
        if zone_labels is not None:
            zid = z["zone_id"]
            zm = (zone_labels == zid) & wm.walkable
            zy, zx = np.where(zm)
            if len(zx) > 0:
                pool = list(zip(zx.tolist(), zy.tolist()))
        for _ in range(count):
            px, py = random.choice(pool)
            spawn_points.append((float(px), float(py)))

    print(f"  {len(spawn_points)} agents to trace")

    # ── Iterative density-aware tracing ──────────────────────
    # Pass 1: trace without density (fast-march only)
    # Pass 2: splat density from pass-1 paths, rebuild φ, retrace
    # This approximates the coupled continuum update cheaply.

    all_paths = []
    agent_states = [(x, y, 0.0, 0.0) for x, y in spawn_points]

    for iteration in range(2):
        print(f"Tracing pass {iteration+1}/2…")
        all_paths = []
        for sx, sy, _, _ in agent_states:
            p = trace_path(sx, sy, grid, exits)
            all_paths.append(p)

        if iteration == 0:
            # Build density from traced paths
            mid_states = []
            for path in all_paths:
                if len(path) > 2:
                    mid = path[len(path)//2]
                    # Estimate velocity at midpoint
                    i = len(path)//2
                    if i+1 < len(path):
                        vx = path[i+1][0]-path[i][0]
                        vy = path[i+1][1]-path[i][1]
                    else:
                        vx, vy = 0, 0
                    mid_states.append((mid[0], mid[1], vx, vy))
                else:
                    mid_states.append((path[0][0], path[0][1], 0, 0))
            grid.splat_density(mid_states)
            grid.build_phi()

    # ── Render ───────────────────────────────────────────────
    print("Rendering…")

    # ── Step 1: background — black + grey wall outlines ──────
    out = np.zeros((wm.h, wm.w, 3), dtype=np.uint8)

    # Draw wall pixels as grey
    wall_px = wm.raw >= 128
    out[wall_px] = CFG["wall_color"]

    # ── Step 2: accumulate path counts per pixel ─────────────
    # Float buffer counts how many path segments cross each pixel
    counts = np.zeros((wm.h, wm.w), dtype=np.float32)
    for path in all_paths:
        for i in range(len(path) - 1):
            x1, y1 = int(round(path[i][0])),   int(round(path[i][1]))
            x2, y2 = int(round(path[i+1][0])), int(round(path[i+1][1]))
            cv2.line(counts, (x1,y1), (x2,y2), 1.0, CFG["line_thickness"])

    # ── Step 3: map count → green brightness (additive look) ──
    # log scale so sparse areas still show, dense areas glow bright
    scale = CFG["additive_scale"]
    # normalise: log(1 + count) / log(1 + scale)
    norm = np.log1p(counts) / math.log(1 + scale)
    norm = np.clip(norm, 0, 1)

    # Interpolate dim→bright green
    cd = np.array(CFG["path_color_dim"],    dtype=np.float32)  # BGR
    cb = np.array(CFG["path_color_bright"], dtype=np.float32)

    path_layer = np.zeros((wm.h, wm.w, 3), dtype=np.float32)
    has_path = counts > 0
    t = norm[has_path, np.newaxis]
    path_layer[has_path] = cd * (1 - t) + cb * t

    # Composite: paths on top of background
    # Where there are paths, replace background completely
    result = out.astype(np.float32)
    result[has_path] = path_layer[has_path]
    result = np.clip(result, 0, 255).astype(np.uint8)

    # ── Step 4: exit markers — yellow circle + label above ───
    ec = CFG["exit_color"]   # BGR
    for ex, ey in exits:
        r = CFG["exit_radius"]
        cv2.circle(result, (ex, ey), r, ec, 2)
        # "EXIT" label centred above the circle
        label = "EXIT"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        cv2.putText(result, label, (ex - tw//2, ey - r - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, ec, 1, cv2.LINE_AA)

    out_path = CFG["output"]
    cv2.imwrite(out_path, result)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()