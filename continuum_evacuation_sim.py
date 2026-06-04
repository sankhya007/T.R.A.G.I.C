"""
continuum_evacuation.py  —  Continuum Crowds Evacuation
Based on: Treuille, Cooper, Popović — "Continuum Crowds" (SIGGRAPH 2006)

Requires:
  - stitched_mask.png        (white=wall, black=walkable)
  - zone_config.json         (zone density + exit positions)

Controls:
  Space   — Pause / Resume
  R       — Reset
  Click   — Inspect agent
"""

import sys, json, math, random
import numpy as np
import cv2
from pathlib import Path
from collections import deque
from scipy.ndimage import label as scipy_label
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QLabel, QFileDialog, QFrame, QSizePolicy,
    QScrollArea, QMessageBox, QCheckBox, QSlider
)
from PyQt6.QtCore import Qt, QTimer, QPointF
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QImage, QPixmap, QPainterPath

# ══════════════════════════════════════════════════════════════════
#  CONFIG — tweak everything here
# ══════════════════════════════════════════════════════════════════
CFG = {
    # Grid resolution for the potential field (pixels per cell)
    # Smaller = more accurate but slower
    "grid_res": 4,

    # Agent parameters
    "agent_radius": 5,          # pixels (for rendering + separation)
    "speed_base": 40,           # px/s at free-flow
    "speed_min": 8,             # px/s at max density
    "speed_variance": 0.20,     # ±20%

    # Continuum density model (Treuille §3.2)
    "rho_min": 0.05,            # below this: topographic speed
    "rho_max": 0.40,            # above this: flow speed dominates
    "density_radius": 8,        # px — splat kernel radius

    # Path cost weights (Treuille eq.4): C = (alpha*f + beta + gamma*g) / f
    "alpha": 0.3,               # path-length weight
    "beta":  0.7,               # time weight
    "gamma": 0.0,               # discomfort weight (0 = off)

    # Discomfort: add ahead of each agent to nudge lane formation
    "predictive_discomfort": True,
    "discomfort_ahead_steps": 6,   # timesteps to project forward

    # BFS pre-pass: flood distances from exits before sim starts
    "bfs_prepass": True,

    # Separation enforcement radius (px)
    "separation_radius": 9,

    # Simulation timestep
    "dt": 0.05,

    # Evacuation radius around exit (px)
    "exit_radius": 18,
}
# ══════════════════════════════════════════════════════════════════


# ── Walkability map ───────────────────────────────────────────────
class WalkMap:
    def __init__(self, path):
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(path)
        self.walkable = img < 128          # white=wall, black=walkable
        self.h, self.w = img.shape
        walk8 = self.walkable.astype(np.uint8) * 255
        self.dist = cv2.distanceTransform(walk8, cv2.DIST_L2, 5)

    def ok(self, x, y):
        ix, iy = int(x), int(y)
        return 0 <= ix < self.w and 0 <= iy < self.h and self.walkable[iy, ix]


# ── Continuum grid (coarse) ───────────────────────────────────────
class ContGrid:
    """
    Coarse grid that holds the potential field φ and density ρ.
    All continuum computations happen here.
    """
    def __init__(self, walk_map: WalkMap, exits, res=4):
        self.res = res
        self.gw = math.ceil(walk_map.w / res)
        self.gh = math.ceil(walk_map.h / res)
        self.walk_map = walk_map

        # Pre-build walkable mask at grid resolution
        self.walkable = np.zeros((self.gh, self.gw), dtype=bool)
        for gy in range(self.gh):
            for gx in range(self.gw):
                px = int((gx + 0.5) * res)
                py = int((gy + 0.5) * res)
                px = min(px, walk_map.w - 1)
                py = min(py, walk_map.h - 1)
                self.walkable[gy, gx] = walk_map.walkable[py, px]

        # Pre-compute BFS distance from exits (in grid cells)
        self.bfs_dist = np.full((self.gh, self.gw), np.inf)
        self._bfs_exits(exits)

        # Runtime fields
        self.rho   = np.zeros((self.gh, self.gw), dtype=np.float32)
        self.vavg  = np.zeros((self.gh, self.gw, 2), dtype=np.float32)
        self.g     = np.zeros((self.gh, self.gw), dtype=np.float32)  # discomfort
        self.phi   = np.full((self.gh, self.gw), np.inf, dtype=np.float32)

        self.exits = exits  # list of (px, py) pixel coords

    # world px → grid cell
    def p2g(self, px, py):
        return int(px / self.res), int(py / self.res)

    def g2p(self, gx, gy):
        return (gx + 0.5) * self.res, (gy + 0.5) * self.res

    def _bfs_exits(self, exits):
        """BFS from all exit cells so we have a baseline distance map."""
        q = deque()
        for ex, ey in exits:
            gx, gy = self.p2g(ex, ey)
            for dy in range(-3, 4):
                for dx in range(-3, 4):
                    nx, ny = gx + dx, gy + dy
                    if 0 <= ny < self.gh and 0 <= nx < self.gw and self.walkable[ny, nx]:
                        if self.bfs_dist[ny, nx] == np.inf:
                            self.bfs_dist[ny, nx] = 0
                            q.append((nx, ny))
        while q:
            cx, cy = q.popleft()
            for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                nx, ny = cx+dx, cy+dy
                if 0 <= ny < self.gh and 0 <= nx < self.gw and self.walkable[ny, nx]:
                    if self.bfs_dist[ny, nx] == np.inf:
                        self.bfs_dist[ny, nx] = self.bfs_dist[cy, cx] + 1
                        q.append((nx, ny))

    def splat_density(self, agents):
        """Convert agents to density + average velocity fields (Treuille §4.1)."""
        self.rho[:] = 0
        self.vavg[:] = 0
        r = CFG["density_radius"] // self.res + 1
        for a in agents:
            if a.evacuated:
                continue
            gx, gy = self.p2g(a.x, a.y)
            for dy in range(-r, r+1):
                for dx in range(-r, r+1):
                    nx, ny = gx+dx, gy+dy
                    if 0 <= ny < self.gh and 0 <= nx < self.gw:
                        w = max(0, 1 - (abs(dx) + abs(dy)) / (r + 1))
                        self.rho[ny, nx] += w
                        self.vavg[ny, nx, 0] += w * a.vx
                        self.vavg[ny, nx, 1] += w * a.vy
        # Normalise vavg
        mask = self.rho > 0
        self.vavg[mask, 0] /= self.rho[mask]
        self.vavg[mask, 1] /= self.rho[mask]

    def splat_discomfort(self, agents, dt):
        """Predictive discomfort: project each agent forward (Treuille §3.3)."""
        self.g[:] = 0
        if not CFG["predictive_discomfort"]:
            return
        steps = CFG["discomfort_ahead_steps"]
        for a in agents:
            if a.evacuated:
                continue
            px, py = a.x, a.y
            for _ in range(steps):
                px += a.vx * dt
                py += a.vy * dt
                gx, gy = self.p2g(px, py)
                if 0 <= gy < self.gh and 0 <= gx < self.gw and self.walkable[gy, gx]:
                    self.g[gy, gx] += 0.5

    def speed_field(self, gx, gy, dx, dy):
        """
        Treuille eq.10: blend topographic and flow speed by density.
        dx,dy is the candidate movement direction (unnormalised ok).
        """
        rho = self.rho[gy, gx]
        f_top = CFG["speed_base"]

        # Flow speed: dot of local avg velocity with direction
        d = math.hypot(dx, dy)
        if d < 1e-6:
            f_flow = CFG["speed_min"]
        else:
            nx_, ny_ = dx/d, dy/d
            f_flow = self.vavg[gy, gx, 0]*nx_ + self.vavg[gy, gx, 1]*ny_
            f_flow = max(CFG["speed_min"], f_flow)

        rho_min = CFG["rho_min"]
        rho_max = CFG["rho_max"]
        if rho <= rho_min:
            return f_top
        if rho >= rho_max:
            return f_flow
        t = (rho - rho_min) / (rho_max - rho_min)
        return f_top + t * (f_flow - f_top)

    def unit_cost(self, gx, gy, dx, dy):
        """C = (alpha*f + beta + gamma*g) / f  (Treuille eq.4)"""
        f = max(1.0, self.speed_field(gx, gy, dx, dy))
        g = self.g[gy, gx]
        return (CFG["alpha"]*f + CFG["beta"] + CFG["gamma"]*g) / f

    def build_potential(self):
        """
        Fast-marching to solve the eikonal equation ||∇φ|| = C.
        Produces φ field that agents follow downhill.
        (Treuille §4.3)
        """
        phi = np.full((self.gh, self.gw), np.inf, dtype=np.float64)

        # Seed: exit cells = 0
        for ex, ey in self.exits:
            gx, gy = self.p2g(ex, ey)
            for dy in range(-3, 4):
                for dx in range(-3, 4):
                    nx, ny = gx+dx, gy+dy
                    if 0 <= ny < self.gh and 0 <= nx < self.gw and self.walkable[ny, nx]:
                        phi[ny, nx] = 0.0

        # Use BFS distance as heuristic initialisation then fast-march
        # Simple fast-marching via priority queue
        import heapq
        heap = []
        visited = np.zeros((self.gh, self.gw), dtype=bool)

        for gy in range(self.gh):
            for gx in range(self.gw):
                if phi[gy, gx] == 0.0:
                    heapq.heappush(heap, (0.0, gx, gy))

        DIRS = [(1,0),(-1,0),(0,1),(0,-1)]

        while heap:
            val, cx, cy = heapq.heappop(heap)
            if visited[cy, cx]:
                continue
            visited[cy, cx] = True
            phi[cy, cx] = val

            for dx, dy in DIRS:
                nx, ny = cx+dx, cy+dy
                if not (0 <= ny < self.gh and 0 <= nx < self.gw):
                    continue
                if visited[ny, nx] or not self.walkable[ny, nx]:
                    continue
                c = self.unit_cost(cx, cy, dx, dy)
                candidate = val + c
                if candidate < phi[ny, nx]:
                    phi[ny, nx] = candidate
                    heapq.heappush(heap, (candidate, nx, ny))

        self.phi = phi.astype(np.float32)

    def gradient_at(self, px, py):
        """
        Bilinear-sample the negative gradient of φ at pixel position.
        Returns (gx, gy) — the direction agents should move.
        """
        gx, gy = self.p2g(px, py)
        gx = int(np.clip(gx, 1, self.gw-2))
        gy = int(np.clip(gy, 1, self.gh-2))

        if not self.walkable[gy, gx]:
            return 0.0, 0.0

        phi = self.phi
        dphi_x = (phi[gy, gx+1] - phi[gy, gx-1]) / 2.0
        dphi_y = (phi[gy+1, gx] - phi[gy-1, gx]) / 2.0

        # Handle inf neighbours — fall back to BFS gradient
        if not np.isfinite(dphi_x) or not np.isfinite(dphi_y):
            d = self.bfs_dist
            dphi_x = (d[gy, min(gx+1, self.gw-1)] - d[gy, max(gx-1, 0)]) / 2.0
            dphi_y = (d[min(gy+1, self.gh-1), gx] - d[max(gy-1, 0), gx]) / 2.0

        mag = math.hypot(dphi_x, dphi_y)
        if mag < 1e-6:
            return 0.0, 0.0
        return -dphi_x/mag, -dphi_y/mag     # negative gradient = toward exit


# ── Agent ─────────────────────────────────────────────────────────
class Agent:
    _ctr = 0
    def __init__(self, x, y):
        Agent._ctr += 1
        self.id = Agent._ctr
        self.x, self.y = float(x), float(y)
        spd = CFG["speed_base"] * random.uniform(1-CFG["speed_variance"],
                                                  1+CFG["speed_variance"])
        self.speed = spd
        self.vx, self.vy = 0.0, 0.0
        self.evacuated = False
        self.selected = False
        self.exit_dist = np.inf   # filled by BFS prepass

    def update(self, grid: ContGrid, dt):
        if self.evacuated:
            return

        # Direction from potential field gradient
        dx, dy = grid.gradient_at(self.x, self.y)

        # Speed from density field
        gx, gy = grid.p2g(self.x, self.y)
        gx = int(np.clip(gx, 0, grid.gw-1))
        gy = int(np.clip(gy, 0, grid.gh-1))
        spd = grid.speed_field(gx, gy, dx, dy)
        spd = np.clip(spd, CFG["speed_min"], CFG["speed_base"])

        self.vx = dx * spd
        self.vy = dy * spd

        nx, ny = self.x + self.vx*dt, self.y + self.vy*dt

        if grid.walk_map.ok(nx, ny):
            self.x, self.y = nx, ny
        elif grid.walk_map.ok(nx, self.y):
            self.x = nx
        elif grid.walk_map.ok(self.x, ny):
            self.y = ny
        else:
            self.vx, self.vy = 0, 0


# ── Simulation ────────────────────────────────────────────────────
class Simulation:
    def __init__(self, walk_map, zone_config):
        self.walk_map = walk_map
        self.t = 0.0
        self.evacuated_count = 0

        exits_raw = zone_config.get("exits", [])
        if not exits_raw:
            raise ValueError("No exits found in zone_config.json")

        self.exit_positions = [(int(e["x"]), int(e["y"])) for e in exits_raw]

        self.grid = ContGrid(walk_map, self.exit_positions, CFG["grid_res"])

        self.agents = []
        self._spawn(zone_config)

        # BFS prepass: annotate each agent with distance to nearest exit
        if CFG["bfs_prepass"]:
            for a in self.agents:
                gx, gy = self.grid.p2g(a.x, a.y)
                gx = int(np.clip(gx, 0, self.grid.gw-1))
                gy = int(np.clip(gy, 0, self.grid.gh-1))
                a.exit_dist = self.grid.bfs_dist[gy, gx]

        # Initial potential field build
        self.grid.build_potential()

    def _rebuild_labels(self, mask_path):
        img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        walkable = cv2.bitwise_not(img)
        _, binary = cv2.threshold(walkable, 127, 255, cv2.THRESH_BINARY)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        dist_n = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)
        coords = peak_local_max(dist_n, min_distance=40, labels=binary)
        sm = np.zeros(dist_n.shape, dtype=bool)
        sm[tuple(coords.T)] = True
        markers, _ = ndi.label(sm)
        return watershed(-dist, markers, mask=binary)

    def _spawn(self, cfg):
        Agent._ctr = 0
        zones = cfg.get("zones", [])
        mask_path = cfg.get("mask_path", "")
        zone_labels = None
        if Path(mask_path).exists():
            try:
                zone_labels = self._rebuild_labels(mask_path)
            except Exception as e:
                print(f"Label rebuild failed: {e}")

        ys, xs = np.where(self.walk_map.walkable)
        pool_all = list(zip(xs.tolist(), ys.tolist()))

        for z in zones:
            if z.get("density_index", 1) <= 0:
                continue
            count = z.get("agents", 0)
            if count <= 0:
                continue
            pool = pool_all
            if zone_labels is not None:
                zid = z["zone_id"]
                zm = (zone_labels == zid) & self.walk_map.walkable
                zy, zx = np.where(zm)
                if len(zx) > 0:
                    pool = list(zip(zx.tolist(), zy.tolist()))
            for _ in range(count):
                px, py = random.choice(pool)
                self.agents.append(Agent(float(px), float(py)))

        print(f"Spawned {len(self.agents)} agents, {len(self.exit_positions)} exits")

    # Rebuild potential every N ticks
    _rebuild_interval = 6
    _tick = 0

    def step(self):
        dt = CFG["dt"]
        self.t += dt
        self._tick += 1

        # Update fields
        self.grid.splat_density(self.agents)
        self.grid.splat_discomfort(self.agents, dt)

        if self._tick % self._rebuild_interval == 0:
            self.grid.build_potential()

        # Move agents
        for a in self.agents:
            a.update(self.grid, dt)

        # Separation push (Treuille §4.5)
        self._enforce_separation()

        # Check exits
        er = CFG["exit_radius"]
        for a in self.agents:
            if a.evacuated:
                continue
            for ex, ey in self.exit_positions:
                if math.hypot(a.x - ex, a.y - ey) < er:
                    a.evacuated = True
                    self.evacuated_count += 1
                    break

    def _enforce_separation(self):
        r = CFG["separation_radius"]
        alive = [a for a in self.agents if not a.evacuated]
        # O(n²) — fine for <300 agents; bin if needed
        for i in range(len(alive)):
            for j in range(i+1, len(alive)):
                ai, aj = alive[i], alive[j]
                dx = ai.x - aj.x
                dy = ai.y - aj.y
                d = math.hypot(dx, dy)
                if 0 < d < r:
                    push = (r - d) / 2.0
                    nx, ny = dx/d, dy/d
                    ai.x += nx*push
                    ai.y += ny*push
                    aj.x -= nx*push
                    aj.y -= ny*push

    def reset(self, zone_config):
        self.t = 0.0
        self.evacuated_count = 0
        self._tick = 0
        self.agents.clear()
        self._spawn(zone_config)
        self.grid.build_potential()


# ── Render widget ─────────────────────────────────────────────────
class SimView(QWidget):
    def __init__(self):
        super().__init__()
        self.sim = None
        self.bg = None
        self.selected = None
        self.show_phi = False
        self.phi_overlay = None
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_sim(self, sim, mask_path):
        self.sim = sim
        img = cv2.imread(str(mask_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, c = img.shape
        self.bg = QPixmap.fromImage(QImage(img.tobytes(), w, h, w*c, QImage.Format.Format_RGB888))
        self._build_phi_overlay()
        self.update()

    def _build_phi_overlay(self):
        """Visualise the potential field as a coloured overlay."""
        if self.sim is None:
            return
        phi = self.sim.grid.phi
        valid = phi[np.isfinite(phi)]
        if len(valid) == 0:
            return
        mn, mx = valid.min(), valid.max()
        norm = np.zeros_like(phi, dtype=np.uint8)
        fin = np.isfinite(phi)
        if mx > mn:
            norm[fin] = ((phi[fin] - mn) / (mx - mn) * 255).astype(np.uint8)
        # Colourmap: low potential (near exit) = green, high = red
        coloured = cv2.applyColorMap(255 - norm, cv2.COLORMAP_JET)
        coloured[~fin] = [20, 20, 20]
        # Scale to full image size
        res = self.sim.grid.res
        full = cv2.resize(coloured,
                          (self.sim.grid.gw * res, self.sim.grid.gh * res),
                          interpolation=cv2.INTER_NEAREST)
        h, w, c = full.shape
        self.phi_overlay = QPixmap.fromImage(
            QImage(full.tobytes(), w, h, w*c, QImage.Format.Format_RGB888))

    def _layout(self):
        if self.bg is None:
            return 1.0, 0.0, 0.0
        s = min(self.width()/self.bg.width(), self.height()/self.bg.height())
        ox = (self.width()  - self.bg.width()*s) / 2
        oy = (self.height() - self.bg.height()*s) / 2
        return s, ox, oy

    def w2s(self, wx, wy):
        s, ox, oy = self._layout()
        return wx*s+ox, wy*s+oy

    def s2w(self, sx, sy):
        s, ox, oy = self._layout()
        return (sx-ox)/s, (sy-oy)/s

    def mousePressEvent(self, e):
        if self.sim is None:
            return
        wx, wy = self.s2w(e.position().x(), e.position().y())
        s, _, _ = self._layout()
        pick = 16/s
        best, bd = None, pick
        for a in self.sim.agents:
            d = math.hypot(a.x-wx, a.y-wy)
            if d < bd:
                bd, best = d, a
        if self.selected:
            self.selected.selected = False
        self.selected = best
        if best:
            best.selected = True
        self.update()

    def paintEvent(self, _):
        if self.sim is None:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        s, ox, oy = self._layout()
        from PyQt6.QtCore import QRectF
        iw, ih = self.bg.width(), self.bg.height()
        dest = QRectF(ox, oy, iw*s, ih*s).toRect()

        if self.show_phi and self.phi_overlay:
            p.setOpacity(0.55)
            p.drawPixmap(dest, self.phi_overlay)
            p.setOpacity(1.0)
        else:
            p.drawPixmap(dest, self.bg)

        # Draw exits
        for ex, ey in self.sim.exit_positions:
            sx, sy = self.w2s(ex, ey)
            r = max(5, CFG["exit_radius"]*s)
            p.setPen(QPen(QColor(0,200,80), 2))
            p.setBrush(QBrush(QColor(0,200,80,60)))
            p.drawEllipse(QPointF(sx, sy), r, r)

        # Draw density field (optional heat tint)
        # — skipped for performance; use phi overlay instead

        # Draw agents
        for a in self.sim.agents:
            if a.evacuated:
                continue
            ax, ay = self.w2s(a.x, a.y)
            r = max(3, CFG["agent_radius"]*s)
            if a.selected:
                p.setPen(QPen(QColor(255,255,0), 2))
                p.setBrush(QBrush(QColor(255,220,0,220)))
            else:
                # colour by potential (how far from exit)
                gx, gy = self.sim.grid.p2g(a.x, a.y)
                gx = int(np.clip(gx, 0, self.sim.grid.gw-1))
                gy = int(np.clip(gy, 0, self.sim.grid.gh-1))
                phi_v = self.sim.grid.phi[gy, gx]
                phi_max = self.sim.grid.phi[np.isfinite(self.sim.grid.phi)].max() + 1e-6
                t = np.clip(phi_v / phi_max, 0, 1) if np.isfinite(phi_v) else 1.0
                red  = int(220*t + 30*(1-t))
                blue = int(200*(1-t))
                p.setPen(QPen(QColor(0,0,0,80), 1))
                p.setBrush(QBrush(QColor(red, 100, blue, 200)))
            p.drawEllipse(QPointF(ax, ay), r, r)
            # velocity arrow
            spd = math.hypot(a.vx, a.vy)
            if spd > 1:
                ex_ = ax + (a.vx/spd)*r*1.8
                ey_ = ay + (a.vy/spd)*r*1.8
                p.setPen(QPen(QColor(255,255,255,180), max(1, r*0.35)))
                p.drawLine(QPointF(ax,ay), QPointF(ex_,ey_))
        p.end()


# ── Inspector ─────────────────────────────────────────────────────
class Inspector(QFrame):
    def __init__(self):
        super().__init__()
        self.setObjectName("card")
        self.setFixedWidth(240)
        lv = QVBoxLayout(self)
        lv.setContentsMargins(10,10,10,10); lv.setSpacing(5)
        title = QLabel("Inspector")
        title.setStyleSheet("font-size:12pt;font-weight:bold;color:#e94560;")
        lv.addWidget(title)
        self.labels = {}
        for k in ["id","pos","speed","phi","dist"]:
            lb = QLabel("—"); lb.setStyleSheet("color:#ccc;font-size:9pt;")
            lv.addWidget(lb); self.labels[k] = lb
        lv.addStretch()

    def update(self, agent, grid):
        if agent is None:
            for lb in self.labels.values(): lb.setText("—")
            return
        gx, gy = grid.p2g(agent.x, agent.y)
        gx = int(np.clip(gx, 0, grid.gw-1))
        gy = int(np.clip(gy, 0, grid.gh-1))
        phi_v = grid.phi[gy, gx]
        self.labels["id"].setText(f"Agent #{agent.id}")
        self.labels["pos"].setText(f"({agent.x:.0f}, {agent.y:.0f})")
        self.labels["speed"].setText(f"speed: {math.hypot(agent.vx,agent.vy):.1f} px/s")
        self.labels["phi"].setText(f"φ: {phi_v:.2f}" if np.isfinite(phi_v) else "φ: ∞")
        self.labels["dist"].setText(f"BFS dist: {grid.bfs_dist[gy,gx]:.0f} cells")


# ── Main Window ───────────────────────────────────────────────────
STYLE = """
QMainWindow,QWidget{background:#1a1a2e;color:#e0e0e0;
  font-family:'Segoe UI',Arial,sans-serif;font-size:10pt;}
QPushButton{background:#16213e;border:1px solid #0f3460;
  border-radius:5px;padding:6px 12px;color:#e0e0e0;}
QPushButton:hover{background:#0f3460;border-color:#e94560;}
QPushButton:pressed{background:#e94560;color:white;}
QPushButton#primary{background:#e94560;color:white;font-weight:bold;}
QFrame#card{background:#16213e;border:1px solid #0f3460;border-radius:8px;}
QLabel{color:#e0e0e0;}
QCheckBox{color:#ccc;}
QSlider::groove:horizontal{background:#0f3460;height:4px;border-radius:2px;}
QSlider::handle:horizontal{background:#e94560;width:14px;height:14px;
  margin:-5px 0;border-radius:7px;}
"""

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TRAGIC — Continuum Crowds")
        self.setMinimumSize(1200, 720)
        self.setStyleSheet(STYLE)
        self.sim = None
        self.zone_cfg = {}
        self.mask_path = ""
        self.paused = True
        self.timer = QTimer()
        self.timer.setInterval(int(CFG["dt"]*1000))
        self.timer.timeout.connect(self._tick)
        self._build_ui()

    def _build_ui(self):
        root = QWidget(); rl = QHBoxLayout(root)
        rl.setContentsMargins(12,12,12,12); rl.setSpacing(12)
        self.setCentralWidget(root)

        # Left panel
        left = QFrame(); left.setObjectName("card"); left.setFixedWidth(210)
        lv = QVBoxLayout(left); lv.setContentsMargins(10,10,10,10); lv.setSpacing(8)
        title = QLabel("<b style='font-size:13pt;color:#e94560'>TRAGIC</b><br>"
                       "<span style='font-size:8pt;color:#888'>Continuum Crowds</span>")
        title.setTextFormat(Qt.TextFormat.RichText); lv.addWidget(title)
        self.load_btn = QPushButton("📂 Load Config"); self.load_btn.clicked.connect(self.load); lv.addWidget(self.load_btn)
        self.run_btn  = QPushButton("▶ Start"); self.run_btn.setObjectName("primary")
        self.run_btn.setEnabled(False); self.run_btn.clicked.connect(self.toggle); lv.addWidget(self.run_btn)
        self.rst_btn  = QPushButton("↺ Reset"); self.rst_btn.setEnabled(False)
        self.rst_btn.clicked.connect(self.reset); lv.addWidget(self.rst_btn)
        lv.addWidget(self._sep())
        lv.addWidget(QLabel("Speed multiplier:"))
        self.spd = QSlider(Qt.Orientation.Horizontal); self.spd.setRange(1,10); self.spd.setValue(3)
        self.spd.valueChanged.connect(lambda v: self.timer.setInterval(max(8, int(CFG["dt"]*1000//v))))
        lv.addWidget(self.spd)
        lv.addWidget(self._sep())
        self.phi_chk = QCheckBox("Show potential field φ"); self.phi_chk.setChecked(False)
        self.phi_chk.toggled.connect(lambda v: setattr(self.view,'show_phi',v) or self.view.update())
        lv.addWidget(self.phi_chk)
        lv.addWidget(self._sep())
        self.stat = QLabel("Load a zone config\nto begin."); self.stat.setStyleSheet("color:#888;font-size:9pt;")
        self.stat.setWordWrap(True); lv.addWidget(self.stat)
        lv.addStretch()

        self.view = SimView()
        self.insp = Inspector()

        rl.addWidget(left); rl.addWidget(self.view, 1); rl.addWidget(self.insp)

    def _sep(self):
        f = QFrame(); f.setFrameShape(QFrame.Shape.HLine); f.setStyleSheet("color:#0f3460;"); return f

    def load(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Zone Config JSON","","JSON (*.json)")
        if not path: return
        with open(path) as f: self.zone_cfg = json.load(f)
        mask_path = self.zone_cfg.get("mask_path","")
        if not Path(mask_path).exists():
            mask_path, _ = QFileDialog.getOpenFileName(self,"Locate Mask Image","","Images (*.png *.jpg *.bmp)")
            if not mask_path: return
            self.zone_cfg["mask_path"] = mask_path
        self.mask_path = mask_path
        try:
            wm = WalkMap(mask_path)
        except Exception as e:
            QMessageBox.critical(self,"Error",str(e)); return
        self.sim = Simulation(wm, self.zone_cfg)
        self.view.set_sim(self.sim, mask_path)
        self.run_btn.setEnabled(True); self.rst_btn.setEnabled(True)
        self.paused = True; self.run_btn.setText("▶ Start")
        self._update_stats()

    def toggle(self):
        if self.sim is None: return
        self.paused = not self.paused
        if self.paused: self.timer.stop(); self.run_btn.setText("▶ Resume")
        else:           self.timer.start(); self.run_btn.setText("⏸ Pause")

    def reset(self):
        if self.sim is None: return
        self.timer.stop(); self.paused = True; self.run_btn.setText("▶ Start")
        self.sim.reset(self.zone_cfg); self.view.selected = None
        self.view._build_phi_overlay(); self.view.update(); self._update_stats()

    def _tick(self):
        if self.sim is None: return
        for _ in range(self.spd.value()):
            self.sim.step()
        if self.view.show_phi and self.sim._tick % 12 == 0:
            self.view._build_phi_overlay()
        self.view.update(); self._update_stats()
        if self.view.selected:
            self.insp.update(self.view.selected, self.sim.grid)

    def _update_stats(self):
        if self.sim is None: return
        n = len(self.sim.agents)
        ev = self.sim.evacuated_count
        alive = n - ev
        self.stat.setText(
            f"Time:      {self.sim.t:.1f}s\n"
            f"Agents:    {n}\n"
            f"Active:    {alive}\n"
            f"Evacuated: {ev} ({100*ev/max(1,n):.0f}%)\n\n"
            f"Space = pause  R = reset")
        if not self.view.selected:
            self.insp.update(None, self.sim.grid)

    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_Space: self.toggle()
        elif e.key() == Qt.Key.Key_R:   self.reset()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())