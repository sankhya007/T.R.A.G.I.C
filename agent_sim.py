"""
agent_sim.py  —  Step 3: Agent Simulation with FOV, Memory & Density Spawning
Run: python agent_sim.py

Requires:
  - A binary mask image (stitched_mask.png or similar)
  - A zone_config.json from zone_editor.py (Step 2)

Controls:
  Space     — Pause / Resume
  Click     — Inspect agent at that position
  R         — Reset simulation
"""

import sys
import json
import math
import random
import numpy as np
import cv2
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from navmesh import NavMesh

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QLabel, QFileDialog, QFrame, QSizePolicy, QScrollArea,
    QMessageBox, QCheckBox, QDoubleSpinBox, QSlider
)
from PyQt6.QtCore import Qt, QTimer, QPointF
from PyQt6.QtGui import (
    QPainter, QColor, QPen, QBrush, QPolygonF,
    QImage, QPixmap, QFont, QPainterPath
)

# ── Simulation constants ───────────────────────────────────────────────
DT              = 0.05      # seconds per tick
AGENT_RADIUS    = 6         # pixels
FOV_DEGREES     = 120       # total field of view
FOV_RANGE       = 80        # pixels — how far an agent can see
PANIC_FOV_RANGE = 140       # pixels — wider look during circle sweep
SPEED_BASE      = 35        # px/s nominal walk speed
SPEED_VARIANCE  = 0.25      # ±25% speed variation per agent
NEIGHBOR_RADIUS = 40        # px — social force range
WALL_BUFFER     = AGENT_RADIUS + 2
CIRCLE_SWEEP_INTERVAL = 3.0 # seconds — how often agents look around
MEMORY_DECAY    = 120.0     # seconds — exit memories fade after this
# ──────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════
#  Walkability map
# ══════════════════════════════════════════════════════════════════════

class WalkMap:
    """Thin wrapper around a binary walkability image."""
    def __init__(self, mask_path: str):
        img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(mask_path)
        # White pixels = wall, black = walkable (typical stitched mask)
        # Auto-detect convention by checking border pixels
        # border_mean = (img[0, :].mean() + img[-1, :].mean() +
        #                img[:, 0].mean() + img[:, -1].mean()) / 4
        # if border_mean > 128:
        #     # White border = wall, invert so walkable = True
        #     self.walkable = img < 128
        # else:
        #     self.walkable = img > 128
        
        # In your mask: white = wall, black = walkable
        # Force this convention directly, no auto-detection
        self.walkable = img < 128
        
        # this is done here tosee if teh agents spawn correctly n the zones in which they are meant to spawn in 
        
        self.h, self.w = img.shape
        # Pre-compute distance transform for wall-avoidance force
        walk_uint8 = self.walkable.astype(np.uint8) * 255
        self.dist = cv2.distanceTransform(walk_uint8, cv2.DIST_L2, 5)

    def is_walkable(self, x: float, y: float) -> bool:
        ix, iy = int(x), int(y)
        if ix < 0 or iy < 0 or ix >= self.w or iy >= self.h:
            return False
        return bool(self.walkable[iy, ix])

    def wall_repulsion(self, x: float, y: float) -> Tuple[float, float]:
        """Push agents away from walls based on distance transform."""
        ix, iy = int(np.clip(x, 0, self.w - 1)), int(np.clip(y, 0, self.h - 1))
        d = self.dist[iy, ix]
        if d >= WALL_BUFFER * 2:
            return 0.0, 0.0
        # Gradient of distance transform
        gx = (self.dist[iy, min(ix + 1, self.w - 1)] -
              self.dist[iy, max(ix - 1, 0)]) / 2
        gy = (self.dist[min(iy + 1, self.h - 1), ix] -
              self.dist[max(iy - 1, 0), ix]) / 2
        strength = max(0, (WALL_BUFFER * 2 - d)) / (WALL_BUFFER * 2)
        return gx * strength * 150, gy * strength * 150


# ══════════════════════════════════════════════════════════════════════
#  Memory entry
# ══════════════════════════════════════════════════════════════════════

@dataclass
class MemoryEntry:
    """A thing the agent has seen and remembered."""
    kind: str                   # "exit" | "hazard" | "crowd"
    position: Tuple[float, float]
    time_seen: float            # simulation time when first seen
    confidence: float = 1.0    # fades with time


# ══════════════════════════════════════════════════════════════════════
#  Agent
# ══════════════════════════════════════════════════════════════════════

class Agent:
    _id_counter = 0

    def __init__(self, x: float, y: float, walk_map: WalkMap):
        Agent._id_counter += 1
        self.id = Agent._id_counter

        self.x = x
        self.y = y
        self.speed = SPEED_BASE * random.uniform(1 - SPEED_VARIANCE, 1 + SPEED_VARIANCE)

        # Direction in radians (0 = right, π/2 = down)
        self.angle = random.uniform(0, math.tau)
        self.vx = math.cos(self.angle) * self.speed
        self.vy = math.sin(self.angle) * self.speed

        self.walk_map = walk_map
        self.memory: List[MemoryEntry] = []         # what this agent has seen
        self.known_map_cells: set = set()           # grid cells explored (10px grid)

        self.panic = 0.0                            # 0–1
        self.last_sweep_time = random.uniform(0, CIRCLE_SWEEP_INTERVAL)
        self.sim_time = 0.0

        # For smooth steering
        self._target_angle = self.angle
        self._wander_timer = random.uniform(0, 2.0)

        # For inspector display
        self.selected = False
        self.fov_visible_ids: set = set()           # IDs of agents in FOV right now

    # ── movement ─────────────────────────────────────────────────────

    def update(self, dt: float, all_agents: List['Agent'], sim_time: float):
        self.sim_time = sim_time

        # 1. Wander — smoothly change direction now and then
        self._wander_timer -= dt
        if self._wander_timer <= 0:
            self._target_angle = self.angle + random.gauss(0, 0.8)
            self._wander_timer = random.uniform(0.8, 2.5)

        # 2. Smooth steer toward target angle
        diff = (self._target_angle - self.angle + math.pi) % math.tau - math.pi
        self.angle += diff * min(1.0, dt * 4)
        self.angle %= math.tau

        # 3. Wall repulsion force
        rx, ry = self.walk_map.wall_repulsion(self.x, self.y)

        # 4. Agent–agent separation force (social force)
        sx, sy = 0.0, 0.0
        self.fov_visible_ids.clear()
        for other in all_agents:
            if other.id == self.id:
                continue
            dx, dy = self.x - other.x, self.y - other.y
            dist = math.hypot(dx, dy)
            if dist < NEIGHBOR_RADIUS and dist > 0:
                # Social separation
                push = max(0, (NEIGHBOR_RADIUS - dist)) / NEIGHBOR_RADIUS
                sx += (dx / dist) * push * 60
                sy += (dy / dist) * push * 60
            # FOV check
            if dist < FOV_RANGE and self._in_fov(other.x, other.y):
                self.fov_visible_ids.add(other.id)

        # 5. Compose velocity
        wx = math.cos(self.angle) * self.speed + rx + sx
        wy = math.sin(self.angle) * self.speed + ry + sy

        # 6. Propose new position, reject if into wall
        nx, ny = self.x + wx * dt, self.y + wy * dt
        if self.walk_map.is_walkable(nx, ny):
            self.x, self.y = nx, ny
        else:
            # Bounce — try axis-aligned slides
            if self.walk_map.is_walkable(nx, self.y):
                self.x = nx
                self._target_angle = math.atan2(-wy, wx)
            elif self.walk_map.is_walkable(self.x, ny):
                self.y = ny
                self._target_angle = math.atan2(wy, -wx)
            else:
                self._target_angle = self.angle + math.pi + random.uniform(-0.5, 0.5)

        # 7. Update explored cells (10px grid)
        cell = (int(self.x) // 10, int(self.y) // 10)
        self.known_map_cells.add(cell)

        # 8. Periodic circle sweep — scan all around
        if sim_time - self.last_sweep_time >= CIRCLE_SWEEP_INTERVAL:
            self._circle_sweep(all_agents, sim_time)
            self.last_sweep_time = sim_time

        # 9. Decay old memories
        self.memory = [
            m for m in self.memory
            if sim_time - m.time_seen < MEMORY_DECAY
        ]

    def _in_fov(self, tx: float, ty: float) -> bool:
        """Is target (tx, ty) within this agent's field of view?"""
        dx, dy = tx - self.x, ty - self.y
        angle_to = math.atan2(dy, dx)
        diff = (angle_to - self.angle + math.pi) % math.tau - math.pi
        return abs(diff) <= math.radians(FOV_DEGREES / 2)

    def _circle_sweep(self, all_agents: List['Agent'], sim_time: float):
        """
        Look in all directions. Record any exits or hazards in memory.
        In this step we just track other agents seen during the sweep
        (exits/hazards would be added by the simulation manager later).
        """
        # Find any agents at panic-sweep range in all directions
        for other in all_agents:
            if other.id == self.id:
                continue
            dist = math.hypot(self.x - other.x, self.y - other.y)
            if dist < PANIC_FOV_RANGE:
                # Store position as "crowd" memory
                self._add_memory("crowd", (other.x, other.y), sim_time)

    def _add_memory(self, kind: str, pos: Tuple[float, float], t: float):
        """Add or refresh memory entry (deduplicates within 20px)."""
        for m in self.memory:
            if m.kind == kind and math.hypot(m.position[0] - pos[0],
                                              m.position[1] - pos[1]) < 20:
                m.time_seen = t
                m.confidence = 1.0
                return
        self.memory.append(MemoryEntry(kind, pos, t))

    def add_exit_memory(self, pos: Tuple[float, float], t: float):
        self._add_memory("exit", pos, t)

    def add_hazard_memory(self, pos: Tuple[float, float], t: float):
        self._add_memory("hazard", pos, t)


# ══════════════════════════════════════════════════════════════════════
#  Simulation Manager
# ══════════════════════════════════════════════════════════════════════

class Simulation:
    def __init__(self, walk_map: WalkMap, zone_config: dict):
        self.walk_map = walk_map
        self.agents: List[Agent] = []
        self.sim_time = 0.0
        nm_path = Path("navmesh.json")
        self.navmesh = NavMesh(
            walk_map,
            cell_size=3,
            clearance=0,
            walkable_threshold=0.50,
            diagonal=True,
        )

        if nm_path.exists():
            try:
                self.navmesh.load("navmesh.json")
            except Exception as e:
                print(f"[NavGrid] Cache load failed: {e}")
                print("[NavGrid] Rebuilding navmesh...")
                self.navmesh.build()
                self.navmesh.save("navmesh.json")
        else:
            self.navmesh.build()
            self.navmesh.save("navmesh.json")
        self._spawn_agents(zone_config)

    def _spawn_agents(self, cfg: dict):
        Agent._id_counter = 0

        zones = cfg.get("zones", [])

        if not zones:
            self._scatter_random(100)
            return

        # All walkable pixels — pre-sampled once
        ys, xs = np.where(self.walk_map.walkable)
        if len(xs) == 0:
            print("Warning: no walkable pixels found in mask!")
            return

        # Try to rebuild zone labels so agents spawn inside the right zone.
        # If that fails we fall back to scattering across ALL walkable pixels
        # (still correct count, just not zone-localised).
        mask_path = cfg.get("mask_path", "")
        zone_labels = None
        if Path(mask_path).exists():
            try:
                zone_labels = self._rebuild_labels(mask_path)
                print("Zone labels rebuilt successfully")
            except Exception as e:
                print(f"Zone label rebuild failed ({e}), using global walkable pool")

        # Build a flat list of walkable pixel indices for fast random sampling
        walkable_pool = list(zip(xs.tolist(), ys.tolist()))

        for zone in zones:
            d = zone.get("density_index", 1.0)
            if d <= 0:
                continue   # density_index 0 = outside/ignore

            count = zone.get("agents", 0)
            if count <= 0:
                continue

            # Pick the pixel pool for this zone
            if zone_labels is not None:
                zid = zone["zone_id"]
                zm = (zone_labels == zid) & self.walk_map.walkable
                zy, zx = np.where(zm)
                if len(zx) == 0:
                    pool = walkable_pool   # fallback
                else:
                    pool = list(zip(zx.tolist(), zy.tolist()))
            else:
                pool = walkable_pool

            for _ in range(count):
                px, py = random.choice(pool)
                self.agents.append(Agent(float(px), float(py), self.walk_map))

        print(f"Spawned {len(self.agents)} agents")

    def _scatter_random(self, n: int):
        ys, xs = np.where(self.walk_map.walkable)
        for _ in range(n):
            idx = random.randint(0, len(xs) - 1)
            self.agents.append(Agent(float(xs[idx]), float(ys[idx]), self.walk_map))

    def _rebuild_labels(self, mask_path: str):
        """Rebuild zone label map using same watershed as zone_editor."""
        import cv2
        from scipy import ndimage as ndi
        from skimage.segmentation import watershed
        from skimage.feature import peak_local_max

        img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        walkable = cv2.bitwise_not(img)
        _, binary = cv2.threshold(walkable, 127, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        dist_norm = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)
        coords = peak_local_max(dist_norm, min_distance=40, labels=binary)
        seed_mask = np.zeros(dist_norm.shape, dtype=bool)
        seed_mask[tuple(coords.T)] = True
        markers, _ = ndi.label(seed_mask)
        labels = watershed(-dist, markers, mask=binary)
        return labels

    def step(self):
        self.sim_time += DT
        for agent in self.agents:
            agent.update(DT, self.agents, self.sim_time)

    def reset(self, zone_config: dict):
        self.agents.clear()
        self.sim_time = 0.0
        self._spawn_agents(zone_config)


# ══════════════════════════════════════════════════════════════════════
#  Render widget
# ══════════════════════════════════════════════════════════════════════

class SimView(QWidget):
    def __init__(self):
        super().__init__()
        self.sim: Optional[Simulation] = None
        self.bg_pixmap: Optional[QPixmap] = None
        self.selected_agent: Optional[Agent] = None
        self.show_fov       = True
        self.show_memory    = True
        self.show_explored  = False
        self.show_navmesh = False
        self.setMinimumSize(400, 400)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)

    def set_sim(self, sim: Simulation, mask_path: str):
        self.sim = sim
        # Build background from mask
        img = cv2.imread(mask_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, c = img.shape
        qimg = QImage(img.tobytes(), w, h, w * c, QImage.Format.Format_RGB888)
        self.bg_pixmap = QPixmap.fromImage(qimg)
        self.update()

    def _layout(self):
        """Uniform scale + letterbox offsets so image keeps its aspect ratio."""
        if self.bg_pixmap is None:
            return 1.0, 0.0, 0.0
        iw, ih = self.bg_pixmap.width(), self.bg_pixmap.height()
        s = min(self.width() / iw, self.height() / ih)
        ox = (self.width()  - iw * s) / 2
        oy = (self.height() - ih * s) / 2
        return s, ox, oy

    def _w2s(self, wx, wy):
        """World → screen coords."""
        s, ox, oy = self._layout()
        return wx * s + ox, wy * s + oy

    def _s2w(self, sx, sy):
        """Screen → world coords."""
        s, ox, oy = self._layout()
        return (sx - ox) / s, (sy - oy) / s

    def mousePressEvent(self, event):
        if self.sim is None:
            return
        wx, wy = self._s2w(event.position().x(), event.position().y())
        s, _, _ = self._layout()
        pick_r = 20 / s
        best, best_d = None, pick_r
        for a in self.sim.agents:
            d = math.hypot(a.x - wx, a.y - wy)
            if d < best_d:
                best_d, best = d, a
        if self.selected_agent:
            self.selected_agent.selected = False
        self.selected_agent = best
        if best:
            best.selected = True
        self.update()

    def paintEvent(self, _):
        if self.sim is None:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        s, ox, oy = self._layout()

        # Background — letterboxed
        if self.bg_pixmap:
            from PyQt6.QtCore import QRectF
            iw, ih = self.bg_pixmap.width(), self.bg_pixmap.height()
            p.drawPixmap(QRectF(ox, oy, iw * s, ih * s).toRect(), self.bg_pixmap)

        if self.show_navmesh and self.sim and hasattr(self.sim, 'navmesh'):
            s, ox, oy = self._layout()
            self.sim.navmesh.draw_debug(p, s, ox, oy)
            
        # Explored overlay
        if self.show_explored and self.selected_agent:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(QColor(100, 200, 255, 40)))
            for (cx, cy) in self.selected_agent.known_map_cells:
                scx, scy = self._w2s(cx * 10, cy * 10)
                p.drawRect(int(scx), int(scy), int(10 * s), int(10 * s))

        for a in self.sim.agents:
            ax, ay = self._w2s(a.x, a.y)
            r = max(3, AGENT_RADIUS * s)

            # FOV cone
            if self.show_fov and (a.selected or len(self.sim.agents) <= 80):
                fov_range_scaled = FOV_RANGE * s
                fov_half = math.radians(FOV_DEGREES / 2)
                path = QPainterPath()
                path.moveTo(ax, ay)
                steps = 12
                for i in range(steps + 1):
                    t = -fov_half + i * (2 * fov_half / steps)
                    angle = a.angle + t
                    path.lineTo(ax + math.cos(angle) * fov_range_scaled,
                                ay + math.sin(angle) * fov_range_scaled)
                path.closeSubpath()
                p.setPen(Qt.PenStyle.NoPen)
                alpha = 60 if a.selected else 18
                p.setBrush(QBrush(QColor(255, 255, 180, alpha)))
                p.drawPath(path)

            # Memory dots (selected agent only)
            if self.show_memory and a.selected:
                for m in a.memory:
                    mx2, my2 = self._w2s(m.position[0], m.position[1])
                    age = (self.sim.sim_time - m.time_seen) / MEMORY_DECAY
                    alpha = int(180 * (1 - age))
                    if m.kind == "exit":
                        color = QColor(0, 255, 100, alpha)
                    elif m.kind == "hazard":
                        color = QColor(255, 80, 0, alpha)
                    else:
                        color = QColor(180, 180, 255, alpha)
                    p.setPen(QPen(color, 2))
                    p.setBrush(QBrush(color))
                    p.drawEllipse(QPointF(mx2, my2), 4, 4)

            # Agent body
            if a.selected:
                p.setPen(QPen(QColor(255, 255, 0), 2))
                p.setBrush(QBrush(QColor(255, 220, 0, 220)))
            else:
                red  = int(200 * a.panic + 60 * (1 - a.panic))
                blue = int(180 * (1 - a.panic))
                p.setPen(QPen(QColor(0, 0, 0, 120), 1))
                p.setBrush(QBrush(QColor(red, 100, blue, 200)))
            p.drawEllipse(QPointF(ax, ay), r, r)

            # Direction arrow
            arrow_len = r * 1.8
            ex = ax + math.cos(a.angle) * arrow_len
            ey = ay + math.sin(a.angle) * arrow_len
            p.setPen(QPen(QColor(255, 255, 255, 200), max(1, r * 0.4)))
            p.drawLine(QPointF(ax, ay), QPointF(ex, ey))

        p.end()


# ══════════════════════════════════════════════════════════════════════
#  Inspector panel (shows selected agent's memory)
# ══════════════════════════════════════════════════════════════════════

class InspectorPanel(QFrame):
    def __init__(self):
        super().__init__()
        self.setObjectName("card")
        self.setFixedWidth(260)
        lv = QVBoxLayout(self)
        lv.setContentsMargins(12, 12, 12, 12)
        lv.setSpacing(6)

        title = QLabel("Agent Inspector")
        title.setStyleSheet("font-size:13pt; font-weight:bold; color:#e94560;")
        lv.addWidget(title)

        self.id_label    = QLabel("Click an agent to inspect")
        self.pos_label   = QLabel("")
        self.speed_label = QLabel("")
        self.panic_label = QLabel("")
        self.cells_label = QLabel("")
        for lbl in [self.id_label, self.pos_label, self.speed_label,
                    self.panic_label, self.cells_label]:
            lbl.setStyleSheet("color:#ccc; font-size:9pt;")
            lv.addWidget(lbl)

        sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color:#333;")
        lv.addWidget(sep)

        lv.addWidget(QLabel("Memory:").setVisible(False) or QLabel("Memory:"))
        self.memory_scroll = QScrollArea()
        self.memory_scroll.setWidgetResizable(True)
        self.memory_content = QWidget()
        self.memory_layout  = QVBoxLayout(self.memory_content)
        self.memory_layout.setSpacing(2)
        self.memory_layout.setContentsMargins(0, 0, 0, 0)
        self.memory_scroll.setWidget(self.memory_content)
        self.memory_scroll.setStyleSheet("background:#111; border:none;")
        lv.addWidget(self.memory_scroll, 1)
        lv.addStretch()

    def update_agent(self, agent: Optional['Agent'], sim_time: float):
        if agent is None:
            self.id_label.setText("Click an agent to inspect")
            self.pos_label.setText("")
            self.speed_label.setText("")
            self.panic_label.setText("")
            self.cells_label.setText("")
            self._clear_memory()
            return

        self.id_label.setText(f"Agent #{agent.id}")
        self.pos_label.setText(f"Position:  ({agent.x:.0f}, {agent.y:.0f})")
        self.speed_label.setText(f"Speed:      {agent.speed:.1f} px/s")
        self.panic_label.setText(f"Panic:      {agent.panic:.2f}")
        self.cells_label.setText(f"Explored:   {len(agent.known_map_cells)} cells")

        self._clear_memory()
        if not agent.memory:
            lbl = QLabel("  (nothing yet)")
            lbl.setStyleSheet("color:#555; font-size:8pt;")
            self.memory_layout.addWidget(lbl)
        for m in sorted(agent.memory, key=lambda x: -x.time_seen):
            age = sim_time - m.time_seen
            text = (f"  [{m.kind:6s}]  ({m.position[0]:.0f},{m.position[1]:.0f})"
                    f"  {age:.1f}s ago")
            if m.kind == "exit":    color = "#00ff66"
            elif m.kind == "hazard": color = "#ff6040"
            else:                    color = "#aaaaff"
            lbl = QLabel(text)
            lbl.setStyleSheet(f"color:{color}; font-size:8pt; font-family:monospace;")
            self.memory_layout.addWidget(lbl)
        self.memory_layout.addStretch()

    def _clear_memory(self):
        while self.memory_layout.count():
            item = self.memory_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()


# ══════════════════════════════════════════════════════════════════════
#  Main Window
# ══════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):

    STYLE = """
    QMainWindow, QWidget { background:#1a1a2e; color:#e0e0e0;
        font-family:'Segoe UI',Arial,sans-serif; font-size:10pt; }
    QPushButton { background:#16213e; border:1px solid #0f3460;
        border-radius:5px; padding:6px 12px; color:#e0e0e0; }
    QPushButton:hover  { background:#0f3460; border-color:#e94560; }
    QPushButton:pressed{ background:#e94560; color:white; }
    QPushButton#primary{ background:#e94560; color:white; font-weight:bold; }
    QFrame#card { background:#16213e; border:1px solid #0f3460; border-radius:8px; }
    QLabel { color:#e0e0e0; }
    QCheckBox { color:#ccc; }
    QSlider::groove:horizontal { background:#0f3460; height:4px; border-radius:2px; }
    QSlider::handle:horizontal { background:#e94560; width:14px; height:14px;
        margin:-5px 0; border-radius:7px; }
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("TRAGIC — Step 3: Agent Simulation")
        self.setMinimumSize(1200, 750)
        self.setStyleSheet(self.STYLE)

        self.sim: Optional[Simulation] = None
        self.zone_config: dict = {}
        self.mask_path: str = ""
        self.paused = True

        self.timer = QTimer()
        self.timer.setInterval(int(DT * 1000))
        self.timer.timeout.connect(self._tick)

        self._build_ui()

    def _build_ui(self):
        root = QWidget()
        rl = QHBoxLayout(root)
        rl.setContentsMargins(12, 12, 12, 12)
        rl.setSpacing(12)
        self.setCentralWidget(root)

        # ── Left controls ──
        left = QFrame(); left.setObjectName("card"); left.setFixedWidth(220)
        lv = QVBoxLayout(left)
        lv.setContentsMargins(10, 10, 10, 10)
        lv.setSpacing(8)

        QLabel("<b style='font-size:13pt;color:#e94560'>TRAGIC</b><br>"
               "<span style='font-size:9pt;color:#888'>Step 3 — Agent Sim</span>").setParent(left)
        title = QLabel("<b style='font-size:13pt;color:#e94560'>TRAGIC</b><br>"
                       "<span style='font-size:9pt;color:#888'>Step 3 — Agent Sim</span>")
        title.setTextFormat(Qt.TextFormat.RichText)
        lv.addWidget(title)

        self.load_btn = QPushButton("📂 Load Zone Config")
        self.load_btn.clicked.connect(self.load_config)
        lv.addWidget(self.load_btn)

        self.run_btn = QPushButton("▶ Start")
        self.run_btn.setObjectName("primary")
        self.run_btn.setEnabled(False)
        self.run_btn.clicked.connect(self.toggle_pause)
        lv.addWidget(self.run_btn)

        self.reset_btn = QPushButton("↺ Reset")
        self.reset_btn.setEnabled(False)
        self.reset_btn.clicked.connect(self.reset_sim)
        lv.addWidget(self.reset_btn)

        lv.addWidget(self._sep())
        lv.addWidget(QLabel("Simulation Speed:"))
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(1, 10)
        self.speed_slider.setValue(3)
        self.speed_slider.valueChanged.connect(self._update_speed)
        lv.addWidget(self.speed_slider)

        lv.addWidget(self._sep())

        self.fov_check = QCheckBox("Show FOV cones")
        self.fov_check.setChecked(True)
        self.fov_check.toggled.connect(lambda v: setattr(self.sim_view, 'show_fov', v) or self.sim_view.update())
        lv.addWidget(self.fov_check)

        self.mem_check = QCheckBox("Show agent memory")
        self.mem_check.setChecked(True)
        self.mem_check.toggled.connect(lambda v: setattr(self.sim_view, 'show_memory', v) or self.sim_view.update())
        lv.addWidget(self.mem_check)

        self.exp_check = QCheckBox("Show explored area")
        self.exp_check.setChecked(False)
        self.exp_check.toggled.connect(lambda v: setattr(self.sim_view, 'show_explored', v) or self.sim_view.update())
        lv.addWidget(self.exp_check)

        self.nav_check = QCheckBox("Show NavMesh")
        self.nav_check.setChecked(False)
        self.nav_check.toggled.connect(lambda v: setattr(self.sim_view, 'show_navmesh', v) or self.sim_view.update())
        lv.addWidget(self.nav_check)

        lv.addWidget(self._sep())

        self.stat_label = QLabel("Load a zone config\nto begin.")
        self.stat_label.setStyleSheet("color:#888; font-size:9pt;")
        self.stat_label.setWordWrap(True)
        lv.addWidget(self.stat_label)
        lv.addStretch()

        # ── Sim view ──
        self.sim_view = SimView()

        # ── Inspector ──
        self.inspector = InspectorPanel()

        rl.addWidget(left)
        rl.addWidget(self.sim_view, 1)
        rl.addWidget(self.inspector)

    def _sep(self):
        sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color:#0f3460;")
        return sep

    def load_config(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Zone Config JSON", "", "JSON (*.json)")
        if not path:
            return

        with open(path) as f:
            self.zone_config = json.load(f)

        mask_path = self.zone_config.get("mask_path", "")
        if not Path(mask_path).exists():
            # Let user pick mask manually
            mask_path, _ = QFileDialog.getOpenFileName(
                self, "Locate Mask Image", "", "Images (*.png *.jpg *.bmp)")
            if not mask_path:
                return
            self.zone_config["mask_path"] = mask_path

        self.mask_path = mask_path
        try:
            walk_map = WalkMap(mask_path)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            return

        self.sim = Simulation(walk_map, self.zone_config)
        self.sim_view.set_sim(self.sim, mask_path)
        self.run_btn.setEnabled(True)
        self.reset_btn.setEnabled(True)
        self._update_stats()
        self.paused = True
        self.run_btn.setText("▶ Start")

    def toggle_pause(self):
        if self.sim is None:
            return
        self.paused = not self.paused
        if self.paused:
            self.timer.stop()
            self.run_btn.setText("▶ Resume")
        else:
            self.timer.start()
            self.run_btn.setText("⏸ Pause")

    def reset_sim(self):
        if self.sim is None:
            return
        self.timer.stop()
        self.paused = True
        self.run_btn.setText("▶ Start")
        self.sim.reset(self.zone_config)
        self.sim_view.selected_agent = None
        self.sim_view.update()
        self._update_stats()

    def _tick(self):
        if self.sim is None:
            return
        steps = self.speed_slider.value()
        for _ in range(steps):
            self.sim.step()
        self.sim_view.update()
        self._update_stats()
        # Refresh inspector if agent selected
        if self.sim_view.selected_agent:
            self.inspector.update_agent(self.sim_view.selected_agent, self.sim.sim_time)

    def _update_speed(self, v):
        self.timer.setInterval(max(10, int(DT * 1000 // v)))

    def _update_stats(self):
        if self.sim is None:
            return
        n = len(self.sim.agents)
        t = self.sim.sim_time
        explored = sum(len(a.known_map_cells) for a in self.sim.agents)
        self.stat_label.setText(
            f"Time:     {t:.1f} s\n"
            f"Agents:   {n}\n"
            f"Explored: {explored} cells total\n\n"
            f"Click agent to inspect.\nSpace = pause.")
        # Update inspector if no agent selected
        if not self.sim_view.selected_agent:
            self.inspector.update_agent(None, t)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Space:
            self.toggle_pause()
        elif event.key() == Qt.Key.Key_R:
            self.reset_sim()


# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())











