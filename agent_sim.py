# this did not spawn any agents in the given json file which had the config right

# """
# agent_sim.py  —  Step 3: Agent Simulation with FOV, Memory & Density Spawning
# Run: python agent_sim.py

# Requires:
#   - A binary mask image (stitched_mask.png or similar)
#   - A zone_config.json from zone_editor.py (Step 2)

# Controls:
#   Space     — Pause / Resume
#   Click     — Inspect agent at that position
#   R         — Reset simulation
# """

# import sys
# import json
# import math
# import random
# import numpy as np
# import cv2
# from pathlib import Path
# from dataclasses import dataclass, field
# from typing import List, Tuple, Optional, Dict

# from PyQt6.QtWidgets import (
#     QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
#     QPushButton, QLabel, QFileDialog, QFrame, QSizePolicy, QScrollArea,
#     QMessageBox, QCheckBox, QDoubleSpinBox, QSlider
# )
# from PyQt6.QtCore import Qt, QTimer, QPointF
# from PyQt6.QtGui import (
#     QPainter, QColor, QPen, QBrush, QPolygonF,
#     QImage, QPixmap, QFont, QPainterPath
# )

# # ── Simulation constants ───────────────────────────────────────────────
# DT              = 0.05      # seconds per tick
# AGENT_RADIUS    = 6         # pixels
# FOV_DEGREES     = 120       # total field of view
# FOV_RANGE       = 80        # pixels — how far an agent can see
# PANIC_FOV_RANGE = 140       # pixels — wider look during circle sweep
# SPEED_BASE      = 35        # px/s nominal walk speed
# SPEED_VARIANCE  = 0.25      # ±25% speed variation per agent
# NEIGHBOR_RADIUS = 40        # px — social force range
# WALL_BUFFER     = AGENT_RADIUS + 2
# CIRCLE_SWEEP_INTERVAL = 3.0 # seconds — how often agents look around
# MEMORY_DECAY    = 120.0     # seconds — exit memories fade after this
# # ──────────────────────────────────────────────────────────────────────


# # ══════════════════════════════════════════════════════════════════════
# #  Walkability map
# # ══════════════════════════════════════════════════════════════════════

# class WalkMap:
#     """Thin wrapper around a binary walkability image."""
#     def __init__(self, mask_path: str):
#         img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
#         if img is None:
#             raise FileNotFoundError(mask_path)
#         # White pixels = wall, black = walkable (typical stitched mask)
#         # Auto-detect convention by checking border pixels
#         border_mean = (img[0, :].mean() + img[-1, :].mean() +
#                        img[:, 0].mean() + img[:, -1].mean()) / 4
#         if border_mean > 128:
#             # White border = wall, invert so walkable = True
#             self.walkable = img < 128
#         else:
#             self.walkable = img > 128
#         self.h, self.w = img.shape
#         # Pre-compute distance transform for wall-avoidance force
#         walk_uint8 = self.walkable.astype(np.uint8) * 255
#         self.dist = cv2.distanceTransform(walk_uint8, cv2.DIST_L2, 5)

#     def is_walkable(self, x: float, y: float) -> bool:
#         ix, iy = int(x), int(y)
#         if ix < 0 or iy < 0 or ix >= self.w or iy >= self.h:
#             return False
#         return bool(self.walkable[iy, ix])

#     def wall_repulsion(self, x: float, y: float) -> Tuple[float, float]:
#         """Push agents away from walls based on distance transform."""
#         ix, iy = int(np.clip(x, 0, self.w - 1)), int(np.clip(y, 0, self.h - 1))
#         d = self.dist[iy, ix]
#         if d >= WALL_BUFFER * 2:
#             return 0.0, 0.0
#         # Gradient of distance transform
#         gx = (self.dist[iy, min(ix + 1, self.w - 1)] -
#               self.dist[iy, max(ix - 1, 0)]) / 2
#         gy = (self.dist[min(iy + 1, self.h - 1), ix] -
#               self.dist[max(iy - 1, 0), ix]) / 2
#         strength = max(0, (WALL_BUFFER * 2 - d)) / (WALL_BUFFER * 2)
#         return gx * strength * 150, gy * strength * 150


# # ══════════════════════════════════════════════════════════════════════
# #  Memory entry
# # ══════════════════════════════════════════════════════════════════════

# @dataclass
# class MemoryEntry:
#     """A thing the agent has seen and remembered."""
#     kind: str                   # "exit" | "hazard" | "crowd"
#     position: Tuple[float, float]
#     time_seen: float            # simulation time when first seen
#     confidence: float = 1.0    # fades with time


# # ══════════════════════════════════════════════════════════════════════
# #  Agent
# # ══════════════════════════════════════════════════════════════════════

# class Agent:
#     _id_counter = 0

#     def __init__(self, x: float, y: float, walk_map: WalkMap):
#         Agent._id_counter += 1
#         self.id = Agent._id_counter

#         self.x = x
#         self.y = y
#         self.speed = SPEED_BASE * random.uniform(1 - SPEED_VARIANCE, 1 + SPEED_VARIANCE)

#         # Direction in radians (0 = right, π/2 = down)
#         self.angle = random.uniform(0, math.tau)
#         self.vx = math.cos(self.angle) * self.speed
#         self.vy = math.sin(self.angle) * self.speed

#         self.walk_map = walk_map
#         self.memory: List[MemoryEntry] = []         # what this agent has seen
#         self.known_map_cells: set = set()           # grid cells explored (10px grid)

#         self.panic = 0.0                            # 0–1
#         self.last_sweep_time = random.uniform(0, CIRCLE_SWEEP_INTERVAL)
#         self.sim_time = 0.0

#         # For smooth steering
#         self._target_angle = self.angle
#         self._wander_timer = random.uniform(0, 2.0)

#         # For inspector display
#         self.selected = False
#         self.fov_visible_ids: set = set()           # IDs of agents in FOV right now

#     # ── movement ─────────────────────────────────────────────────────

#     def update(self, dt: float, all_agents: List['Agent'], sim_time: float):
#         self.sim_time = sim_time

#         # 1. Wander — smoothly change direction now and then
#         self._wander_timer -= dt
#         if self._wander_timer <= 0:
#             self._target_angle = self.angle + random.gauss(0, 0.8)
#             self._wander_timer = random.uniform(0.8, 2.5)

#         # 2. Smooth steer toward target angle
#         diff = (self._target_angle - self.angle + math.pi) % math.tau - math.pi
#         self.angle += diff * min(1.0, dt * 4)
#         self.angle %= math.tau

#         # 3. Wall repulsion force
#         rx, ry = self.walk_map.wall_repulsion(self.x, self.y)

#         # 4. Agent–agent separation force (social force)
#         sx, sy = 0.0, 0.0
#         self.fov_visible_ids.clear()
#         for other in all_agents:
#             if other.id == self.id:
#                 continue
#             dx, dy = self.x - other.x, self.y - other.y
#             dist = math.hypot(dx, dy)
#             if dist < NEIGHBOR_RADIUS and dist > 0:
#                 # Social separation
#                 push = max(0, (NEIGHBOR_RADIUS - dist)) / NEIGHBOR_RADIUS
#                 sx += (dx / dist) * push * 60
#                 sy += (dy / dist) * push * 60
#             # FOV check
#             if dist < FOV_RANGE and self._in_fov(other.x, other.y):
#                 self.fov_visible_ids.add(other.id)

#         # 5. Compose velocity
#         wx = math.cos(self.angle) * self.speed + rx + sx
#         wy = math.sin(self.angle) * self.speed + ry + sy

#         # 6. Propose new position, reject if into wall
#         nx, ny = self.x + wx * dt, self.y + wy * dt
#         if self.walk_map.is_walkable(nx, ny):
#             self.x, self.y = nx, ny
#         else:
#             # Bounce — try axis-aligned slides
#             if self.walk_map.is_walkable(nx, self.y):
#                 self.x = nx
#                 self._target_angle = math.atan2(-wy, wx)
#             elif self.walk_map.is_walkable(self.x, ny):
#                 self.y = ny
#                 self._target_angle = math.atan2(wy, -wx)
#             else:
#                 self._target_angle = self.angle + math.pi + random.uniform(-0.5, 0.5)

#         # 7. Update explored cells (10px grid)
#         cell = (int(self.x) // 10, int(self.y) // 10)
#         self.known_map_cells.add(cell)

#         # 8. Periodic circle sweep — scan all around
#         if sim_time - self.last_sweep_time >= CIRCLE_SWEEP_INTERVAL:
#             self._circle_sweep(all_agents, sim_time)
#             self.last_sweep_time = sim_time

#         # 9. Decay old memories
#         self.memory = [
#             m for m in self.memory
#             if sim_time - m.time_seen < MEMORY_DECAY
#         ]

#     def _in_fov(self, tx: float, ty: float) -> bool:
#         """Is target (tx, ty) within this agent's field of view?"""
#         dx, dy = tx - self.x, ty - self.y
#         angle_to = math.atan2(dy, dx)
#         diff = (angle_to - self.angle + math.pi) % math.tau - math.pi
#         return abs(diff) <= math.radians(FOV_DEGREES / 2)

#     def _circle_sweep(self, all_agents: List['Agent'], sim_time: float):
#         """
#         Look in all directions. Record any exits or hazards in memory.
#         In this step we just track other agents seen during the sweep
#         (exits/hazards would be added by the simulation manager later).
#         """
#         # Find any agents at panic-sweep range in all directions
#         for other in all_agents:
#             if other.id == self.id:
#                 continue
#             dist = math.hypot(self.x - other.x, self.y - other.y)
#             if dist < PANIC_FOV_RANGE:
#                 # Store position as "crowd" memory
#                 self._add_memory("crowd", (other.x, other.y), sim_time)

#     def _add_memory(self, kind: str, pos: Tuple[float, float], t: float):
#         """Add or refresh memory entry (deduplicates within 20px)."""
#         for m in self.memory:
#             if m.kind == kind and math.hypot(m.position[0] - pos[0],
#                                               m.position[1] - pos[1]) < 20:
#                 m.time_seen = t
#                 m.confidence = 1.0
#                 return
#         self.memory.append(MemoryEntry(kind, pos, t))

#     def add_exit_memory(self, pos: Tuple[float, float], t: float):
#         self._add_memory("exit", pos, t)

#     def add_hazard_memory(self, pos: Tuple[float, float], t: float):
#         self._add_memory("hazard", pos, t)


# # ══════════════════════════════════════════════════════════════════════
# #  Simulation Manager
# # ══════════════════════════════════════════════════════════════════════

# class Simulation:
#     def __init__(self, walk_map: WalkMap, zone_config: dict):
#         self.walk_map = walk_map
#         self.agents: List[Agent] = []
#         self.sim_time = 0.0
#         self._spawn_agents(zone_config)

#     def _spawn_agents(self, cfg: dict):
#         Agent._id_counter = 0
#         w, h = self.walk_map.w, self.walk_map.h

#         # Load zone label image to know which pixel belongs to which zone
#         # We don't have it here, so we fall back to proportional random spawn
#         # across walkable pixels, weighted by density_index per zone.
#         # If zone_config has mask_path, we regenerate zone labels.
#         zones = cfg.get("zones", [])
#         base_density = cfg.get("base_density", 1.0)
#         agent_scale  = cfg.get("agent_scale", 1000)

#         if not zones:
#             # Fallback: just scatter agents randomly
#             self._scatter_random(100)
#             return

#         # Collect all walkable pixels
#         ys, xs = np.where(self.walk_map.walkable)
#         if len(xs) == 0:
#             return

#         # Try to re-segment zones to get pixel->zone mapping
#         mask_path = cfg.get("mask_path", "")
#         zone_labels = None
#         if Path(mask_path).exists():
#             try:
#                 zone_labels = self._rebuild_labels(mask_path)
#             except Exception:
#                 pass

#         if zone_labels is not None:
#             # Spawn per zone according to density_index
#             for zone in zones:
#                 d = zone.get("density_index", 1.0)
#                 if d <= 0:
#                     continue
#                 zid = zone["zone_id"]
#                 count = zone.get("agents", 0)
#                 if count == 0:
#                     area = zone.get("area_px", 0)
#                     count = int(area * d * base_density / agent_scale)
#                 count = max(1, count)

#                 # Get pixels belonging to this zone that are walkable
#                 zone_mask = (zone_labels == zid) & self.walk_map.walkable
#                 zy, zx = np.where(zone_mask)
#                 if len(zx) == 0:
#                     continue

#                 for _ in range(count):
#                     idx = random.randint(0, len(zx) - 1)
#                     self.agents.append(Agent(float(zx[idx]), float(zy[idx]), self.walk_map))
#         else:
#             # No label map — distribute proportionally across all walkable
#             total = sum(max(0, z.get("agents", 0)) for z in zones if z.get("density_index", 1) > 0)
#             if total == 0:
#                 total = 50
#             self._scatter_random(total)

#         print(f"Spawned {len(self.agents)} agents")

#     def _scatter_random(self, n: int):
#         ys, xs = np.where(self.walk_map.walkable)
#         for _ in range(n):
#             idx = random.randint(0, len(xs) - 1)
#             self.agents.append(Agent(float(xs[idx]), float(ys[idx]), self.walk_map))

#     def _rebuild_labels(self, mask_path: str):
#         """Rebuild zone label map using same watershed as zone_editor."""
#         import cv2
#         from scipy import ndimage as ndi
#         from skimage.segmentation import watershed
#         from skimage.feature import peak_local_max

#         img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
#         walkable = cv2.bitwise_not(img)
#         _, binary = cv2.threshold(walkable, 127, 255, cv2.THRESH_BINARY)
#         kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
#         binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

#         dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
#         dist_norm = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)
#         coords = peak_local_max(dist_norm, min_distance=40, labels=binary)
#         seed_mask = np.zeros(dist_norm.shape, dtype=bool)
#         seed_mask[tuple(coords.T)] = True
#         markers, _ = ndi.label(seed_mask)
#         labels = watershed(-dist, markers, mask=binary)
#         return labels

#     def step(self):
#         self.sim_time += DT
#         for agent in self.agents:
#             agent.update(DT, self.agents, self.sim_time)

#     def reset(self, zone_config: dict):
#         self.agents.clear()
#         self.sim_time = 0.0
#         self._spawn_agents(zone_config)


# # ══════════════════════════════════════════════════════════════════════
# #  Render widget
# # ══════════════════════════════════════════════════════════════════════

# class SimView(QWidget):
#     def __init__(self):
#         super().__init__()
#         self.sim: Optional[Simulation] = None
#         self.bg_pixmap: Optional[QPixmap] = None
#         self.selected_agent: Optional[Agent] = None
#         self.show_fov       = True
#         self.show_memory    = True
#         self.show_explored  = False
#         self.setMinimumSize(400, 400)
#         self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
#         self.setMouseTracking(True)

#     def set_sim(self, sim: Simulation, mask_path: str):
#         self.sim = sim
#         # Build background from mask
#         img = cv2.imread(mask_path)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         h, w, c = img.shape
#         qimg = QImage(img.tobytes(), w, h, w * c, QImage.Format.Format_RGB888)
#         self.bg_pixmap = QPixmap.fromImage(qimg)
#         self.update()

#     def mousePressEvent(self, event):
#         if self.sim is None:
#             return
#         mx, my = event.position().x(), event.position().y()
#         sx, sy = self._scale()
#         wx, wy = mx / sx, my / sy
#         # Find nearest agent
#         best, best_d = None, 20 / min(sx, sy)
#         for a in self.sim.agents:
#             d = math.hypot(a.x - wx, a.y - wy)
#             if d < best_d:
#                 best_d, best = d, a
#         if self.selected_agent:
#             self.selected_agent.selected = False
#         self.selected_agent = best
#         if best:
#             best.selected = True
#         self.update()

#     def _scale(self) -> Tuple[float, float]:
#         if self.bg_pixmap is None:
#             return 1.0, 1.0
#         return (self.width() / self.bg_pixmap.width(),
#                 self.height() / self.bg_pixmap.height())

#     def paintEvent(self, _):
#         if self.sim is None:
#             return
#         p = QPainter(self)
#         p.setRenderHint(QPainter.RenderHint.Antialiasing)
#         sx, sy = self._scale()

#         # Background
#         if self.bg_pixmap:
#             p.drawPixmap(self.rect(), self.bg_pixmap)

#         # Explored overlay
#         if self.show_explored and self.selected_agent:
#             p.setPen(Qt.PenStyle.NoPen)
#             p.setBrush(QBrush(QColor(100, 200, 255, 40)))
#             for (cx, cy) in self.selected_agent.known_map_cells:
#                 p.drawRect(int(cx * 10 * sx), int(cy * 10 * sy),
#                            int(10 * sx), int(10 * sy))

#         for a in self.sim.agents:
#             ax, ay = a.x * sx, a.y * sy
#             r = max(3, AGENT_RADIUS * sx)

#             # FOV cone (only for selected or all if enabled)
#             if self.show_fov and (a.selected or len(self.sim.agents) <= 80):
#                 fov_range_scaled = FOV_RANGE * sx
#                 fov_half = math.radians(FOV_DEGREES / 2)
#                 path = QPainterPath()
#                 path.moveTo(ax, ay)
#                 steps = 12
#                 for i in range(steps + 1):
#                     t = -fov_half + i * (2 * fov_half / steps)
#                     angle = a.angle + t
#                     path.lineTo(ax + math.cos(angle) * fov_range_scaled,
#                                 ay + math.sin(angle) * fov_range_scaled)
#                 path.closeSubpath()
#                 p.setPen(Qt.PenStyle.NoPen)
#                 alpha = 60 if a.selected else 18
#                 p.setBrush(QBrush(QColor(255, 255, 180, alpha)))
#                 p.drawPath(path)

#             # Memory dots (only for selected agent)
#             if self.show_memory and a.selected:
#                 for m in a.memory:
#                     mx2 = m.position[0] * sx
#                     my2 = m.position[1] * sy
#                     age = (self.sim.sim_time - m.time_seen) / MEMORY_DECAY
#                     alpha = int(180 * (1 - age))
#                     if m.kind == "exit":
#                         color = QColor(0, 255, 100, alpha)
#                     elif m.kind == "hazard":
#                         color = QColor(255, 80, 0, alpha)
#                     else:  # crowd
#                         color = QColor(180, 180, 255, alpha)
#                     p.setPen(QPen(color, 2))
#                     p.setBrush(QBrush(color))
#                     p.drawEllipse(QPointF(mx2, my2), 4, 4)

#             # Agent body
#             if a.selected:
#                 p.setPen(QPen(QColor(255, 255, 0), 2))
#                 p.setBrush(QBrush(QColor(255, 220, 0, 220)))
#             else:
#                 # Color by panic
#                 red  = int(200 * a.panic + 60 * (1 - a.panic))
#                 blue = int(180 * (1 - a.panic))
#                 p.setPen(QPen(QColor(0, 0, 0, 120), 1))
#                 p.setBrush(QBrush(QColor(red, 100, blue, 200)))
#             p.drawEllipse(QPointF(ax, ay), r, r)

#             # Direction arrow (facing indicator)
#             arrow_len = r * 1.8
#             ex = ax + math.cos(a.angle) * arrow_len
#             ey = ay + math.sin(a.angle) * arrow_len
#             p.setPen(QPen(QColor(255, 255, 255, 200), max(1, r * 0.4)))
#             p.drawLine(QPointF(ax, ay), QPointF(ex, ey))

#         p.end()


# # ══════════════════════════════════════════════════════════════════════
# #  Inspector panel (shows selected agent's memory)
# # ══════════════════════════════════════════════════════════════════════

# class InspectorPanel(QFrame):
#     def __init__(self):
#         super().__init__()
#         self.setObjectName("card")
#         self.setFixedWidth(260)
#         lv = QVBoxLayout(self)
#         lv.setContentsMargins(12, 12, 12, 12)
#         lv.setSpacing(6)

#         title = QLabel("Agent Inspector")
#         title.setStyleSheet("font-size:13pt; font-weight:bold; color:#e94560;")
#         lv.addWidget(title)

#         self.id_label    = QLabel("Click an agent to inspect")
#         self.pos_label   = QLabel("")
#         self.speed_label = QLabel("")
#         self.panic_label = QLabel("")
#         self.cells_label = QLabel("")
#         for lbl in [self.id_label, self.pos_label, self.speed_label,
#                     self.panic_label, self.cells_label]:
#             lbl.setStyleSheet("color:#ccc; font-size:9pt;")
#             lv.addWidget(lbl)

#         sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine)
#         sep.setStyleSheet("color:#333;")
#         lv.addWidget(sep)

#         lv.addWidget(QLabel("Memory:").setVisible(False) or QLabel("Memory:"))
#         self.memory_scroll = QScrollArea()
#         self.memory_scroll.setWidgetResizable(True)
#         self.memory_content = QWidget()
#         self.memory_layout  = QVBoxLayout(self.memory_content)
#         self.memory_layout.setSpacing(2)
#         self.memory_layout.setContentsMargins(0, 0, 0, 0)
#         self.memory_scroll.setWidget(self.memory_content)
#         self.memory_scroll.setStyleSheet("background:#111; border:none;")
#         lv.addWidget(self.memory_scroll, 1)
#         lv.addStretch()

#     def update_agent(self, agent: Optional['Agent'], sim_time: float):
#         if agent is None:
#             self.id_label.setText("Click an agent to inspect")
#             self.pos_label.setText("")
#             self.speed_label.setText("")
#             self.panic_label.setText("")
#             self.cells_label.setText("")
#             self._clear_memory()
#             return

#         self.id_label.setText(f"Agent #{agent.id}")
#         self.pos_label.setText(f"Position:  ({agent.x:.0f}, {agent.y:.0f})")
#         self.speed_label.setText(f"Speed:      {agent.speed:.1f} px/s")
#         self.panic_label.setText(f"Panic:      {agent.panic:.2f}")
#         self.cells_label.setText(f"Explored:   {len(agent.known_map_cells)} cells")

#         self._clear_memory()
#         if not agent.memory:
#             lbl = QLabel("  (nothing yet)")
#             lbl.setStyleSheet("color:#555; font-size:8pt;")
#             self.memory_layout.addWidget(lbl)
#         for m in sorted(agent.memory, key=lambda x: -x.time_seen):
#             age = sim_time - m.time_seen
#             text = (f"  [{m.kind:6s}]  ({m.position[0]:.0f},{m.position[1]:.0f})"
#                     f"  {age:.1f}s ago")
#             if m.kind == "exit":    color = "#00ff66"
#             elif m.kind == "hazard": color = "#ff6040"
#             else:                    color = "#aaaaff"
#             lbl = QLabel(text)
#             lbl.setStyleSheet(f"color:{color}; font-size:8pt; font-family:monospace;")
#             self.memory_layout.addWidget(lbl)
#         self.memory_layout.addStretch()

#     def _clear_memory(self):
#         while self.memory_layout.count():
#             item = self.memory_layout.takeAt(0)
#             if item.widget():
#                 item.widget().deleteLater()


# # ══════════════════════════════════════════════════════════════════════
# #  Main Window
# # ══════════════════════════════════════════════════════════════════════

# class MainWindow(QMainWindow):

#     STYLE = """
#     QMainWindow, QWidget { background:#1a1a2e; color:#e0e0e0;
#         font-family:'Segoe UI',Arial,sans-serif; font-size:10pt; }
#     QPushButton { background:#16213e; border:1px solid #0f3460;
#         border-radius:5px; padding:6px 12px; color:#e0e0e0; }
#     QPushButton:hover  { background:#0f3460; border-color:#e94560; }
#     QPushButton:pressed{ background:#e94560; color:white; }
#     QPushButton#primary{ background:#e94560; color:white; font-weight:bold; }
#     QFrame#card { background:#16213e; border:1px solid #0f3460; border-radius:8px; }
#     QLabel { color:#e0e0e0; }
#     QCheckBox { color:#ccc; }
#     QSlider::groove:horizontal { background:#0f3460; height:4px; border-radius:2px; }
#     QSlider::handle:horizontal { background:#e94560; width:14px; height:14px;
#         margin:-5px 0; border-radius:7px; }
#     """

#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("TRAGIC — Step 3: Agent Simulation")
#         self.setMinimumSize(1200, 750)
#         self.setStyleSheet(self.STYLE)

#         self.sim: Optional[Simulation] = None
#         self.zone_config: dict = {}
#         self.mask_path: str = ""
#         self.paused = True

#         self.timer = QTimer()
#         self.timer.setInterval(int(DT * 1000))
#         self.timer.timeout.connect(self._tick)

#         self._build_ui()

#     def _build_ui(self):
#         root = QWidget()
#         rl = QHBoxLayout(root)
#         rl.setContentsMargins(12, 12, 12, 12)
#         rl.setSpacing(12)
#         self.setCentralWidget(root)

#         # ── Left controls ──
#         left = QFrame(); left.setObjectName("card"); left.setFixedWidth(220)
#         lv = QVBoxLayout(left)
#         lv.setContentsMargins(10, 10, 10, 10)
#         lv.setSpacing(8)

#         QLabel("<b style='font-size:13pt;color:#e94560'>TRAGIC</b><br>"
#                "<span style='font-size:9pt;color:#888'>Step 3 — Agent Sim</span>").setParent(left)
#         title = QLabel("<b style='font-size:13pt;color:#e94560'>TRAGIC</b><br>"
#                        "<span style='font-size:9pt;color:#888'>Step 3 — Agent Sim</span>")
#         title.setTextFormat(Qt.TextFormat.RichText)
#         lv.addWidget(title)

#         self.load_btn = QPushButton("📂 Load Zone Config")
#         self.load_btn.clicked.connect(self.load_config)
#         lv.addWidget(self.load_btn)

#         self.run_btn = QPushButton("▶ Start")
#         self.run_btn.setObjectName("primary")
#         self.run_btn.setEnabled(False)
#         self.run_btn.clicked.connect(self.toggle_pause)
#         lv.addWidget(self.run_btn)

#         self.reset_btn = QPushButton("↺ Reset")
#         self.reset_btn.setEnabled(False)
#         self.reset_btn.clicked.connect(self.reset_sim)
#         lv.addWidget(self.reset_btn)

#         lv.addWidget(self._sep())
#         lv.addWidget(QLabel("Simulation Speed:"))
#         self.speed_slider = QSlider(Qt.Orientation.Horizontal)
#         self.speed_slider.setRange(1, 10)
#         self.speed_slider.setValue(3)
#         self.speed_slider.valueChanged.connect(self._update_speed)
#         lv.addWidget(self.speed_slider)

#         lv.addWidget(self._sep())

#         self.fov_check = QCheckBox("Show FOV cones")
#         self.fov_check.setChecked(True)
#         self.fov_check.toggled.connect(lambda v: setattr(self.sim_view, 'show_fov', v) or self.sim_view.update())
#         lv.addWidget(self.fov_check)

#         self.mem_check = QCheckBox("Show agent memory")
#         self.mem_check.setChecked(True)
#         self.mem_check.toggled.connect(lambda v: setattr(self.sim_view, 'show_memory', v) or self.sim_view.update())
#         lv.addWidget(self.mem_check)

#         self.exp_check = QCheckBox("Show explored area")
#         self.exp_check.setChecked(False)
#         self.exp_check.toggled.connect(lambda v: setattr(self.sim_view, 'show_explored', v) or self.sim_view.update())
#         lv.addWidget(self.exp_check)

#         lv.addWidget(self._sep())

#         self.stat_label = QLabel("Load a zone config\nto begin.")
#         self.stat_label.setStyleSheet("color:#888; font-size:9pt;")
#         self.stat_label.setWordWrap(True)
#         lv.addWidget(self.stat_label)
#         lv.addStretch()

#         # ── Sim view ──
#         self.sim_view = SimView()

#         # ── Inspector ──
#         self.inspector = InspectorPanel()

#         rl.addWidget(left)
#         rl.addWidget(self.sim_view, 1)
#         rl.addWidget(self.inspector)

#     def _sep(self):
#         sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine)
#         sep.setStyleSheet("color:#0f3460;")
#         return sep

#     def load_config(self):
#         path, _ = QFileDialog.getOpenFileName(
#             self, "Select Zone Config JSON", "", "JSON (*.json)")
#         if not path:
#             return

#         with open(path) as f:
#             self.zone_config = json.load(f)

#         mask_path = self.zone_config.get("mask_path", "")
#         if not Path(mask_path).exists():
#             # Let user pick mask manually
#             mask_path, _ = QFileDialog.getOpenFileName(
#                 self, "Locate Mask Image", "", "Images (*.png *.jpg *.bmp)")
#             if not mask_path:
#                 return
#             self.zone_config["mask_path"] = mask_path

#         self.mask_path = mask_path
#         try:
#             walk_map = WalkMap(mask_path)
#         except Exception as e:
#             QMessageBox.critical(self, "Error", str(e))
#             return

#         self.sim = Simulation(walk_map, self.zone_config)
#         self.sim_view.set_sim(self.sim, mask_path)
#         self.run_btn.setEnabled(True)
#         self.reset_btn.setEnabled(True)
#         self._update_stats()
#         self.paused = True
#         self.run_btn.setText("▶ Start")

#     def toggle_pause(self):
#         if self.sim is None:
#             return
#         self.paused = not self.paused
#         if self.paused:
#             self.timer.stop()
#             self.run_btn.setText("▶ Resume")
#         else:
#             self.timer.start()
#             self.run_btn.setText("⏸ Pause")

#     def reset_sim(self):
#         if self.sim is None:
#             return
#         self.timer.stop()
#         self.paused = True
#         self.run_btn.setText("▶ Start")
#         self.sim.reset(self.zone_config)
#         self.sim_view.selected_agent = None
#         self.sim_view.update()
#         self._update_stats()

#     def _tick(self):
#         if self.sim is None:
#             return
#         steps = self.speed_slider.value()
#         for _ in range(steps):
#             self.sim.step()
#         self.sim_view.update()
#         self._update_stats()
#         # Refresh inspector if agent selected
#         if self.sim_view.selected_agent:
#             self.inspector.update_agent(self.sim_view.selected_agent, self.sim.sim_time)

#     def _update_speed(self, v):
#         self.timer.setInterval(max(10, int(DT * 1000 // v)))

#     def _update_stats(self):
#         if self.sim is None:
#             return
#         n = len(self.sim.agents)
#         t = self.sim.sim_time
#         explored = sum(len(a.known_map_cells) for a in self.sim.agents)
#         self.stat_label.setText(
#             f"Time:     {t:.1f} s\n"
#             f"Agents:   {n}\n"
#             f"Explored: {explored} cells total\n\n"
#             f"Click agent to inspect.\nSpace = pause.")
#         # Update inspector if no agent selected
#         if not self.sim_view.selected_agent:
#             self.inspector.update_agent(None, t)

#     def keyPressEvent(self, event):
#         if event.key() == Qt.Key.Key_Space:
#             self.toggle_pause()
#         elif event.key() == Qt.Key.Key_R:
#             self.reset_sim()


# # ══════════════════════════════════════════════════════════════════════
# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     win = MainWindow()
#     win.show()
#     sys.exit(app.exec())













# """
# agent_sim.py  —  Step 3: Agent Simulation with FOV, Memory & Density Spawning
# Run: python agent_sim.py

# Requires:
#   - A binary mask image (stitched_mask.png or similar)
#   - A zone_config.json from zone_editor.py (Step 2)

# Controls:
#   Space     — Pause / Resume
#   Click     — Inspect agent at that position
#   R         — Reset simulation
# """

# import sys
# import json
# import math
# import random
# import numpy as np
# import cv2
# from pathlib import Path
# from dataclasses import dataclass, field
# from typing import List, Tuple, Optional, Dict

# from PyQt6.QtWidgets import (
#     QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
#     QPushButton, QLabel, QFileDialog, QFrame, QSizePolicy, QScrollArea,
#     QMessageBox, QCheckBox, QDoubleSpinBox, QSlider
# )
# from PyQt6.QtCore import Qt, QTimer, QPointF
# from PyQt6.QtGui import (
#     QPainter, QColor, QPen, QBrush, QPolygonF,
#     QImage, QPixmap, QFont, QPainterPath
# )

# # ── Simulation constants ───────────────────────────────────────────────
# DT              = 0.05      # seconds per tick
# AGENT_RADIUS    = 6         # pixels
# FOV_DEGREES     = 120       # total field of view
# FOV_RANGE       = 80        # pixels — how far an agent can see
# PANIC_FOV_RANGE = 140       # pixels — wider look during circle sweep
# SPEED_BASE      = 35        # px/s nominal walk speed
# SPEED_VARIANCE  = 0.25      # ±25% speed variation per agent
# NEIGHBOR_RADIUS = 40        # px — social force range
# WALL_BUFFER     = AGENT_RADIUS + 2
# CIRCLE_SWEEP_INTERVAL = 3.0 # seconds — how often agents look around
# MEMORY_DECAY    = 120.0     # seconds — exit memories fade after this
# # ──────────────────────────────────────────────────────────────────────


# # ══════════════════════════════════════════════════════════════════════
# #  Walkability map
# # ══════════════════════════════════════════════════════════════════════

# class WalkMap:
#     """Thin wrapper around a binary walkability image."""
#     def __init__(self, mask_path: str):
#         img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
#         if img is None:
#             raise FileNotFoundError(mask_path)
#         # White pixels = wall, black = walkable (typical stitched mask)
#         # Auto-detect convention by checking border pixels
#         # border_mean = (img[0, :].mean() + img[-1, :].mean() +
#         #                img[:, 0].mean() + img[:, -1].mean()) / 4
#         # if border_mean > 128:
#         #     # White border = wall, invert so walkable = True
#         #     self.walkable = img < 128
#         # else:
#         #     self.walkable = img > 128
        
#         # In your mask: white = wall, black = walkable
#         # Force this convention directly, no auto-detection
#         self.walkable = img < 128
        
#         # this is done here tosee if teh agents spawn correctly n the zones in which they are meant to spawn in 
        
#         self.h, self.w = img.shape
#         # Pre-compute distance transform for wall-avoidance force
#         walk_uint8 = self.walkable.astype(np.uint8) * 255
#         self.dist = cv2.distanceTransform(walk_uint8, cv2.DIST_L2, 5)

#     def is_walkable(self, x: float, y: float) -> bool:
#         ix, iy = int(x), int(y)
#         if ix < 0 or iy < 0 or ix >= self.w or iy >= self.h:
#             return False
#         return bool(self.walkable[iy, ix])

#     def wall_repulsion(self, x: float, y: float) -> Tuple[float, float]:
#         """Push agents away from walls based on distance transform."""
#         ix, iy = int(np.clip(x, 0, self.w - 1)), int(np.clip(y, 0, self.h - 1))
#         d = self.dist[iy, ix]
#         if d >= WALL_BUFFER * 2:
#             return 0.0, 0.0
#         # Gradient of distance transform
#         gx = (self.dist[iy, min(ix + 1, self.w - 1)] -
#               self.dist[iy, max(ix - 1, 0)]) / 2
#         gy = (self.dist[min(iy + 1, self.h - 1), ix] -
#               self.dist[max(iy - 1, 0), ix]) / 2
#         strength = max(0, (WALL_BUFFER * 2 - d)) / (WALL_BUFFER * 2)
#         return gx * strength * 150, gy * strength * 150


# # ══════════════════════════════════════════════════════════════════════
# #  Memory entry
# # ══════════════════════════════════════════════════════════════════════

# @dataclass
# class MemoryEntry:
#     """A thing the agent has seen and remembered."""
#     kind: str                   # "exit" | "hazard" | "crowd"
#     position: Tuple[float, float]
#     time_seen: float            # simulation time when first seen
#     confidence: float = 1.0    # fades with time


# # ══════════════════════════════════════════════════════════════════════
# #  Agent
# # ══════════════════════════════════════════════════════════════════════

# class Agent:
#     _id_counter = 0

#     def __init__(self, x: float, y: float, walk_map: WalkMap):
#         Agent._id_counter += 1
#         self.id = Agent._id_counter

#         self.x = x
#         self.y = y
#         self.speed = SPEED_BASE * random.uniform(1 - SPEED_VARIANCE, 1 + SPEED_VARIANCE)

#         # Direction in radians (0 = right, π/2 = down)
#         self.angle = random.uniform(0, math.tau)
#         self.vx = math.cos(self.angle) * self.speed
#         self.vy = math.sin(self.angle) * self.speed

#         self.walk_map = walk_map
#         self.memory: List[MemoryEntry] = []         # what this agent has seen
#         self.known_map_cells: set = set()           # grid cells explored (10px grid)

#         self.panic = 0.0                            # 0–1
#         self.last_sweep_time = random.uniform(0, CIRCLE_SWEEP_INTERVAL)
#         self.sim_time = 0.0

#         # For smooth steering
#         self._target_angle = self.angle
#         self._wander_timer = random.uniform(0, 2.0)

#         # For inspector display
#         self.selected = False
#         self.fov_visible_ids: set = set()           # IDs of agents in FOV right now

#     # ── movement ─────────────────────────────────────────────────────

#     def update(self, dt: float, all_agents: List['Agent'], sim_time: float):
#         self.sim_time = sim_time

#         # 1. Wander — smoothly change direction now and then
#         self._wander_timer -= dt
#         if self._wander_timer <= 0:
#             self._target_angle = self.angle + random.gauss(0, 0.8)
#             self._wander_timer = random.uniform(0.8, 2.5)

#         # 2. Smooth steer toward target angle
#         diff = (self._target_angle - self.angle + math.pi) % math.tau - math.pi
#         self.angle += diff * min(1.0, dt * 4)
#         self.angle %= math.tau

#         # 3. Wall repulsion force
#         rx, ry = self.walk_map.wall_repulsion(self.x, self.y)

#         # 4. Agent–agent separation force (social force)
#         sx, sy = 0.0, 0.0
#         self.fov_visible_ids.clear()
#         for other in all_agents:
#             if other.id == self.id:
#                 continue
#             dx, dy = self.x - other.x, self.y - other.y
#             dist = math.hypot(dx, dy)
#             if dist < NEIGHBOR_RADIUS and dist > 0:
#                 # Social separation
#                 push = max(0, (NEIGHBOR_RADIUS - dist)) / NEIGHBOR_RADIUS
#                 sx += (dx / dist) * push * 60
#                 sy += (dy / dist) * push * 60
#             # FOV check
#             if dist < FOV_RANGE and self._in_fov(other.x, other.y):
#                 self.fov_visible_ids.add(other.id)

#         # 5. Compose velocity
#         wx = math.cos(self.angle) * self.speed + rx + sx
#         wy = math.sin(self.angle) * self.speed + ry + sy

#         # 6. Propose new position, reject if into wall
#         nx, ny = self.x + wx * dt, self.y + wy * dt
#         if self.walk_map.is_walkable(nx, ny):
#             self.x, self.y = nx, ny
#         else:
#             # Bounce — try axis-aligned slides
#             if self.walk_map.is_walkable(nx, self.y):
#                 self.x = nx
#                 self._target_angle = math.atan2(-wy, wx)
#             elif self.walk_map.is_walkable(self.x, ny):
#                 self.y = ny
#                 self._target_angle = math.atan2(wy, -wx)
#             else:
#                 self._target_angle = self.angle + math.pi + random.uniform(-0.5, 0.5)

#         # 7. Update explored cells (10px grid)
#         cell = (int(self.x) // 10, int(self.y) // 10)
#         self.known_map_cells.add(cell)

#         # 8. Periodic circle sweep — scan all around
#         if sim_time - self.last_sweep_time >= CIRCLE_SWEEP_INTERVAL:
#             self._circle_sweep(all_agents, sim_time)
#             self.last_sweep_time = sim_time

#         # 9. Decay old memories
#         self.memory = [
#             m for m in self.memory
#             if sim_time - m.time_seen < MEMORY_DECAY
#         ]

#     def _in_fov(self, tx: float, ty: float) -> bool:
#         """Is target (tx, ty) within this agent's field of view?"""
#         dx, dy = tx - self.x, ty - self.y
#         angle_to = math.atan2(dy, dx)
#         diff = (angle_to - self.angle + math.pi) % math.tau - math.pi
#         return abs(diff) <= math.radians(FOV_DEGREES / 2)

#     def _circle_sweep(self, all_agents: List['Agent'], sim_time: float):
#         """
#         Look in all directions. Record any exits or hazards in memory.
#         In this step we just track other agents seen during the sweep
#         (exits/hazards would be added by the simulation manager later).
#         """
#         # Find any agents at panic-sweep range in all directions
#         for other in all_agents:
#             if other.id == self.id:
#                 continue
#             dist = math.hypot(self.x - other.x, self.y - other.y)
#             if dist < PANIC_FOV_RANGE:
#                 # Store position as "crowd" memory
#                 self._add_memory("crowd", (other.x, other.y), sim_time)

#     def _add_memory(self, kind: str, pos: Tuple[float, float], t: float):
#         """Add or refresh memory entry (deduplicates within 20px)."""
#         for m in self.memory:
#             if m.kind == kind and math.hypot(m.position[0] - pos[0],
#                                               m.position[1] - pos[1]) < 20:
#                 m.time_seen = t
#                 m.confidence = 1.0
#                 return
#         self.memory.append(MemoryEntry(kind, pos, t))

#     def add_exit_memory(self, pos: Tuple[float, float], t: float):
#         self._add_memory("exit", pos, t)

#     def add_hazard_memory(self, pos: Tuple[float, float], t: float):
#         self._add_memory("hazard", pos, t)


# # ══════════════════════════════════════════════════════════════════════
# #  Simulation Manager
# # ══════════════════════════════════════════════════════════════════════

# class Simulation:
#     def __init__(self, walk_map: WalkMap, zone_config: dict):
#         self.walk_map = walk_map
#         self.agents: List[Agent] = []
#         self.sim_time = 0.0
#         self._spawn_agents(zone_config)

#     def _spawn_agents(self, cfg: dict):
#         Agent._id_counter = 0

#         zones = cfg.get("zones", [])

#         if not zones:
#             self._scatter_random(100)
#             return

#         # All walkable pixels — pre-sampled once
#         ys, xs = np.where(self.walk_map.walkable)
#         if len(xs) == 0:
#             print("Warning: no walkable pixels found in mask!")
#             return

#         # Try to rebuild zone labels so agents spawn inside the right zone.
#         # If that fails we fall back to scattering across ALL walkable pixels
#         # (still correct count, just not zone-localised).
#         mask_path = cfg.get("mask_path", "")
#         zone_labels = None
#         if Path(mask_path).exists():
#             try:
#                 zone_labels = self._rebuild_labels(mask_path)
#                 print("Zone labels rebuilt successfully")
#             except Exception as e:
#                 print(f"Zone label rebuild failed ({e}), using global walkable pool")

#         # Build a flat list of walkable pixel indices for fast random sampling
#         walkable_pool = list(zip(xs.tolist(), ys.tolist()))

#         for zone in zones:
#             d = zone.get("density_index", 1.0)
#             if d <= 0:
#                 continue   # density_index 0 = outside/ignore

#             count = zone.get("agents", 0)
#             if count <= 0:
#                 continue

#             # Pick the pixel pool for this zone
#             if zone_labels is not None:
#                 zid = zone["zone_id"]
#                 zm = (zone_labels == zid) & self.walk_map.walkable
#                 zy, zx = np.where(zm)
#                 if len(zx) == 0:
#                     pool = walkable_pool   # fallback
#                 else:
#                     pool = list(zip(zx.tolist(), zy.tolist()))
#             else:
#                 pool = walkable_pool

#             for _ in range(count):
#                 px, py = random.choice(pool)
#                 self.agents.append(Agent(float(px), float(py), self.walk_map))

#         print(f"Spawned {len(self.agents)} agents")

#     def _scatter_random(self, n: int):
#         ys, xs = np.where(self.walk_map.walkable)
#         for _ in range(n):
#             idx = random.randint(0, len(xs) - 1)
#             self.agents.append(Agent(float(xs[idx]), float(ys[idx]), self.walk_map))

#     def _rebuild_labels(self, mask_path: str):
#         """Rebuild zone label map using same watershed as zone_editor."""
#         import cv2
#         from scipy import ndimage as ndi
#         from skimage.segmentation import watershed
#         from skimage.feature import peak_local_max

#         img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
#         walkable = cv2.bitwise_not(img)
#         _, binary = cv2.threshold(walkable, 127, 255, cv2.THRESH_BINARY)
#         kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
#         binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

#         dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
#         dist_norm = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)
#         coords = peak_local_max(dist_norm, min_distance=40, labels=binary)
#         seed_mask = np.zeros(dist_norm.shape, dtype=bool)
#         seed_mask[tuple(coords.T)] = True
#         markers, _ = ndi.label(seed_mask)
#         labels = watershed(-dist, markers, mask=binary)
#         return labels

#     def step(self):
#         self.sim_time += DT
#         for agent in self.agents:
#             agent.update(DT, self.agents, self.sim_time)

#     def reset(self, zone_config: dict):
#         self.agents.clear()
#         self.sim_time = 0.0
#         self._spawn_agents(zone_config)


# # ══════════════════════════════════════════════════════════════════════
# #  Render widget
# # ══════════════════════════════════════════════════════════════════════

# class SimView(QWidget):
#     def __init__(self):
#         super().__init__()
#         self.sim: Optional[Simulation] = None
#         self.bg_pixmap: Optional[QPixmap] = None
#         self.selected_agent: Optional[Agent] = None
#         self.show_fov       = True
#         self.show_memory    = True
#         self.show_explored  = False
#         self.setMinimumSize(400, 400)
#         self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
#         self.setMouseTracking(True)

#     def set_sim(self, sim: Simulation, mask_path: str):
#         self.sim = sim
#         # Build background from mask
#         img = cv2.imread(mask_path)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         h, w, c = img.shape
#         qimg = QImage(img.tobytes(), w, h, w * c, QImage.Format.Format_RGB888)
#         self.bg_pixmap = QPixmap.fromImage(qimg)
#         self.update()

#     def _layout(self):
#         """Uniform scale + letterbox offsets so image keeps its aspect ratio."""
#         if self.bg_pixmap is None:
#             return 1.0, 0.0, 0.0
#         iw, ih = self.bg_pixmap.width(), self.bg_pixmap.height()
#         s = min(self.width() / iw, self.height() / ih)
#         ox = (self.width()  - iw * s) / 2
#         oy = (self.height() - ih * s) / 2
#         return s, ox, oy

#     def _w2s(self, wx, wy):
#         """World → screen coords."""
#         s, ox, oy = self._layout()
#         return wx * s + ox, wy * s + oy

#     def _s2w(self, sx, sy):
#         """Screen → world coords."""
#         s, ox, oy = self._layout()
#         return (sx - ox) / s, (sy - oy) / s

#     def mousePressEvent(self, event):
#         if self.sim is None:
#             return
#         wx, wy = self._s2w(event.position().x(), event.position().y())
#         s, _, _ = self._layout()
#         pick_r = 20 / s
#         best, best_d = None, pick_r
#         for a in self.sim.agents:
#             d = math.hypot(a.x - wx, a.y - wy)
#             if d < best_d:
#                 best_d, best = d, a
#         if self.selected_agent:
#             self.selected_agent.selected = False
#         self.selected_agent = best
#         if best:
#             best.selected = True
#         self.update()

#     def paintEvent(self, _):
#         if self.sim is None:
#             return
#         p = QPainter(self)
#         p.setRenderHint(QPainter.RenderHint.Antialiasing)
#         s, ox, oy = self._layout()

#         # Background — letterboxed
#         if self.bg_pixmap:
#             from PyQt6.QtCore import QRectF
#             iw, ih = self.bg_pixmap.width(), self.bg_pixmap.height()
#             p.drawPixmap(QRectF(ox, oy, iw * s, ih * s).toRect(), self.bg_pixmap)

#         # Explored overlay
#         if self.show_explored and self.selected_agent:
#             p.setPen(Qt.PenStyle.NoPen)
#             p.setBrush(QBrush(QColor(100, 200, 255, 40)))
#             for (cx, cy) in self.selected_agent.known_map_cells:
#                 scx, scy = self._w2s(cx * 10, cy * 10)
#                 p.drawRect(int(scx), int(scy), int(10 * s), int(10 * s))

#         for a in self.sim.agents:
#             ax, ay = self._w2s(a.x, a.y)
#             r = max(3, AGENT_RADIUS * s)

#             # FOV cone
#             if self.show_fov and (a.selected or len(self.sim.agents) <= 80):
#                 fov_range_scaled = FOV_RANGE * s
#                 fov_half = math.radians(FOV_DEGREES / 2)
#                 path = QPainterPath()
#                 path.moveTo(ax, ay)
#                 steps = 12
#                 for i in range(steps + 1):
#                     t = -fov_half + i * (2 * fov_half / steps)
#                     angle = a.angle + t
#                     path.lineTo(ax + math.cos(angle) * fov_range_scaled,
#                                 ay + math.sin(angle) * fov_range_scaled)
#                 path.closeSubpath()
#                 p.setPen(Qt.PenStyle.NoPen)
#                 alpha = 60 if a.selected else 18
#                 p.setBrush(QBrush(QColor(255, 255, 180, alpha)))
#                 p.drawPath(path)

#             # Memory dots (selected agent only)
#             if self.show_memory and a.selected:
#                 for m in a.memory:
#                     mx2, my2 = self._w2s(m.position[0], m.position[1])
#                     age = (self.sim.sim_time - m.time_seen) / MEMORY_DECAY
#                     alpha = int(180 * (1 - age))
#                     if m.kind == "exit":
#                         color = QColor(0, 255, 100, alpha)
#                     elif m.kind == "hazard":
#                         color = QColor(255, 80, 0, alpha)
#                     else:
#                         color = QColor(180, 180, 255, alpha)
#                     p.setPen(QPen(color, 2))
#                     p.setBrush(QBrush(color))
#                     p.drawEllipse(QPointF(mx2, my2), 4, 4)

#             # Agent body
#             if a.selected:
#                 p.setPen(QPen(QColor(255, 255, 0), 2))
#                 p.setBrush(QBrush(QColor(255, 220, 0, 220)))
#             else:
#                 red  = int(200 * a.panic + 60 * (1 - a.panic))
#                 blue = int(180 * (1 - a.panic))
#                 p.setPen(QPen(QColor(0, 0, 0, 120), 1))
#                 p.setBrush(QBrush(QColor(red, 100, blue, 200)))
#             p.drawEllipse(QPointF(ax, ay), r, r)

#             # Direction arrow
#             arrow_len = r * 1.8
#             ex = ax + math.cos(a.angle) * arrow_len
#             ey = ay + math.sin(a.angle) * arrow_len
#             p.setPen(QPen(QColor(255, 255, 255, 200), max(1, r * 0.4)))
#             p.drawLine(QPointF(ax, ay), QPointF(ex, ey))

#         p.end()


# # ══════════════════════════════════════════════════════════════════════
# #  Inspector panel (shows selected agent's memory)
# # ══════════════════════════════════════════════════════════════════════

# class InspectorPanel(QFrame):
#     def __init__(self):
#         super().__init__()
#         self.setObjectName("card")
#         self.setFixedWidth(260)
#         lv = QVBoxLayout(self)
#         lv.setContentsMargins(12, 12, 12, 12)
#         lv.setSpacing(6)

#         title = QLabel("Agent Inspector")
#         title.setStyleSheet("font-size:13pt; font-weight:bold; color:#e94560;")
#         lv.addWidget(title)

#         self.id_label    = QLabel("Click an agent to inspect")
#         self.pos_label   = QLabel("")
#         self.speed_label = QLabel("")
#         self.panic_label = QLabel("")
#         self.cells_label = QLabel("")
#         for lbl in [self.id_label, self.pos_label, self.speed_label,
#                     self.panic_label, self.cells_label]:
#             lbl.setStyleSheet("color:#ccc; font-size:9pt;")
#             lv.addWidget(lbl)

#         sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine)
#         sep.setStyleSheet("color:#333;")
#         lv.addWidget(sep)

#         lv.addWidget(QLabel("Memory:").setVisible(False) or QLabel("Memory:"))
#         self.memory_scroll = QScrollArea()
#         self.memory_scroll.setWidgetResizable(True)
#         self.memory_content = QWidget()
#         self.memory_layout  = QVBoxLayout(self.memory_content)
#         self.memory_layout.setSpacing(2)
#         self.memory_layout.setContentsMargins(0, 0, 0, 0)
#         self.memory_scroll.setWidget(self.memory_content)
#         self.memory_scroll.setStyleSheet("background:#111; border:none;")
#         lv.addWidget(self.memory_scroll, 1)
#         lv.addStretch()

#     def update_agent(self, agent: Optional['Agent'], sim_time: float):
#         if agent is None:
#             self.id_label.setText("Click an agent to inspect")
#             self.pos_label.setText("")
#             self.speed_label.setText("")
#             self.panic_label.setText("")
#             self.cells_label.setText("")
#             self._clear_memory()
#             return

#         self.id_label.setText(f"Agent #{agent.id}")
#         self.pos_label.setText(f"Position:  ({agent.x:.0f}, {agent.y:.0f})")
#         self.speed_label.setText(f"Speed:      {agent.speed:.1f} px/s")
#         self.panic_label.setText(f"Panic:      {agent.panic:.2f}")
#         self.cells_label.setText(f"Explored:   {len(agent.known_map_cells)} cells")

#         self._clear_memory()
#         if not agent.memory:
#             lbl = QLabel("  (nothing yet)")
#             lbl.setStyleSheet("color:#555; font-size:8pt;")
#             self.memory_layout.addWidget(lbl)
#         for m in sorted(agent.memory, key=lambda x: -x.time_seen):
#             age = sim_time - m.time_seen
#             text = (f"  [{m.kind:6s}]  ({m.position[0]:.0f},{m.position[1]:.0f})"
#                     f"  {age:.1f}s ago")
#             if m.kind == "exit":    color = "#00ff66"
#             elif m.kind == "hazard": color = "#ff6040"
#             else:                    color = "#aaaaff"
#             lbl = QLabel(text)
#             lbl.setStyleSheet(f"color:{color}; font-size:8pt; font-family:monospace;")
#             self.memory_layout.addWidget(lbl)
#         self.memory_layout.addStretch()

#     def _clear_memory(self):
#         while self.memory_layout.count():
#             item = self.memory_layout.takeAt(0)
#             if item.widget():
#                 item.widget().deleteLater()


# # ══════════════════════════════════════════════════════════════════════
# #  Main Window
# # ══════════════════════════════════════════════════════════════════════

# class MainWindow(QMainWindow):

#     STYLE = """
#     QMainWindow, QWidget { background:#1a1a2e; color:#e0e0e0;
#         font-family:'Segoe UI',Arial,sans-serif; font-size:10pt; }
#     QPushButton { background:#16213e; border:1px solid #0f3460;
#         border-radius:5px; padding:6px 12px; color:#e0e0e0; }
#     QPushButton:hover  { background:#0f3460; border-color:#e94560; }
#     QPushButton:pressed{ background:#e94560; color:white; }
#     QPushButton#primary{ background:#e94560; color:white; font-weight:bold; }
#     QFrame#card { background:#16213e; border:1px solid #0f3460; border-radius:8px; }
#     QLabel { color:#e0e0e0; }
#     QCheckBox { color:#ccc; }
#     QSlider::groove:horizontal { background:#0f3460; height:4px; border-radius:2px; }
#     QSlider::handle:horizontal { background:#e94560; width:14px; height:14px;
#         margin:-5px 0; border-radius:7px; }
#     """

#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("TRAGIC — Step 3: Agent Simulation")
#         self.setMinimumSize(1200, 750)
#         self.setStyleSheet(self.STYLE)

#         self.sim: Optional[Simulation] = None
#         self.zone_config: dict = {}
#         self.mask_path: str = ""
#         self.paused = True

#         self.timer = QTimer()
#         self.timer.setInterval(int(DT * 1000))
#         self.timer.timeout.connect(self._tick)

#         self._build_ui()

#     def _build_ui(self):
#         root = QWidget()
#         rl = QHBoxLayout(root)
#         rl.setContentsMargins(12, 12, 12, 12)
#         rl.setSpacing(12)
#         self.setCentralWidget(root)

#         # ── Left controls ──
#         left = QFrame(); left.setObjectName("card"); left.setFixedWidth(220)
#         lv = QVBoxLayout(left)
#         lv.setContentsMargins(10, 10, 10, 10)
#         lv.setSpacing(8)

#         QLabel("<b style='font-size:13pt;color:#e94560'>TRAGIC</b><br>"
#                "<span style='font-size:9pt;color:#888'>Step 3 — Agent Sim</span>").setParent(left)
#         title = QLabel("<b style='font-size:13pt;color:#e94560'>TRAGIC</b><br>"
#                        "<span style='font-size:9pt;color:#888'>Step 3 — Agent Sim</span>")
#         title.setTextFormat(Qt.TextFormat.RichText)
#         lv.addWidget(title)

#         self.load_btn = QPushButton("📂 Load Zone Config")
#         self.load_btn.clicked.connect(self.load_config)
#         lv.addWidget(self.load_btn)

#         self.run_btn = QPushButton("▶ Start")
#         self.run_btn.setObjectName("primary")
#         self.run_btn.setEnabled(False)
#         self.run_btn.clicked.connect(self.toggle_pause)
#         lv.addWidget(self.run_btn)

#         self.reset_btn = QPushButton("↺ Reset")
#         self.reset_btn.setEnabled(False)
#         self.reset_btn.clicked.connect(self.reset_sim)
#         lv.addWidget(self.reset_btn)

#         lv.addWidget(self._sep())
#         lv.addWidget(QLabel("Simulation Speed:"))
#         self.speed_slider = QSlider(Qt.Orientation.Horizontal)
#         self.speed_slider.setRange(1, 10)
#         self.speed_slider.setValue(3)
#         self.speed_slider.valueChanged.connect(self._update_speed)
#         lv.addWidget(self.speed_slider)

#         lv.addWidget(self._sep())

#         self.fov_check = QCheckBox("Show FOV cones")
#         self.fov_check.setChecked(True)
#         self.fov_check.toggled.connect(lambda v: setattr(self.sim_view, 'show_fov', v) or self.sim_view.update())
#         lv.addWidget(self.fov_check)

#         self.mem_check = QCheckBox("Show agent memory")
#         self.mem_check.setChecked(True)
#         self.mem_check.toggled.connect(lambda v: setattr(self.sim_view, 'show_memory', v) or self.sim_view.update())
#         lv.addWidget(self.mem_check)

#         self.exp_check = QCheckBox("Show explored area")
#         self.exp_check.setChecked(False)
#         self.exp_check.toggled.connect(lambda v: setattr(self.sim_view, 'show_explored', v) or self.sim_view.update())
#         lv.addWidget(self.exp_check)

#         lv.addWidget(self._sep())

#         self.stat_label = QLabel("Load a zone config\nto begin.")
#         self.stat_label.setStyleSheet("color:#888; font-size:9pt;")
#         self.stat_label.setWordWrap(True)
#         lv.addWidget(self.stat_label)
#         lv.addStretch()

#         # ── Sim view ──
#         self.sim_view = SimView()

#         # ── Inspector ──
#         self.inspector = InspectorPanel()

#         rl.addWidget(left)
#         rl.addWidget(self.sim_view, 1)
#         rl.addWidget(self.inspector)

#     def _sep(self):
#         sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine)
#         sep.setStyleSheet("color:#0f3460;")
#         return sep

#     def load_config(self):
#         path, _ = QFileDialog.getOpenFileName(
#             self, "Select Zone Config JSON", "", "JSON (*.json)")
#         if not path:
#             return

#         with open(path) as f:
#             self.zone_config = json.load(f)

#         mask_path = self.zone_config.get("mask_path", "")
#         if not Path(mask_path).exists():
#             # Let user pick mask manually
#             mask_path, _ = QFileDialog.getOpenFileName(
#                 self, "Locate Mask Image", "", "Images (*.png *.jpg *.bmp)")
#             if not mask_path:
#                 return
#             self.zone_config["mask_path"] = mask_path

#         self.mask_path = mask_path
#         try:
#             walk_map = WalkMap(mask_path)
#         except Exception as e:
#             QMessageBox.critical(self, "Error", str(e))
#             return

#         self.sim = Simulation(walk_map, self.zone_config)
#         self.sim_view.set_sim(self.sim, mask_path)
#         self.run_btn.setEnabled(True)
#         self.reset_btn.setEnabled(True)
#         self._update_stats()
#         self.paused = True
#         self.run_btn.setText("▶ Start")

#     def toggle_pause(self):
#         if self.sim is None:
#             return
#         self.paused = not self.paused
#         if self.paused:
#             self.timer.stop()
#             self.run_btn.setText("▶ Resume")
#         else:
#             self.timer.start()
#             self.run_btn.setText("⏸ Pause")

#     def reset_sim(self):
#         if self.sim is None:
#             return
#         self.timer.stop()
#         self.paused = True
#         self.run_btn.setText("▶ Start")
#         self.sim.reset(self.zone_config)
#         self.sim_view.selected_agent = None
#         self.sim_view.update()
#         self._update_stats()

#     def _tick(self):
#         if self.sim is None:
#             return
#         steps = self.speed_slider.value()
#         for _ in range(steps):
#             self.sim.step()
#         self.sim_view.update()
#         self._update_stats()
#         # Refresh inspector if agent selected
#         if self.sim_view.selected_agent:
#             self.inspector.update_agent(self.sim_view.selected_agent, self.sim.sim_time)

#     def _update_speed(self, v):
#         self.timer.setInterval(max(10, int(DT * 1000 // v)))

#     def _update_stats(self):
#         if self.sim is None:
#             return
#         n = len(self.sim.agents)
#         t = self.sim.sim_time
#         explored = sum(len(a.known_map_cells) for a in self.sim.agents)
#         self.stat_label.setText(
#             f"Time:     {t:.1f} s\n"
#             f"Agents:   {n}\n"
#             f"Explored: {explored} cells total\n\n"
#             f"Click agent to inspect.\nSpace = pause.")
#         # Update inspector if no agent selected
#         if not self.sim_view.selected_agent:
#             self.inspector.update_agent(None, t)

#     def keyPressEvent(self, event):
#         if event.key() == Qt.Key.Key_Space:
#             self.toggle_pause()
#         elif event.key() == Qt.Key.Key_R:
#             self.reset_sim()


# # ══════════════════════════════════════════════════════════════════════
# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     win = MainWindow()
#     win.show()
#     sys.exit(app.exec())






"""
agent_sim.py  —  Crowd Evacuation Simulation
Behavioral model based on:
  - GAS (General Adaptation Syndrome) stress-driven state transitions
  - Collision-Free Speed Model for smooth movement (inspired by JuPedSim)
  - 120° FOV with circle sweep on stress trigger (per planning doc)
  - Explored-area memory rendered as imshow overlay (single draw call)
  - Exit detection with evacuation

Run: python agent_sim.py
Controls: Space = pause/resume, R = reset, click = inspect agent
"""

import sys
import json
import math
import random
import numpy as np
import cv2
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QLabel, QFileDialog, QFrame, QSizePolicy,
    QScrollArea, QMessageBox, QCheckBox, QSlider, QSpinBox
)
from PyQt6.QtCore import Qt, QTimer, QPointF
from PyQt6.QtGui import (
    QPainter, QColor, QPen, QBrush, QImage, QPixmap, QPainterPath
)

# ── constants ──────────────────────────────────────────────────────────
DT             = 0.05    # seconds per tick
AGENT_RADIUS   = 5       # pixels display radius
FOV_DEG        = 120     # field of view in degrees
SPEED_BASE     = 40      # px/s base walk speed
SPEED_VARIANCE = 0.2     # ±20%
WALL_BUFFER    = 8       # px — keep agents this far from walls

# Stress thresholds (GAS model)
STRESS_ALERT    = 0.25   # idle → alert
STRESS_EVACUATE = 0.55   # alert → evacuate
STRESS_DECAY    = 0.02   # per second when no hazard
STRESS_RISE     = 0.35   # per second near hazard

# Circle sweep — how far agent looks during sweep (px)
SWEEP_RADIUS = 150

# Exit detection radius (px)
EXIT_RADIUS = 15

# Wander: new target every N seconds
WANDER_INTERVAL = (4.0, 9.0)

# Explored memory: how fast unexplored fades (not used for now, kept static)
EXPLORED_ALPHA = 0.18   # opacity of explored overlay
# ───────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════
#  WalkMap
# ══════════════════════════════════════════════════════════════════════

class WalkMap:
    """Binary walkability map built from stitched mask."""

    def __init__(self, mask_path: str):
        img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(mask_path)
        # White = wall, black = walkable
        self.walkable = img < 128
        self.h, self.w = img.shape

        walk_u8 = self.walkable.astype(np.uint8) * 255
        # Distance transform: every walkable px gets distance to nearest wall
        self.dist = cv2.distanceTransform(walk_u8, cv2.DIST_L2, 5)

    def is_walkable(self, x: float, y: float) -> bool:
        ix, iy = int(x), int(y)
        if not (0 <= ix < self.w and 0 <= iy < self.h):
            return False
        return bool(self.walkable[iy, ix])

    def wall_force(self, x: float, y: float) -> Tuple[float, float]:
        """Repulsion gradient away from walls using distance transform."""
        ix = int(np.clip(x, 0, self.w - 1))
        iy = int(np.clip(y, 0, self.h - 1))
        d = self.dist[iy, ix]
        if d >= WALL_BUFFER * 2:
            return 0.0, 0.0
        gx = (self.dist[iy, min(ix+1, self.w-1)] - self.dist[iy, max(ix-1, 0)]) / 2
        gy = (self.dist[min(iy+1, self.h-1), ix] - self.dist[max(iy-1, 0), ix]) / 2
        strength = max(0.0, WALL_BUFFER * 2 - d) / (WALL_BUFFER * 2)
        return gx * strength * 180, gy * strength * 180

    def random_walkable(self) -> Optional[Tuple[float, float]]:
        """Return a random walkable pixel coordinate."""
        ys, xs = np.where(self.walkable)
        if len(xs) == 0:
            return None
        i = random.randint(0, len(xs) - 1)
        return float(xs[i]), float(ys[i])


# ══════════════════════════════════════════════════════════════════════
#  Exit
# ══════════════════════════════════════════════════════════════════════

@dataclass
class Exit:
    x: float
    y: float
    radius: float = EXIT_RADIUS
    blocked: bool = False


# ══════════════════════════════════════════════════════════════════════
#  Agent
# ══════════════════════════════════════════════════════════════════════

class Agent:
    """
    Individual pedestrian with:
      - GAS stress model (idle / alert / evacuate)
      - 120° FOV + circle sweep when stress threshold hit
      - Memory of exits seen (coordinates stored on first sight)
      - Collision-free speed model movement
    """

    IDLE     = 'idle'
    ALERT    = 'alert'
    EVACUATE = 'evacuate'

    _id_counter = 0

    def __init__(self, x: float, y: float, walk_map: WalkMap):
        Agent._id_counter += 1
        self.id = Agent._id_counter

        self.x = float(x)
        self.y = float(y)
        self.angle = random.uniform(0, math.tau)  # facing direction (radians)
        self.speed = SPEED_BASE * random.uniform(1 - SPEED_VARIANCE, 1 + SPEED_VARIANCE)

        self.walk_map = walk_map

        # GAS state
        self.stress = 0.0
        self.state = Agent.IDLE

        # Navigation
        self.goal: Optional[Tuple[float, float]] = None  # current target (wander or exit)
        self.known_exits: List[Tuple[float, float]] = []  # exits seen and memorised

        # Wander
        self.wander_cooldown = random.uniform(*WANDER_INTERVAL)

        # Evacuation
        self.evacuated = False
        self.swept = False  # has this agent done its circle sweep yet

        # Inspector
        self.selected = False

    # ── stress / state ────────────────────────────────────────────────

    def update_stress(self, hazard_dist: float, neighbor_stress: List[float], dt: float):
        """
        hazard_dist: distance in px to nearest hazard (None = no hazard)
        GAS: stress rises fast near hazard, spreads slowly from neighbors, decays otherwise.
        """
        if hazard_dist is not None:
            # Closer = faster stress rise; normalise over 300px
            proximity = max(0.0, 1.0 - hazard_dist / 300.0)
            self.stress = min(1.0, self.stress + STRESS_RISE * proximity * dt)
        else:
            self.stress = max(0.0, self.stress - STRESS_DECAY * dt)

        # Social contagion — a fraction of neighbor stress bleeds in
        if neighbor_stress:
            avg = np.mean(neighbor_stress)
            self.stress = min(1.0, self.stress + max(0, avg - self.stress) * 0.06 * dt)

        # State transitions (hysteresis — can't go backward easily)
        if self.state == Agent.IDLE and self.stress >= STRESS_ALERT:
            self.state = Agent.ALERT
        elif self.state == Agent.ALERT and self.stress >= STRESS_EVACUATE:
            self.state = Agent.EVACUATE
        elif self.state == Agent.EVACUATE and self.stress < STRESS_ALERT * 0.5:
            self.state = Agent.ALERT  # only calm down partially

    # ── FOV ───────────────────────────────────────────────────────────

    def in_fov(self, tx: float, ty: float) -> bool:
        """Is (tx, ty) within the 120° FOV cone?"""
        dx, dy = tx - self.x, ty - self.y
        dist = math.hypot(dx, dy)
        if dist > SWEEP_RADIUS:
            return False
        angle_to = math.atan2(dy, dx)
        diff = (angle_to - self.angle + math.pi) % math.tau - math.pi
        return abs(diff) <= math.radians(FOV_DEG / 2)

    def scan_fov(self, exits: List[Exit]):
        """Check if any exit is in current FOV — if so, memorise it."""
        for ex in exits:
            if ex.blocked:
                continue
            if self.in_fov(ex.x, ex.y):
                coord = (ex.x, ex.y)
                if coord not in self.known_exits:
                    self.known_exits.append(coord)

    def circle_sweep(self, exits: List[Exit]):
        """
        Full 360° look — done once when stress threshold is crossed.
        Stores all visible exits (within SWEEP_RADIUS) into memory.
        """
        for ex in exits:
            if ex.blocked:
                continue
            dist = math.hypot(ex.x - self.x, ex.y - self.y)
            if dist <= SWEEP_RADIUS:
                coord = (ex.x, ex.y)
                if coord not in self.known_exits:
                    self.known_exits.append(coord)
        self.swept = True

    def pick_best_exit(self) -> Optional[Tuple[float, float]]:
        """
        Choose exit from memory:
          - nearest if calm enough
          - random-ish if stress > 0.8 (panic degrades decision quality)
        """
        if not self.known_exits:
            return None
        if self.stress > 0.8:
            return random.choice(self.known_exits)
        return min(self.known_exits, key=lambda e: math.hypot(e[0]-self.x, e[1]-self.y))

    # ── movement ──────────────────────────────────────────────────────

    def get_speed(self) -> float:
        if self.state == Agent.IDLE:
            return self.speed * 0.55       # strolling
        elif self.state == Agent.ALERT:
            return self.speed * 1.0        # brisk walk
        else:
            return self.speed * (1.0 + self.stress * 0.7)  # rushing

    def update(self, dt: float, all_agents: List['Agent'],
               exits: List[Exit], hazard_pos: Optional[Tuple[float, float]]):

        if self.evacuated:
            return

        # --- Stress ---
        hazard_dist = None
        if hazard_pos is not None:
            hazard_dist = math.hypot(self.x - hazard_pos[0], self.y - hazard_pos[1])

        neighbor_stress = [
            a.stress for a in all_agents
            if a.id != self.id and math.hypot(a.x - self.x, a.y - self.y) < 80
        ]
        self.update_stress(hazard_dist, neighbor_stress, dt)

        # --- Trigger circle sweep once on alert ---
        if self.state in (Agent.ALERT, Agent.EVACUATE) and not self.swept:
            self.circle_sweep(exits)

        # --- Continuous FOV scan (cheap — just checks exits) ---
        self.scan_fov(exits)

        # --- Goal assignment ---
        if self.state == Agent.IDLE:
            # Wander: pick a new random target when cooldown expires
            self.wander_cooldown -= dt
            if self.goal is None or self.wander_cooldown <= 0:
                pos = self.walk_map.random_walkable()
                if pos:
                    self.goal = pos
                self.wander_cooldown = random.uniform(*WANDER_INTERVAL)

        else:
            # Alert or Evacuate: head to best known exit
            best = self.pick_best_exit()
            if best is not None:
                self.goal = best
            elif self.goal is None:
                # No exit in memory yet — keep wandering until one appears
                pos = self.walk_map.random_walkable()
                if pos:
                    self.goal = pos

        # --- Compute desired direction ---
        if self.goal is None:
            return

        gx, gy = self.goal
        dx, dy = gx - self.x, gy - self.y
        dist_to_goal = math.hypot(dx, dy)

        # Reached goal
        if dist_to_goal < 6.0:
            if self.state == Agent.IDLE:
                self.goal = None   # wander will pick a new one
            return

        nx_dir, ny_dir = dx / dist_to_goal, dy / dist_to_goal

        # --- Collision-free speed model: avoid other agents ---
        avoid_x, avoid_y = 0.0, 0.0
        for other in all_agents:
            if other.id == self.id or other.evacuated:
                continue
            odx, ody = self.x - other.x, self.y - other.y
            od = math.hypot(odx, ody)
            if 0 < od < AGENT_RADIUS * 6:
                push = max(0.0, (AGENT_RADIUS * 6 - od)) / (AGENT_RADIUS * 6)
                avoid_x += (odx / od) * push * 50
                avoid_y += (ody / od) * push * 50

        # --- Wall repulsion ---
        wx, wy = self.walk_map.wall_force(self.x, self.y)

        # --- Final velocity ---
        spd = self.get_speed()
        vx = nx_dir * spd + avoid_x + wx
        vy = ny_dir * spd + avoid_y + wy

        # Smooth facing direction
        move_angle = math.atan2(vy, vx)
        diff = (move_angle - self.angle + math.pi) % math.tau - math.pi
        self.angle += diff * min(1.0, dt * 5)
        self.angle %= math.tau

        # --- Step with wall check ---
        nx_, ny_ = self.x + vx * dt, self.y + vy * dt
        if self.walk_map.is_walkable(nx_, ny_):
            self.x, self.y = nx_, ny_
        elif self.walk_map.is_walkable(nx_, self.y):
            self.x = nx_
        elif self.walk_map.is_walkable(self.x, ny_):
            self.y = ny_
        else:
            # Stuck — pick a new goal
            self.goal = None

        # --- Check exit ---
        for ex in exits:
            if ex.blocked:
                continue
            if math.hypot(self.x - ex.x, self.y - ex.y) < ex.radius:
                self.evacuated = True
                return


# ══════════════════════════════════════════════════════════════════════
#  Simulation
# ══════════════════════════════════════════════════════════════════════

class Simulation:

    def __init__(self, walk_map: WalkMap, zone_config: dict):
        self.walk_map = walk_map
        self.agents: List[Agent] = []
        self.exits: List[Exit] = []
        self.hazard: Optional[Tuple[float, float]] = None  # (x, y) or None
        self.sim_time = 0.0

        # Explored area — numpy array same size as mask, accumulates FOV visits
        self.explored = np.zeros((walk_map.h, walk_map.w), dtype=np.float32)

        self._zone_labels = None
        self._spawn(zone_config)

    def _build_labels(self, mask_path: str):
        img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        walkable = cv2.bitwise_not(img)
        _, binary = cv2.threshold(walkable, 127, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        dist_norm = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)
        coords = peak_local_max(dist_norm, min_distance=40, labels=binary)
        seed = np.zeros(dist_norm.shape, dtype=bool)
        seed[tuple(coords.T)] = True
        markers, _ = ndi.label(seed)
        return watershed(-dist, markers, mask=binary)

    def _spawn(self, cfg: dict):
        Agent._id_counter = 0
        zones = cfg.get('zones', [])
        mask_path = cfg.get('mask_path', '')

        if Path(mask_path).exists():
            try:
                self._zone_labels = self._build_labels(mask_path)
            except Exception as e:
                print(f"Zone label build failed: {e}")

        ys, xs = np.where(self.walk_map.walkable)
        global_pool = list(zip(xs.tolist(), ys.tolist()))

        for zone in zones:
            if zone.get('density_index', 0) <= 0:
                continue
            count = zone.get('agents', 0)
            if count <= 0:
                continue

            pool = global_pool
            if self._zone_labels is not None:
                zid = zone['zone_id']
                zm = (self._zone_labels == zid) & self.walk_map.walkable
                zy, zx = np.where(zm)
                if len(zx) > 0:
                    pool = list(zip(zx.tolist(), zy.tolist()))

            for _ in range(count):
                px, py = random.choice(pool)
                self.agents.append(Agent(float(px), float(py), self.walk_map))

        print(f"Spawned {len(self.agents)} agents, {len(self.exits)} exits")

    def add_exit(self, x: float, y: float):
        self.exits.append(Exit(x, y))
        print(f"Exit added at ({x:.0f}, {y:.0f})")

    def set_hazard(self, x: float, y: float):
        self.hazard = (x, y)
        print(f"Hazard at ({x:.0f}, {y:.0f})")

    def _update_explored(self):
        """
        Stamp the explored map for each active agent's FOV.
        Uses a circle mask centered at agent position — one numpy slice per agent.
        Fast because it's just array indexing, no Python loops over pixels.
        """
        fov_r = int(SWEEP_RADIUS * 0.6)   # idle FOV range in px
        for a in self.agents:
            if a.evacuated:
                continue
            ix, iy = int(a.x), int(a.y)
            x0, x1 = max(0, ix - fov_r), min(self.walk_map.w, ix + fov_r + 1)
            y0, y1 = max(0, iy - fov_r), min(self.walk_map.h, iy + fov_r + 1)
            self.explored[y0:y1, x0:x1] = np.minimum(
                1.0, self.explored[y0:y1, x0:x1] + 0.4
            )

    def step(self):
        self.sim_time += DT
        for a in self.agents:
            a.update(DT, self.agents, self.exits, self.hazard)
        self._update_explored()

    def reset(self, cfg: dict):
        self.agents.clear()
        self.exits.clear()
        self.hazard = None
        self.sim_time = 0.0
        self.explored[:] = 0
        self._spawn(cfg)

    @property
    def active_count(self):
        return sum(1 for a in self.agents if not a.evacuated)

    @property
    def evacuated_count(self):
        return sum(1 for a in self.agents if a.evacuated)


# ══════════════════════════════════════════════════════════════════════
#  SimView — render widget
# ══════════════════════════════════════════════════════════════════════

class SimView(QWidget):

    def __init__(self):
        super().__init__()
        self.sim: Optional[Simulation] = None
        self.bg_pixmap: Optional[QPixmap] = None
        self.selected: Optional[Agent] = None
        self.show_fov = True
        self.show_memory = True
        self._exit_mode = False
        self._hazard_mode = False
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)

    def load(self, sim: Simulation, mask_path: str):
        self.sim = sim
        img = cv2.imread(mask_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, c = img.shape
        qimg = QImage(img.tobytes(), w, h, w * c, QImage.Format.Format_RGB888)
        self.bg_pixmap = QPixmap.fromImage(qimg)
        self.update()

    # ── coordinate transforms ─────────────────────────────────────────

    def _layout(self):
        if self.bg_pixmap is None:
            return 1.0, 0.0, 0.0
        s = min(self.width() / self.bg_pixmap.width(),
                self.height() / self.bg_pixmap.height())
        ox = (self.width()  - self.bg_pixmap.width()  * s) / 2
        oy = (self.height() - self.bg_pixmap.height() * s) / 2
        return s, ox, oy

    def w2s(self, wx, wy):
        s, ox, oy = self._layout()
        return wx * s + ox, wy * s + oy

    def s2w(self, sx, sy):
        s, ox, oy = self._layout()
        return (sx - ox) / s, (sy - oy) / s

    # ── mouse ─────────────────────────────────────────────────────────

    def mousePressEvent(self, event):
        if self.sim is None:
            return
        wx, wy = self.s2w(event.position().x(), event.position().y())
        s, _, _ = self._layout()

        if self._exit_mode:
            self.sim.add_exit(wx, wy)
            self._exit_mode = False
            self.update()
            return

        if self._hazard_mode:
            self.sim.set_hazard(wx, wy)
            self._hazard_mode = False
            self.update()
            return

        # Inspector click
        pick_r = 15 / s
        best, best_d = None, pick_r
        for a in self.sim.agents:
            if a.evacuated:
                continue
            d = math.hypot(a.x - wx, a.y - wy)
            if d < best_d:
                best_d, best = d, a
        if self.selected:
            self.selected.selected = False
        self.selected = best
        if best:
            best.selected = True
        self.update()

    # ── paint ─────────────────────────────────────────────────────────

    def paintEvent(self, _):
        if self.sim is None:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        s, ox, oy = self._layout()

        # Background map
        if self.bg_pixmap:
            from PyQt6.QtCore import QRectF
            iw, ih = self.bg_pixmap.width(), self.bg_pixmap.height()
            p.drawPixmap(QRectF(ox, oy, iw * s, ih * s).toRect(), self.bg_pixmap)

        # Explored area overlay — single imshow-style draw using QImage
        if self.show_memory and self.sim is not None:
            exp = self.sim.explored
            # Build RGBA array: cyan tint where explored
            rgba = np.zeros((exp.shape[0], exp.shape[1], 4), dtype=np.uint8)
            alpha = (np.clip(exp, 0, 1) * 255 * EXPLORED_ALPHA).astype(np.uint8)
            rgba[:, :, 0] = 0
            rgba[:, :, 1] = 200
            rgba[:, :, 2] = 200
            rgba[:, :, 3] = alpha
            h_e, w_e = exp.shape
            mem_img = QImage(rgba.tobytes(), w_e, h_e, w_e * 4, QImage.Format.Format_RGBA8888)
            mem_pix = QPixmap.fromImage(mem_img)
            from PyQt6.QtCore import QRectF
            p.drawPixmap(QRectF(ox, oy, w_e * s, h_e * s).toRect(), mem_pix)

        # Exits
        for ex in self.sim.exits:
            sx, sy = self.w2s(ex.x, ex.y)
            r = ex.radius * s
            color = QColor(255, 80, 80) if ex.blocked else QColor(0, 220, 100)
            p.setPen(QPen(color.darker(130), 2))
            p.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 160)))
            p.drawEllipse(QPointF(sx, sy), r, r)
            p.setPen(QPen(Qt.GlobalColor.white))
            p.drawText(int(sx - 8), int(sy + 4), "EXIT")

        # Hazard
        if self.sim.hazard:
            hx, hy = self.w2s(*self.sim.hazard)
            p.setPen(QPen(QColor(255, 120, 0), 2))
            p.setBrush(QBrush(QColor(255, 60, 0, 140)))
            p.drawEllipse(QPointF(hx, hy), 18 * s, 18 * s)

        # Agents
        for a in self.sim.agents:
            if a.evacuated:
                continue
            ax, ay = self.w2s(a.x, a.y)
            r = max(3, AGENT_RADIUS * s)

            # FOV cone
            if self.show_fov and (a.selected or len(self.sim.agents) <= 60):
                fov_range_s = (SWEEP_RADIUS * 0.6) * s
                half = math.radians(FOV_DEG / 2)
                path = QPainterPath()
                path.moveTo(ax, ay)
                for i in range(13):
                    t = -half + i * (2 * half / 12)
                    angle = a.angle + t
                    path.lineTo(ax + math.cos(angle) * fov_range_s,
                                ay + math.sin(angle) * fov_range_s)
                path.closeSubpath()
                p.setPen(Qt.PenStyle.NoPen)
                alpha = 70 if a.selected else 20
                p.setBrush(QBrush(QColor(255, 255, 180, alpha)))
                p.drawPath(path)

            # Body color by state
            if a.selected:
                body_color = QColor(255, 220, 0)
                p.setPen(QPen(QColor(255, 180, 0), 2))
            elif a.state == Agent.EVACUATE:
                body_color = QColor(220, 50, 50)
                p.setPen(QPen(QColor(150, 0, 0), 1))
            elif a.state == Agent.ALERT:
                body_color = QColor(230, 150, 30)
                p.setPen(QPen(QColor(160, 100, 0), 1))
            else:
                body_color = QColor(70, 140, 220)
                p.setPen(QPen(QColor(30, 80, 160), 1))

            p.setBrush(QBrush(body_color))
            p.drawEllipse(QPointF(ax, ay), r, r)

            # Facing arrow
            arrow_len = r * 2.0
            ex2 = ax + math.cos(a.angle) * arrow_len
            ey2 = ay + math.sin(a.angle) * arrow_len
            p.setPen(QPen(QColor(255, 255, 255, 200), max(1, r * 0.35)))
            p.drawLine(QPointF(ax, ay), QPointF(ex2, ey2))

        p.end()


# ══════════════════════════════════════════════════════════════════════
#  Inspector panel
# ══════════════════════════════════════════════════════════════════════

class InspectorPanel(QFrame):

    def __init__(self):
        super().__init__()
        self.setFixedWidth(220)
        lv = QVBoxLayout(self)
        lv.setContentsMargins(10, 10, 10, 10)
        lv.setSpacing(5)

        title = QLabel("Inspector")
        title.setStyleSheet("font-size:12pt; font-weight:bold; color:#e94560;")
        lv.addWidget(title)

        self.id_lbl     = QLabel("Click an agent")
        self.state_lbl  = QLabel("")
        self.stress_lbl = QLabel("")
        self.exits_lbl  = QLabel("")
        self.pos_lbl    = QLabel("")
        for lbl in [self.id_lbl, self.state_lbl, self.stress_lbl,
                    self.exits_lbl, self.pos_lbl]:
            lbl.setStyleSheet("color:#ccc; font-size:9pt;")
            lbl.setWordWrap(True)
            lv.addWidget(lbl)

        lv.addStretch()

    def refresh(self, agent: Optional[Agent]):
        if agent is None:
            self.id_lbl.setText("Click an agent")
            for lbl in [self.state_lbl, self.stress_lbl, self.exits_lbl, self.pos_lbl]:
                lbl.setText("")
            return
        self.id_lbl.setText(f"Agent #{agent.id}")
        self.state_lbl.setText(f"State:   {agent.state}")
        self.stress_lbl.setText(f"Stress:  {agent.stress:.2f}")
        self.pos_lbl.setText(f"Pos:  ({agent.x:.0f}, {agent.y:.0f})")
        if agent.known_exits:
            exits_str = "\n".join(f"  ({ex[0]:.0f}, {ex[1]:.0f})" for ex in agent.known_exits)
            self.exits_lbl.setText(f"Known exits:\n{exits_str}")
        else:
            self.exits_lbl.setText("Known exits: none yet")


# ══════════════════════════════════════════════════════════════════════
#  Main Window
# ══════════════════════════════════════════════════════════════════════

STYLE = """
QMainWindow, QWidget { background:#1a1a2e; color:#e0e0e0;
    font-family:'Segoe UI', Arial, sans-serif; font-size:10pt; }
QPushButton { background:#16213e; border:1px solid #0f3460;
    border-radius:5px; padding:6px 12px; }
QPushButton:hover  { background:#0f3460; border-color:#e94560; }
QPushButton:pressed { background:#e94560; color:white; }
QPushButton#primary { background:#e94560; color:white; font-weight:bold; }
QFrame { border:1px solid #0f3460; border-radius:6px; }
QLabel { border:none; }
QSlider::groove:horizontal { background:#0f3460; height:4px; border-radius:2px; }
QSlider::handle:horizontal { background:#e94560; width:14px; height:14px;
    margin:-5px 0; border-radius:7px; }
"""


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("TRAGIC — Agent Simulation")
        self.setMinimumSize(1200, 750)
        self.setStyleSheet(STYLE)

        self.sim: Optional[Simulation] = None
        self.zone_cfg: dict = {}
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

        # Left panel
        left = QFrame()
        left.setFixedWidth(210)
        lv = QVBoxLayout(left)
        lv.setContentsMargins(10, 10, 10, 10)
        lv.setSpacing(8)

        title = QLabel("TRAGIC")
        title.setStyleSheet("font-size:15pt; font-weight:bold; color:#e94560; border:none;")
        lv.addWidget(title)

        self.load_btn = QPushButton("📂  Load Config")
        self.load_btn.clicked.connect(self.load_config)
        lv.addWidget(self.load_btn)

        self.run_btn = QPushButton("▶  Start")
        self.run_btn.setObjectName("primary")
        self.run_btn.setEnabled(False)
        self.run_btn.clicked.connect(self.toggle_pause)
        lv.addWidget(self.run_btn)

        self.reset_btn = QPushButton("↺  Reset")
        self.reset_btn.setEnabled(False)
        self.reset_btn.clicked.connect(self.reset_sim)
        lv.addWidget(self.reset_btn)

        lv.addWidget(self._sep())

        self.exit_btn = QPushButton("🚪  Place Exit")
        self.exit_btn.setEnabled(False)
        self.exit_btn.clicked.connect(self._place_exit)
        lv.addWidget(self.exit_btn)

        self.hazard_btn = QPushButton("🔥  Place Hazard")
        self.hazard_btn.setEnabled(False)
        self.hazard_btn.clicked.connect(self._place_hazard)
        lv.addWidget(self.hazard_btn)

        lv.addWidget(self._sep())

        lv.addWidget(QLabel("Speed:"))
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(1, 10)
        self.speed_slider.setValue(3)
        self.speed_slider.valueChanged.connect(self._update_speed)
        lv.addWidget(self.speed_slider)

        lv.addWidget(self._sep())

        self.fov_cb = QCheckBox("Show FOV")
        self.fov_cb.setChecked(True)
        self.fov_cb.toggled.connect(lambda v: setattr(self.view, 'show_fov', v) or self.view.update())
        lv.addWidget(self.fov_cb)

        self.mem_cb = QCheckBox("Show explored area")
        self.mem_cb.setChecked(True)
        self.mem_cb.toggled.connect(lambda v: setattr(self.view, 'show_memory', v) or self.view.update())
        lv.addWidget(self.mem_cb)

        lv.addWidget(self._sep())

        self.stat_lbl = QLabel("Load a zone config\nto begin.")
        self.stat_lbl.setStyleSheet("color:#888; font-size:9pt; border:none;")
        self.stat_lbl.setWordWrap(True)
        lv.addWidget(self.stat_lbl)
        lv.addStretch()

        # Sim view
        self.view = SimView()

        # Inspector
        self.inspector = InspectorPanel()

        rl.addWidget(left)
        rl.addWidget(self.view, 1)
        rl.addWidget(self.inspector)

    def _sep(self):
        f = QFrame()
        f.setFrameShape(QFrame.Shape.HLine)
        f.setFixedHeight(1)
        f.setStyleSheet("border:none; background:#0f3460;")
        return f

    # ── loading ───────────────────────────────────────────────────────

    def load_config(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Zone Config JSON", "", "JSON (*.json)")
        if not path:
            return

        with open(path) as f:
            self.zone_cfg = json.load(f)

        mask_path = self.zone_cfg.get('mask_path', '')
        if not Path(mask_path).exists():
            mask_path, _ = QFileDialog.getOpenFileName(
                self, "Locate Mask Image", "", "Images (*.png *.jpg *.bmp)")
            if not mask_path:
                return
            self.zone_cfg['mask_path'] = mask_path

        try:
            walk_map = WalkMap(mask_path)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            return

        self.sim = Simulation(walk_map, self.zone_cfg)
        self.view.load(self.sim, mask_path)

        for btn in [self.run_btn, self.reset_btn, self.exit_btn, self.hazard_btn]:
            btn.setEnabled(True)

        self.paused = True
        self.run_btn.setText("▶  Start")
        self._update_stats()

    # ── controls ─────────────────────────────────────────────────────

    def toggle_pause(self):
        if self.sim is None:
            return
        self.paused = not self.paused
        if self.paused:
            self.timer.stop()
            self.run_btn.setText("▶  Resume")
        else:
            self.timer.start()
            self.run_btn.setText("⏸  Pause")

    def reset_sim(self):
        if self.sim is None:
            return
        self.timer.stop()
        self.paused = True
        self.run_btn.setText("▶  Start")
        self.sim.reset(self.zone_cfg)
        self.view.selected = None
        self.view.update()
        self._update_stats()

    def _place_exit(self):
        if self.view:
            self.view._exit_mode = True

    def _place_hazard(self):
        if self.view:
            self.view._hazard_mode = True

    def _update_speed(self, v):
        self.timer.setInterval(max(5, int(DT * 1000 // v)))

    # ── tick ──────────────────────────────────────────────────────────

    def _tick(self):
        if self.sim is None:
            return
        steps = self.speed_slider.value()
        for _ in range(steps):
            self.sim.step()
        self.view.update()
        self._update_stats()
        if self.view.selected:
            self.inspector.refresh(self.view.selected)

    def _update_stats(self):
        if self.sim is None:
            return
        self.stat_lbl.setText(
            f"Time:      {self.sim.sim_time:.1f}s\n"
            f"Active:    {self.sim.active_count}\n"
            f"Evacuated: {self.sim.evacuated_count}\n\n"
            f"🔵 idle  🟠 alert  🔴 evacuate\n"
            f"Space = pause,  R = reset\n"
            f"Click agent to inspect"
        )
        if not self.view.selected:
            self.inspector.refresh(None)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Space:
            self.toggle_pause()
        elif event.key() == Qt.Key.Key_R:
            self.reset_sim()


# ══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())