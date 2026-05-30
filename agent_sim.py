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
import heapq
import numpy as np
import cv2
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

from navmesh import NavMesh

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QLabel, QFileDialog, QFrame, QSizePolicy, QScrollArea,
    QMessageBox, QCheckBox, QSlider
)
from PyQt6.QtCore import Qt, QTimer, QPointF, QRectF
from PyQt6.QtGui import (
    QPainter, QColor, QPen, QBrush,
    QImage, QPixmap, QPainterPath
)

from agents import Agent as CoreAgent, AgentState, AgentProfile


# ── Simulation constants ───────────────────────────────────────────────
DT              = 0.05
AGENT_RADIUS    = 6
FOV_DEGREES     = 120
FOV_RANGE       = 80
PANIC_FOV_RANGE = 140
SPEED_BASE      = 35
SPEED_VARIANCE  = 0.25
NEIGHBOR_RADIUS = 40
WALL_BUFFER     = AGENT_RADIUS + 2
CIRCLE_SWEEP_INTERVAL = 3.0
MEMORY_DECAY    = 120.0

# Perception / memory
RAY_COUNT       = 12
PANIC_SWEEP_THR = 0.35
MEM_DEGRADE_THR = 0.70
MEM_DEGRADE_P   = 0.30

# Panic model
PANIC_RADIUS          = 140.0
AWARE_HERD_WEIGHT     = 0.30
DISTRESS_HERD_WEIGHT  = 0.60
PANIC_HERD_WEIGHT     = 1.00
DISTRESS_FOV_DEG      = 90.0
PANIC_FOV_MIN_DEG     = 60.0
DISTRESS_RADIUS_SCALE = 0.85
PANIC_RADIUS_SCALE    = 0.70
CALM_SPEED_SCALE      = 0.95
AWARE_SPEED_SCALE     = 1.05
DISTRESS_SPEED_SCALE  = 1.18
PANIC_SPEED_SCALE     = 1.30

# Hierarchical navigation
WAYPOINT_REACHED_DIST    = 14.0
WAYPOINT_BLOCKED_DENSITY = 6
GRAPH_LINK_RADIUS        = 220.0
LOCAL_TARGET_WEIGHT      = 1.25
WANDER_WEIGHT            = 0.15
REPLAN_COOLDOWN          = 0.35
STUCK_REPLAN_TICKS       = 20

MAX_KNOWN_CELLS    = 400
MAX_MEMORY_ENTRIES = 16


# ══════════════════════════════════════════════════════════════════════
#  Walkability map
# ══════════════════════════════════════════════════════════════════════

class WalkMap:
    """Thin wrapper around a binary walkability image."""
    def __init__(self, mask_path: str):
        img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(mask_path)

        # White = wall, black = walkable
        self.walkable = img < 128
        self.h, self.w = img.shape

        walk_uint8 = self.walkable.astype(np.uint8) * 255
        self.dist = cv2.distanceTransform(walk_uint8, cv2.DIST_L2, 5)

    def is_walkable(self, x: float, y: float) -> bool:
        ix, iy = int(x), int(y)
        if ix < 0 or iy < 0 or ix >= self.w or iy >= self.h:
            return False
        return bool(self.walkable[iy, ix])

    def wall_repulsion(self, x: float, y: float) -> Tuple[float, float]:
        ix = int(np.clip(x, 0, self.w - 1))
        iy = int(np.clip(y, 0, self.h - 1))
        d = self.dist[iy, ix]
        if d >= WALL_BUFFER * 2:
            return 0.0, 0.0

        gx = (self.dist[iy, min(ix + 1, self.w - 1)] -
              self.dist[iy, max(ix - 1, 0)]) / 2
        gy = (self.dist[min(iy + 1, self.h - 1), ix] -
              self.dist[max(iy - 1, 0), ix]) / 2
        strength = max(0, (WALL_BUFFER * 2 - d)) / (WALL_BUFFER * 2)
        return gx * strength * 150, gy * strength * 150


# ══════════════════════════════════════════════════════════════════════
#  Data helpers
# ══════════════════════════════════════════════════════════════════════

@dataclass
class MemoryEntry:
    kind: str
    position: Tuple[float, float]
    time_seen: float
    confidence: float = 1.0


@dataclass
class NavNode:
    node_id: int
    kind: str
    pos: Tuple[float, float]


# ══════════════════════════════════════════════════════════════════════
#  Agent_IM
# ══════════════════════════════════════════════════════════════════════

class Agent(CoreAgent):
    _id_counter = 0

    def __init__(self, x: float, y: float, walk_map: WalkMap,
                 exits: Optional[Dict[int, Tuple[float, float]]] = None,
                 hazards: Optional[Dict[int, Tuple[float, float]]] = None):
        Agent._id_counter += 1
        profile = random.choice(list(AgentProfile))

        super().__init__(
            id=Agent._id_counter,
            pos=np.array([x, y], dtype=float),
            velocity=np.zeros(2, dtype=float),
            facing=random.uniform(0, math.tau),
            panic=0.0,
            memory={},
            goal=None,
            state=AgentState.CALM,
            profile=profile,
            speed_max=SPEED_BASE * random.uniform(1 - SPEED_VARIANCE, 1 + SPEED_VARIANCE),
            shoulder_radius=AGENT_RADIUS / 20.0,
        )

        self.x, self.y = x, y
        self.speed = self.speed_max
        self.angle = self.facing
        self.vx = math.cos(self.angle) * self.speed
        self.vy = math.sin(self.angle) * self.speed
        self.walk_map = walk_map

        # Shared world registers
        self._exits = exits if exits is not None else {}
        self._hazards = hazards if hazards is not None else {}

        self.memory_entries: List[MemoryEntry] = []
        self.known_map_cells: set = set()

        # Sweep state
        self._sweep_active = False
        self._sweep_tick = 0
        self._sweep_total_ticks = math.ceil(360 / FOV_DEGREES)
        self._sweep_angle_offset = 0.0
        self._sweep_triggered = False
        self.panic_delay_remaining = 0.0

        self.last_sweep_time = random.uniform(0, CIRCLE_SWEEP_INTERVAL)
        self.sim_time = 0.0
        self._target_angle = self.angle
        self._wander_timer = random.uniform(0.0, 2.0)
        self.selected = False
        self.fov_visible_ids: set = set()

        # Panic model
        self.alpha = {
            AgentProfile.ADULT: 0.045,
            AgentProfile.CHILD: 0.060,
            AgentProfile.ELDERLY: 0.040,
            AgentProfile.MOBILITY_IMPAIRED: 0.050,
        }.get(profile, 0.045)

        self.beta = {
            AgentProfile.ADULT: 0.018,
            AgentProfile.CHILD: 0.015,
            AgentProfile.ELDERLY: 0.020,
            AgentProfile.MOBILITY_IMPAIRED: 0.016,
        }.get(profile, 0.018)

        self.herd_weight = 0.0
        self.goal_weight = 1.0
        self.current_fov_deg = FOV_DEGREES
        self.collision_scale = 1.0
        self.pathfinding_quality = 1.0
        self.panic_state_name = "CALM"

        # Hierarchical navigation state
        self.current_exit_id: Optional[int] = None
        self.current_path_nodes: List[int] = []
        self.current_path_points: List[Tuple[float, float]] = []
        self.path_index = 0
        self.last_planned_exit_count = 0
        self.last_replan_time = -999.0
        self.follow_crowd_only = False
        self.stuck_ticks = 0
        self._last_pos_for_stuck = (self.x, self.y)

    # ── Geometry ────────────────────────────────────────────────────

    def _in_fov(self, tx: float, ty: float, angle_override: Optional[float] = None) -> bool:
        dx, dy = tx - self.x, ty - self.y
        base = self.angle if angle_override is None else angle_override
        angle_to = math.atan2(dy, dx)
        diff = (angle_to - base + math.pi) % math.tau - math.pi
        return abs(diff) <= math.radians(self.current_fov_deg / 2)

    def _ray_clear(self, tx: float, ty: float, max_range: Optional[float] = None) -> bool:
        dx, dy = tx - self.x, ty - self.y
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            return True
        if max_range is not None and dist > max_range:
            return False

        steps = int(min(dist, max_range if max_range is not None else FOV_RANGE))
        if steps <= 0:
            return True

        sx, sy = dx / dist, dy / dist
        for i in range(1, steps + 1):
            px = self.x + sx * i
            py = self.y + sy * i
            if not self.walk_map.is_walkable(px, py):
                return False
        return True

    def _point_visible(self, tx: float, ty: float, max_range: float = FOV_RANGE) -> bool:
        dist = math.hypot(tx - self.x, ty - self.y)
        if dist > max_range:
            return False
        return self._in_fov(tx, ty) and self._ray_clear(tx, ty, max_range)

    # ── Panic model ─────────────────────────────────────────────────

    def _nearest_hazard_distance(self) -> float:
        if not self._hazards:
            return float("inf")
        return min(math.hypot(self.x - hx, self.y - hy) for hx, hy in self._hazards.values())

    def _is_open_air_safe(self) -> bool:
        return self._nearest_hazard_distance() > PANIC_RADIUS

    def update_panic_pac(self, dt: float):
        dist = self._nearest_hazard_distance()
        hazard_signal = max(0.0, 1.0 - dist / PANIC_RADIUS) if math.isfinite(dist) else 0.0
        open_air_dt = dt if self._is_open_air_safe() else 0.0

        self.panic = float(np.clip(
            self.panic + (self.alpha * hazard_signal) - (self.beta * open_air_dt),
            0.0, 1.0
        ))
        self._apply_panic_state()

    def _apply_panic_state(self):
        if self.panic <= 0.2:
            self.panic_state_name = "CALM"
            self.state = AgentState.CALM
            self.herd_weight = 0.0
            self.goal_weight = 1.0
            self.current_fov_deg = FOV_DEGREES
            self.collision_scale = 1.0
            self.pathfinding_quality = 1.0
            self.speed = self.speed_max * CALM_SPEED_SCALE

        elif self.panic <= 0.5:
            self.panic_state_name = "AWARE"
            self.state = getattr(AgentState, "AWARE", AgentState.CALM)
            self.herd_weight = AWARE_HERD_WEIGHT
            self.goal_weight = 1.0 - self.herd_weight
            self.current_fov_deg = FOV_DEGREES
            self.collision_scale = 1.0
            self.pathfinding_quality = 0.95
            self.speed = self.speed_max * AWARE_SPEED_SCALE

        elif self.panic <= 0.75:
            self.panic_state_name = "DISTRESSED"
            self.state = getattr(AgentState, "PANICKING", AgentState.CALM)
            self.herd_weight = DISTRESS_HERD_WEIGHT
            self.goal_weight = 1.0 - self.herd_weight
            self.current_fov_deg = DISTRESS_FOV_DEG
            self.collision_scale = DISTRESS_RADIUS_SCALE
            self.pathfinding_quality = 0.65
            self.speed = self.speed_max * DISTRESS_SPEED_SCALE

        else:
            self.panic_state_name = "PANIC"
            self.state = getattr(AgentState, "PANICKING", AgentState.CALM)
            self.herd_weight = PANIC_HERD_WEIGHT
            self.goal_weight = 0.0
            self.current_fov_deg = PANIC_FOV_MIN_DEG
            self.collision_scale = PANIC_RADIUS_SCALE
            self.pathfinding_quality = 0.25
            self.speed = self.speed_max * PANIC_SPEED_SCALE

    # ── Memory ──────────────────────────────────────────────────────

    def _store_memory(self, uid: int, kind: str, pos: Tuple[float, float], t: float):
        self.memory[uid] = {"kind": kind, "pos": pos, "t": t, "certain": True}

        for m in self.memory_entries:
            if m.kind == kind and math.hypot(m.position[0] - pos[0], m.position[1] - pos[1]) < 20:
                m.time_seen = t
                m.confidence = 1.0
                return

        self.memory_entries.append(MemoryEntry(kind, pos, t))
        if len(self.memory_entries) > MAX_MEMORY_ENTRIES:
            self.memory_entries.sort(key=lambda m: m.time_seen, reverse=True)
            self.memory_entries = self.memory_entries[:MAX_MEMORY_ENTRIES]

    def add_exit_memory(self, pos: Tuple[float, float], t: float, uid: int = -1):
        self._store_memory(uid if uid != -1 else hash(pos), "exit", pos, t)

    def add_hazard_memory(self, pos: Tuple[float, float], t: float, uid: int = -1):
        self._store_memory(uid if uid != -1 else hash(pos), "hazard", pos, t)

    def manage_memory(self, all_agents: List['Agent'], sim_time: float):
        self.memory_entries = [m for m in self.memory_entries if sim_time - m.time_seen < MEMORY_DECAY]

        stale = [uid for uid, v in self.memory.items() if sim_time - v["t"] >= MEMORY_DECAY]
        for uid in stale:
            del self.memory[uid]

        if self.panic > MEM_DEGRADE_THR:
            if random.random() < self.panic * MEM_DEGRADE_P:
                for v in self.memory.values():
                    if v["kind"] == "exit":
                        v["certain"] = False
                self._do_herding(all_agents)

    # ── Herding ─────────────────────────────────────────────────────

    def _do_herding(self, all_agents: List['Agent']):
        best_d, best = float("inf"), None
        for other in all_agents:
            if other.id == self.id:
                continue
            d = math.hypot(self.x - other.x, self.y - other.y)
            if d < FOV_RANGE and d < best_d and self._in_fov(other.x, other.y):
                best_d, best = d, other

        if best is None:
            return

        herd_angle = math.atan2(best.vy, best.vx)
        goal_angle = self._target_angle

        hx, hy = math.cos(herd_angle), math.sin(herd_angle)
        gx, gy = math.cos(goal_angle), math.sin(goal_angle)

        bx = self.herd_weight * hx + self.goal_weight * gx
        by = self.herd_weight * hy + self.goal_weight * gy

        if abs(bx) > 1e-6 or abs(by) > 1e-6:
            self._target_angle = math.atan2(by, bx)

    # ── Perception ──────────────────────────────────────────────────

    def update_perception(self, sim_time: float):
        # Exits in normal FOV
        for eid, epos in self._exits.items():
            if self._point_visible(epos[0], epos[1], FOV_RANGE):
                self._store_memory(eid, "exit", epos, sim_time)

        # Hazards in normal FOV
        for hid, hpos in self._hazards.items():
            if self._point_visible(hpos[0], hpos[1], FOV_RANGE):
                self._store_memory(hid, "hazard", hpos, sim_time)
                self.panic = max(self.panic, PANIC_SWEEP_THR)
                self._apply_panic_state()

    def check_immediate_awareness(self, hazard_pos: Tuple[float, float]) -> bool:
        if self._point_visible(hazard_pos[0], hazard_pos[1], FOV_RANGE):
            self._store_memory(hash(("haz0", hazard_pos)), "hazard", hazard_pos, self.sim_time)
            self.panic = max(self.panic, PANIC_SWEEP_THR)
            self._apply_panic_state()
            self._sweep_triggered = True
            return True
        return False

    # ── Sweeps ──────────────────────────────────────────────────────

    def check_sweeps(self, sim_time: float):
        if (not self._sweep_triggered
                and self.panic >= PANIC_SWEEP_THR
                and self.panic_delay_remaining <= 0.0):
            self._sweep_active = True
            self._sweep_tick = 0
            self._sweep_triggered = True
            self._sweep_angle_offset = self.angle

        if (not self._sweep_active
                and sim_time - self.last_sweep_time >= CIRCLE_SWEEP_INTERVAL):
            self._sweep_active = True
            self._sweep_tick = 0
            self._sweep_angle_offset = self.angle
            self.last_sweep_time = sim_time

        if self._sweep_active:
            self._do_sweep_tick(sim_time)

    def _do_sweep_tick(self, sim_time: float):
        tick_deg = 360.0 / self._sweep_total_ticks
        scan_base = self._sweep_angle_offset + math.radians(tick_deg * self._sweep_tick)

        def visible_with_base(tx: float, ty: float, max_range: float) -> bool:
            dx, dy = tx - self.x, ty - self.y
            dist = math.hypot(dx, dy)
            if dist > max_range:
                return False
            angle_to = math.atan2(dy, dx)
            diff = (angle_to - scan_base + math.pi) % math.tau - math.pi
            if abs(diff) > math.radians(self.current_fov_deg / 2):
                return False
            return self._ray_clear(tx, ty, max_range)

        for eid, epos in self._exits.items():
            if visible_with_base(epos[0], epos[1], PANIC_FOV_RANGE):
                self._store_memory(eid, "exit", epos, sim_time)

        for hid, hpos in self._hazards.items():
            if visible_with_base(hpos[0], hpos[1], PANIC_FOV_RANGE):
                self._store_memory(hid, "hazard", hpos, sim_time)
                self.panic = max(self.panic, PANIC_SWEEP_THR)
                self._apply_panic_state()

        self._sweep_tick += 1
        if self._sweep_tick >= self._sweep_total_ticks:
            self._sweep_active = False
            self._sweep_tick = 0

    # ── Hierarchical planning ───────────────────────────────────────

    def get_best_exit_from_memory(self) -> Optional[int]:
        exit_candidates = []
        for uid, item in self.memory.items():
            if item.get("kind") == "exit":
                pos = item["pos"]
                d = math.hypot(pos[0] - self.x, pos[1] - self.y)
                certainty_bonus = 0.0 if item.get("certain", True) else 25.0
                exit_candidates.append((d + certainty_bonus, uid))

        if not exit_candidates:
            return None
        exit_candidates.sort(key=lambda x: x[0])
        return exit_candidates[0][1]

    def is_waypoint_blocked(self, sim: 'Simulation', waypoint: Tuple[float, float]) -> bool:
        if not self.walk_map.is_walkable(*waypoint):
            return True

        crowd_count = 0
        for other in sim.agents:
            if other.id == self.id:
                continue
            if math.hypot(other.x - waypoint[0], other.y - waypoint[1]) < NEIGHBOR_RADIUS:
                crowd_count += 1
                if crowd_count >= WAYPOINT_BLOCKED_DENSITY:
                    return True

        for _, hpos in sim.hazards.items():
            if math.hypot(hpos[0] - waypoint[0], hpos[1] - waypoint[1]) < AGENT_RADIUS * 6:
                return True

        return False

    def plan_high_level_path(self, sim: 'Simulation'):
        if self.panic >= 0.75:
            self.follow_crowd_only = True
            self.current_exit_id = None
            self.current_path_nodes = []
            self.current_path_points = []
            self.path_index = 0
            self.last_replan_time = self.sim_time
            return

        target_exit_id = self.get_best_exit_from_memory()
        if target_exit_id is None:
            self.current_exit_id = None
            self.current_path_nodes = []
            self.current_path_points = []
            self.path_index = 0
            self.follow_crowd_only = False
            self.last_planned_exit_count = 0
            self.last_replan_time = self.sim_time
            return

        start_node = sim.find_nearest_graph_node((self.x, self.y))
        goal_node = sim.exit_node_lookup.get(target_exit_id)

        if start_node is None or goal_node is None:
            self.current_exit_id = target_exit_id
            self.current_path_nodes = []
            self.current_path_points = [self._exits[target_exit_id]] if target_exit_id in self._exits else []
            self.path_index = 0
            self.follow_crowd_only = False
            self.last_planned_exit_count = len([k for k, v in self.memory.items() if v.get("kind") == "exit"])
            self.last_replan_time = self.sim_time
            self.goal = self._exits.get(target_exit_id)
            return

        node_path = sim.astar_path(start_node, goal_node)
        if not node_path:
            self.current_exit_id = target_exit_id
            self.current_path_nodes = []
            self.current_path_points = [self._exits[target_exit_id]] if target_exit_id in self._exits else []
            self.path_index = 0
            self.follow_crowd_only = False
            self.last_planned_exit_count = len([k for k, v in self.memory.items() if v.get("kind") == "exit"])
            self.last_replan_time = self.sim_time
            self.goal = self._exits.get(target_exit_id)
            return

        self.current_exit_id = target_exit_id
        self.current_path_nodes = node_path
        self.current_path_points = [sim.nav_nodes[nid].pos for nid in node_path]
        self.path_index = 0
        self.follow_crowd_only = False
        self.last_planned_exit_count = len([k for k, v in self.memory.items() if v.get("kind") == "exit"])
        self.last_replan_time = self.sim_time
        self.goal = self._exits.get(target_exit_id)

    def should_replan(self, sim: 'Simulation') -> bool:
        if self.sim_time - self.last_replan_time < REPLAN_COOLDOWN:
            return False

        if self.panic >= 0.75:
            return not self.follow_crowd_only

        current_exit_count = len([k for k, v in self.memory.items() if v.get("kind") == "exit"])
        if current_exit_count > self.last_planned_exit_count:
            return True

        if self.current_exit_id is None:
            return current_exit_count > 0

        if self.path_index < len(self.current_path_points):
            wp = self.current_path_points[self.path_index]
            if self.is_waypoint_blocked(sim, wp):
                return True

        if self.current_exit_id in self._exits:
            if self.is_waypoint_blocked(sim, self._exits[self.current_exit_id]):
                return True

        moved = math.hypot(self.x - self._last_pos_for_stuck[0], self.y - self._last_pos_for_stuck[1])
        if moved < 2.0:
            self.stuck_ticks += 1
        else:
            self.stuck_ticks = 0
            self._last_pos_for_stuck = (self.x, self.y)

        if self.stuck_ticks > STUCK_REPLAN_TICKS:
            self.stuck_ticks = 0
            self._last_pos_for_stuck = (self.x, self.y)
            return True

        return False

    def get_next_steering_target(self, sim: 'Simulation') -> Optional[Tuple[float, float]]:
        if self.follow_crowd_only:
            return None

        while self.path_index < len(self.current_path_points):
            wp = self.current_path_points[self.path_index]
            if math.hypot(self.x - wp[0], self.y - wp[1]) <= WAYPOINT_REACHED_DIST:
                self.path_index += 1
            else:
                return wp

        if self.current_exit_id is not None and self.current_exit_id in self._exits:
            return self._exits[self.current_exit_id]

        return None

    def update_navigation(self, sim: 'Simulation', all_agents: List['Agent']):
        if self.should_replan(sim):
            self.plan_high_level_path(sim)

        if self.panic >= 0.75:
            self.follow_crowd_only = True
            self._do_herding(all_agents)
            return

        steering_target = self.get_next_steering_target(sim)
        if steering_target is not None:
            tx, ty = steering_target
            goal_angle = math.atan2(ty - self.y, tx - self.x)

            gx, gy = math.cos(goal_angle), math.sin(goal_angle)
            wx, wy = math.cos(self._target_angle), math.sin(self._target_angle)

            blend_x = LOCAL_TARGET_WEIGHT * gx + WANDER_WEIGHT * wx
            blend_y = LOCAL_TARGET_WEIGHT * gy + WANDER_WEIGHT * wy
            if abs(blend_x) > 1e-6 or abs(blend_y) > 1e-6:
                self._target_angle = math.atan2(blend_y, blend_x)

    # ── Main update ─────────────────────────────────────────────────

    def update(self, dt: float, all_agents: List['Agent'], sim_time: float, sim: 'Simulation'):
        self.sim_time = sim_time

        # PAC panic update must happen every tick
        self.update_panic_pac(dt)

        if self.panic_delay_remaining > 0.0:
            self.panic_delay_remaining = max(0.0, self.panic_delay_remaining - dt)

        self.update_perception(sim_time)
        self.check_sweeps(sim_time)
        self.update_navigation(sim, all_agents)

        self._wander_timer -= dt
        if self._wander_timer <= 0:
            if not self.current_path_points and not self.follow_crowd_only:
                self._target_angle = self.angle + random.gauss(0, 0.8)
            self._wander_timer = random.uniform(0.8, 2.5)

        diff = (self._target_angle - self.angle + math.pi) % math.tau - math.pi
        self.angle = (self.angle + diff * min(1.0, dt * 4.0)) % math.tau

        rx, ry = self.walk_map.wall_repulsion(self.x, self.y)

        sx, sy = 0.0, 0.0
        self.fov_visible_ids.clear()
        effective_neighbor_radius = NEIGHBOR_RADIUS * self.collision_scale

        for other in all_agents:
            if other.id == self.id:
                continue
            dx, dy = self.x - other.x, self.y - other.y
            dist = math.hypot(dx, dy)

            if dist < effective_neighbor_radius and dist > 1e-6:
                push = max(0.0, (effective_neighbor_radius - dist)) / max(effective_neighbor_radius, 1e-6)
                sx += (dx / dist) * push * 60
                sy += (dy / dist) * push * 60

            if dist < FOV_RANGE and self._in_fov(other.x, other.y):
                self.fov_visible_ids.add(other.id)

        if self.herd_weight > 0.0:
            self._do_herding(all_agents)

        wx = math.cos(self.angle) * self.speed + rx + sx
        wy = math.sin(self.angle) * self.speed + ry + sy
        self.vx, self.vy = wx, wy

        nx, ny = self.x + wx * dt, self.y + wy * dt
        if self.walk_map.is_walkable(nx, ny):
            self.x, self.y = nx, ny
        elif self.walk_map.is_walkable(nx, self.y):
            self.x = nx
            self._target_angle = math.atan2(-wy, wx)
        elif self.walk_map.is_walkable(self.x, ny):
            self.y = ny
            self._target_angle = math.atan2(wy, -wx)
        else:
            self._target_angle = self.angle + math.pi + random.uniform(-0.5, 0.5)

        self.pos = np.array([self.x, self.y], dtype=float)
        self.velocity = np.array([self.vx, self.vy], dtype=float)
        self.facing = self.angle

        if len(self.known_map_cells) < MAX_KNOWN_CELLS:
            self.known_map_cells.add((int(self.x) // 10, int(self.y) // 10))

        self.manage_memory(all_agents, sim_time)


# ══════════════════════════════════════════════════════════════════════
#  Simulation
# ══════════════════════════════════════════════════════════════════════

class Simulation:
    def __init__(self, walk_map: WalkMap, zone_config: dict):
        self.walk_map = walk_map
        self.agents: List[Agent] = []
        self.sim_time = 0.0

        # Shared world registers
        self.exits: Dict[int, Tuple[float, float]] = {}
        self.hazards: Dict[int, Tuple[float, float]] = {}

        # Navigation graph
        self.nav_nodes: Dict[int, NavNode] = {}
        self.nav_graph: Dict[int, List[Tuple[int, float]]] = {}
        self.exit_node_lookup: Dict[int, int] = {}
        self._next_nav_node_id = 1

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
        self.build_navigation_graph()

    def _spawn_agents(self, cfg: dict):
        Agent._id_counter = 0
        self.agents.clear()

        zones = cfg.get("zones", [])
        if not zones:
            self._scatter_random(100)
            print(f"Spawned {len(self.agents)} agents")
            return

        ys, xs = np.where(self.walk_map.walkable)
        if len(xs) == 0:
            print("Warning: no walkable pixels found in mask!")
            return

        mask_path = cfg.get("mask_path", "")
        zone_labels = None
        if Path(mask_path).exists():
            try:
                zone_labels = self._rebuild_labels(mask_path)
                print("Zone labels rebuilt successfully")
            except Exception as e:
                print(f"Zone label rebuild failed ({e}), using global walkable pool")

        walkable_pool = list(zip(xs.tolist(), ys.tolist()))

        for zone in zones:
            d = zone.get("density_index", 1.0)
            if d <= 0:
                continue
            count = zone.get("agents", 0)
            if count <= 0:
                continue

            if zone_labels is not None:
                zid = zone["zone_id"]
                zm = (zone_labels == zid) & self.walk_map.walkable
                zy, zx = np.where(zm)
                pool = list(zip(zx.tolist(), zy.tolist())) if len(zx) > 0 else walkable_pool
            else:
                pool = walkable_pool

            for _ in range(count):
                px, py = random.choice(pool)
                self.agents.append(
                    Agent(float(px), float(py), self.walk_map,
                          exits=self.exits, hazards=self.hazards)
                )

        print(f"Spawned {len(self.agents)} agents")

    def _scatter_random(self, n: int):
        ys, xs = np.where(self.walk_map.walkable)
        for _ in range(n):
            idx = random.randint(0, len(xs) - 1)
            self.agents.append(
                Agent(float(xs[idx]), float(ys[idx]), self.walk_map,
                      exits=self.exits, hazards=self.hazards)
            )

    def _rebuild_labels(self, mask_path: str):
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

    # ── Navigation graph ────────────────────────────────────────────

    def _add_nav_node(self, kind: str, pos: Tuple[float, float]) -> int:
        nid = self._next_nav_node_id
        self._next_nav_node_id += 1
        self.nav_nodes[nid] = NavNode(nid, kind, pos)
        self.nav_graph[nid] = []
        return nid

    def _link_nav_nodes(self, a: int, b: int):
        pa = self.nav_nodes[a].pos
        pb = self.nav_nodes[b].pos
        cost = math.hypot(pa[0] - pb[0], pa[1] - pb[1])
        self.nav_graph[a].append((b, cost))
        self.nav_graph[b].append((a, cost))

    def _line_walkable(self, a: Tuple[float, float], b: Tuple[float, float], step: float = 6.0) -> bool:
        dx, dy = b[0] - a[0], b[1] - a[1]
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            return True
        n = max(1, int(dist / step))
        for i in range(n + 1):
            t = i / n
            x = a[0] + dx * t
            y = a[1] + dy * t
            if not self.walk_map.is_walkable(x, y):
                return False
        return True

    def build_navigation_graph(self):
        self.nav_nodes.clear()
        self.nav_graph.clear()
        self.exit_node_lookup.clear()
        self._next_nav_node_id = 1

        for exit_id, pos in self.exits.items():
            nid = self._add_nav_node("exit", pos)
            self.exit_node_lookup[exit_id] = nid

        sampled = []
        step = 80
        margin = 20
        for y in range(margin, self.walk_map.h - margin, step):
            for x in range(margin, self.walk_map.w - margin, step):
                if self.walk_map.is_walkable(x, y) and self.walk_map.dist[y, x] > AGENT_RADIUS * 2:
                    sampled.append((float(x), float(y)))

        for pos in sampled:
            self._add_nav_node("waypoint", pos)

        node_ids = list(self.nav_nodes.keys())
        for i, a in enumerate(node_ids):
            pa = self.nav_nodes[a].pos
            for b in node_ids[i + 1:]:
                pb = self.nav_nodes[b].pos
                d = math.hypot(pa[0] - pb[0], pa[1] - pb[1])
                if d <= GRAPH_LINK_RADIUS and self._line_walkable(pa, pb):
                    self._link_nav_nodes(a, b)

    def find_nearest_graph_node(self, pos: Tuple[float, float]) -> Optional[int]:
        best_id, best_d = None, float("inf")
        for nid, node in self.nav_nodes.items():
            d = math.hypot(node.pos[0] - pos[0], node.pos[1] - pos[1])
            if d < best_d and self._line_walkable(pos, node.pos):
                best_d, best_id = d, nid
        return best_id

    def astar_path(self, start_node: int, goal_node: int) -> List[int]:
        if start_node == goal_node:
            return [start_node]

        def heuristic(a: int, b: int) -> float:
            pa = self.nav_nodes[a].pos
            pb = self.nav_nodes[b].pos
            return math.hypot(pa[0] - pb[0], pa[1] - pb[1])

        open_heap = [(heuristic(start_node, goal_node), 0.0, start_node)]
        came_from: Dict[int, Optional[int]] = {start_node: None}
        g_score: Dict[int, float] = {start_node: 0.0}
        closed = set()

        while open_heap:
            _, g_curr, current = heapq.heappop(open_heap)
            if current in closed:
                continue

            if current == goal_node:
                path = []
                node = current
                while node is not None:
                    path.append(node)
                    node = came_from[node]
                path.reverse()
                return path

            closed.add(current)

            for neighbor, cost in self.nav_graph.get(current, []):
                if neighbor in closed:
                    continue
                tentative = g_curr + cost
                if tentative < g_score.get(neighbor, float("inf")):
                    g_score[neighbor] = tentative
                    came_from[neighbor] = current
                    f = tentative + heuristic(neighbor, goal_node)
                    heapq.heappush(open_heap, (f, tentative, neighbor))

        return []

    # ── Hazard trigger ──────────────────────────────────────────────

    def trigger_hazard(self, hazard_id: int, hazard_pos: Tuple[float, float], propagation_speed: float = 50.0):
        self.hazards[hazard_id] = hazard_pos

        for agent in self.agents:
            dist = math.hypot(agent.x - hazard_pos[0], agent.y - hazard_pos[1])

            if agent.check_immediate_awareness(hazard_pos):
                agent.panic_delay_remaining = 0.0
                continue

            agent.panic_delay_remaining = dist / propagation_speed

    # ── Tick/reset ──────────────────────────────────────────────────

    def step(self):
        self.sim_time += DT
        for agent in self.agents:
            agent.update(DT, self.agents, self.sim_time, self)

    def reset(self, zone_config: dict):
        self.agents.clear()
        self.sim_time = 0.0
        self.hazards.clear()
        self.exits.clear()
        self.build_navigation_graph()
        self._spawn_agents(zone_config)
        self.build_navigation_graph()


# ══════════════════════════════════════════════════════════════════════
#  View
# ══════════════════════════════════════════════════════════════════════

class SimView(QWidget):
    def __init__(self):
        super().__init__()
        self.sim: Optional[Simulation] = None
        self.bg_pixmap: Optional[QPixmap] = None
        self.selected_agent: Optional[Agent] = None
        self.show_fov = True
        self.show_memory = True
        self.show_explored = False
        self.show_navmesh = False
        self.setMinimumSize(400, 400)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)

    def set_sim(self, sim: Simulation, mask_path: str):
        self.sim = sim
        img = cv2.imread(mask_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, c = img.shape
        qimg = QImage(img.tobytes(), w, h, w * c, QImage.Format.Format_RGB888)
        self.bg_pixmap = QPixmap.fromImage(qimg)
        self.update()

    def _layout(self):
        if self.bg_pixmap is None:
            return 1.0, 0.0, 0.0
        iw, ih = self.bg_pixmap.width(), self.bg_pixmap.height()
        s = min(self.width() / iw, self.height() / ih)
        ox = (self.width() - iw * s) / 2
        oy = (self.height() - ih * s) / 2
        return s, ox, oy

    def _w2s(self, wx, wy):
        s, ox, oy = self._layout()
        return wx * s + ox, wy * s + oy

    def _s2w(self, sx, sy):
        s, ox, oy = self._layout()
        return (sx - ox) / s, (sy - oy) / s

    def mousePressEvent(self, event):
        if self.sim is None:
            return

        wx, wy = self._s2w(event.position().x(), event.position().y())
        win = self.window()
        if hasattr(win, "handle_map_click") and win.handle_map_click(wx, wy):
            self.update()
            return

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

        if self.bg_pixmap:
            iw, ih = self.bg_pixmap.width(), self.bg_pixmap.height()
            p.drawPixmap(QRectF(ox, oy, iw * s, ih * s).toRect(), self.bg_pixmap)

        if self.show_navmesh and hasattr(self.sim, "navmesh"):
            self.sim.navmesh.draw_debug(p, s, ox, oy)

        if self.show_explored and self.selected_agent:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(QColor(100, 200, 255, 40)))
            for (cx, cy) in self.selected_agent.known_map_cells:
                scx, scy = self._w2s(cx * 10, cy * 10)
                p.drawRect(int(scx), int(scy), int(10 * s), int(10 * s))

        # Draw exits
        for eid, (x, y) in self.sim.exits.items():
            ex, ey = self._w2s(x, y)
            p.setPen(QPen(QColor(0, 255, 120), 2))
            p.setBrush(QBrush(QColor(0, 255, 120, 120)))
            p.drawEllipse(QPointF(ex, ey), 8, 8)

        # Draw hazards
        for hid, (x, y) in self.sim.hazards.items():
            hx, hy = self._w2s(x, y)
            p.setPen(QPen(QColor(255, 80, 40), 2))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawLine(QPointF(hx - 8, hy - 8), QPointF(hx + 8, hy + 8))
            p.drawLine(QPointF(hx - 8, hy + 8), QPointF(hx + 8, hy - 8))

        if self.selected_agent and self.selected_agent.current_path_points:
            p.setPen(QPen(QColor(80, 220, 255, 180), 2))
            prev = (self.selected_agent.x, self.selected_agent.y)
            for wp in self.selected_agent.current_path_points[self.selected_agent.path_index:]:
                x1, y1 = self._w2s(*prev)
                x2, y2 = self._w2s(*wp)
                p.drawLine(QPointF(x1, y1), QPointF(x2, y2))
                p.setBrush(QBrush(QColor(80, 220, 255, 180)))
                p.drawEllipse(QPointF(x2, y2), 4, 4)
                prev = wp

        for a in self.sim.agents:
            ax, ay = self._w2s(a.x, a.y)
            r = max(3, AGENT_RADIUS * s)

            if self.show_fov and (a.selected or len(self.sim.agents) <= 80):
                fov_range_scaled = FOV_RANGE * s
                fov_half = math.radians(a.current_fov_deg / 2)
                path = QPainterPath()
                path.moveTo(ax, ay)
                steps = 12
                for i in range(steps + 1):
                    t = -fov_half + i * (2 * fov_half / steps)
                    ang = a.angle + t
                    path.lineTo(ax + math.cos(ang) * fov_range_scaled,
                                ay + math.sin(ang) * fov_range_scaled)
                path.closeSubpath()
                p.setPen(Qt.PenStyle.NoPen)
                alpha = 60 if a.selected else 18
                p.setBrush(QBrush(QColor(255, 255, 180, alpha)))
                p.drawPath(path)

            if self.show_memory and a.selected:
                for m in a.memory_entries:
                    mx, my = self._w2s(*m.position)
                    age = (self.sim.sim_time - m.time_seen) / MEMORY_DECAY
                    alpha = int(180 * max(0.0, 1.0 - age))
                    if m.kind == "exit":
                        color = QColor(0, 255, 100, alpha)
                    elif m.kind == "hazard":
                        color = QColor(255, 80, 0, alpha)
                    else:
                        color = QColor(180, 180, 255, alpha)
                    p.setPen(QPen(color, 2))
                    p.setBrush(QBrush(color))
                    p.drawEllipse(QPointF(mx, my), 4, 4)

            if a.selected:
                p.setPen(QPen(QColor(255, 255, 0), 2))
                p.setBrush(QBrush(QColor(255, 220, 0, 220)))
            else:
                red = int(200 * a.panic + 60 * (1 - a.panic))
                blue = int(180 * (1 - a.panic))
                p.setPen(QPen(QColor(0, 0, 0, 120), 1))
                p.setBrush(QBrush(QColor(red, 100, blue, 200)))
            p.drawEllipse(QPointF(ax, ay), r, r)

            arrow_len = r * 1.8
            ex = ax + math.cos(a.angle) * arrow_len
            ey = ay + math.sin(a.angle) * arrow_len
            p.setPen(QPen(QColor(255, 255, 255, 200), max(1, int(r * 0.4))))
            p.drawLine(QPointF(ax, ay), QPointF(ex, ey))

        p.end()


# ══════════════════════════════════════════════════════════════════════
#  Inspector
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

        self.id_label = QLabel("Click an agent to inspect")
        self.pos_label = QLabel("")
        self.speed_label = QLabel("")
        self.panic_label = QLabel("")
        self.cells_label = QLabel("")
        self.goal_label = QLabel("")
        for lbl in [self.id_label, self.pos_label, self.speed_label, self.panic_label, self.cells_label, self.goal_label]:
            lbl.setStyleSheet("color:#ccc; font-size:9pt;")
            lv.addWidget(lbl)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color:#333;")
        lv.addWidget(sep)

        mem_title = QLabel("Memory:")
        lv.addWidget(mem_title)

        self.memory_scroll = QScrollArea()
        self.memory_scroll.setWidgetResizable(True)
        self.memory_content = QWidget()
        self.memory_layout = QVBoxLayout(self.memory_content)
        self.memory_layout.setSpacing(2)
        self.memory_layout.setContentsMargins(0, 0, 0, 0)
        self.memory_scroll.setWidget(self.memory_content)
        self.memory_scroll.setStyleSheet("background:#111; border:none;")
        lv.addWidget(self.memory_scroll, 1)
        lv.addStretch()

    def update_agent(self, agent: Optional[Agent], sim_time: float):
        if agent is None:
            self.id_label.setText("Click an agent to inspect")
            self.pos_label.setText("")
            self.speed_label.setText("")
            self.panic_label.setText("")
            self.cells_label.setText("")
            self.goal_label.setText("")
            self._clear_memory()
            return

        self.id_label.setText(f"Agent #{agent.id}")
        self.pos_label.setText(f"Position:  ({agent.x:.0f}, {agent.y:.0f})")
        self.speed_label.setText(f"Speed:      {agent.speed:.1f} px/s")
        self.panic_label.setText(f"Panic:      {agent.panic:.2f} ({agent.panic_state_name})")
        self.cells_label.setText(f"Explored:   {len(agent.known_map_cells)} cells")

        if agent.follow_crowd_only:
            nav_text = "Nav:        crowd-follow"
        elif agent.current_exit_id is not None:
            nav_text = f"Nav:        exit {agent.current_exit_id}, wp {agent.path_index + 1}/{max(1, len(agent.current_path_points))}"
        else:
            nav_text = "Nav:        no route"
        self.goal_label.setText(nav_text)

        self._clear_memory()
        if not agent.memory_entries:
            lbl = QLabel("  (nothing yet)")
            lbl.setStyleSheet("color:#555; font-size:8pt;")
            self.memory_layout.addWidget(lbl)

        for m in sorted(agent.memory_entries, key=lambda x: -x.time_seen):
            age = sim_time - m.time_seen
            text = f"  [{m.kind:6s}]  ({m.position[0]:.0f},{m.position[1]:.0f})  {age:.1f}s ago"
            if m.kind == "exit":
                color = "#00ff66"
            elif m.kind == "hazard":
                color = "#ff6040"
            else:
                color = "#aaaaff"
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
#  Main window
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
        self.add_exit_mode = False
        self.add_hazard_mode = False
        self.next_exit_id = 1
        self.next_hazard_id = 1
        self._stats_skip = 0

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

        left = QFrame()
        left.setObjectName("card")
        left.setFixedWidth(220)
        lv = QVBoxLayout(left)
        lv.setContentsMargins(10, 10, 10, 10)
        lv.setSpacing(8)

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

        self.add_exit_btn = QPushButton("➕ Add Exit")
        self.add_exit_btn.setCheckable(True)
        self.add_exit_btn.setEnabled(False)
        self.add_exit_btn.clicked.connect(self._toggle_add_exit_mode)
        lv.addWidget(self.add_exit_btn)

        self.start_hazard_btn = QPushButton("🔥 Start Hazard")
        self.start_hazard_btn.setCheckable(True)
        self.start_hazard_btn.setEnabled(False)
        self.start_hazard_btn.clicked.connect(self._toggle_hazard_mode)
        lv.addWidget(self.start_hazard_btn)

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

        self.sim_view = SimView()
        self.inspector = InspectorPanel()

        rl.addWidget(left)
        rl.addWidget(self.sim_view, 1)
        rl.addWidget(self.inspector)

    def _sep(self):
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color:#0f3460;")
        return sep

    def _toggle_add_exit_mode(self):
        self.add_exit_mode = self.add_exit_btn.isChecked()
        if self.add_exit_mode:
            self.add_hazard_mode = False
            self.start_hazard_btn.setChecked(False)

    def _toggle_hazard_mode(self):
        self.add_hazard_mode = self.start_hazard_btn.isChecked()
        if self.add_hazard_mode:
            self.add_exit_mode = False
            self.add_exit_btn.setChecked(False)

    def handle_map_click(self, x: float, y: float) -> bool:
        if self.sim is None:
            return False
        if not self.sim.walk_map.is_walkable(x, y):
            return True

        if self.add_exit_mode:
            self.sim.exits[self.next_exit_id] = (x, y)
            self.sim.build_navigation_graph()
            self.next_exit_id += 1
            self.add_exit_mode = False
            self.add_exit_btn.setChecked(False)
            self.sim_view.update()
            return True

        if self.add_hazard_mode:
            self.sim.trigger_hazard(self.next_hazard_id, (x, y))
            self.next_hazard_id += 1
            self.add_hazard_mode = False
            self.start_hazard_btn.setChecked(False)
            self.sim_view.update()
            return True

        return False

    def load_config(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Zone Config JSON", "", "JSON (*.json)")
        if not path:
            return

        with open(path) as f:
            self.zone_config = json.load(f)

        mask_path = self.zone_config.get("mask_path", "")
        if not Path(mask_path).exists():
            mask_path, _ = QFileDialog.getOpenFileName(self, "Locate Mask Image", "", "Images (*.png *.jpg *.bmp)")
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
        self.add_exit_btn.setEnabled(True)
        self.start_hazard_btn.setEnabled(True)
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

        self._stats_skip += 1
        if self._stats_skip >= 5:
            self._update_stats()
            self._stats_skip = 0

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
            f"Exits:    {len(self.sim.exits)}\n"
            f"Hazards:  {len(self.sim.hazards)}\n"
            f"Explored: {explored} cells total\n\n"
            f"Click agent to inspect.\n"
            f"Use Add Exit / Start Hazard to place markers."
        )
        if not self.sim_view.selected_agent:
            self.inspector.update_agent(None, t)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Space:
            self.toggle_pause()
        elif event.key() == Qt.Key.Key_R:
            self.reset_sim()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
