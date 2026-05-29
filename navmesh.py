"""
navmesh.py — High-resolution occupancy-grid navigation for TRAGIC

Why this version exists:
- your input is a 2D binary floorplan mask
- a dense occupancy grid fits that input better than a fake polygon graph
- it covers narrow gaps and room interiors much more faithfully
- it is easy to understand, tune, and later replace with Recast if needed

Core idea:
- downsample the walkable mask into small square cells
- each cell becomes either walkable or blocked
- pathfinding runs on those cells (8-connected A*)
- the resulting path is smoothed with line-of-sight checks
- debug drawing shows the actual navigable area, not just sparse points
"""

import json
import math
import heapq
import numpy as np
import cv2
from typing import Tuple, List, Optional, Dict, Set


# ---------------------------------------------------------------------
# Tunable defaults
# ---------------------------------------------------------------------
CELL_SIZE = 4                 # pixels per nav cell; smaller = more precise
CLEARANCE = 1                 # erosion in pixels before grid generation
WALKABLE_THRESHOLD = 0.60     # fraction of open pixels required inside a cell
DIAGONAL = True               # allow 8-connected movement


class NavMesh:
    def __init__(self, walk_map,
                 cell_size: int = CELL_SIZE,
                 clearance: int = CLEARANCE,
                 walkable_threshold: float = WALKABLE_THRESHOLD,
                 diagonal: bool = DIAGONAL):
        self.walk_map = walk_map
        self.cell_size = int(cell_size)
        self.clearance = int(clearance)
        self.walkable_threshold = float(walkable_threshold)
        self.diagonal = bool(diagonal)

        self.grid: Optional[np.ndarray] = None      # True = walkable
        self.gh = 0
        self.gw = 0
        self._built = False

    # -----------------------------------------------------------------
    # Build
    # -----------------------------------------------------------------
    def build(self):
        print("[NavGrid] Building occupancy-grid navigation...")

        walk = self.walk_map.walkable.astype(np.uint8) * 255

        if self.clearance > 0:
            k = self.clearance * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            walk = cv2.erode(walk, kernel)

        h, w = walk.shape
        self.gh = math.ceil(h / self.cell_size)
        self.gw = math.ceil(w / self.cell_size)
        self.grid = np.zeros((self.gh, self.gw), dtype=bool)

        for gy in range(self.gh):
            for gx in range(self.gw):
                y0 = gy * self.cell_size
                y1 = min(h, y0 + self.cell_size)
                x0 = gx * self.cell_size
                x1 = min(w, x0 + self.cell_size)

                patch = walk[y0:y1, x0:x1]
                if patch.size == 0:
                    continue

                open_ratio = np.count_nonzero(patch) / patch.size
                self.grid[gy, gx] = open_ratio >= self.walkable_threshold

        self._built = True
        print(f"[NavGrid] Done. Grid: {self.gw}x{self.gh}")

    # -----------------------------------------------------------------
    # Coordinate conversion
    # -----------------------------------------------------------------
    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        gx = int(np.clip(x // self.cell_size, 0, self.gw - 1))
        gy = int(np.clip(y // self.cell_size, 0, self.gh - 1))
        return gx, gy

    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        x = gx * self.cell_size + self.cell_size / 2.0
        y = gy * self.cell_size + self.cell_size / 2.0
        return x, y

    # -----------------------------------------------------------------
    # Queries
    # -----------------------------------------------------------------
    def is_walkable_world(self, x: float, y: float) -> bool:
        if not self._built or self.grid is None:
            return False
        gx, gy = self.world_to_grid(x, y)
        return bool(self.grid[gy, gx])

    def nearest_walkable(self, x: float, y: float, max_radius_cells: int = 20) -> Optional[Tuple[int, int]]:
        if not self._built or self.grid is None:
            return None

        sx, sy = self.world_to_grid(x, y)
        if self.grid[sy, sx]:
            return sx, sy

        for r in range(1, max_radius_cells + 1):
            x0 = max(0, sx - r)
            x1 = min(self.gw - 1, sx + r)
            y0 = max(0, sy - r)
            y1 = min(self.gh - 1, sy + r)

            for gy in range(y0, y1 + 1):
                for gx in range(x0, x1 + 1):
                    if self.grid[gy, gx]:
                        return gx, gy
        return None

    # -----------------------------------------------------------------
    # Pathfinding
    # -----------------------------------------------------------------
    def find_path(self, start: Tuple[float, float], goal: Tuple[float, float]) -> List[Tuple[float, float]]:
        if not self._built or self.grid is None:
            return []

        s = self.nearest_walkable(start[0], start[1])
        g = self.nearest_walkable(goal[0], goal[1])
        if s is None or g is None:
            return []

        if s == g:
            return [goal]

        path_cells = self._astar(s, g)
        if not path_cells:
            return []

        path_points = [self.grid_to_world(gx, gy) for gx, gy in path_cells]
        path_points = self._smooth_path(path_points)

        if path_points:
            path_points[-1] = goal
        return path_points

    def _astar(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        open_heap = []
        heapq.heappush(open_heap, (0.0, start))

        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], float] = {start: 0.0}
        visited: Set[Tuple[int, int]] = set()

        while open_heap:
            _, current = heapq.heappop(open_heap)
            if current in visited:
                continue
            visited.add(current)

            if current == goal:
                return self._reconstruct_cells(came_from, current)

            cx, cy = current
            for nx, ny, cost in self._neighbors(cx, cy):
                tentative = g_score[current] + cost
                nxt = (nx, ny)
                if tentative < g_score.get(nxt, float("inf")):
                    came_from[nxt] = current
                    g_score[nxt] = tentative
                    f = tentative + self._heuristic(nx, ny, goal[0], goal[1])
                    heapq.heappush(open_heap, (f, nxt))

        return []

    def _neighbors(self, gx: int, gy: int):
        directions = [
            (-1, 0, 1.0), (1, 0, 1.0),
            (0, -1, 1.0), (0, 1, 1.0),
        ]
        if self.diagonal:
            d = math.sqrt(2)
            directions += [
                (-1, -1, d), (1, -1, d),
                (-1, 1, d), (1, 1, d),
            ]

        for dx, dy, cost in directions:
            nx, ny = gx + dx, gy + dy
            if nx < 0 or ny < 0 or nx >= self.gw or ny >= self.gh:
                continue
            if not self.grid[ny, nx]:
                continue

            # prevent diagonal corner cutting through tight walls
            if dx != 0 and dy != 0:
                if not self.grid[gy, nx] or not self.grid[ny, gx]:
                    continue

            yield nx, ny, cost

    def _heuristic(self, x0: int, y0: int, x1: int, y1: int) -> float:
        return math.hypot(x1 - x0, y1 - y0)

    def _reconstruct_cells(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    # -----------------------------------------------------------------
    # Smoothing
    # -----------------------------------------------------------------
    def _smooth_path(self, points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        if len(points) <= 2:
            return points

        out = [points[0]]
        i = 0
        while i < len(points) - 1:
            j = len(points) - 1
            while j > i + 1:
                if self._has_los_world(points[i], points[j]):
                    break
                j -= 1
            out.append(points[j])
            i = j
        return out

    def _has_los_world(self, a: Tuple[float, float], b: Tuple[float, float]) -> bool:
        x0, y0 = a
        x1, y1 = b
        dist = max(1.0, math.hypot(x1 - x0, y1 - y0))
        steps = max(4, int(dist / max(1, self.cell_size / 2)))
        for i in range(1, steps):
            t = i / steps
            x = x0 + (x1 - x0) * t
            y = y0 + (y1 - y0) * t
            if not self.walk_map.is_walkable(x, y):
                return False
        return True

    # -----------------------------------------------------------------
    # Save / Load
    # -----------------------------------------------------------------
    def save(self, path: str):
        data = {
            "cell_size": self.cell_size,
            "clearance": self.clearance,
            "walkable_threshold": self.walkable_threshold,
            "diagonal": self.diagonal,
            "grid": self.grid.astype(np.uint8).tolist() if self.grid is not None else [],
        }
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"[NavGrid] Saved to {path}")

    def load(self, path: str):
        with open(path) as f:
            data = json.load(f)
        self.cell_size = int(data.get("cell_size", CELL_SIZE))
        self.clearance = int(data.get("clearance", CLEARANCE))
        self.walkable_threshold = float(data.get("walkable_threshold", WALKABLE_THRESHOLD))
        self.diagonal = bool(data.get("diagonal", DIAGONAL))
        self.grid = np.array(data["grid"], dtype=np.uint8).astype(bool)
        self.gh, self.gw = self.grid.shape
        self._built = True
        print(f"[NavGrid] Loaded {self.gw}x{self.gh} from {path}")

    # -----------------------------------------------------------------
    # Debug drawing
    # -----------------------------------------------------------------
    def draw_debug(self, painter, scale: float, ox: float, oy: float,
                   show_fill: bool = True,
                   show_grid_lines: bool = False):
        from PyQt6.QtGui import QColor, QBrush, QPen
        from PyQt6.QtCore import Qt

        if not self._built or self.grid is None:
            return

        cell = self.cell_size * scale
        if cell <= 0:
            return

        if show_fill:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor(0, 220, 255, 55)))
            for gy in range(self.gh):
                for gx in range(self.gw):
                    if not self.grid[gy, gx]:
                        continue
                    x = ox + gx * self.cell_size * scale
                    y = oy + gy * self.cell_size * scale
                    painter.drawRect(int(x), int(y), max(1, int(cell)), max(1, int(cell)))

        if show_grid_lines and cell >= 4:
            painter.setPen(QPen(QColor(0, 220, 255, 35), 1))
            for gy in range(self.gh):
                for gx in range(self.gw):
                    if not self.grid[gy, gx]:
                        continue
                    x = ox + gx * self.cell_size * scale
                    y = oy + gy * self.cell_size * scale
                    painter.drawRect(int(x), int(y), max(1, int(cell)), max(1, int(cell)))

    def draw_path(self, painter, path: List[Tuple[float, float]], scale: float, ox: float, oy: float):
        from PyQt6.QtGui import QPen, QColor
        from PyQt6.QtCore import QPointF

        if len(path) < 2:
            return
        painter.setPen(QPen(QColor(255, 220, 0, 220), 2))
        for i in range(len(path) - 1):
            x0 = path[i][0] * scale + ox
            y0 = path[i][1] * scale + oy
            x1 = path[i + 1][0] * scale + ox
            y1 = path[i + 1][1] * scale + oy
            painter.drawLine(QPointF(x0, y0), QPointF(x1, y1))