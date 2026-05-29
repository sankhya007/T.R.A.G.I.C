# import cv2
# import numpy as np
# import random

# # ========================== CONFIG ====================================
# MASK_PATH      = "stitched_mask.png"   # output from predict_tiled
# OUTPUT_PATH    = "zone_map.png"

# MIN_ZONE_AREA  = 500      # pixels — blobs smaller than this are noise, skip them
# AGENT_DENSITY  = 0.005    # agents per pixel  (tune this up/down freely)

# # density_index per zone — default is 1.0 for every zone.
# # After you run this once and see the zone IDs printed, you can override
# # specific zones here like:  DENSITY_OVERRIDES = {3: 2.0, 7: 0.5}
# DENSITY_OVERRIDES = {}
# # ======================================================================

# # ---------- load + invert ----------
# mask = cv2.imread(MASK_PATH, cv2.IMREAD_GRAYSCALE)
# assert mask is not None, f"Could not load {MASK_PATH}"

# # white=walls → invert so walkable = 255, walls = 0
# walkable = cv2.bitwise_not(mask)

# # ---------- connected components ----------
# num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
#     walkable, connectivity=8
# )
# # label 0 is always background (walls) — skip it

# print(f"\nFound {num_labels - 1} raw blobs")

# # ---------- filter noise, collect zones ----------
# zones = []   # list of dicts
# for label_id in range(1, num_labels):
#     area = int(stats[label_id, cv2.CC_STAT_AREA])
#     if area < MIN_ZONE_AREA:
#         continue
#     density_index = DENSITY_OVERRIDES.get(label_id, 1.0)
#     zones.append({
#         "id":            label_id,
#         "area":          area,
#         "density_index": density_index,
#         "centroid":      centroids[label_id],
#     })

# print(f"Kept {len(zones)} zones after filtering (min area = {MIN_ZONE_AREA} px)\n")

# # ---------- assign a distinct colour per zone ----------
# rng = random.Random(42)
# def rand_colour():
#     # pastel-ish, stays readable on dark background
#     return (rng.randint(60, 220), rng.randint(60, 220), rng.randint(60, 220))

# zone_colours = {z["id"]: rand_colour() for z in zones}

# # ---------- build output image ----------
# H, W = mask.shape
# viz = np.zeros((H, W, 3), dtype=np.uint8)

# # draw walls in dark grey so the floor plan is still visible
# viz[mask > 127] = (50, 50, 50)

# # fill each zone with its colour
# for z in zones:
#     lid = z["id"]
#     viz[labels == lid] = zone_colours[lid]

# # ---------- spawn agents ----------
# total_agents = 0

# for z in zones:
#     lid    = z["id"]
#     di     = z["density_index"]
#     n      = max(1, int(z["area"] * AGENT_DENSITY * di))

#     # get all pixel coords inside this zone
#     ys, xs = np.where(labels == lid)
#     if len(xs) == 0:
#         continue

#     # pick n random ones
#     indices = np.random.choice(len(xs), size=min(n, len(xs)), replace=False)
#     agent_xs = xs[indices]
#     agent_ys = ys[indices]

#     # draw dots
#     for ax, ay in zip(agent_xs, agent_ys):
#         cv2.circle(viz, (int(ax), int(ay)), 3, (255, 255, 255), -1)

#     total_agents += n
#     cx, cy = int(z["centroid"][0]), int(z["centroid"][1])
#     label_text = f"Z{lid} d={di:.1f}"
#     cv2.putText(viz, label_text, (cx - 20, cy),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

#     print(f"  Zone {lid:>3}  area={z['area']:>6}px  density={di:.1f}  agents={n}")

# print(f"\nTotal agents spawned: {total_agents}")

# # ---------- save ----------
# cv2.imwrite(OUTPUT_PATH, viz)
# print(f"Saved: {OUTPUT_PATH}\n")









# this one works but what i want is to have the control to set the number of the dencity threshold by myself which is not gonna happen in this but this is the best ever zone selection that i have ever gotten tell now



# """
# zone_detector.py
# Step 2: Spawnable area detection, room segmentation, density-based agent spawning

# Input : stitched binary mask (white=walls, black=walkable)
# Output: zone_map.png  — colored zones + spawned agent dots

# Room splitting strategy:
#   Distance transform on walkable mask → peaks = room centers
#   Watershed from those peaks → separates rooms even when connected via corridors
# """

# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# from scipy import ndimage as ndi
# from skimage.segmentation import watershed
# from skimage.feature import peak_local_max
# import sys
# from pathlib import Path


# # ── tuneable constants ────────────────────────────────────────────────
# MIN_ZONE_AREA      = 800    # px²  — ignore zones smaller than this
# DEFAULT_DENSITY    = 1.0    # agents per 1000 px²
# AGENT_SCALE        = 1000   # px² per density unit
# PEAK_MIN_DISTANCE  = 40     # px  — min distance between room centers (tune per image)
# # ──────────────────────────────────────────────────────────────────────


# def load_mask(path: str) -> np.ndarray:
#     """Load image, return clean binary walkable mask (255=walkable, 0=wall)."""
#     img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#     if img is None:
#         raise FileNotFoundError(f"Cannot load: {path}")

#     # white=wall → invert → walkable=255
#     walkable = cv2.bitwise_not(img)
#     _, binary = cv2.threshold(walkable, 127, 255, cv2.THRESH_BINARY)

#     # small close to seal hairline cracks in walls
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
#     binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

#     return binary


# def segment_rooms(binary: np.ndarray) -> np.ndarray:
#     """
#     Use distance transform + watershed to split connected rooms.
#     Returns a label map (0=wall, 1..N=zone ids).
#     """
#     # Distance transform: each walkable pixel gets its distance to the nearest wall
#     dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

#     # Normalize for peak detection
#     dist_norm = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)

#     # Find local maxima (room centers) — these become watershed seeds
#     coords = peak_local_max(
#         dist_norm,
#         min_distance=PEAK_MIN_DISTANCE,
#         labels=binary
#     )

#     # Create seed mask
#     seed_mask = np.zeros(dist_norm.shape, dtype=bool)
#     seed_mask[tuple(coords.T)] = True
#     markers, _ = ndi.label(seed_mask)

#     # Watershed: flood from seeds, split at narrow passages (corridors)
#     labels = watershed(-dist, markers, mask=binary)

#     return labels


# def find_zones(labels: np.ndarray) -> tuple:
#     """Filter out tiny zones, return valid zone ids + their stats."""
#     unique_ids = np.unique(labels)
#     unique_ids = unique_ids[unique_ids > 0]  # skip 0 (wall)

#     valid_zones = []
#     stats = {}
#     for zid in unique_ids:
#         area = int(np.sum(labels == zid))
#         if area >= MIN_ZONE_AREA:
#             valid_zones.append(zid)
#             stats[zid] = area

#     print(f"Found {len(valid_zones)} rooms/zones (min area={MIN_ZONE_AREA}px²)")
#     return valid_zones, stats


# def spawn_agents(labels: np.ndarray, valid_zones: list,
#                  zone_stats: dict, density_overrides: dict = None) -> dict:
#     """
#     Spawn agents per zone based on density_index.
#     density_overrides: {zone_list_index: density_value}
#     """
#     if density_overrides is None:
#         density_overrides = {}

#     zone_data = {}
#     for i, zid in enumerate(valid_zones):
#         area    = zone_stats[zid]
#         density = density_overrides.get(i, DEFAULT_DENSITY)
#         n_agents = max(1, int(area * density / AGENT_SCALE))

#         ys, xs = np.where(labels == zid)
#         indices = np.random.choice(len(xs), size=min(n_agents, len(xs)), replace=False)

#         zone_data[i] = {
#             "zone_id"  : zid,
#             "area_px"  : area,
#             "density"  : density,
#             "n_agents" : len(indices),
#             "agents"   : list(zip(xs[indices].tolist(), ys[indices].tolist()))
#         }
#         print(f"  Zone {i:2d} | area={area:7d}px² | density={density:.1f} | agents={len(indices)}")

#     return zone_data


# def visualize(binary: np.ndarray, labels: np.ndarray,
#               valid_zones: list, zone_data: dict, out_path: str):
#     """Colored zone map with agent dots."""

#     h, w = binary.shape
#     rgb = np.zeros((h, w, 3), dtype=np.uint8)

#     # Distinct colors via HSV
#     np.random.seed(42)
#     colors = {}
#     for i, zid in enumerate(valid_zones):
#         hue = int(179 * i / max(len(valid_zones), 1))
#         hsv_px = np.uint8([[[hue, 190, 170]]])
#         bgr = cv2.cvtColor(hsv_px, cv2.COLOR_HSV2BGR)[0][0]
#         colors[i] = tuple(int(c) for c in bgr)

#     # Paint zones
#     for i, zid in enumerate(valid_zones):
#         rgb[labels == zid] = colors[i]

#     # Walls — dark
#     rgb[binary == 0] = (30, 30, 30)

#     rgb_plot = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

#     fig, ax = plt.subplots(figsize=(14, 10))
#     ax.imshow(rgb_plot, origin="upper")
#     ax.axis("off")
#     ax.set_title("Zone Map — Rooms & Agent Spawning", fontsize=14, fontweight="bold")

#     # Agent dots
#     for i, data in zone_data.items():
#         xs = [p[0] for p in data["agents"]]
#         ys = [p[1] for p in data["agents"]]
#         ax.scatter(xs, ys, s=5, c=[[1.0, 1.0, 0.15]], linewidths=0, zorder=5)

#     # Legend
#     patches = []
#     for i, data in zone_data.items():
#         c = tuple(v / 255.0 for v in colors[i][::-1])
#         patches.append(mpatches.Patch(color=c,
#             label=f"Zone {i} | {data['n_agents']} agents | d={data['density']:.1f}"))

#     ax.legend(handles=patches, loc="upper right", fontsize=7,
#               framealpha=0.85, ncol=max(1, len(patches) // 20))

#     plt.tight_layout()
#     plt.savefig(out_path, dpi=150, bbox_inches="tight")
#     plt.close()
#     print(f"\nSaved → {out_path}")


# if __name__ == "__main__":
#     if len(sys.argv) < 2:
#         print("Usage: python zone_detector.py <mask.png> [zone_index:density ...]")
#         print("Example: python zone_detector.py stitched_mask.png 0:2.0 3:0.5")
#         sys.exit(1)

#     mask_path = sys.argv[1]

#     density_overrides = {}
#     for arg in sys.argv[2:]:
#         idx, val = arg.split(":")
#         density_overrides[int(idx)] = float(val)

#     print(f"Loading: {mask_path}")
#     binary = load_mask(mask_path)

#     print("Segmenting rooms via distance transform + watershed...")
#     labels = segment_rooms(binary)

#     valid_zones, zone_stats = find_zones(labels)

#     print(f"\nSpawning agents (default density={DEFAULT_DENSITY}):")
#     zone_data = spawn_agents(labels, valid_zones, zone_stats, density_overrides)

#     total = sum(d["n_agents"] for d in zone_data.values())
#     print(f"\nTotal agents: {total}")

#     out_path = str(Path(mask_path).stem) + "_zone_map.png"
#     visualize(binary, labels, valid_zones, zone_data, out_path)










# holy fucking shit it works


"""
zone_editor.py  —  Step 2: Interactive Zone Density Editor
Run: python zone_editor.py

Workflow:
  1. Click "Load Mask" → select your stitched binary mask
  2. Zones are auto-detected and colored
  3. Click any zone on the map → type its density index (0 = outside/ignore)
  4. Set "Base agents per 1000px²" (what density index 1 equals)
  5. Click "Save Config" → writes zone_config.json for the simulation
"""

import sys
import json
import numpy as np
import cv2
from pathlib import Path

from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QLabel, QFileDialog, QInputDialog, QDoubleSpinBox,
    QScrollArea, QFrame, QSizePolicy, QMessageBox, QSpinBox
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPixmap, QImage, QPainter, QColor, QFont, QPen

# ── constants ──────────────────────────────────────────────────────────
MIN_ZONE_AREA     = 800    # px² — ignore blobs smaller than this
PEAK_MIN_DISTANCE = 40     # px  — tune if too many/few zones detected
AGENT_SCALE       = 1000   # px² per density unit
# ──────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════
#  Core processing (same logic as zone_detector.py)
# ══════════════════════════════════════════════════════════════════════

def load_and_segment(path: str):
    """Load mask → return (binary, label_map, valid_zones, zone_stats)."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)

    walkable = cv2.bitwise_not(img)
    _, binary = cv2.threshold(walkable, 127, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Distance transform + watershed
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    dist_norm = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)
    coords = peak_local_max(dist_norm, min_distance=PEAK_MIN_DISTANCE, labels=binary)

    seed_mask = np.zeros(dist_norm.shape, dtype=bool)
    seed_mask[tuple(coords.T)] = True
    markers, _ = ndi.label(seed_mask)
    labels = watershed(-dist, markers, mask=binary)

    unique_ids = np.unique(labels)
    unique_ids = unique_ids[unique_ids > 0]

    valid_zones = []
    zone_stats  = {}
    for zid in unique_ids:
        area = int(np.sum(labels == zid))
        if area >= MIN_ZONE_AREA:
            valid_zones.append(int(zid))
            zone_stats[int(zid)] = area

    return binary, labels, valid_zones, zone_stats


def build_color_map(valid_zones):
    """Return {zone_id: (R,G,B)} with distinct colors."""
    np.random.seed(42)
    colors = {}
    for i, zid in enumerate(valid_zones):
        hue = int(179 * i / max(len(valid_zones), 1))
        hsv = np.uint8([[[hue, 190, 170]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        colors[zid] = (int(bgr[2]), int(bgr[1]), int(bgr[0]))  # BGR→RGB
    return colors


def render_zone_image(binary, labels, valid_zones, color_map,
                      density_map, highlight_id=None):
    """
    Return an RGBA QImage of the zone map.
    density_map: {zone_id: float}  — 0 = outside (hatched gray)
    highlight_id: zone_id to draw with bright border
    """
    h, w = binary.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    for zid in valid_zones:
        d = density_map.get(zid, 1.0)
        if d == 0:
            rgb[labels == zid] = (80, 80, 80)   # outside — dark gray
        else:
            rgb[labels == zid] = color_map[zid]

    rgb[binary == 0] = (20, 20, 20)   # walls

    # Highlight selected zone with bright border
    if highlight_id is not None and highlight_id in valid_zones:
        zone_mask = (labels == highlight_id).astype(np.uint8) * 255
        contours, _ = cv2.findContours(zone_mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(rgb, contours, -1, (255, 255, 0), 3)

    # Draw zone index numbers at centroids
    for i, zid in enumerate(valid_zones):
        ys, xs = np.where(labels == zid)
        if len(xs) == 0:
            continue
        cx, cy = int(xs.mean()), int(ys.mean())
        d = density_map.get(zid, 1.0)
        text = f"{i}:{d:.1f}"
        cv2.putText(rgb, text, (cx - 15, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # Convert to QImage
    h, w, ch = rgb.shape
    qimg = QImage(rgb.tobytes(), w, h, w * ch, QImage.Format.Format_RGB888)
    return qimg


# ══════════════════════════════════════════════════════════════════════
#  Clickable Map Widget
# ══════════════════════════════════════════════════════════════════════

class ZoneMapWidget(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.labels      = None
        self.valid_zones = []
        self.scale_x     = 1.0
        self.scale_y     = 1.0
        self.on_click    = None   # callback(zone_id)
        self.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.setCursor(Qt.CursorShape.CrossCursor)

    def set_data(self, labels, valid_zones, orig_w, orig_h):
        self.labels      = labels
        self.valid_zones = valid_zones
        self.orig_w      = orig_w
        self.orig_h      = orig_h

    def update_scale(self):
        if self.pixmap():
            self.scale_x = self.orig_w / self.pixmap().width()
            self.scale_y = self.orig_h / self.pixmap().height()

    def mousePressEvent(self, event):
        if self.labels is None or self.on_click is None:
            return
        self.update_scale()
        px = int(event.position().x() * self.scale_x)
        py = int(event.position().y() * self.scale_y)
        px = np.clip(px, 0, self.labels.shape[1] - 1)
        py = np.clip(py, 0, self.labels.shape[0] - 1)
        zid = int(self.labels[py, px])
        if zid in self.valid_zones:
            self.on_click(zid)


# ══════════════════════════════════════════════════════════════════════
#  Main Window
# ══════════════════════════════════════════════════════════════════════

class ZoneEditor(QMainWindow):

    STYLE = """
    QMainWindow, QWidget { background:#1a1a2e; color:#e0e0e0;
        font-family:'Segoe UI',Arial,sans-serif; font-size:10pt; }
    QPushButton { background:#16213e; border:1px solid #0f3460;
        border-radius:5px; padding:7px 14px; color:#e0e0e0; }
    QPushButton:hover  { background:#0f3460; border-color:#e94560; }
    QPushButton:pressed{ background:#e94560; color:white; }
    QPushButton#primary{ background:#e94560; color:white; font-weight:bold; }
    QPushButton#primary:hover{ background:#ff6b6b; }
    QLabel#title { font-size:15pt; font-weight:bold; color:#e94560; }
    QLabel#info  { color:#aaa; font-size:9pt; }
    QFrame#card  { background:#16213e; border:1px solid #0f3460;
        border-radius:8px; }
    QDoubleSpinBox, QSpinBox { background:#0f3460; border:1px solid #e94560;
        border-radius:4px; padding:4px; color:white; }
    QScrollArea  { border:none; }
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("TRAGIC — Zone Density Editor")
        self.setMinimumSize(1200, 750)
        self.setStyleSheet(self.STYLE)

        # State
        self.mask_path   = None
        self.binary      = None
        self.labels      = None
        self.valid_zones = []
        self.zone_stats  = {}
        self.color_map   = {}
        self.density_map = {}      # {zone_id: float}
        self.highlight   = None

        self._build_ui()

    # ── UI construction ───────────────────────────────────────────────

    def _build_ui(self):
        root = QWidget()
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(15, 15, 15, 15)
        root_layout.setSpacing(15)
        self.setCentralWidget(root)

        # ── Left panel ──
        left = QFrame(); left.setObjectName("card")
        left.setFixedWidth(280)
        lv = QVBoxLayout(left)
        lv.setContentsMargins(12, 12, 12, 12)
        lv.setSpacing(10)

        title = QLabel("Zone Editor"); title.setObjectName("title")
        lv.addWidget(title)

        self.load_btn = QPushButton("📂  Load Mask")
        self.load_btn.clicked.connect(self.load_mask)
        lv.addWidget(self.load_btn)

        # Base density
        lv.addWidget(QLabel("Agents per 1000px²\nat density index = 1:"))
        self.base_spin = QDoubleSpinBox()
        self.base_spin.setRange(0.1, 20.0)
        self.base_spin.setValue(1.0)
        self.base_spin.setSingleStep(0.1)
        lv.addWidget(self.base_spin)

        lv.addWidget(self._sep())

        # Selected zone info
        self.zone_label = QLabel("Click a zone on the map")
        self.zone_label.setObjectName("info")
        self.zone_label.setWordWrap(True)
        lv.addWidget(self.zone_label)

        lv.addWidget(QLabel("Set density index:"))
        self.density_spin = QDoubleSpinBox()
        self.density_spin.setRange(0.0, 10.0)
        self.density_spin.setValue(1.0)
        self.density_spin.setSingleStep(0.5)
        self.density_spin.setToolTip("0 = outside / ignore")
        lv.addWidget(self.density_spin)

        self.apply_btn = QPushButton("Apply to Zone")
        self.apply_btn.setEnabled(False)
        self.apply_btn.clicked.connect(self.apply_density)
        lv.addWidget(self.apply_btn)

        lv.addWidget(self._sep())

        # Zone list scroll
        lv.addWidget(QLabel("All zones:"))
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.zone_list_widget = QWidget()
        self.zone_list_layout = QVBoxLayout(self.zone_list_widget)
        self.zone_list_layout.setSpacing(3)
        self.zone_list_layout.setContentsMargins(0, 0, 0, 0)
        scroll.setWidget(self.zone_list_widget)
        lv.addWidget(scroll, 1)

        lv.addWidget(self._sep())

        save_btn = QPushButton("💾  Save Config")
        save_btn.setObjectName("primary")
        save_btn.clicked.connect(self.save_config)
        lv.addWidget(save_btn)

        # ── Map area ──
        map_scroll = QScrollArea()
        map_scroll.setWidgetResizable(True)
        self.map_widget = ZoneMapWidget()
        self.map_widget.on_click = self.zone_clicked
        self.map_widget.setSizePolicy(QSizePolicy.Policy.Expanding,
                                       QSizePolicy.Policy.Expanding)
        map_scroll.setWidget(self.map_widget)

        root_layout.addWidget(left)
        root_layout.addWidget(map_scroll, 1)

    def _sep(self):
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("color:#0f3460;")
        return line

    # ── Actions ──────────────────────────────────────────────────────

    def load_mask(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Binary Mask", "",
            "Images (*.png *.jpg *.jpeg *.bmp)")
        if not path:
            return

        self.mask_path = path
        self.setWindowTitle(f"TRAGIC — Zone Editor  [{Path(path).name}]")

        try:
            self.binary, self.labels, self.valid_zones, self.zone_stats = \
                load_and_segment(path)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            return

        self.color_map   = build_color_map(self.valid_zones)
        self.density_map = {zid: 1.0 for zid in self.valid_zones}
        self.highlight   = None

        self._refresh_map()
        self._rebuild_zone_list()
        self.zone_label.setText(
            f"{len(self.valid_zones)} zones detected.\nClick a zone to set its density.")

    def zone_clicked(self, zid):
        self.highlight = zid
        idx = self.valid_zones.index(zid)
        area = self.zone_stats[zid]
        d    = self.density_map.get(zid, 1.0)
        agents = int(area * d * self.base_spin.value() / AGENT_SCALE)

        self.zone_label.setText(
            f"Zone {idx}  (id={zid})\n"
            f"Area: {area:,} px²\n"
            f"Current density: {d:.1f}\n"
            f"→ ~{agents} agents at base={self.base_spin.value():.1f}")

        self.density_spin.setValue(d)
        self.apply_btn.setEnabled(True)
        self._refresh_map()

    def apply_density(self):
        if self.highlight is None:
            return
        val = self.density_spin.value()
        self.density_map[self.highlight] = val
        self.zone_clicked(self.highlight)   # refresh info
        self._rebuild_zone_list()
        self._refresh_map()

    def save_config(self):
        if not self.mask_path:
            QMessageBox.warning(self, "No mask", "Load a mask first.")
            return

        out_path = str(Path(self.mask_path).with_suffix("")) + "_zone_config.json"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Zone Config", out_path, "JSON (*.json)")
        if not path:
            return

        config = {
            "mask_path"   : self.mask_path,
            "base_density": self.base_spin.value(),
            "agent_scale" : AGENT_SCALE,
            "zones"       : []
        }

        for i, zid in enumerate(self.valid_zones):
            area    = self.zone_stats[zid]
            d       = self.density_map.get(zid, 1.0)
            agents  = int(area * d * self.base_spin.value() / AGENT_SCALE)
            config["zones"].append({
                "zone_index"   : i,
                "zone_id"      : zid,
                "area_px"      : area,
                "density_index": d,
                "agents"       : agents
            })

        with open(path, "w") as f:
            json.dump(config, f, indent=2)

        total = sum(z["agents"] for z in config["zones"] if z["density_index"] > 0)
        QMessageBox.information(self, "Saved",
            f"Config saved to:\n{path}\n\nTotal agents: {total}")

    # ── Helpers ──────────────────────────────────────────────────────

    def _refresh_map(self):
        if self.binary is None:
            return
        qimg = render_zone_image(
            self.binary, self.labels, self.valid_zones,
            self.color_map, self.density_map, self.highlight)

        pix = QPixmap.fromImage(qimg)
        # Scale to fit nicely, keep aspect ratio
        max_w = self.width() - 320
        max_h = self.height() - 40
        pix = pix.scaled(max_w, max_h,
                         Qt.AspectRatioMode.KeepAspectRatio,
                         Qt.TransformationMode.SmoothTransformation)
        self.map_widget.setPixmap(pix)
        self.map_widget.set_data(self.labels, self.valid_zones,
                                  self.binary.shape[1], self.binary.shape[0])
        self.map_widget.resize(pix.width(), pix.height())

    def _rebuild_zone_list(self):
        # Clear old
        while self.zone_list_layout.count():
            item = self.zone_list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for i, zid in enumerate(self.valid_zones):
            d     = self.density_map.get(zid, 1.0)
            area  = self.zone_stats[zid]
            c     = self.color_map[zid]
            label = "outside" if d == 0 else f"d={d:.1f}"

            row = QLabel(f"  Zone {i:2d} | {label:8s} | {area//1000}k px²")
            row.setStyleSheet(
                f"background:rgb({c[0]},{c[1]},{c[2]});"
                f"color:{'#111' if sum(c)>400 else '#eee'};"
                f"border-radius:3px; padding:2px 4px;")
            row.setFixedHeight(22)
            self.zone_list_layout.addWidget(row)

        self.zone_list_layout.addStretch()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        QTimer.singleShot(50, self._refresh_map)


# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = ZoneEditor()
    win.show()
    sys.exit(app.exec())
