# """
# zone_editor.py  —  Step 2: Interactive Zone Density Editor
# Run: python zone_editor.py

# Workflow:
#   1. Click "Load Mask" → select your stitched binary mask
#   2. Zones are auto-detected and colored
#   3. Click any zone on the map → type its density index (0 = outside/ignore)
#   4. Set "Base agents per 1000px²" (what density index 1 equals)
#   5. Click "Save Config" → writes zone_config.json for the simulation
# """

# import sys
# import json
# import numpy as np
# import cv2
# from pathlib import Path

# from scipy import ndimage as ndi
# from skimage.segmentation import watershed
# from skimage.feature import peak_local_max

# from PyQt6.QtWidgets import (
#     QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
#     QPushButton, QLabel, QFileDialog, QInputDialog, QDoubleSpinBox,
#     QScrollArea, QFrame, QSizePolicy, QMessageBox, QSpinBox
# )
# from PyQt6.QtCore import Qt, QTimer
# from PyQt6.QtGui import QPixmap, QImage, QPainter, QColor, QFont, QPen

# # ── constants ──────────────────────────────────────────────────────────
# MIN_ZONE_AREA     = 800    # px² — ignore blobs smaller than this
# PEAK_MIN_DISTANCE = 40     # px  — tune if too many/few zones detected
# AGENT_SCALE       = 1000   # px² per density unit
# # ──────────────────────────────────────────────────────────────────────


# # ══════════════════════════════════════════════════════════════════════
# #  Core processing (same logic as zone_detector.py)
# # ══════════════════════════════════════════════════════════════════════

# def load_and_segment(path: str):
#     """Load mask → return (binary, label_map, valid_zones, zone_stats)."""
#     img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#     if img is None:
#         raise FileNotFoundError(path)

#     walkable = cv2.bitwise_not(img)
#     _, binary = cv2.threshold(walkable, 127, 255, cv2.THRESH_BINARY)
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
#     binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

#     # Distance transform + watershed
#     dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
#     dist_norm = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)
#     coords = peak_local_max(dist_norm, min_distance=PEAK_MIN_DISTANCE, labels=binary)

#     seed_mask = np.zeros(dist_norm.shape, dtype=bool)
#     seed_mask[tuple(coords.T)] = True
#     markers, _ = ndi.label(seed_mask)
#     labels = watershed(-dist, markers, mask=binary)

#     unique_ids = np.unique(labels)
#     unique_ids = unique_ids[unique_ids > 0]

#     valid_zones = []
#     zone_stats  = {}
#     for zid in unique_ids:
#         area = int(np.sum(labels == zid))
#         if area >= MIN_ZONE_AREA:
#             valid_zones.append(int(zid))
#             zone_stats[int(zid)] = area

#     return binary, labels, valid_zones, zone_stats


# def build_color_map(valid_zones):
#     """Return {zone_id: (R,G,B)} with distinct colors."""
#     np.random.seed(42)
#     colors = {}
#     for i, zid in enumerate(valid_zones):
#         hue = int(179 * i / max(len(valid_zones), 1))
#         hsv = np.uint8([[[hue, 190, 170]]])
#         bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
#         colors[zid] = (int(bgr[2]), int(bgr[1]), int(bgr[0]))  # BGR→RGB
#     return colors


# def render_zone_image(binary, labels, valid_zones, color_map,
#                       density_map, highlight_id=None):
#     """
#     Return an RGBA QImage of the zone map.
#     density_map: {zone_id: float}  — 0 = outside (hatched gray)
#     highlight_id: zone_id to draw with bright border
#     """
#     h, w = binary.shape
#     rgb = np.zeros((h, w, 3), dtype=np.uint8)

#     for zid in valid_zones:
#         d = density_map.get(zid, 1.0)
#         if d == 0:
#             rgb[labels == zid] = (80, 80, 80)   # outside — dark gray
#         else:
#             rgb[labels == zid] = color_map[zid]

#     rgb[binary == 0] = (20, 20, 20)   # walls

#     # Highlight selected zone with bright border
#     if highlight_id is not None and highlight_id in valid_zones:
#         zone_mask = (labels == highlight_id).astype(np.uint8) * 255
#         contours, _ = cv2.findContours(zone_mask, cv2.RETR_EXTERNAL,
#                                         cv2.CHAIN_APPROX_SIMPLE)
#         cv2.drawContours(rgb, contours, -1, (255, 255, 0), 3)

#     # Draw zone index numbers at centroids
#     for i, zid in enumerate(valid_zones):
#         ys, xs = np.where(labels == zid)
#         if len(xs) == 0:
#             continue
#         cx, cy = int(xs.mean()), int(ys.mean())
#         d = density_map.get(zid, 1.0)
#         text = f"{i}:{d:.1f}"
#         cv2.putText(rgb, text, (cx - 15, cy + 5),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

#     # Convert to QImage
#     h, w, ch = rgb.shape
#     qimg = QImage(rgb.tobytes(), w, h, w * ch, QImage.Format.Format_RGB888)
#     return qimg


# # ══════════════════════════════════════════════════════════════════════
# #  Clickable Map Widget
# # ══════════════════════════════════════════════════════════════════════

# class ZoneMapWidget(QLabel):
#     def __init__(self, parent=None):
#         super().__init__(parent)
#         self.labels      = None
#         self.valid_zones = []
#         self.scale_x     = 1.0
#         self.scale_y     = 1.0
#         self.on_click    = None   # callback(zone_id)
#         self.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
#         self.setCursor(Qt.CursorShape.CrossCursor)

#     def set_data(self, labels, valid_zones, orig_w, orig_h):
#         self.labels      = labels
#         self.valid_zones = valid_zones
#         self.orig_w      = orig_w
#         self.orig_h      = orig_h

#     def update_scale(self):
#         if self.pixmap():
#             self.scale_x = self.orig_w / self.pixmap().width()
#             self.scale_y = self.orig_h / self.pixmap().height()

#     def mousePressEvent(self, event):
#         if self.labels is None or self.on_click is None:
#             return
#         self.update_scale()
#         px = int(event.position().x() * self.scale_x)
#         py = int(event.position().y() * self.scale_y)
#         px = np.clip(px, 0, self.labels.shape[1] - 1)
#         py = np.clip(py, 0, self.labels.shape[0] - 1)
#         zid = int(self.labels[py, px])
#         if zid in self.valid_zones:
#             self.on_click(zid)


# # ══════════════════════════════════════════════════════════════════════
# #  Main Window
# # ══════════════════════════════════════════════════════════════════════

# class ZoneEditor(QMainWindow):

#     STYLE = """
#     QMainWindow, QWidget { background:#1a1a2e; color:#e0e0e0;
#         font-family:'Segoe UI',Arial,sans-serif; font-size:10pt; }
#     QPushButton { background:#16213e; border:1px solid #0f3460;
#         border-radius:5px; padding:7px 14px; color:#e0e0e0; }
#     QPushButton:hover  { background:#0f3460; border-color:#e94560; }
#     QPushButton:pressed{ background:#e94560; color:white; }
#     QPushButton#primary{ background:#e94560; color:white; font-weight:bold; }
#     QPushButton#primary:hover{ background:#ff6b6b; }
#     QLabel#title { font-size:15pt; font-weight:bold; color:#e94560; }
#     QLabel#info  { color:#aaa; font-size:9pt; }
#     QFrame#card  { background:#16213e; border:1px solid #0f3460;
#         border-radius:8px; }
#     QDoubleSpinBox, QSpinBox { background:#0f3460; border:1px solid #e94560;
#         border-radius:4px; padding:4px; color:white; }
#     QScrollArea  { border:none; }
#     """

#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("TRAGIC — Zone Density Editor")
#         self.setMinimumSize(1200, 750)
#         self.setStyleSheet(self.STYLE)

#         # State
#         self.mask_path   = None
#         self.binary      = None
#         self.labels      = None
#         self.valid_zones = []
#         self.zone_stats  = {}
#         self.color_map   = {}
#         self.density_map = {}      # {zone_id: float}
#         self.highlight   = None

#         self._build_ui()

#     # ── UI construction ───────────────────────────────────────────────

#     def _build_ui(self):
#         root = QWidget()
#         root_layout = QHBoxLayout(root)
#         root_layout.setContentsMargins(15, 15, 15, 15)
#         root_layout.setSpacing(15)
#         self.setCentralWidget(root)

#         # ── Left panel ──
#         left = QFrame(); left.setObjectName("card")
#         left.setFixedWidth(280)
#         lv = QVBoxLayout(left)
#         lv.setContentsMargins(12, 12, 12, 12)
#         lv.setSpacing(10)

#         title = QLabel("Zone Editor"); title.setObjectName("title")
#         lv.addWidget(title)

#         self.load_btn = QPushButton("📂  Load Mask")
#         self.load_btn.clicked.connect(self.load_mask)
#         lv.addWidget(self.load_btn)

#         # Base density
#         lv.addWidget(QLabel("Agents per 1000px²\nat density index = 1:"))
#         self.base_spin = QDoubleSpinBox()
#         self.base_spin.setRange(0.1, 20.0)
#         self.base_spin.setValue(1.0)
#         self.base_spin.setSingleStep(0.1)
#         lv.addWidget(self.base_spin)

#         lv.addWidget(self._sep())

#         # Selected zone info
#         self.zone_label = QLabel("Click a zone on the map")
#         self.zone_label.setObjectName("info")
#         self.zone_label.setWordWrap(True)
#         lv.addWidget(self.zone_label)

#         lv.addWidget(QLabel("Set density index:"))
#         self.density_spin = QDoubleSpinBox()
#         self.density_spin.setRange(0.0, 10.0)
#         self.density_spin.setValue(1.0)
#         self.density_spin.setSingleStep(0.5)
#         self.density_spin.setToolTip("0 = outside / ignore")
#         lv.addWidget(self.density_spin)

#         self.apply_btn = QPushButton("Apply to Zone")
#         self.apply_btn.setEnabled(False)
#         self.apply_btn.clicked.connect(self.apply_density)
#         lv.addWidget(self.apply_btn)

#         lv.addWidget(self._sep())

#         # Zone list scroll
#         lv.addWidget(QLabel("All zones:"))
#         scroll = QScrollArea()
#         scroll.setWidgetResizable(True)
#         self.zone_list_widget = QWidget()
#         self.zone_list_layout = QVBoxLayout(self.zone_list_widget)
#         self.zone_list_layout.setSpacing(3)
#         self.zone_list_layout.setContentsMargins(0, 0, 0, 0)
#         scroll.setWidget(self.zone_list_widget)
#         lv.addWidget(scroll, 1)

#         lv.addWidget(self._sep())

#         save_btn = QPushButton("💾  Save Config")
#         save_btn.setObjectName("primary")
#         save_btn.clicked.connect(self.save_config)
#         lv.addWidget(save_btn)

#         # ── Map area ──
#         map_scroll = QScrollArea()
#         map_scroll.setWidgetResizable(True)
#         self.map_widget = ZoneMapWidget()
#         self.map_widget.on_click = self.zone_clicked
#         self.map_widget.setSizePolicy(QSizePolicy.Policy.Expanding,
#                                        QSizePolicy.Policy.Expanding)
#         map_scroll.setWidget(self.map_widget)

#         root_layout.addWidget(left)
#         root_layout.addWidget(map_scroll, 1)

#     def _sep(self):
#         line = QFrame()
#         line.setFrameShape(QFrame.Shape.HLine)
#         line.setStyleSheet("color:#0f3460;")
#         return line

#     # ── Actions ──────────────────────────────────────────────────────

#     def load_mask(self):
#         path, _ = QFileDialog.getOpenFileName(
#             self, "Select Binary Mask", "",
#             "Images (*.png *.jpg *.jpeg *.bmp)")
#         if not path:
#             return

#         self.mask_path = path
#         self.setWindowTitle(f"TRAGIC — Zone Editor  [{Path(path).name}]")

#         try:
#             self.binary, self.labels, self.valid_zones, self.zone_stats = \
#                 load_and_segment(path)
#         except Exception as e:
#             QMessageBox.critical(self, "Error", str(e))
#             return

#         self.color_map   = build_color_map(self.valid_zones)
#         self.density_map = {zid: 1.0 for zid in self.valid_zones}
#         self.highlight   = None

#         self._refresh_map()
#         self._rebuild_zone_list()
#         self.zone_label.setText(
#             f"{len(self.valid_zones)} zones detected.\nClick a zone to set its density.")

#     def zone_clicked(self, zid):
#         self.highlight = zid
#         idx = self.valid_zones.index(zid)
#         area = self.zone_stats[zid]
#         d    = self.density_map.get(zid, 1.0)
#         agents = int(area * d * self.base_spin.value() / AGENT_SCALE)

#         self.zone_label.setText(
#             f"Zone {idx}  (id={zid})\n"
#             f"Area: {area:,} px²\n"
#             f"Current density: {d:.1f}\n"
#             f"→ ~{agents} agents at base={self.base_spin.value():.1f}")

#         self.density_spin.setValue(d)
#         self.apply_btn.setEnabled(True)
#         self._refresh_map()

#     def apply_density(self):
#         if self.highlight is None:
#             return
#         val = self.density_spin.value()
#         self.density_map[self.highlight] = val
#         self.zone_clicked(self.highlight)   # refresh info
#         self._rebuild_zone_list()
#         self._refresh_map()

#     def save_config(self):
#         if not self.mask_path:
#             QMessageBox.warning(self, "No mask", "Load a mask first.")
#             return

#         out_path = str(Path(self.mask_path).with_suffix("")) + "_zone_config.json"
#         path, _ = QFileDialog.getSaveFileName(
#             self, "Save Zone Config", out_path, "JSON (*.json)")
#         if not path:
#             return

#         config = {
#             "mask_path"   : self.mask_path,
#             "base_density": self.base_spin.value(),
#             "agent_scale" : AGENT_SCALE,
#             "zones"       : []
#         }

#         for i, zid in enumerate(self.valid_zones):
#             area    = self.zone_stats[zid]
#             d       = self.density_map.get(zid, 1.0)
#             agents  = int(area * d * self.base_spin.value() / AGENT_SCALE)
#             config["zones"].append({
#                 "zone_index"   : i,
#                 "zone_id"      : zid,
#                 "area_px"      : area,
#                 "density_index": d,
#                 "agents"       : agents
#             })

#         with open(path, "w") as f:
#             json.dump(config, f, indent=2)

#         total = sum(z["agents"] for z in config["zones"] if z["density_index"] > 0)
#         QMessageBox.information(self, "Saved",
#             f"Config saved to:\n{path}\n\nTotal agents: {total}")

#     # ── Helpers ──────────────────────────────────────────────────────

#     def _refresh_map(self):
#         if self.binary is None:
#             return
#         qimg = render_zone_image(
#             self.binary, self.labels, self.valid_zones,
#             self.color_map, self.density_map, self.highlight)

#         pix = QPixmap.fromImage(qimg)
#         # Scale to fit nicely, keep aspect ratio
#         max_w = self.width() - 320
#         max_h = self.height() - 40
#         pix = pix.scaled(max_w, max_h,
#                          Qt.AspectRatioMode.KeepAspectRatio,
#                          Qt.TransformationMode.SmoothTransformation)
#         self.map_widget.setPixmap(pix)
#         self.map_widget.set_data(self.labels, self.valid_zones,
#                                   self.binary.shape[1], self.binary.shape[0])
#         self.map_widget.resize(pix.width(), pix.height())

#     def _rebuild_zone_list(self):
#         # Clear old
#         while self.zone_list_layout.count():
#             item = self.zone_list_layout.takeAt(0)
#             if item.widget():
#                 item.widget().deleteLater()

#         for i, zid in enumerate(self.valid_zones):
#             d     = self.density_map.get(zid, 1.0)
#             area  = self.zone_stats[zid]
#             c     = self.color_map[zid]
#             label = "outside" if d == 0 else f"d={d:.1f}"

#             row = QLabel(f"  Zone {i:2d} | {label:8s} | {area//1000}k px²")
#             row.setStyleSheet(
#                 f"background:rgb({c[0]},{c[1]},{c[2]});"
#                 f"color:{'#111' if sum(c)>400 else '#eee'};"
#                 f"border-radius:3px; padding:2px 4px;")
#             row.setFixedHeight(22)
#             self.zone_list_layout.addWidget(row)

#         self.zone_list_layout.addStretch()

#     def resizeEvent(self, event):
#         super().resizeEvent(event)
#         QTimer.singleShot(50, self._refresh_map)


# # ══════════════════════════════════════════════════════════════════════
# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     win = ZoneEditor()
#     win.show()
#     sys.exit(app.exec())



"""
zone_editor.py  —  Step 2: Interactive Zone Density Editor
Run: python zone_editor.py

Workflow:
  1. Click "Load Mask" → select your stitched binary mask
  2. Zones are auto-detected and colored
  3. Click any zone on the map → type its density index (0 = outside/ignore)
  4. Set "Base agents per 1000px²" (what density index 1 equals)
  5. Toggle "Exit Mode" → click the map to place/remove exits
  6. Click "Save Config" → writes zone_config.json for the simulation
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
MIN_ZONE_AREA     = 800
PEAK_MIN_DISTANCE = 40
AGENT_SCALE       = 1000

EXIT_RADIUS       = 10    # px — radius of exit circle drawn on map
EXIT_SNAP_DIST    = 30    # px — if click is within this of existing exit, remove it
# ──────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════
#  Core processing
# ══════════════════════════════════════════════════════════════════════

def load_and_segment(path: str):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)

    walkable = cv2.bitwise_not(img)
    _, binary = cv2.threshold(walkable, 127, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

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
    np.random.seed(42)
    colors = {}
    for i, zid in enumerate(valid_zones):
        hue = int(179 * i / max(len(valid_zones), 1))
        hsv = np.uint8([[[hue, 190, 170]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        colors[zid] = (int(bgr[2]), int(bgr[1]), int(bgr[0]))
    return colors


def render_zone_image(binary, labels, valid_zones, color_map,
                      density_map, exits, highlight_id=None, exit_mode=False):
    """
    Return an RGB QImage of the zone map with exit markers overlaid.
    exits: list of {"x": int, "y": int} in original pixel coords.
    exit_mode: if True, draws a green crosshair cursor hint border.
    """
    h, w = binary.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    for zid in valid_zones:
        d = density_map.get(zid, 1.0)
        if d == 0:
            rgb[labels == zid] = (80, 80, 80)
        else:
            rgb[labels == zid] = color_map[zid]

    rgb[binary == 0] = (20, 20, 20)

    if highlight_id is not None and highlight_id in valid_zones:
        zone_mask = (labels == highlight_id).astype(np.uint8) * 255
        contours, _ = cv2.findContours(zone_mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(rgb, contours, -1, (255, 255, 0), 3)

    for i, zid in enumerate(valid_zones):
        ys, xs = np.where(labels == zid)
        if len(xs) == 0:
            continue
        cx, cy = int(xs.mean()), int(ys.mean())
        d = density_map.get(zid, 1.0)
        text = f"{i}:{d:.1f}"
        cv2.putText(rgb, text, (cx - 15, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # ── Draw exit markers ──────────────────────────────────────────────
    for idx, ex in enumerate(exits):
        px, py = int(ex["x"]), int(ex["y"])
        # Filled green circle with black border
        cv2.circle(rgb, (px, py), EXIT_RADIUS + 3, (0, 0, 0), -1)
        cv2.circle(rgb, (px, py), EXIT_RADIUS + 2, (0, 220, 80), -1)
        # "E" label
        cv2.putText(rgb, f"E{idx+1}", (px - 9, py + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(rgb, f"E{idx+1}", (px - 9, py + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

    # Exit mode: green border around entire map as visual cue
    if exit_mode:
        cv2.rectangle(rgb, (2, 2), (w - 3, h - 3), (0, 220, 80), 4)

    h2, w2, ch = rgb.shape
    qimg = QImage(rgb.tobytes(), w2, h2, w2 * ch, QImage.Format.Format_RGB888)
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
        self.on_click    = None   # callback(zone_id)          — zone mode
        self.on_exit_click = None # callback(orig_x, orig_y)   — exit mode
        self.exit_mode   = False
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
        if self.labels is None:
            return
        self.update_scale()
        px = int(event.position().x() * self.scale_x)
        py = int(event.position().y() * self.scale_y)
        px = int(np.clip(px, 0, self.orig_w - 1))
        py = int(np.clip(py, 0, self.orig_h - 1))

        if self.exit_mode and self.on_exit_click is not None:
            self.on_exit_click(px, py)
        elif not self.exit_mode and self.on_click is not None:
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
    QPushButton#exit_active { background:#00a550; color:white; font-weight:bold;
        border:2px solid #00dc6e; }
    QPushButton#exit_active:hover { background:#00dc6e; }
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

        self.mask_path   = None
        self.binary      = None
        self.labels      = None
        self.valid_zones = []
        self.zone_stats  = {}
        self.color_map   = {}
        self.density_map = {}
        self.highlight   = None
        self.exits       = []        # list of {"x": int, "y": int}
        self.exit_mode   = False     # True = clicks place exits

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

        lv.addWidget(QLabel("Agents per 1000px²\nat density index = 1:"))
        self.base_spin = QDoubleSpinBox()
        self.base_spin.setRange(0.1, 20.0)
        self.base_spin.setValue(1.0)
        self.base_spin.setSingleStep(0.1)
        lv.addWidget(self.base_spin)

        lv.addWidget(self._sep())

        # ── Zone section ──
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

        # ── Exit editor section ────────────────────────────────────────
        exit_title = QLabel("Exit Placement")
        exit_title.setStyleSheet("font-weight:bold; color:#00dc6e;")
        lv.addWidget(exit_title)

        self.exit_mode_btn = QPushButton("🚪  Enter Exit Mode")
        self.exit_mode_btn.setToolTip(
            "Toggle Exit Mode.\n"
            "LEFT CLICK on map → place exit\n"
            "LEFT CLICK near existing exit → remove it")
        self.exit_mode_btn.clicked.connect(self.toggle_exit_mode)
        self.exit_mode_btn.setEnabled(False)
        lv.addWidget(self.exit_mode_btn)

        self.exit_info = QLabel("No exits placed yet.")
        self.exit_info.setObjectName("info")
        self.exit_info.setWordWrap(True)
        lv.addWidget(self.exit_info)

        self.clear_exits_btn = QPushButton("✕  Clear All Exits")
        self.clear_exits_btn.clicked.connect(self.clear_exits)
        self.clear_exits_btn.setEnabled(False)
        lv.addWidget(self.clear_exits_btn)

        lv.addWidget(self._sep())

        # ── Zone list ──
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
        self.map_widget.on_click       = self.zone_clicked
        self.map_widget.on_exit_click  = self.exit_clicked   # ← new
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
        self.exits       = []
        self.exit_mode   = False
        self._set_exit_mode(False)

        self.exit_mode_btn.setEnabled(True)
        self.clear_exits_btn.setEnabled(True)

        self._refresh_map()
        self._rebuild_zone_list()
        self.zone_label.setText(
            f"{len(self.valid_zones)} zones detected.\nClick a zone to set its density.")
        self._update_exit_info()

    def zone_clicked(self, zid):
        self.highlight = zid
        idx  = self.valid_zones.index(zid)
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
        self.zone_clicked(self.highlight)
        self._rebuild_zone_list()
        self._refresh_map()

    # ── Exit placement ────────────────────────────────────────────────

    def toggle_exit_mode(self):
        self._set_exit_mode(not self.exit_mode)

    def _set_exit_mode(self, active: bool):
        """Switch between zone-select mode and exit-placement mode."""
        self.exit_mode = active
        self.map_widget.exit_mode = active

        if active:
            self.exit_mode_btn.setText("✅  Exit Mode ON  (click to disable)")
            self.exit_mode_btn.setObjectName("exit_active")
            self.apply_btn.setEnabled(False)
            self.highlight = None
        else:
            self.exit_mode_btn.setText("🚪  Enter Exit Mode")
            self.exit_mode_btn.setObjectName("")

        # Force stylesheet refresh
        self.exit_mode_btn.style().unpolish(self.exit_mode_btn)
        self.exit_mode_btn.style().polish(self.exit_mode_btn)
        self._refresh_map()

    def exit_clicked(self, orig_x: int, orig_y: int):
        """
        Called when the user clicks in exit mode.
        - If click is near an existing exit (within EXIT_SNAP_DIST px) → remove it.
        - Otherwise → place a new exit at that position.
        """
        # Check if clicking near an existing exit to remove it
        for i, ex in enumerate(self.exits):
            dist = ((ex["x"] - orig_x) ** 2 + (ex["y"] - orig_y) ** 2) ** 0.5
            if dist <= EXIT_SNAP_DIST:
                self.exits.pop(i)
                self._update_exit_info()
                self._refresh_map()
                return

        # Place new exit
        self.exits.append({"x": orig_x, "y": orig_y})
        self._update_exit_info()
        self._refresh_map()

    def clear_exits(self):
        if not self.exits:
            return
        reply = QMessageBox.question(
            self, "Clear exits",
            f"Remove all {len(self.exits)} exit(s)?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self.exits = []
            self._update_exit_info()
            self._refresh_map()

    def _update_exit_info(self):
        n = len(self.exits)
        if n == 0:
            self.exit_info.setText("No exits placed yet.\nEnter Exit Mode and click the map.")
        elif n == 1:
            ex = self.exits[0]
            self.exit_info.setText(f"1 exit placed at ({ex['x']}, {ex['y']}).\n"
                                    "Click near it to remove.")
        else:
            coords = ", ".join(f"({e['x']},{e['y']})" for e in self.exits[:3])
            more   = f" +{n-3} more" if n > 3 else ""
            self.exit_info.setText(f"{n} exits: {coords}{more}\n"
                                    "Click near any exit to remove it.")

    # ── Save ─────────────────────────────────────────────────────────

    def save_config(self):
        if not self.mask_path:
            QMessageBox.warning(self, "No mask", "Load a mask first.")
            return
        if not self.exits:
            reply = QMessageBox.question(
                self, "No exits",
                "No exits have been placed.\n"
                "The analyser requires at least one exit to run.\n\n"
                "Save anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.No:
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
            "exits"       : self.exits,
            "zones"       : []
        }

        for i, zid in enumerate(self.valid_zones):
            area   = self.zone_stats[zid]
            d      = self.density_map.get(zid, 1.0)
            agents = int(area * d * self.base_spin.value() / AGENT_SCALE)
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
            f"Config saved to:\n{path}\n\n"
            f"Exits saved: {len(self.exits)}\n"
            f"Total agents: {total}")

    # ── Helpers ──────────────────────────────────────────────────────

    def _refresh_map(self):
        if self.binary is None:
            return
        qimg = render_zone_image(
            self.binary, self.labels, self.valid_zones,
            self.color_map, self.density_map,
            self.exits, self.highlight, self.exit_mode)

        pix = QPixmap.fromImage(qimg)
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
        while self.zone_list_layout.count():
            item = self.zone_list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for i, zid in enumerate(self.valid_zones):
            d    = self.density_map.get(zid, 1.0)
            area = self.zone_stats[zid]
            c    = self.color_map[zid]
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