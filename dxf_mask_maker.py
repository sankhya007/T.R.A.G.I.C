"""
dxf_mask_maker.py — DXF → Binary Mask Tool
============================================
Standalone tool for TRAGIC. Opens a DXF file, shows all its layers,
lets you toggle which ones to include, previews the result, then exports
a black-and-white PNG mask compatible with the TRAGIC pipeline.

Black  = walkable space (corridors, rooms)
White  = walls / obstacles (what you selected from layers)

Usage:
    python dxf_mask_maker.py

Requires:
    pip install ezdxf PyQt6 numpy opencv-python
"""

import sys
import math
import traceback
from pathlib import Path

import numpy as np
import cv2
import ezdxf
from ezdxf.math import BoundingBox2d

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QHBoxLayout, QVBoxLayout, QFormLayout,
    QPushButton, QLabel, QFileDialog, QFrame,
    QScrollArea, QCheckBox, QSpinBox,
    QMessageBox, QSizePolicy, QGroupBox,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QSplitter, QDoubleSpinBox, QSlider,
)
from PyQt6.QtCore import Qt, QRectF, QThread, pyqtSignal
from PyQt6.QtGui import (
    QPixmap, QImage, QPainter, QColor, QPen,
    QFont, QWheelEvent,
)


# ──────────────────────────────────────────────────────────
#  COLOUR THEME  (matches TRAGIC launcher dark palette)
# ──────────────────────────────────────────────────────────

DARK = {
    "bg":       "#0f1117",
    "panel":    "#1a1d27",
    "card":     "#20243a",
    "border":   "#2e3350",
    "accent":   "#4f8ef7",
    "success":  "#22c55e",
    "danger":   "#ef4444",
    "text":     "#e2e8f0",
    "subtext":  "#94a3b8",
    "input_bg": "#161928",
    "warning":  "#f59e0b",
}

STYLE = f"""
QMainWindow, QWidget {{
    background: {DARK['bg']};
    color: {DARK['text']};
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 10pt;
}}
QFrame#card {{
    background: {DARK['card']};
    border: 1px solid {DARK['border']};
    border-radius: 10px;
}}
QPushButton {{
    background: {DARK['card']};
    border: 1px solid {DARK['border']};
    border-radius: 6px;
    padding: 8px 14px;
    color: {DARK['text']};
}}
QPushButton:hover {{ background: {DARK['border']}; border-color: {DARK['accent']}; }}
QPushButton:pressed {{ background: {DARK['accent']}; color: white; }}
QPushButton#primary {{
    background: {DARK['accent']}; border: none;
    color: white; font-weight: bold; font-size: 11pt;
}}
QPushButton#primary:hover {{ background: #6ba3ff; }}
QPushButton#primary:disabled {{ background: {DARK['border']}; color: {DARK['subtext']}; }}
QPushButton#success {{
    background: {DARK['success']}; border: none; color: white; font-weight: bold;
}}
QCheckBox {{
    color: {DARK['text']};
    spacing: 8px;
}}
QCheckBox::indicator {{
    width: 16px; height: 16px;
    border: 1px solid {DARK['border']};
    border-radius: 3px;
    background: {DARK['input_bg']};
}}
QCheckBox::indicator:checked {{
    background: {DARK['accent']};
    border-color: {DARK['accent']};
}}
QSpinBox, QDoubleSpinBox {{
    background: {DARK['input_bg']};
    border: 1px solid {DARK['border']};
    border-radius: 5px;
    padding: 4px 8px;
    color: {DARK['text']};
}}
QScrollArea {{ border: none; background: transparent; }}
QScrollBar:vertical {{
    background: {DARK['panel']}; width: 8px; border-radius: 4px;
}}
QScrollBar::handle:vertical {{
    background: {DARK['border']}; border-radius: 4px; min-height: 20px;
}}
QScrollBar::handle:vertical:hover {{ background: {DARK['accent']}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QGroupBox {{
    border: 1px solid {DARK['border']};
    border-radius: 8px; margin-top: 12px; padding-top: 8px;
    color: {DARK['subtext']}; font-size: 9pt; font-weight: bold;
}}
QGroupBox::title {{
    subcontrol-origin: margin; left: 10px; padding: 0 4px; color: {DARK['subtext']};
}}
QLabel#title {{ font-size: 16pt; font-weight: bold; color: {DARK['text']}; }}
QLabel#sub   {{ font-size: 9pt;  color: {DARK['subtext']}; }}
QLabel#hint  {{ font-size: 8pt;  color: {DARK['subtext']}; font-style: italic; }}
QSlider::groove:horizontal {{
    height: 4px; background: {DARK['border']}; border-radius: 2px;
}}
QSlider::handle:horizontal {{
    width: 14px; height: 14px; margin: -5px 0;
    background: {DARK['accent']}; border-radius: 7px;
}}
QSplitter::handle {{ background: {DARK['border']}; }}
"""


# ──────────────────────────────────────────────────────────
#  ZOOMABLE CANVAS  (reused from TRAGIC launcher pattern)
# ──────────────────────────────────────────────────────────

class Canvas(QGraphicsView):
    def __init__(self, placeholder="Load a DXF file to begin"):
        super().__init__()
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._item = None
        self._placeholder = placeholder
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setStyleSheet(f"""
            QGraphicsView {{
                background: {DARK['input_bg']};
                border: 1px solid {DARK['border']};
                border-radius: 8px;
            }}
        """)
        self.setMinimumHeight(400)

    def show_image(self, img_bgr: np.ndarray):
        """Feed an OpenCV BGR or single-channel image to the canvas."""
        if img_bgr.ndim == 2:
            h, w = img_bgr.shape
            qimg = QImage(img_bgr.tobytes(), w, h, w, QImage.Format.Format_Grayscale8)
        else:
            h, w, _ = img_bgr.shape
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            qimg = QImage(rgb.tobytes(), w, h, w * 3, QImage.Format.Format_RGB888)

        pix = QPixmap.fromImage(qimg)
        self._scene.clear()
        self._item = QGraphicsPixmapItem(pix)
        self._scene.addItem(self._item)
        self._scene.setSceneRect(QRectF(pix.rect()))
        self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def fit(self):
        if self._item:
            self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event: QWheelEvent):
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.scale(factor, factor)

    def drawBackground(self, painter, rect):
        super().drawBackground(painter, rect)
        if self._item is None:
            painter.setPen(QPen(QColor(DARK['subtext'])))
            painter.setFont(QFont("Segoe UI", 12))
            painter.drawText(QRectF(rect), Qt.AlignmentFlag.AlignCenter, self._placeholder)


# ──────────────────────────────────────────────────────────
#  DXF RENDERING LOGIC
# ──────────────────────────────────────────────────────────

def collect_layer_names(doc) -> list[str]:
    """Return sorted list of all layer names that actually have geometry."""
    used = set()
    msp = doc.modelspace()
    for entity in msp:
        layer = getattr(entity.dxf, "layer", "0")
        used.add(layer)
    return sorted(used)


def rasterize_dxf(doc, selected_layers: set[str],
                  resolution: int = 1000,
                  wall_thickness: int = 6) -> np.ndarray | None:
    """
    Rasterize selected DXF layers into a binary mask.

    How it works:
    1.  Walk every entity in modelspace on a selected layer.
    2.  Project all geometry into a bounding box.
    3.  Scale to `resolution` pixels on the longest axis.
    4.  Draw lines / polylines / arcs / circles / rectangles into an image.
    5.  Dilate thin lines so walls have physical thickness in the mask.

    Returns a numpy array where:
        255 = wall / obstacle  (white)
        0   = walkable space   (black)
    """
    msp = doc.modelspace()

    # ── collect all vertex points to compute bounding box ──
    all_pts = []
    for entity in msp:
        layer = getattr(entity.dxf, "layer", "0")
        if layer not in selected_layers:
            continue
        pts = _entity_points(entity)
        all_pts.extend(pts)

    if not all_pts:
        return None

    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    span_x = max_x - min_x
    span_y = max_y - min_y
    if span_x < 1e-9 or span_y < 1e-9:
        return None

    # scale so longest axis = resolution
    scale = resolution / max(span_x, span_y)
    pad = wall_thickness * 2
    W = int(math.ceil(span_x * scale)) + pad * 2
    H = int(math.ceil(span_y * scale)) + pad * 2

    canvas_img = np.zeros((H, W), dtype=np.uint8)

    def to_px(x, y):
        # DXF Y axis is up; image Y axis is down — flip it
        px = int((x - min_x) * scale) + pad
        py = H - 1 - (int((y - min_y) * scale) + pad)
        return px, py

    # ── draw each entity ──
    for entity in msp:
        layer = getattr(entity.dxf, "layer", "0")
        if layer not in selected_layers:
            continue
        _draw_entity(canvas_img, entity, to_px, wall_thickness)

    # ── dilate: thin centerlines → thick solid walls ──
    if wall_thickness > 1:
        k = wall_thickness
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (k * 2 + 1, k * 2 + 1))
        canvas_img = cv2.dilate(canvas_img, kernel, iterations=1)

    return canvas_img   # 255 = wall, 0 = walkable


def _entity_points(entity) -> list[tuple[float, float]]:
    """Extract a rough bounding sample of points from an entity."""
    t = entity.dxftype()
    pts = []
    try:
        if t == "LINE":
            s, e = entity.dxf.start, entity.dxf.end
            pts = [(s.x, s.y), (e.x, e.y)]
        elif t in ("LWPOLYLINE", "POLYLINE"):
            pts = [(v[0], v[1]) for v in entity.vertices()]
        elif t == "ARC":
            c = entity.dxf.center
            r = entity.dxf.radius
            for a in range(0, 360, 30):
                rad = math.radians(a)
                pts.append((c.x + r * math.cos(rad), c.y + r * math.sin(rad)))
        elif t == "CIRCLE":
            c = entity.dxf.center
            r = entity.dxf.radius
            pts = [(c.x + r, c.y), (c.x - r, c.y), (c.x, c.y + r), (c.x, c.y - r)]
        elif t in ("SPLINE", "ELLIPSE"):
            for pt in entity.flattening(0.1):
                pts.append((pt[0], pt[1]))
        elif t == "INSERT":
            p = entity.dxf.insert
            pts = [(p.x, p.y)]
    except Exception:
        pass
    return pts


def _draw_entity(img, entity, to_px, thickness):
    """Draw a single DXF entity onto the image."""
    t = entity.dxftype()
    try:
        if t == "LINE":
            s, e = entity.dxf.start, entity.dxf.end
            cv2.line(img, to_px(s.x, s.y), to_px(e.x, e.y), 255, 1)

        elif t == "LWPOLYLINE":
            pts = [(v[0], v[1]) for v in entity.vertices()]
            closed = entity.closed
            for i in range(len(pts) - 1):
                cv2.line(img, to_px(*pts[i]), to_px(*pts[i + 1]), 255, 1)
            if closed and len(pts) > 1:
                cv2.line(img, to_px(*pts[-1]), to_px(*pts[0]), 255, 1)

        elif t == "POLYLINE":
            pts = [(v.dxf.location.x, v.dxf.location.y)
                   for v in entity.vertices
                   if hasattr(v.dxf, "location")]
            closed = bool(entity.is_closed)
            for i in range(len(pts) - 1):
                cv2.line(img, to_px(*pts[i]), to_px(*pts[i + 1]), 255, 1)
            if closed and len(pts) > 1:
                cv2.line(img, to_px(*pts[-1]), to_px(*pts[0]), 255, 1)

        elif t == "ARC":
            c = entity.dxf.center
            r = entity.dxf.radius
            cx, cy = to_px(c.x, c.y)
            start_a = entity.dxf.start_angle
            end_a   = entity.dxf.end_angle
            # OpenCV angles: clockwise from 3 o'clock; DXF: CCW from 3 o'clock
            # Also the Y axis flip inverts the arc direction
            scale_r = r  # we need pixel radius
            # estimate pixel radius from a single radial sample
            px1, py1 = to_px(c.x + r, c.y)
            pr = abs(px1 - cx)
            cv2.ellipse(img, (cx, cy), (max(1, pr), max(1, pr)),
                        0, -end_a, -start_a, 255, 1)

        elif t == "CIRCLE":
            c = entity.dxf.center
            px1, _ = to_px(c.x + entity.dxf.radius, c.y)
            cx, cy = to_px(c.x, c.y)
            pr = abs(px1 - cx)
            cv2.circle(img, (cx, cy), max(1, pr), 255, 1)

        elif t in ("SPLINE", "ELLIPSE"):
            pts = [(p[0], p[1]) for p in entity.flattening(0.5)]
            for i in range(len(pts) - 1):
                cv2.line(img, to_px(*pts[i]), to_px(*pts[i + 1]), 255, 1)

    except Exception:
        pass  # silently skip malformed entities


# ──────────────────────────────────────────────────────────
#  WORKER THREAD  (keeps UI responsive during rasterization)
# ──────────────────────────────────────────────────────────

class RasterWorker(QThread):
    done    = pyqtSignal(object)   # numpy array or None
    error   = pyqtSignal(str)

    def __init__(self, doc, layers, resolution, thickness):
        super().__init__()
        self._doc   = doc
        self._layers  = layers
        self._res     = resolution
        self._thick   = thickness

    def run(self):
        try:
            result = rasterize_dxf(self._doc, self._layers,
                                   self._res, self._thick)
            self.done.emit(result)
        except Exception as e:
            self.error.emit(traceback.format_exc())


# ──────────────────────────────────────────────────────────
#  LAYER ROW WIDGET
# ──────────────────────────────────────────────────────────

class LayerRow(QWidget):
    """One row in the layer list: toggle + layer name."""

    def __init__(self, name: str, on_change):
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(10)

        self._cb = QCheckBox(name)
        self._cb.setChecked(False)
        self._cb.stateChanged.connect(on_change)
        layout.addWidget(self._cb, 1)

        # colour swatch (placeholder — just shows layer is recognised)
        swatch = QFrame()
        swatch.setFixedSize(12, 12)
        swatch.setStyleSheet(f"background: {DARK['accent']}; border-radius: 3px;")
        layout.addWidget(swatch)

        self.setStyleSheet(f"""
            QWidget {{ background: transparent; }}
            QWidget:hover {{ background: {DARK['border']}; border-radius: 4px; }}
        """)

    @property
    def name(self):
        return self._cb.text()

    @property
    def enabled(self):
        return self._cb.isChecked()

    def set_enabled(self, v: bool):
        self._cb.setChecked(v)


# ──────────────────────────────────────────────────────────
#  MAIN WINDOW
# ──────────────────────────────────────────────────────────

class DXFMaskMaker(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DXF → Binary Mask   |   TRAGIC Floorplan Parser")
        self.setMinimumSize(1280, 780)
        self.setStyleSheet(STYLE)

        self._doc       = None      # ezdxf document
        self._layer_rows: list[LayerRow] = []
        self._mask      = None      # numpy uint8 array (255=wall, 0=walkable)
        self._worker    = None

        self._build_ui()

    # ── UI ──────────────────────────────────────────────────

    def _build_ui(self):
        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self.setCentralWidget(root)

        # ── header ──
        header = QWidget()
        header.setFixedHeight(60)
        header.setStyleSheet(f"background: {DARK['panel']}; border-bottom: 1px solid {DARK['border']};")
        hl = QHBoxLayout(header)
        hl.setContentsMargins(24, 0, 24, 0)
        logo = QLabel("⚠ TRAGIC  —  DXF Mask Maker")
        logo.setStyleSheet(f"font-size: 13pt; font-weight: bold; color: {DARK['accent']};")
        hl.addWidget(logo)
        hl.addStretch()
        tagline = QLabel("Turn a CAD floorplan into a walkability mask for crowd simulation")
        tagline.setStyleSheet(f"color: {DARK['subtext']}; font-size: 9pt;")
        hl.addWidget(tagline)
        outer.addWidget(header)

        # ── main splitter: left panel | canvas ──
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(4)
        outer.addWidget(splitter, 1)

        # ── LEFT PANEL ──────────────────────────────────────
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFixedWidth(320)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(16, 16, 16, 16)
        lv.setSpacing(14)

        # ── Step 1: load file ──
        step1 = self._group("Step 1 — Open DXF File")
        s1v = QVBoxLayout(step1)

        load_hint = QLabel(
            "Load a .dxf file exported from AutoCAD, Revit, SketchUp, or any CAD tool. "
            "DWG is not supported — use 'Save As DXF' in your CAD app first."
        )
        load_hint.setObjectName("hint"); load_hint.setWordWrap(True)
        s1v.addWidget(load_hint)

        self._load_btn = QPushButton("📂  Open DXF File")
        self._load_btn.clicked.connect(self._open_file)
        s1v.addWidget(self._load_btn)

        self._file_label = QLabel("No file loaded.")
        self._file_label.setObjectName("hint"); self._file_label.setWordWrap(True)
        s1v.addWidget(self._file_label)

        lv.addWidget(step1)

        # ── Step 2: layer picker ──
        step2 = self._group("Step 2 — Pick Wall / Obstacle Layers")
        s2v = QVBoxLayout(step2)

        layer_hint = QLabel(
            "Each checkbox is one DXF layer. Check layers that contain walls, columns, "
            "or fixed obstacles. Leave corridors, doors, furniture, and text unchecked — "
            "those will become walkable space.\n\n"
            "Tip: layer names like 'A-WALL', 'WALLS', or '0' usually hold the structure."
        )
        layer_hint.setObjectName("hint"); layer_hint.setWordWrap(True)
        s2v.addWidget(layer_hint)

        sel_row = QHBoxLayout()
        sel_all = QPushButton("Select All")
        sel_none = QPushButton("Clear All")
        sel_all.clicked.connect(lambda: self._select_all(True))
        sel_none.clicked.connect(lambda: self._select_all(False))
        sel_row.addWidget(sel_all); sel_row.addWidget(sel_none)
        s2v.addLayout(sel_row)

        # scrollable layer list
        self._layer_scroll = QScrollArea()
        self._layer_scroll.setWidgetResizable(True)
        self._layer_scroll.setMinimumHeight(180)
        self._layer_scroll.setMaximumHeight(280)
        self._layer_container = QWidget()
        self._layer_layout = QVBoxLayout(self._layer_container)
        self._layer_layout.setSpacing(2)
        self._layer_layout.setContentsMargins(0, 0, 0, 0)
        self._layer_layout.addStretch()
        self._layer_scroll.setWidget(self._layer_container)
        s2v.addWidget(self._layer_scroll)

        self._layer_count = QLabel("No layers loaded.")
        self._layer_count.setObjectName("hint")
        s2v.addWidget(self._layer_count)

        lv.addWidget(step2)

        # ── Step 3: export settings ──
        step3 = self._group("Step 3 — Raster Settings")
        s3v = QFormLayout(step3)
        s3v.setSpacing(8)

        self._res_spin = QSpinBox()
        self._res_spin.setRange(256, 4096)
        self._res_spin.setValue(1024)
        self._res_spin.setSingleStep(256)
        self._res_spin.setToolTip("Longer axis of the output image in pixels.")
        s3v.addRow("Output size (px):", self._res_spin)

        self._thick_spin = QSpinBox()
        self._thick_spin.setRange(1, 30)
        self._thick_spin.setValue(6)
        self._thick_spin.setToolTip(
            "DXF walls are often drawn as thin centerlines. "
            "This value thickens them so agents can't clip through. "
            "6–10 px works well for most plans."
        )
        s3v.addRow("Wall thickness (px):", self._thick_spin)

        thick_hint = QLabel(
            "Wall thickness matters because CAD walls are centerlines, not filled shapes. "
            "Too thin → agents clip through walls. Too thick → doorways close up."
        )
        thick_hint.setObjectName("hint"); thick_hint.setWordWrap(True)
        step3.layout().addRow(thick_hint)  # type: ignore

        lv.addWidget(step3)

        # ── Step 4: generate + save ──
        step4 = self._group("Step 4 — Generate & Save")
        s4v = QVBoxLayout(step4)

        gen_hint = QLabel(
            "Generate Preview renders the mask in the viewer. "
            "Save Mask exports it as stitched_mask.png ready to drop into TRAGIC."
        )
        gen_hint.setObjectName("hint"); gen_hint.setWordWrap(True)
        s4v.addWidget(gen_hint)

        self._gen_btn = QPushButton("▶  Generate Preview")
        self._gen_btn.setObjectName("primary")
        self._gen_btn.setEnabled(False)
        self._gen_btn.clicked.connect(self._generate)
        s4v.addWidget(self._gen_btn)

        self._status = QLabel("")
        self._status.setObjectName("hint"); self._status.setWordWrap(True)
        s4v.addWidget(self._status)

        self._save_btn = QPushButton("💾  Save Mask as PNG")
        self._save_btn.setObjectName("success")
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(self._save_mask)
        s4v.addWidget(self._save_btn)

        lv.addWidget(step4)
        lv.addStretch()

        left_scroll.setWidget(left)
        splitter.addWidget(left_scroll)

        # ── RIGHT: canvas ──────────────────────────────────
        right = QFrame()
        right.setObjectName("card")
        rv = QVBoxLayout(right)
        rv.setContentsMargins(12, 12, 12, 12)
        rv.setSpacing(8)

        canvas_header = QHBoxLayout()
        self._canvas_title = QLabel("Preview")
        self._canvas_title.setStyleSheet("font-weight: bold; font-size: 11pt;")
        fit_btn = QPushButton("Fit")
        fit_btn.setFixedWidth(50)
        fit_btn.clicked.connect(lambda: self._canvas.fit())
        self._mode_label = QLabel("")
        self._mode_label.setObjectName("hint")
        canvas_header.addWidget(self._canvas_title)
        canvas_header.addWidget(self._mode_label)
        canvas_header.addStretch()
        canvas_header.addWidget(fit_btn)
        rv.addLayout(canvas_header)

        self._canvas = Canvas(
            "Open a DXF file to see the floorplan here.\n\n"
            "Scroll to zoom  ·  Drag to pan"
        )
        rv.addWidget(self._canvas, 1)

        # toggle between wire preview and binary mask
        toggle_row = QHBoxLayout()
        self._wire_btn = QPushButton("Show Wire Preview")
        self._mask_btn = QPushButton("Show Binary Mask")
        self._wire_btn.setEnabled(False)
        self._mask_btn.setEnabled(False)
        self._wire_btn.clicked.connect(self._show_wire)
        self._mask_btn.clicked.connect(self._show_mask_view)
        toggle_row.addWidget(self._wire_btn)
        toggle_row.addWidget(self._mask_btn)
        rv.addLayout(toggle_row)

        splitter.addWidget(right)
        splitter.setSizes([320, 960])

        # internal state for preview toggle
        self._wire_img = None   # BGR numpy array of wire frame

    # ── helpers ─────────────────────────────────────────────

    def _group(self, title: str) -> QGroupBox:
        g = QGroupBox(title)
        return g

    def _sep(self):
        f = QFrame(); f.setFrameShape(QFrame.Shape.HLine)
        f.setStyleSheet(f"color: {DARK['border']};"); return f

    # ── file loading ─────────────────────────────────────────

    def _open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open DXF File", "", "DXF Files (*.dxf *.DXF)")
        if not path:
            return
        try:
            self._doc = ezdxf.readfile(path)
        except Exception as e:
            QMessageBox.critical(self, "Load Error",
                f"Could not read DXF:\n{e}\n\nMake sure the file is a valid DXF (not DWG).")
            return

        self._file_label.setText(Path(path).name)
        self._file_label.setStyleSheet(f"color: {DARK['success']}; font-size: 9pt;")
        self._populate_layers()
        self._render_wire_preview()
        self._gen_btn.setEnabled(True)
        self._save_btn.setEnabled(False)
        self._mask = None
        self._status.setText("")

    def _populate_layers(self):
        # clear old rows
        while self._layer_layout.count() > 1:
            item = self._layer_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._layer_rows.clear()

        names = collect_layer_names(self._doc)
        for name in names:
            row = LayerRow(name, self._on_layer_change)
            self._layer_rows.append(row)
            self._layer_layout.insertWidget(self._layer_layout.count() - 1, row)

        n = len(names)
        self._layer_count.setText(
            f"{n} layer{'s' if n != 1 else ''} found in this file.")

    def _select_all(self, state: bool):
        for row in self._layer_rows:
            row.set_enabled(state)

    def _on_layer_change(self):
        n = sum(1 for r in self._layer_rows if r.enabled)
        self._status.setText(f"{n} layer(s) selected.")

    # ── wire preview (shows all geometry, coloured by layer) ──

    def _render_wire_preview(self):
        """Draw all geometry in white on black — quick overview before picking layers."""
        if self._doc is None:
            return

        # gather all points for bounding box
        all_pts = []
        msp = self._doc.modelspace()
        for entity in msp:
            all_pts.extend(_entity_points(entity))

        if not all_pts:
            return

        xs = [p[0] for p in all_pts]
        ys = [p[1] for p in all_pts]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        span_x = max_x - min_x
        span_y = max_y - min_y
        if span_x < 1e-9 or span_y < 1e-9:
            return

        WIRE_RES = 1200
        scale = WIRE_RES / max(span_x, span_y)
        pad = 20
        W = int(math.ceil(span_x * scale)) + pad * 2
        H = int(math.ceil(span_y * scale)) + pad * 2

        img = np.zeros((H, W, 3), dtype=np.uint8)

        # assign a different colour per layer so user can identify them
        layer_names = sorted(set(
            getattr(e.dxf, "layer", "0") for e in msp))
        palette = [
            (0x4f, 0x8e, 0xf7),  # blue
            (0x22, 0xc5, 0x5e),  # green
            (0xef, 0x44, 0x44),  # red
            (0xf5, 0x9e, 0x0b),  # amber
            (0xa8, 0x5c, 0xff),  # purple
            (0x06, 0xb6, 0xd4),  # cyan
            (0xf9, 0x73, 0x16),  # orange
            (0xec, 0x48, 0x99),  # pink
        ]
        layer_color = {
            name: palette[i % len(palette)]
            for i, name in enumerate(layer_names)
        }

        def to_px(x, y):
            px = int((x - min_x) * scale) + pad
            py = H - 1 - (int((y - min_y) * scale) + pad)
            return px, py

        for entity in msp:
            layer = getattr(entity.dxf, "layer", "0")
            col = layer_color.get(layer, (200, 200, 200))
            _draw_entity_colour(img, entity, to_px, col)

        self._wire_img = img
        self._canvas.show_image(img)
        self._canvas_title.setText("Wire Preview  (colours = layers)")
        self._mode_label.setText(
            "Each colour is a different DXF layer. Use this to identify which layers hold walls.")
        self._wire_btn.setEnabled(True)
        self._canvas.fit()

    # ── generate binary mask ──────────────────────────────────

    def _generate(self):
        selected = {r.name for r in self._layer_rows if r.enabled}
        if not selected:
            QMessageBox.warning(self, "No Layers",
                "Select at least one layer to include in the mask.")
            return

        self._gen_btn.setEnabled(False)
        self._save_btn.setEnabled(False)
        self._status.setText("Rasterizing…")

        self._worker = RasterWorker(
            self._doc, selected,
            self._res_spin.value(),
            self._thick_spin.value(),
        )
        self._worker.done.connect(self._on_raster_done)
        self._worker.error.connect(self._on_raster_error)
        self._worker.start()

    def _on_raster_done(self, result):
        self._gen_btn.setEnabled(True)
        if result is None:
            self._status.setText("No geometry found on selected layers.")
            return
        self._mask = result
        self._canvas.show_image(result)
        self._canvas_title.setText("Binary Mask  (white = wall  ·  black = walkable)")
        self._mode_label.setText("")
        self._mask_btn.setEnabled(True)
        self._save_btn.setEnabled(True)
        h, w = result.shape
        wall_pct = int(100 * result.sum() / (255 * w * h))
        self._status.setText(
            f"Mask generated  —  {w}×{h}px  |  Walls {wall_pct}%  |  Walkable {100-wall_pct}%\n"
            "Save it as stitched_mask.png and open it in TRAGIC's Zone Editor."
        )
        self._status.setStyleSheet(f"color: {DARK['success']}; font-size: 9pt;")
        self._canvas.fit()

    def _on_raster_error(self, msg):
        self._gen_btn.setEnabled(True)
        self._status.setText(f"Error during rasterization.")
        self._status.setStyleSheet(f"color: {DARK['danger']}; font-size: 9pt;")
        QMessageBox.critical(self, "Rasterization Error", msg)

    # ── preview toggle ────────────────────────────────────────

    def _show_wire(self):
        if self._wire_img is not None:
            self._canvas.show_image(self._wire_img)
            self._canvas_title.setText("Wire Preview  (colours = layers)")
            self._mode_label.setText(
                "Each colour is a different DXF layer. Use this to identify which layers hold walls.")
            self._canvas.fit()

    def _show_mask_view(self):
        if self._mask is not None:
            self._canvas.show_image(self._mask)
            self._canvas_title.setText("Binary Mask  (white = wall  ·  black = walkable)")
            self._mode_label.setText("")
            self._canvas.fit()

    # ── save ──────────────────────────────────────────────────

    def _save_mask(self):
        if self._mask is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Binary Mask", "stitched_mask.png",
            "PNG Image (*.png)")
        if not path:
            return
        cv2.imwrite(path, self._mask)
        self._status.setText(f"Saved → {Path(path).name}")
        self._status.setStyleSheet(f"color: {DARK['success']}; font-size: 9pt;")


# ──────────────────────────────────────────────────────────
#  COLOUR WIRE HELPER  (used only for the preview, not the mask)
# ──────────────────────────────────────────────────────────

def _draw_entity_colour(img, entity, to_px, colour):
    t = entity.dxftype()
    col = tuple(reversed(colour))   # BGR
    try:
        if t == "LINE":
            s, e = entity.dxf.start, entity.dxf.end
            cv2.line(img, to_px(s.x, s.y), to_px(e.x, e.y), col, 1)
        elif t == "LWPOLYLINE":
            pts = [(v[0], v[1]) for v in entity.vertices()]
            closed = entity.closed
            for i in range(len(pts) - 1):
                cv2.line(img, to_px(*pts[i]), to_px(*pts[i+1]), col, 1)
            if closed and len(pts) > 1:
                cv2.line(img, to_px(*pts[-1]), to_px(*pts[0]), col, 1)
        elif t == "POLYLINE":
            pts = [(v.dxf.location.x, v.dxf.location.y)
                   for v in entity.vertices if hasattr(v.dxf, "location")]
            closed = bool(entity.is_closed)
            for i in range(len(pts) - 1):
                cv2.line(img, to_px(*pts[i]), to_px(*pts[i+1]), col, 1)
            if closed and len(pts) > 1:
                cv2.line(img, to_px(*pts[-1]), to_px(*pts[0]), col, 1)
        elif t == "ARC":
            c = entity.dxf.center
            r = entity.dxf.radius
            cx, cy = to_px(c.x, c.y)
            px1, _ = to_px(c.x + r, c.y)
            pr = max(1, abs(px1 - cx))
            cv2.ellipse(img, (cx, cy), (pr, pr), 0,
                        -entity.dxf.end_angle, -entity.dxf.start_angle, col, 1)
        elif t == "CIRCLE":
            c = entity.dxf.center
            cx, cy = to_px(c.x, c.y)
            px1, _ = to_px(c.x + entity.dxf.radius, c.y)
            pr = max(1, abs(px1 - cx))
            cv2.circle(img, (cx, cy), pr, col, 1)
        elif t in ("SPLINE", "ELLIPSE"):
            pts = [(p[0], p[1]) for p in entity.flattening(0.5)]
            for i in range(len(pts) - 1):
                cv2.line(img, to_px(*pts[i]), to_px(*pts[i+1]), col, 1)
    except Exception:
        pass


# ──────────────────────────────────────────────────────────
#  ENTRY POINT
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = DXFMaskMaker()
    win.show()
    sys.exit(app.exec())