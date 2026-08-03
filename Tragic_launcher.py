import sys
import json
import time
import threading
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import cv2

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QStackedWidget,
    QHBoxLayout, QVBoxLayout, QGridLayout, QFormLayout,
    QPushButton, QLabel, QFileDialog, QFrame, QSizePolicy,
    QScrollArea, QDoubleSpinBox, QSpinBox, QSlider,
    QProgressBar, QMessageBox, QLineEdit, QGroupBox,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QDialog, QTextEdit,
)
from PyQt6.QtCore import (
    Qt, QTimer, QThread, pyqtSignal, QPointF, QRectF,
    QPropertyAnimation, QEasingCurve, QSize,
)
from PyQt6.QtGui import (
    QPixmap, QImage, QPainter, QColor, QPen, QBrush,
    QFont, QWheelEvent, QTransform, QCursor,
)

HAZARD_BLOCK_RADIUS = 90  # px — must match the sim scripts' hazard_block_radius

# ══════════════════════════════════════════════════════════
#  SHARED STATE
# ══════════════════════════════════════════════════════════

@dataclass
class AppState:
    """Shared state passed between all three views."""
    image_path: str = ""
    mask_path: str = ""          # output of View 1
    zone_config_path: str = ""   # output of View 2
    selected_model: str = "SFM"
    output_image_path: str = ""  # output of View 3
    hazard: dict = field(default_factory=dict)  # optional hazard position: {"x":int,"y":int}


# ══════════════════════════════════════════════════════════
#  THEME
# ══════════════════════════════════════════════════════════

DARK = {
    "bg":       "#0f1117",
    "panel":    "#1a1d27",
    "card":     "#20243a",
    "border":   "#2e3350",
    "accent":   "#4f8ef7",
    "accent2":  "#7c5cfc",
    "success":  "#22c55e",
    "warning":  "#f59e0b",
    "danger":   "#ef4444",
    "text":     "#e2e8f0",
    "subtext":  "#94a3b8",
    "input_bg": "#161928",
}

STYLESHEET = f"""
QMainWindow, QWidget {{
    background: {DARK['bg']};
    color: {DARK['text']};
    font-family: 'Segoe UI', 'Inter', Arial, sans-serif;
    font-size: 10pt;
}}
QFrame#card {{
    background: {DARK['card']};
    border: 1px solid {DARK['border']};
    border-radius: 10px;
}}
QFrame#panel {{
    background: {DARK['panel']};
    border: 1px solid {DARK['border']};
    border-radius: 8px;
}}
QPushButton {{
    background: {DARK['card']};
    border: 1px solid {DARK['border']};
    border-radius: 6px;
    padding: 8px 16px;
    color: {DARK['text']};
    font-weight: 500;
}}
QPushButton:hover {{
    background: {DARK['border']};
    border-color: {DARK['accent']};
}}
QPushButton:pressed {{
    background: {DARK['accent']};
    color: white;
}}
QPushButton#primary {{
    background: {DARK['accent']};
    border: none;
    color: white;
    font-weight: bold;
    font-size: 11pt;
}}
QPushButton#primary:hover {{
    background: #6ba3ff;
}}
QPushButton#primary:disabled {{
    background: {DARK['border']};
    color: {DARK['subtext']};
}}
QPushButton#danger {{
    background: {DARK['danger']};
    border: none;
    color: white;
    font-weight: bold;
}}
QPushButton#success {{
    background: {DARK['success']};
    border: none;
    color: white;
    font-weight: bold;
}}
QPushButton#model_card {{
    background: {DARK['card']};
    border: 2px solid {DARK['border']};
    border-radius: 10px;
    padding: 16px;
    text-align: left;
    font-size: 11pt;
}}
QPushButton#model_card:hover {{
    border-color: {DARK['accent']};
    background: {DARK['panel']};
}}
QPushButton#model_card_selected {{
    background: {DARK['panel']};
    border: 2px solid {DARK['accent']};
    border-radius: 10px;
    padding: 16px;
    text-align: left;
    font-size: 11pt;
    color: {DARK['accent']};
    font-weight: bold;
}}
QSpinBox, QDoubleSpinBox, QLineEdit {{
    background: {DARK['input_bg']};
    border: 1px solid {DARK['border']};
    border-radius: 5px;
    padding: 5px 8px;
    color: {DARK['text']};
    selection-background-color: {DARK['accent']};
}}
QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus {{
    border-color: {DARK['accent']};
}}
QScrollArea {{ border: none; background: transparent; }}
QScrollBar:vertical {{
    background: {DARK['panel']};
    width: 8px;
    border-radius: 4px;
}}
QScrollBar::handle:vertical {{
    background: {DARK['border']};
    border-radius: 4px;
    min-height: 20px;
}}
QScrollBar::handle:vertical:hover {{
    background: {DARK['accent']};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QProgressBar {{
    background: {DARK['input_bg']};
    border: 1px solid {DARK['border']};
    border-radius: 4px;
    height: 8px;
    text-align: center;
}}
QProgressBar::chunk {{
    background: {DARK['accent']};
    border-radius: 4px;
}}
QLabel#title {{
    font-size: 18pt;
    font-weight: bold;
    color: {DARK['text']};
}}
QLabel#subtitle {{
    font-size: 10pt;
    color: {DARK['subtext']};
}}
QLabel#section {{
    font-size: 10pt;
    font-weight: bold;
    color: {DARK['subtext']};
    text-transform: uppercase;
    letter-spacing: 1px;
}}
QLabel#badge {{
    background: {DARK['accent']};
    color: white;
    border-radius: 10px;
    padding: 2px 10px;
    font-size: 9pt;
    font-weight: bold;
}}
QGroupBox {{
    border: 1px solid {DARK['border']};
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 8px;
    color: {DARK['subtext']};
    font-size: 9pt;
    font-weight: bold;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    color: {DARK['subtext']};
}}
"""


# ══════════════════════════════════════════════════════════
#  WORKER THREAD
# ══════════════════════════════════════════════════════════

class Worker(QThread):
    """Generic worker thread. Pass a callable and it runs it off the UI thread."""
    progress = pyqtSignal(int, str)   # percent, message
    finished = pyqtSignal(bool, str)  # success, message

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs

    def run(self):
        try:
            self._fn(*self._args, progress_cb=self.progress.emit, **self._kwargs)
            self.finished.emit(True, "Done")
        except Exception as e:
            self.finished.emit(False, str(e))


# ══════════════════════════════════════════════════════════
#  NAV BAR
# ══════════════════════════════════════════════════════════

class NavBar(QWidget):
    # Emitted when the user clicks a step button; carries the target index
    nav_clicked = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.setFixedHeight(56)
        self._max_visited = 0   # highest step index the user has reached
        self.setObjectName("navbar")
        self.setStyleSheet(f"""
            QWidget#navbar {{
                background: {DARK['panel']};
                border-bottom: 1px solid {DARK['border']};
            }}
        """)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(24, 0, 24, 0)

        # Logo
        logo = QLabel("TRAGIC")
        logo.setStyleSheet(f"font-size: 14pt; font-weight: bold; color: {DARK['accent']}; letter-spacing: 2px;")
        layout.addWidget(logo)

        layout.addSpacing(40)

        # Step buttons — clickable to jump back to any visited step
        self.steps = []
        step_names = ["Map Parser", "Zone Editor", "Simulation"]
        for i, name in enumerate(step_names):
            btn = QPushButton(f"  {i+1}. {name}  ")
            btn.setFixedHeight(32)
            btn.setEnabled(False)   # enabled only once the step has been visited
            btn.setFlat(True)
            btn.setStyleSheet(f"""
                QPushButton {{
                    color: {DARK['subtext']};
                    border: none;
                    border-radius: 6px;
                    padding: 4px 12px;
                    font-size: 10pt;
                    background: transparent;
                }}
            """)
            btn.clicked.connect(lambda checked, idx=i: self.nav_clicked.emit(idx))
            self.steps.append(btn)
            layout.addWidget(btn)
            if i < len(step_names) - 1:
                arrow = QLabel(">")
                arrow.setStyleSheet(f"color: {DARK['border']}; font-size: 12pt;")
                layout.addWidget(arrow)

        layout.addStretch()

        info = QLabel("Crowd Evacuation Intelligence System")
        info.setStyleSheet(f"color: {DARK['subtext']}; font-size: 9pt;")
        layout.addWidget(info)

    def set_active(self, index: int):
        # Track the furthest step reached so we know which buttons to enable
        if index > self._max_visited:
            self._max_visited = index

        for i, btn in enumerate(self.steps):
            # Only enable steps the user has already been to
            btn.setEnabled(i <= self._max_visited)

            if i == index:
                # Current step — highlighted blue
                btn.setStyleSheet(f"""
                    QPushButton {{
                        color: white;
                        background: {DARK['accent']};
                        border: none;
                        border-radius: 6px;
                        padding: 4px 12px;
                        font-weight: bold;
                        font-size: 10pt;
                    }}
                """)
            elif i < self._max_visited:
                # Previously visited step — green, clickable cursor
                btn.setStyleSheet(f"""
                    QPushButton {{
                        color: {DARK['success']};
                        background: transparent;
                        border: none;
                        border-radius: 6px;
                        padding: 4px 12px;
                        font-size: 10pt;
                    }}
                    QPushButton:hover {{
                        background: {DARK['border']};
                    }}
                """)
                btn.setCursor(Qt.CursorShape.PointingHandCursor)
            else:
                # Not yet visited — greyed out, not clickable
                btn.setStyleSheet(f"""
                    QPushButton {{
                        color: {DARK['subtext']};
                        background: transparent;
                        border: none;
                        border-radius: 6px;
                        padding: 4px 12px;
                        font-size: 10pt;
                    }}
                """)
                btn.setCursor(Qt.CursorShape.ArrowCursor)


# ══════════════════════════════════════════════════════════
#  ZOOMABLE IMAGE VIEW  (shared by View 1 preview + View 3 output)
# ══════════════════════════════════════════════════════════

class ZoomableImageView(QGraphicsView):
    def __init__(self, placeholder_text="No image loaded"):
        super().__init__()
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item: Optional[QGraphicsPixmapItem] = None
        self._placeholder = placeholder_text
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
        self.setMinimumHeight(300)

    def load_image(self, path: str):
        self._scene.clear()
        pix = QPixmap(path)
        if pix.isNull():
            return
        self._pixmap_item = QGraphicsPixmapItem(pix)
        self._scene.addItem(self._pixmap_item)
        self._scene.setSceneRect(QRectF(pix.rect()))
        self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def load_pixmap(self, pix: QPixmap):
        self._scene.clear()
        self._pixmap_item = QGraphicsPixmapItem(pix)
        self._scene.addItem(self._pixmap_item)
        self._scene.setSceneRect(QRectF(pix.rect()))
        self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event: QWheelEvent):
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.scale(factor, factor)

    def drawBackground(self, painter, rect):
        super().drawBackground(painter, rect)
        if self._pixmap_item is None:
            painter.setPen(QPen(QColor(DARK['subtext'])))
            painter.setFont(QFont("Segoe UI", 12))
            painter.drawText(
                QRectF(rect),
                Qt.AlignmentFlag.AlignCenter,
                self._placeholder
            )

    def reset_zoom(self):
        if self._pixmap_item:
            self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)


class EditableImageView(ZoomableImageView):
    """ZoomableImageView that can optionally intercept mouse events to draw
    white lines (walls) directly onto the mask image."""

    def __init__(self, placeholder_text="No image loaded"):
        super().__init__(placeholder_text)
        self.edit_mode = False
        self.brush_size = 6
        self._canvas: Optional[np.ndarray] = None
        self._last_pt: Optional[tuple] = None
        self._undo_stack: list = []
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    # ── public API ──────────────────────────────────────────────────

    def load_canvas(self, path: str):
        """Load image and keep a numpy copy for drawing."""
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return
        self._canvas = img.copy()
        self._undo_stack.clear()
        self._refresh_pixmap()
        self._scene.setSceneRect(QRectF(0, 0, img.shape[1], img.shape[0]))
        self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def get_canvas(self) -> Optional[np.ndarray]:
        return self._canvas

    def undo(self):
        if self._undo_stack:
            self._canvas = self._undo_stack.pop()
            self._refresh_pixmap()

    # ── mouse drawing ───────────────────────────────────────────────

    def mousePressEvent(self, event):
        if self.edit_mode and event.button() == Qt.MouseButton.LeftButton:
            self._undo_stack.append(self._canvas.copy())  # snapshot before stroke
            pt = self._to_image(event.position())
            if pt:
                self._last_pt = pt
                cv2.circle(self._canvas, pt, self.brush_size // 2, 255, -1)
                self._refresh_pixmap()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.edit_mode and (event.buttons() & Qt.MouseButton.LeftButton):
            pt = self._to_image(event.position())
            if pt and self._last_pt:
                cv2.line(self._canvas, self._last_pt, pt, 255, self.brush_size)
                self._refresh_pixmap()
            self._last_pt = pt
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.edit_mode:
            self._last_pt = None
        else:
            super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if self.edit_mode and event.key() == Qt.Key.Key_Z and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.undo()
        else:
            super().keyPressEvent(event)

    # ── helpers ─────────────────────────────────────────────────────

    def _to_image(self, qpointf) -> Optional[tuple]:
        """Map a view-space QPointF → integer image pixel (x, y)."""
        if self._canvas is None:
            return None
        scene_pt = self.mapToScene(int(qpointf.x()), int(qpointf.y()))
        h, w = self._canvas.shape[:2]
        x = int(np.clip(scene_pt.x(), 0, w - 1))
        y = int(np.clip(scene_pt.y(), 0, h - 1))
        return (x, y)

    def _refresh_pixmap(self):
        if self._canvas is None:
            return
        h, w = self._canvas.shape
        qimg = QImage(self._canvas.data, w, h, w, QImage.Format.Format_Grayscale8)
        pix = QPixmap.fromImage(qimg)
        if self._pixmap_item is None:
            self._pixmap_item = QGraphicsPixmapItem(pix)
            self._scene.addItem(self._pixmap_item)
        else:
            self._pixmap_item.setPixmap(pix)


# ══════════════════════════════════════════════════════════
#  TOAST NOTIFICATION
# ══════════════════════════════════════════════════════════

class ToastNotification(QFrame):
    def __init__(self, parent):
        super().__init__(parent)
        self.setFixedWidth(320)
        self.setFixedHeight(64)
        self.setObjectName("toast")
        self.setStyleSheet(f"""
            QFrame#toast {{
                background: {DARK['success']};
                border-radius: 10px;
                border: none;
            }}
            QLabel {{ color: white; background: transparent; border: none; }}
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 8, 16, 8)
        self.icon_label = QLabel("Config Saved")
        self.icon_label.setStyleSheet("font-weight: bold; font-size: 11pt;")
        self.msg_label = QLabel("")
        self.msg_label.setStyleSheet("font-size: 9pt;")
        layout.addWidget(self.icon_label)
        layout.addWidget(self.msg_label)
        self.hide()
        self._timer = QTimer()
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.hide)

    def show_message(self, title: str, detail: str):
        self.icon_label.setText(f"{title}")
        self.msg_label.setText(detail)
        self._position()
        self.show()
        self.raise_()
        self._timer.start(3500)

    def _position(self):
        if self.parent():
            pw = self.parent().width()
            ph = self.parent().height()
            self.move(20, ph - self.height() - 20)


# ══════════════════════════════════════════════════════════
#  VIEW 1 — MAP PARSER  (modified sections only)
# ══════════════════════════════════════════════════════════

PATCH_SIZE  = 256
MAX_DIRECT  = 2_250_000   # ~1500×1500 — mirrors predict_combined.py

# ── Drop-zone widget ────────────────────────────────────────────────────────

class _DropZone(QLabel):
    """
    Drag-and-drop target for image files.
    Emits file_dropped(str) with the resolved path.
    """
    file_dropped = pyqtSignal(str)

    _ACCEPTED = {".png", ".jpg", ".jpeg", ".bmp"}

    def __init__(self):
        super().__init__("⬆  Drop floorplan here")
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setFixedHeight(64)
        self._normal_ss  = (
            f"border: 2px dashed {DARK['border']}; border-radius: 8px; "
            f"color: {DARK['subtext']}; font-size: 9pt; background: {DARK['input_bg']};"
        )
        self._active_ss  = (
            f"border: 2px dashed {DARK['accent']}; border-radius: 8px; "
            f"color: {DARK['accent']}; font-size: 9pt; background: {DARK['card']};"
        )
        self.setStyleSheet(self._normal_ss)

    def dragEnterEvent(self, ev):
        if ev.mimeData().hasUrls():
            urls = ev.mimeData().urls()
            if any(Path(u.toLocalFile()).suffix.lower() in self._ACCEPTED
                   for u in urls):
                ev.acceptProposedAction()
                self.setStyleSheet(self._active_ss)
                return
        ev.ignore()

    def dragLeaveEvent(self, ev):
        self.setStyleSheet(self._normal_ss)

    def dropEvent(self, ev):
        self.setStyleSheet(self._normal_ss)
        for url in ev.mimeData().urls():
            path = url.toLocalFile()
            if Path(path).suffix.lower() in self._ACCEPTED:
                self.file_dropped.emit(path)
                return


# ── Pill toggle switch ───────────────────────────────────────────────────────

class _ToggleSwitch(QWidget):
    """
    iOS-style toggle switch.
    Exposes .isChecked() and .toggled(bool) to match QCheckBox API.
    """
    toggled = pyqtSignal(bool)

    _W, _H, _PAD = 44, 24, 3   # overall size and thumb padding

    def __init__(self, parent=None):
        super().__init__(parent)
        self._checked = False
        self.setFixedSize(self._W, self._H)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def isChecked(self) -> bool:
        return self._checked

    def setChecked(self, val: bool):
        if val != self._checked:
            self._checked = val
            self.update()
            self.toggled.emit(val)

    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            self.setChecked(not self._checked)

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H, PAD = self._W, self._H, self._PAD
        # track
        track_color = QColor(DARK["accent2"]) if self._checked else QColor(DARK["border"])
        p.setBrush(QBrush(track_color))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(0, 0, W, H, H / 2, H / 2)
        # thumb
        thumb_x = W - H + PAD if self._checked else PAD
        p.setBrush(QBrush(QColor("white")))
        p.drawEllipse(thumb_x, PAD, H - 2 * PAD, H - 2 * PAD)


def _run_predict_tiled(image_path, stride, max_patches, threshold, output_path,
                       progress_cb=None):
    """Generate a mask using overlapping, Gaussian-blended 256 px tiles."""
    import torch
    from model import UNet

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet()
    model.load_state_dict(torch.load("unet.pth", map_location=device))
    model.to(device)
    model.eval()

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]

    pad_y, pad_x = int(0.05 * height), int(0.05 * width)
    padded = cv2.copyMakeBorder(
        image, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_REFLECT_101)
    padded_height, padded_width = padded.shape[:2]

    def positions(dimension):
        # Include the final position so every source pixel is covered.
        last = max(0, dimension - PATCH_SIZE)
        values = list(range(0, last, stride))
        if not values or values[-1] != last:
            values.append(last)
        return values

    tiles = [(y, x) for y in positions(padded_height) for x in positions(padded_width)]
    if len(tiles) > max_patches:
        step = len(tiles) / max_patches
        tiles = [tiles[int(i * step)] for i in range(max_patches)]

    coords = np.linspace(-1, 1, PATCH_SIZE, dtype=np.float32)
    y_grid, x_grid = np.meshgrid(coords, coords, indexing="ij")
    weights = np.exp(-(x_grid ** 2 + y_grid ** 2) * 4).astype(np.float32)
    mask_sum = np.zeros((padded_height, padded_width), dtype=np.float32)
    weight_sum = np.zeros_like(mask_sum)

    for index, (y, x) in enumerate(tiles, start=1):
        source_tile = padded[y:y + PATCH_SIZE, x:x + PATCH_SIZE]
        actual_height, actual_width = source_tile.shape[:2]
        if actual_height < PATCH_SIZE or actual_width < PATCH_SIZE:
            source_tile = cv2.copyMakeBorder(
                source_tile, 0, PATCH_SIZE - actual_height, 0,
                PATCH_SIZE - actual_width, cv2.BORDER_REFLECT_101)

        tensor = source_tile.astype(np.float32) / 255.0
        tensor = torch.from_numpy(tensor.transpose(2, 0, 1)).unsqueeze(0).to(device)
        with torch.no_grad():
            prediction = torch.sigmoid(model(tensor)).squeeze().cpu().numpy()

        # Small images may make an edge tile smaller than PATCH_SIZE; only
        # blend the portion that corresponds to real padded-image pixels.
        prediction = prediction[:actual_height, :actual_width]
        tile_weights = weights[:actual_height, :actual_width]
        mask_sum[y:y + actual_height, x:x + actual_width] += prediction * tile_weights
        weight_sum[y:y + actual_height, x:x + actual_width] += tile_weights
        if progress_cb:
            progress_cb(int(index / len(tiles) * 85), f"Patch {index}/{len(tiles)}")

    blended = mask_sum / np.maximum(weight_sum, 1e-8)
    blended = blended[pad_y:pad_y + height, pad_x:pad_x + width]
    if progress_cb:
        progress_cb(88, "Cleaning mask…")

    binary = (blended > threshold).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.dilate(binary, np.ones((2, 2), np.uint8), iterations=1)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary)
    for label in range(1, count):
        area = stats[label, cv2.CC_STAT_AREA]
        component_width = stats[label, cv2.CC_STAT_WIDTH]
        component_height = stats[label, cv2.CC_STAT_HEIGHT]
        aspect_ratio = max(component_width, component_height) / (min(component_width, component_height) + 1e-6)
        if area >= 80 or aspect_ratio >= 3.0:
            cleaned[labels == label] = 1

    if not cv2.imwrite(output_path, cleaned * 255):
        raise OSError(f"Could not write mask to: {output_path}")
    if progress_cb:
        progress_cb(100, "Mask saved")


# ── new helper: run direct (no-tile) inference ──────────────────────────────
def _run_predict_direct(image_path, threshold, output_path, progress_cb=None):
    """
    Direct (no-tiling) inference — mirrors infer_direct() in predict_combined.py.
    Pads to multiples of 32, single forward pass, strips padding, binarizes.
    """
    import torch
    from model import UNet

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet()
    model.load_state_dict(torch.load("unet.pth", map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    if progress_cb:
        progress_cb(10, "Loading image…")

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W    = img_rgb.shape[:2]

    pad_h = (32 - H % 32) % 32
    pad_w = (32 - W % 32) % 32
    padded = cv2.copyMakeBorder(img_rgb, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)

    if progress_cb:
        progress_cb(30, "Running direct inference…")

    t = padded.astype(np.float32) / 255.0
    t = torch.from_numpy(t).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred = model(t)
    pred = torch.sigmoid(pred).squeeze().cpu().numpy()
    pred = pred[:H, :W]

    if progress_cb:
        progress_cb(80, "Cleaning mask…")

    binary = (pred > threshold).astype(np.uint8)
    k = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  k)
    binary = cv2.dilate(binary, np.ones((2, 2), np.uint8), iterations=1)

    # remove small text/symbol blobs
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary)
    min_area = (H * W) * 0.00005
    for i in range(1, num_labels):
        area  = stats[i, cv2.CC_STAT_AREA]
        w_box = stats[i, cv2.CC_STAT_WIDTH]
        h_box = stats[i, cv2.CC_STAT_HEIGHT]
        aspect = max(w_box, h_box) / (min(w_box, h_box) + 1e-6)
        if area < min_area and aspect < 3.0:
            continue
        cleaned[labels == i] = 1
    binary = cleaned

    cv2.imwrite(output_path, binary * 255)
    if progress_cb:
        progress_cb(100, "Mask saved")


class View1_MapParser(QWidget):
    proceed_signal = pyqtSignal()

    def __init__(self, state: AppState):
        super().__init__()
        self.state  = state
        self._worker: Optional[Worker] = None
        self._img_pixels: int = 0          # cached pixel count of loaded image
        self._build_ui()

    # ── UI CONSTRUCTION ────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(20)

        # ── Left: controls ──────────────────────────────
        left = QFrame(); left.setObjectName("card")
        left.setFixedWidth(340)
        lv = QVBoxLayout(left)
        lv.setContentsMargins(20, 20, 20, 20)
        lv.setSpacing(12)

        title = QLabel("Map Parser"); title.setObjectName("title")
        sub = QLabel("Parse a floorplan image into a walkability mask")
        sub.setObjectName("subtitle"); sub.setWordWrap(True)
        lv.addWidget(title)
        lv.addWidget(sub)
        lv.addWidget(self._sep())

        # ── Dropzone / upload ────────────────────────────
        lv.addWidget(self._section_label("INPUT IMAGE"))

        self.dropzone = _DropZone()
        self.dropzone.file_dropped.connect(self._on_file_chosen)
        lv.addWidget(self.dropzone)

        file_row = QHBoxLayout()
        self.file_label = QLabel("No file selected")
        self.file_label.setStyleSheet(f"color: {DARK['subtext']}; font-size: 9pt;")
        self.file_label.setWordWrap(True)
        self.browse_btn = QPushButton("Browse…")
        self.browse_btn.setFixedWidth(80)
        self.browse_btn.clicked.connect(self._browse)
        file_row.addWidget(self.file_label, 1)
        file_row.addWidget(self.browse_btn)
        lv.addLayout(file_row)

        self.res_label = QLabel("")
        self.res_label.setStyleSheet(f"color: {DARK['subtext']}; font-size: 8pt;")
        lv.addWidget(self.res_label)

        self.load_mask_btn = QPushButton("Load Existing Mask")
        self.load_mask_btn.clicked.connect(self._load_existing_mask)
        lv.addWidget(self.load_mask_btn)

        lv.addWidget(self._sep())

        # ── Force-tiled toggle ───────────────────────────
        toggle_row = QHBoxLayout()
        toggle_lbl = QLabel("Force Tiled Parsing (STITCH Framework)")
        toggle_lbl.setStyleSheet(f"color: {DARK['text']}; font-size: 9pt;")
        toggle_lbl.setWordWrap(True)

        self.stitch_toggle = _ToggleSwitch()
        self.stitch_toggle.toggled.connect(self._on_mode_changed)

        toggle_row.addWidget(toggle_lbl, 1)
        toggle_row.addWidget(self.stitch_toggle)
        lv.addLayout(toggle_row)

        # mode indicator badge
        self.mode_badge = QLabel("AUTO")
        self.mode_badge.setObjectName("badge")
        self.mode_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.mode_badge.setFixedWidth(90)
        lv.addWidget(self.mode_badge)

        lv.addWidget(self._sep())

        # ── Tiling parameters ────────────────────────────
        lv.addWidget(self._section_label("TILING PARAMETERS"))

        form = QFormLayout()
        form.setSpacing(8)

        self.stride_spin = QSpinBox()
        self.stride_spin.setRange(32, 256)
        self.stride_spin.setSingleStep(32)
        self.stride_spin.setValue(128)
        self.stride_spin.setToolTip(
            "Patch stride in pixels. Lower = more overlap, smoother seams, slower. "
            "Default 128 = 50% overlap on 256×256 patches.")
        self.stride_spin.valueChanged.connect(self._refresh_ideal_patches)
        form.addRow("Stride (px):", self.stride_spin)

        self.max_patches_spin = QSpinBox()
        self.max_patches_spin.setRange(1, 9999)
        self.max_patches_spin.setValue(40)
        self.max_patches_spin.setToolTip(
            "Auto-filled on image load with the ideal patch count for this "
            "image size and stride. Override manually to cap processing.")
        form.addRow("Max Patches:", self.max_patches_spin)

        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.1, 0.9)
        self.threshold_spin.setSingleStep(0.05)
        self.threshold_spin.setValue(0.50)
        self.threshold_spin.setDecimals(2)
        self.threshold_spin.setToolTip("Binarization threshold. Lower = more pixels classified as walls.")
        form.addRow("Threshold:", self.threshold_spin)

        lv.addLayout(form)

        # collect tiling widgets so we can enable/disable them together
        self._tiling_widgets = [
            self.stride_spin, self.max_patches_spin, self.threshold_spin,
        ]

        lv.addWidget(self._sep())

        self.run_btn = QPushButton("Run Parser")
        self.run_btn.setObjectName("primary")
        self.run_btn.setEnabled(False)
        self.run_btn.clicked.connect(self._run)
        lv.addWidget(self.run_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        lv.addWidget(self.progress_bar)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet(f"color: {DARK['subtext']}; font-size: 9pt;")
        self.status_label.setWordWrap(True)
        lv.addWidget(self.status_label)

        lv.addStretch()
        lv.addWidget(self._sep())

        btn_row = QHBoxLayout()
        self.tweak_btn = QPushButton("↺ Re-run")
        self.tweak_btn.setToolTip("Adjust the parameters above and click to re-run the parser.")
        self.tweak_btn.setVisible(False)
        self.tweak_btn.clicked.connect(self._run)

        self.debug_btn = QPushButton("🔍 Debug Mask")
        self.debug_btn.setToolTip("Gaussian blur + re-threshold to clean up the mask.")
        self.debug_btn.setVisible(False)
        self.debug_btn.clicked.connect(self._debug_mask)

        self.proceed_btn = QPushButton("Proceed to Zones →")
        self.proceed_btn.setObjectName("primary")
        self.proceed_btn.setVisible(False)
        self.proceed_btn.clicked.connect(self._proceed)

        btn_row.addWidget(self.tweak_btn)
        btn_row.addWidget(self.debug_btn)
        btn_row.addWidget(self.proceed_btn)
        lv.addLayout(btn_row)

        # ── Right: preview ───────────────────────────────
        right = QFrame(); right.setObjectName("panel")
        rv = QVBoxLayout(right)
        rv.setContentsMargins(16, 16, 16, 16)
        rv.setSpacing(8)

        preview_header = QHBoxLayout()
        self.preview_title = QLabel("Output Preview")
        self.preview_title.setStyleSheet("font-weight: bold; font-size: 11pt;")
        self.reset_zoom_btn = QPushButton("Fit")
        self.reset_zoom_btn.setFixedWidth(50)
        self.reset_zoom_btn.clicked.connect(lambda: self.preview.reset_zoom())

        self.compare_btn = QPushButton("⇄ Show Original")
        self.compare_btn.setCheckable(True)
        self.compare_btn.setVisible(False)
        self.compare_btn.setToolTip("Toggle between the binary mask and the original floorplan")
        self.compare_btn.clicked.connect(self._toggle_compare)
        self._showing_mask = True

        self.edit_btn = QPushButton("✏ Edit Mask")
        self.edit_btn.setCheckable(True)
        self.edit_btn.setVisible(False)
        self.edit_btn.clicked.connect(self._toggle_edit)

        self.brush_spin = QSpinBox()
        self.brush_spin.setRange(1, 40)
        self.brush_spin.setValue(6)
        self.brush_spin.setFixedWidth(55)
        self.brush_spin.setToolTip("Brush thickness in pixels")
        self.brush_spin.setVisible(False)
        self.brush_spin.valueChanged.connect(lambda v: setattr(self.preview, 'brush_size', v))

        self.save_edits_btn = QPushButton("💾 Save")
        self.save_edits_btn.setVisible(False)
        self.save_edits_btn.clicked.connect(self._save_edits)

        preview_header.addWidget(self.preview_title)
        preview_header.addStretch()
        preview_header.addWidget(self.compare_btn)
        preview_header.addWidget(self.edit_btn)
        preview_header.addWidget(self.brush_spin)
        preview_header.addWidget(self.save_edits_btn)
        preview_header.addWidget(self.reset_zoom_btn)
        rv.addLayout(preview_header)

        self.preview = EditableImageView(
            "Run the parser to see the output mask here.\n"
            "White = walls  |  Black = walkable space"
        )
        rv.addWidget(self.preview)

        self.preview_info = QLabel("")
        self.preview_info.setStyleSheet(f"color: {DARK['subtext']}; font-size: 9pt;")
        rv.addWidget(self.preview_info)

        root.addWidget(left)
        root.addWidget(right, 1)

        # initial state: no image loaded → apply correct enabled states
        self._apply_mode_state()

    # ── helpers ────────────────────────────────────────────────────────────────

    def _sep(self):
        f = QFrame(); f.setFrameShape(QFrame.Shape.HLine)
        f.setStyleSheet(f"color: {DARK['border']};")
        return f

    def _section_label(self, text):
        l = QLabel(text); l.setObjectName("section")
        return l

    def _use_tiled(self) -> bool:
        """True if tiled mode should be used (toggle forced ON, or image over threshold)."""
        return self.stitch_toggle.isChecked() or self._img_pixels > MAX_DIRECT

    def _apply_mode_state(self):
        """
        Enable / disable tiling parameter inputs and update the mode badge
        depending on image resolution and the force-tiled toggle.
        """
        tiled = self._use_tiled()
        for w in self._tiling_widgets:
            w.setEnabled(tiled)

        if not self.state.image_path:
            # no image yet — neutral badge
            self.mode_badge.setText("AUTO")
            self.mode_badge.setStyleSheet(
                f"background: {DARK['border']}; color: {DARK['subtext']}; "
                "border-radius: 10px; padding: 2px 10px; font-size: 9pt; font-weight: bold;"
            )
            return

        if self.stitch_toggle.isChecked():
            label, color = "STITCH (FORCED)", DARK["accent2"]
        elif self._img_pixels > MAX_DIRECT:
            label, color = "STITCH (AUTO)",   DARK["warning"]
        else:
            label, color = "DIRECT",          DARK["success"]

        self.mode_badge.setText(label)
        self.mode_badge.setStyleSheet(
            f"background: {color}; color: white; border-radius: 10px; "
            "padding: 2px 10px; font-size: 9pt; font-weight: bold;"
        )

    # ── signal handlers ────────────────────────────────────────────────────────

    def _on_file_chosen(self, path: str):
        """Central handler called from both the dropzone and Browse button."""
        if not path:
            return
        self.state.image_path = path
        short = Path(path).name
        self.file_label.setText(short)
        self.file_label.setStyleSheet(f"color: {DARK['text']}; font-size: 9pt;")
        self.run_btn.setEnabled(True)
        self.tweak_btn.setVisible(False)
        self.proceed_btn.setVisible(False)
        self.preview_info.setText("")

        # measure resolution
        img = cv2.imread(path)
        if img is not None:
            H, W = img.shape[:2]
            self._img_pixels = H * W
            self.res_label.setText(
                f"Resolution: {W}×{H}  ({self._img_pixels:,} px)  "
                f"— threshold {MAX_DIRECT:,} px"
            )
            # auto-fill ideal patch count
            ideal = self._compute_ideal_patches(path, self.stride_spin.value())
            self.max_patches_spin.setValue(ideal)
        else:
            self._img_pixels = 0
            self.res_label.setText("")

        self._apply_mode_state()
        self.preview.load_image(path)   # show original before parsing

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Floorplan Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp)")
        if path:
            self._on_file_chosen(path)

    def _on_mode_changed(self, _checked: bool):
        """Called when the STITCH toggle is flipped."""
        self._apply_mode_state()

    def _compute_ideal_patches(self, image_path, stride):
        img = cv2.imread(image_path)
        if img is None:
            return 40
        H, W = img.shape[:2]
        pad_h, pad_w = int(0.05 * H), int(0.05 * W)
        pH, pW = H + 2 * pad_h, W + 2 * pad_w

        def n_pos(dim):
            pos = list(range(0, dim - PATCH_SIZE, stride))
            if not pos or pos[-1] != dim - PATCH_SIZE:
                pos.append(max(0, dim - PATCH_SIZE))
            return len(pos)

        return max(1, n_pos(pH) * n_pos(pW))

    def _refresh_ideal_patches(self):
        if self.state.image_path:
            ideal = self._compute_ideal_patches(
                self.state.image_path, self.stride_spin.value())
            self.max_patches_spin.setValue(ideal)

    def _run(self):
        if not self.state.image_path:
            return
        if not Path("unet.pth").exists():
            reply = QMessageBox.question(
                self, "Model Weights Missing",
                "unet.pth not found.\n\n"
                "Download it now from HuggingFace? (~30 MB)\n"
                "https://huggingface.co/sankhya007/Floorplan_parser_STITCH\n\n"
                "Click Yes to auto-download, No to cancel.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
            self.run_btn.setEnabled(False)
            self.status_label.setText("Downloading unet.pth…")
            self.status_label.setStyleSheet(f"color: {DARK['warning']}; font-size: 9pt;")
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)

            def _download(progress_cb=None):
                import urllib.request
                url = (
                    "https://huggingface.co/sankhya007/Floorplan_parser_STITCH"
                    "/resolve/main/unet.pth"
                )
                dest = Path("unet.pth")
                with urllib.request.urlopen(url) as response:
                    total = int(response.headers.get("Content-Length", 0))
                    downloaded = 0
                    chunk = 1024 * 64
                    with open(dest, "wb") as f:
                        while True:
                            buf = response.read(chunk)
                            if not buf:
                                break
                            f.write(buf)
                            downloaded += len(buf)
                            if total and progress_cb:
                                progress_cb(int(downloaded / total * 100),
                                            f"Downloading… {downloaded // 1024 // 1024} MB")
                if progress_cb:
                    progress_cb(100, "Download complete")

            self._dl_worker = Worker(_download)
            self._dl_worker.progress.connect(self._on_progress)
            self._dl_worker.finished.connect(self._on_download_done)
            self._dl_worker.start()
            return

        self.run_btn.setEnabled(False)
        self.tweak_btn.setVisible(False)
        self.debug_btn.setVisible(False)
        self.edit_btn.setVisible(False)
        self.edit_btn.setChecked(False)
        self.brush_spin.setVisible(False)
        self.save_edits_btn.setVisible(False)
        self.proceed_btn.setVisible(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        output = str(Path(__file__).parent / "stitched_mask.png")
        self.state.mask_path = output

        # ── choose inference mode ────────────────────────────────────────────
        if self._use_tiled():
            mode_str = (
                "STITCH (forced)" if self.stitch_toggle.isChecked()
                else f"STITCH (image {self._img_pixels:,} px > threshold)"
            )
            self.status_label.setText(f"Mode: {mode_str}")
            self._worker = Worker(
                _run_predict_tiled,
                self.state.image_path,
                self.stride_spin.value(),
                self.max_patches_spin.value(),
                self.threshold_spin.value(),
                output,
            )
        else:
            self.status_label.setText("Mode: Direct (single-pass inference)")
            self._worker = Worker(
                _run_predict_direct,
                self.state.image_path,
                self.threshold_spin.value(),
                output,
            )

        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_done)
        self._worker.start()


    def _on_progress(self, pct, msg):
        self.progress_bar.setValue(pct)
        self.status_label.setText(msg)

    def _on_done(self, success, msg):
        self.progress_bar.setVisible(False)
        self.run_btn.setEnabled(True)
        if success:
            self.status_label.setText("Mask generated successfully")
            self.status_label.setStyleSheet(f"color: {DARK['success']}; font-size: 9pt;")
            self.preview.load_canvas(self.state.mask_path)
            img = cv2.imread(self.state.mask_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                white = np.sum(img > 127)
                black = np.sum(img <= 127)
                total = img.size
                self.preview_info.setText(
                    f"Size: {img.shape[1]}×{img.shape[0]}px  |  "
                    f"Walls: {100*white//total}%  |  Walkable: {100*black//total}%"
                )
            self.tweak_btn.setVisible(True)
            self.debug_btn.setVisible(True)
            self.compare_btn.setVisible(True)
            self.compare_btn.setChecked(False)
            self.compare_btn.setText("⇄ Show Original")
            self._showing_mask = True
            self.preview_title.setText("Output Preview — Mask")
            self.edit_btn.setVisible(True)
            self.brush_spin.setVisible(True)
            self.save_edits_btn.setVisible(True)
            self.proceed_btn.setVisible(True)
        else:
            self.status_label.setText(f"Error: {msg}")
            self.status_label.setStyleSheet(f"color: {DARK['danger']}; font-size: 9pt;")

    def _debug_mask(self):
        if not self.state.mask_path or not Path(self.state.mask_path).exists():
            return

        img = cv2.imread(self.state.mask_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return

        # Gaussian blur — smooths salt-and-pepper noise and fills small gaps
        blurred = cv2.GaussianBlur(img.astype(np.float32) / 255.0, (0, 0), sigmaX=2.0)

        # Re-threshold at 0.5 on the blurred float image
        binary = (blurred > 0.5).astype(np.uint8)

        # Same morphological cleanup as the tiled predictor
        k = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)
        binary = cv2.dilate(binary, np.ones((2, 2), np.uint8), iterations=1)

        cv2.imwrite(self.state.mask_path, binary * 255)

        self.preview.load_canvas(self.state.mask_path)
        white = int((binary > 0).sum())
        total = binary.size
        self.preview_info.setText(
            f"Size: {binary.shape[1]}×{binary.shape[0]}px  |  "
            f"Walls: {100*white//total}%  |  Walkable: {100*(total-white)//total}%  "
            f"[debug applied]"
        )
        self.status_label.setText("Debug mask applied — proceed when ready")
        self.status_label.setStyleSheet(f"color: {DARK['warning']}; font-size: 9pt;")

    def _toggle_compare(self, checked):
        """Switch the preview between the generated mask and source floorplan."""
        if checked:
            if not self.state.image_path or not Path(self.state.image_path).exists():
                # A mask loaded without its source image cannot be compared.
                self.compare_btn.setChecked(False)
                return
            self.preview.load_image(self.state.image_path)
            self.compare_btn.setText("⇄ Show Mask")
            self.preview_title.setText("Output Preview — Original")
            self._showing_mask = False
            self.edit_btn.setEnabled(False)
            self.save_edits_btn.setEnabled(False)
        else:
            if self.state.mask_path and Path(self.state.mask_path).exists():
                self.preview.load_canvas(self.state.mask_path)
            self.compare_btn.setText("⇄ Show Original")
            self.preview_title.setText("Output Preview — Mask")
            self._showing_mask = True
            self.edit_btn.setEnabled(True)
            self.save_edits_btn.setEnabled(True)

    def _toggle_edit(self, checked):
        self.preview.edit_mode = checked
        if checked:
            # if the user is viewing the original, snap back to mask before editing
            if not self._showing_mask:
                self.compare_btn.setChecked(False)
                self._toggle_compare(False)
            self.edit_btn.setText("✏ Editing...")
            self.edit_btn.setStyleSheet(f"background: {DARK['warning']}; color: black; font-weight: bold;")
            self.preview.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.preview.setFocus()
            self.status_label.setText("Draw mode: click and drag to paint walls. Ctrl+Z to undo. Save when done.")
            self.status_label.setStyleSheet(f"color: {DARK['warning']}; font-size: 9pt;")
        else:
            self.edit_btn.setText("✏ Edit Mask")
            self.edit_btn.setStyleSheet("")
            self.preview.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.status_label.setText("Edit mode off.")
            self.status_label.setStyleSheet(f"color: {DARK['subtext']}; font-size: 9pt;")

    def _save_edits(self):
        canvas = self.preview.get_canvas()
        if canvas is None or not self.state.mask_path:
            return
        cv2.imwrite(self.state.mask_path, canvas)
        self.status_label.setText("Edits saved — proceed when ready")
        self.status_label.setStyleSheet(f"color: {DARK['success']}; font-size: 9pt;")

    def _on_download_done(self, success, msg):
        self.progress_bar.setVisible(False)
        self.run_btn.setEnabled(True)
        if success:
            self.status_label.setText("Download complete — running parser now")
            self.status_label.setStyleSheet(f"color: {DARK['success']}; font-size: 9pt;")
            self._run()   # kick off the parse immediately
        else:
            self.status_label.setText(f"Download failed: {msg}")
            self.status_label.setStyleSheet(f"color: {DARK['danger']}; font-size: 9pt;")
            QMessageBox.warning(self, "Download Failed",
                f"Could not download unet.pth:\n{msg}\n\n"
                "Download it manually from:\n"
                "https://huggingface.co/sankhya007/Floorplan_parser_STITCH")

    def _load_existing_mask(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Existing Mask or Zone Config", "",
            "All supported (*.png *.jpg *.bmp *.json);;Images (*.png *.jpg *.bmp);;Zone Config (*.json)")
        if not path:
            return

        # JSON → hand off to View 2 and jump straight there
        if path.lower().endswith(".json"):
            self.window().view2._load_config_from_path(path)
            self.window()._go_to(1)
            return

        self.state.mask_path = path
        self.preview.load_canvas(path)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            white = np.sum(img > 127)
            black = np.sum(img <= 127)
            total = img.size
            self.preview_info.setText(
                f"Size: {img.shape[1]}x{img.shape[0]}px  |  "
                f"Walls: {100*white//total}%  |  Walkable: {100*black//total}%"
            )
        self.status_label.setText(f"Loaded: {Path(path).name}")
        self.status_label.setStyleSheet(f"color: {DARK['success']}; font-size: 9pt;")
        self.tweak_btn.setVisible(False)
        self.debug_btn.setVisible(True)
        self.edit_btn.setVisible(True)
        self.brush_spin.setVisible(True)
        self.save_edits_btn.setVisible(True)
        self.proceed_btn.setVisible(True)

    def _proceed(self):
        self.proceed_signal.emit()

    def restore_with_image(self):
        """Called when coming back from tweak — preserves loaded image."""
        if self.state.image_path:
            self.file_label.setText(Path(self.state.image_path).name)
            self.run_btn.setEnabled(True)
        if self.state.mask_path and Path(self.state.mask_path).exists():
            self.preview.load_image(self.state.mask_path)


# ══════════════════════════════════════════════════════════
#  VIEW 2 — ZONE EDITOR  (wraps existing ZoneEditor logic)
# ══════════════════════════════════════════════════════════

class View2_ZoneEditor(QWidget):
    proceed_signal = pyqtSignal()

    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self.binary = None
        self.labels = None
        self.valid_zones = []
        self.zone_stats = {}
        self.color_map = {}
        self.density_map = {}
        self.highlight = None
        self.highlight_set = set()   # multi-zone selection (shift-click)
        self.exits = []   # list of {"x": int, "y": int}
        self.hazard = None   # single hazard: {"x": int, "y": int} or None
        self.exit_mode = False
        self.hazard_mode = False
        self._build_ui()

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(20)

        # ── Left panel ──────────────────────────────────
        left = QFrame(); left.setObjectName("card")
        left.setFixedWidth(250)
        lv = QVBoxLayout(left)
        lv.setContentsMargins(20, 20, 20, 20)
        lv.setSpacing(12)

        title = QLabel("Zone Editor"); title.setObjectName("title")
        sub = QLabel("Segment walkable zones and assign agent density")
        sub.setObjectName("subtitle"); sub.setWordWrap(True)
        lv.addWidget(title)
        lv.addWidget(sub)
        lv.addWidget(self._sep())

        self.load_mask_btn = QPushButton("Load Mask")
        self.load_mask_btn.clicked.connect(self._load_mask)
        lv.addWidget(self.load_mask_btn)

        self.load_config_btn = QPushButton("Load Existing Config")
        self.load_config_btn.clicked.connect(self._load_config)
        lv.addWidget(self.load_config_btn)

        # self.auto_load_label = QLabel("")
        # self.auto_load_label.setStyleSheet(f"color: {DARK['success']}; font-size: 8pt;")
        # lv.addWidget(self.auto_load_label)
        
        self.auto_load_label = QLabel("")
        self.auto_load_label.setStyleSheet(
            f"color: {DARK['success']}; font-size: 10pt; font-weight: bold;")
        self.auto_load_label.setWordWrap(True)
        lv.addWidget(self.auto_load_label)

        lv.addWidget(self._sep())
        lv.addWidget(self._section_label("BASE DENSITY"))

        density_form = QFormLayout()
        self.base_spin = QDoubleSpinBox()
        self.base_spin.setRange(0.01, 20.0)
        self.base_spin.setValue(0.5)
        self.base_spin.setSingleStep(0.05)
        self.base_spin.setDecimals(2)
        self.base_spin.setToolTip("Agents per 1000px² at density index = 1. Use small values (0.1–0.5) for large DXF masks.")
        density_form.addRow("Agents / 1000px²:", self.base_spin)
        lv.addLayout(density_form)

        lv.addWidget(self._sep())
        lv.addWidget(self._section_label("SELECTED ZONE"))

        self.zone_info = QLabel("Click a zone on the map\nShift+click to select multiple")
        self.zone_info.setStyleSheet(f"color: {DARK['subtext']}; font-size: 9pt;")
        self.zone_info.setWordWrap(True)
        lv.addWidget(self.zone_info)

        density_row = QHBoxLayout()
        density_row.addWidget(QLabel("Density Index:"))
        self.zone_density_spin = QDoubleSpinBox()
        self.zone_density_spin.setRange(0.0, 10.0)
        self.zone_density_spin.setValue(1.0)
        self.zone_density_spin.setSingleStep(0.5)
        self.zone_density_spin.setDecimals(1)
        self.zone_density_spin.setToolTip("0 = outside / ignore this zone")
        density_row.addWidget(self.zone_density_spin)
        lv.addLayout(density_row)

        self.apply_btn = QPushButton("Apply to Zone")
        self.apply_btn.setEnabled(False)
        self.apply_btn.clicked.connect(self._apply_density)
        lv.addWidget(self.apply_btn)

        lv.addWidget(self._sep())
        lv.addWidget(self._section_label("ZONE LIST"))

        zone_scroll = QScrollArea()
        zone_scroll.setWidgetResizable(True)
        zone_scroll.setFixedHeight(160)
        self.zone_list_widget = QWidget()
        self.zone_list_layout = QVBoxLayout(self.zone_list_widget)
        self.zone_list_layout.setSpacing(2)
        self.zone_list_layout.setContentsMargins(0, 0, 0, 0)
        zone_scroll.setWidget(self.zone_list_widget)
        lv.addWidget(zone_scroll)

        # exit placement block 
        lv.addWidget(self._sep())
        lv.addWidget(self._section_label("EXIT PLACEMENT"))

        self.exit_mode_btn = QPushButton("Place Exits")
        self.exit_mode_btn.setCheckable(True)
        self.exit_mode_btn.setEnabled(False)
        self.exit_mode_btn.clicked.connect(self._toggle_exit_mode)
        lv.addWidget(self.exit_mode_btn)

        self.exit_info_label = QLabel("No exits placed yet.")
        self.exit_info_label.setStyleSheet(f"color: {DARK['subtext']}; font-size: 9pt;")
        self.exit_info_label.setWordWrap(True)
        lv.addWidget(self.exit_info_label)

        clear_exits_btn = QPushButton("Clear Exits")
        clear_exits_btn.clicked.connect(self._clear_exits)
        lv.addWidget(clear_exits_btn)

        # hazard placement block
        lv.addWidget(self._sep())
        lv.addWidget(self._section_label("HAZARD PLACEMENT"))

        self.hazard_mode_btn = QPushButton("Place Hazard")
        self.hazard_mode_btn.setCheckable(True)
        self.hazard_mode_btn.setEnabled(False)
        self.hazard_mode_btn.clicked.connect(self._toggle_hazard_mode)
        lv.addWidget(self.hazard_mode_btn)

        self.hazard_info_label = QLabel("No hazard placed yet.")
        self.hazard_info_label.setStyleSheet(f"color: {DARK['subtext']}; font-size: 9pt;")
        self.hazard_info_label.setWordWrap(True)
        lv.addWidget(self.hazard_info_label)

        clear_hazard_btn = QPushButton("Clear Hazard")
        clear_hazard_btn.clicked.connect(self._clear_hazard)
        lv.addWidget(clear_hazard_btn)

        lv.addWidget(self._sep())
        lv.addWidget(self._section_label("SAVE CONFIG"))

        filename_row = QHBoxLayout()
        filename_row.addWidget(QLabel("Filename:"))
        self.filename_input = QLineEdit("zone_config")
        self.filename_input.setPlaceholderText("zone_config")
        filename_row.addWidget(self.filename_input)
        lv.addLayout(filename_row)

        self.save_btn = QPushButton("Save Config")
        self.save_btn.setObjectName("primary")
        self.save_btn.clicked.connect(self._save_config)
        lv.addWidget(self.save_btn)

        lv.addStretch()
        lv.addWidget(self._sep())
        
        back_btn2 = QPushButton("← Back to Parser")
        back_btn2.clicked.connect(self._go_back)
        lv.addWidget(back_btn2)

        self.proceed_btn = QPushButton("Proceed to Simulation →")
        self.proceed_btn.setObjectName("primary")
        self.proceed_btn.setEnabled(False)
        self.proceed_btn.clicked.connect(self.proceed_signal.emit)
        lv.addWidget(self.proceed_btn)

        # ── Right: map view ──────────────────────────────
        right = QFrame(); right.setObjectName("panel")
        rv = QVBoxLayout(right)
        rv.setContentsMargins(16, 16, 16, 16)
        rv.setSpacing(8)

        map_header = QHBoxLayout()
        map_title = QLabel("Zone Map")
        map_title.setStyleSheet("font-weight: bold; font-size: 11pt;")
        self.zone_count_label = QLabel("")
        self.zone_count_label.setStyleSheet(f"color: {DARK['subtext']}; font-size: 9pt;")
        map_header.addWidget(map_title)
        map_header.addStretch()
        map_header.addWidget(self.zone_count_label)
        rv.addLayout(map_header)

        # Create a custom map widget that supports hazard mode
        from zone_detector import ZoneMapWidget as ZoneMapWidgetClass
        self.map_label = ZoneMapWidgetClass()
        self.map_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.map_label.setStyleSheet(f"""
            background: {DARK['input_bg']};
            border: 1px solid {DARK['border']};
            border-radius: 8px;
            color: {DARK['subtext']};
            font-size: 12pt;
        """)
        self.map_label.setText("Load a mask to see zone segmentation")
        self.map_label.setMinimumHeight(400)
        self.map_label.setMouseTracking(True)
        self.map_label.mousePressEvent = self._map_click
        rv.addWidget(self.map_label, 1)

        # Wrap left panel in scroll area
        scroll = QScrollArea()
        scroll.setWidget(left)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet(f"""
            QScrollArea {{
                background: transparent;
                border: none;
            }}
            QScrollBar:vertical {{
                background: {DARK['panel']};
                width: 12px;
                border-radius: 6px;
            }}
            QScrollBar::handle:vertical {{
                background: {DARK['border']};
                border-radius: 6px;
                min-height: 20px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: {DARK['accent']};
            }}
        """)
        
        root.addWidget(scroll)
        root.addWidget(right, 1)

        # Toast
        self._toast = ToastNotification(self)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_map()
        if self._toast:
            self._toast._position()

    def _sep(self):
        f = QFrame(); f.setFrameShape(QFrame.Shape.HLine)
        f.setStyleSheet(f"color: {DARK['border']};")
        return f

    def _section_label(self, text):
        l = QLabel(text); l.setObjectName("section")
        return l

    def on_enter(self):
        if self.binary is None and self.state.mask_path and Path(self.state.mask_path).exists():
            self._load_mask_from_path(self.state.mask_path)
            self.auto_load_label.setText(f"Auto-loaded: {Path(self.state.mask_path).name}")

    def _load_mask(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Mask or Config", "",
            "All supported (*.png *.jpg *.bmp *.json);;Images (*.png *.jpg *.bmp);;JSON Config (*.json)")
        if path:
            if path.endswith(".json"):
                self._load_config_from_path(path)
            else:
                self._load_mask_from_path(path)
                
    def _go_back(self):
        # Navigate to the nav bar's step 0 via the main window
        self.window()._go_to(0)

    def _load_mask_from_path(self, path: str):
        try:
            from scipy import ndimage as ndi
            from skimage.segmentation import watershed
            from skimage.feature import peak_local_max
        except ImportError:
            QMessageBox.critical(self, "Missing dependency",
                "scipy and scikit-image are required.\n"
                "Run: pip install scipy scikit-image")
            return

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return

        # White = wall, black = walkable
        walkable = cv2.bitwise_not(img)
        _, binary = cv2.threshold(walkable, 127, 255, cv2.THRESH_BINARY)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)

        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        dist_norm = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)
        # Must match the hardcoded min_distance=40 used in all four sim scripts
        # so that zone IDs produced here are the same ones the sim scripts produce
        coords = peak_local_max(dist_norm, min_distance=40, labels=binary)
        seed_mask = np.zeros(dist_norm.shape, dtype=bool)
        seed_mask[tuple(coords.T)] = True
        markers, _ = ndi.label(seed_mask)
        labels = watershed(-dist, markers, mask=binary)

        h_img, w_img = binary.shape
        min_area = 800  # matches the original zone_detector.py threshold
        valid_zones, zone_stats = [], {}
        for zid in np.unique(labels):
            if zid == 0:
                continue
            area = int(np.sum(labels == zid))
            if area >= min_area:
                valid_zones.append(int(zid))
                zone_stats[int(zid)] = area

        self.binary = binary
        self.labels = labels
        self.valid_zones = valid_zones
        self.zone_stats = zone_stats
        self.color_map = self._build_colors(valid_zones)
        self.density_map = {zid: 1.0 for zid in valid_zones}
        self.highlight = None
        self.highlight_set = set()

        # Build and cache the base zone image once — _refresh_map copies it instead of recomputing
        self._base_zone_img = self._build_base_zone_img()

        self.zone_info.setText(f"{len(valid_zones)} zones detected.\nClick any zone to set its density.")
        self._refresh_map()
        
        # zone selection using mouse click 
        self._rebuild_zone_list()
        self.exit_mode_btn.setEnabled(True)
        self.hazard_mode_btn.setEnabled(True)
        self.exits = []
        self.hazard = None
        self.exit_mode = False
        self.hazard_mode = False
        self.exit_mode_btn.setChecked(False)
        self.hazard_mode_btn.setChecked(False)
        self._update_exit_info()
        self._update_hazard_info() 

    def _load_config(self):
        """Load and edit existing zone config JSON."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Zone Config", "", "JSON Files (*.json)")
        if path:
            self._load_config_from_path(path)

    def _load_config_from_path(self, path: str):
        """Parse zone config JSON and populate UI state."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                config = json.load(f)
        except Exception as e:
            QMessageBox.critical(self, "Error loading config", str(e))
            return

        # Validate that mask exists and load it
        mask_path = config.get("mask_path")
        if not mask_path or not Path(mask_path).exists():
            QMessageBox.warning(self, "Mask not found",
                f"Mask path not found: {mask_path}")
            return

        # Load the mask first
        self._load_mask_from_path(mask_path)

        # Now populate config data into the UI
        self.base_spin.setValue(config.get("base_density", 1.0))
        # Default all zones to 0 — only set density for zones explicitly listed in config
        self.density_map = {zid: 0.0 for zid in self.valid_zones}
        for zone_data in config.get("zones", []):
            zid = zone_data.get("zone_id")
            d = zone_data.get("density_index", 0.0)
            if zid in self.valid_zones:
                self.density_map[zid] = d

        # Rebuild base image now that density_map is correct
        self._base_zone_img = self._build_base_zone_img()

        self.exits = config.get("exits", [])
        self.hazard = config.get("hazard", None)
        self.state.zone_config_path = path
        self.state.hazard = self.hazard if self.hazard else {}

        self.hazard_mode_btn.setEnabled(True)
        self._rebuild_zone_list()
        self._update_exit_info()
        self._update_hazard_info()
        self._refresh_map()
        self.auto_load_label.setText(f"Loaded: {Path(path).name}")
        self.proceed_btn.setEnabled(True)

    def _build_base_zone_img(self):
        """Build the full-res coloured zone image once. Called at load time and when density changes."""
        if self.binary is None:
            return None
        h, w = self.binary.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for zid in self.valid_zones:
            d = self.density_map.get(zid, 1.0)
            rgb[self.labels == zid] = (80, 80, 80) if d == 0 else self.color_map[zid]
        rgb[self.binary == 0] = (20, 20, 20)
        # Draw zone labels (centroid text) into the base image
        for i, zid in enumerate(self.valid_zones):
            ys, xs = np.where(self.labels == zid)
            if len(xs) == 0:
                continue
            cx, cy = int(xs.mean()), int(ys.mean())
            d = self.density_map.get(zid, 1.0)
            cv2.putText(rgb, f"{i}:{d:.1f}", (cx - 15, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        return rgb

    def _build_colors(self, zones):
        colors = {}
        for i, zid in enumerate(zones):
            hue = int(179 * i / max(len(zones), 1))
            hsv = np.uint8([[[hue, 190, 170]]])
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
            colors[zid] = (int(bgr[2]), int(bgr[1]), int(bgr[0]))
        return colors

    def _refresh_map(self):
        if self.binary is None:
            return
        h, w = self.binary.shape

        # Use cached base image — copy it so we can draw overlays without dirtying the cache
        base = getattr(self, '_base_zone_img', None)
        if base is None:
            base = self._build_base_zone_img()
            self._base_zone_img = base
        rgb = base.copy()

        # Marker size scales with image so they're visible on big DXF masks
        marker_r = max(10, int(max(h, w) / 80))
        font_scale = max(0.4, marker_r / 25)
        text_off = max(8, marker_r // 2)

        # Selection outlines
        if self.highlight is not None:
            zm = (self.labels == self.highlight).astype(np.uint8) * 255
            contours, _ = cv2.findContours(zm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(rgb, contours, -1, (255, 255, 0), max(2, marker_r // 5))
        for sel_zid in self.highlight_set:
            if sel_zid != self.highlight:
                zm2 = (self.labels == sel_zid).astype(np.uint8) * 255
                c2, _ = cv2.findContours(zm2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(rgb, c2, -1, (255, 165, 0), max(2, marker_r // 6))

        # Draw exits
        for i, ex in enumerate(self.exits):
            ex_x, ex_y = int(ex["x"]), int(ex["y"])
            cv2.circle(rgb, (ex_x, ex_y), marker_r, (0, 220, 80), -1)
            cv2.putText(rgb, f"E{i+1}", (ex_x - text_off, ex_y + text_off // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), max(1, marker_r // 10))

        # Draw hazard — scale the no-go radius circle to image size too
        if self.hazard:
            hx, hy = int(self.hazard["x"]), int(self.hazard["y"])
            hazard_display_r = max(30, int(max(h, w) * HAZARD_BLOCK_RADIUS / 1000))
            cv2.circle(rgb, (hx, hy), hazard_display_r, (0, 140, 255), max(2, marker_r // 8))
            cv2.circle(rgb, (hx, hy), marker_r, (0, 0, 255), -1)
            cv2.circle(rgb, (hx, hy), marker_r + 3, (0, 0, 200), max(2, marker_r // 8))
            cv2.putText(rgb, "H", (hx - text_off // 2, hy + text_off // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), max(1, marker_r // 8))

        qimg = QImage(rgb.tobytes(), w, h, w * 3, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        lw = self.map_label.width() - 4
        lh = self.map_label.height() - 4
        if lw > 0 and lh > 0:
            pix = pix.scaled(lw, lh, Qt.AspectRatioMode.KeepAspectRatio,
                             Qt.TransformationMode.SmoothTransformation)
        self.map_label.setPixmap(pix)
        self._map_scale = (self.binary.shape[1] / pix.width(),
                           self.binary.shape[0] / pix.height()) if pix.width() > 0 else (1, 1)
        self._map_offset = ((self.map_label.width() - pix.width()) // 2,
                            (self.map_label.height() - pix.height()) // 2)

    def _map_click(self, event): # modified to handle both zone selection and exit placement
        if self.labels is None:
            return
        ox, oy = self._map_offset if hasattr(self, '_map_offset') else (0, 0)
        sx, sy = self._map_scale if hasattr(self, '_map_scale') else (1, 1)
        px = int((event.position().x() - ox) * sx)
        py = int((event.position().y() - oy) * sy)
        px = np.clip(px, 0, self.labels.shape[1] - 1)
        py = np.clip(py, 0, self.labels.shape[0] - 1)

        # HAZARD MODE — place or remove hazard
        if self.hazard_mode:
            self._map_hazard_click(px, py)
            return

        # EXIT MODE — place or remove exits
        if self.exit_mode:
            SNAP = 30
            for i, ex in enumerate(self.exits):
                if abs(ex["x"] - px) < SNAP and abs(ex["y"] - py) < SNAP:
                    self.exits.pop(i)
                    self._update_exit_info()
                    self._refresh_map()
                    return
            self.exits.append({"x": int(px), "y": int(py)})
            self._update_exit_info()
            self._refresh_map()
            return

        # zone selection — shift-click adds to selection, normal click resets
        zid = int(self.labels[py, px])
        if zid not in self.valid_zones:
            return

        shift_held = event.modifiers() & Qt.KeyboardModifier.ShiftModifier
        if shift_held:
            # Toggle this zone in the multi-select set
            if zid in self.highlight_set:
                self.highlight_set.discard(zid)
                if self.highlight == zid:
                    self.highlight = next(iter(self.highlight_set), None)
            else:
                self.highlight_set.add(zid)
                self.highlight = zid
        else:
            self.highlight = zid
            self.highlight_set = {zid}

        if self.highlight is not None:
            idx = self.valid_zones.index(self.highlight)
            area = self.zone_stats[self.highlight]
            d = self.density_map.get(self.highlight, 1.0)
            agents = int(area * d * self.base_spin.value() / 1000)
            multi_note = f"\n(+{len(self.highlight_set)-1} more selected)" if len(self.highlight_set) > 1 else ""
            self.zone_info.setText(
                f"Zone {idx}  (id={self.highlight})\n"
                f"Area: {area:,} px²\n"
                f"Density: {d:.1f}  ->  ~{agents} agents{multi_note}")
            self.zone_density_spin.setValue(d)
        self.apply_btn.setEnabled(bool(self.highlight_set))
        self._refresh_map()

    def _apply_density(self):
        if not self.highlight_set:
            return
        val = self.zone_density_spin.value()
        for zid in self.highlight_set:
            self.density_map[zid] = val
        if self.highlight is not None:
            self._map_click_refresh()
        self._rebuild_zone_list()
        # Density changed → rebuild cached base image
        self._base_zone_img = self._build_base_zone_img()
        self._refresh_map()

    def _map_click_refresh(self):
        if self.highlight is None:
            return
        zid = self.highlight
        idx = self.valid_zones.index(zid)
        area = self.zone_stats[zid]
        d = self.density_map.get(zid, 1.0)
        agents = int(area * d * self.base_spin.value() / 1000)
        self.zone_info.setText(
            f"Zone {idx}  (id={zid})\n"
            f"Area: {area:,} px²\n"
            f"Density: {d:.1f}  ->  ~{agents} agents")

    def _rebuild_zone_list(self):
        while self.zone_list_layout.count():
            item = self.zone_list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for i, zid in enumerate(self.valid_zones):
            d = self.density_map.get(zid, 1.0)
            area = self.zone_stats[zid]
            c = self.color_map[zid]
            label_text = "outside" if d == 0 else f"d={d:.1f}"
            lbl = QLabel(f"  Zone {i:2d} | {label_text:8s} | {area//1000}k px²")
            brightness = sum(c)
            text_color = "#111" if brightness > 400 else "#eee"
            lbl.setStyleSheet(
                f"background: rgb({c[0]},{c[1]},{c[2]});"
                f"color: {text_color};"
                f"border-radius: 3px; padding: 2px 4px; font-size: 9pt;")
            lbl.setFixedHeight(22)
            self.zone_list_layout.addWidget(lbl)
        self.zone_list_layout.addStretch()

    # three blocks underneath are for exit placement and management
    def _toggle_exit_mode(self, checked):
        self.exit_mode = checked
        if checked:
            self.hazard_mode = False
            self.hazard_mode_btn.setChecked(False)
            self.exit_mode_btn.setText("Exit Mode ON (click map)")
            self.exit_mode_btn.setStyleSheet(f"background: {DARK['success']}; color: white; font-weight: bold;")
        else:
            self.exit_mode_btn.setText("Place Exits")
            self.exit_mode_btn.setStyleSheet("")
        self._refresh_map()

    def _clear_exits(self):
        self.exits = []
        self._update_exit_info()
        self._refresh_map()

    def _update_exit_info(self):
        n = len(self.exits)
        if n == 0:
            self.exit_info_label.setText("No exits placed yet.")
        else:
            self.exit_info_label.setText(f"{n} exit(s) placed. Click near one to remove it.")

    # ── Hazard mode ──────────────────────────────────────────────
    def _toggle_hazard_mode(self, checked):
        self.hazard_mode = checked
        if checked:
            self.exit_mode = False
            self.exit_mode_btn.setChecked(False)
            self.hazard_mode_btn.setText("Hazard Mode ON (click map)")
            self.hazard_mode_btn.setStyleSheet(f"background: {DARK['danger']}; color: white; font-weight: bold;")
        else:
            self.hazard_mode_btn.setText("Place Hazard")
            self.hazard_mode_btn.setStyleSheet("")
        self._refresh_map()

    def _map_hazard_click(self, orig_x: int, orig_y: int):
        """Handle hazard placement/removal on map click."""
        if self.hazard:
            dist = ((self.hazard["x"] - orig_x) ** 2 + (self.hazard["y"] - orig_y) ** 2) ** 0.5
            if dist <= 40:  # HAZARD_SNAP_DIST
                self.hazard = None
                self._update_hazard_info()
                self._refresh_map()
                return
        self.hazard = {"x": orig_x, "y": orig_y}
        self._update_hazard_info()
        self._refresh_map()

    def _clear_hazard(self):
        if not self.hazard:
            return
        self.hazard = None
        self._update_hazard_info()
        self._refresh_map()

    def _update_hazard_info(self):
        if not self.hazard:
            self.hazard_info_label.setText("No hazard placed yet.")
        else:
            h = self.hazard
            self.hazard_info_label.setText(f"Hazard at ({h['x']}, {h['y']}). Click near to remove.")

    def _save_config(self):
        if not self.valid_zones:
            QMessageBox.warning(self, "No zones", "Load a mask and detect zones first.")
            return
        name = self.filename_input.text().strip() or "zone_config"
        if not name.endswith(".json"):
            name += ".json"
        path, _ = QFileDialog.getSaveFileName(self, "Save Zone Config", name, "JSON (*.json)")
        if not path:
            return

        def _normalize_json_value(val):
            if isinstance(val, np.integer):
                return int(val)
            if isinstance(val, np.floating):
                return float(val)
            if isinstance(val, dict):
                return {k: _normalize_json_value(v) for k, v in val.items()}
            if isinstance(val, list):
                return [_normalize_json_value(v) for v in val]
            return val

        config = {
            "mask_path": self.state.mask_path,
            "base_density": float(self.base_spin.value()),
            "agent_scale": 1000,
            "exits": self.exits,
            "hazard": self.hazard,
            "zones": []
        }
        for i, zid in enumerate(self.valid_zones):
            area = int(self.zone_stats[zid])
            d = float(self.density_map.get(zid, 0.0))
            agents = int(area * d * float(self.base_spin.value()) / 1000)
            config["zones"].append({
                "zone_index": int(i),
                "zone_id": int(zid),
                "area_px": area,
                "density_index": d,
                "agents": agents,
            })
        config = _normalize_json_value(config)
        with open(path, "w") as f:
            json.dump(config, f, indent=2)

        # persist hazard into shared AppState for other views
        self.state.zone_config_path = path
        try:
            self.state.hazard = config.get("hazard", {})
        except Exception:
            self.state.hazard = {}
        total = sum(z["agents"] for z in config["zones"] if z["density_index"] > 0)
        short_name = Path(path).name
        self._toast.show_message(
            f"JSON Saved: {short_name}",
            f"Total agents: {total}  |  Zones: {len(self.valid_zones)}"
        )
        self.proceed_btn.setEnabled(True)


# ══════════════════════════════════════════════════════════
#  VIEW 3 — SIMULATION
# ══════════════════════════════════════════════════════════

# Per-model config definitions:  (label, param_key, type, min, max, default, step, decimals, tooltip)
MODEL_CONFIGS = {
    "SFM": {
        "display": "Social Force Model",
        "desc": "Physics-based pedestrian dynamics using attractive/repulsive force fields.",
        "script": "SFM_evacuation.py",
        "output": "output/sfm_agent_paths.png",
        "params": [
            ("Speed Min (px/s)",  "speed_min",         "float", 0.5, 5.0,  0.8,  0.1, 1, "Minimum agent walking speed"),
            ("Speed Max (px/s)",  "speed_max",         "float", 0.5, 5.0,  1.8,  0.1, 1, "Maximum agent walking speed"),
            ("Relaxation Time",   "relaxation_time",   "float", 0.1, 2.0,  0.5,  0.1, 2, "How quickly agents reach desired speed (τ)"),
            ("Agent Strength",    "agent_strength",    "float", 100, 5000, 2000, 100, 0, "Repulsion force magnitude between agents"),
            ("Wall Strength",     "wall_strength",     "float", 100, 5000, 2000, 100, 0, "Repulsion force magnitude from walls"),
            ("Panic Threshold",   "panic_threshold",   "float", 0.0, 1.0,  0.3,  0.05, 2, "Panic level at which agents start evacuating"),
            ("Max Sim Time (s)",  "max_time",          "float", 10,  600,  300,  10, 0, "Maximum simulation duration in seconds"),
            ("Fire Spread Speed",     "fire_spread_speed",     "float", 0.1, 5.0, 1.0, 0.1, 1, "Diffusion rate multiplier"),
            ("Fire Intensity Factor", "fire_intensity_factor", "float", 0.1, 5.0, 1.0, 0.1, 1, "Growth-to-saturation rate multiplier"),
        ]
    },
    "RVO": {
        "display": "Reciprocal Velocity Obstacles",
        "desc": "Geometric collision avoidance — agents compute collision-free velocities in real time.",
        "script": "RVO_evacuation.py",
        "output": "output/rvo_agent_paths.png",
        "params": [
            ("Speed Min (px/s)",  "speed_min",         "float", 0.5, 5.0,  0.8,  0.1, 1, "Minimum agent walking speed"),
            ("Speed Max (px/s)",  "speed_max",         "float", 0.5, 5.0,  1.8,  0.1, 1, "Maximum agent walking speed"),
            ("Time Horizon (s)",  "time_horizon",      "float", 0.5, 10.0, 2.0,  0.5, 1, "How far ahead agents look for collisions"),
            ("Neighbor Distance", "neighbor_dist",     "float", 10,  200,  50,   5,  0, "Radius (px) in which agents consider others"),
            ("Max Neighbors",     "max_neighbors",     "int",   1,   50,   10,   1,  0, "Max agents each agent considers per step"),
            ("Panic Threshold",   "panic_threshold",   "float", 0.0, 1.0,  0.3,  0.05, 2, "Panic level at which agents start evacuating"),
            ("Max Sim Time (s)",  "max_time",          "float", 10,  600,  300,  10, 0, "Maximum simulation duration in seconds"),
            ("Fire Spread Speed",     "fire_spread_speed",     "float", 0.1, 5.0, 1.0, 0.1, 1, "Diffusion rate multiplier"),
            ("Fire Intensity Factor", "fire_intensity_factor", "float", 0.1, 5.0, 1.0, 0.1, 1, "Growth-to-saturation rate multiplier"),
        ]
    },
    "Continuum": {
        "display": "Continuum Crowds",
        "desc": "Treuille et al. 2006 — flow field approach treating the crowd as a fluid continuum.",
        "script": "continuum_evacuation_path.py",
        "output": "output/continuum_agent_paths.png",
        "params": [
            ("Time Step (DT)",    "DT",            "float", 0.01, 0.2,  0.05,  0.01, 2, "Simulation timestep in seconds"),
            ("Max Time (s)",      "MAX_TIME",      "float", 10,   200,  40,    10,   0, "Hard cap on simulation duration"),
            ("Speed (px/s)",      "speed_px_s",    "float", 10,   300,  150,   10,   0, "Agent movement speed in pixels per second"),
            ("Grid Resolution",   "grid_res",      "int",   1,    8,    4,     1,    0, "Pixels per potential field cell. Higher = faster but less accurate"),
            ("Density Radius",    "density_radius","int",   2,    20,   6,     1,    0, "Pixel radius for agent density splatting"),
            ("Agent Radius",      "agent_radius",  "int",   2,    20,   6,     1,    0, "Agent body radius for repulsion"),
            ("Repulse Range",     "repulse_range", "float", 4,    40,   14,    1,    0, "Range in pixels of agent-agent repulsion"),
            ("Relax Time",        "relax_time",    "float", 0.1,  2.0,  0.3,   0.1,  1, "Velocity relaxation time constant"),
            ("Fire Spread Speed",     "fire_spread_speed",     "float", 0.1, 5.0, 1.0, 0.1, 1, "Diffusion rate multiplier"),
            ("Fire Intensity Factor", "fire_intensity_factor", "float", 0.1, 5.0, 1.0, 0.1, 1, "Growth-to-saturation rate multiplier"),
        ]
    },
    "CA": {
        "display": "Cellular Automata",
        "desc": "Grid-based discrete-time model — agents on cells, local rules govern movement.",
        "script": "CA_evacuation.py",
        "output": "output/ca_paths.png",
        "params": [
            ("Time Step (DT)",       "dt",             "float", 0.01, 0.2,  0.05,  0.01, 2, "Seconds per simulation tick"),
            ("Max Time (s)",         "max_time",       "float", 10,   600,  120,   10,   0, "Hard cap on simulation duration"),
            ("Desired Speed (px/s)", "desired_speed",  "float", 10,   200,  55,    5,    0, "Target movement speed for each agent"),
            ("Wall Clearance (px)",  "agent_wall_min", "int",   1,    30,   6,     1,    0, "Minimum spawn distance from walls"),
            ("Randomness",           "randomness",     "float", 0.0,  0.5,  0.06,  0.01, 2, "Amount of stochastic movement noise"),
            ("Exit Radius (px)",     "exit_radius",    "int",   5,    60,   22,    1,    0, "Distance from an exit counted as evacuated"),
            ("Fire Spread Speed",     "fire_spread_speed",     "float", 0.1, 5.0, 1.0, 0.1, 1, "Diffusion rate multiplier"),
            ("Fire Intensity Factor", "fire_intensity_factor", "float", 0.1, 5.0, 1.0, 0.1, 1, "Growth-to-saturation rate multiplier"),
        ]
    },
}


def _run_simulation(script_name, params, mask_path, zone_config_path,
                    output_path, progress_cb=None):
    import os, shutil

    Path("output").mkdir(exist_ok=True)

    runtime_mask = Path("stitched_mask.png")
    runtime_zone_config = Path("zone_config.json")
    runtime_stitched_config = Path("stitched_mask_zone_config.json")
    runtime_params = Path("output") / "simulation_params.json"

    if Path(mask_path).resolve() != runtime_mask.resolve():
        shutil.copy2(mask_path, runtime_mask)

    with open(zone_config_path, "r", encoding="utf-8") as f:
        zone_config = json.load(f)
    zone_config["mask_path"] = str(runtime_mask)

    # Ensure hazard is included in the config
    zone_config["hazard"] = zone_config.get("hazard", None)

    for target in (runtime_zone_config, runtime_stitched_config):
        with open(target, "w", encoding="utf-8") as f:
            json.dump(zone_config, f, indent=2)

    with open(runtime_params, "w", encoding="utf-8") as f:
        json.dump(params or {}, f, indent=2)

    if progress_cb:
        progress_cb(10, f"Running {script_name}...")

    script_args = {
        "SFM_evacuation.py": ["stitched_mask.png", "zone_config.json", str(runtime_params)],
        "RVO_evacuation.py": ["stitched_mask.png", "stitched_mask_zone_config.json", str(runtime_params)],
        "continuum_evacuation_path.py": ["stitched_mask.png", "stitched_mask_zone_config.json", str(runtime_params)],
        "CA_evacuation.py": ["stitched_mask.png", "zone_config.json", str(runtime_params)],
    }
    
    def _resolve_runner(script_name):
        if getattr(sys, "frozen", False):          # we're running as a compiled exe
            exe_name = script_name.replace(".py", ".exe")
            return [str(Path(sys.executable).parent / exe_name)]
        return [sys.executable, script_name]       # normal dev mode

    proc = subprocess.Popen(
        _resolve_runner(script_name) + script_args.get(script_name, []),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )

    pct = 10
    output_lines = []
    for line in proc.stdout:
        line = line.strip()
        if line:
            output_lines.append(line)
            pct = min(pct + 2, 90)
            if progress_cb:
                progress_cb(pct, line[:80])

    proc.wait()

    if proc.returncode != 0:
        detail = "\n".join(output_lines[-10:])
        if detail:
            raise RuntimeError(f"{script_name} exited with code {proc.returncode}\n{detail}")
        raise RuntimeError(f"{script_name} exited with code {proc.returncode}")

    # where each script actually writes its image output
    script_outputs = {
        "SFM_evacuation.py":            "output/sfm_agent_paths.png",
        "RVO_evacuation.py":            "output/rvo_agent_paths.png",
        "continuum_evacuation_path.py": "output/continuum_agent_paths.png",
        "CA_evacuation.py":             "output/ca_paths.png",
    }
    # where each script writes its text report
    script_reports = {
        "SFM_evacuation.py":            "output/SFM_output_report.txt",
        "RVO_evacuation.py":            "output/RVO_output_report.txt",
        "continuum_evacuation_path.py": "output/continuum_report.txt",
        "CA_evacuation.py":             "output/ca_report.txt",
    }

    # copy image to output/ destination expected by MODEL_CONFIGS
    actual_out = script_outputs.get(script_name)
    if actual_out and Path(actual_out).exists():
        if Path(actual_out).resolve() != Path(output_path).resolve():
            import shutil as _sh
            _sh.copy2(actual_out, output_path)
    elif not Path(output_path).exists():
        raise RuntimeError(f"Output image not found: {output_path}")

    # copy report to last_sim_report.txt so View Report button works
    report_src = script_reports.get(script_name)
    if report_src and Path(report_src).exists():
        import shutil as _sh
        _sh.copy2(report_src, "last_sim_report.txt")
    elif Path("last_sim_report.txt").exists():
        Path("last_sim_report.txt").unlink()

    if progress_cb:
        progress_cb(100, "Simulation complete")


class View3_Simulation(QWidget):
    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        # self._worker: Optional[Worker] = None
        # self._param_widgets = {}
        # self._saved_params = {}   # {model_key: {param_key: value}} — persists user edits
        # self._build_ui()
        self._worker: Optional[Worker] = None
        self._param_widgets = {}
        self._model_btns = {}
        self._saved_params = {}   # {model_key: {param_key: value}} — persists user edits
        self._results = []        # list of (image_path, report_path, label) — all completed runs
        self._current_idx = -1   # which result is currently shown
        self._build_ui()

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(20)

        # ── Left: model selection ────────────────────────
        left = QFrame(); left.setObjectName("card")
        left.setFixedWidth(280)
        lv = QVBoxLayout(left)
        lv.setContentsMargins(20, 20, 20, 20)
        lv.setSpacing(10)

        title = QLabel("Simulation"); title.setObjectName("title")
        sub = QLabel("Select a model and configure parameters")
        sub.setObjectName("subtitle"); sub.setWordWrap(True)
        lv.addWidget(title)
        lv.addWidget(sub)
        lv.addWidget(self._sep())
        lv.addWidget(self._section_label("SELECT MODEL"))

        # self._model_btns = {}
        # for key, cfg in MODEL_CONFIGS.items():
        #     btn = QPushButton(f"{cfg['display']}\n{cfg['desc'][:50]}…" if len(cfg['desc']) > 50 else f"{cfg['display']}\n{cfg['desc']}")
        #     btn.setObjectName("model_card")
        #     btn.setFixedHeight(68)
        
        # full model description 
        for key, cfg in MODEL_CONFIGS.items():
            btn = QPushButton(f"{cfg['display']}\n{cfg['desc']}")
            btn.setObjectName("model_card")
            btn.setFixedHeight(68)
            btn.clicked.connect(lambda checked, k=key: self._select_model(k))
            self._model_btns[key] = btn
            lv.addWidget(btn)

        lv.addStretch()
        lv.addWidget(self._sep())
        
        back_btn3 = QPushButton("← Back to Zones")
        back_btn3.clicked.connect(lambda: self.window()._go_to(1))
        lv.addWidget(back_btn3)

        self.status_label = QLabel("Configure and run a simulation")
        self.status_label.setStyleSheet(f"color: {DARK['subtext']}; font-size: 9pt;")
        self.status_label.setWordWrap(True)
        lv.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        lv.addWidget(self.progress_bar)

        # ── Middle: config panel ─────────────────────────
        mid = QFrame(); mid.setObjectName("panel")
        mid.setFixedWidth(340)
        mv = QVBoxLayout(mid)
        mv.setContentsMargins(20, 20, 20, 20)
        mv.setSpacing(10)

        self.config_title = QLabel("Select a model")
        self.config_title.setStyleSheet("font-size: 13pt; font-weight: bold;")
        self.config_desc = QLabel("")
        self.config_desc.setStyleSheet(f"color: {DARK['subtext']}; font-size: 9pt;")
        self.config_desc.setWordWrap(True)
        mv.addWidget(self.config_title)
        mv.addWidget(self.config_desc)
        mv.addWidget(self._sep())

        self.params_scroll = QScrollArea()
        self.params_scroll.setWidgetResizable(True)
        self.params_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.params_container = QWidget()
        self.params_layout = QFormLayout(self.params_container)
        self.params_layout.setSpacing(8)
        self.params_scroll.setWidget(self.params_container)
        mv.addWidget(self.params_scroll, 1)

        mv.addWidget(self._sep())

        io_group = QGroupBox("Input Files")
        io_layout = QFormLayout()
        io_layout.setSpacing(4)
        self.mask_display = QLabel("(none)")
        self.mask_display.setStyleSheet(f"color: {DARK['subtext']}; font-size: 8pt;")
        self.zone_display = QLabel("(none)")
        self.zone_display.setStyleSheet(f"color: {DARK['subtext']}; font-size: 8pt;")
        io_layout.addRow("Mask:", self.mask_display)
        io_layout.addRow("Zones:", self.zone_display)
        io_group.setLayout(io_layout)
        mv.addWidget(io_group)

        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.setToolTip("Restore all parameters to their default values.")
        reset_btn.clicked.connect(self._reset_params)
        mv.addWidget(reset_btn)

        self.run_btn = QPushButton("▶  Run Simulation")
        self.run_btn.setObjectName("primary")
        self.run_btn.setEnabled(False)
        self.run_btn.clicked.connect(self._run)
        mv.addWidget(self.run_btn)

        self.compare_btn = QPushButton("⚖  Compare All Models")
        self.compare_btn.setEnabled(False)
        self.compare_btn.setToolTip("Run all four models on the same config and show a score table.")
        self.compare_btn.clicked.connect(self._run_compare)
        mv.addWidget(self.compare_btn)

        # ── Right: output viewer ─────────────────────────
        right = QFrame(); right.setObjectName("panel")
        rv = QVBoxLayout(right)
        rv.setContentsMargins(16, 16, 16, 16)
        rv.setSpacing(8)

        output_header = QHBoxLayout()
        output_title = QLabel("Simulation Output")
        output_title.setStyleSheet("font-weight: bold; font-size: 11pt;")

        self.nav_prev_btn = QPushButton("<")
        self.nav_prev_btn.setFixedWidth(32)
        self.nav_prev_btn.setEnabled(False)
        self.nav_prev_btn.clicked.connect(self._show_prev)

        self.nav_label = QLabel("")
        self.nav_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.nav_label.setStyleSheet(f"color: {DARK['subtext']}; font-size: 9pt; min-width: 80px;")

        self.nav_next_btn = QPushButton(">")
        self.nav_next_btn.setFixedWidth(32)
        self.nav_next_btn.setEnabled(False)
        self.nav_next_btn.clicked.connect(self._show_next)

        self.fit_btn = QPushButton("Fit")
        self.fit_btn.setFixedWidth(50)
        self.fit_btn.clicked.connect(lambda: self.output_view.reset_zoom())
        self.save_img_btn = QPushButton("Save Image")
        self.save_img_btn.setEnabled(False)
        self.save_img_btn.clicked.connect(self._save_image)
        self.view_report_btn = QPushButton("View Report")
        self.view_report_btn.setEnabled(False)
        self.view_report_btn.clicked.connect(self._view_report)

        output_header.addWidget(output_title)
        output_header.addStretch()
        output_header.addWidget(self.nav_prev_btn)
        output_header.addWidget(self.nav_label)
        output_header.addWidget(self.nav_next_btn)
        output_header.addSpacing(8)
        output_header.addWidget(self.fit_btn)
        output_header.addWidget(self.save_img_btn)
        output_header.addWidget(self.view_report_btn)
        rv.addLayout(output_header)

        self.output_view = ZoomableImageView(
            "Run a simulation to see output here.\n\n"
            "Controls:\n  Scroll = zoom  |  Drag = pan"
        )
        rv.addWidget(self.output_view, 1)

        self.output_info = QLabel("")
        self.output_info.setStyleSheet(f"color: {DARK['subtext']}; font-size: 9pt;")
        rv.addWidget(self.output_info)

        root.addWidget(left)
        root.addWidget(mid)
        root.addWidget(right, 1)

        # Select SFM by default
        self._select_model("SFM")

    def _sep(self):
        f = QFrame(); f.setFrameShape(QFrame.Shape.HLine)
        f.setStyleSheet(f"color: {DARK['border']};")
        return f

    def _section_label(self, text):
        l = QLabel(text); l.setObjectName("section")
        return l

    def on_enter(self):
        """Refresh displayed paths when entering this view."""
        if self.state.mask_path:
            self.mask_display.setText(Path(self.state.mask_path).name)
            self.mask_display.setStyleSheet(f"color: {DARK['success']}; font-size: 8pt;")
        if self.state.zone_config_path:
            self.zone_display.setText(Path(self.state.zone_config_path).name)
            self.zone_display.setStyleSheet(f"color: {DARK['success']}; font-size: 8pt;")
        self._update_run_btn()

    def _update_run_btn(self):
        has_inputs = bool(self.state.mask_path and self.state.zone_config_path)
        self.run_btn.setEnabled(has_inputs and self.state.selected_model != "")
        self.compare_btn.setEnabled(has_inputs)
        
    # reset to default for the peremeters of the selected model
    def _reset_params(self):
        key = self.state.selected_model
        if not key:
            return
        for (label, key_p, typ, mn, mx, default, step, dec, tip) in MODEL_CONFIGS[key]["params"]:
            if key_p in self._param_widgets:
                self._param_widgets[key_p].setValue(default)

    def _select_model(self, key: str):
        self.state.selected_model = key
        for k, btn in self._model_btns.items():
            btn.setObjectName("model_card_selected" if k == key else "model_card")
            btn.style().unpolish(btn)
            btn.style().polish(btn)

        cfg = MODEL_CONFIGS[key]
        self.config_title.setText(cfg["display"])
        self.config_desc.setText(cfg["desc"])

        # Save current widget values before clearing (so switching models doesn't lose edits)
        if self._param_widgets:
            prev_key = self.state.selected_model  # still the old model at this point
            self._saved_params[prev_key] = {k: w.value() for k, w in self._param_widgets.items()}

        # Clear old params
        while self.params_layout.count():
            item = self.params_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._param_widgets.clear()

        # Restore previously saved values for this model (or fall back to defaults)
        saved = self._saved_params.get(key, {})

        # Build new param widgets
        for (label, key_p, typ, mn, mx, default, step, dec, tip) in cfg["params"]:
            value = saved.get(key_p, default)   # use saved value if available
            if typ == "int":
                w = QSpinBox()
                w.setRange(int(mn), int(mx))
                w.setValue(int(value))
                w.setSingleStep(int(step))
            else:
                w = QDoubleSpinBox()
                w.setRange(float(mn), float(mx))
                w.setValue(float(value))
                w.setSingleStep(float(step))
                w.setDecimals(int(dec))
            w.setToolTip(tip)
            self._param_widgets[key_p] = w
            self.params_layout.addRow(label + ":", w)

        self._update_run_btn()

    def _run_compare(self):
        if not self.state.mask_path or not self.state.zone_config_path:
            QMessageBox.warning(self, "Missing Inputs",
                "Both mask and zone config are required.")
            return
        if not Path("compare_models.py").exists():
            QMessageBox.warning(self, "Script Not Found",
                "compare_models.py not found in the project root.")
            return

        self.compare_btn.setEnabled(False)
        self.run_btn.setEnabled(False)
        for btn in self._model_btns.values():
            btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Running all models — this will take a while...")

        import os, shutil
        runtime_mask = Path("stitched_mask.png")
        runtime_zone_config = Path("zone_config.json")
        runtime_stitched_config = Path("stitched_mask_zone_config.json")
        if Path(self.state.mask_path).resolve() != runtime_mask.resolve():
            shutil.copy2(self.state.mask_path, runtime_mask)
        with open(self.state.zone_config_path, "r", encoding="utf-8") as f:
            zone_config = json.load(f)
        zone_config["mask_path"] = str(runtime_mask)
        for target in (runtime_zone_config, runtime_stitched_config):
            with open(target, "w", encoding="utf-8") as f:
                json.dump(zone_config, f, indent=2)

        def _do_compare(progress_cb=None):
            proc = subprocess.Popen(
                [sys.executable, "compare_models.py", "stitched_mask.png", "zone_config.json"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            )
            pct = 0
            for line in proc.stdout:
                line = line.strip()
                if line and progress_cb:
                    pct = min(pct + 3, 95)
                    progress_cb(pct, line[:80])
            proc.wait()
            if proc.returncode != 0:
                raise RuntimeError(f"compare_models.py exited with code {proc.returncode}")
            if progress_cb:
                progress_cb(100, "Done")

        self._compare_worker = Worker(_do_compare)
        self._compare_worker.progress.connect(self._on_progress)
        self._compare_worker.finished.connect(self._on_compare_done)
        self._compare_worker.start()

    def _run(self):
        if not self.state.mask_path or not self.state.zone_config_path:
            QMessageBox.warning(self, "Missing Inputs",
                "Both mask (View 1) and zone config (View 2) are required.")
            return

        key = self.state.selected_model
        cfg = MODEL_CONFIGS[key]
        script = cfg["script"]

        if not Path(script).exists():
            QMessageBox.warning(self, "Script Not Found",
                f"{script} not found in the current directory.\n"
                "Make sure all simulation scripts are in the project root.")
            return

        params = {k: w.value() for k, w in self._param_widgets.items()}
        self._saved_params[key] = params   # persist so re-selecting this model keeps the values
        output = cfg["output"]
        self.state.output_image_path = output
        
        self.run_btn.setEnabled(False)
        self.save_img_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Starting simulation...")
        for btn in self._model_btns.values():
            btn.setEnabled(False)

        self._worker = Worker(
            _run_simulation,
            script, params,
            self.state.mask_path,
            self.state.zone_config_path,
            output,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_done)
        self._worker.start()

    def _on_progress(self, pct, msg):
        self.progress_bar.setValue(pct)
        self.status_label.setText(msg)

    def _on_done(self, success, msg):
        self.progress_bar.setVisible(False)
        self.run_btn.setEnabled(True)
        for btn in self._model_btns.values():
            btn.setEnabled(True)
        if success:
            self.status_label.setText("Simulation complete")
            self.status_label.setStyleSheet(f"color: {DARK['success']}; font-size: 9pt;")

            key = self.state.selected_model
            img_path = MODEL_CONFIGS[key]["output"]
            report_src = {
                "SFM_evacuation.py":            "output/SFM_output_report.txt",
                "RVO_evacuation.py":            "output/RVO_output_report.txt",
                "continuum_evacuation_path.py": "output/continuum_report.txt",
                "CA_evacuation.py":             "output/ca_report.txt",
            }.get(MODEL_CONFIGS[key]["script"], "last_sim_report.txt")
            report_path = report_src if Path(report_src).exists() else ""
            self._results.append((img_path, report_path, MODEL_CONFIGS[key]["display"]))
            self._show_result(len(self._results) - 1)

        else:
            self.status_label.setText("Simulation failed — see error dialog")
            self.status_label.setStyleSheet(f"color: {DARK['danger']}; font-size: 9pt;")
            print(f"SIMULATION ERROR: {msg}")
            dlg = QDialog(self)
            dlg.setWindowTitle("Simulation Error")
            dlg.resize(640, 400)
            dlg.setStyleSheet(f"background: {DARK['panel']}; color: {DARK['text']};")
            dl = QVBoxLayout(dlg)
            dl.setContentsMargins(16, 16, 16, 16)
            tb = QTextEdit()
            tb.setReadOnly(True)
            tb.setPlainText(msg)
            tb.setStyleSheet(f"""
                QTextEdit {{
                    background: {DARK['input_bg']};
                    color: {DARK['danger']};
                    border: 1px solid {DARK['border']};
                    border-radius: 6px;
                    font-family: 'Consolas', 'Courier New', monospace;
                    font-size: 9pt;
                    padding: 8px;
                }}
            """)
            dl.addWidget(tb)
            close = QPushButton("Close")
            close.setFixedWidth(80)
            close.clicked.connect(dlg.accept)
            row = QHBoxLayout()
            row.addStretch(); row.addWidget(close)
            dl.addLayout(row)
            dlg.exec()

    def _on_compare_done(self, success, msg):
        self.progress_bar.setVisible(False)
        for btn in self._model_btns.values():
            btn.setEnabled(True)
        self._update_run_btn()
        if not success:
            self.status_label.setText("Compare failed")
            self.status_label.setStyleSheet(f"color: {DARK['danger']}; font-size: 9pt;")
            return

        self.status_label.setText("All models complete")
        self.status_label.setStyleSheet(f"color: {DARK['success']}; font-size: 9pt;")

        # Add each model's result individually so the user can nav through them
        script_map = {
            "SFM":       ("output/sfm_agent_paths.png",        "output/SFM_output_report.txt"),
            "RVO":       ("output/rvo_agent_paths.png",        "output/RVO_output_report.txt"),
            "CA":        ("output/ca_paths.png",               "output/ca_report.txt"),
            "Continuum": ("output/continuum_agent_paths.png",  "output/continuum_report.txt"),
        }
        for label, (img_p, rep_p) in script_map.items():
            if Path(img_p).exists():
                self._results.append((img_p, rep_p if Path(rep_p).exists() else "", label))
        if self._results:
            self._show_result(len(self._results) - len([v for v in script_map.values() if Path(v[0]).exists()]))

        # Also show the summary table in a dialog
        report_path = "output/model_comparison.txt"
        if Path(report_path).exists():
            text = Path(report_path).read_text(encoding="utf-8", errors="replace")
            dlg = QDialog(self)
            dlg.setWindowTitle("Model Comparison")
            dlg.resize(680, 300)
            dlg.setStyleSheet(f"background: {DARK['panel']}; color: {DARK['text']};")
            dl = QVBoxLayout(dlg)
            dl.setContentsMargins(16, 16, 16, 16)
            tb = QTextEdit()
            tb.setReadOnly(True)
            tb.setPlainText(text)
            tb.setStyleSheet(f"""
                QTextEdit {{
                    background: {DARK['input_bg']};
                    color: {DARK['text']};
                    border: 1px solid {DARK['border']};
                    border-radius: 6px;
                    font-family: 'Consolas', 'Courier New', monospace;
                    font-size: 10pt;
                    padding: 8px;
                }}
            """)
            dl.addWidget(tb)
            close = QPushButton("Close")
            close.setFixedWidth(80)
            close.clicked.connect(dlg.accept)
            row = QHBoxLayout()
            row.addStretch(); row.addWidget(close)
            dl.addLayout(row)
            dlg.exec()

    def _show_result(self, idx: int):
        """Display the result at idx and update nav buttons."""
        if not self._results or idx < 0 or idx >= len(self._results):
            return
        self._current_idx = idx
        img_path, report_path, label = self._results[idx]

        self.output_view.load_image(img_path)
        img = cv2.imread(img_path)
        if img is not None:
            self.output_info.setText(
                f"{label}  |  {img_path}  |  {img.shape[1]}×{img.shape[0]}px")
        self.state.output_image_path = img_path

        self.save_img_btn.setEnabled(True)
        self.view_report_btn.setEnabled(bool(report_path and Path(report_path).exists()))

        n = len(self._results)
        self.nav_label.setText(f"{idx + 1} / {n}")
        self.nav_prev_btn.setEnabled(idx > 0)
        self.nav_next_btn.setEnabled(idx < n - 1)

    def _show_prev(self):
        self._show_result(self._current_idx - 1)

    def _show_next(self):
        self._show_result(self._current_idx + 1)

    def _save_image(self):
        if not self.state.output_image_path or not Path(self.state.output_image_path).exists():
            return
        dest, _ = QFileDialog.getSaveFileName(
            self, "Save Output Image",
            self.state.output_image_path,
            "Images (*.png *.jpg)")
        if dest:
            import shutil
            shutil.copy2(self.state.output_image_path, dest)

    def _view_report(self):
        # Use the report that belongs to the currently displayed result
        report_path = ""
        if 0 <= self._current_idx < len(self._results):
            report_path = self._results[self._current_idx][1]
        if not report_path or not Path(report_path).exists():
            QMessageBox.information(self, "No Report", "No report file found for this simulation.")
            return
        try:
            text = Path(report_path).read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not read report: {e}")
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Simulation Report")
        dlg.resize(680, 520)
        dlg.setStyleSheet(f"background: {DARK['panel']}; color: {DARK['text']};")
        dlg_layout = QVBoxLayout(dlg)
        dlg_layout.setContentsMargins(16, 16, 16, 16)
        dlg_layout.setSpacing(10)

        text_box = QTextEdit()
        text_box.setReadOnly(True)
        text_box.setPlainText(text)
        text_box.setStyleSheet(f"""
            QTextEdit {{
                background: {DARK['input_bg']};
                color: {DARK['text']};
                border: 1px solid {DARK['border']};
                border-radius: 6px;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 10pt;
                padding: 8px;
            }}
        """)
        dlg_layout.addWidget(text_box)

        close_btn = QPushButton("Close")
        close_btn.setFixedWidth(100)
        close_btn.clicked.connect(dlg.accept)
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(close_btn)
        dlg_layout.addLayout(btn_row)

        dlg.exec()


# ══════════════════════════════════════════════════════════
#  MAIN WINDOW
# ══════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TRAGIC  —  Crowd Evacuation Intelligence System")
        self.setMinimumSize(1280, 780)

        self.state = AppState()

        # Central widget
        central = QWidget()
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        self.setCentralWidget(central)

        # Nav bar
        self.nav = NavBar()
        main_layout.addWidget(self.nav)

        # Stack
        self.stack = QStackedWidget()
        main_layout.addWidget(self.stack, 1)

        # Views
        self.view1 = View1_MapParser(self.state)
        self.view2 = View2_ZoneEditor(self.state)
        self.view3 = View3_Simulation(self.state)

        self.stack.addWidget(self.view1)   # index 0
        self.stack.addWidget(self.view2)   # index 1
        self.stack.addWidget(self.view3)   # index 2

        # Wire proceed signals
        self.view1.proceed_signal.connect(lambda: self._go_to(1))
        self.view2.proceed_signal.connect(lambda: self._go_to(2))
        self.nav.nav_clicked.connect(self._go_to)

        self._go_to(0)

    def _go_to(self, index: int):
        self.stack.setCurrentIndex(index)
        self.nav.set_active(index)
        if index == 1:
            self.view2.on_enter()
        elif index == 2:
            self.view3.on_enter()


# ══════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLESHEET)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
