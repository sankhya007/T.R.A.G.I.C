"""
tragic_launcher.py  —  TRAGIC Unified Launcher
Three views in one window, shared application state.

Run: python tragic_launcher.py

Requires: PyQt6, numpy, opencv-python, scipy, scikit-image, torch
"""

import sys
import json
import math
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
)
from PyQt6.QtCore import (
    Qt, QTimer, QThread, pyqtSignal, QPointF, QRectF,
    QPropertyAnimation, QEasingCurve, QSize,
)
from PyQt6.QtGui import (
    QPixmap, QImage, QPainter, QColor, QPen, QBrush,
    QFont, QWheelEvent, QTransform, QCursor,
)


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
    def __init__(self):
        super().__init__()
        self.setFixedHeight(56)
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
        logo = QLabel("⚠ TRAGIC")
        logo.setStyleSheet(f"font-size: 14pt; font-weight: bold; color: {DARK['accent']}; letter-spacing: 2px;")
        layout.addWidget(logo)

        layout.addSpacing(40)

        # Step indicators
        self.steps = []
        step_names = ["Map Parser", "Zone Editor", "Simulation"]
        for i, name in enumerate(step_names):
            btn = QLabel(f"  {i+1}. {name}  ")
            btn.setAlignment(Qt.AlignmentFlag.AlignCenter)
            btn.setFixedHeight(32)
            btn.setStyleSheet(f"""
                color: {DARK['subtext']};
                border-radius: 6px;
                padding: 4px 12px;
                font-size: 10pt;
            """)
            self.steps.append(btn)
            layout.addWidget(btn)
            if i < len(step_names) - 1:
                arrow = QLabel("→")
                arrow.setStyleSheet(f"color: {DARK['border']}; font-size: 12pt;")
                layout.addWidget(arrow)

        layout.addStretch()

        info = QLabel("Crowd Evacuation Intelligence System")
        info.setStyleSheet(f"color: {DARK['subtext']}; font-size: 9pt;")
        layout.addWidget(info)

    def set_active(self, index: int):
        for i, btn in enumerate(self.steps):
            if i == index:
                btn.setStyleSheet(f"""
                    color: white;
                    background: {DARK['accent']};
                    border-radius: 6px;
                    padding: 4px 12px;
                    font-weight: bold;
                    font-size: 10pt;
                """)
            elif i < index:
                btn.setStyleSheet(f"""
                    color: {DARK['success']};
                    border-radius: 6px;
                    padding: 4px 12px;
                    font-size: 10pt;
                """)
            else:
                btn.setStyleSheet(f"""
                    color: {DARK['subtext']};
                    border-radius: 6px;
                    padding: 4px 12px;
                    font-size: 10pt;
                """)


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
        self.icon_label = QLabel("✓  Config Saved")
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
        self.icon_label.setText(f"✓  {title}")
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
#  VIEW 1 — MAP PARSER
# ══════════════════════════════════════════════════════════

def _run_predict_tiled(image_path, max_patches, overlap_ratio,
                       window_min, threshold, output_path,
                       progress_cb=None):
    """Core predict_tiled logic extracted into a callable."""
    import torch
    from model import UNet

    MODEL_PATH  = "unet.pth"
    OVERLAP_RATIO = overlap_ratio
    MODEL_INPUT   = 256
    WINDOW_MIN    = window_min
    MAX_PATCHES   = max_patches

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    def preprocess(patch):
        patch = cv2.resize(patch, (MODEL_INPUT, MODEL_INPUT), interpolation=cv2.INTER_AREA)
        patch = patch.astype(np.float32) / 255.0
        patch = np.transpose(patch, (2, 0, 1))
        return torch.from_numpy(patch).unsqueeze(0).to(DEVICE)

    def weight_map(h, w):
        y, x = np.ogrid[-1:1:h*1j, -1:1:w*1j]
        return np.exp(-(x**2 + y**2) * 4).astype(np.float32)

    def make_positions(total, win, stride):
        pos = list(range(0, total - win, stride))
        if not pos or pos[-1] + win < total:
            pos.append(total - win)
        return pos

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W, _ = img.shape

    pad_h, pad_w = int(0.05 * H), int(0.05 * W)
    pH = H + 2 * pad_h
    pW = W + 2 * pad_w

    aspect = pW / pH
    n_cols_est = math.sqrt(MAX_PATCHES * aspect)
    n_rows_est = math.sqrt(MAX_PATCHES / aspect)
    win_c = pW / (n_cols_est * (1.0 - OVERLAP_RATIO))
    win_r = pH / (n_rows_est * (1.0 - OVERLAP_RATIO))
    WINDOW = max(WINDOW_MIN, int(math.ceil(max(win_c, win_r))))

    def n_pos(dim, win, stride):
        p = list(range(0, dim - win, stride))
        if not p or p[-1] + win < dim:
            p.append(dim - win)
        return len(p)

    while True:
        STRIDE = max(1, int(round(WINDOW * (1.0 - OVERLAP_RATIO))))
        if n_pos(pW, WINDOW, STRIDE) * n_pos(pH, WINDOW, STRIDE) <= MAX_PATCHES:
            break
        WINDOW += 1

    wmap = weight_map(WINDOW, WINDOW)
    img_pad = cv2.copyMakeBorder(img, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_REFLECT_101)

    final_mask = np.zeros((pH, pW), dtype=np.float32)
    weight_sum  = np.zeros((pH, pW), dtype=np.float32)

    y_pos = make_positions(pH, WINDOW, STRIDE)
    x_pos = make_positions(pW, WINDOW, STRIDE)
    total = len(y_pos) * len(x_pos)
    done = 0

    for y1 in y_pos:
        for x1 in x_pos:
            patch = img_pad[y1:y1+WINDOW, x1:x1+WINDOW]
            dh, dw = WINDOW - patch.shape[0], WINDOW - patch.shape[1]
            if dh > 0 or dw > 0:
                patch = cv2.copyMakeBorder(patch, 0, dh, 0, dw, cv2.BORDER_REFLECT_101)
            t = preprocess(patch)
            with torch.no_grad():
                pred = model(t)
            pred = torch.sigmoid(pred).squeeze().cpu().numpy()
            pred = cv2.resize(pred, (WINDOW, WINDOW), interpolation=cv2.INTER_LINEAR)
            pred = np.clip(pred, 0.05, 0.95)
            final_mask[y1:y1+WINDOW, x1:x1+WINDOW] += pred * wmap
            weight_sum [y1:y1+WINDOW, x1:x1+WINDOW] += wmap
            done += 1
            if progress_cb:
                progress_cb(int(done / total * 90), f"Processing patch {done}/{total}")

    weight_sum[weight_sum == 0] = 1e-8
    final_mask = final_mask / weight_sum
    final_mask = final_mask[pad_h:pad_h+H, pad_w:pad_w+W]

    binary = (final_mask > threshold).astype(np.uint8)
    k = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)
    binary = cv2.dilate(binary, np.ones((2, 2), np.uint8), iterations=1)

    cv2.imwrite(output_path, binary * 255)
    if progress_cb:
        progress_cb(100, "Mask saved")


class View1_MapParser(QWidget):
    proceed_signal = pyqtSignal()

    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self._worker: Optional[Worker] = None
        self._build_ui()

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

        # File selector
        lv.addWidget(QLabel("Input Image:").setParent(None) or QLabel("Input Image:"))
        file_row = QHBoxLayout()
        self.file_label = QLabel("No file selected")
        self.file_label.setStyleSheet(f"color: {DARK['subtext']}; font-size: 9pt;")
        self.file_label.setWordWrap(True)
        self.browse_btn = QPushButton("Browse")
        self.browse_btn.setFixedWidth(80)
        self.browse_btn.clicked.connect(self._browse)
        file_row.addWidget(self.file_label, 1)
        file_row.addWidget(self.browse_btn)
        lv.addLayout(file_row)

        lv.addWidget(self._sep())
        lv.addWidget(self._section_label("TILING PARAMETERS"))

        # Parameters
        form = QFormLayout()
        form.setSpacing(8)

        self.max_patches_spin = QSpinBox()
        self.max_patches_spin.setRange(10, 200)
        self.max_patches_spin.setValue(40)
        self.max_patches_spin.setToolTip("Hard ceiling on total patch count. Higher = slower but finer detail.")
        form.addRow("Max Patches:", self.max_patches_spin)

        self.overlap_spin = QDoubleSpinBox()
        self.overlap_spin.setRange(0.1, 0.9)
        self.overlap_spin.setSingleStep(0.05)
        self.overlap_spin.setValue(0.50)
        self.overlap_spin.setDecimals(2)
        self.overlap_spin.setToolTip("Overlap between adjacent patches (0.5 = 50%). Higher = smoother boundaries.")
        form.addRow("Overlap Ratio:", self.overlap_spin)

        self.window_min_spin = QSpinBox()
        self.window_min_spin.setRange(64, 512)
        self.window_min_spin.setSingleStep(32)
        self.window_min_spin.setValue(256)
        self.window_min_spin.setToolTip("Minimum patch window size in pixels. Never goes below this.")
        form.addRow("Window Min (px):", self.window_min_spin)

        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.1, 0.9)
        self.threshold_spin.setSingleStep(0.05)
        self.threshold_spin.setValue(0.50)
        self.threshold_spin.setDecimals(2)
        self.threshold_spin.setToolTip("Binarization threshold. Lower = more pixels classified as walls.")
        form.addRow("Threshold:", self.threshold_spin)

        lv.addLayout(form)
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
        self.tweak_btn = QPushButton("↺ Tweak Settings")
        self.tweak_btn.setVisible(False)
        self.tweak_btn.clicked.connect(self._tweak)

        self.proceed_btn = QPushButton("Proceed to Zones →")
        self.proceed_btn.setObjectName("success")
        self.proceed_btn.setObjectName("primary")
        self.proceed_btn.setVisible(False)
        self.proceed_btn.clicked.connect(self._proceed)

        btn_row.addWidget(self.tweak_btn)
        btn_row.addWidget(self.proceed_btn)
        lv.addLayout(btn_row)

        # ── Right: preview ───────────────────────────────
        right = QFrame(); right.setObjectName("panel")
        rv = QVBoxLayout(right)
        rv.setContentsMargins(16, 16, 16, 16)
        rv.setSpacing(8)

        preview_header = QHBoxLayout()
        preview_title = QLabel("Output Preview")
        preview_title.setStyleSheet("font-weight: bold; font-size: 11pt;")
        self.reset_zoom_btn = QPushButton("Fit")
        self.reset_zoom_btn.setFixedWidth(50)
        self.reset_zoom_btn.clicked.connect(lambda: self.preview.reset_zoom())
        preview_header.addWidget(preview_title)
        preview_header.addStretch()
        preview_header.addWidget(self.reset_zoom_btn)
        rv.addLayout(preview_header)

        self.preview = ZoomableImageView("Run the parser to see the output mask here.\nWhite = walls  |  Black = walkable space")
        rv.addWidget(self.preview)

        self.preview_info = QLabel("")
        self.preview_info.setStyleSheet(f"color: {DARK['subtext']}; font-size: 9pt;")
        rv.addWidget(self.preview_info)

        root.addWidget(left)
        root.addWidget(right, 1)

    def _sep(self):
        f = QFrame(); f.setFrameShape(QFrame.Shape.HLine)
        f.setStyleSheet(f"color: {DARK['border']};")
        return f

    def _section_label(self, text):
        l = QLabel(text); l.setObjectName("section")
        return l

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Floorplan Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp)")
        if path:
            self.state.image_path = path
            short = Path(path).name
            self.file_label.setText(short)
            self.file_label.setStyleSheet(f"color: {DARK['text']}; font-size: 9pt;")
            self.run_btn.setEnabled(True)
            self.tweak_btn.setVisible(False)
            self.proceed_btn.setVisible(False)
            self.preview_info.setText("")

    def _run(self):
        if not self.state.image_path:
            return
        if not Path("unet.pth").exists():
            QMessageBox.warning(self, "Missing Model",
                "unet.pth not found in the current directory.\n"
                "Make sure you run this from the project root.")
            return

        self.run_btn.setEnabled(False)
        self.tweak_btn.setVisible(False)
        self.proceed_btn.setVisible(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Starting...")

        output = "stitched_mask.png"
        self.state.mask_path = output

        self._worker = Worker(
            _run_predict_tiled,
            self.state.image_path,
            self.max_patches_spin.value(),
            self.overlap_spin.value(),
            self.window_min_spin.value(),
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
            self.status_label.setText("✓ Mask generated successfully")
            self.status_label.setStyleSheet(f"color: {DARK['success']}; font-size: 9pt;")
            self.preview.load_image("stitched_mask.png")
            # Show image dimensions
            img = cv2.imread("stitched_mask.png", cv2.IMREAD_GRAYSCALE)
            if img is not None:
                white = np.sum(img > 127)
                black = np.sum(img <= 127)
                total = img.size
                self.preview_info.setText(
                    f"Size: {img.shape[1]}×{img.shape[0]}px  |  "
                    f"Walls: {100*white//total}%  |  Walkable: {100*black//total}%"
                )
            self.tweak_btn.setVisible(True)
            self.proceed_btn.setVisible(True)
        else:
            self.status_label.setText(f"✗ Error: {msg}")
            self.status_label.setStyleSheet(f"color: {DARK['danger']}; font-size: 9pt;")

    def _tweak(self):
        # Stay on this view — just reset the action buttons so user can re-run
        self.tweak_btn.setVisible(False)
        self.proceed_btn.setVisible(False)
        self.status_label.setText("Adjust parameters and run again.")
        self.status_label.setStyleSheet(f"color: {DARK['subtext']}; font-size: 9pt;")

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
        self.exits = []   # list of {"x": int, "y": int}
        self.exit_mode = False
        self._build_ui()

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(20)

        # ── Left panel ──────────────────────────────────
        left = QFrame(); left.setObjectName("card")
        left.setFixedWidth(300)
        lv = QVBoxLayout(left)
        lv.setContentsMargins(20, 20, 20, 20)
        lv.setSpacing(10)

        title = QLabel("Zone Editor"); title.setObjectName("title")
        sub = QLabel("Segment walkable zones and assign agent density")
        sub.setObjectName("subtitle"); sub.setWordWrap(True)
        lv.addWidget(title)
        lv.addWidget(sub)
        lv.addWidget(self._sep())

        self.load_mask_btn = QPushButton("Load Mask")
        self.load_mask_btn.clicked.connect(self._load_mask)
        lv.addWidget(self.load_mask_btn)

        self.auto_load_label = QLabel("")
        self.auto_load_label.setStyleSheet(f"color: {DARK['success']}; font-size: 8pt;")
        lv.addWidget(self.auto_load_label)

        lv.addWidget(self._sep())
        lv.addWidget(self._section_label("BASE DENSITY"))

        density_form = QFormLayout()
        self.base_spin = QDoubleSpinBox()
        self.base_spin.setRange(0.1, 20.0)
        self.base_spin.setValue(1.0)
        self.base_spin.setSingleStep(0.1)
        self.base_spin.setDecimals(1)
        self.base_spin.setToolTip("Agents per 1000px² at density index = 1")
        density_form.addRow("Agents / 1000px²:", self.base_spin)
        lv.addLayout(density_form)

        lv.addWidget(self._sep())
        lv.addWidget(self._section_label("SELECTED ZONE"))

        self.zone_info = QLabel("Click a zone on the map")
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

        lv.addWidget(self._sep())
        lv.addWidget(self._section_label("SAVE CONFIG"))

        filename_row = QHBoxLayout()
        filename_row.addWidget(QLabel("Filename:"))
        self.filename_input = QLineEdit("zone_config")
        self.filename_input.setPlaceholderText("zone_config")
        filename_row.addWidget(self.filename_input)
        lv.addLayout(filename_row)

        self.save_btn = QPushButton("💾 Save Config")
        self.save_btn.setObjectName("primary")
        self.save_btn.clicked.connect(self._save_config)
        lv.addWidget(self.save_btn)

        lv.addStretch()
        lv.addWidget(self._sep())

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

        self.map_label = QLabel()
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

        root.addWidget(left)
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
        """Called when this view becomes active. Auto-load if mask exists."""
        if self.binary is None and self.state.mask_path and Path(self.state.mask_path).exists():
            self._load_mask_from_path(self.state.mask_path)
            self.auto_load_label.setText(f"✓ Auto-loaded: {Path(self.state.mask_path).name}")

    def _load_mask(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Mask", "", "Images (*.png *.jpg *.bmp)")
        if path:
            self._load_mask_from_path(path)

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
        coords = peak_local_max(dist_norm, min_distance=40, labels=binary)
        seed_mask = np.zeros(dist_norm.shape, dtype=bool)
        seed_mask[tuple(coords.T)] = True
        markers, _ = ndi.label(seed_mask)
        labels = watershed(-dist, markers, mask=binary)

        valid_zones, zone_stats = [], {}
        for zid in np.unique(labels):
            if zid == 0:
                continue
            area = int(np.sum(labels == zid))
            if area >= 800:
                valid_zones.append(int(zid))
                zone_stats[int(zid)] = area

        self.binary = binary
        self.labels = labels
        self.valid_zones = valid_zones
        self.zone_stats = zone_stats
        self.color_map = self._build_colors(valid_zones)
        self.density_map = {zid: 1.0 for zid in valid_zones}
        self.highlight = None

        self.zone_count_label.setText(f"{len(valid_zones)} zones detected")
        self.zone_info.setText(f"{len(valid_zones)} zones detected.\nClick any zone to set its density.")
        self._refresh_map()
        
        # zone selection using mouse click 
        self._rebuild_zone_list()
        self.exit_mode_btn.setEnabled(True)
        self.exits = []
        self._update_exit_info()
        
        self.exit_mode_btn.setEnabled(True)
        self.exits = []
        self._update_exit_info() # exit updates 

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
        rgb = np.zeros((h, w, 3), dtype=np.uint8)

        for zid in self.valid_zones:
            d = self.density_map.get(zid, 1.0)
            rgb[self.labels == zid] = (80, 80, 80) if d == 0 else self.color_map[zid]
        rgb[self.binary == 0] = (20, 20, 20)

        if self.highlight is not None:
            zm = (self.labels == self.highlight).astype(np.uint8) * 255
            contours, _ = cv2.findContours(zm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(rgb, contours, -1, (255, 255, 0), 3)

        for i, zid in enumerate(self.valid_zones):
            ys, xs = np.where(self.labels == zid)
            if len(xs) == 0:
                continue
            cx, cy = int(xs.mean()), int(ys.mean())
            d = self.density_map.get(zid, 1.0)
            cv2.putText(rgb, f"{i}:{d:.1f}", (cx - 15, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
        # Draw exits
        for i, ex in enumerate(self.exits):
            cv2.circle(rgb, (int(ex["x"]), int(ex["y"])), 18, (0, 220, 80), -1)
            cv2.putText(rgb, f"E{i+1}", (int(ex["x"]) - 12, int(ex["y"]) + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

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

        # zone selection
        zid = int(self.labels[py, px])
        if zid not in self.valid_zones:
            return
        self.highlight = zid
        idx = self.valid_zones.index(zid)
        area = self.zone_stats[zid]
        d = self.density_map.get(zid, 1.0)
        agents = int(area * d * self.base_spin.value() / 1000)
        self.zone_info.setText(
            f"Zone {idx}  (id={zid})\n"
            f"Area: {area:,} px²\n"
            f"Density: {d:.1f}  →  ~{agents} agents")
        self.zone_density_spin.setValue(d)
        self.apply_btn.setEnabled(True)
        self._refresh_map()

    def _apply_density(self):
        if self.highlight is None:
            return
        self.density_map[self.highlight] = self.zone_density_spin.value()
        self._map_click_refresh()
        self._rebuild_zone_list()
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
            f"Density: {d:.1f}  →  ~{agents} agents")

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
            self.exit_mode_btn.setText("Exit Mode ON (click map)")
            self.exit_mode_btn.setStyleSheet(f"background: {DARK['success']}; color: white; font-weight: bold;")
        else:
            self.exit_mode_btn.setText("Place Exits")
            self.exit_mode_btn.setStyleSheet("")

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

        config = {
            "mask_path": self.state.mask_path,
            "base_density": self.base_spin.value(),
            "agent_scale": 1000,
            "exits": self.exits,
            "zones": []
        }
        for i, zid in enumerate(self.valid_zones):
            area = self.zone_stats[zid]
            d = self.density_map.get(zid, 1.0)
            agents = int(area * d * self.base_spin.value() / 1000)
            config["zones"].append({
                "zone_index": i,
                "zone_id": zid,
                "area_px": area,
                "density_index": d,
                "agents": agents,
            })
        with open(path, "w") as f:
            json.dump(config, f, indent=2)

        self.state.zone_config_path = path
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
        "output": "SFM_agent_paths.png",
        "params": [
            ("Agent Count",       "agent_count",       "int",   10,  2000, 100,  10, 0, "Number of agents to simulate"),
            ("Speed Min (px/s)",  "speed_min",         "float", 0.5, 5.0,  0.8,  0.1, 1, "Minimum agent walking speed"),
            ("Speed Max (px/s)",  "speed_max",         "float", 0.5, 5.0,  1.8,  0.1, 1, "Maximum agent walking speed"),
            ("Relaxation Time",   "relaxation_time",   "float", 0.1, 2.0,  0.5,  0.1, 2, "How quickly agents reach desired speed (τ)"),
            ("Agent Strength",    "agent_strength",    "float", 100, 5000, 2000, 100, 0, "Repulsion force magnitude between agents"),
            ("Wall Strength",     "wall_strength",     "float", 100, 5000, 2000, 100, 0, "Repulsion force magnitude from walls"),
            ("Panic Threshold",   "panic_threshold",   "float", 0.0, 1.0,  0.3,  0.05, 2, "Panic level at which agents start evacuating"),
            ("Max Sim Time (s)",  "max_time",          "float", 10,  600,  300,  10, 0, "Maximum simulation duration in seconds"),
        ]
    },
    "RVO": {
        "display": "Reciprocal Velocity Obstacles",
        "desc": "Geometric collision avoidance — agents compute collision-free velocities in real time.",
        "script": "RVO_Evacuation.py",
        "output": "RVO_agent_paths.png",
        "params": [
            ("Agent Count",       "agent_count",       "int",   10,  2000, 100,  10, 0, "Number of agents to simulate"),
            ("Speed Min (px/s)",  "speed_min",         "float", 0.5, 5.0,  0.8,  0.1, 1, "Minimum agent walking speed"),
            ("Speed Max (px/s)",  "speed_max",         "float", 0.5, 5.0,  1.8,  0.1, 1, "Maximum agent walking speed"),
            ("Time Horizon (s)",  "time_horizon",      "float", 0.5, 10.0, 2.0,  0.5, 1, "How far ahead agents look for collisions"),
            ("Neighbor Distance", "neighbor_dist",     "float", 10,  200,  50,   5,  0, "Radius (px) in which agents consider others"),
            ("Max Neighbors",     "max_neighbors",     "int",   1,   50,   10,   1,  0, "Max agents each agent considers per step"),
            ("Panic Threshold",   "panic_threshold",   "float", 0.0, 1.0,  0.3,  0.05, 2, "Panic level at which agents start evacuating"),
            ("Max Sim Time (s)",  "max_time",          "float", 10,  600,  300,  10, 0, "Maximum simulation duration in seconds"),
        ]
    },
    "Continuum": {
        "display": "Continuum Crowds",
        "desc": "Treuille et al. 2006 — flow field approach treating the crowd as a fluid continuum.",
        "script": "continuum_evacuation_path.py",
        "output": "continuum_agent_paths.png",
        "params": [
            ("Agent Count",       "agent_count",       "int",   10,  2000, 100,  10, 0, "Number of agents to simulate"),
            ("Speed (px/s)",      "speed",             "float", 0.5, 10.0, 2.0,  0.5, 1, "Agent movement speed along flow field"),
            ("Step Size",         "step_size",         "float", 0.1, 5.0,  1.0,  0.1, 1, "Integration step size for path tracing"),
            ("Max Steps",         "max_steps",         "int",   100, 5000, 2000, 100, 0, "Maximum path trace steps per agent"),
            ("Gradient Blur σ",   "gradient_blur",     "float", 0.5, 5.0,  1.0,  0.5, 1, "Gaussian blur on distance field before gradient"),
            ("Path Accumulation", "log_accumulate",    "int",   1,   10,   3,    1,  0, "Log-scale path brightness multiplier"),
        ]
    },
    "CA": {
        "display": "Cellular Automata",
        "desc": "Grid-based discrete-time model — agents on cells, local rules govern movement.",
        "script": "CA_evacuation.py",
        "output": "CA_agent_paths.png",
        "params": [
            ("Agent Count",       "agent_count",       "int",   10,  2000, 100,  10, 0, "Number of agents to simulate"),
            ("Cell Size (px)",    "cell_size",         "int",   4,   32,   8,    2,  0, "Size of each grid cell in pixels"),
            ("Time Steps",        "time_steps",        "int",   50,  5000, 500,  50, 0, "Number of simulation time steps"),
            ("Move Probability",  "move_prob",         "float", 0.1, 1.0,  0.9,  0.05, 2, "Probability an agent moves each step"),
            ("Panic Spread Rad.", "panic_radius",      "int",   1,   20,   5,    1,  0, "Cell radius for panic contagion"),
            ("Panic Threshold",   "panic_threshold",   "float", 0.0, 1.0,  0.3,  0.05, 2, "Density above which panic triggers"),
        ]
    },
}


def _run_simulation(script_name, params, mask_path, zone_config_path,
                    output_path, progress_cb=None):
    """Launch simulation script as a subprocess, passing params as env vars."""
    import os
    env = os.environ.copy()
    env["TRAGIC_MASK_PATH"]       = mask_path
    env["TRAGIC_ZONE_CONFIG"]     = zone_config_path
    env["TRAGIC_OUTPUT_PATH"]     = output_path
    for k, v in params.items():
        env[f"TRAGIC_{k.upper()}"] = str(v)

    if progress_cb:
        progress_cb(5, f"Launching {script_name}...")

    proc = subprocess.Popen(
        [sys.executable, script_name],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    if progress_cb:
        progress_cb(10, "Simulation running...")

    # Stream output and look for progress hints
    pct = 10
    for line in proc.stdout:
        line = line.strip()
        if "%" in line or "step" in line.lower() or "agent" in line.lower():
            pct = min(pct + 5, 90)
            if progress_cb:
                progress_cb(pct, line[:80])

    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"{script_name} exited with code {proc.returncode}")

    if not Path(output_path).exists():
        raise RuntimeError(f"Output file not found: {output_path}")

    if progress_cb:
        progress_cb(100, "Simulation complete")


class View3_Simulation(QWidget):
    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self._worker: Optional[Worker] = None
        self._param_widgets = {}
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

        self._model_btns = {}
        for key, cfg in MODEL_CONFIGS.items():
            btn = QPushButton(f"{cfg['display']}\n{cfg['desc'][:50]}…" if len(cfg['desc']) > 50 else f"{cfg['display']}\n{cfg['desc']}")
            btn.setObjectName("model_card")
            btn.setFixedHeight(68)
            btn.clicked.connect(lambda checked, k=key: self._select_model(k))
            self._model_btns[key] = btn
            lv.addWidget(btn)

        lv.addStretch()
        lv.addWidget(self._sep())

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

        self.run_btn = QPushButton("▶  Run Simulation")
        self.run_btn.setObjectName("primary")
        self.run_btn.setEnabled(False)
        self.run_btn.clicked.connect(self._run)
        mv.addWidget(self.run_btn)

        # ── Right: output viewer ─────────────────────────
        right = QFrame(); right.setObjectName("panel")
        rv = QVBoxLayout(right)
        rv.setContentsMargins(16, 16, 16, 16)
        rv.setSpacing(8)

        output_header = QHBoxLayout()
        output_title = QLabel("Simulation Output")
        output_title.setStyleSheet("font-weight: bold; font-size: 11pt;")
        self.fit_btn = QPushButton("Fit")
        self.fit_btn.setFixedWidth(50)
        self.fit_btn.clicked.connect(lambda: self.output_view.reset_zoom())
        self.save_img_btn = QPushButton("💾 Save Image")
        self.save_img_btn.setEnabled(False)
        self.save_img_btn.clicked.connect(self._save_image)
        output_header.addWidget(output_title)
        output_header.addStretch()
        output_header.addWidget(self.fit_btn)
        output_header.addWidget(self.save_img_btn)
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

    def _select_model(self, key: str):
        self.state.selected_model = key
        for k, btn in self._model_btns.items():
            btn.setObjectName("model_card_selected" if k == key else "model_card")
            btn.style().unpolish(btn)
            btn.style().polish(btn)

        cfg = MODEL_CONFIGS[key]
        self.config_title.setText(cfg["display"])
        self.config_desc.setText(cfg["desc"])

        # Clear old params
        while self.params_layout.count():
            item = self.params_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._param_widgets.clear()

        # Build new param widgets
        for (label, key_p, typ, mn, mx, default, step, dec, tip) in cfg["params"]:
            if typ == "int":
                w = QSpinBox()
                w.setRange(int(mn), int(mx))
                w.setValue(int(default))
                w.setSingleStep(int(step))
            else:
                w = QDoubleSpinBox()
                w.setRange(float(mn), float(mx))
                w.setValue(float(default))
                w.setSingleStep(float(step))
                w.setDecimals(int(dec))
            w.setToolTip(tip)
            self._param_widgets[key_p] = w
            self.params_layout.addRow(label + ":", w)

        self._update_run_btn()

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
        output = cfg["output"]
        self.state.output_image_path = output

        self.run_btn.setEnabled(False)
        self.save_img_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Starting simulation...")

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
        if success:
            self.status_label.setText("✓ Simulation complete")
            self.status_label.setStyleSheet(f"color: {DARK['success']}; font-size: 9pt;")
            self.output_view.load_image(self.state.output_image_path)
            img = cv2.imread(self.state.output_image_path)
            if img is not None:
                self.output_info.setText(
                    f"Output: {self.state.output_image_path}  |  "
                    f"Size: {img.shape[1]}×{img.shape[0]}px")
            self.save_img_btn.setEnabled(True)
        else:
            self.status_label.setText(f"✗ {msg}")
            self.status_label.setStyleSheet(f"color: {DARK['danger']}; font-size: 9pt;")

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