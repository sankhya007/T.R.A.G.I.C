# cool this one works perfectly 


import torch
import cv2
import numpy as np
import math

# ========================== CONFIG ====================================
MODEL_PATH  = "unet.pth"
IMAGE_PATH  = r"test7.jpg"

# ------ Universal tiling: budget-controlled ------
# No matter the image resolution, the grid will never exceed MAX_PATCHES
# total patches. The window size is computed BACKWARDS from this budget
# so that cols × rows ≤ MAX_PATCHES always holds.
# This is the "percentage" behaviour you want: bigger image → bigger
# window, same patch count.
MAX_PATCHES   = 40     # hard ceiling on total patches

# Overlap ratio O = 0.5  →  stride = window × 0.5
# Same as the original 256-px / 128-stride pair from the working code.
OVERLAP_RATIO = 0.50

# The model was trained on exactly 256×256 images.
# Every window is resized to this before inference — never changes.
MODEL_INPUT   = 256

# Safety floor: window is never fed to the model smaller than this
# (below 256 native px the features become too tiny to recognise).
WINDOW_MIN    = 256

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ======================================================================

from model import UNet

model = UNet()
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()


# ---------- helpers ----------

def preprocess_patch(patch):
    """Resize window to model input size and convert to normalised tensor."""
    patch = cv2.resize(patch, (MODEL_INPUT, MODEL_INPUT),
                       interpolation=cv2.INTER_AREA)   # AREA = best for downscale
    patch = patch.astype(np.float32) / 255.0
    patch = np.transpose(patch, (2, 0, 1))
    return torch.from_numpy(patch).unsqueeze(0).to(DEVICE)


def create_weight_map(h, w):
    """Gaussian weight map: patch centres matter more than edges.
       Accepts non-square windows so the weight map always matches the window."""
    y, x = np.ogrid[-1:1:h*1j, -1:1:w*1j]
    weight = np.exp(-(x**2 + y**2) * 4)
    return weight.astype(np.float32)


def make_positions(total_len, window, stride):
    """Return every start offset so the last window always covers the far edge."""
    positions = list(range(0, total_len - window, stride))
    if not positions or positions[-1] + window < total_len:
        positions.append(total_len - window)
    return positions


def progress_bar(done, total, width=30):
    """Return an ASCII progress bar string."""
    filled = int(width * done / total)
    bar    = "█" * filled + "░" * (width - filled)
    pct    = 100 * done / total
    return f"[{bar}] {pct:5.1f}%  ({done}/{total})"


# ======================== LOAD IMAGE ==================================
img = cv2.imread(IMAGE_PATH)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
orig_H, orig_W, _ = img.shape
print(f"\n{'='*55}")
print(f"  Image loaded  :  {orig_W} × {orig_H} px")

# ======================== COMPUTE WINDOW SIZE =========================
# Work BACKWARDS from the patch budget, using the PADDED dimensions
# since that is what the grid actually runs on.
#
# Step 1: analytic estimate using the aspect-aware formula.
# Step 2: iterative bump-up until the real patch count is within budget.
# This guarantees cols × rows ≤ MAX_PATCHES for every image.

pad_h = int(0.05 * orig_H)
pad_w = int(0.05 * orig_W)
pH_pre = orig_H + 2 * pad_h          # padded height used for grid calc
pW_pre = orig_W + 2 * pad_w          # padded width  used for grid calc

aspect        = pW_pre / pH_pre
n_cols_est    = math.sqrt(MAX_PATCHES * aspect)
n_rows_est    = math.sqrt(MAX_PATCHES / aspect)
win_cols      = pW_pre / (n_cols_est * (1.0 - OVERLAP_RATIO))
win_rows      = pH_pre / (n_rows_est * (1.0 - OVERLAP_RATIO))
WINDOW        = max(WINDOW_MIN, int(math.ceil(max(win_cols, win_rows))))

# Iterative correction: bump window by 1px until real count ≤ budget.
def _n_positions(dim, win, stride):
    pos = list(range(0, dim - win, stride))
    if not pos or pos[-1] + win < dim:
        pos.append(dim - win)
    return len(pos)

while True:
    STRIDE  = max(1, int(round(WINDOW * (1.0 - OVERLAP_RATIO))))
    n_cols_check = _n_positions(pW_pre, WINDOW, STRIDE)
    n_rows_check = _n_positions(pH_pre, WINDOW, STRIDE)
    if n_cols_check * n_rows_check <= MAX_PATCHES:
        break
    WINDOW += 1

print(f"  Budget        :  ≤ {MAX_PATCHES} patches")
print(f"  Window        :  {WINDOW} px")
print(f"  Stride (50%)  :  {STRIDE} px  (overlap = {OVERLAP_RATIO:.0%})")
print(f"  Expected grid :  {n_cols_check} cols × {n_rows_check} rows = "
      f"{n_cols_check * n_rows_check} patches")
print(f"  Model input   :  {MODEL_INPUT} × {MODEL_INPUT} px")
print(f"{'='*55}\n")

# Pre-compute weight map once (shape = WINDOW × WINDOW)
weight_map = create_weight_map(WINDOW, WINDOW)

# ===================== ADD 5% PADDING =================================
img_pad = cv2.copyMakeBorder(img, pad_h, pad_h, pad_w, pad_w,
                              cv2.BORDER_REFLECT_101)
pH, pW, _ = img_pad.shape

# ===================== ACCUMULATION BUFFERS ===========================
final_mask = np.zeros((pH, pW), dtype=np.float32)
weight_sum  = np.zeros((pH, pW), dtype=np.float32)

# ===================== PATCH GRID =====================================
y_positions = make_positions(pH, WINDOW, STRIDE)
x_positions = make_positions(pW, WINDOW, STRIDE)
n_rows      = len(y_positions)
n_cols      = len(x_positions)
total_patches = n_rows * n_cols

print(f"  Grid : {n_cols} cols × {n_rows} rows = {total_patches} patches\n")

# ===================== INFERENCE LOOP =================================
done = 0

for row_idx, y1 in enumerate(y_positions):
    row_label = f"Row {row_idx+1:>3}/{n_rows}"

    for col_idx, x1 in enumerate(x_positions):
        # --- extract window (always WINDOW×WINDOW because make_positions ensures fit) ---
        patch = img_pad[y1:y1+WINDOW, x1:x1+WINDOW]

        # safety pad (should never trigger, but defensive)
        dh = WINDOW - patch.shape[0]
        dw = WINDOW - patch.shape[1]
        if dh > 0 or dw > 0:
            patch = cv2.copyMakeBorder(patch, 0, dh, 0, dw,
                                        cv2.BORDER_REFLECT_101)

        # --- inference ---
        patch_tensor = preprocess_patch(patch)
        with torch.no_grad():
            pred = model(patch_tensor)

        # pred shape: (1, 1, MODEL_INPUT, MODEL_INPUT) or (1, MODEL_INPUT, MODEL_INPUT)
        pred = torch.sigmoid(pred).squeeze().cpu().numpy()  # (MODEL_INPUT, MODEL_INPUT)

        # resize prediction back to window size for correct spatial placement
        pred = cv2.resize(pred, (WINDOW, WINDOW), interpolation=cv2.INTER_LINEAR)
        pred = np.clip(pred, 0.05, 0.95)

        # --- weighted accumulation ---
        weighted_pred = pred * weight_map
        final_mask[y1:y1+WINDOW, x1:x1+WINDOW] += weighted_pred
        weight_sum [y1:y1+WINDOW, x1:x1+WINDOW] += weight_map

        done += 1

    # --- print row-complete progress ---
    cols_done  = done
    bar        = progress_bar(done, total_patches)
    print(f"  {row_label}  {bar}", flush=True)

print()

# ===================== NORMALISE ======================================
weight_sum[weight_sum == 0] = 1e-8
final_mask = final_mask / weight_sum

# ===================== REMOVE PADDING =================================
final_mask = final_mask[pad_h:pad_h+orig_H, pad_w:pad_w+orig_W]

# ===================== BINARIZE + CLEAN ===============================
binary_mask = (final_mask > 0.5).astype(np.uint8)

kernel      = np.ones((3, 3), np.uint8)
binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN,  kernel)
binary_mask = cv2.dilate(binary_mask, np.ones((2, 2), np.uint8), iterations=1)

print(f"  Unique values in output : {np.unique(binary_mask)}")

# ===================== SAVE ===========================================
cv2.imwrite("debug_raw_mask.png",  (final_mask * 255).astype(np.uint8))
cv2.imwrite("stitched_mask.png",   binary_mask * 255)
print("  Saved : stitched_mask.png")
print(f"{'='*55}\n")