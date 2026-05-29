# import torch
# import cv2
# import numpy as np

# #--------------------------CONFIG------------------------------------
# MODEL_PATH = "unet.pth"   # change if needed
# IMAGE_PATH = r"C:\Users\Asus\parser-model\test.jpg"

# PATCH_SIZE = 256
# STRIDE = 150   # 50% overlap (IMPORTANT)

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# #---------------------------------------------------------------------

# from model import UNet   # import  model

# model = UNet()          # create model

# state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
# model.load_state_dict(state_dict)  # load weights
# model.to(DEVICE)
# model.eval()

# def preprocess_patch(patch):
#     # patch = patch / 255.0   # this one worked when the image was a bit smaller 
#     patch = patch.astype(np.float32) / 255.0
#     patch = np.transpose(patch, (2, 0, 1))
#     patch = torch.from_numpy(patch).unsqueeze(0)
#     return patch.to(DEVICE)

# #here we are using gaussian blur to blend thats why the corners are getting fucked 
# def create_weight_map(size):
#     h, w = size, size
#     y, x = np.ogrid[-1:1:h*1j, -1:1:w*1j]
#     weight = np.exp(-(x**2 + y**2) * 4)   # gaussian
#     return weight.astype(np.float32)

# weight_map = create_weight_map(PATCH_SIZE)

# img = cv2.imread(IMAGE_PATH)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# H, W, _ = img.shape
# print("H, W:", H, W)

# final_mask = np.zeros((H, W), dtype=np.float32)
# weight_sum = np.zeros((H, W), dtype=np.float32)

# y_positions = list(range(0, H - PATCH_SIZE, STRIDE))
# x_positions = list(range(0, W - PATCH_SIZE, STRIDE))

# if y_positions[-1] != H - PATCH_SIZE:
#     y_positions.append(H - PATCH_SIZE)

# if x_positions[-1] != W - PATCH_SIZE:
#     x_positions.append(W - PATCH_SIZE)

# for y1 in y_positions:
#     for x1 in x_positions:

#         print("Processing patch at:", y1, x1)

#         patch = img[y1:y1+PATCH_SIZE, x1:x1+PATCH_SIZE]
#         if patch.shape[0] != PATCH_SIZE or patch.shape[1] != PATCH_SIZE:
#             pad_h = PATCH_SIZE - patch.shape[0]
#             pad_w = PATCH_SIZE - patch.shape[1]
#             patch = cv2.copyMakeBorder(
#                 patch, 
#                 0, pad_h, 
#                 0, pad_w, 
#                 cv2.BORDER_REFLECT
#             )

#         patch_tensor = preprocess_patch(patch)

#         with torch.no_grad():
#             pred = model(patch_tensor)

#         pred = torch.sigmoid(pred).squeeze().cpu().numpy()
#         pred = np.clip(pred, 0.05, 0.95)

#         weighted_pred = pred * weight_map

#         final_mask[y1:y1+PATCH_SIZE, x1:x1+PATCH_SIZE] += weighted_pred
#         weight_sum[y1:y1+PATCH_SIZE, x1:x1+PATCH_SIZE] += weight_map
        

# print("Before normalize min/max:", final_mask.min(), final_mask.max())
# print("Weight sum min/max:", weight_sum.min(), weight_sum.max())

# weight_sum[weight_sum == 0] = 1e-8
# final_mask = final_mask / weight_sum

# # DEBUG (visualize raw probabilities)
# cv2.imwrite("debug_raw_mask.png", (final_mask * 255).astype(np.uint8))
# binary_mask = (final_mask > 0.5).astype(np.uint8)


# kernel = np.ones((3,3), np.uint8)
# binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
# binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
# binary_mask = cv2.dilate(binary_mask, np.ones((2,2), np.uint8), iterations=1)

# print("Binary mask unique values:", np.unique(binary_mask))
# cv2.imwrite("stitched_mask.png", binary_mask * 255)

# print("Saved: stitched_mask.png")










# this is the original predict.py code without tiling, for reference and comparison

# import torch
# import cv2
# import numpy as np

# # -------------------------- CONFIG ------------------------------------
# MODEL_PATH = "unet.pth"
# IMAGE_PATH = r"C:\Users\Asus\parser-model\test.jpg"

# PATCH_SIZE = 256
# STRIDE = 128   #50% for each block 

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # ----------------------------------------------------------------------

# from model import UNet

# model = UNet()
# state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
# model.load_state_dict(state_dict)
# model.to(DEVICE)
# model.eval()


# def preprocess_patch(patch):
#     patch = patch.astype(np.float32) / 255.0
#     patch = np.transpose(patch, (2, 0, 1))
#     patch = torch.from_numpy(patch).unsqueeze(0)
#     return patch.to(DEVICE)


# def create_weight_map(size):
#     h, w = size, size
#     y, x = np.ogrid[-1:1:h*1j, -1:1:w*1j]
#     weight = np.exp(-(x**2 + y**2) * 4)
#     return weight.astype(np.float32)


# weight_map = create_weight_map(PATCH_SIZE)

# # -------------------- LOAD IMAGE --------------------
# img = cv2.imread(IMAGE_PATH)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# H, W, _ = img.shape
# print("Original H, W:", H, W)

# # -------------------- ADD 5% PADDING --------------------
# pad_h = int(0.05 * H)
# pad_w = int(0.05 * W)

# img = cv2.copyMakeBorder(
#     img,
#     pad_h, pad_h,
#     pad_w, pad_w,
#     cv2.BORDER_REFLECT_101
# )

# padded_H, padded_W, _ = img.shape
# print("Padded H, W:", padded_H, padded_W)

# # -------------------- INIT MASKS --------------------
# final_mask = np.zeros((padded_H, padded_W), dtype=np.float32)
# weight_sum = np.zeros((padded_H, padded_W), dtype=np.float32)

# # -------------------- PATCH GRID --------------------
# y_positions = list(range(0, padded_H - PATCH_SIZE, STRIDE))
# x_positions = list(range(0, padded_W - PATCH_SIZE, STRIDE))

# if y_positions[-1] != padded_H - PATCH_SIZE:
#     y_positions.append(padded_H - PATCH_SIZE)

# if x_positions[-1] != padded_W - PATCH_SIZE:
#     x_positions.append(padded_W - PATCH_SIZE)

# # -------------------- INFERENCE LOOP --------------------
# for y1 in y_positions:
#     for x1 in x_positions:

#         print("Processing patch at:", y1, x1)

#         patch = img[y1:y1+PATCH_SIZE, x1:x1+PATCH_SIZE]

#         if patch.shape[0] != PATCH_SIZE or patch.shape[1] != PATCH_SIZE:
#             pad_h2 = PATCH_SIZE - patch.shape[0]
#             pad_w2 = PATCH_SIZE - patch.shape[1]
#             patch = cv2.copyMakeBorder(
#                 patch,
#                 0, pad_h2,
#                 0, pad_w2,
#                 cv2.BORDER_REFLECT_101
#             )

#         patch_tensor = preprocess_patch(patch)

#         with torch.no_grad():
#             pred = model(patch_tensor)

#         pred = torch.sigmoid(pred).squeeze().cpu().numpy()
#         pred = np.clip(pred, 0.05, 0.95)

#         weighted_pred = pred * weight_map

#         final_mask[y1:y1+PATCH_SIZE, x1:x1+PATCH_SIZE] += weighted_pred
#         weight_sum[y1:y1+PATCH_SIZE, x1:x1+PATCH_SIZE] += weight_map

# # -------------------- NORMALIZE --------------------
# print("Before normalize min/max:", final_mask.min(), final_mask.max())
# print("Weight sum min/max:", weight_sum.min(), weight_sum.max())

# weight_sum[weight_sum == 0] = 1e-8
# final_mask = final_mask / weight_sum

# # -------------------- REMOVE PADDING --------------------
# final_mask = final_mask[
#     pad_h:pad_h + H,
#     pad_w:pad_w + W
# ]

# # -------------------- SAVE DEBUG --------------------
# cv2.imwrite("debug_raw_mask.png", (final_mask * 255).astype(np.uint8))

# # -------------------- BINARIZE --------------------
# binary_mask = (final_mask > 0.5).astype(np.uint8)

# kernel = np.ones((3,3), np.uint8)
# binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
# binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
# binary_mask = cv2.dilate(binary_mask, np.ones((2,2), np.uint8), iterations=1)

# print("Binary mask unique values:", np.unique(binary_mask))

# cv2.imwrite("stitched_mask.png", binary_mask * 255)

# print("Saved: stitched_mask.png")













# this one did not work at all 

# import torch
# import cv2
# import numpy as np

# # --------------------------CONFIG------------------------------------
# MODEL_PATH = "unet.pth"
# IMAGE_PATH = r"test.jpg"

# # The model was trained on 256x256 patches.
# # We normalize the image to this canonical resolution first,
# # then tile using percentages so it works for any image size.
# CANONICAL_SIZE = 1024   # internal working resolution (longest side)
# PATCH_PERCENT  = 0.20   # each patch = 20% of the canonical dimension
# OVERLAP_RATIO  = 0.50   # stride = patch * (1 - overlap) = 50% of patch
#                         # matches STITCH paper recommendation (O = 0.5)
# MODEL_INPUT    = 256    # what the UNet expects (resize patch to this)

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # ---------------------------------------------------------------------

# from model import UNet

# model = UNet()
# model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
# model.to(DEVICE)
# model.eval()


# def preprocess_patch(patch):
#     """Resize patch to model input size and convert to tensor."""
#     patch = cv2.resize(patch, (MODEL_INPUT, MODEL_INPUT),
#                        interpolation=cv2.INTER_LINEAR)
#     patch = patch.astype(np.float32) / 255.0
#     patch = np.transpose(patch, (2, 0, 1))
#     return torch.from_numpy(patch).unsqueeze(0).to(DEVICE)


# def create_weight_map(size):
#     """Gaussian weight map so patch centres matter more than edges."""
#     h, w = size, size
#     y, x = np.ogrid[-1:1:h*1j, -1:1:w*1j]
#     weight = np.exp(-(x**2 + y**2) * 4)
#     return weight.astype(np.float32)


# # -------------------- LOAD IMAGE --------------------
# img = cv2.imread(IMAGE_PATH)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# orig_H, orig_W, _ = img.shape
# print(f"Original size: {orig_W}x{orig_H}")

# # -------------------- NORMALIZE TO CANONICAL SIZE --------------------
# # Scale so the longest side = CANONICAL_SIZE.
# # This makes patch/stride percentages map to sensible pixel counts
# # regardless of whether the input is 300px or 8000px.
# scale = CANONICAL_SIZE / max(orig_H, orig_W)
# canon_W = int(round(orig_W * scale))
# canon_H = int(round(orig_H * scale))
# img_canon = cv2.resize(img, (canon_W, canon_H), interpolation=cv2.INTER_LINEAR)
# print(f"Canonical size: {canon_W}x{canon_H}  (scale={scale:.3f})")

# # -------------------- COMPUTE PATCH / STRIDE --------------------
# # Use the SMALLER of the two canonical dimensions so patches are always
# # fully inside the image even on portrait/landscape maps.
# short_side   = min(canon_H, canon_W)
# PATCH_SIZE   = max(64, int(round(short_side * PATCH_PERCENT)))
# STRIDE       = max(32, int(round(PATCH_SIZE * (1.0 - OVERLAP_RATIO))))
# print(f"Patch size: {PATCH_SIZE}px  Stride: {STRIDE}px  "
#       f"(overlap={OVERLAP_RATIO:.0%})")

# weight_map = create_weight_map(PATCH_SIZE)

# # -------------------- ADD 5% PADDING --------------------
# pad_h = int(0.05 * canon_H)
# pad_w = int(0.05 * canon_W)
# img_padded = cv2.copyMakeBorder(
#     img_canon, pad_h, pad_h, pad_w, pad_w,
#     cv2.BORDER_REFLECT_101
# )
# pH, pW, _ = img_padded.shape
# print(f"Padded size: {pW}x{pH}")

# # -------------------- INIT ACCUMULATION BUFFERS --------------------
# final_mask = np.zeros((pH, pW), dtype=np.float32)
# weight_sum  = np.zeros((pH, pW), dtype=np.float32)

# # -------------------- PATCH GRID --------------------
# def make_positions(total, patch, stride):
#     """Return start positions so the last patch always covers the edge."""
#     positions = list(range(0, total - patch, stride))
#     if not positions or positions[-1] != total - patch:
#         positions.append(total - patch)
#     return positions

# y_positions = make_positions(pH, PATCH_SIZE, STRIDE)
# x_positions = make_positions(pW, PATCH_SIZE, STRIDE)
# total_patches = len(y_positions) * len(x_positions)
# print(f"Total patches: {total_patches} "
#       f"({len(x_positions)} cols × {len(y_positions)} rows)")

# # -------------------- INFERENCE LOOP --------------------
# for i, y1 in enumerate(y_positions):
#     for j, x1 in enumerate(x_positions):
#         patch = img_padded[y1:y1+PATCH_SIZE, x1:x1+PATCH_SIZE]

#         # Safety pad if edge patch is too small (shouldn't happen but just in case)
#         if patch.shape[0] != PATCH_SIZE or patch.shape[1] != PATCH_SIZE:
#             pad_h2 = PATCH_SIZE - patch.shape[0]
#             pad_w2 = PATCH_SIZE - patch.shape[1]
#             patch = cv2.copyMakeBorder(
#                 patch, 0, pad_h2, 0, pad_w2,
#                 cv2.BORDER_REFLECT_101
#             )

#         patch_tensor = preprocess_patch(patch)

#         with torch.no_grad():
#             pred = model(patch_tensor)

#         # Get probability map and resize back to PATCH_SIZE
#         pred = torch.sigmoid(pred).squeeze().cpu().numpy()
#         pred = cv2.resize(pred, (PATCH_SIZE, PATCH_SIZE),
#                           interpolation=cv2.INTER_LINEAR)
#         pred = np.clip(pred, 0.05, 0.95)

#         weighted_pred = pred * weight_map
#         final_mask[y1:y1+PATCH_SIZE, x1:x1+PATCH_SIZE] += weighted_pred
#         weight_sum [y1:y1+PATCH_SIZE, x1:x1+PATCH_SIZE] += weight_map

# # -------------------- NORMALIZE --------------------
# weight_sum[weight_sum == 0] = 1e-8
# final_mask = final_mask / weight_sum

# # -------------------- REMOVE PADDING --------------------
# final_mask = final_mask[pad_h:pad_h+canon_H, pad_w:pad_w+canon_W]

# # -------------------- SCALE BACK TO ORIGINAL SIZE --------------------
# final_mask = cv2.resize(final_mask, (orig_W, orig_H),
#                         interpolation=cv2.INTER_LINEAR)

# # -------------------- BINARIZE + CLEAN --------------------
# binary_mask = (final_mask > 0.5).astype(np.uint8)

# kernel = np.ones((3, 3), np.uint8)
# binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
# binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN,  kernel)
# binary_mask = cv2.dilate(binary_mask, np.ones((2, 2), np.uint8), iterations=1)

# print("Unique values in output:", np.unique(binary_mask))

# cv2.imwrite("debug_raw_mask.png",  (final_mask * 255).astype(np.uint8))
# cv2.imwrite("stitched_mask.png",   binary_mask * 255)
# print("Saved: stitched_mask.png")










# this one works but this  works on the onces that re lil low on resolution when the total number of patches goes over the threshold of 100 it stops working and akes a lot of problems in the image that we are prsing the stitched masks mecomes a mess, trying to have a maski that will not go ove rthe number of 45 because when ever we have masks that are 35 to 40 patches it works reallt great 

# import torch
# import cv2
# import numpy as np
# import math

# # ========================== CONFIG ====================================
# MODEL_PATH  = "unet.pth"
# IMAGE_PATH  = r"test.jpg"

# # ------ Percentage-based tiling (your universal design) ------
# # Window that scans across the image = 20% of the shorter side.
# # Clamped: never smaller than 256 (model training size),
# #           never larger than 512 (avoids too-zoomed-out view).
# PATCH_PERCENT = 0.20   # window = 20% of shorter side
# PATCH_MIN     = 256    # absolute floor  (px)
# PATCH_MAX     = 512    # absolute ceiling (px)

# # Stride = 50% of window → overlap ratio O = 0.5
# # Matches STITCH paper recommendation and the original 256/128 pair.
# OVERLAP_RATIO = 0.50

# # The model was trained on exactly 256×256 images.
# # Every window is resized to this before inference.
# MODEL_INPUT   = 256

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # ======================================================================

# from model import UNet

# model = UNet()
# model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
# model.to(DEVICE)
# model.eval()


# # ---------- helpers ----------

# def preprocess_patch(patch):
#     """Resize window to model input size and convert to normalised tensor."""
#     patch = cv2.resize(patch, (MODEL_INPUT, MODEL_INPUT),
#                        interpolation=cv2.INTER_AREA)   # AREA = best for downscale
#     patch = patch.astype(np.float32) / 255.0
#     patch = np.transpose(patch, (2, 0, 1))
#     return torch.from_numpy(patch).unsqueeze(0).to(DEVICE)


# def create_weight_map(h, w):
#     """Gaussian weight map: patch centres matter more than edges.
#        Accepts non-square windows so the weight map always matches the window."""
#     y, x = np.ogrid[-1:1:h*1j, -1:1:w*1j]
#     weight = np.exp(-(x**2 + y**2) * 4)
#     return weight.astype(np.float32)


# def make_positions(total_len, window, stride):
#     """Return every start offset so the last window always covers the far edge."""
#     positions = list(range(0, total_len - window, stride))
#     if not positions or positions[-1] + window < total_len:
#         positions.append(total_len - window)
#     return positions


# def progress_bar(done, total, width=30):
#     """Return an ASCII progress bar string."""
#     filled = int(width * done / total)
#     bar    = "█" * filled + "░" * (width - filled)
#     pct    = 100 * done / total
#     return f"[{bar}] {pct:5.1f}%  ({done}/{total})"


# # ======================== LOAD IMAGE ==================================
# img = cv2.imread(IMAGE_PATH)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# orig_H, orig_W, _ = img.shape
# print(f"\n{'='*55}")
# print(f"  Image loaded  :  {orig_W} × {orig_H} px")

# # ======================== COMPUTE WINDOW SIZE =========================
# # Use the SHORTER side for the percentage so the window always fits
# # inside the image on both axes.
# short_side  = min(orig_H, orig_W)
# raw_window  = int(round(short_side * PATCH_PERCENT))
# WINDOW      = max(PATCH_MIN, min(PATCH_MAX, raw_window))
# STRIDE      = max(1, int(round(WINDOW * (1.0 - OVERLAP_RATIO))))

# print(f"  Short side    :  {short_side} px")
# print(f"  Window (20%)  :  {raw_window} px  →  clamped to {WINDOW} px")
# print(f"  Stride (50%)  :  {STRIDE} px  (overlap = {OVERLAP_RATIO:.0%})")
# print(f"  Model input   :  {MODEL_INPUT} × {MODEL_INPUT} px")
# print(f"{'='*55}\n")

# # Pre-compute weight map once (shape = WINDOW × WINDOW)
# weight_map = create_weight_map(WINDOW, WINDOW)

# # ===================== ADD 5% PADDING =================================
# pad_h = int(0.05 * orig_H)
# pad_w = int(0.05 * orig_W)
# img_pad = cv2.copyMakeBorder(img, pad_h, pad_h, pad_w, pad_w,
#                               cv2.BORDER_REFLECT_101)
# pH, pW, _ = img_pad.shape

# # ===================== ACCUMULATION BUFFERS ===========================
# final_mask = np.zeros((pH, pW), dtype=np.float32)
# weight_sum  = np.zeros((pH, pW), dtype=np.float32)

# # ===================== PATCH GRID =====================================
# y_positions = make_positions(pH, WINDOW, STRIDE)
# x_positions = make_positions(pW, WINDOW, STRIDE)
# n_rows      = len(y_positions)
# n_cols      = len(x_positions)
# total_patches = n_rows * n_cols

# print(f"  Grid : {n_cols} cols × {n_rows} rows = {total_patches} patches\n")

# # ===================== INFERENCE LOOP =================================
# done = 0

# for row_idx, y1 in enumerate(y_positions):
#     row_label = f"Row {row_idx+1:>3}/{n_rows}"

#     for col_idx, x1 in enumerate(x_positions):
#         # --- extract window (always WINDOW×WINDOW because make_positions ensures fit) ---
#         patch = img_pad[y1:y1+WINDOW, x1:x1+WINDOW]

#         # safety pad (should never trigger, but defensive)
#         dh = WINDOW - patch.shape[0]
#         dw = WINDOW - patch.shape[1]
#         if dh > 0 or dw > 0:
#             patch = cv2.copyMakeBorder(patch, 0, dh, 0, dw,
#                                         cv2.BORDER_REFLECT_101)

#         # --- inference ---
#         patch_tensor = preprocess_patch(patch)
#         with torch.no_grad():
#             pred = model(patch_tensor)

#         # pred shape: (1, 1, MODEL_INPUT, MODEL_INPUT) or (1, MODEL_INPUT, MODEL_INPUT)
#         pred = torch.sigmoid(pred).squeeze().cpu().numpy()  # (MODEL_INPUT, MODEL_INPUT)

#         # resize prediction back to window size for correct spatial placement
#         pred = cv2.resize(pred, (WINDOW, WINDOW), interpolation=cv2.INTER_LINEAR)
#         pred = np.clip(pred, 0.05, 0.95)

#         # --- weighted accumulation ---
#         weighted_pred = pred * weight_map
#         final_mask[y1:y1+WINDOW, x1:x1+WINDOW] += weighted_pred
#         weight_sum [y1:y1+WINDOW, x1:x1+WINDOW] += weight_map

#         done += 1

#     # --- print row-complete progress ---
#     cols_done  = done
#     bar        = progress_bar(done, total_patches)
#     print(f"  {row_label}  {bar}", flush=True)

# print()

# # ===================== NORMALISE ======================================
# weight_sum[weight_sum == 0] = 1e-8
# final_mask = final_mask / weight_sum

# # ===================== REMOVE PADDING =================================
# final_mask = final_mask[pad_h:pad_h+orig_H, pad_w:pad_w+orig_W]

# # ===================== BINARIZE + CLEAN ===============================
# binary_mask = (final_mask > 0.5).astype(np.uint8)

# kernel      = np.ones((3, 3), np.uint8)
# binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
# binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN,  kernel)
# binary_mask = cv2.dilate(binary_mask, np.ones((2, 2), np.uint8), iterations=1)

# print(f"  Unique values in output : {np.unique(binary_mask)}")

# # ===================== SAVE ===========================================
# cv2.imwrite("debug_raw_mask.png",  (final_mask * 255).astype(np.uint8))
# cv2.imwrite("stitched_mask.png",   binary_mask * 255)
# print("  Saved : stitched_mask.png")
# print(f"{'='*55}\n")













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