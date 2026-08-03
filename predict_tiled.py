import torch
import cv2
import numpy as np
import argparse
import os
from tqdm import tqdm
from model import UNet

"""
Tiled inference script for S.T.I.T.C.H floorplan segmentation.

Usage:
  python predict_tiled.py --image path/to/floorplan.jpg

Optional:
  --model   path to weights file (default: unet.pth)
  --stride  patch stride in pixels (default: 128, lower = slower but smoother)
  --output  output filename (default: stitched_mask.png)
"""

# ------------------------------ CONFIG ------------------------------
PATCH_SIZE = 256
# --------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image",  required=True,              help="Path to input floorplan image")
    p.add_argument("--model",  default="unet.pth",         help="Path to model weights")
    p.add_argument("--stride", type=int, default=128,      help="Patch stride (lower = smoother)")
    p.add_argument("--output", default="stitched_mask.png", help="Output filename")
    return p.parse_args()


def preprocess_patch(patch, device):
    patch = patch.astype(np.float32) / 255.0
    patch = np.transpose(patch, (2, 0, 1))
    return torch.from_numpy(patch).unsqueeze(0).to(device)


def create_weight_map(size):
    y, x = np.ogrid[-1:1:size*1j, -1:1:size*1j]
    weight = np.exp(-(x**2 + y**2) * 4)
    return weight.astype(np.float32)


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model weights not found: {args.model}")
    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")

    model = UNet()
    model.load_state_dict(torch.load(
        args.model, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    print(f"Model loaded: {args.model}")

    img = cv2.imread(args.image)
    if img is None:
        raise ValueError(f"Could not read image: {args.image}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    H, W, _ = img.shape
    print(f"Original H, W: {H}, {W}")

    # 5% reflective padding — reduces edge artifacts
    pad_h = int(0.05 * H)
    pad_w = int(0.05 * W)
    img = cv2.copyMakeBorder(img, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_REFLECT_101)
    padded_H, padded_W, _ = img.shape
    print(f"Padded H, W: {padded_H}, {padded_W}")

    weight_map = create_weight_map(PATCH_SIZE)
    final_mask = np.zeros((padded_H, padded_W), dtype=np.float32)
    weight_sum = np.zeros((padded_H, padded_W), dtype=np.float32)

    STRIDE = args.stride
    y_positions = list(range(0, padded_H - PATCH_SIZE, STRIDE))
    x_positions = list(range(0, padded_W - PATCH_SIZE, STRIDE))
    if y_positions[-1] != padded_H - PATCH_SIZE:
        y_positions.append(padded_H - PATCH_SIZE)
    if x_positions[-1] != padded_W - PATCH_SIZE:
        x_positions.append(padded_W - PATCH_SIZE)

    total_patches = len(y_positions) * len(x_positions)
    print(f"Running inference on {total_patches} patches...")

    pbar = tqdm(total=total_patches, desc="Patching", unit="patch")

    for y1 in y_positions:
        for x1 in x_positions:
            patch = img[y1:y1+PATCH_SIZE, x1:x1+PATCH_SIZE]
            if patch.shape[0] != PATCH_SIZE or patch.shape[1] != PATCH_SIZE:
                ph = PATCH_SIZE - patch.shape[0]
                pw = PATCH_SIZE - patch.shape[1]
                patch = cv2.copyMakeBorder(patch, 0, ph, 0, pw, cv2.BORDER_REFLECT_101)

            patch_tensor = preprocess_patch(patch, device)
            with torch.no_grad():
                pred = model(patch_tensor)
            pred = torch.sigmoid(pred).squeeze().cpu().numpy()
            pred = np.clip(pred, 0.05, 0.95)

            final_mask[y1:y1+PATCH_SIZE, x1:x1+PATCH_SIZE] += pred * weight_map
            weight_sum[y1:y1+PATCH_SIZE, x1:x1+PATCH_SIZE] += weight_map

            pbar.update(1)

    pbar.close()

    weight_sum[weight_sum == 0] = 1e-8
    final_mask = final_mask / weight_sum

    # remove padding
    final_mask = final_mask[pad_h:pad_h + H, pad_w:pad_w + W]

    cv2.imwrite("debug_raw_mask.png", (final_mask * 255).astype(np.uint8))
    print("Debug mask saved: debug_raw_mask.png")

    # binarize
    binary_mask = (final_mask > 0.5).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    binary_mask = cv2.dilate(binary_mask, np.ones((2, 2), np.uint8), iterations=1)

    # remove text / symbols
    # text blobs are small AND roughly square; walls are large AND elongated
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    cleaned = np.zeros_like(binary_mask)
    for i in range(1, num_labels):
        area   = stats[i, cv2.CC_STAT_AREA]
        w_box  = stats[i, cv2.CC_STAT_WIDTH]
        h_box  = stats[i, cv2.CC_STAT_HEIGHT]
        aspect = max(w_box, h_box) / (min(w_box, h_box) + 1e-6)
        if area < 80 and aspect < 3.0:
            continue
        cleaned[labels == i] = 1
    binary_mask = cleaned

    cv2.imwrite(args.output, binary_mask * 255)
    print(f"✅ Saved: {args.output}")


if __name__ == "__main__":
    main()
