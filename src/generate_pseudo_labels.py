import os
import cv2
import torch
import numpy as np
from .models import get_baseline_model
from .dataset import json_to_mask

# --- Cấu hình mặc định trỏ vào data/ ---
# Device selection with MPS support for macOS
if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
BASELINE_MODEL_PATH = "models/baseline_unetpp.pth"

ALL_FRAMES_DIR = "data/video"            # frames directory (or where you extract frames)
LABELED_MASK_DIR = "data/labeled_masks"  # masks saved from JSON (png per frame)
PSEUDO_MASK_DIR = "data/pseudo_masks"    # where generated pseudo masks are saved

os.makedirs(PSEUDO_MASK_DIR, exist_ok=True)

def calculate_optical_flow(prev_img, next_img):
    prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_img, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None,
                                        0.5, 3, 15, 3, 5, 1.2, 0)
    return flow

def propagate_mask(mask, flow):
    h, w = mask.shape[:2]
    grid_y, grid_x = np.mgrid[0:h, 0:w].astype(np.float32)
    new_x = grid_x + flow[..., 0]
    new_y = grid_y + flow[..., 1]
    warped_mask = cv2.remap(mask.astype(np.float32), new_x, new_y, cv2.INTER_NEAREST,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return (warped_mask > 0.5).astype(np.uint8)

def run_pseudo_label_generation(window=5):
    """
    Simple pseudo-label generator:
      - load baseline model
      - for each labeled frame in LABELED_MASK_DIR, propagate mask across nearby frames using optical flow
      - fallback: if no labeled mask, run baseline model to predict mask and save as pseudo
    """
    if not os.path.exists(BASELINE_MODEL_PATH):
        print(f"Missing baseline model at {BASELINE_MODEL_PATH}")
        return

    model = get_baseline_model()
    model.load_state_dict(torch.load(BASELINE_MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    frame_files = sorted([f for f in os.listdir(ALL_FRAMES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if not frame_files:
        print(f"No frames found under {ALL_FRAMES_DIR}")
        return

    # helper to save mask
    def save_mask(mask, out_path):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, (mask * 255).astype(np.uint8))

    for i, fname in enumerate(frame_files):
        frame_path = os.path.join(ALL_FRAMES_DIR, fname)
        labeled_mask_path = os.path.join(LABELED_MASK_DIR, os.path.splitext(fname)[0] + ".png")
        pseudo_out = os.path.join(PSEUDO_MASK_DIR, os.path.splitext(fname)[0] + ".png")

        if os.path.exists(labeled_mask_path):
            # propagate from labeled mask across a short window
            base_mask = cv2.imread(labeled_mask_path, cv2.IMREAD_GRAYSCALE).astype(np.uint8) // 255
            save_mask(base_mask, pseudo_out)  # also copy labeled -> pseudo folder
            # propagate forward/backward
            for j in range(1, window):
                # forward
                if i + j < len(frame_files):
                    prev = cv2.imread(frame_path)
                    nxt = cv2.imread(os.path.join(ALL_FRAMES_DIR, frame_files[i+j]))
                    flow = calculate_optical_flow(prev, nxt)
                    warped = propagate_mask(base_mask, flow)
                    outp = os.path.join(PSEUDO_MASK_DIR, os.path.splitext(frame_files[i+j])[0] + ".png")
                    save_mask(warped, outp)
                # backward
                if i - j >= 0:
                    prev = cv2.imread(os.path.join(ALL_FRAMES_DIR, frame_files[i-j]))
                    nxt = cv2.imread(frame_path)
                    flow = calculate_optical_flow(prev, nxt)
                    warped = propagate_mask(base_mask, flow)
                    outp = os.path.join(PSEUDO_MASK_DIR, os.path.splitext(frame_files[i-j])[0] + ".png")
                    save_mask(warped, outp)
        else:
            # fallback: use baseline to predict and save
            img = cv2.imread(frame_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_t = torch.from_numpy(img_rgb / 255.).permute(2,0,1).unsqueeze(0).float().to(DEVICE)
            with torch.no_grad():
                out = model(img_t)
                # model may return tensor or dict
                logits = out if isinstance(out, torch.Tensor) else out.get('out', next(iter(out.values())))
                probs = torch.sigmoid(logits)
                pred = (probs > 0.5).cpu().numpy().squeeze().astype(np.uint8)
                save_mask(pred, pseudo_out)

    print("Pseudo-label generation completed. Saved to", PSEUDO_MASK_DIR)

def _extract_logits(output):
    if isinstance(output, dict):
        return output.get('out', output.get('logits', next(iter(output.values()))))
    return output

def infer_and_save(model, img_path, out_path, device="cpu", size=(256,256)):
    img = cv2.imread(img_path)
    if img is None:
        return False
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, size)
    img_t = torch.from_numpy(img_resized/255.).permute(2,0,1).unsqueeze(0).float().to(device)
    with torch.no_grad():
        out = model(img_t)
        logits = _extract_logits(out)
        probs = torch.sigmoid(logits)
        pred = (probs > 0.5).cpu().numpy().squeeze().astype(np.uint8)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, (pred*255).astype(np.uint8))
    return True

if __name__ == '__main__':
    print("Run run_pseudo_label_generation() after updating paths if needed.")
