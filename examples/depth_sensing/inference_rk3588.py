"""
inference_rk3588.py
Run GLPN depth estimation on RK3588 using rknnlite.
Install on device: pip install rknn_toolkit_lite2
"""

import sys
import requests
import numpy as np
import cv2
from rknnlite.api import RKNNLite

# ─── CONFIG ──────────────────────────────────────────────────────────────────
RKNN_MODEL   = "glpn_nyu_depth.rknn"
INPUT_H      = 480
INPUT_W      = 640
COCO_URL     = "http://images.cocodataset.org/val2017/000000039769.jpg"
SAVE_INPUT   = "sample_input.jpg"
SAVE_DEPTH   = "sample_depth.png"
SAVE_COMBO   = "sample_combined.jpg"

# ─── INIT ────────────────────────────────────────────────────────────────────
rknn = RKNNLite(verbose=False)
rknn.load_rknn(RKNN_MODEL)

ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_ALL)
assert ret == 0, "init_runtime() failed"
print("Runtime initialised on RK3588 NPU")


# ─── DOWNLOAD ────────────────────────────────────────────────────────────────
def download_image(url: str, save_path: str) -> np.ndarray:
    print(f"Downloading {url} ...")
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    arr   = np.frombuffer(r.content, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    cv2.imwrite(save_path, frame)
    print(f"Saved → {save_path}  ({frame.shape[1]}×{frame.shape[0]})")
    return frame


# ─── PREPROCESS ──────────────────────────────────────────────────────────────
def preprocess(frame_bgr: np.ndarray) -> tuple[np.ndarray, tuple]:
    """
    BGR uint8 [H, W, 3]
        → resize
        → RGB float32 [-1, 1]
        → NHWC [1, H, W, 3]

    rknn.inference() always expects NHWC numpy input.
    The ONNX graph is NCHW internally; RKNN transposes it
    automatically when data_format="nhwc" is passed.
    """
    h, w          = frame_bgr.shape[:2]
    original_size = (w, h)

    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (INPUT_W, INPUT_H), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0      # [0, 1]
    img = (img - 0.5) / 0.5                   # [-1, 1]
    img = np.expand_dims(img, axis=0)          # HWC → NHWC [1, H, W, 3]

    return np.ascontiguousarray(img), original_size


# ─── INFERENCE ───────────────────────────────────────────────────────────────
def infer(x_nhwc: np.ndarray) -> np.ndarray:
    """
    Input:  NHWC float32 [1, 480, 640, 3] in [-1, 1]
    Output: depth float32 [480, 640]
    """
    outputs = rknn.inference(inputs=[x_nhwc], data_format="nhwc")
    out     = outputs[0]

    print(f"Raw output shape: {out.shape}  dtype: {out.dtype}")

    if out.ndim == 4:
        out = out[0]   # remove batch → [C or 1, H, W]
    if out.ndim == 3:
        out = out[0]   # remove channel/pred dim → [H, W]

    print(f"Depth range: min={out.min():.4f}  max={out.max():.4f}")
    return out.astype(np.float32)


# ─── POSTPROCESS ─────────────────────────────────────────────────────────────
def postprocess(depth: np.ndarray, original_size: tuple) -> np.ndarray:
    """
    Upscale to original size, normalise to uint8, apply Inferno colormap.
    np.clip guards against bicubic overshoot producing banding artifacts.
    """
    depth = cv2.resize(depth, original_size, interpolation=cv2.INTER_CUBIC)

    d_min, d_max = depth.min(), depth.max()
    if d_max - d_min > 1e-6:
        depth = (depth - d_min) / (d_max - d_min)

    depth = np.clip(depth, 0.0, 1.0)
    gray  = (depth * 255).astype(np.uint8)
    return cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)


# ─── SIDE-BY-SIDE ────────────────────────────────────────────────────────────
def save_combined(frame_bgr: np.ndarray, depth_color: np.ndarray, path: str = SAVE_COMBO):
    depth_resized = cv2.resize(depth_color, (frame_bgr.shape[1], frame_bgr.shape[0]))
    combined      = np.hstack([frame_bgr, depth_resized])
    cv2.imwrite(path, combined)
    print(f"Side-by-side → {path}")


# ─── PIPELINE ────────────────────────────────────────────────────────────────
def run(frame_bgr: np.ndarray):
    x_nhwc, original_size = preprocess(frame_bgr)
    raw_depth             = infer(x_nhwc)
    depth_color           = postprocess(raw_depth, original_size)

    cv2.imwrite(SAVE_DEPTH, depth_color)
    print(f"Depth saved  → {SAVE_DEPTH}")

    save_combined(frame_bgr, depth_color)


# ─── ENTRY POINT ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) > 1:
        frame = cv2.imread(sys.argv[1])
        assert frame is not None, f"Cannot read: {sys.argv[1]}"
    else:
        frame = download_image(COCO_URL, SAVE_INPUT)

    run(frame)
    rknn.release()
