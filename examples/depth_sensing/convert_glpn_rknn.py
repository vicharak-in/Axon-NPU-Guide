"""
convert_glpn_rknn.py
GLPN-NYU → TorchScript .pt → ONNX → .rknn
Run on PC (not on RK3588).
Requires: torch, transformers, rknn-toolkit2, onnx==1.18.0
"""

import sys

import cv2
import numpy as np
import torch
from transformers import GLPNForDepthEstimation

# ─── CONFIG ──────────────────────────────────────────────────────────────────
MODEL_ID = "vinvino02/glpn-nyu"
PT_MODEL = "glpn_nyu_depth.pt"
ONNX_MODEL = "glpn_nyu_depth.onnx"
RKNN_MODEL = "glpn_nyu_depth.rknn"
TARGET_PLATFORM = "rk3588"
INPUT_H = 480
INPUT_W = 640


# ─── WRAPPER ─────────────────────────────────────────────────────────────────
class GLPNWrapper(torch.nn.Module):
    """
    Contract: input  = NCHW float32 in [-1, 1]
              output = [1, H, W] float32 depth
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values=x).predicted_depth


# ─── STEP 1: TRACE ───────────────────────────────────────────────────────────
def trace():
    print("\n=== Step 1: Tracing GLPN to TorchScript ===")

    model = GLPNForDepthEstimation.from_pretrained(
        MODEL_ID,
        attn_implementation="eager",
    )
    model.eval()

    dummy = torch.randn(1, 3, INPUT_H, INPUT_W)
    wrapper = GLPNWrapper(model)
    wrapper.eval()

    with torch.no_grad():
        out = wrapper(dummy)
        print(f"Wrapper output shape: {out.shape}")
        assert out.ndim == 3, f"Expected 3D output, got {out.shape}"
        traced = torch.jit.trace(wrapper, dummy)

    torch.jit.save(traced, PT_MODEL)
    print(f"Saved → {PT_MODEL}")

    # Verify round-trip
    loaded = torch.jit.load(PT_MODEL)
    loaded.eval()
    with torch.no_grad():
        out2 = loaded(dummy)
    print(f"Round-trip output shape: {out2.shape}")
    print(f"Max diff from original:  {(out - out2).abs().max().item():.6f}")


# ─── STEP 2: EXPORT TO ONNX ──────────────────────────────────────────────────
def export_onnx():
    print("\n=== Step 2: Exporting TorchScript → ONNX ===")

    loaded = torch.jit.load(PT_MODEL)
    loaded.eval()

    dummy = torch.randn(1, 3, INPUT_H, INPUT_W)

    with torch.no_grad():
        torch.onnx.export(
            loaded,
            dummy,
            ONNX_MODEL,
            input_names=["pixel_values"],
            output_names=["predicted_depth"],
            opset_version=12,  # RKNN supports up to opset 19; 12 is safest
            do_constant_folding=True,
            dynamic_axes=None,  # static shapes — RKNN requires this
        )

    print(f"Saved → {ONNX_MODEL}")

    # Verify ONNX graph is valid
    import onnx

    model_onnx = onnx.load(ONNX_MODEL)
    onnx.checker.check_model(model_onnx)
    print("ONNX graph check passed")


# ─── STEP 3: EXPORT TO RKNN ──────────────────────────────────────────────────
def export_rknn():
    print("\n=== Step 3: ONNX → RKNN ===")
    from rknn.api import RKNN

    rknn = RKNN(verbose=True)

    # mean=0 std=1 → no normalisation inside RKNN.
    # Input is already float32 [-1, 1] from our preprocess().
    rknn.config(
        mean_values=[[0.0, 0.0, 0.0]],
        std_values=[[1.0, 1.0, 1.0]],
        target_platform=TARGET_PLATFORM,
        quantized_method="channel",
        optimization_level=3,
    )

    print("--> load_onnx")
    # input_size_list not needed — shape is embedded in the ONNX graph
    ret = rknn.load_onnx(model=ONNX_MODEL)
    assert ret == 0, "load_onnx failed"

    print("--> build")
    ret = rknn.build(do_quantization=False)
    assert ret == 0, "build failed"

    print("--> export_rknn")
    ret = rknn.export_rknn(RKNN_MODEL)
    assert ret == 0, "export_rknn failed"

    print(f"Saved → {RKNN_MODEL}")
    rknn.release()


# ─── STEP 4: PC SIMULATOR VERIFICATION ───────────────────────────────────────
def verify_pc(image_path: str):
    print("\n=== Step 4: PC Simulator Verification ===")
    from rknn.api import RKNN

    frame = cv2.imread(image_path)
    assert frame is not None, f"Cannot read {image_path}"

    x_nchw, _ = preprocess_pytorch(frame)
    x_nhwc, orig_size = preprocess_rknn(frame)

    # ── PyTorch reference ────────────────────────────────────────────────────
    loaded = torch.jit.load(PT_MODEL)
    loaded.eval()
    with torch.no_grad():
        depth_pt = loaded(torch.from_numpy(x_nchw)).squeeze().numpy()

    # ── RKNN simulator ───────────────────────────────────────────────────────
    rknn = RKNN()
    rknn.config(
        mean_values=[[0.0, 0.0, 0.0]],
        std_values=[[1.0, 1.0, 1.0]],
        target_platform=TARGET_PLATFORM,
        quantized_method="channel",
        optimization_level=3,
    )
    assert rknn.load_onnx(model=ONNX_MODEL) == 0, "load_onnx failed"
    assert rknn.build(do_quantization=False) == 0, "build failed"
    assert rknn.init_runtime() == 0, "init_runtime failed"

    outputs = rknn.inference(inputs=[x_nhwc], data_format="nhwc")
    depth_rk = parse_output(outputs)
    rknn.release()

    # ── Compare ──────────────────────────────────────────────────────────────
    w, h = orig_size
    print(
        f"PyTorch  — shape: {depth_pt.shape}  range: {depth_pt.min():.4f} .. {depth_pt.max():.4f}"
    )
    print(
        f"RKNN sim — shape: {depth_rk.shape}  range: {depth_rk.min():.4f} .. {depth_rk.max():.4f}"
    )
    print(f"Max abs diff:     {np.abs(depth_pt - depth_rk).max():.6f}")
    print(f"Mean abs diff:    {np.abs(depth_pt - depth_rk).mean():.6f}")

    save_depth_vis(depth_pt, orig_size, "verify_pytorch.png")
    save_depth_vis(depth_rk, orig_size, "verify_rknn_sim.png")
    print("Saved → verify_pytorch.png  verify_rknn_sim.png")


# ─── SHARED HELPERS ──────────────────────────────────────────────────────────
def preprocess_base(frame_bgr: np.ndarray) -> tuple[np.ndarray, tuple]:
    """Common resize + normalise. Returns HWC float32 [-1,1] + original size."""
    h, w = frame_bgr.shape[:2]
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (INPUT_W, INPUT_H), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    return img, (w, h)  # HWC float32 [-1, 1]


def preprocess_pytorch(frame_bgr: np.ndarray) -> tuple[np.ndarray, tuple]:
    """HWC → NCHW float32 for PyTorch / TorchScript."""
    img, original_size = preprocess_base(frame_bgr)
    img = np.transpose(img, (2, 0, 1))  # HWC → CHW
    img = np.expand_dims(img, axis=0)  # → NCHW [1, 3, H, W]
    return np.ascontiguousarray(img), original_size


def preprocess_rknn(frame_bgr: np.ndarray) -> tuple[np.ndarray, tuple]:
    """HWC → NHWC float32 for rknn.inference(data_format='nhwc')."""
    img, original_size = preprocess_base(frame_bgr)
    img = np.expand_dims(img, axis=0)  # HWC → NHWC [1, H, W, 3]
    return np.ascontiguousarray(img), original_size


def parse_output(outputs) -> np.ndarray:
    out = outputs[0]
    if out.ndim == 4:
        out = out[0]
    if out.ndim == 3:
        out = out[0]
    return out.astype(np.float32)


def save_depth_vis(depth: np.ndarray, original_size: tuple, path: str):
    depth = cv2.resize(depth, original_size, interpolation=cv2.INTER_CUBIC)
    d_min, d_max = depth.min(), depth.max()
    if d_max - d_min > 1e-6:
        depth = (depth - d_min) / (d_max - d_min)
    depth = np.clip(depth, 0.0, 1.0)
    vis = (depth * 255).astype(np.uint8)
    cv2.imwrite(path, cv2.applyColorMap(vis, cv2.COLORMAP_INFERNO))


# ─── ENTRY POINT ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    trace()
    export_onnx()
    export_rknn()
    if len(sys.argv) > 1:
        verify_pc(sys.argv[1])
    else:
        print("\nTip: pass an image path to also run PC simulator verification:")
        print("     python convert_glpn_rknn.py /path/to/image.jpg")
