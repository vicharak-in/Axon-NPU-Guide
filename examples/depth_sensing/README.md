# GLPN Depth Estimation on RK3588 (RKNN)

This project runs the Hugging Face **GLPN NYU** depth–estimation model on a Rockchip **RK3588** NPU using **RKNN Toolkit 2** and `rknnlite`.  
The RKNN output is numerically and visually aligned with the original PyTorch implementation.

---

## Features

- Uses the pretrained model `vinvino02/glpn-nyu` from Hugging Face.
- End‑to‑end toolchain:
  - PyTorch → TorchScript → ONNX → RKNN.
- Explicit, shared preprocessing pipeline for PyTorch and RKNN.
- Handling of NCHW (model) vs NHWC (runtime buffer) layouts.
- PC simulator verification against a PyTorch reference.
- RK3588 inference script with side‑by‑side visualization.

---

## Example Structure

- `convert_glpn_rknn.py`  
  Conversion + verification pipeline (runs on PC):
  - Trace GLPN to TorchScript.
  - Export TorchScript → ONNX.
  - Convert ONNX → RKNN.
  - Optionally verify PC simulator vs PyTorch.


- `inference_rk3588.py`  
  Runtime script (runs on RK3588):
  - Preprocess input image.
  - Run NPU inference via `rknnlite`.
  - Save depth map + combined visualization.

- `glpn_nyu_depth.rknn`  
  RKNN model for deployment on RK3588 (generated).

---

## Design Overview

### 1. Single, explicit preprocessing pipeline

All preprocessing is done in Python. RKNN is configured with:

```python
rknn.config(
    mean_values=[[0.0, 0.0, 0.0]],
    std_values=[[1.0, 1.0, 1.0]],
    target_platform="rk3588",
    quantized_method="channel",
    optimization_level=3,
)
```

So RKNN **does not apply any additional normalization**; instead, we perform:

1. BGR → RGB (OpenCV to model convention).
2. Resize to `INPUT_W × INPUT_H` (640×480).
3. Convert to `float32` and scale to `[0, 1]`.
4. Normalize to `[-1, 1]` using `(x - 0.5) / 0.5`.

This is done in a shared helper:

```python
def preprocess_base(frame_bgr: np.ndarray) -> tuple[np.ndarray, tuple]:
    h, w          = frame_bgr.shape[:2]
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (INPUT_W, INPUT_H), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0        #[1]
    img = (img - 0.5) / 0.5                     # [-1, 1]
    return img, (w, h)
```

Then:

- For **PyTorch / TorchScript** we reshape to NCHW:

  ```python
  img, original_size = preprocess_base(frame_bgr)
  img = np.transpose(img, (2, 0, 1))           # HWC → CHW
  img = np.expand_dims(img, axis=0)            # → NCHW [1, 3, H, W]
  ```

- For **RKNN** we reshape to NHWC:

  ```python
  img, original_size = preprocess_base(frame_bgr)
  img = np.expand_dims(img, axis=0)            # → NHWC [1, H, W, 3]
  ```

**Why:** Centralizing preprocessing avoids mismatches between PyTorch and RKNN, and matches Rockchip’s recommendation to keep image normalization consistent with training‑time preprocessing.

---

### 2. NCHW inside the model, NHWC at inference

Internally, the GLPN model and ONNX graph are **NCHW**, which is the native PyTorch layout.  
However, the RKNN Python API expects **NHWC** numpy buffers at inference time unless specified otherwise.

To reconcile this:

- **Model / ONNX / RKNN graph:** NCHW `[1, 3, H, W]`.
- **Inference input buffer:** NHWC `[1, H, W, 3]`, with `data_format="nhwc"`.

Example (PC simulator and RK3588 use the same pattern):

```python
# x_nhwc: [1, H, W, 3] float32 in [-1, 1]
outputs = rknn.inference(inputs=[x_nhwc], data_format="nhwc")
out     = outputs
```

RKNN automatically permutes the input to the graph’s internal layout based on `data_format`.

---

### 3. Attention backend: `attn_implementation="eager"`

The model is loaded with:

```python
from transformers import GLPNForDepthEstimation

model = GLPNForDepthEstimation.from_pretrained(
    "vinvino02/glpn-nyu",
    attn_implementation="eager",
)
```

Modern versions of Transformers can use more optimized attention backends such as SDPA (scaled dot‑product attention) or Flash‑Attention, which are great for GPU but may translate to ONNX ops that RKNN cannot handle cleanly.
Using the `"eager"` backend:

- Forces attention to be implemented using standard PyTorch operations.
- Simplifies TorchScript and ONNX export.
- Avoids dependency on GPU‑only kernels or exotic ONNX ops.

**If you encounter unsupported attention mechanisms:**

- Symptoms:
  - Export errors mentioning `scaled_dot_product_attention`, Flash‑Attention, or missing attention kernels.
  - Runtime errors about unsupported ops in RKNN.

- Mitigation:
  1. Load the model with `attn_implementation="eager"` as above.
  2. Ensure your PyTorch + Transformers versions are compatible (sometimes a specific version combo is required for clean ONNX export).
  3. If problems persist:
     - Check the model’s issue tracker for notes on export.
     - Consider using the PyTorch → TorchScript → RKNN path directly instead of ONNX.

---

### 4. No quantization (FP32 RKNN)

The project currently builds a **non‑quantized** RKNN model:

```python
rknn.build(do_quantization=False)
```

**Reasons:**

- Primary goal is correctness and matching PyTorch output, not maximum performance.
- FP32 (or FP16) preserves numeric behavior, making debugging easier.
- Rockchip’s documentation recommends first validating accuracy with FP32 models before introducing quantization.

You can later enable quantization:

```python
rknn.build(do_quantization=True, dataset="./dataset.txt")
```

Then follow Rockchip’s precision debugging flow if accuracy drifts.

---

### 5. Postprocessing and avoiding visual artifacts

Postprocessing steps:

1. Resize the raw depth map back to the original image size.
2. Normalise via min–max.
3. Clip to `[0, 1]`.
4. Multiply by 255, cast to `uint8`.
5. Apply the Inferno colormap.

```python
def postprocess(depth: np.ndarray, original_size: tuple) -> np.ndarray:
    depth = cv2.resize(depth, original_size, interpolation=cv2.INTER_CUBIC)

    d_min, d_max = depth.min(), depth.max()
    if d_max - d_min > 1e-6:
        depth = (depth - d_min) / (d_max - d_min)

    depth = np.clip(depth, 0.0, 1.0)          # guard against overshoot
    gray  = (depth * 255).astype(np.uint8)
    return cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
```

**Why clip?**  
`INTER_CUBIC` interpolation on float32 data can overshoot slightly beyond the original range, producing values <0 or >1. Casting those directly to `uint8` can cause banding or “tiled” artifacts. Clipping ensures a stable [0,1] range before visualization.

---

## Conversion workflow (PC)

### 1. Environment

Install the dependencies:

```bash
pip install torch transformers[torch] onnx==1.18.0 opencv-python torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 2. Trace GLPN to TorchScript

Inside `convert_glpn_rknn.py`:

```python
model = GLPNForDepthEstimation.from_pretrained(
    MODEL_ID,
    attn_implementation="eager",
)
model.eval()

dummy   = torch.randn(1, 3, INPUT_H, INPUT_W)
wrapper = GLPNWrapper(model)  # wraps model(pixel_values=x).predicted_depth
wrapper.eval()

with torch.no_grad():
    traced = torch.jit.trace(wrapper, dummy)

torch.jit.save(traced, PT_MODEL)
```

The wrapper ensures the model’s interface is simply:

- Input:  `x` (NCHW float32 tensor in `[-1, 1]`).
- Output: `predicted_depth` `[1, H, W]`.

### 3. Export TorchScript → ONNX

```python
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
        opset_version=12,
        do_constant_folding=True,
        dynamic_axes=None,  # static shapes for RKNN
    )
```

Optionally verify ONNX:

```python
import onnx
model_onnx = onnx.load(ONNX_MODEL)
onnx.checker.check_model(model_onnx)
```

### 4. Convert ONNX → RKNN

```python
from rknn.api import RKNN

rknn = RKNN(verbose=True)

rknn.config(
    mean_values=[[0.0, 0.0, 0.0]],
    std_values=[[1.0, 1.0, 1.0]],
    target_platform="rk3588",
    quantized_method="channel",
    optimization_level=3,
)

rknn.load_onnx(model=ONNX_MODEL)
rknn.build(do_quantization=False)
rknn.export_rknn(RKNN_MODEL)
rknn.release()
```

### 5. PC simulator verification (optional)

To compare RKNN vs PyTorch on the host:

```python
from rknn.api import RKNN

frame = cv2.imread(image_path)
x_nchw, _  = preprocess_pytorch(frame)
x_nhwc, sz = preprocess_rknn(frame)

# PyTorch reference
loaded = torch.jit.load(PT_MODEL)
loaded.eval()
with torch.no_grad():
    depth_pt = loaded(torch.from_numpy(x_nchw)).squeeze().numpy()

# RKNN sim
rknn = RKNN()
rknn.config(...)
rknn.load_onnx(model=ONNX_MODEL)
rknn.build(do_quantization=False)
rknn.init_runtime()  # no target → PC simulator[2][3]

outputs  = rknn.inference(inputs=[x_nhwc], data_format="nhwc")
depth_rk = parse_output(outputs)
rknn.release()
```

---

## Inference on RK3588

### 1. Environment

On the RK3588 board:

```bash
pip install rknn_toolkit_lite2 opencv-python numpy requests
```

(Use the Rockchip wheel for `rknn_toolkit_lite2` per their docs.)

### 2. Run the script

Copy:

- `glpn_nyu_depth.rknn`
- `inference_rk3588.py`

Then:

```bash
# With a local image:
python inference_rk3588.py /path/to/image.jpg

# Or let it fetch a sample image:
python inference_rk3588.py
```

The script:

1. **Preprocesses** the input image to NHWC float32 in `[-1, 1]`.
2. **Runs inference**:

   ```python
   rknn = RKNNLite()
   rknn.load_rknn("glpn_nyu_depth.rknn")
   rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)

   outputs = rknn.inference(inputs=[x_nhwc], data_format="nhwc")
   ```

3. **Postprocesses** the depth map with the clipping+colormap pipeline.
4. **Saves**:
   - `sample_depth.png` — coloured depth map.
   - `sample_combined.jpg` — original + depth side‑by‑side.

---

## Troubleshooting

### Wrong input shape / “expect nhwc”

- Error (simulator example):

  > The input(ndarray) shape (1, 3, 480, 640) is wrong, expect 'nhwc' like (1, 480, 640, 3)!

- Fix:

  - Ensure you use NHWC input (`[1, H, W, 3]`) and set `data_format="nhwc"` for **all** RKNN inference calls, both in PC simulator and on RK3588.

### Tiled or blocky depth maps

- Cause:

  - Treating a 3D tensor as image, or
  - Interpolation overshoot causing wrap‑around when casting to `uint8`.

- Fixes:

  1. Ensure you collapse all size‑1 dimensions:

     ```python
     out = outputs
     if out.ndim == 4:
         out = out
     if out.ndim == 3:
         out = out
     ```

  2. Clip before casting:

     ```python
     depth = np.clip(depth, 0.0, 1.0)
     gray  = (depth * 255).astype(np.uint8)
     ```

### Attention / ONNX export errors

- If you see SDPA or Flash‑Attention related errors during export:

  1. Force `attn_implementation="eager"` in the model loader.
  2. Double‑check your Transformers + PyTorch version compatibility if export still fails.
  3. If necessary, search the model’s issues for export notes or fall back to using TorchScript → RKNN directly.
