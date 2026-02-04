# YOLOv8n/11n Object Detection — RKNN Conversion Pipeline for Rockchip NPU

This guide provides a complete walkthrough for converting YOLOv8n/11n PyTorch models (`.pt`) to RKNN format (`.rknn`) for deployment on Rockchip NPU devices (RK3588, RK3576, etc.).

> **Note on YOLOv11n:** YOLOv11n models are supported but with a limitation - they can only be converted with **single static shapes** (one shape at a time). Multi-shape RKNN conversion from dynamic ONNX does not work reliably for YOLOv11n. See the dedicated [YOLOv11n section](#yolov11n-specific-pipeline) below.

---

## Overview

The conversion pipeline consists of four main stages:


### YOLOv8n Pipeline

| Stage | Tool | Environment | Input | Output |
|-------|------|-------------|-------|--------|
| 1. Export | Ultralytics YOLO | venv-onnx | `.pt` | `.onnx` (static) |
| 2. Node Removal | removenodes.py | venv-onnx | `.onnx` | `.onnx` (modified) |
| 3. Dynamic Conversion | static2dynamic.py | venv-onnx | `.onnx` | `.onnx` (dynamic) |
| 4. RKNN Conversion | onnx2rknn.py | venv-rknn | `.onnx` | `.rknn` (multi-shape) |

### YOLOv11n Pipeline (Static Shape Only)

| Stage | Tool | Environment | Input | Output |
|-------|------|-------------|-------|--------|
| 1. Export | Ultralytics YOLO | venv-onnx | `.pt` | `.onnx` (static) |
| 2. Node Removal | **removenodes11n.py** | venv-onnx | `.onnx` | `.onnx` (modified) |
| 3. RKNN Conversion | onnx2rknn.py | venv-rknn | `.onnx` (static) | `.rknn` (single shape) |

> **YOLOv11n Note:** Skip the dynamic conversion step. Convert the static modified ONNX directly to RKNN with a single shape only.

---

## Why This Pipeline?

### Why Export Static First, Then Convert to Dynamic?

During our testing, we found that directly exporting YOLOv8 models with `dynamic=True` from Ultralytics produces ONNX models that do not work reliably with the RKNN conversion toolkit.

Our solution:
1. Export a **static** ONNX model (fixed input shape)
2. Use our custom `static2dynamic.py` script to properly convert dimensions to dynamic
3. This approach produces consistent, reliable dynamic ONNX models that convert cleanly to RKNN

#### What We Remove:
```
Original YOLOv8 Head:
  Conv (64ch) -> Reshape -> Softmax -> DFL Processing -> Concat -> Output
  Conv (80ch) -> Reshape -> Concat -> Output

Modified Head (after node removal):
  Conv (64ch) -> Output (bbox features)
  Conv (80ch) -> Sigmoid -> Output (class scores)
```
---

**Recommended Python versions:** 3.8 to 3.12

## Clone the Repository

```bash
git clone https://github.com/vicharak-in/Axon-NPU-Guide.git
cd Axon-NPU-Guide/examples/yolov8-11_model_conversion_n_deployment
```

## Environment Setup

We use **two separate virtual environments** to avoid package conflicts:

| Environment | Purpose | Key Packages |
|-------------|---------|--------------|
| `venv-onnx` | PyTorch export, ONNX manipulation | ultralytics, onnx, onnx-graphsurgeon |
| `venv-rknn` | RKNN conversion | rknn-toolkit2 |

### Create Environment 1: venv-onnx

```bash
# Create and activate virtual environment
python3 -m venv venv-onnx
source venv-onnx/bin/activate

# Install required packages
pip install --upgrade pip
pip install ultralytics
pip install onnx
pip install onnx-graphsurgeon==0.5.2  # Specific version required for compatibility
pip install numpy
```

> **Important:** We use `onnx-graphsurgeon==0.5.2` specifically because newer versions have changes that break compatibility with our node removal script.

### Create Environment 2: venv-rknn

```bash
# Deactivate previous environment if active
deactivate

# Create and activate new virtual environment
python3 -m venv venv-rknn
source venv-rknn/bin/activate

# Clone RKNN Toolkit2 repository
git clone https://github.com/airockchip/rknn-toolkit2.git
cd rknn-toolkit2/rknn-toolkit2

# Install dependencies
pip install --upgrade pip
pip install -r packages/x86_64/requirements_cp312-2.3.2.txt # Adjust for your Python and Tookit version

# Install RKNN Toolkit2 wheel (select based on your Python version)
# For Python 3.12:
pip install packages/x86_64/rknn_toolkit2-2.3.2-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

# For other Python versions, use the appropriate wheel:
# Python 3.8:  rknn_toolkit2-2.3.2-cp38-cp38-manylinux_2_17_x86_64.whl
# Python 3.10: rknn_toolkit2-2.3.2-cp310-cp310-manylinux_2_17_x86_64.whl
# Python 3.11: rknn_toolkit2-2.3.2-cp311-cp311-manylinux_2_17_x86_64.whl

# Verify installation
$python
>>>from rknn.api import RKNN
>>>
# CTRL+D to exit

# Return to project directory
cd ../..
```

---

## Step 1: Export PyTorch to ONNX

Export the YOLO model from PyTorch format to ONNX format using Ultralytics. The process is identical for both YOLOv8n and YOLOv11n.

### Activate Environment

```bash
source venv-onnx/bin/activate
```

### Example Export Commands

**For YOLOv8n:**
```bash
yolo export model=yolov8n.pt format=onnx opset=19 dynamic=False
```

**For YOLOv11n:**
```bash
yolo export model=yolo11n.pt format=onnx opset=19 dynamic=False imgsz=1280
```

for more information on the parameters, refer to the ultralytics export arguments in [ultralytics docs](https://docs.ultralytics.com/modes/export/#arguments)

### Output

This creates `yolov8n.onnx` or `yolo11n.onnx` with static input shape `[1, 3, 640, 640]` (default) or `[1, 3, 1280, 1280]` based on `imgsz`, in your current directory.

#### Verify Export (optional)

```bash
python -c "
import onnx
model = onnx.load('yolov8n.onnx')
print('Inputs:')
for inp in model.graph.input:
    shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    print(f'  {inp.name}: {shape}')
print('Outputs:')
for out in model.graph.output:
    shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
    print(f'  {out.name}: {shape}')
"
```

Expected output:
```
Inputs:
  images: [1, 3, 640, 640]
Outputs:
  output0: [1, 84, 8400]
```

---

## Step 2: Remove Post-Processing Nodes

This step removes post-processing nodes to optimize the model for NPU inference.

### Run Node Removal Script

**For YOLOv8n:**
```bash
# ensure venv-onnx is active
source venv-onnx/bin/activate

# run the script (see usage in the file)
python utils/removenodes.py yolov8n.onnx modified_yolov8n.onnx
```

**For YOLOv11n:**
```bash
# ensure venv-onnx is active
source venv-onnx/bin/activate

# run the script using the YOLOv11n-specific version
python utils/removenodes11n.py yolo11n.onnx modified_yolo11n.onnx
```

### Output

Creates `modified_yolov8n.onnx` with 6 outputs (3 scales × 2 branches):
- 3 bbox feature maps: `[1, 64, H, W]` for H,W ∈ {80, 40, 20}
- 3 class score maps: `[1, 80, H, W]` for H,W ∈ {80, 40, 20}

---

## Step 3: Convert Static to Dynamic ONNX

> **⚠️ YOLOv8n Only:** This step is **only for YOLOv8n**. For YOLOv11n, skip this step and proceed directly to Step 4 with the static modified ONNX.

Convert the static ONNX model to support dynamic input dimensions.

### Run Conversion Script (YOLOv8n)

```bash
# ensure venv-onnx is active
source venv-onnx/bin/activate

# Convert with dynamic height/width (check code for usage or --help)
python utils/static2dynamic.py --model modified_yolov8n.onnx --no-batch --hw
```

### Command Options

```bash
python utils/static2dynamic.py --help
```

### Output

Creates `modified_yolov8n_dynamic.onnx` with input shape `[batch, 3, height, width]`.

#### Verify Dynamic Dimensions (optional)

```bash
python -c "
import onnx
model = onnx.load('modified_yolov8n_dynamic.onnx')
print('Dynamic model inputs:')
for inp in model.graph.input:
    shape = [d.dim_value if d.dim_value else d.dim_param for d in inp.type.tensor_type.shape.dim]
    print(f'  {inp.name}: {shape}')
"
```

Expected output:
```
Dynamic model inputs:
  images: ['batch', 3, 'height', 'width']
```

---

## Step 4: Convert ONNX to RKNN

Convert the ONNX model to RKNN format for deployment on Rockchip NPU.

### Switch to RKNN Environment

```bash
# Deactivate current environment
deactivate

# Activate RKNN environment
source venv-rknn/bin/activate
```

### Basic Conversion

**For YOLOv8n (supports multiple shapes):**
```bash
# Single shape
python utils/onnx2rknn.py modified_yolov8n_dynamic.onnx --graphsz 640,640

# Multiple shapes
python utils/onnx2rknn.py modified_yolov8n_dynamic.onnx --graphsz 320,320 640,640 1280,1280
```

**For YOLOv11n (single shape only):**
```bash
# Use the static modified ONNX, only one shape at a time
python utils/onnx2rknn.py modified_yolo11n.onnx --graphsz 640,640
```

### Full Command Options

```bash
python utils/onnx2rknn.py --help
```

| Few Important Options | Description |
|--------|-------------|
| `--graphsz H,W H,W...` | Graph sizes as H,W pairs (required or uses defaults) |
| `--platform` | Target platform: rk3588, rk3576, etc (default: rk3588) |
| `--quantize` | Enable INT8 quantization (requires --dataset) |
| `--dataset` | Path to calibration dataset file for quantization |

### Example Commands

#### YOLOv8n - Single Square Size
```bash
python utils/onnx2rknn.py modified_yolov8n_dynamic.onnx --graphsz 640,640
```

#### YOLOv8n - Multiple Square Sizes
```bash
python utils/onnx2rknn.py modified_yolov8n_dynamic.onnx --graphsz 320,320 480,480 640,640 1792,1792
```

#### YOLOv8n - Rectangular Sizes
```bash
python utils/onnx2rknn.py modified_yolov8n_dynamic.onnx --graphsz 640,640 480,640 1280,736
```

#### YOLOv11n - Single Shape Only
```bash
# YOLOv11n only supports one shape per conversion
python utils/onnx2rknn.py modified_yolo11n.onnx --graphsz 640,640

# For a different shape, convert separately
python utils/onnx2rknn.py modified_yolo11n.onnx modified_yolo11n_1280x1280.rknn --graphsz 1280,1280
```

#### With Quantization
```bash
# First, create a dataset.txt file with paths to calibration images
# One image path per line, ~40-100 images recommended

python utils/onnx2rknn.py modified_yolov8n_dynamic.onnx \
    --graphsz 320,320 480,480 640,640 \
    --quantize \
    --dataset dataset.txt
```

#### Custom Output Name and Platform
```bash
python utils/onnx2rknn.py modified_yolov8n_dynamic.onnx \
    my_custom_model.rknn \
    --graphsz 640,640 \
    --platform rk3566
```

### Understanding Multi-Shape Support

RKNN does **not** support true dynamic shapes like ONNX. Instead, it uses "discrete dynamic input":

- You pre-define all input shapes at conversion time
- RKNN compiles a separate optimized graph for each shape
- At runtime, you must use one of the predefined shapes
- RKNN automatically selects the matching graph based on input tensor dimensions

```
┌─────────────────────────────────────────────────────────────┐
│                    RKNN Model File                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │ 320×320     │  │ 640×640     │  │ 1792×1792   │          │
│  │ Graph       │  │ Graph       │  │ Graph       │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
│                                                             │
│  At runtime: RKNN selects graph based on input shape        │
└─────────────────────────────────────────────────────────────┘
```

### Output

Creates an RKNN file with auto-generated name based on shapes:
```
modified_yolov8n_dynamic_dynamic_320x320-480x480-640x640.rknn
```

### Quantization Dataset Format

Create a text file with one image path per line:

```
images/img001.jpg
images/img002.jpg
images/img003.jpg
...
```

> **Tip:** Use at least 40-100 diverse images from your target domain for best quantization accuracy.

---

---

## YOLOv11n Specific Pipeline

This section provides a complete walkthrough specifically for YOLOv11n models, which have limitations compared to YOLOv8n.

### Key Differences from YOLOv8n

1. **No Dynamic Shape Support**: YOLOv11n cannot be reliably converted to RKNN with multiple shapes
2. **Static Conversion Only**: Use the static modified ONNX directly for RKNN conversion
3. **Different Node Removal Script**: Use `removenodes11n.py` instead of `removenodes.py`
4. **One Shape Per Model**: Each RKNN model file contains only one input shape

### Complete YOLOv11n Workflow

#### 1. Export to ONNX (venv-onnx)

```bash
source venv-onnx/bin/activate
yolo export model=yolo11n.pt format=onnx opset=19 dynamic=False
```

#### 2. Remove Post-Processing Nodes (venv-onnx)

```bash
python utils/removenodes11n.py yolo11n.onnx modified_yolo11n.onnx
```

#### 3. Skip Dynamic Conversion

**Do NOT run static2dynamic.py for YOLOv11n.** The dynamic conversion works, but the resulting dynamic ONNX cannot be reliably converted to RKNN with multiple shapes.

#### 4. Convert to RKNN with Single Shape (venv-rknn)

```bash
deactivate
source venv-rknn/bin/activate

# Convert with one shape at a time
python utils/onnx2rknn.py modified_yolo11n.onnx --graphsz 640,640
```

**Output:** `modified_yolo11n_640x640.rknn`

#### Creating Multiple Shape Models for YOLOv11n

If you need different input sizes, create separate RKNN files:

```bash
# 640x640 version
python utils/onnx2rknn.py modified_yolo11n.onnx modified_yolo11n_640x640.rknn --graphsz 640,640

# 1280x1280 version
python utils/onnx2rknn.py modified_yolo11n.onnx modified_yolo11n_1280x1280.rknn --graphsz 1280,1280

# 1920x1080 version
python utils/onnx2rknn.py modified_yolo11n.onnx modified_yolo11n_1920x1080.rknn --graphsz 1920,1080
```

At runtime, load the appropriate RKNN model based on your required input size.

### Why Multi-Shape Doesn't Work for YOLOv11n

While the `static2dynamic.py` script successfully converts YOLOv11n models to dynamic ONNX format, the RKNN toolkit encounters issues when attempting to compile multi-shape models from these dynamic ONNX files. This appears to be related to architectural differences in YOLOv11n's structure.

**Workaround:** Use static shapes and create separate RKNN model files for each required input size.

---

## Troubleshooting

### Common Issues

#### "ONNX opset version not supported"
```
ERROR: ONNX opset version 21 is not supported
```
**Solution:** Re-export with lower opset version:
```bash
yolo export model=yolov8n.pt format=onnx opset=19 dynamic=False
```

#### "No module named 'onnx_graphsurgeon'" or float value errors
```
ModuleNotFoundError: No module named 'onnx_graphsurgeon'
or
AttributeError: module 'onnx.helper' has no attribute 'float32_to_bfloat16'
```
**Solution:** Install the correct version:
```bash
pip install onnx-graphsurgeon==0.5.2
```

#### "No objects detected" during inference
- Verify `--size` matches one of the compiled graph sizes
- Check if model was quantized but `--quantized` flag wasn't used

#### YOLOv11n multi-shape conversion fails
```
ERROR: Failed to build RKNN model with multiple shapes
```
**Solution:** YOLOv11n only supports single static shapes. Convert with one shape at a time:
```bash
python utils/onnx2rknn.py modified_yolo11n.onnx --graphsz 640,640
```
For multiple sizes, create separate RKNN files.

---

## Automated Conversion Script

For convenience, we provide an automated script that handles the entire conversion pipeline described above.

### Script Usage (YOLOv8n)

```bash
# Make it an executable
chmod +x /script/convert_yolov8_to_rknn.sh
or
chmod +x /script/convert_yolo11n_to_rknn.sh

# Basic conversion with single shape
./script/convert_yolov8_to_rknn.sh -m yolov8n -s "640,640"

# Multiple shapes (ensure dimensions are divisible by 32)
./script/convert_yolov8_to_rknn.sh -m yolov8n -s "320,320 640,640 1280,1280"

# Rectangular shapes
./script/convert_yolov8_to_rknn.sh -m yolov8n -s "640,640 1280,736"

# With quantization
./script/convert_yolov8_to_rknn.sh -m yolov8n -s "640,640" -q -d dataset.txt

# Different target platform
./script/convert_yolov8_to_rknn.sh -m yolov8n -s "640,640" -p rk3576
```

> **Script usage is similar for YOLOv11n,** for more info on the flags, use the `-h` or `--help` to show help message.

### Important Notes

- **Dimension Requirement**: All input dimensions (height, width) must be divisible by 32 due to YOLOv8's architecture
- **Environment Isolation**: The script manages separate environments to avoid package conflicts

The automated script is recommended for users who want a streamlined conversion process without manually managing environments and dependencies.

---

## Additional Resources

- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- [RKNN-Toolkit2 GitHub](https://github.com/airockchip/rknn-toolkit2)
- [ONNX GraphSurgeon](https://github.com/NVIDIA/TensorRT/tree/main/tools/onnx-graphsurgeon)
- [Vicharak Documentation](https://docs.vicharak.in)

---