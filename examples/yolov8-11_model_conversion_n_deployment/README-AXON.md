# RKNN Model Inference Guide for Vicharak Axon Boards

This guide walks you through setting up and running inference with `.rknn` models on the Vicharak Axon board.

---

## 1. Prerequisites

- Vicharak Axon board with RK35xx SoC
- `.rknn` model file (converted to .rknn using the rknn-toolkit2) (we have also made ready some precompiled `.rknn` model files that have been tested on Axon with accurate results.)

---

## 2. Environment Setup

### Create a Python Virtual Environment

```bash
# Create virtual environment
python3 -m venv env-rknn

# Activate the environment
source env-rknn/bin/activate
```

> **Note:** Always activate the environment before running inference or installing any kind of packages:
> ```bash
> source env-rknn/bin/activate
> ```

---

## 3. Install Dependencies

### run these commands:

```bash
sudo apt update
sudo apt-get install -y python3-dev python3-pip
sudo apt install -y python3-opencv python3-numpy python3-setuptools
```

---

## 4. Install RKNN Toolkit Lite2

RKNN toolkit lite2 is designed for deploying models directly on the board. For additional dependency requirements and detailed usage instructions, refer to the [Rockchip_RKNPU_User_Guide_RKNN_SDK](https://github.com/rockchip-linux/rknn-toolkit2/tree/master/doc) documentation.

You can obtain RKNN toolkit lite2 by downloading it from the [official github repo](https://github.com/airockchip/rknn-toolkit2).

### Clone the Repository

```bash
git clone https://github.com/airockchip/rknn-toolkit2.git
cd rknn-toolkit2/rknn-toolkit-lite2/
```

### Install the Package

For Python 3.12 (latest):
```bash
pip install packages/rknn_toolkit_lite2-2.3.2-cp312-cp312-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
```

For other Python versions, select the appropriate wheel file:
| Python Version | Wheel File |
|----------------|------------|
| Python 3.8 | `rknn_toolkit_lite2-2.3.2-cp38-cp38-manylinux_2_17_aarch64.whl` |
| Python 3.10 | `rknn_toolkit_lite2-2.3.2-cp310-cp310-manylinux_2_17_aarch64.whl` |
| Python 3.11 | `rknn_toolkit_lite2-2.3.2-cp311-cp311-manylinux_2_17_aarch64.whl` |
| Python 3.12 | `rknn_toolkit_lite2-2.3.2-cp312-cp312-manylinux_2_17_aarch64.whl` |

---

## 5. Runtime Library Setup

The RKNN runtime shared object (`librknnrt.so`) must be available on the board.

### Check if Already Present

```bash
ls /usr/lib/librknnrt.so
```

### If Not Present, Download and Place

```bash
# Download the runtime library
wget https://github.com/airockchip/rknn-toolkit2/blob/master/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so

# Copy to system library path
sudo cp librknnrt.so /usr/lib/
```

> **Important:** The runtime library version must match the RKNN-Toolkit2 version used to convert your model. Mismatched versions will cause `Invalid RKNN model version` errors.

---

## 6. Verify Installation

Open a Python shell and test the import:

```bash
python
>>>from rknnlite.api import RKNNLite
>>>
```

If no errors appear, the installation was successful.

---

## 7. Running Inference

### Command Syntax

```bash
python rknn_inference.py --model <MODEL_PATH> --image <IMAGE_PATH> --size <H,W or SIZE> --imgname <OUTPUT_NAME> [--quantized]
```

### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--model` | Yes | Path to the `.rknn` model file |
| `--image` | Yes | Path to the input image |
| `--size` | No | Input size as `H,W` (e.g., `480,640`) or single int for square (e.g., `640`). Default: `640` |
| `--imgname` | No | Output image filename. Default: `result_rknn_output.jpg` |
| `--quantized` | No | **Add this flag if the model was quantized** during conversion |

> **Note:** The `--size` must match one of the graph sizes compiled into the `.rknn` model. 

### Example Commands

### For Quantized Models:

**Rectangular Input (H,W format):**
```bash
python rknn_inference.py \
    --model models/modified_yolov8n_640x640-480x640-1792x1280.rknn \
    --image images/orange320.jpg \
    --size 480,640 \
    --imgname orange_output_mixeddims.jpg \
    --quantized
```

**Square Input (single int):**
```bash
python rknn_inference.py \
    --model models/quant_dyanmic320_480_640_1792.rknn \
    --image images/cat480.jpg \
    --size 640 \
    --imgname cat_output_quant.jpg \
    --quantized
```

**For Non-Quantized (Float) Models:**
```bash
python rknn_inference.py \
    --model models/yolov8n_float.rknn \
    --image images/test.jpg \
    --size 640 \
    --imgname result_output.jpg
```

---

## 8. Troubleshooting

### Error: `ModuleNotFoundError: No module named 'rknnlite'`

**Cause:** might be that your virtual environment is not activated or package is not installed.

**Solution:**
```bash
source env-rknn/bin/activate
pip install rknn_toolkit_lite2-2.3.2-cp312-cp312-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
```

### No Objects Detected?

- Ensure the `--size` matches one of the compiled graph sizes in the model
- Check the model filename for available sizes (e.g., `640x640-480x640` means you can use `--size 640,640` or `--size 480,640`)
- Verify the model was converted correctly

---

## Resources

- [RKNN-Toolkit2 GitHub](https://github.com/airockchip/rknn-toolkit2)
- [RKNPU2 Runtime](https://github.com/airockchip/rknn-toolkit2/blob/master/rknpu2/)
- [Vicharak Documentation](https://docs.vicharak.in)

---