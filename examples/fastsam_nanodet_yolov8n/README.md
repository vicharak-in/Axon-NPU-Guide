# Tri-Model RKNN Inference

Run YOLOv8n, NanoDet-Plus, and FastSAM on the RK3588 NPU. Each script uses all 3 NPU cores for maximum throughput.

## Models

| Model | Input Size | Format | File |
|-------|-----------|--------|------|
| YOLOv8n | 640×640 | INT8 RKNN | `yolov8n_i8.rknn` |
| NanoDet-Plus-m-1.5x | 416×416 | INT8 RKNN | `nanodet-plus-m-1.5x_416_i8.rknn` |
| FastSAM-s | 640×640 | INT8 RKNN (auto-hybrid) | `fastSAM-s_i8.rknn` |

>All three pre-converted INT8 RKNN models are provided in this directory and ready to use.

## Benchmarks (1280×720 @ 30fps input, 3 NPU cores)

| Model | Output FPS | Inference |
|-------|-----------|-----------|
| YOLOv8n | ~30 fps | ~19 ms |
| NanoDet-Plus | ~30 fps | ~14.4 ms |
| FastSAM-s | ~14 fps | ~ 114 ms |

## Running Each Model Separately

## YOLOv8n (3-core)

#### Prerequisites

Follow the environment setup guide here: [Environment Setup](https://github.com/vicharak-in/Axon-NPU-Guide/blob/main/examples/yolov8-11_model_conversion_n_deployment/README-AXON.md#2-environment-setup)

#### Step 1: Export ONNX

Export YOLOv8n to ONNX from Ultralytics:

```bash
yolo export model=yolov8n.pt format=onnx opset=19
```

> For postprocess node removal and other ONNX modifications, refer to the [YOLOv8/11 conversion guide](https://github.com/vicharak-in/Axon-NPU-Guide/tree/main/examples/yolov8-11_model_conversion_n_deployment).

#### Step 2: Convert ONNX to RKNN

**FP16 (no quantization)**

```bash
python3 fastsam/convert_onnx_to_rknn.py \
    -i <path/to/yolov8n.onnx> \
    -o <path/to/yolov8n.rknn>
```

**INT8 (quantized)**

```bash
python3 fastsam/convert_onnx_to_rknn.py \
    -i <path/to/yolov8n.onnx> \
    -o <path/to/yolov8n_i8.rknn> \
    --quantize \
    --dataset <path/to/dataset.txt>
```

#### Step 3: Run Inference (3-core)

> **Environment:** on-device

```bash
python3 yolo_multicore.py \
    --input <input_stream_ip> \
    --model yolov8n_i8.rknn
```

Options:
```
--input_size 640        # Input size (default: 640)
--score_thresh 0.45     # Score threshold
--nms_thresh 0.45       # NMS IoU threshold
--post_workers 1        # Postprocess workers
--queue_size 4          # Queue size (drop-oldest)
```

## NanoDet-Plus (3-core)

Deploy [NanoDet](https://github.com/RangiLyu/nanodet) on the **Vicharak Axon** NPU using RKNN Toolkit2.

#### Prerequisites

Two Python virtual environments are needed:

- **venv-rknn** (RKNN conversion & on-device inference): [follow this link](https://github.com/vicharak-in/Axon-NPU-Guide/tree/main/examples/yolov8-11_model_conversion_n_deployment#create-environment-2-venv-rknn)
- **venv-nanodet** (NanoDet ONNX export — Python 3.8 only):

```bash
python3.8 -m venv venv-nanodet
source venv-nanodet/bin/activate

git clone https://github.com/RangiLyu/nanodet.git
pip install --upgrade pip
pip install -r /path/to/this/repo/requirements.txt

cd nanodet
python setup.py develop
python -c "import nanodet; print('NanoDet OK')"
```

> **Note:** Do NOT use NanoDet's official `requirements.txt`. Use this repo's `requirements.txt` instead (it avoids known compatibility issues).

#### Step 1: Download Weights and Export ONNX

> **Environment:** `venv-nanodet`

1. Download NanoDet weights from the [Model Zoo](https://github.com/RangiLyu/nanodet?tab=readme-ov-file#model-zoo)
2. Export ONNX using the [official steps](https://github.com/RangiLyu/nanodet?tab=readme-ov-file#export-model-to-onnx)

> Use the matching config YAML for your chosen weights, and keep model/input size consistent (e.g. a `416×416` weight with a `416×416` config).

#### Step 2: Convert ONNX to RKNN

> **Environment:** `venv-rknn`

**FP model (no quantization)**

```bash
python nanodet2rknn.py \
  --onnx /path/to/model.onnx \
  --config /path/to/nanodet_config.yml \
  --output /path/to/model.rknn \
  --platform rk3588
```

**INT8 (quantized)**

```bash
python nanodet2rknn.py \
  --onnx /path/to/model.onnx \
  --config /path/to/nanodet_config.yml \
  --output /path/to/model_int8.rknn \
  --platform rk3588 \
  --quantize \
  --dataset /path/to/dataset.txt
```

> `--config` should be the NanoDet YAML from your NanoDet repo clone (e.g. `nanodet/config/...`) corresponding to your weights/ONNX. If `--output` is omitted, the script auto-generates an RKNN filename.

#### Step 3: Run Inference (3-core)

> **Environment:** `venv-rknn` (on-device)

```bash
python3 nanodet_multicore.py \
    --input <input_stream_ip> \
    --model nanodet-plus-m-1.5x_416_i8.rknn
```

Options:
```
--input_size 416,416    # Input W,H (default: 416,416)
--num_classes 80
--reg_max 7
--strides 8,16,32,64
--score_thresh 0.60     # Score threshold
--nms_thresh 0.6        # NMS IoU threshold
--post_workers 1        # Postprocess workers
--queue_size 4          # Queue size (drop-oldest)
```

## FastSAM (3-core)

Deploy [FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM) (Segment Anything) on the **Vicharak Axon** NPU using RKNN Toolkit2.

#### Prerequisites

Two Python virtual environments are needed - one for ONNX manipulation (Steps 1–2) and one for RKNN conversion + inference (Steps 3–4).

Follow the environment setup guides here:

- **venv-onnx** (ONNX export & node removal): [follow this link](https://github.com/vicharak-in/Axon-NPU-Guide/tree/main/examples/yolov8-11_model_conversion_n_deployment#create-environment-1-venv-onnx)
- **venv-rknn** (RKNN conversion & on-device inference): [follow this link](https://github.com/vicharak-in/Axon-NPU-Guide/tree/main/examples/yolov8-11_model_conversion_n_deployment#create-environment-2-venv-rknnn)

#### Step 1: Export ONNX from Ultralytics

> **Environment:** `venv-onnx`

```bash
yolo export model="FastSAM-s.pt" format=onnx opset=19
```

This produces a `.onnx` file alongside the `.pt` model.

#### Step 2: Remove Postprocess Nodes

> **Environment:** `venv-onnx`

This step strips postprocessing nodes and exposes the raw per-scale head outputs.

```bash
python3 remove_fastsam_postprocess.py -i <path/to/FastSAM-s.onnx> -o <path/to/output.onnx>
```

#### Step 3: Convert ONNX to RKNN

> **Environment:** `venv-rknn`

**FP16 (no quantization)**

```bash
python3 convert_onnx_to_rknn.py -i <path/to/mod_FastSAM-s.onnx> -o <path/to/mod_FastSAM-s.rknn>
```

**INT8 (quantized)**

Requires a calibration dataset (a text file listing image paths, one per line).

```bash
python3 convert_onnx_to_rknn.py \
    -i <path/to/mod_FastSAM-s.onnx> \
    -o <path/to/quant_FastSAM-s.rknn> \
    --quantize \
    --dataset <path/to/dataset.txt> \
    --mean-values 0,0,0 \
    --std-values 255.0,255.0,255.0 \
    --auto-hybrid
```

#### Step 4: Run Inference (3-core)

> **Environment:** `venv-rknn` (on-device)

```bash
python3 fastsam_multicore.py \
    --input <input_stream_ip> \
    --model fastSAM-s_i8.rknn
```

Options:
```
--input_size 640        # Input size (default: 640)
--conf_thresh 0.25      # Confidence threshold
--iou_thresh 0.9        # NMS IoU threshold
--topk 120              # Max candidates before NMS
--max_det 64            # Max detections after NMS
--draw_boxes            # Draw bounding boxes on output
--post_workers 2        # Postprocess workers (default: 2)
--queue_size 4          # Queue size (drop-oldest)
```

## Running All Three Models Together

`dynamic_tri_infer.py` runs YOLOv8n, NanoDet-Plus, and FastSAM simultaneously - all 3 models are loaded on each of the 3 NPU cores (9 instances total). Each core worker picks the globally oldest pending task across all models, so cores are never idle waiting on one slow model.

```bash
python3 dynamic_tri_infer.py \
    --input <input_stream_ip> \
    --yolo_model yolov8n_i8.rknn \
    --nano_model quant_nanodet-plus-m-1.5x_416_416x416.rknn \
    --fastsam_model quant_modified_FastSAM-s.rknn
```

Options:
```
--queue_size 4              # Per-model deque size (drop-oldest)
--yolo_post_workers 1
--nano_post_workers 1
--fastsam_post_workers 2    # Default 2; FastSAM post is heavy
--fastsam_boxes             # Draw bounding boxes on FastSAM output
```

Each model gets its own display window. Frames are shared from a single capture thread with drop-oldest queues so faster models don't block on slower ones.
