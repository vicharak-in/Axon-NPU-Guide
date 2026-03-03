# Tri-Model RKNN Inference

Run YOLOv8n, NanoDet-Plus, and FastSAM on the RK3588 NPU. Each script uses all 3 NPU cores for maximum throughput.

## Models

| Model | Input Size | Format | File |
|-------|-----------|--------|------|
| YOLOv8n | 640×640 | INT8 RKNN | `yolov8n_i8.rknn` |
| NanoDet-Plus-m-1.5x | 416×416 | INT8 RKNN | `nanodet-plus-m-1.5x_416_i8.rknn` |
| FastSAM-s | 640×640 | INT8 RKNN | `fastSAM-s.rknn` |

>All three pre-converted INT8 RKNN models are provided in this directory and ready to use.

## Benchmarks (1280×720 @ 30fps input, 3 NPU cores)

| Model | Output FPS | Inference |
|-------|-----------|-----------|
| YOLOv8n | ~30 fps | ~19 ms |
| NanoDet-Plus | ~30 fps | ~14.4 ms |
| FastSAM-s | ~14 fps | ~ 114 ms |

## Running Each Model Separately

### YOLOv8n (3-core)

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

### NanoDet-Plus (3-core)

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

### FastSAM (3-core)

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

`dynamic_tri_infer.py` runs YOLOv8n, NanoDet-Plus, and FastSAM simultaneously — all 3 models are loaded on each of the 3 NPU cores (9 instances total). Each core worker picks the globally oldest pending task across all models, so cores are never idle waiting on one slow model.

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