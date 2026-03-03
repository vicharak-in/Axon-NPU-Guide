#!/usr/bin/env python3
import argparse
import math
import queue
import threading
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from rknnlite.api import RKNNLite
except ImportError:
    print("ERROR: rknnlite not found. Install rknn-toolkit2-lite for RK35xx.")
    exit(1)

COCO_CLASSES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
)

SENTINEL = object()
WARMUP_SECONDS = 10.0


class SharedTaskQueue:
    """Per-model deques with drop-oldest; get() returns the globally oldest task.

    put(model_key, task) appends to the model's deque, dropping the oldest
    item if the deque is full.

    get(timeout) scans *all* model deques and returns the task with the
    smallest frame_id (oldest-first priority).  Blocks with a condition
    variable so core workers sleep efficiently when no work is available.
    """

    def __init__(self, model_keys: List[str], maxlen: int):
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._deques: Dict[str, deque] = {k: deque(maxlen=maxlen) for k in model_keys}
        self._closed = False

    def put(self, model_key: str, task: dict):
        """Non-blocking put; drops oldest if deque is full."""
        with self._not_empty:
            self._deques[model_key].append(task)
            self._not_empty.notify()

    def get(self, timeout: float = 0.05) -> Optional[Tuple[str, dict]]:
        """Return (model_key, task) for the globally oldest task, or None on timeout."""
        deadline = time.monotonic() + timeout
        with self._not_empty:
            while True:
                best_key = None
                best_fid = float("inf")
                for key, dq in self._deques.items():
                    if dq:
                        fid = dq[0].get("frame_id", 0)
                        if fid < best_fid:
                            best_fid = fid
                            best_key = key
                if best_key is not None:
                    task = self._deques[best_key].popleft()
                    return (best_key, task)
                if self._closed:
                    return None
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._not_empty.wait(timeout=remaining)

    def close(self):
        """Wake all waiters so they can exit."""
        with self._not_empty:
            self._closed = True
            self._not_empty.notify_all()


def make_model_stats() -> dict:
    return {
        "samples": 0,
        "inf_ms_sum": 0.0,
        "post_ms_sum": 0.0,
        "last_ts": None,
    }


def update_avg_stats(
    stats: Dict[str, dict],
    stats_lock: threading.Lock,
    model_key: str,
    now: float,
    warmup_end: float,
    inf_ms: float,
    post_ms: float,
):
    if now < warmup_end:
        return
    with stats_lock:
        s = stats[model_key]
        s["samples"] += 1
        s["inf_ms_sum"] += inf_ms
        s["post_ms_sum"] += post_ms
        s["last_ts"] = now


def print_avg_stats(stats: Dict[str, dict], warmup_end: float):
    print(f"\n[Main] Average stats (after first {int(WARMUP_SECONDS)}s warmup):")
    for model_key, label in (("yolo", "YOLO"), ("nano", "NanoDet"), ("fastsam", "FastSAM")):
        s = stats[model_key]
        n = s["samples"]
        if n == 0:
            print(f"  {label}: no samples collected after warmup")
            continue
        avg_inf = s["inf_ms_sum"] / n
        avg_post = s["post_ms_sum"] / n
        active_secs = max((s["last_ts"] or warmup_end) - warmup_end, 1e-6)
        avg_fps = n / active_secs
        print(f"  {label}: fps={avg_fps:.2f}  inf={avg_inf:.2f}ms  post={avg_post:.2f}ms  samples={n}")


# Preprocessing

def letterbox_topleft(img: np.ndarray, target_h: int, target_w: int) -> Tuple[np.ndarray, float, int, int]:
    h, w = img.shape[:2]
    scale = min(target_h / h, target_w / w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (new_w, new_h))
    canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
    canvas[:new_h, :new_w] = resized
    return canvas, scale, new_h, new_w


def preprocess_640_rgb(frame_bgr: np.ndarray) -> Tuple[np.ndarray, dict]:
    canvas_bgr, scale, new_h, new_w = letterbox_topleft(frame_bgr, 640, 640)
    canvas_rgb = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB)
    inp = np.expand_dims(canvas_rgb, 0)
    meta = {"scale": scale, "new_h": new_h, "new_w": new_w}
    return inp, meta


def preprocess_nano(frame_bgr: np.ndarray, input_w: int, input_h: int) -> Tuple[np.ndarray, dict]:
    resized = cv2.resize(frame_bgr, (input_w, input_h))
    inp = np.expand_dims(resized, 0)
    meta = {"orig_shape": frame_bgr.shape}
    return inp, meta

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88.0, 88.0)))


def _softmax(x: np.ndarray, axis: int) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> np.ndarray:
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        ix1 = np.maximum(x1[i], x1[order[1:]])
        iy1 = np.maximum(y1[i], y1[order[1:]])
        ix2 = np.minimum(x2[i], x2[order[1:]])
        iy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-7)
        order = order[1:][iou <= iou_thresh]
    return np.array(keep, dtype=np.int32)


def _multiclass_nms(boxes, scores, class_ids, nms_thresh):
    keep = []
    for c in np.unique(class_ids):
        mask = class_ids == c
        indices = np.where(mask)[0]
        k = _nms(boxes[mask], scores[mask], nms_thresh)
        keep.extend(indices[k])
    return keep


# YOLOv8 postprocess
def _dfl(position: np.ndarray) -> np.ndarray:
    n, c, h, w = position.shape
    mc = c // 4
    y = position.reshape(n, 4, mc, h, w)
    y = np.exp(y - np.max(y, axis=2, keepdims=True))
    y /= np.sum(y, axis=2, keepdims=True)
    acc = np.arange(mc, dtype=np.float32).reshape(1, 1, mc, 1, 1)
    return (y * acc).sum(2)


def _yolo_box_process(position: np.ndarray, input_h: int, input_w: int) -> np.ndarray:
    feat_h, feat_w = position.shape[2:]
    grid_x, grid_y = np.meshgrid(np.arange(feat_w, dtype=np.float32),
                                  np.arange(feat_h, dtype=np.float32))
    grid = np.stack((grid_x, grid_y), axis=0).reshape(1, 2, feat_h, feat_w)
    stride_h = input_h / feat_h
    stride_w = input_w / feat_w
    stride = np.array([stride_w, stride_h], dtype=np.float32).reshape(1, 2, 1, 1)
    position = _dfl(position)
    box1 = grid + 0.5 - position[:, 0:2]
    box2 = grid + 0.5 + position[:, 2:4]
    return np.concatenate((box1 * stride, box2 * stride), axis=1)


def _yolo_organize_outputs(outputs: list, num_classes: int = 80) -> list:
    boxes = {}
    classes = {}
    for out in outputs:
        _, c, h, w = out.shape
        key = (h, w)
        if c == 64 or c == 4:
            boxes[key] = out
        elif c == num_classes:
            classes[key] = out
    ordered = []
    for k in sorted(boxes.keys(), key=lambda x: x[0], reverse=True):
        ordered.append(boxes[k])
        ordered.append(classes[k])
    return ordered


def yolo_postprocess(
    raw_outputs: list,
    scale: float,
    orig_h: int,
    orig_w: int,
    score_thresh: float,
    nms_thresh: float,
    num_classes: int = 80,
    input_size: int = 640,
) -> List[Tuple[float, float, float, float, float, int]]:
    outputs_f32 = [o.astype(np.float32) if o.dtype != np.float32 else o for o in raw_outputs]
    outputs_4d = [o for o in outputs_f32 if o.ndim == 4]
    organized = _yolo_organize_outputs(outputs_4d, num_classes)

    boxes_all, cls_all, scores_all = [], [], []
    for i in range(len(organized) // 2):
        box_out = organized[i * 2]
        cls_out = organized[i * 2 + 1]
        boxes = _yolo_box_process(box_out, input_size, input_size)
        boxes = boxes.transpose(0, 2, 3, 1).reshape(-1, 4)
        cls_out = cls_out.transpose(0, 2, 3, 1).reshape(-1, num_classes)
        cls_score = np.max(cls_out, axis=1)
        cls_id = np.argmax(cls_out, axis=1)
        mask = cls_score >= score_thresh
        boxes_all.append(boxes[mask])
        cls_all.append(cls_id[mask])
        scores_all.append(cls_score[mask])

    if not boxes_all or all(len(b) == 0 for b in boxes_all):
        return []

    boxes = np.concatenate(boxes_all)
    classes = np.concatenate(cls_all)
    scores = np.concatenate(scores_all)
    if len(boxes) == 0:
        return []

    keep = _multiclass_nms(boxes, scores, classes, nms_thresh)
    if not keep:
        return []

    boxes = boxes[keep]
    classes = classes[keep]
    scores = scores[keep]

    boxes /= scale
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, orig_w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, orig_h)

    return [[b[0], b[1], b[2], b[3], s, int(c)] for b, s, c in zip(boxes, scores, classes)]


# NanoDet-Plus postprocess
class NanoDetDecoder:
    def __init__(self, input_shape: Tuple[int, int], num_classes: int,
                 reg_max: int, strides: List[int],
                 score_thresh: float, nms_thresh: float):
        self.input_w, self.input_h = input_shape
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.strides = strides
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self._build_anchors()

    def _build_anchors(self):
        anchors, strides_flat = [], []
        for stride in self.strides:
            h = math.ceil(self.input_h / stride)
            w = math.ceil(self.input_w / stride)
            sx, sy = np.meshgrid(np.arange(w, dtype=np.float32),
                                 np.arange(h, dtype=np.float32))
            anchor = np.stack([sx, sy], axis=-1).reshape(-1, 2)
            anchors.append(anchor)
            strides_flat.append(np.full(anchor.shape[0], stride, dtype=np.float32))
        self.all_anchors = np.concatenate(anchors, axis=0)
        self.all_strides = np.concatenate(strides_flat, axis=0)

    def decode(self, raw_outputs: list, orig_shape: tuple) -> list:
        output = raw_outputs[0]
        if output.ndim == 3:
            output = output[0]
        output = output.astype(np.float32)

        cls_scores = output[:, :self.num_classes]
        reg_preds = output[:, self.num_classes:]

        scores = cls_scores.max(axis=1)
        class_ids = cls_scores.argmax(axis=1)
        valid = scores > self.score_thresh
        if not valid.any():
            return []

        v_scores = scores[valid]
        v_cls = class_ids[valid]
        v_anchors = self.all_anchors[valid]
        v_strides = self.all_strides[valid]
        v_reg = reg_preds[valid]

        dis = v_reg.reshape(-1, 4, self.reg_max + 1)
        dis = _softmax(dis, axis=-1)
        proj = np.arange(self.reg_max + 1, dtype=np.float32).reshape(1, 1, -1)
        dis = (dis * proj).sum(axis=-1)

        x1 = (v_anchors[:, 0] - dis[:, 0]) * v_strides
        y1 = (v_anchors[:, 1] - dis[:, 1]) * v_strides
        x2 = (v_anchors[:, 0] + dis[:, 2]) * v_strides
        y2 = (v_anchors[:, 1] + dis[:, 3]) * v_strides

        scale_x = orig_shape[1] / self.input_w
        scale_y = orig_shape[0] / self.input_h
        x1 *= scale_x; x2 *= scale_x
        y1 *= scale_y; y2 *= scale_y

        boxes = np.stack([x1, y1, x2, y2], axis=1)
        keep = _multiclass_nms(boxes, v_scores, v_cls, self.nms_thresh)
        if not keep:
            return []

        return [[boxes[i, 0], boxes[i, 1], boxes[i, 2], boxes[i, 3],
                 v_scores[i], int(v_cls[i])] for i in keep]


# FastSAM postprocess
def _fastsam_make_anchors(feat_sizes, strides):
    all_a, all_s = [], []
    for (h, w), stride in zip(feat_sizes, strides):
        sx = np.arange(w, dtype=np.float32) + 0.5
        sy = np.arange(h, dtype=np.float32) + 0.5
        gy, gx = np.meshgrid(sy, sx, indexing="ij")
        all_a.append(np.stack([gx.ravel(), gy.ravel()], axis=-1))
        all_s.append(np.full((h * w, 1), stride, dtype=np.float32))
    return np.concatenate(all_a, 0), np.concatenate(all_s, 0)


def _fastsam_dfl(x, reg_max):
    b, c, h, w = x.shape
    x = x.reshape(b, 4, reg_max, h, w)
    x = _softmax(x, axis=2)
    wt = np.arange(reg_max, dtype=np.float32).reshape(1, 1, reg_max, 1, 1)
    return (x * wt).sum(axis=2)


def _fastsam_dist2bbox(dist, anchors, strides):
    lt, rb = dist[:, :2], dist[:, 2:]
    x1y1 = anchors - lt
    x2y2 = anchors + rb
    cxy = (x1y1 + x2y2) / 2.0
    wh = x2y2 - x1y1
    return np.concatenate([cxy, wh], axis=1) * strides


def fastsam_postprocess(
    raw_outputs: list,
    scale: float,
    new_h: int,
    new_w: int,
    orig_h: int,
    orig_w: int,
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.9,
    topk: int = 120,
    max_det: int = 64,
    imgsz: int = 640,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    outputs_f32 = [o.astype(np.float32) if o.dtype != np.float32 else o for o in raw_outputs]

    tensors_4d = []
    for i, o in enumerate(outputs_f32):
        if o.ndim != 4:
            continue
        _, c, h, w = o.shape
        tensors_4d.append({"idx": i, "arr": o, "c": c, "h": h, "w": w})

    if len(tensors_4d) < 4:
        empty_m = np.empty((0, orig_h, orig_w), dtype=np.float32)
        return empty_m, np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)

    proto_info = max(tensors_4d, key=lambda t: t["h"] * t["w"])
    proto = proto_info["arr"][0]
    n_coeffs = proto.shape[0]
    mh, mw = proto.shape[1], proto.shape[2]

    per_scale: Dict[Tuple[int, int], list] = {}
    for t in tensors_4d:
        if t["idx"] == proto_info["idx"]:
            continue
        per_scale.setdefault((t["h"], t["w"]), []).append(t)

    scale_groups = []
    for (sh, sw), g in per_scale.items():
        if len(g) < 2:
            continue
        bbox_cands = [x for x in g if x["c"] % 4 == 0 and x["c"] >= 8]
        if not bbox_cands:
            continue
        bbox = max(bbox_cands, key=lambda x: x["c"])
        rem = [x for x in g if x["idx"] != bbox["idx"]]
        if not rem:
            continue
        mask_t = None
        if len(rem) >= 2:
            mask_t = min(rem, key=lambda x: abs(x["c"] - n_coeffs))
        cls_pool = rem if mask_t is None else [x for x in rem if x["idx"] != mask_t["idx"]]
        if not cls_pool:
            cls_pool = rem
        cls_t = min(cls_pool, key=lambda x: x["c"])
        stride = imgsz / float(sh)
        scale_groups.append({"h": sh, "w": sw, "stride": stride,
                             "bbox": bbox, "cls": cls_t, "mask": mask_t})

    if not scale_groups:
        empty_m = np.empty((0, orig_h, orig_w), dtype=np.float32)
        return empty_m, np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)

    scale_groups.sort(key=lambda s: s["stride"])
    reg_max = scale_groups[0]["bbox"]["c"] // 4

    feat_sizes, strides_list = [], []
    all_dist, all_raw_cls, all_coeffs = [], [], []
    all_have_masks = True

    for s in scale_groups:
        bbox_raw = s["bbox"]["arr"]
        cls_raw = s["cls"]["arr"]
        sh, sw = bbox_raw.shape[2], bbox_raw.shape[3]
        feat_sizes.append((sh, sw))
        strides_list.append(s["stride"])

        dist = _fastsam_dfl(bbox_raw, reg_max)[0].reshape(4, -1).T
        all_dist.append(dist)

        raw_cls = cls_raw[0].reshape(cls_raw.shape[1], -1).max(axis=0)
        all_raw_cls.append(raw_cls)

        if s["mask"] is not None and s["mask"]["c"] == n_coeffs:
            mr = s["mask"]["arr"]
            all_coeffs.append(mr[0].reshape(mr.shape[1], -1).T)
        else:
            all_have_masks = False

    anchor_pts, stride_tensor = _fastsam_make_anchors(feat_sizes, strides_list)
    all_dist = np.concatenate(all_dist, 0)
    all_raw_cls = np.concatenate(all_raw_cls, 0).astype(np.float32)

    if all_raw_cls.min() < 0.0 or all_raw_cls.max() > 1.0:
        scores = _sigmoid(all_raw_cls)
    else:
        scores = all_raw_cls

    boxes_cxcywh = _fastsam_dist2bbox(all_dist, anchor_pts, stride_tensor)

    coeffs_all = None
    if all_have_masks and len(all_coeffs) == len(scale_groups):
        coeffs_all = np.concatenate(all_coeffs, 0)

    keep = scores > conf_thresh
    if not keep.any():
        empty_m = np.empty((0, orig_h, orig_w), dtype=np.float32)
        return empty_m, np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)

    scores = scores[keep]
    boxes_k = boxes_cxcywh[keep]
    coeffs = coeffs_all[keep] if coeffs_all is not None else None

    if len(scores) > topk:
        topk_idx = np.argpartition(-scores, topk)[:topk]
        scores, boxes_k = scores[topk_idx], boxes_k[topk_idx]
        if coeffs is not None:
            coeffs = coeffs[topk_idx]

    boxes = np.empty_like(boxes_k)
    boxes[:, 0] = boxes_k[:, 0] - boxes_k[:, 2] / 2
    boxes[:, 1] = boxes_k[:, 1] - boxes_k[:, 3] / 2
    boxes[:, 2] = boxes_k[:, 0] + boxes_k[:, 2] / 2
    boxes[:, 3] = boxes_k[:, 1] + boxes_k[:, 3] / 2
    np.clip(boxes, 0, imgsz, out=boxes)

    idx = _nms(boxes, scores, iou_thresh)
    boxes, scores = boxes[idx], scores[idx]
    if coeffs is not None:
        coeffs = coeffs[idx]
    if len(scores) > max_det:
        boxes, scores = boxes[:max_det], scores[:max_det]
        if coeffs is not None:
            coeffs = coeffs[:max_det]

    if coeffs is not None:
        masks = _sigmoid(coeffs @ proto.reshape(n_coeffs, -1)).reshape(-1, mh, mw)
        sx_m, sy_m = mw / imgsz, mh / imgsz
        for i in range(len(masks)):
            px1, py1 = max(0, int(boxes[i, 0] * sx_m)), max(0, int(boxes[i, 1] * sy_m))
            px2, py2 = min(mw, int(boxes[i, 2] * sx_m + 0.5)), min(mh, int(boxes[i, 3] * sy_m + 0.5))
            cropped = np.zeros_like(masks[i])
            cropped[py1:py2, px1:px2] = masks[i, py1:py2, px1:px2]
            masks[i] = cropped
        proto_new_h = int(round(new_h * (mh / float(imgsz))))
        proto_new_w = int(round(new_w * (mw / float(imgsz))))
        pbot = min(mh, proto_new_h)
        pright = min(mw, proto_new_w)
        masks = masks[:, :pbot, :pright]
    else:
        masks = np.empty((0, 0, 0), dtype=np.float32)

    boxes_orig = boxes.copy()
    boxes_orig /= scale
    boxes_orig[:, [0, 2]] = boxes_orig[:, [0, 2]].clip(0, orig_w)
    boxes_orig[:, [1, 3]] = boxes_orig[:, [1, 3]].clip(0, orig_h)

    return masks, boxes_orig, scores


# Visualization
def draw_detections(img: np.ndarray, detections: list,
                    class_names: tuple, color: tuple,
                    label_prefix: str = "",
                    score_thresh: float = 0.0) -> np.ndarray:
    for det in detections:
        x1, y1, x2, y2, score, cls_id = det
        if score < score_thresh:
            continue
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cls_id = int(cls_id)
        label = f"{label_prefix}{class_names[cls_id]}: {score:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return img


def draw_fastsam(img: np.ndarray, masks: np.ndarray,
                 boxes: np.ndarray, scores: np.ndarray,
                 draw_boxes: bool = True) -> np.ndarray:
    vis = img.copy()
    n = len(masks)
    if n == 0 or masks.shape[1] == 0:
        return vis

    ph, pw = masks.shape[1], masks.shape[2]
    oh, ow = img.shape[:2]

    overlay_small = np.zeros((ph, pw, 3), dtype=np.uint8)
    mask_small = np.zeros((ph, pw), dtype=np.uint8)

    rng = np.random.default_rng(0)
    for i in range(n):
        thr = max(0.4, 0.5 * scores[i]) if i < len(scores) else 0.5
        region = masks[i] > thr
        if not region.any():
            continue
        color = rng.integers(0, 256, size=3, dtype=np.uint8)
        overlay_small[region] = color
        mask_small[region] = 255

    overlay_full = cv2.resize(overlay_small, (ow, oh), interpolation=cv2.INTER_NEAREST)
    mask_full = cv2.resize(mask_small, (ow, oh), interpolation=cv2.INTER_NEAREST)

    blended = cv2.addWeighted(vis, 0.45, overlay_full, 0.55, 0)
    mask_3ch = cv2.merge([mask_full, mask_full, mask_full])
    inv_mask = cv2.bitwise_not(mask_3ch)
    vis = cv2.add(cv2.bitwise_and(blended, mask_3ch),
                  cv2.bitwise_and(vis, inv_mask))

    if draw_boxes:
        for i in range(min(n, len(boxes))):
            bx1, by1, bx2, by2 = boxes[i].astype(int)
            cv2.rectangle(vis, (bx1, by1), (bx2, by2), (255, 0, 0), 2)
            if i < len(scores):
                cv2.putText(vis, f"{scores[i]:.2f}", (bx1, max(by1 - 5, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return vis

def load_rknn_model(model_path: str, core_mask) -> RKNNLite:
    rknn = RKNNLite(verbose=False)
    ret = rknn.load_rknn(model_path)
    if ret != 0:
        raise RuntimeError(f"Failed to load RKNN: {model_path} (ret={ret})")
    try:
        ret = rknn.init_runtime(core_mask=core_mask)
    except TypeError:
        ret = rknn.init_runtime()
    if ret != 0:
        rknn.release()
        raise RuntimeError(f"Failed to init RKNN runtime: {model_path} (ret={ret})")
    return rknn


# Queue helpers
def put_drop_oldest(q: queue.Queue, item):
    try:
        q.put_nowait(item)
        return
    except queue.Full:
        pass
    try:
        q.get_nowait()
    except queue.Empty:
        pass
    try:
        q.put_nowait(item)
    except queue.Full:
        pass


def safe_put_sentinel(q: queue.Queue, count: int):
    for _ in range(count):
        while True:
            try:
                q.put_nowait(SENTINEL)
                break
            except queue.Full:
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass


# Thread workers
def capture_worker(
    source,
    nano_input_wh: Tuple[int, int],
    shared_queue: SharedTaskQueue,
    frame_slots: dict,
    frame_lock: threading.Lock,
    max_stored_frames: int,
    shutdown: threading.Event,
):
    """Read frames, preprocess, push all 3 tasks into the shared queue."""
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[Capture] ERROR: cannot open {source}")
        shutdown.set()
        shared_queue.close()
        return

    nano_w, nano_h = nano_input_wh
    frame_id = 0

    while not shutdown.is_set():
        ok, frame = cap.read()
        if not ok:
            break

        with frame_lock:
            frame_slots[frame_id] = frame
            stale = [fid for fid in frame_slots if fid < frame_id - max_stored_frames]
            for fid in stale:
                del frame_slots[fid]

        # Shared 640 preprocess (YOLO + FastSAM share the same input)
        inp_640, meta_640 = preprocess_640_rgb(frame)
        # NanoDet 416 preprocess
        inp_nano, meta_nano = preprocess_nano(frame, nano_w, nano_h)

        packet_base = {
            "frame_id": frame_id,
            "orig_h": frame.shape[0],
            "orig_w": frame.shape[1],
        }

        shared_queue.put("yolo", {**packet_base, "inp": inp_640, "meta": meta_640})
        shared_queue.put("nano", {**packet_base, "inp": inp_nano, "meta": meta_nano})
        shared_queue.put("fastsam", {**packet_base, "inp": inp_640, "meta": meta_640})

        frame_id += 1

    cap.release()
    shared_queue.close()
    print(f"[Capture] stopped, {frame_id} frames captured")


def core_worker(
    name: str,
    models: Dict[str, RKNNLite],
    shared_queue: SharedTaskQueue,
    raw_queues: Dict[str, queue.Queue],
    shutdown: threading.Event,
):
    """Dynamic core worker: picks the oldest task from any model and runs it."""
    counts = {k: 0 for k in models}

    while not shutdown.is_set():
        result = shared_queue.get(timeout=0.05)
        if result is None:
            if shutdown.is_set():
                break
            continue

        model_key, task = result
        rknn = models[model_key]

        t0 = time.perf_counter()
        try:
            outputs = rknn.inference(inputs=[task["inp"]], data_format="nhwc")
        except Exception as e:
            print(f"[{name}] inference error ({model_key}): {e}, skipping")
            continue
        inf_ms = (time.perf_counter() - t0) * 1000.0

        if outputs is None:
            continue

        raw = {
            "frame_id": task["frame_id"],
            "orig_h": task["orig_h"],
            "orig_w": task["orig_w"],
            "meta": task["meta"],
            "outputs": outputs,
            "inf_ms": inf_ms,
        }
        put_drop_oldest(raw_queues[model_key], raw)
        counts[model_key] += 1

    print(f"[{name}] stopped, ran: " + ", ".join(f"{k}={v}" for k, v in counts.items()))


def yolo_post_worker(
    name: str,
    raw_q: queue.Queue,
    result_slot: dict,
    slot_lock: threading.Lock,
    frame_slots: dict,
    frame_lock: threading.Lock,
    score_thresh: float,
    nms_thresh: float,
    stats: Dict[str, dict],
    stats_lock: threading.Lock,
    warmup_end: float,
    shutdown: threading.Event,
):
    count = 0
    t_start = time.perf_counter()
    last_log = t_start
    while not shutdown.is_set():
        try:
            item = raw_q.get(timeout=0.05)
        except queue.Empty:
            continue
        if item is SENTINEL:
            break

        with frame_lock:
            frame = frame_slots.get(item["frame_id"])
        if frame is None:
            continue

        t0 = time.perf_counter()
        dets = yolo_postprocess(
            item["outputs"], item["meta"]["scale"],
            item["orig_h"], item["orig_w"],
            score_thresh, nms_thresh,
        )
        vis = frame.copy()
        draw_detections(vis, dets, COCO_CLASSES, (255, 128, 0), "[Y] ", score_thresh)
        post_ms = (time.perf_counter() - t0) * 1000.0
        count += 1

        now = time.perf_counter()
        update_avg_stats(stats, stats_lock, "yolo", now, warmup_end, item["inf_ms"], post_ms)
        elapsed = now - t_start
        fps = count / elapsed if elapsed > 0 else 0.0
        if now - last_log >= 2.0:
            print(f"[{name}] output fps: {fps:.1f}  inf: {item['inf_ms']:.1f}ms  post: {post_ms:.1f}ms")
            last_log = now

        with slot_lock:
            result_slot["pending"][item["frame_id"]] = vis

    print(f"[{name}] stopped, processed {count} frames")


def nano_post_worker(
    name: str,
    raw_q: queue.Queue,
    decoder: NanoDetDecoder,
    result_slot: dict,
    slot_lock: threading.Lock,
    frame_slots: dict,
    frame_lock: threading.Lock,
    stats: Dict[str, dict],
    stats_lock: threading.Lock,
    warmup_end: float,
    shutdown: threading.Event,
):
    count = 0
    t_start = time.perf_counter()
    last_log = t_start
    while not shutdown.is_set():
        try:
            item = raw_q.get(timeout=0.05)
        except queue.Empty:
            continue
        if item is SENTINEL:
            break

        with frame_lock:
            frame = frame_slots.get(item["frame_id"])
        if frame is None:
            continue

        t0 = time.perf_counter()
        dets = decoder.decode(item["outputs"], item["meta"]["orig_shape"])
        vis = frame.copy()
        draw_detections(vis, dets, COCO_CLASSES, (0, 255, 0), "[N] ", decoder.score_thresh)
        post_ms = (time.perf_counter() - t0) * 1000.0
        count += 1

        now = time.perf_counter()
        update_avg_stats(stats, stats_lock, "nano", now, warmup_end, item["inf_ms"], post_ms)
        elapsed = now - t_start
        fps = count / elapsed if elapsed > 0 else 0.0
        if now - last_log >= 2.0:
            print(f"[{name}] output fps: {fps:.1f}  inf: {item['inf_ms']:.1f}ms  post: {post_ms:.1f}ms")
            last_log = now

        with slot_lock:
            result_slot["pending"][item["frame_id"]] = vis

    print(f"[{name}] stopped, processed {count} frames")


def fastsam_post_worker(
    name: str,
    raw_q: queue.Queue,
    result_slot: dict,
    slot_lock: threading.Lock,
    frame_slots: dict,
    frame_lock: threading.Lock,
    conf_thresh: float,
    iou_thresh: float,
    topk: int,
    max_det: int,
    draw_boxes: bool,
    stats: Dict[str, dict],
    stats_lock: threading.Lock,
    warmup_end: float,
    shutdown: threading.Event,
):
    count = 0
    t_start = time.perf_counter()
    last_log = t_start
    while not shutdown.is_set():
        try:
            item = raw_q.get(timeout=0.05)
        except queue.Empty:
            continue
        if item is SENTINEL:
            break

        with frame_lock:
            frame = frame_slots.get(item["frame_id"])
        if frame is None:
            continue

        t0 = time.perf_counter()
        masks, boxes, scores = fastsam_postprocess(
            item["outputs"],
            item["meta"]["scale"],
            item["meta"]["new_h"],
            item["meta"]["new_w"],
            item["orig_h"],
            item["orig_w"],
            conf_thresh=conf_thresh,
            iou_thresh=iou_thresh,
            topk=topk,
            max_det=max_det,
        )
        vis = draw_fastsam(frame, masks, boxes, scores, draw_boxes=draw_boxes)
        post_ms = (time.perf_counter() - t0) * 1000.0
        count += 1

        now = time.perf_counter()
        update_avg_stats(stats, stats_lock, "fastsam", now, warmup_end, item["inf_ms"], post_ms)
        elapsed = now - t_start
        fps = count / elapsed if elapsed > 0 else 0.0
        if now - last_log >= 2.0:
            print(f"[{name}] output fps: {fps:.1f}  inf: {item['inf_ms']:.1f}ms  post: {post_ms:.1f}ms")
            last_log = now

        with slot_lock:
            result_slot["pending"][item["frame_id"]] = vis

    print(f"[{name}] stopped, processed {count} frames")


def display_worker(
    slots: Dict[str, dict],
    slot_locks: Dict[str, threading.Lock],
    shutdown: threading.Event,
):
    win_names = {
        "yolo": "YOLOv8n Detection",
        "nano": "NanoDet+ Detection",
        "fastsam": "FastSAM Segmentation",
    }
    for wn in win_names.values():
        cv2.namedWindow(wn, cv2.WINDOW_NORMAL)

    while not shutdown.is_set():
        for model_key, wn in win_names.items():
            with slot_locks[model_key]:
                pending = slots[model_key]["pending"]
                nid = slots[model_key]["next_id"]
                img = None
                if nid in pending:
                    img = pending.pop(nid)
                    slots[model_key]["next_id"] = nid + 1
                elif pending:
                    oldest = min(pending)
                    img = pending.pop(oldest)
                    slots[model_key]["next_id"] = oldest + 1
            if img is not None:
                cv2.imshow(wn, img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            shutdown.set()
            break

    cv2.destroyAllWindows()
    print("[Display] stopped")

def parse_args():
    p = argparse.ArgumentParser(
        description="Dynamic-priority tri-model RKNN pipeline: YOLOv8n + NanoDet-Plus + FastSAM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Example:
  python3 dynamic_tri_infer.py \\
    --input "<video_stream_ip>" \\
    --yolo_model yolov8n_i8.rknn \\
    --nano_model nanodet_i8.rknn \\
    --fastsam_model fastsam_i8.rknn
""",
    )

    p.add_argument("--input", required=True, help="Video path, camera index, or IP webcam URL")

    p.add_argument("--yolo_model", required=True, help="YOLOv8n RKNN model path")
    p.add_argument("--nano_model", required=True, help="NanoDet-Plus RKNN model path")
    p.add_argument("--fastsam_model", required=True, help="FastSAM RKNN model path")

    yg = p.add_argument_group("YOLOv8")
    yg.add_argument("--yolo_size", type=int, default=640)
    yg.add_argument("--yolo_score", type=float, default=0.45)
    yg.add_argument("--yolo_nms", type=float, default=0.45)

    ng = p.add_argument_group("NanoDet")
    ng.add_argument("--nano_input_size", type=str, default="416,416", help="NanoDet input W,H")
    ng.add_argument("--nano_num_classes", type=int, default=80)
    ng.add_argument("--nano_reg_max", type=int, default=7)
    ng.add_argument("--nano_strides", type=str, default="8,16,32,64")
    ng.add_argument("--nano_score", type=float, default=0.60)
    ng.add_argument("--nano_nms", type=float, default=0.6)

    fg = p.add_argument_group("FastSAM")
    fg.add_argument("--fastsam_size", type=int, default=640)
    fg.add_argument("--fastsam_conf", type=float, default=0.25)
    fg.add_argument("--fastsam_iou", type=float, default=0.9)
    fg.add_argument("--fastsam_topk", type=int, default=120)
    fg.add_argument("--fastsam_max_det", type=int, default=64)
    fg.add_argument("--fastsam_boxes", action="store_true")

    tg = p.add_argument_group("Pipeline")
    tg.add_argument("--queue_size", type=int, default=4, help="Per-model deque size (drop-oldest)")
    tg.add_argument("--fastsam_post_workers", type=int, default=2,
                    help="FastSAM postprocess workers (default: 2 to match higher inference rate)")
    tg.add_argument("--yolo_post_workers", type=int, default=1)
    tg.add_argument("--nano_post_workers", type=int, default=1)

    return p.parse_args()

def main():
    args = parse_args()

    shutdown = threading.Event()

    core0 = getattr(RKNNLite, "NPU_CORE_0", None)
    core1 = getattr(RKNNLite, "NPU_CORE_1", None)
    core2 = getattr(RKNNLite, "NPU_CORE_2", None)
    cores = [("CORE_0", core0), ("CORE_1", core1), ("CORE_2", core2)]

    # Load all 3 models on each core = 9 model instances total
    core_models: List[Dict[str, RKNNLite]] = []
    for core_name, core_mask in cores:
        models = {}
        print(f"[{core_name}] Loading YOLOv8n ...")
        models["yolo"] = load_rknn_model(args.yolo_model, core_mask)
        print(f"[{core_name}] Loading NanoDet+ ...")
        models["nano"] = load_rknn_model(args.nano_model, core_mask)
        print(f"[{core_name}] Loading FastSAM ...")
        models["fastsam"] = load_rknn_model(args.fastsam_model, core_mask)
        core_models.append(models)

    # NanoDet decoder
    nano_input_wh = tuple(map(int, args.nano_input_size.split(",")))
    nano_strides = list(map(int, args.nano_strides.split(",")))
    nano_decoder = NanoDetDecoder(
        input_shape=nano_input_wh,
        num_classes=args.nano_num_classes,
        reg_max=args.nano_reg_max,
        strides=nano_strides,
        score_thresh=args.nano_score,
        nms_thresh=args.nano_nms,
    )

    # Shared task queue
    qs = max(args.queue_size, 1)
    shared_queue = SharedTaskQueue(["yolo", "nano", "fastsam"], maxlen=qs)

    # Per-model raw output queues (inference -> postprocess)
    q_yolo_raw = queue.Queue(maxsize=qs)
    q_nano_raw = queue.Queue(maxsize=qs)
    q_fsam_raw = queue.Queue(maxsize=qs)
    raw_queues = {"yolo": q_yolo_raw, "nano": q_nano_raw, "fastsam": q_fsam_raw}

    # Frame store
    frame_slots: Dict[int, np.ndarray] = {}
    frame_lock = threading.Lock()

    # Display result slots (ordered pending buffer per model)
    result_slots = {
        "yolo": {"pending": {}, "next_id": 0},
        "nano": {"pending": {}, "next_id": 0},
        "fastsam": {"pending": {}, "next_id": 0},
    }
    slot_locks = {
        "yolo": threading.Lock(),
        "nano": threading.Lock(),
        "fastsam": threading.Lock(),
    }
    stats = {
        "yolo": make_model_stats(),
        "nano": make_model_stats(),
        "fastsam": make_model_stats(),
    }
    stats_lock = threading.Lock()
    warmup_end = time.perf_counter() + WARMUP_SECONDS

    # Build threads
    threads: List[threading.Thread] = []

    # Capture
    threads.append(threading.Thread(
        target=capture_worker, name="capture",
        args=(args.input, nano_input_wh, shared_queue,
              frame_slots, frame_lock, qs * 4, shutdown),
    ))

    # 3 core workers (each has all 3 models, picks oldest task dynamically)
    for i, models in enumerate(core_models):
        threads.append(threading.Thread(
            target=core_worker, name=f"core-{i}",
            args=(f"core-{i}", models, shared_queue, raw_queues, shutdown),
        ))

    # YOLO postprocess
    for i in range(args.yolo_post_workers):
        threads.append(threading.Thread(
            target=yolo_post_worker, name=f"yolo-post-{i}",
            args=(f"yolo-post-{i}", q_yolo_raw,
                  result_slots["yolo"], slot_locks["yolo"],
                  frame_slots, frame_lock,
                  args.yolo_score, args.yolo_nms,
                  stats, stats_lock, warmup_end, shutdown),
        ))

    # NanoDet postprocess
    for i in range(args.nano_post_workers):
        threads.append(threading.Thread(
            target=nano_post_worker, name=f"nano-post-{i}",
            args=(f"nano-post-{i}", q_nano_raw, nano_decoder,
                  result_slots["nano"], slot_locks["nano"],
                  frame_slots, frame_lock,
                  stats, stats_lock, warmup_end, shutdown),
        ))

    # FastSAM postprocess (2 workers by default)
    for i in range(args.fastsam_post_workers):
        threads.append(threading.Thread(
            target=fastsam_post_worker, name=f"fsam-post-{i}",
            args=(f"fsam-post-{i}", q_fsam_raw,
                  result_slots["fastsam"], slot_locks["fastsam"],
                  frame_slots, frame_lock,
                  args.fastsam_conf, args.fastsam_iou,
                  args.fastsam_topk, args.fastsam_max_det,
                  args.fastsam_boxes,
                  stats, stats_lock, warmup_end, shutdown),
        ))

    # Display
    disp_thread = threading.Thread(
        target=display_worker, name="display",
        args=(result_slots, slot_locks, shutdown),
        daemon=True,
    )

    # start
    n_cores = len(core_models)
    print(f"\n[Pipeline] Dynamic scheduling: {n_cores} core workers, each with all 3 models")
    print(f"  Postprocess: YOLO={args.yolo_post_workers}  Nano={args.nano_post_workers}  FastSAM={args.fastsam_post_workers}")
    print(f"  Queue size={qs}")
    print(f"  Total model instances: {n_cores * 3}")
    print(f"  Scheduling: oldest task first on whichever core finishes first")
    print(f"[Pipeline] Ignoring first {int(WARMUP_SECONDS)}s for average stats\n")

    for t in threads:
        t.start()
    disp_thread.start()

    try:
        while not shutdown.is_set():
            shutdown.wait(timeout=0.5)
    except KeyboardInterrupt:
        print("\n[Main] Ctrl+C received, shutting down ...")
        shutdown.set()

    # Signal shared queue and raw queues
    shared_queue.close()
    for q in raw_queues.values():
        safe_put_sentinel(q, 8)

    for t in threads:
        t.join(timeout=3.0)

    print_avg_stats(stats, warmup_end)

    # Cleanup all 9 model instances
    for models in core_models:
        for rknn in models.values():
            try:
                rknn.release()
            except Exception:
                pass

    print("[Main] Done.")


if __name__ == "__main__":
    main()
