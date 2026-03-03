#!/usr/bin/env python3
"""
NanoDet-Plus multicore benchmark — runs a single NanoDet-Plus model across all
3 NPU cores for maximum throughput on a live video stream.

Architecture:
  capture → preprocess (direct resize 416×416, BGR) → SharedQueue (drop-oldest)
  → 3 core workers (one RKNN instance per core) → raw output queue
  → postprocess worker(s) → display
"""

import argparse
import math
import queue
import threading
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

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


# ── Shared Queue ───────────────────────────────────────────────────────────────

class DropOldestQueue:
    """Thread-safe FIFO with drop-oldest overflow and efficient blocking get."""

    def __init__(self, maxlen: int):
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._dq: deque = deque(maxlen=maxlen)
        self._closed = False

    def put(self, item):
        with self._not_empty:
            self._dq.append(item)
            self._not_empty.notify()

    def get(self, timeout: float = 0.05):
        deadline = time.monotonic() + timeout
        with self._not_empty:
            while True:
                if self._dq:
                    return self._dq.popleft()
                if self._closed:
                    return None
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._not_empty.wait(timeout=remaining)

    def close(self):
        with self._not_empty:
            self._closed = True
            self._not_empty.notify_all()


# ── Stats ──────────────────────────────────────────────────────────────────────

def make_stats() -> dict:
    return {"samples": 0, "inf_ms_sum": 0.0, "post_ms_sum": 0.0, "last_ts": None}


def update_stats(stats, lock, now, warmup_end, inf_ms, post_ms):
    if now < warmup_end:
        return
    with lock:
        stats["samples"] += 1
        stats["inf_ms_sum"] += inf_ms
        stats["post_ms_sum"] += post_ms
        stats["last_ts"] = now


def print_stats(stats, warmup_end):
    n = stats["samples"]
    if n == 0:
        print("  NanoDet: no samples collected after warmup")
        return
    avg_inf = stats["inf_ms_sum"] / n
    avg_post = stats["post_ms_sum"] / n
    active_secs = max((stats["last_ts"] or warmup_end) - warmup_end, 1e-6)
    avg_fps = n / active_secs
    print(f"  NanoDet: fps={avg_fps:.2f}  inf={avg_inf:.2f}ms  post={avg_post:.2f}ms  samples={n}")


# ── Preprocessing ──────────────────────────────────────────────────────────────

def preprocess_nano(frame_bgr, input_w, input_h):
    resized = cv2.resize(frame_bgr, (input_w, input_h))
    inp = np.expand_dims(resized, 0)  # (1, H, W, 3) uint8 BGR
    meta = {"orig_shape": frame_bgr.shape}
    return inp, meta


# ── Math helpers ───────────────────────────────────────────────────────────────

def _softmax(x, axis):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def _nms(boxes, scores, iou_thresh):
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


# ── NanoDet-Plus postprocess ──────────────────────────────────────────────────

class NanoDetDecoder:
    def __init__(self, input_shape, num_classes, reg_max, strides,
                 score_thresh, nms_thresh):
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

    def decode(self, raw_outputs, orig_shape):
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


# ── Visualization ──────────────────────────────────────────────────────────────

def draw_detections(img, detections, class_names, color, label_prefix="", score_thresh=0.0):
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


# ── Model loading ──────────────────────────────────────────────────────────────

def load_rknn_model(model_path, core_mask):
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


# ── Queue helpers ──────────────────────────────────────────────────────────────

def put_drop_oldest(q, item):
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


def safe_put_sentinel(q, count):
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


# ── Thread workers ─────────────────────────────────────────────────────────────

def capture_worker(source, input_w, input_h, input_queue, frame_slots,
                   frame_lock, max_stored_frames, shutdown):
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[Capture] ERROR: cannot open {source}")
        shutdown.set()
        input_queue.close()
        return

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

        inp, meta = preprocess_nano(frame, input_w, input_h)
        input_queue.put({
            "frame_id": frame_id,
            "orig_h": frame.shape[0],
            "orig_w": frame.shape[1],
            "inp": inp,
            "meta": meta,
        })
        frame_id += 1

    cap.release()
    input_queue.close()
    print(f"[Capture] stopped, {frame_id} frames captured")


def core_worker(name, rknn, input_queue, raw_queue, shutdown):
    count = 0
    while not shutdown.is_set():
        task = input_queue.get(timeout=0.05)
        if task is None:
            if shutdown.is_set():
                break
            continue

        t0 = time.perf_counter()
        try:
            outputs = rknn.inference(inputs=[task["inp"]], data_format="nhwc")
        except Exception as e:
            print(f"[{name}] inference error: {e}, skipping")
            continue
        inf_ms = (time.perf_counter() - t0) * 1000.0

        if outputs is None:
            continue

        put_drop_oldest(raw_queue, {
            "frame_id": task["frame_id"],
            "orig_h": task["orig_h"],
            "orig_w": task["orig_w"],
            "meta": task["meta"],
            "outputs": outputs,
            "inf_ms": inf_ms,
        })
        count += 1

    print(f"[{name}] stopped, ran {count} inferences")


def post_worker(name, raw_q, decoder, result_slot, slot_lock, frame_slots,
                frame_lock, stats, stats_lock, warmup_end, shutdown):
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
        update_stats(stats, stats_lock, now, warmup_end, item["inf_ms"], post_ms)
        elapsed = now - t_start
        fps = count / elapsed if elapsed > 0 else 0.0
        if now - last_log >= 2.0:
            print(f"[{name}] fps: {fps:.1f}  inf: {item['inf_ms']:.1f}ms  post: {post_ms:.1f}ms")
            last_log = now

        with slot_lock:
            result_slot["pending"][item["frame_id"]] = vis

    print(f"[{name}] stopped, processed {count} frames")


def display_worker(result_slot, slot_lock, shutdown):
    win_name = "NanoDet+ Detection (3-core)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    while not shutdown.is_set():
        with slot_lock:
            pending = result_slot["pending"]
            nid = result_slot["next_id"]
            img = None
            if nid in pending:
                img = pending.pop(nid)
                result_slot["next_id"] = nid + 1
            elif pending:
                # frame nid was dropped before inference; jump to oldest available
                oldest = min(pending)
                img = pending.pop(oldest)
                result_slot["next_id"] = oldest + 1
        if img is not None:
            cv2.imshow(win_name, img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            shutdown.set()
            break

    cv2.destroyAllWindows()
    print("[Display] stopped")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="NanoDet-Plus multicore benchmark — all 3 NPU cores",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Example:
  python3 nanodet_multicore.py \\
    --input "<video_stream_ip>" \\
    --model <modelname.rknn>
""",
    )
    p.add_argument("--input", required=True, help="Video path, camera index, or IP webcam URL")
    p.add_argument("--model", required=True, help="NanoDet-Plus RKNN model path")
    p.add_argument("--input_size", type=str, default="416,416", help="Input W,H (default: 416,416)")
    p.add_argument("--num_classes", type=int, default=80)
    p.add_argument("--reg_max", type=int, default=7)
    p.add_argument("--strides", type=str, default="8,16,32,64")
    p.add_argument("--score_thresh", type=float, default=0.60, help="Score threshold")
    p.add_argument("--nms_thresh", type=float, default=0.6, help="NMS IoU threshold")
    p.add_argument("--post_workers", type=int, default=1, help="Postprocess workers")
    p.add_argument("--queue_size", type=int, default=4, help="Queue size (drop-oldest)")
    return p.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    shutdown = threading.Event()

    core0 = getattr(RKNNLite, "NPU_CORE_0", None)
    core1 = getattr(RKNNLite, "NPU_CORE_1", None)
    core2 = getattr(RKNNLite, "NPU_CORE_2", None)
    cores = [("CORE_0", core0), ("CORE_1", core1), ("CORE_2", core2)]

    # Load model on all 3 cores
    rknns = []
    for core_name, core_mask in cores:
        print(f"[{core_name}] Loading NanoDet+ ...")
        rknns.append(load_rknn_model(args.model, core_mask))

    # Decoder
    input_wh = tuple(map(int, args.input_size.split(",")))
    strides = list(map(int, args.strides.split(",")))
    decoder = NanoDetDecoder(
        input_shape=input_wh,
        num_classes=args.num_classes,
        reg_max=args.reg_max,
        strides=strides,
        score_thresh=args.score_thresh,
        nms_thresh=args.nms_thresh,
    )

    qs = max(args.queue_size, 1)
    input_queue = DropOldestQueue(maxlen=qs)
    raw_queue = queue.Queue(maxsize=qs)

    frame_slots = {}
    frame_lock = threading.Lock()
    result_slot = {"pending": {}, "next_id": 0}
    slot_lock = threading.Lock()
    stats = make_stats()
    stats_lock = threading.Lock()
    warmup_end = time.perf_counter() + WARMUP_SECONDS

    threads = []

    input_w, input_h = input_wh

    # Capture
    threads.append(threading.Thread(
        target=capture_worker, name="capture",
        args=(args.input, input_w, input_h, input_queue,
              frame_slots, frame_lock, qs * 4, shutdown),
    ))

    # 3 core workers
    for i, rknn in enumerate(rknns):
        threads.append(threading.Thread(
            target=core_worker, name=f"core-{i}",
            args=(f"core-{i}", rknn, input_queue, raw_queue, shutdown),
        ))

    # Postprocess workers
    for i in range(args.post_workers):
        threads.append(threading.Thread(
            target=post_worker, name=f"nano-post-{i}",
            args=(f"nano-post-{i}", raw_queue, decoder, result_slot, slot_lock,
                  frame_slots, frame_lock, stats, stats_lock, warmup_end, shutdown),
        ))

    # Display
    disp_thread = threading.Thread(
        target=display_worker, name="display",
        args=(result_slot, slot_lock, shutdown),
        daemon=True,
    )

    print(f"\n[Pipeline] NanoDet+ on 3 NPU cores, {args.post_workers} post worker(s)")
    print(f"  Input: {input_w}x{input_h}  Queue size={qs}")
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

    input_queue.close()
    safe_put_sentinel(raw_queue, 8)

    for t in threads:
        t.join(timeout=3.0)

    print(f"\n[Main] Average stats (after first {int(WARMUP_SECONDS)}s warmup):")
    print_stats(stats, warmup_end)

    for rknn in rknns:
        try:
            rknn.release()
        except Exception:
            pass

    print("[Main] Done.")


if __name__ == "__main__":
    main()
