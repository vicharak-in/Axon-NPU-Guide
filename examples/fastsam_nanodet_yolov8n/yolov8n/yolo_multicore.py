#!/usr/bin/env python3
"""
YOLOv8n multicore benchmark — runs a single YOLOv8n model across all 3 NPU cores for maximum throughput on a live video stream.
"""

import argparse
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


class DropOldestQueue:
    
    def __init__(self, maxlen: int):
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._dq: deque = deque(maxlen=maxlen)
        self._closed = False

    def put(self, item):
        with self._not_empty:
            self._dq.append(item)  # auto-drops oldest if full
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
        print("  YOLO: no samples collected after warmup")
        return
    avg_inf = stats["inf_ms_sum"] / n
    avg_post = stats["post_ms_sum"] / n
    active_secs = max((stats["last_ts"] or warmup_end) - warmup_end, 1e-6)
    avg_fps = n / active_secs
    print(f"  YOLO: fps={avg_fps:.2f}  inf={avg_inf:.2f}ms  post={avg_post:.2f}ms  samples={n}")


# Preprocessing
def letterbox_topleft(img, target_h, target_w):
    h, w = img.shape[:2]
    scale = min(target_h / h, target_w / w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (new_w, new_h))
    canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
    canvas[:new_h, :new_w] = resized
    return canvas, scale, new_h, new_w


def preprocess_640_rgb(frame_bgr):
    canvas_bgr, scale, new_h, new_w = letterbox_topleft(frame_bgr, 640, 640)
    canvas_rgb = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB)
    inp = np.expand_dims(canvas_rgb, 0)
    meta = {"scale": scale, "new_h": new_h, "new_w": new_w}
    return inp, meta


# YOLOv8 postprocess
def _dfl(position):
    n, c, h, w = position.shape
    mc = c // 4
    y = position.reshape(n, 4, mc, h, w)
    y = np.exp(y - np.max(y, axis=2, keepdims=True))
    y /= np.sum(y, axis=2, keepdims=True)
    acc = np.arange(mc, dtype=np.float32).reshape(1, 1, mc, 1, 1)
    return (y * acc).sum(2)


def _yolo_box_process(position, input_h, input_w):
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


def _yolo_organize_outputs(outputs, num_classes=80):
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


def yolo_postprocess(raw_outputs, scale, orig_h, orig_w,
                     score_thresh, nms_thresh, num_classes=80, input_size=640):
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

def capture_worker(source, input_queue, frame_slots, frame_lock,
                   max_stored_frames, shutdown):
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

        inp, meta = preprocess_640_rgb(frame)
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


def post_worker(name, raw_q, result_slot, slot_lock, frame_slots, frame_lock,
                score_thresh, nms_thresh, stats, stats_lock, warmup_end, shutdown):
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
    win_name = "YOLOv8n Detection (3-core)"
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

def parse_args():
    p = argparse.ArgumentParser(
        description="YOLOv8n multicore benchmark — all 3 NPU cores",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Example:
  python3 yolo_multicore.py \\
    --input "<video_stream_ip>" \\
    --model <modelname.rknn>
""",
    )
    p.add_argument("--input", required=True, help="Video path, camera index, or IP webcam URL")
    p.add_argument("--model", required=True, help="YOLOv8n RKNN model path")
    p.add_argument("--input_size", type=int, default=640, help="Input size (default: 640)")
    p.add_argument("--score_thresh", type=float, default=0.45, help="Score threshold")
    p.add_argument("--nms_thresh", type=float, default=0.45, help="NMS IoU threshold")
    p.add_argument("--post_workers", type=int, default=1, help="Postprocess workers")
    p.add_argument("--queue_size", type=int, default=4, help="Queue size (drop-oldest)")
    return p.parse_args()

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
        print(f"[{core_name}] Loading YOLOv8n ...")
        rknns.append(load_rknn_model(args.model, core_mask))

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

    threads.append(threading.Thread(
        target=capture_worker, name="capture",
        args=(args.input, input_queue, frame_slots, frame_lock, qs * 4, shutdown),
    ))

    for i, rknn in enumerate(rknns):
        threads.append(threading.Thread(
            target=core_worker, name=f"core-{i}",
            args=(f"core-{i}", rknn, input_queue, raw_queue, shutdown),
        ))

    # Postprocess workers
    for i in range(args.post_workers):
        threads.append(threading.Thread(
            target=post_worker, name=f"yolo-post-{i}",
            args=(f"yolo-post-{i}", raw_queue, result_slot, slot_lock,
                  frame_slots, frame_lock, args.score_thresh, args.nms_thresh,
                  stats, stats_lock, warmup_end, shutdown),
        ))

    disp_thread = threading.Thread(
        target=display_worker, name="display",
        args=(result_slot, slot_lock, shutdown),
        daemon=True,
    )

    print(f"\n[Pipeline] YOLOv8n on 3 NPU cores, {args.post_workers} post worker(s)")
    print(f"  Queue size={qs}")
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
