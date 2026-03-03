#!/usr/bin/env python3
"""
FastSAM multicore benchmark — runs a single FastSAM model across all 3 NPU
cores for maximum throughput on a live video stream.

Architecture:
  capture → preprocess (top-left letterbox 640, RGB) → SharedQueue (drop-oldest)
  → 3 core workers (one RKNN instance per core) → raw output queue
  → postprocess worker(s) (default 2, since post≈128ms) → display
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

SENTINEL = object()
WARMUP_SECONDS = 10.0


# Shared Queue

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
        print("  FastSAM: no samples collected after warmup")
        return
    avg_inf = stats["inf_ms_sum"] / n
    avg_post = stats["post_ms_sum"] / n
    active_secs = max((stats["last_ts"] or warmup_end) - warmup_end, 1e-6)
    avg_fps = n / active_secs
    print(f"  FastSAM: fps={avg_fps:.2f}  inf={avg_inf:.2f}ms  post={avg_post:.2f}ms  samples={n}")


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

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88.0, 88.0)))


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
    raw_outputs, scale, new_h, new_w, orig_h, orig_w,
    conf_thresh=0.25, iou_thresh=0.9, topk=120, max_det=64, imgsz=640,
):
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

def draw_fastsam(img, masks, boxes, scores, draw_boxes=True):
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
                conf_thresh, iou_thresh, topk, max_det, draw_boxes,
                stats, stats_lock, warmup_end, shutdown):
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

        # Skip frames older than what's already displayed (saves ~200ms post work)
        with slot_lock:
            current_fid = result_slot.get("frame_id", -1)
        if item["frame_id"] <= current_fid:
            continue

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
        update_stats(stats, stats_lock, now, warmup_end, item["inf_ms"], post_ms)
        elapsed = now - t_start
        fps = count / elapsed if elapsed > 0 else 0.0
        if now - last_log >= 2.0:
            print(f"[{name}] fps: {fps:.1f}  inf: {item['inf_ms']:.1f}ms  post: {post_ms:.1f}ms")
            last_log = now

        with slot_lock:
            if item["frame_id"] > result_slot.get("frame_id", -1):
                result_slot["img"] = vis
                result_slot["frame_id"] = item["frame_id"]

    print(f"[{name}] stopped, processed {count} frames")


def display_worker(result_slot, slot_lock, shutdown):
    win_name = "FastSAM Segmentation (3-core)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    while not shutdown.is_set():
        with slot_lock:
            img = result_slot.get("img")
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
        description="FastSAM multicore benchmark — all 3 NPU cores",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Example:
  python3 fastsam_multicore.py \\
    --input "<video_stream_ip>" \\
    --model <modelname.rknn>
""",
    )
    p.add_argument("--input", required=True, help="Video path, camera index, or IP webcam URL")
    p.add_argument("--model", required=True, help="FastSAM RKNN model path")
    p.add_argument("--input_size", type=int, default=640, help="Input size (default: 640)")
    p.add_argument("--conf_thresh", type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--iou_thresh", type=float, default=0.9, help="NMS IoU threshold")
    p.add_argument("--topk", type=int, default=120, help="Max candidates before NMS")
    p.add_argument("--max_det", type=int, default=64, help="Max detections after NMS")
    p.add_argument("--draw_boxes", action="store_true", help="Draw bounding boxes on output")
    p.add_argument("--post_workers", type=int, default=2,
                   help="Postprocess workers (default: 2, FastSAM post is heavy)")
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
        print(f"[{core_name}] Loading FastSAM ...")
        rknns.append(load_rknn_model(args.model, core_mask))

    qs = max(args.queue_size, 1)
    input_queue = DropOldestQueue(maxlen=qs)
    raw_queue = queue.Queue(maxsize=qs * 2)  # slightly larger for 2 post workers

    frame_slots = {}
    frame_lock = threading.Lock()
    result_slot = {}
    slot_lock = threading.Lock()
    stats = make_stats()
    stats_lock = threading.Lock()
    warmup_end = time.perf_counter() + WARMUP_SECONDS

    threads = []

    # Capture
    threads.append(threading.Thread(
        target=capture_worker, name="capture",
        args=(args.input, input_queue, frame_slots, frame_lock, qs * 4, shutdown),
    ))

    # 3 core workers
    for i, rknn in enumerate(rknns):
        threads.append(threading.Thread(
            target=core_worker, name=f"core-{i}",
            args=(f"core-{i}", rknn, input_queue, raw_queue, shutdown),
        ))

    # Postprocess workers (default 2)
    for i in range(args.post_workers):
        threads.append(threading.Thread(
            target=post_worker, name=f"fsam-post-{i}",
            args=(f"fsam-post-{i}", raw_queue, result_slot, slot_lock,
                  frame_slots, frame_lock,
                  args.conf_thresh, args.iou_thresh,
                  args.topk, args.max_det, args.draw_boxes,
                  stats, stats_lock, warmup_end, shutdown),
        ))

    # Display
    disp_thread = threading.Thread(
        target=display_worker, name="display",
        args=(result_slot, slot_lock, shutdown),
        daemon=True,
    )

    print(f"\n[Pipeline] FastSAM on 3 NPU cores, {args.post_workers} post worker(s)")
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
