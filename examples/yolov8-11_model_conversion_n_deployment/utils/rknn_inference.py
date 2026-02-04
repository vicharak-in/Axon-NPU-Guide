#this file is only to be run on RK35xx hardware for yolov8 inference

import cv2
import numpy as np
import argparse
from rknnlite.api import RKNNLite


# ---------------- Config ----------------

OBJ_THRESH = 0.25
NMS_THRESH = 0.45

CLASSES = (
    "person","bicycle","car","motorbike","aeroplane","bus","train","truck","boat","traffic light",
    "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
    "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
    "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","sofa",
    "pottedplant","bed","diningtable","toilet","tvmonitor","laptop","mouse","remote","keyboard",
    "cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase",
    "scissors","teddy bear","hair drier","toothbrush"
)


# ---------------- Utils ----------------

def letterbox(im, new_shape):
    """
    Letterbox an image to target shape (H, W) with padding.
    """
    # Handle both int (square) and tuple (rectangular)
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    
    target_h, target_w = new_shape
    h, w = im.shape[:2]
    
    # Scale ratio (new / old)
    r = min(target_h / h, target_w / w)
    
    # Compute new unpadded dimensions
    new_unpad_h, new_unpad_w = int(h * r), int(w * r)
    im = cv2.resize(im, (new_unpad_w, new_unpad_h))
    
    # Compute padding
    pad_h = target_h - new_unpad_h
    pad_w = target_w - new_unpad_w
    top = pad_h // 2
    left = pad_w // 2
    
    # Create padded output
    out = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
    out[top:top+new_unpad_h, left:left+new_unpad_w] = im
    
    return out, r, (left, top)


def dfl(position):
    n, c, h, w = position.shape
    mc = c // 4
    y = position.reshape(n, 4, mc, h, w)
    y = np.exp(y - np.max(y, axis=2, keepdims=True))
    y /= np.sum(y, axis=2, keepdims=True)
    acc = np.arange(mc).reshape(1, 1, mc, 1, 1)
    return (y * acc).sum(2)


def box_process(position, input_size):

    if isinstance(input_size, int):
        input_size = (input_size, input_size)
    
    input_h, input_w = input_size
    feat_h, feat_w = position.shape[2:]
    
    grid_x, grid_y = np.meshgrid(np.arange(feat_w), np.arange(feat_h))
    grid = np.stack((grid_x, grid_y), axis=0).reshape(1, 2, feat_h, feat_w)
    
    # Separate strides for H and W (important for rectangular inputs)
    stride_h = input_h / feat_h
    stride_w = input_w / feat_w
    stride = np.array([stride_w, stride_h]).reshape(1, 2, 1, 1)  # x, y order
    
    position = dfl(position)
    
    box1 = grid + 0.5 - position[:, 0:2]
    box2 = grid + 0.5 + position[:, 2:4]
    return np.concatenate((box1 * stride, box2 * stride), axis=1)


def nms_boxes(boxes, scores):
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(ovr <= NMS_THRESH)[0] + 1]

    return keep


def organize_outputs(outputs):
    boxes = {}
    classes = {}

    for out in outputs:
        _, c, h, w = out.shape
        key = (h, w)

        if c == 64 or c == 4:
            boxes[key] = out
        elif c == len(CLASSES):
            classes[key] = out

    ordered = []
    for k in sorted(boxes.keys(), key=lambda x: x[0]):
        ordered.append(boxes[k])
        ordered.append(classes[k])

    return ordered


# ---------------- Postprocess ----------------

def post_process(outputs, input_size):

    outputs = organize_outputs(outputs)

    boxes_all, cls_all, scores_all = [], [], []

    for i in range(3):
        box_out = outputs[i * 2]
        cls_out = outputs[i * 2 + 1]

        boxes = box_process(box_out, input_size)
        boxes = boxes.transpose(0, 2, 3, 1).reshape(-1, 4)

        cls_out = cls_out.transpose(0, 2, 3, 1).reshape(-1, len(CLASSES))
        cls_score = np.max(cls_out, axis=1)
        cls_id = np.argmax(cls_out, axis=1)

        mask = cls_score >= OBJ_THRESH
        boxes_all.append(boxes[mask])
        cls_all.append(cls_id[mask])
        scores_all.append(cls_score[mask])

    boxes = np.concatenate(boxes_all)
    classes = np.concatenate(cls_all)
    scores = np.concatenate(scores_all)

    final_boxes, final_cls, final_scores = [], [], []

    for c in np.unique(classes):
        idx = np.where(classes == c)
        keep = nms_boxes(boxes[idx], scores[idx])
        final_boxes.append(boxes[idx][keep])
        final_cls.append(classes[idx][keep])
        final_scores.append(scores[idx][keep])

    if not final_boxes:
        return None, None, None

    return (
        np.concatenate(final_boxes),
        np.concatenate(final_cls),
        np.concatenate(final_scores),
    )


def parse_size(size_str):
    """
    Parse size argument: accepts 'H,W' or single int for square.
    
    Examples:
        '640' -> (640, 640)
        '480,640' -> (480, 640)
    """
    if ',' in size_str:
        h, w = map(int, size_str.split(','))
        return (h, w)
    else:
        s = int(size_str)
        return (s, s)


# ---------------- Main ----------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='YOLOv8 RKNN Inference (RK35xx)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Square input (traditional)
  python rknn_inference.py --model model.rknn --image test.jpg --size 640

  # Rectangular input
  python rknn_inference.py --model model.rknn --image test.jpg --size 480,640

  # With quantized model
  python rknn_inference.py --model model.rknn --image test.jpg --size 1792,1280 --quantized
        """
    )
    parser.add_argument("--model", required=True, help="Path to RKNN model")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--size", type=str, default="640",
                        help="Input size: single int for square (640) or H,W for rectangular (480,640)")
    parser.add_argument("--imgname", type=str, default="result_rknn_output.jpg", 
                        help="Output image filename (default: result_rknn_output.jpg)")
    parser.add_argument("--quantized", action="store_true",
                        help="Use this flag if model was quantized with mean/std values")
    
    args = parser.parse_args()
    
    # Parse size (supports both '640' and '480,640')
    input_size = parse_size(args.size)
    print(f"[Input size] {input_size[0]}x{input_size[1]} (HxW)")

    rknn = RKNNLite()
    
    ret = rknn.load_rknn(args.model)
    if ret != 0:
        print(f"Load RKNN model failed! Error code: {ret}")
        exit(ret)

    ret = rknn.init_runtime()
    if ret != 0:
        print(f"Init runtime failed! Error code: {ret}")
        print("Make sure you are running on RK35xx hardware with proper RKNN runtime installed.")
        exit(ret)

    img = cv2.imread(args.image)
    if img is None:
        print(f"Failed to load image: {args.image}")
        exit(1)
        
    img_lb, ratio, pad = letterbox(img, input_size)
    inp = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)

    # CRITICAL DIFFERENCE:
    # Quantized models have mean/std baked in during conversion
    # They expect UINT8 input in NHWC format, normalization happens internally
    # Non-quantized (float) models expect normalized float32 in NCHW format
    
    if args.quantized:
        # Quantized model: pass uint8, NHWC format
        # RKNN handles normalization internally (mean=0, std=255)
        print("[Quantized mode] Using uint8 input, NHWC format")
        inp = np.expand_dims(inp, axis=0)  # (1, H, W, 3)
        outputs = rknn.inference(inputs=[inp], data_format='nhwc')
    else:
        # Non-quantized model: normalize manually, NCHW format
        print("[Float mode] Using float32 input, NCHW format")
        inp = inp.astype(np.float32) / 255.0
        inp = inp.transpose(2, 0, 1)[None]  # (1, 3, H, W)
        outputs = rknn.inference(inputs=[inp], data_format='nchw')

    if outputs is None:
        print("Inference failed!")
        print("Make sure --size matches one of the compiled graph sizes in the model.")
        exit(1)

    boxes, classes, scores = post_process(outputs, input_size)

    if boxes is not None:
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad[0]) / ratio
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad[1]) / ratio

        print(f"Detected {len(boxes)} objects:")
        for b, c, s in zip(boxes, classes, scores):
            x1, y1, x2, y2 = map(int, b)
            print(f"  {CLASSES[c]}: {s:.2f} @ ({x1}, {y1}, {x2}, {y2})")
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{CLASSES[c]} {s:.2f}",
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        print("No objects detected")

    cv2.imwrite(args.imgname, img)
    print(f"Saved {args.imgname}")
    
    rknn.release()