#!/usr/bin/env python3
import argparse
import os
import re
from collections import deque

import numpy as np
import onnx
import onnx_graphsurgeon as gs


CV_BRANCH_RE = re.compile(r"(?:^|/)(cv2|cv3)\.(\d+)")


def sanitize_name_part(value):
    return re.sub(r"[^0-9a-zA-Z_]+", "_", str(value))


def make_output_name(prefix, shape, idx):
    if shape is not None and len(shape) == 4:
        h = sanitize_name_part(shape[2])
        w = sanitize_name_part(shape[3])
        return f"{prefix}_{h}x{w}"
    return f"{prefix}_{idx}"


def parse_branch_and_scale(node_name):
    if not node_name:
        return None, None
    match = CV_BRANCH_RE.search(node_name.lower())
    if not match:
        return None, None
    branch = "reg" if match.group(1) == "cv2" else "cls"
    scale = int(match.group(2))
    return branch, scale


def remove_nodes_downstream(graph, start_nodes):
    """
    Remove all nodes downstream from the given start nodes.
    """
    queue = deque(start_nodes)
    visited = set()
    to_remove = []

    while queue:
        node = queue.popleft()
        node_id = id(node)
        if node_id in visited:
            continue

        visited.add(node_id)
        to_remove.append(node)

        for output in node.outputs:
            for consumer in output.outputs:
                if id(consumer) not in visited:
                    queue.append(consumer)

    for node in to_remove:
        if node in graph.nodes:
            graph.nodes.remove(node)

    return len(to_remove)


def collect_downstream_ops(start_node, max_nodes=2000):
    """
    Collect downstream op types from a start node. Used as a fallback classifier.
    """
    queue = deque([start_node])
    visited = set()
    ops = set()

    while queue and len(visited) < max_nodes:
        node = queue.popleft()
        node_id = id(node)
        if node_id in visited:
            continue

        visited.add(node_id)
        ops.add(node.op)

        for out in node.outputs:
            for consumer in out.outputs:
                if id(consumer) not in visited:
                    queue.append(consumer)

    return ops


def collect_conv_reshape_candidates(graph):
    candidates = []

    for node in graph.nodes:
        if node.op != "Conv" or len(node.outputs) != 1:
            continue

        out = node.outputs[0]
        if len(out.outputs) != 1:
            continue

        reshape = out.outputs[0]
        if reshape.op != "Reshape":
            continue

        if out.shape is not None and len(out.shape) != 4:
            continue

        candidates.append((node, out, reshape))

    return candidates


def detect_head_branches(graph):
    """
    Detect YOLO11 detection head branches.
    """
    candidates = collect_conv_reshape_candidates(graph)

    named = []
    unnamed = []
    for node, out, reshape in candidates:
        branch, scale = parse_branch_and_scale(node.name)
        if branch is None:
            unnamed.append((node, out, reshape))
            continue
        named.append(
            {
                "branch": branch,
                "scale": scale,
                "node": node,
                "out": out,
                "reshape": reshape,
                "source": "name",
            }
        )

    if named:
        return named

    # Fallback for models with stripped names.
    fallback = []
    for node, out, reshape in unnamed:
        ops = collect_downstream_ops(reshape)
        if "Softmax" in ops or "MatMul" in ops:
            branch = "reg"
        elif "Sigmoid" in ops:
            branch = "cls"
        else:
            continue
        fallback.append(
            {
                "branch": branch,
                "scale": None,
                "node": node,
                "out": out,
                "reshape": reshape,
                "source": "heuristic",
            }
        )

    return fallback


def shape_sort_key(shape):
    if shape is None or len(shape) != 4:
        return (1, 0, "")

    h, w = shape[2], shape[3]
    if isinstance(h, int) and isinstance(w, int):
        return (0, -(h * w), f"{h}x{w}")

    return (1, 0, f"{h}x{w}")


def sort_branches(branches):
    # Keep deterministic ordering: by parsed scale if available, otherwise by spatial size.
    with_scale = [b for b in branches if b["scale"] is not None]
    without_scale = [b for b in branches if b["scale"] is None]

    with_scale.sort(key=lambda x: x["scale"])
    without_scale.sort(key=lambda x: shape_sort_key(x["out"].shape))

    return with_scale + without_scale


def modify_yolo11_head(input_model, output_model):
    print(f"Loading {input_model}...")
    model = onnx.load(input_model)
    graph = gs.import_onnx(model)
    graph.cleanup().toposort()

    print(f"Original model has {len(graph.nodes)} nodes")
    print(f"Original outputs: {[o.name for o in graph.outputs]}\n")

    branches = detect_head_branches(graph)
    reg_branches = sort_branches([b for b in branches if b["branch"] == "reg"])
    cls_branches = sort_branches([b for b in branches if b["branch"] == "cls"])

    print(f"Detected {len(reg_branches)} regression and {len(cls_branches)} classification branches")
    for b in reg_branches:
        print(f"  [REG/{b['source']}] {b['node'].name} shape={b['out'].shape}")
    for b in cls_branches:
        print(f"  [CLS/{b['source']}] {b['node'].name} shape={b['out'].shape}")
    print()

    if not reg_branches or not cls_branches:
        raise RuntimeError(
            "Could not detect both regression and classification head branches. "
            "Aborting to avoid exporting a broken model."
        )

    if len(reg_branches) != len(cls_branches):
        raise RuntimeError(
            f"Unmatched head branches: reg={len(reg_branches)}, cls={len(cls_branches)}. "
            "Aborting to avoid ambiguous output mapping."
        )

    # Remove post-processing graph once from all reshape heads.
    reshape_heads = [b["reshape"] for b in reg_branches + cls_branches]
    removed = remove_nodes_downstream(graph, reshape_heads)
    print(f"Removed {removed} downstream post-processing node(s)\n")

    new_outputs = []

    for idx, reg in enumerate(reg_branches):
        reg_out = reg["out"]
        reg_out.name = make_output_name("reg", reg_out.shape, idx)
        new_outputs.append(reg_out)
        print(f"  -> Output: {reg_out.name} {reg_out.shape}")

    for idx, cls in enumerate(cls_branches):
        cls_out = cls["out"]
        sigmoid_out = gs.Variable(
            name=make_output_name("cls", cls_out.shape, idx),
            dtype=cls_out.dtype if cls_out.dtype is not None else np.float32,
            shape=cls_out.shape,
        )
        sigmoid_node = gs.Node(
            op="Sigmoid",
            name=(cls["node"].name or f"cls_{idx}") + "_Sigmoid",
            inputs=[cls_out],
            outputs=[sigmoid_out],
        )
        graph.nodes.append(sigmoid_node)
        new_outputs.append(sigmoid_out)
        print(f"  -> Output: {sigmoid_out.name} {sigmoid_out.shape} (with Sigmoid)")

    graph.outputs = new_outputs
    graph.cleanup().toposort()

    print(f"\nModified model has {len(graph.nodes)} nodes")
    print(f"New outputs ({len(graph.outputs)}):")
    for out in graph.outputs:
        print(f"  - {out.name} {out.shape}")

    modified_model = gs.export_onnx(graph)

    print(f"\nOriginal exported IR version: {modified_model.ir_version}")
    if modified_model.ir_version > 9:
        print("Setting IR version to 9 for compatibility...")
        modified_model.ir_version = 9
        for opset in modified_model.opset_import:
            if opset.domain == "" or opset.domain == "ai.onnx":
                if opset.version > 13:
                    opset.version = 13
        print(f"Updated to IR version: {modified_model.ir_version}")

    onnx.save(modified_model, output_model)
    print(f"\nDone. Saved as {output_model}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Remove post-processing nodes from YOLO11 ONNX model and expose per-scale outputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python utils/removenodes11.py yolo11n.onnx
  python utils/removenodes11.py yolo11m.onnx modified_yolo11m.onnx
        """,
    )
    parser.add_argument("input_model", type=str, help="Path to input ONNX model")
    parser.add_argument(
        "output_model",
        type=str,
        nargs="?",
        help="Path to output ONNX model (optional, auto-generated if not provided)",
    )
    args = parser.parse_args()

    if args.output_model is None:
        base_dir = os.path.dirname(args.input_model) or "."
        base_name = os.path.basename(args.input_model)
        args.output_model = os.path.join(base_dir, f"modified_{base_name}")
        print(f"Output model not specified, using: {args.output_model}\n")

    modify_yolo11_head(args.input_model, args.output_model)