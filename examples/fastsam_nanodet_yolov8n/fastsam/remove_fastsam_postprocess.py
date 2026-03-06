#!/usr/bin/env python3

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, helper


# The 9 per-scale Conv outputs to expose (bbox, cls, mask × 3 scales)
NEW_OUTPUT_NAMES = [
    "/model.22/cv2.0/cv2.0.2/Conv_output_0",  # bbox  scale 0 (80×80)
    "/model.22/cv3.0/cv3.0.2/Conv_output_0",  # cls   scale 0
    "/model.22/cv4.0/cv4.0.2/Conv_output_0",  # mask  scale 0
    "/model.22/cv2.1/cv2.1.2/Conv_output_0",  # bbox  scale 1 (40×40)
    "/model.22/cv3.1/cv3.1.2/Conv_output_0",  # cls   scale 1
    "/model.22/cv4.1/cv4.1.2/Conv_output_0",  # mask  scale 1
    "/model.22/cv2.2/cv2.2.2/Conv_output_0",  # bbox  scale 2 (20×20)
    "/model.22/cv3.2/cv3.2.2/Conv_output_0",  # cls   scale 2
    "/model.22/cv4.2/cv4.2.2/Conv_output_0",  # mask  scale 2
]

PROTO_OUTPUT_NAME = "output1"


def remove_postprocess(input_path: str, output_path: str) -> None:
    model = onnx.load(input_path)
    graph = model.graph

    # Collect all tensor names that the new outputs need
    # Build a map: tensor_name -> node that produces it
    producer = {}
    for node in graph.node:
        for out in node.output:
            producer[out] = node

    # BFS backwards from the new output tensors to find all required nodes
    required_tensors = set()
    queue = list(NEW_OUTPUT_NAMES) + [PROTO_OUTPUT_NAME]
    while queue:
        tensor = queue.pop()
        if tensor in required_tensors:
            continue
        required_tensors.add(tensor)
        if tensor in producer:
            for inp in producer[tensor].input:
                queue.append(inp)

    # Remove nodes not needed by the new outputs
    required_nodes = []
    for node in graph.node:
        # Keep node if any of its outputs are required
        if any(o in required_tensors for o in node.output):
            required_nodes.append(node)

    removed_count = len(graph.node) - len(required_nodes)

    del graph.node[:]
    graph.node.extend(required_nodes)

    # Remove unused initializers
    all_node_inputs = set()
    for node in graph.node:
        for inp in node.input:
            all_node_inputs.add(inp)

    used_inits = [init for init in graph.initializer if init.name in all_node_inputs]
    removed_inits = len(graph.initializer) - len(used_inits)

    del graph.initializer[:]
    graph.initializer.extend(used_inits)

    # Replace graph outputs
    # Keep the proto output (output1), add 9 new per-scale outputs
    del graph.output[:]

    # Re-add proto output
    graph.output.append(
        helper.make_tensor_value_info(PROTO_OUTPUT_NAME, TensorProto.FLOAT, [1, 32, 160, 160])
    )
    scale_hw = {0: (80, 80), 1: (40, 40), 2: (20, 20)}
    head_channels = {"cv2": 64, "cv3": 1, "cv4": 32}

    for name in NEW_OUTPUT_NAMES:
        # Parse head type and scale from the name
        # e.g. "/model.22/cv2.0/cv2.0.2/Conv_output_0" → cv2, scale 0
        parts = name.split("/")
        head_part = parts[2]  # "cv2.0" or "cv3.1" etc.
        head_type = head_part.split(".")[0]  # "cv2", "cv3", "cv4"
        scale_idx = int(head_part.split(".")[1])  # 0, 1, 2

        c = head_channels[head_type]
        h, w = scale_hw[scale_idx]

        graph.output.append(
            helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, c, h, w])
        )

    # Clean up unused graph inputs
    main_input_name = graph.input[0].name
    init_names = {init.name for init in graph.initializer}
    needed_inputs = []
    for inp in graph.input:
        if inp.name == main_input_name or inp.name in init_names or inp.name in all_node_inputs:
            needed_inputs.append(inp)

    del graph.input[:]
    graph.input.extend(needed_inputs)

    onnx.checker.check_model(model)
    onnx.save(model, output_path)

    print(f"Input model : {input_path}")
    print(f"Output model: {output_path}")
    print(f"Nodes removed     : {removed_count}")
    print(f"Initializers removed: {removed_inits}")
    print(f"Outputs: {len(graph.output)} (1 proto + 9 per-scale heads)")
    for o in graph.output:
        dims = [d.dim_value for d in o.type.tensor_type.shape.dim]
        print(f"  {o.name}: {dims}")


def main():
    parser = argparse.ArgumentParser(
        description="Remove DFL/bbox-decode postprocessing from FastSAM ONNX, "
                    "exposing raw per-scale head outputs."
    )
    parser.add_argument("-i", "--input", required=True, help="Input ONNX model path")
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output ONNX model path (default: modified_<input>)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input model not found: {input_path}")

    if args.output:
        output_path = args.output
    else:
        output_path = str(input_path.parent / f"modified_{input_path.name}")

    remove_postprocess(str(input_path), output_path)


if __name__ == "__main__":
    main()
