#!/usr/bin/env python3
import onnx
import onnx_graphsurgeon as gs
import numpy as np
import argparse
import os

def remove_nodes_downstream(graph, start_nodes):
    """
    Remove all nodes downstream from the given start nodes.
    Uses BFS traversal to find all connected downstream nodes.
    """
    to_remove = []
    queue = list(start_nodes)
    visited_ids = set()
    
    while queue:
        node = queue.pop(0)
        node_id = id(node)
        if node_id in visited_ids:
            continue
        visited_ids.add(node_id)
        to_remove.append(node)
        
        # Add all consumers to the queue
        for output in node.outputs:
            for consumer in output.outputs:
                if id(consumer) not in visited_ids:
                    queue.append(consumer)
    
    # Remove nodes from graph
    for node in to_remove:
        if node in graph.nodes:
            graph.nodes.remove(node)

def modify_yolo11_head(input_model, output_model):
    # ---------- Load model ----------
    print(f"Loading {input_model}...")
    model = onnx.load(input_model)
    graph = gs.import_onnx(model)
    graph.cleanup().toposort()
    print(f"Original model has {len(graph.nodes)} nodes")
    print(f"Original outputs: {[o.name for o in graph.outputs]}\n")

    # Collect Conv nodes to process
    reg_convs = []  # Regression heads: 64 channels
    cls_convs = []  # Classification heads: 80 channels

    # ---------- Pass 1: find Conv heads ----------
    # We need to find Conv nodes that feed directly into Reshape nodes
    # These are the detection heads in model.23
    for node in graph.nodes:
        if node.op != "Conv":
            continue

        # Check if this Conv has outputs
        if len(node.outputs) == 0:
            continue
        
        out = node.outputs[0]
        
        # Check for valid shape
        if not hasattr(out, 'shape') or out.shape is None:
            continue
        
        shape = out.shape
        if len(shape) != 4:
            continue

        _, c, h, w = shape

        # Only process Conv nodes with the right channel count and spatial dimensions
        if c not in [64, 80] or h not in [160, 80, 40]:
            continue
        
        # Check if this Conv feeds directly into a Reshape (detection head pattern)
        consumers = out.outputs
        if len(consumers) != 1:
            continue
        
        if consumers[0].op != "Reshape":
            continue

        # This is a detection head!
        if c == 64:
            print(f"[FOUND REG HEAD] {node.name}: {list(shape)}")
            reg_convs.append((node, out))
        elif c == 80:
            print(f"[FOUND CLS HEAD] {node.name}: {list(shape)}")
            cls_convs.append((node, out))

    print(f"\nFound {len(reg_convs)} regression heads and {len(cls_convs)} classification heads\n")

    new_outputs = []

    # ---------- Process regression heads ----------
    for conv_node, conv_out in reg_convs:
        print(f"Processing regression: {conv_node.name}")
        
        # Find all downstream nodes and remove them
        downstream_nodes = []
        for consumer in conv_out.outputs:
            downstream_nodes.append(consumer)
        
        if downstream_nodes:
            remove_nodes_downstream(graph, downstream_nodes)
            print(f"  -> Removed {len(downstream_nodes)} downstream node(s)")
        
        # Use Conv output directly
        conv_out.name = f"reg_{conv_out.shape[2]}"
        new_outputs.append(conv_out)
        print(f"  -> Output: {conv_out.name} {list(conv_out.shape)}")

    # ---------- Process classification heads ----------
    for conv_node, conv_out in cls_convs:
        print(f"Processing classification: {conv_node.name}")
        
        # Find all downstream nodes and remove them
        downstream_nodes = []
        for consumer in conv_out.outputs:
            downstream_nodes.append(consumer)
        
        if downstream_nodes:
            remove_nodes_downstream(graph, downstream_nodes)
            print(f"  -> Removed {len(downstream_nodes)} downstream node(s)")
        
        # Create Sigmoid node
        sigmoid_out = gs.Variable(
            name=f"cls_{conv_out.shape[2]}",
            dtype=np.float32,
            shape=conv_out.shape
        )
        
        sigmoid_node = gs.Node(
            op="Sigmoid",
            name=f"{conv_node.name}_Sigmoid",
            inputs=[conv_out],
            outputs=[sigmoid_out]
        )
        
        graph.nodes.append(sigmoid_node)
        new_outputs.append(sigmoid_out)
        print(f"  -> Added Sigmoid")
        print(f"  -> Output: {sigmoid_out.name} {list(sigmoid_out.shape)}")

    # ---------- Finalize graph ----------
    graph.outputs = new_outputs
    graph.cleanup().toposort()

    print(f"\nModified model has {len(graph.nodes)} nodes")
    print(f"New outputs ({len(new_outputs)}):")
    for out in new_outputs:
        print(f"  - {out.name} {list(out.shape)}")

    # Export the modified model
    modified_model = gs.export_onnx(graph)

    # Set IR version for compatibility
    print(f"\nOriginal model IR version: {modified_model.ir_version}")
    if modified_model.ir_version > 9:
        print("Setting IR version to 9 for compatibility...")
        modified_model.ir_version = 9
        for opset in modified_model.opset_import:
            if opset.domain == '' or opset.domain == 'ai.onnx':
                if opset.version > 13:
                    opset.version = 13
        print(f"Updated to IR version: {modified_model.ir_version}")

    onnx.save(modified_model, output_model)

    print(f"\nDone. Saved as {output_model}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Remove post-processing nodes from YOLOv11 ONNX model and add 6 outputs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (auto-generates output name)
  python removenodes11n.py yolo11n.onnx
  
  # Specify custom output path
  python removenodes11n.py yolo11n.onnx modified_yolo11n.onnx
        """
    )
    parser.add_argument('input_model', type=str, 
                        help='Path to input ONNX model')
    parser.add_argument('output_model', type=str, nargs='?',
                        help='Path to output ONNX model (optional, auto-generated if not provided)')
    
    args = parser.parse_args()

    # Generate output filename if not provided
    if args.output_model is None:
        base_dir = os.path.dirname(args.input_model) or '.'
        base_name = os.path.basename(args.input_model)
        name_without_ext = os.path.splitext(base_name)[0]
        args.output_model = os.path.join(base_dir, f"modified_{base_name}")
        print(f"Output model not specified, using: {args.output_model}\n")
    
    modify_yolo11_head(
        input_model=args.input_model,
        output_model=args.output_model
    )