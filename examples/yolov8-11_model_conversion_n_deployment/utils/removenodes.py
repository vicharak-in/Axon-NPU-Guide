#!/usr/bin/env python3
import onnx
import onnx_graphsurgeon as gs
import numpy as np
import argparse
import os


def remove_node_and_reconnect(graph, node):
    """
    Remove a node and reconnect its inputs to outputs if possible.
    Used for removing Reshape nodes while maintaining connectivity.
    """
    if len(node.inputs) > 0 and len(node.outputs) > 0:
        # Get the input tensor
        inp_tensor = node.inputs[0]
        out_tensor = node.outputs[0]
        
        # Reconnect all consumers of output to use input instead
        for consumer in out_tensor.outputs:
            for i, inp in enumerate(consumer.inputs):
                if inp == out_tensor:
                    consumer.inputs[i] = inp_tensor
    
    # Remove the node
    graph.nodes.remove(node)


def find_reshape_after_conv(node):
    """
    Find if there's a Reshape node directly connected to Conv output.
    Returns the Reshape node if found, None otherwise.
    """
    if len(node.outputs) == 0:
        return None
    
    conv_out = node.outputs[0]
    for consumer in conv_out.outputs:
        if consumer.op == "Reshape":
            return consumer
    
    return None


def remove_nodes_downstream(graph, start_nodes):
    """
    Remove all nodes downstream from the given start nodes.
    Uses BFS traversal to find all connected downstream nodes.
    """
    to_remove = []
    queue = list(start_nodes)
    visited_names = set()
    
    while queue:
        node = queue.pop(0)
        # Use node name for tracking since Node objects aren't hashable
        node_id = id(node)
        if node_id in visited_names:
            continue
        visited_names.add(node_id)
        to_remove.append(node)
        
        # Add all consumers to the queue
        for output in node.outputs:
            for consumer in output.outputs:
                if id(consumer) not in visited_names:
                    queue.append(consumer)
    
    # Remove nodes from graph
    for node in to_remove:
        if node in graph.nodes:
            graph.nodes.remove(node)


def modify_yolov8_head(input_model, output_model):
   
    # Load and parse the model
    model = onnx.load(input_model)
    graph = gs.import_onnx(model)
    graph.cleanup().toposort()
    
    print(f"Original model has {len(graph.nodes)} nodes")
    print(f"Original outputs: {[o.name for o in graph.outputs]}\n")
    
    # Categorize Conv nodes by their output channel patterns
    bbox_convs = []
    cls_convs = []
    
    for node in graph.nodes:
        if node.op != "Conv":
            continue
        
        # Get output tensor and shape
        if len(node.outputs) == 0:
            continue
        
        out_tensor = node.outputs[0]
        if not hasattr(out_tensor, 'shape') or out_tensor.shape is None:
            continue
        
        shape = out_tensor.shape
        if len(shape) != 4:
            continue
        
        # Channel dimension (NCHW format)
        channels = shape[1]
        
        # Check if this Conv has a Reshape consumer (indicates it's a detection head)
        reshape_node = find_reshape_after_conv(node)
        if reshape_node is None:
            continue
        
        # Use node name to identify the branch
        if 'cv2' in node.name:
            # This is bbox regression branch
            bbox_convs.append((node, out_tensor, reshape_node))
            print(f"Found bbox Conv: {node.name}, output: {out_tensor.name}, shape: {shape}")
        elif 'cv3' in node.name:
            # This is classification branch
            cls_convs.append((node, out_tensor, reshape_node))
            print(f"Found cls Conv: {node.name}, output: {out_tensor.name}, shape: {shape}")
        elif 'dfl' in node.name.lower():
            # Skip DFL processing nodes - these will be removed with downstream cleanup
            print(f"Skipping DFL node: {node.name}, output: {out_tensor.name}, shape: {shape}")
        else:
            # Fallback: use channel count heuristic
            if channels == 64 or (channels == 16 and channels % 4 == 0):
                bbox_convs.append((node, out_tensor, reshape_node))
                print(f"Found bbox Conv (by channels): {node.name}, output: {out_tensor.name}, shape: {shape}")
            else:
                cls_convs.append((node, out_tensor, reshape_node))
                print(f"Found cls Conv (by channels): {node.name}, output: {out_tensor.name}, shape: {shape}")
    
    print(f"\nFound {len(bbox_convs)} bbox Conv nodes and {len(cls_convs)} classification Conv nodes\n")
    
    # New outputs for the modified model
    new_outputs = []
    
    # Process bbox Conv nodes: remove Reshape and everything downstream
    for conv_node, conv_out, reshape_node in bbox_convs:
        print(f"Processing bbox branch: {conv_node.name}")
        
        # Remove the Reshape and all downstream nodes
        remove_nodes_downstream(graph, [reshape_node])
        
        # Use Conv output directly as a model output
        new_outputs.append(conv_out)
        print(f"  -> Output: {conv_out.name} {conv_out.shape}")
    
    # Process classification Conv nodes: remove Reshape, add Sigmoid
    for conv_node, conv_out, reshape_node in cls_convs:
        print(f"Processing classification branch: {conv_node.name}")
        
        # Remove the Reshape and all downstream nodes
        remove_nodes_downstream(graph, [reshape_node])
        
        # Create Sigmoid node
        sigmoid_out = gs.Variable(
            name=conv_out.name + "_sigmoid",
            dtype=conv_out.dtype if hasattr(conv_out, 'dtype') else np.float32,
            shape=conv_out.shape
        )
        
        sigmoid_node = gs.Node(
            op="Sigmoid",
            name=conv_node.name + "_Sigmoid",
            inputs=[conv_out],
            outputs=[sigmoid_out]
        )
        
        graph.nodes.append(sigmoid_node)
        new_outputs.append(sigmoid_out)
        print(f"  -> Added Sigmoid")
        print(f"  -> Output: {sigmoid_out.name} {sigmoid_out.shape}")
    
    # Update graph outputs
    graph.outputs = new_outputs
    
    # Cleanup and export
    graph.cleanup().toposort()
    
    print(f"\nModified model has {len(graph.nodes)} nodes")
    print(f"New outputs ({len(new_outputs)}):")
    for out in new_outputs:
        print(f"  - {out.name} {out.shape}")
    
    # Export the modified model
    modified_model = gs.export_onnx(graph)
    
    # Set IR version to 9 for compatibility with older ONNX Runtime
    # This is a simpler approach than version_converter which may fail
    print(f"\nOriginal model IR version: {modified_model.ir_version}")
    if modified_model.ir_version > 9:
        print("Setting IR version to 9 for compatibility...")
        modified_model.ir_version = 9
        # Also update opset imports to match
        for opset in modified_model.opset_import:
            if opset.domain == '' or opset.domain == 'ai.onnx':
                # Set to opset 13 which is compatible with IR 9
                # and has all the ops we need (Conv, Sigmoid, etc.)
                if opset.version > 13:
                    opset.version = 13
        print(f"Updated to IR version: {modified_model.ir_version}")
    
    # Save the modified model
    onnx.save(modified_model, output_model)
    print(f"\nModel saved to {output_model}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Remove post-processing nodes from YOLOv8 ONNX model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (auto-generates output name)
  python utils/removenodes.py yolov8n.onnx
  
  # Specify custom output path
  python utils/removenodes.py yolov8n.onnx modified_yolov8n.onnx
  
  # With full paths
  python utils/removenodes.py models/yolov8n.onnx models/modified_yolov8n.onnx
        """
    )
    parser.add_argument('input_model', type=str, 
                        help='Path to input ONNX model')
    parser.add_argument('output_model', type=str, nargs='?',
                        help='Path to output ONNX model (optional, auto-generated if not provided)')
    
    args = parser.parse_args()
    
    # generate output filename if not provided
    if args.output_model is None:
        base_dir = os.path.dirname(args.input_model)
        base_name = os.path.basename(args.input_model)
        name_without_ext = os.path.splitext(base_name)[0]
        args.output_model = os.path.join(base_dir, f"modified_{base_name}")
        print(f"Output model not specified, using: {args.output_model}\n")
    
    modify_yolov8_head(
        input_model=args.input_model,
        output_model=args.output_model
    )