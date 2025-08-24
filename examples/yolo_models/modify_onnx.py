#!/usr/bin/env python3
import sys
import onnx
import onnx_graphsurgeon as gs
import numpy as np

def add_post_nodes(input_path, output_path, num_classes=80, verbose=True):
    model = onnx.load(input_path)
    graph = gs.import_onnx(model)
    graph.cleanup().toposort()

    orig_outputs = list(graph.outputs)

    if verbose:
        print(f"args: {input_path=}; {output_path=}; {num_classes=}\n")
        print("[INFO] Original outputs:")
        for o in orig_outputs:
            print(f"  {o.name}: {o.shape}")

    new_outputs = []

    for out in orig_outputs:
        shape = [int(x) if isinstance(x, (int, np.int64)) else x for x in out.shape]
        if verbose:
            print(f"[INFO] Processing output {out.name} with shape {shape}")


        if len(shape) == 4 and shape[1] == num_classes:
            sig_out = gs.Variable(name=out.name + "_sigmoid", dtype=np.float32, shape=shape)
            sig_node = gs.Node(op="Sigmoid", name=out.name + "_sigmoid_node", inputs=[out], outputs=[sig_out])
            graph.nodes.append(sig_node)

            new_outputs.append(sig_out)  

            sum_shape = [shape[0], 1, shape[2], shape[3]]
            reduce_out = gs.Variable(name=out.name + "_reducesum", dtype=np.float32, shape=sum_shape)

            axes_tensor = gs.Constant(
                name=out.name + '_axes',
                values=np.array([1], dtype=np.int64)
            )
            reduce_node = gs.Node(
                op="ReduceSum",
                name=out.name + "_reducesum_node",
                inputs=[sig_out, axes_tensor],
                outputs=[reduce_out],
                attrs={"keepdims": 1},
            )
            graph.nodes.append(reduce_node)

            clip_out = gs.Variable(name=out.name + "_clip", dtype=np.float32, shape=sum_shape)
            clip_node = gs.Node(
                op="Clip",
                name=out.name + "_clip_node",
                inputs=[reduce_out],
                outputs=[clip_out],
                attrs={"min": 0.0, "max": 1.0},
            )
            graph.nodes.append(clip_node)

            new_outputs.append(clip_out)
        else:
            new_outputs.append(out) 

    graph.outputs = new_outputs
    graph.cleanup().toposort()

    if verbose:
        print("[INFO] Final outputs:")
        for o in graph.outputs:
            print(f"  {o.name}: {o.shape}")

    onnx.save(gs.export_onnx(graph), output_path)
    print(f"[OK] Saved new model to {output_path}")


def main():
    num_classes = 80

    if len(sys.argv) < 3:
        print("Usage: python modify_onnx.py <input.onnx> <output.onnx> <num_classes>[optional, default: 80]")
        sys.exit(1)
    
    if len(sys.argv) > 3:
        num_classes = int(sys.argv[3])
    
    add_post_nodes(sys.argv[1], sys.argv[2], num_classes)

if __name__ == "__main__":
    main()

