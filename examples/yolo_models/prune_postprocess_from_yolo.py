import sys
import onnx
import onnx_graphsurgeon as gs

def prune_postprocess(input_path, output_path, verbose=True):
    model = onnx.load(input_path)
    graph = gs.import_onnx(model)
    graph.cleanup().toposort()

    last_split = None
    for node in reversed(graph.nodes):
        if node.op == "Split":
            last_split = node
            break
    if last_split is None:
        raise RuntimeError("No Split node found in the graph.")
    if verbose:
        print(f"[INFO] Last Split node: {last_split.name}")

    concat_final = None
    for inp in last_split.inputs:
        if inp.inputs and inp.inputs[0].op == "Concat":
            concat_final = inp.inputs[0]
            break
    if concat_final is None:
        raise RuntimeError("No Concat feeding the Split was found.")
    if verbose:
        print(f"[INFO] Found final Concat before Split: {concat_final.name}")

    reshape_nodes = []
    for t in concat_final.inputs:
        if t.inputs and t.inputs[0].op == "Reshape":
            reshape_nodes.append(t.inputs[0])
    if len(reshape_nodes) != 3:
        raise RuntimeError(f"Expected 3 Reshapes into final Concat, found {len(reshape_nodes)}")
    if verbose:
        print(f"[INFO] Found 3 Reshape nodes: {[r.name for r in reshape_nodes]}")

    new_outputs = []
    for r in reshape_nodes:
        data_in = r.inputs[0]  
        if not data_in.inputs:
            raise RuntimeError(f"Reshape {r.name} input has no producer.")
        prod = data_in.inputs[0]
        if prod.op != "Concat":
            raise RuntimeError(f"Reshape {r.name} is not fed by a Concat, but {prod.op}")

        for inp in prod.inputs:
            new_outputs.append(inp)

    if verbose:
        print(f"[INFO] New outputs are inputs to the 3 branch concats: {[o.name for o in new_outputs]}")

    graph.outputs = new_outputs

    graph.cleanup().toposort()
    onnx.save(gs.export_onnx(graph), output_path)
    if verbose:
        print(f"[OK] Saved pruned model to {output_path}")

def main():
    if len(sys.argv) < 3:
        print("Usage: python prune_postprocess_from_yolo.py <input.onnx> <output.onnx>")
        sys.exit(1)
    prune_above_branch_concats(sys.argv[1], sys.argv[2])

if __name__ == "__main__":
    main()

