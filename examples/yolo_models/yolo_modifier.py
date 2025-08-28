import sys
import prune_postprocess_from_yolo
import modify_onnx

def re_export_onnx(input_onnx_path, num_classes=80, output_onnx_path=None):
    if (output_onnx_path is None):
        output_onnx_path = input_onnx_path[:-5] + 'modified.onnx'
    
    intermediate_onnx_path = input_onnx_path[:-5] + 'intermediate.onnx'
    
    prune_postprocess(input_onnx_path, intermediate_onnx_path)
    
    add_post_nodes(intermediate_onnx_path, output_onnx_path, num_classes)


def main():
    num_classes = 80

    if len(sys.argv) < 3:
        print("Usage: python yolo_modifier.py <input.onnx> <output.onnx> <num_classes>[optional, default: 80]")
        sys.exit(1)
    
    if len(sys.argv) > 3:
        num_classes = int(sys.argv[3])
    
    re_export_onnx(sys.argv[1], sys.argv[2], num_classes)

if __name__ == "__main__":
    main()
