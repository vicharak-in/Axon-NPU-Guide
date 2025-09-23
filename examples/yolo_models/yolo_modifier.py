import sys
from prune_postprocess_from_yolo import prune_postprocess
from modify_onnx import add_post_nodes
from export_rknn import export_rknn_from_static_onnx
import os


def re_export_onnx(input_onnx_path, output_rknn_path=None, num_classes=80):
    if (output_rknn_path is None):
        output_rknn_path = input_onnx_path[:-5] + '.rknn'
    
    intermediate_onnx_path = input_onnx_path[:-5] + 'intermediate.onnx'
    intermediate_onnx_path2 = input_onnx_path[:-5] + 'intermediate2.onnx'
    
    prune_postprocess(input_onnx_path, intermediate_onnx_path)
    
    add_post_nodes(intermediate_onnx_path, intermediate_onnx_path2, num_classes)

    export_rknn_from_static_onnx(intermediate_onnx_path2, output_rknn_path)

    if os.path.exists(intermediate_onnx_path):
        os.remove(intermediate_onnx_path)
    if os.path.exists(intermediate_onnx_path2):
        os.remove(intermediate_onnx_path2)


def main():
    num_classes = 80
    output_model_name = ""

    if len(sys.argv) < 2:
        print("Usage: python yolo_modifier.py <input.onnx> <output.rknn>[optional, default: <input_model_name>.rknn] <num_classes>[optional, default: 80]")
        sys.exit(1)

    if(len(sys.argv) > 2):
        output_model_name = sys.argv[2]
    else:
        output_model_name = sys.argv[1][:-5] + ".rknn"
    
    if len(sys.argv) > 3:
        num_classes = int(sys.argv[3])
    
    re_export_onnx(sys.argv[1], output_model_name, num_classes)

if __name__ == "__main__":
    main()
