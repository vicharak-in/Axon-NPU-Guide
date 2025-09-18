from rknn.api import RKNN
import sys


def export_rknn_from_static_onnx(input_onnx_path, output_rknn_path=None):
    if (output_rknn_path is None):
        output_rknn_path = input_onnx_path[:-5] + '.rknn'

    model = RKNN()
    model.config(target_platform="rk3588", dynamic_input=None, remove_reshape=True)

    ret = model.load_onnx(model=input_onnx_path)
    if ret != 0:
        print('Could not load onnx model')
        exit(ret)

    ret = model.build(do_quantization=False)
    if ret != 0:
        print('Build model failed')
        exit(ret)
    else:
        print('Build model done')

    ret = model.export_rknn(output_rknn_path)
    if ret != 0:
        print('Export model failed')
        exit(ret)
   

def main():
    if len(sys.argv) < 2:
        print("Usage: python export_rknn.py <input_model_name.onnx> <output_model_name.rknn>[optional, default: <input_model_name>.rknn]")
        sys.exit(1)

    output_model_name = ""

    if(len(sys.argv) > 2):
        output_model_name = sys.argv[2]
    else:
        output_model_name = sys.argv[1][:-5] + ".rknn"
    
    export_rknn_from_static_onnx(sys.argv[1], output_model_name)

if __name__ == "__main__":
    main()
