from rknn.api import RKNN
onnx_model_path = './mobilenetv2-12.onnx'
rknn_model_path = './mobilenetv2-12-b4.rknn'

rknn = RKNN()
rknn.config(mean_values=[[103.94, 116.78, 123.68]], std_values=[[58.82, 58.82, 58.82]], target_platform='rk3588')

ret = rknn.load_onnx(model=onnx_model_path)

if ret != 0:
    print('Could not load onnx model')
    exit(ret)


dataset_path = './dataset-mobil/dataset.txt'

ret = rknn.build(do_quantization=True,
                  dataset=dataset_path,
                  rknn_batch_size=4)

if ret != 0:
    print('Build model failed')
    exit(ret)

ret = rknn.export_rknn(rknn_model_path)
