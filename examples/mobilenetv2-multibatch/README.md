## Steps for model conversion and inference

```
.
├── dataset-mobil/          # Dataset folder for model conversion or testing
├── export_rknn.py          # Script to convert ONNX model to RKNN format
├── labels.txt              # Class labels for the MobileNet model
├── mobilenetv2-12-b4.rknn  # Prebuilt RKNN model with batch size 4
├── mobilenetv2-12.onnx     # Original ONNX model
└── mobilenetv2.py          # Inference script supporting multi-batch processing
```

- Make sure you have `rknn-toolkit2` if you want to export model on axon and `rknn-toolkit-lite2` for model inference. Follow readme [guide](https://github.com/vicharak-in/Axon-NPU-Guide?tab=readme-ov-file#axon-npu-guide) for install instructions.
- Install required python libraries `pip install numpy opencv-python`.
- Export RKNN model(optional)
```
python export_rknn.py
```
- Run model with `python mobilenetv2.py -m ./mobilenetv2-12-b4.rknn -i <video-stream>`.
