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

- Run `python export_rknn.py` for generating the rknn model.
- Run model with `python mobilenetv2.py -m ./mobilenetv2-12-b4.rknn -i <video-stream>`
