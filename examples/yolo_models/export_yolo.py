from ultralytics import YOLO
import sys

model_path = sys.argv[1] # Custom YoloV8 or Yolo11 model path
model = YOLO(model_path)

model.export(format="onnx", dynamic=False, simplify=True, opset=11, imgsz=640)
