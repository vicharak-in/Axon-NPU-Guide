# Objection detection using Yolo in c++

This repo contains optimized code to do object detection using any YoloV8 or Yolo11 rknn models (quantized or fp16 precision) on input video stream

To run please follow below steps:
- First compile the code by running `./compile.sh`
- Then a binary executable file named `./build/object_detection` will be created.
- To execute that file run `./build/object_detection  <rknn_model_path>  <input_video_stream_path_readable_by_cv::VideoCapture>`

This code gives 70+ FPS with yolo11s(int8 quantized) and around 30 FPS with yolo11s(fp16 model) on **Axon**
