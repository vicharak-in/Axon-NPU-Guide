# Running YoloV8 or Yolo11 models on Axon

To run Yolo models optimally there needs some optimizations to be done on the model.  
This section contains program which can do all such optimization easily on any custom trained YoloV8 or Yolo11 object detection models.  
This program requires onnx exported of yolo model and it outputs the modified optimized onnx model.  
Then, that onnx model needs to be converted to rknn model for rk3588 platform and after inferencing with that rknn model, the output should be post-processed as done in this [Yolo11 example](https://github.com/airockchip/rknn_model_zoo/tree/main/examples/yolo11).


The optimizations that are done on the model using this program are:  

- Change output node, remove post-process from the model. (post-process block in model is unfriendly for quantization)
- Remove dfl structure at the end of the model. (which slowdown the inference speed on NPU device)
- Add a score-sum output branch to speedup post-process.

*All the removed operation will be done on CPU as postprocess as done in this [Yolo11 example](https://github.com/airockchip/rknn_model_zoo/tree/main/examples/yolo11).


To run the program to modify the model, run the following command after installing all the necessary libraries:  
`python3 yolo_modifier.py <input_onnx_path>/<model_name>.onnx <output_onnx_path><model_name>.onnx <num_classes>`