# Axon-NPU-Guide
This repository contains guide on how to setup toolkits to use NPU present on Axon for running various AI/ML/DL models


To use NPU and other resources on axon at full capacity (or maximum frequencies) run the [max_freq.sh](https://github.com/vicharak-in/Axon-NPU-Guide/blob/main/max_freq.sh) script with sudo permissions as:  
`sudo bash max_freq.sh`

### How to convert your custom CNN model to rknn format to run it on NPU
To run CNN models on NPU using rknn-toolkit-lite2, first we need to have models in model.rknn format.  
This conversion needs to be first done on user's personal computer having amd64/x86 based cpus. Below is walkthrough to setup rknn-toolkit2(that can be used for model conversion), and convert your model to .rknn format.  

#### Follow below steps to install rknn toolkit 2 on your PC (x86/amd64 ) having linux os on it  
- clone the rknn-toolkit2 repo https://github.com/airockchip/rknn-toolkit2.git   
command: `git clone https://github.com/airockchip/rknn-toolkit2.git`
- go to directory rknn-toolkit2/rknn-toolkit2/packages  
command: `cd rknn-toolkit2/rknn-toolkit2/packages`
- install all requirements needed to install rknn-toolkit2 as per your python version from given requirements*.txt files. For eg, if you have python3.10 then use below command   
command: `pip install -r requirements_cp310-2.2.0.txt`
- from given whl files install the whl file as per your environment. For eg, if you have python3.10 then use below command   
command: `pip install rknn_toolkit2-2.2.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl`


Now you have rknn-toolkit2 on your PC. You can explore rknn-toolkit2/rknn-toolkit2/examples from the same repo for use cases. Also there are many more examples on [rknn model zoo repo](https://github.com/airockchip/rknn_model_zoo) for various computer vision related tasks.  

To learn more about rknn-toolkit2, you can follow api reference of rknn-toolkit2 [here](https://github.com/airockchip/rknn-toolkit2/blob/master/doc/03_Rockchip_RKNPU_API_Reference_RKNN_Toolkit2_V2.2.0_EN.pdf)


To convert your particular CNN model to .rknn format first export it to onnx format.  
Then following below code example you can convert your model.onnx to model.rknn that can be used on axon for running the model.  
```python
from rknn.api import RKNN
onnx_model_path = 'path_to_onnx_model/model.onnx'
rknn_model_path = 'path_to_save_rknn_model/model.rknn'

rknn = RKNN()
rknn.config(mean_values=[[mu1, mu2, mu3]], std_values=[[del1, del2, del3]], target_platform='rk3588')
# choose mean and std values as per you model

ret = rknn.load_onnx(model=onnx_model_path)

if ret != 0:
    print('Could not load onnx model')
    exit(ret)

# Set quantization_flag True if you want to quantize your model for faster inference, this will some images as dataset
quantization_flag = True  

# dataset.txt will contain images(in jpg format) name that should be present in the same directory as this code
dataset_path = 'path_to_dataset/dataset.txt'  #this is required only if quantization is performed

ret = rknn.build(do_quantization=quantization_flag, dataset=dataset_path)

if ret != 0:
    print('Build model failed')
    exit(ret)

ret = rknn.export_rknn(rknn_model_path)
    
# then you will have rknn model file for your CNN model
```

Now after you have your model.rknn file, then you need to have rknn-toolkit-lite2 on your Axon SBC.  

#### Follow below steps to setup rknn-toolkit-lite2 on Axon.
#### All of these steps would be performed on top of Axon only

- clone the rknn-toolkit2 repo https://github.com/airockchip/rknn-toolkit2.git   
command: `git clone https://github.com/airockchip/rknn-toolkit2.git`
- go to directory rknn-toolkit2/rknn-toolkit2/packages  
command: `cd rknn-toolkit2/rknn-toolkit-lite2/packages`
- from given whl files install the whl file as per your environment. For eg, if you have python3.10 then use below command   
command: `pip install rknn_toolkit_lite2-2.2.0-cp310-cp310-linux_aarch64.whl`
- download librknn.so file from this [link](https://github.com/airockchip/rknn-toolkit2/blob/master/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so) and move it to location `/usr/lib/`

Now you have rknn-toolkit-lite2 on your PC. You can explore rknn-toolkit2/rknn-toolkit-lite2/examples from the same repo for use cases.

#### How to use rknn-toolkit-lite2 on Axon

You can find api reference to rknn-toolkit-lite [here](file:///home/darklord/Downloads/03_Rockchip_User_Guide_RKNN_Toolkit_Lite_V1.7.5_EN%20(2).pdf)

Basic usage of rknn-toolkit-lite2 api:
```python
# to import the toolkit package 
from rknnlite.api import RKNNLite 

# to instatiate the toolkit
rknn_lite = RKNNLite()

# to load model.rknn file on toolkit
rknn_model_path = 'path_to_rknn_model/model.rknn'
ret = rknn_lite.load_rknn(rknn_model_path)
# this returns 0 on successful loading of model.rknn

if ret != 0:
    print('Load RKNN model failed')
    exit(ret)

# then we need image to be numpy array in 'nhwc' format with channels preferably formatted as RGB or as per your onnx model

image = preprocessed_image_in_required_format # you need to preprocess image as per your cnn model

# to initialize runtime environment and select npu core(s) on which this toolkit will perform inference  
ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
'''
RKNNLite.NPU_CORE_0 : this will direct toolkit to use only core 0
RKNNLite.NPU_CORE_1 : this will direct toolkit to use only core 1
RKNNLite.NPU_CORE_2 : this will direct toolkit to use only core 2
RKNNLite.NPU_CORE_AUTO : this will direct toolkit to automatically detect and use any free npu core
RKNNLite.NPU_CORE_0_1_2 : this will direct toolkit to use all 3 cores on npu, and this is useful only when passing images in batches of size>1 for inference
'''

# on successful initialization of runtime environment it will return 0
if ret != 0:
    print('Init runtime environment failed')
    exit(ret)


# to finally perform inference, this will give output in format according to particular cnn model
outputs = rknn_lite.inference(inputs=[image])

```
