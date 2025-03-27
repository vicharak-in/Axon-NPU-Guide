# Axon-NPU-Guide
This repository contains guide on how to setup toolkits to use NPU present on Axon for running various AI/ML/DL models


To use NPU and other resources on axon at full capacity (or maximum frequencies) run the [max_freq.sh](https://github.com/vicharak-in/Axon-NPU-Guide/blob/main/max_freq.sh) script with sudo permissions as:  
```bash
sudo bash max_freq.sh
```

### LLM Benchmarks on Axon

|Model                          | quantization type | tokens per second |
| :---:                         | :---:             | :---:             |
| Llama3.2-1B                   | w8a8              | 21.23             |
| Llama3.2-3B                   | w8a8              | 8.51              |
| Qwen2-VL-2B                   | w8a8              | 16.38             |
| DeepSeek-R1-Distill-Qwen-1.5B | w8a8              | 16.67             |
| DeepSeek-R1-Distill-Qwen-1.5B | None              | 9.19              |
| TinyLlama-1.1B                | w8a8              | 24.47             |
| MiniCPM3-4B                   | w8a8              | 5.78              |


### How to convert your custom CNN model to rknn format to run it on NPU
To run CNN models on NPU using rknn-toolkit-lite2, first we need to have models in model.rknn format.  
This conversion can be done on user's personal computer having amd64/x86 based cpus, or on Axon also. But it is recommended to be done on PC for faster quantization of models.  

Below is walkthrough to setup rknn-toolkit2(that can be used for model conversion), and convert your model to .rknn format. This toolkit is either to be installed on your PC(recommended) or Axon  

#### Follow below steps to install rknn toolkit 2 on your PC (x86/amd64) having linux os on it  
- clone the rknn-toolkit2 repo https://github.com/airockchip/rknn-toolkit2.git   
command: `git clone https://github.com/airockchip/rknn-toolkit2.git`
- go to directory rknn-toolkit2/rknn-toolkit2/packages/x86_64  
command: `cd rknn-toolkit2/rknn-toolkit2/packages/x86_64`
- install all requirements needed to install rknn-toolkit2 as per your python version from given requirements*.txt files. For eg, if you have python3.10 then use below command   
command: `pip install -r requirements_cp310-2.2.0.txt`
- from given whl files install the whl file as per your environment. For eg, if you have python3.10 then use below command   
command: `pip install rknn_toolkit2-2.2.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl`


#### Follow below steps to install rknn toolkit 2 on your Axon (or any other arm cpu based PC) having linux os on it   
- (optional) you may install all below libraries on [python virtual environment](https://docs.python.org/3/library/venv.html) as mostly recommended.   
   Commands to create and activate/deactivate:  
    ```bash
    python3 -m venv env_rknn
    source env_rknn/bin/activate

    # it can be later deactivated after getting task done by entering below command
    deactivate
    ```
- setuptools maybe required to install few packages of some version. Install it using following  
command: `pip install setuptools`
- clone the rknn-toolkit2 repo https://github.com/airockchip/rknn-toolkit2.git   
command: `git clone https://github.com/airockchip/rknn-toolkit2.git`
- go to directory rknn-toolkit2/rknn-toolkit2/packages/arm64  
command: `cd rknn-toolkit2/rknn-toolkit2/packages/arm64`
- install all requirements needed to install rknn-toolkit2 as per your python version from given arm64_requirements*.txt files. For eg, if you have python3.10 then use below command   
command: `pip install -r arm64_requirements_cp310-2.2.0.txt`
- from given whl files install the whl file as per your environment. For eg, if you have python3.10 then use below command   
command: `pip install rknn_toolkit2-2.3.0-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl`

To learn more about rknn-toolkit2, you can follow api reference of rknn-toolkit2 [here](https://github.com/airockchip/rknn-toolkit2/blob/master/doc/03_Rockchip_RKNPU_API_Reference_RKNN_Toolkit2_V2.3.0_EN.pdf)

Now you have rknn-toolkit2 on your PC. You can explore rknn-toolkit2/rknn-toolkit2/examples from the same repo for use cases. Also there are many more examples on [rknn model zoo repo](https://github.com/airockchip/rknn_model_zoo) for various computer vision related tasks.   

To convert your particular CNN model to .rknn format, it is easier to first export it to onnx format.  
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


-------------------------------------------------------------------


### How to convert LLM models from huggingface or gguf file to rkllm format and run on Axon

#### Step 1: Model Conversion on PC having x86/amd64 linux on it.

Follow below steps to first install the required libraries to convert the model inn rkllm format:
- clone the rknn-toolkit2 repo https://github.com/airockchip/rknn-llm.git   
command: `git clone https://github.com/airockchip/rknn-llm.git`
- go to directory rknn-llm/rkllm-toolkit/packages  
command: `cd rknn-llm/rkllm-toolkit/packages`
- from given whl files install the whl file as per your environment (you may also create python virtual environment and install libraries there). For eg, if you have python3.10 then use below command   
command: `pip install rkllm_toolkit-1.1.4-cp310-cp310-linux_x86_64.whl`


Follow below steps to convert the LLM models to run on Axon:

```python
# import the necessary libraries 
from rkllm.api import RKLLM

# initialise rkllm toolkit
rkllm = RKLLM()

# load LLM model as per your model files
ret = rkllm.load_huggingface(model='./path_to_model_directory', model_lora=None, device='cpu')

# if you have gguf file then load as below
# ret = rkllm.load_gguf(model = modelpath) 

if ret != 0:
    print('Load model failed!')
    exit(ret)


# if quantization will be performed then dataset is needed 
dataset_path = 'path_to_dataset/dataset.json'

# build model with your choice of parameters
# make sure in target platform you choose 'rk3588'

ret = rkllm.build(
    do_quantization=True,
    quantized_dtype='w8a8',
    optimization_level=1,
    target_platform='rk3588',
    quantized_algorithm='normal',
    target_platform='rk3588', 
    num_npu_core=3, 
    extra_qparams=None, 
    dataset=dataset_path # required if performing quantization
)

if ret != 0:
    print('Build model failed!')
    exit(ret)

# Export rkllm model
ret = llm.export_rkllm("model_path/model_name.rkllm")

if ret != 0:
    print('Export model failed!')
    exit(ret)
```


#### Step 2: Running model on Axon SBC using RKLLM C API

- First make sure kernel version is >= 5.10.228, check it using `uname -r`. 
- If kernel version is older then update it as:
    - Update the package list as: `sudo apt update`
    - Install the updated kernel as: `sudo apt install linux-image-5.10.228-axon`
    - Remove the older kernel as: `sudo apt remove linux-image-5.10.<version-num>-axon`
    - Reboot the Axon as: `sudo reboot`
- Download "rkllm.h" header file from [here](https://github.com/airockchip/rknn-llm/blob/main/rkllm-runtime/Linux/librkllm_api/include/rkllm.h)
- You can copy this header file to /usr/include or define source path to header file while compiling c++ source code
- Download librkllm.so shared object library from [here](https://github.com/airockchip/rknn-llm/blob/main/rkllm-runtime/Linux/librkllm_api/aarch64/librkllmrt.so)
- You can copy this shared object to /usr/lib and link it while compiling source code
- Then you need to write c++ code to use rkllm library to run LLM on Axon. 
- Example code is given below, compile it as 
    ```bash 
    g++ llm.cpp -lrkllmrt -o llm
    ```
- Or if rkllm.h is not at /usr/include/ then manually pass directory location while compiling as: 
    ```bash
    g++ llm.cpp -lrkllmrt -I<path_to_directory_containing_rkllm.h> -o llm
    ```
- Then you can run the compiled example code as:
    ```bash 
    ./llm <rkllm_model_path> <max_new_tokens> <max_context_len>
    # <max_new_tokens> and <max_context_len> should be replaced by integer value
    ```

#### Example llm code:
```cpp
#include <iostream>
#include <fstream>
#include <string.h>
#include <string>
#include "rkllm.h"


// callback function which will be called by LLMs every time they generate a token.
// generated text is recorded in text attribute RKLLMResult struct
// when last callback is called in a conversation LLMCallState parameter is equal to RKLLM_RUN_FINISH
// LLMCallState is originally defined as: 
// typedef enum {
//     RKLLM_RUN_NORMAL  = 0, /**< The LLM call is in a normal running state. */
//     RKLLM_RUN_WAITING = 1, /**< The LLM call is waiting for complete UTF-8 encoded character. */
//     RKLLM_RUN_FINISH  = 2, /**< The LLM call has finished execution. */
//     RKLLM_RUN_ERROR   = 3, /**< An error occurred during the LLM call. */
//     RKLLM_RUN_GET_LAST_HIDDEN_LAYER = 4 /**< Retrieve the last hidden layer during inference. */
// } LLMCallState;
void callback(RKLLMResult *result, void *userdata, LLMCallState state)
{
    if (state == RKLLM_RUN_FINISH)
    {
        std::cout << std::endl;
    } else if (state == RKLLM_RUN_ERROR) {
        std::cout << "Runtime error" << std::endl;
    } else if (state == RKLLM_RUN_GET_LAST_HIDDEN_LAYER) {
        if (result->last_hidden_layer.embd_size != 0 && result->last_hidden_layer.num_tokens != 0) {
            int data_size = result->last_hidden_layer.embd_size * result->last_hidden_layer.num_tokens * sizeof(float);
            std::cout << "data_size: " << data_size << std::endl;
            std::ofstream outFile("last_hidden_layer.bin", std::ios::binary);
            if (outFile.is_open()) {
                outFile.write(reinterpret_cast<const char*>(result->last_hidden_layer.hidden_states), data_size);
                outFile.close();
                std::cout << "Data saved to output.bin successfully!" << std::endl;
            } else {
                std::cerr << "Failed to open the file for writing!" << std::endl;
            }
        }
    } else if (state == RKLLM_RUN_NORMAL) {
        std::cout << result->text;
    }
}


int main(int argc, char **argv)
{
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " model_path max_new_tokens max_context_len\n";
        return 1;
    }

    // initialize llmHandle to store pointer to loaded llm model
    LLMHandle llmHandle = nullptr;

    // initializing RKLLMParam struct for setting parameters required for LLMs
    // RKLLMParam struct is originally defined as:
    // typedef struct {
    //     const char* model_path;         /**< Path to the model file. */
    //     int32_t max_context_len;        /**< Maximum number of tokens in the context window. */
    //     int32_t max_new_tokens;         /**< Maximum number of new tokens to generate. */
    //     int32_t top_k;                  /**< Top-K sampling parameter for token generation. */
    //     float top_p;                    /**< Top-P (nucleus) sampling parameter. */
    //     float temperature;              /**< Sampling temperature, affecting the randomness of token selection. */
    //     float repeat_penalty;           /**< Penalty for repeating tokens in generation. */
    //     float frequency_penalty;        /**< Penalizes frequent tokens during generation. */
    //     float presence_penalty;         /**< Penalizes tokens based on their presence in the input. */
    //     int32_t mirostat;               /**< Mirostat sampling strategy flag (0 to disable). */
    //     float mirostat_tau;             /**< Tau parameter for Mirostat sampling. */
    //     float mirostat_eta;             /**< Eta parameter for Mirostat sampling. */
    //     bool skip_special_token;        /**< Whether to skip special tokens during generation. */
    //     bool is_async;                  /**< Whether to run inference asynchronously. */
    //     const char* img_start;          /**< Starting position of an image in multimodal input. */
    //     const char* img_end;            /**< Ending position of an image in multimodal input. */
    //     const char* img_content;        /**< Pointer to the image content. */
    //     RKLLMExtendParam extend_param; /**< Extend parameters. */
    // } RKLLMParam;
    RKLLMParam param = rkllm_createDefaultParam();
    param.model_path = argv[1];

    param.top_k = 1;
    param.top_p = 0.95;
    param.temperature = 0.8;
    param.repeat_penalty = 1.1;
    param.frequency_penalty = 0.0;
    param.presence_penalty = 0.0;

    param.max_new_tokens = std::atoi(argv[2]);
    param.max_context_len = std::atoi(argv[3]);
    param.skip_special_token = true;
    param.extend_param.base_domain_id = 0;

    // loading LLM model onto memory for conversations
    int ret = rkllm_init(&llmHandle, &param, callback);
    if (ret == 0){
        std::cout << "rkllm init success" << std::endl ;
    } else {
        std::cerr << "rkllm init failed" << std::endl;
    }

    std::string text;

    // struct to pass input to LLM
    RKLLMInput rkllm_input;

    // struct define kind of interaction with LLM
    RKLLMInferParam rkllm_infer_params;
    memset(&rkllm_infer_params, 0, sizeof(RKLLMInferParam)); 

    rkllm_infer_params.mode = RKLLM_INFER_GENERATE;

    while (true)
    {
        std::cout << std::endl;
        std::cout << "User: ";
        std::getline(std::cin, text);
        if (text == "exit")
        {
            break;
        }
        std::cout << std::endl;
        // input_type attribute is of enum type and offered input types are:
        // RKLLM_INPUT_PROMPT      = 0,  Input is a text prompt. 
        // RKLLM_INPUT_TOKEN       = 1,  Input is a sequence of tokens. 
        // RKLLM_INPUT_EMBED       = 2,  Input is an embedding vector. 
        // RKLLM_INPUT_MULTIMODAL  = 3,  Input is multimodal (e.g., text and image). 
        rkllm_input.input_type = RKLLM_INPUT_PROMPT;

        rkllm_input.prompt_input = (char *)text.c_str();
        printf("AI: ");

        // Running LLM model
        rkllm_run(llmHandle, &rkllm_input, &rkllm_infer_params, NULL);
    }

    // releasing llmHandle after work is done.
    rkllm_destroy(llmHandle);

    return 0;
}

```

