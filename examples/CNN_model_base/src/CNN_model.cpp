#include "CNN_model.h"
#include <stdlib.h>
#include <math.h>

static int readDataFromFile(const char *path, char **out_data)
{
    FILE *fp = fopen(path, "rb");
    if(fp == NULL) {
        printf("fopen %s fail!\n", path);
        return -1;
    }
    fseek(fp, 0, SEEK_END);
    int file_size = ftell(fp);
    char *data = (char *)malloc(file_size+1);
    data[file_size] = 0;
    fseek(fp, 0, SEEK_SET);
    if(file_size != fread(data, 1, file_size, fp)) {
        printf("fread %s fail!\n", path);
        free(data);
        fclose(fp);
        return -1;
    }
    if(fp) {
        fclose(fp);
    }
    *out_data = data;
    return file_size;
}


CNNModel::CNNModel(std::vector<std::string>& modelPaths_):numModels(modelPaths_.size()),
                        modelPaths(modelPaths_.begin(), modelPaths_.end()) {
                            ;
}

CNNModel::CNNModel(std::string modelPaths_[], int numModels_):numModels(numModels_),
                        modelPaths(modelPaths_, modelPaths_+numModels_) {
                            ;
}

rknn_output* CNNModel::getRKNNOutput(int modelNum) {
    size_t n_output = io_nums[modelNum].n_output;
    rknn_output* outputs = (rknn_output*)malloc(n_output * sizeof(rknn_output));
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < n_output; i++) {
        outputs[i].index = i;
        outputs[i].want_float = !isQuant[modelNum];
        outputs[i].is_prealloc = true;
        outputs[i].buf = malloc(outputAttrs[modelNum][i].size << (isQuant[modelNum]?0:1));
        outputs[i].size = outputAttrs[modelNum][i].size << (isQuant[modelNum]?0:1);
    }  
    
    return outputs;
}

void CNNModel::freeRKNNOutput(int modelNum, rknn_output* rknnOutput) {
    for(int i=0; i<io_nums[modelNum].n_output; ++i) 
        free(rknnOutput[i].buf);
    free(rknnOutput);
    rknnOutput = nullptr;
    return;
}

bool CNNModel::loadModels() {
    int ret, modelLen = 0;
    char* model;
    for(int i=0; i<numModels; ++i) {
        modelLen = readDataFromFile(modelPaths[i].c_str(), &model);
        if (model == NULL) {
            std::cerr << "load_model fail!" << std::endl;
            return false;
        }
        for(int j=0; j<numInferenceWorker; ++j) {
            rknn_context ctxI;
            ret = rknn_init(&ctxI, model, modelLen, 0, NULL);
            if (ret < 0) {
                std::cerr << "rknn_init fail! ret=" << ret << std::endl;
                return false;
            }
            if(j == 0) {
                rknn_input_output_num io_num;
                ret = rknn_query(ctxI, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
                if (ret != RKNN_SUCC) {
                    std::cerr << "rknn io num query fail! ret=" << ret << std::endl;
                    return false;
                }

                io_nums.push_back(io_num);

                rknn_tensor_attr* inputAttr = (rknn_tensor_attr*)calloc(io_num.n_input, sizeof(rknn_tensor_attr));
                for (int k = 0; k < io_num.n_input; ++k) {
                    inputAttr[k].index = k;
                    ret = rknn_query(ctxI, RKNN_QUERY_INPUT_ATTR, &(inputAttr[k]), sizeof(rknn_tensor_attr));
                    if (ret != RKNN_SUCC) {
                        std::cerr << "rknn input attr query fail! ret=" << ret << std::endl;
                        return false;
                    }
                }
                inputAttrs.push_back(inputAttr);

                rknn_tensor_attr* outputAttr = (rknn_tensor_attr*)calloc(io_num.n_output, sizeof(rknn_tensor_attr));
                for (int k = 0; k < io_num.n_output; ++k) {
                    outputAttr[k].index = k;
                    ret = rknn_query(ctxI, RKNN_QUERY_OUTPUT_ATTR, &(outputAttr[k]), sizeof(rknn_tensor_attr));
                    if (ret != RKNN_SUCC) {
                        std::cerr << "rknn output attr query fail! ret=" << ret << std::endl;
                        return false;
                    }
                }
                outputAttrs.push_back(outputAttr);

                if (outputAttr[0].qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC && outputAttr[0].type == RKNN_TENSOR_INT8) {
                    isQuant.push_back(true);
                }
                else {
                    isQuant.push_back(false);
                }
            }
            modelContexts[j].push_back(ctxI);
            rknn_input* _inputs = (rknn_input*)calloc(io_nums[i].n_input, sizeof(rknn_input));
            for(int k=0; k<io_nums.back().n_input; ++k) {
                _inputs[k].index = k;
                _inputs[k].type = inputAttrs.back()[k].type;
                _inputs[k].fmt = inputAttrs.back()[k].fmt;
                _inputs[k].size = inputAttrs.back()[k].size * (_inputs[k].type == RKNN_TENSOR_INT8 ? 1:2);
            }
            modelInputs[j].push_back(_inputs);
        }
        free(model);
    }
    return true;
}

void CNNModel::inferenceWorker(int workerId) {
    std::unique_lock<std::mutex> lck(inferMtx, std::defer_lock);
    int modelId, ret;
    while(true) {
        lck.lock();
        inferCv.wait(lck, [this](){return !inferQ.empty() || stopFlag;});

        if(inferQ.empty()) {
            lck.unlock();
            break;
        }

        auto item = inferQ.top();
        inferQ.pop();
        lck.unlock();

        rknn_output* outputs = (rknn_output*)item->out;
        modelId = item->id;
        for(int i=0; i<io_nums[modelId].n_input; ++i) {
            modelInputs[workerId][modelId][i].buf = item->inp[i]; 
        }

        ret = rknn_inputs_set(modelContexts[workerId][modelId], 
                io_nums[modelId].n_input, modelInputs[workerId][modelId]);

        if (ret < 0) {
            std::cerr << "rknn_input_set fail! ret=" << ret << std::endl;
            return;
        }

        ret = rknn_run(modelContexts[workerId][modelId], nullptr);

        if (ret < 0){
            std::cerr << "rknn_run fail! ret=" << ret << std::endl;
            return;
        }

        ret = rknn_outputs_get(modelContexts[workerId][modelId], io_nums[modelId].n_output, outputs, NULL);
        if (ret < 0) {
            std::cerr << "rknn_outputs_get fail! ret=" << ret << std::endl;
            return;
        }
        item->done.set_value();
        rknn_outputs_release(modelContexts[workerId][modelId], io_nums[modelId].n_output, outputs);
    }
}

void CNNModel::freeMemories() {
    for(int i=0; i<numInferenceWorker; ++i) {
        for(int j=0; j<numModels; ++j) {
            if(modelContexts[i][j] != 0) {
                rknn_destroy(modelContexts[i][j]);
                modelContexts[i][j] = 0;
            }
        }
    }
}

std::future<void> CNNModel::infer(inferTaskPtr& T) {
    std::future<void> fut = T->done.get_future();
    
    {
        std::lock_guard<std::mutex> lock(inferMtx);
        inferQ.push(T);
    }
    inferCv.notify_one();
    return fut;
}

void CNNModel::streamer() {
    size_t frameNum = 0, totFrames = 0;
    std::chrono::steady_clock::time_point startTime, endTime;
    std::chrono::duration<double, std::milli> elapsed;
    startTime = std::chrono::steady_clock::now();
    streamT.start();
    while(doStream) {
        cv::Mat frame;
        if(!syncedQ.pop(frame)) {
            continue;
        }
	    ++frameNum;
        ++totFrames;
        cv::imshow("processed stream", frame);
        cv::waitKey(1);
        frame.release();
        
        if(frameNum == 20) {
            endTime = std::chrono::steady_clock::now();
            elapsed = endTime-startTime;
            startTime = endTime;
            frameNum = 0;
            std::cout << "\nFPS: " << 20000.0/elapsed.count() << std::endl;
        }
    }
    streamT.stop();
    std::cout << "Avg FPS over " << totFrames << ": " << totFrames*1000.0/streamT.getTotalMilliseconds() << std::endl;
}
