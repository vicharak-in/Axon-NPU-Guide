#pragma once

#include "rknn_api.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <thread>
#include <queue>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>
#include <atomic>
#include <memory>
#include <chrono>
#include "thread_timer.h"
#include "sync_frames_queue.h"
#include "structs.h"


class CNNModel {
public:
    CNNModel(std::string modelPaths_[], int numModels_);

    CNNModel(std::vector<std::string>& modelPaths_);

    bool loadModels();

    void freeMemories();
    
protected:
    void inferenceWorker(int workerId);

    void streamer();
    
    rknn_output* getRKNNOutput(int modelNum);
    
    void freeRKNNOutput(int modelNum, rknn_output* rknnOutput);

    std::future<void> infer(inferTaskPtr& T);

    const int numInferenceWorker = 3;
    const int numModels;
    std::vector<std::string> modelPaths;
    std::vector<rknn_input_output_num> io_nums;
    std::vector<rknn_tensor_attr*> inputAttrs;
    std::vector<rknn_tensor_attr*> outputAttrs;
    std::vector<rknn_input*> modelInputs[3];
    std::vector<rknn_context> modelContexts[3];
    std::vector<bool> isQuant;
    bool autoRefresh = false;
    std::atomic<bool> stopFlag = false, doStream = false;
    std::priority_queue<inferTaskPtr, std::vector<inferTaskPtr>, inferTaskCompare> inferQ;
    SyncFramesQueue syncedQ;
    ThreadTimer streamT;
    std::mutex inferMtx;
    std::condition_variable inferCv;

};
