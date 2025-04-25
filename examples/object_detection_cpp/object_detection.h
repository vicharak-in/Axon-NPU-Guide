#pragma once

#include "rknn_api.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <memory>
#include <chrono>
#include "thread_timer.h"
#include "sync_frames_queue.h"

#define OBJ_NAME_MAX_SIZE 64
#define OBJ_NUMB_MAX_SIZE 128
#define OBJ_CLASS_NUM 80
#define NMS_THRESH 0.4
#define BOX_THRESH 0.3


typedef struct _inferStruct {
    cv::Mat frame;
    cv::Mat processedFrame;
    int iter;
    rknn_output* outputs;
} inferStruct;

typedef struct _postStruct {
    cv::Mat frame;
    rknn_output* outputs;
    int iter;
} postStruct;

struct compareFinalFrames {
    bool operator()(const std::shared_ptr<finalFrames>& a, const std::shared_ptr<finalFrames>& b) {
        return a->iter > b->iter;
    }
};

class ObjectDetection {
    rknn_context ctx;
    rknn_input_output_num io_num;
    rknn_tensor_attr* input_attrs;
    rknn_tensor_attr* output_attrs;
    bool isQuant;
    std::string modelPath;
    size_t modelHeight = 640;
    size_t modelWidth = 640;
    size_t modelChannel = 3;
    float aspectRatio, xPad, yPad;
    cv::Size modelSize;
    cv::Size rSize;
    bool autoRefresh = false;

    std::queue<std::shared_ptr<inferStruct>> inferQ;
    std::queue<std::shared_ptr<postStruct>> postQ;

    std::priority_queue<std::shared_ptr<finalFrames>, std::vector<std::shared_ptr<finalFrames>>, compareFinalFrames> streamQ;
    SyncFramesQueue syncedQ;

    std::mutex inferMtx, postMtx, streamMtx, inputAttrMtx;
    std::condition_variable readCv, inferCv, postCv, streamCv, inputAttrCv;

    std::atomic<bool> stopFlag = false, stopPost = false, stopStream = false, inputAttrSet = false;
    ThreadTimer inferT, postT, streamT;

    size_t numInferenceWorkers = 3;
    size_t numPostprocessWorkers = 1;

    const std::vector<std::pair<int, int>> skeleton; 

    const std::vector<std::string> labels;
    
    public:
    ObjectDetection(const std::string& modelPath_);

    cv::Mat preprocess(const cv::Mat& image);

    void postprocess(cv::Mat& frame, rknn_output* outputs, int iter=0);

    void inference(cv::Mat& frame, cv::Mat& image);

    void inferenceWorker(int workerId);

    void postprocessWorker(int workerId);

    void streamer();

    void run(const std::string& streamPath);
};
