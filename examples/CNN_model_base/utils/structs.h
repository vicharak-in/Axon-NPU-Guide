#pragma once
#include <opencv2/opencv.hpp>
#include "rknn_api.h"
#include <memory>
#include <future>

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

typedef struct _finalFrames {
    cv::Mat frame;
    int iter;
} finalFrames;

struct compareFinalFrames {
    bool operator()(const finalFrames& a, const finalFrames& b) {
        return a.iter > b.iter;
    }
};

struct compareFinalFramesPtr {
    bool operator()(const std::shared_ptr<finalFrames>& a, const std::shared_ptr<finalFrames>& b) {
        return a->iter > b->iter;
    }
};

typedef struct _inferTask {
    int id;
    int priority=-2;
    int iter=0;
    void** inp;
    void* out;
    std::promise<void> done; 
    bool operator<(const _inferTask& other) const {
        return priority < other.priority;  
    }
} inferTask;

typedef std::shared_ptr<inferTask> inferTaskPtr;

struct inferTaskCompare {
    bool operator()(const inferTaskPtr& a, const inferTaskPtr& b) const {
        return (*a) < (*b);
    }
};

