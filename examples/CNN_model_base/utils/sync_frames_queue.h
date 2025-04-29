#pragma once

#include <opencv2/opencv.hpp>
#include <queue>
#include <vector>
#include <mutex>
#include <condition_variable>
#include "structs.h"

class SyncFramesQueue {
public:
    SyncFramesQueue(int start = 0, int wait_ms = 50);

    void push(const finalFrames& frame);
    bool pop(cv::Mat& outFrame);

private:
    std::priority_queue<finalFrames, std::vector<finalFrames>, compareFinalFrames> framesQ;
    int expected_iter;
    int wait_time;

    std::mutex mtx;
    std::condition_variable cv;
};
