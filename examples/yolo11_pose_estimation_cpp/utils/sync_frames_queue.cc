#include "sync_frames_queue.h"
#include <chrono>

SyncFramesQueue::SyncFramesQueue(int start, int wait_ms)
    : expected_iter(start), wait_time(wait_ms) {}

void SyncFramesQueue::push(const finalFrames& frame) {
    std::lock_guard<std::mutex> lock(mtx);
    if (frame.iter < expected_iter) return;  

    // buffer[frame.iter] = frame;
    framesQ.push(frame);
    cv.notify_all();
}

bool SyncFramesQueue::pop(cv::Mat& outFrame) {
    std::unique_lock<std::mutex> lock(mtx);

    bool gotNext = cv.wait_for(lock, std::chrono::milliseconds(wait_time), [&] {
        return !framesQ.empty() && framesQ.top().iter == expected_iter;
    });

    if (gotNext) {
        outFrame = framesQ.top().frame;
        framesQ.pop();
        ++expected_iter;
        return true;
    } else if(!framesQ.empty()) {
        expected_iter = framesQ.top().iter + 1;
        outFrame = framesQ.top().frame;
        framesQ.pop();  
        return true;
    }
    return false;
}
