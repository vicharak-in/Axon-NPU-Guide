#pragma once
#include <chrono>
#include <vector>

class ThreadTimer {
public:
    ThreadTimer();
    explicit ThreadTimer(int numThreads);

    void start(int id = 0);
    void stop(int id = 0);

    double getTotalMilliseconds() const;
    double getAverageMilliseconds() const;
    int getCallCount() const;

    void reset();

private:
    struct TimeData {
        std::chrono::steady_clock::time_point startTime;
        double elapsed = 0.0;
        int calls = 0;
    };

    std::vector<TimeData> times;
};
