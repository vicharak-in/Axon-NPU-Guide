#include "thread_timer.h"

ThreadTimer::ThreadTimer() {
    times.resize(1);
}

ThreadTimer::ThreadTimer(int numThreads) {
    times.resize(numThreads);
}

void ThreadTimer::start(int id) {
    times[id].startTime = std::chrono::steady_clock::now();
}

void ThreadTimer::stop(int id) {
    auto end = std::chrono::steady_clock::now();
    auto& t = times[id];
    std::chrono::duration<double, std::milli> elapsed = end - t.startTime;
    t.elapsed += elapsed.count();
    t.calls += 1;
}

double ThreadTimer::getTotalMilliseconds() const {
    double total = 0.0;
    for (const auto& t : times) total += t.elapsed;
    return total;
}

double ThreadTimer::getAverageMilliseconds() const {
    int totalCalls = getCallCount();
    return (totalCalls == 0) ? 0.0 : getTotalMilliseconds() / totalCalls;
}

int ThreadTimer::getCallCount() const {
    int total = 0;
    for (const auto& t : times) total += t.calls;
    return total;
}

void ThreadTimer::reset() {
    for (auto& t : times) {
        t.elapsed = 0.0;
        t.calls = 0;
    }
}
