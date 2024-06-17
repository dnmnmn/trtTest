#ifndef __TIMER_H__
#define __TIMER_H__

#include <chrono>

class Timer
{
public:
    double elapsed_time;
    std::chrono::high_resolution_clock::time_point t1;
    std::chrono::high_resolution_clock::time_point t2;
    std::chrono::duration<double> time_span;
    void start();
    double end();
    inline int64_t now()
    {
        return (std::chrono::system_clock::now().time_since_epoch()).count();
    }
};

#endif