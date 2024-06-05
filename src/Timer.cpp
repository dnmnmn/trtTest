#include "Timer.h"

void Timer::start()
{
    t1 = std::chrono::high_resolution_clock::now();
}

double Timer::end()
{
    t2 = std::chrono::high_resolution_clock::now();
    time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    if (time_span.count() >= 0)
    {
        elapsed_time = time_span.count() * 1000;
    }
    else
    {
        elapsed_time = 0;
    }

    return elapsed_time;
}