#include "host.h"

__host__ void
setEvent (event * evt)
{
#ifdef __linux__
  gettimeofday (evt, NULL);
#else
  QueryPerformanceCounter (evt);
#endif
}

__host__ double
eventDiff (event * first_evt, event * last_evt)
{
#ifdef __linux__
  return ((double) (last_evt->tv_sec + (double) last_evt->tv_usec / 1000000) -
          (double) (first_evt->tv_sec +
                    (double) first_evt->tv_usec / 1000000)) * 1000.0;
#else
  event perf_counter;
  QueryPerformanceFrequency (&perf_counter);
  return (double) (last_evt->QuadPart -
                   first_evt->QuadPart) / (double) perf_counter.QuadPart *
    1000.0;
#endif
}
