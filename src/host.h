#ifndef HOST_H
#define HOST_H
#include "common.h"

__host__ void setEvent (event * evt);

__host__ double eventDiff (event * first_event, event * second_event);

#endif
