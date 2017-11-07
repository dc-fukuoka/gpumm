#include <time.h>

double dclock(void) {
    struct timespec tp;
    
    clock_gettime(CLOCK_REALTIME, &tp);
    return (double)(1.0*tp.tv_sec+1.0e-9*tp.tv_nsec);
}
	
