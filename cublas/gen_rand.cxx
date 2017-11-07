#include <stdlib.h>
#include <assert.h>
#include "gen_rand.h"

void gen_rand(MKL_INT seed, double min, double max, size_t size, double *arr) {
    VSLStreamStatePtr stream;
    MKL_INT brng   = VSL_BRNG_MT19937;
    MKL_INT method = VSL_RNG_METHOD_UNIFORM_STD_ACCURATE;
    MKL_INT ret;
    
    ret = vslNewStream(&stream, brng, seed);                 assert(ret==VSL_ERROR_OK);
    ret = vdRngUniform(method, stream, size, arr, min, max); assert(ret==VSL_ERROR_OK);
    ret = vslDeleteStream(&stream);                          assert(ret==VSL_ERROR_OK);
}

