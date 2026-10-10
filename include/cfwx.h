/* Public contracts for cfwx. */

#ifndef CFWX_H
#define CFWX_H
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
/* Owners must be zero-initialized. No shallow copies. Row-major doubles.
 * Outputs preallocated and disjoint unless exact in-place use is documented.
 * Operations require valid library-created objects. Errors may alter outputs. */
typedef enum
{
    CFWX_OK = 0,
    CFWX_INVALID,
    CFWX_SHAPE,
    CFWX_ALLOC,
    CFWX_NUMERIC
} cfwx_status;
static inline int cfwx_count(size_t a, size_t b, size_t *n)
{
    if (!a || !b || a > SIZE_MAX / b || a * b > SIZE_MAX / sizeof(double))
        return 0;
    *n = a * b;
    return 1;
}
#endif
