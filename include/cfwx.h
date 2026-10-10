/**
 * @file include/cfwx.h
 * @brief cfwx interfaces and implementation.
 * @ingroup core
 * @details See the module guide for shape, ownership, failure and execution-policy
 * constraints. Source listings include implementation helpers and ordinary
 * explanatory comments; generated source documentation does not execute code.
 */

/** Public contracts for cfwx. */

#ifndef CFWX_H
#define CFWX_H
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
/** Owners must be zero-initialized. No shallow copies. Row-major doubles.
 * Outputs preallocated and disjoint unless exact in-place use is documented.
 * Operations require valid library-created objects. Errors may alter outputs. */
/** @brief Record/enumeration used by this module; member contracts describe storage and policy. */
typedef enum
{
    CFWX_OK = 0, /**< cfwx ok enum choice. */
    CFWX_INVALID, /**< cfwx invalid enum choice. */
    CFWX_SHAPE, /**< cfwx shape enum choice. */
    CFWX_ALLOC, /**< cfwx alloc enum choice. */
    CFWX_NUMERIC /**< cfwx numeric enum choice. */
} cfwx_status;
/**
 * @anchor api_include_cfwx_h_cfwx_count
 * @brief Check a positive element-count product and double-storage overflow.
 * @ingroup core
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @param a First borrowed numerical operand or internal node/state as declared.
 * @param b Second borrowed numerical operand or internal node/state as declared.
 * @param n Positive scalar element count; each associated vector/array must contain at least n entries.
 * @return Computed value or status described by the adjacent contract.
 */
static inline int cfwx_count(size_t a, size_t b, size_t *n)
{
    if (!a || !b || a > SIZE_MAX / b || a * b > SIZE_MAX / sizeof(double))
        return 0;
    *n = a * b;
    return 1;
}
#endif
