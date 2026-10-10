/**
 * @file include/datastructures/matrix.h
 * @brief matrix interfaces and implementation.
 * @ingroup containers
 * @details See the module guide for shape, ownership, failure and execution-policy
 * constraints. Source listings include implementation helpers and ordinary
 * explanatory comments; generated source documentation does not execute code.
 */

/** Public contracts for matrix. */

#ifndef CFWX_MATRIX_H
#define CFWX_MATRIX_H
#include "cfwx.h"

/** @brief Record/enumeration used by this module; member contracts describe storage and policy. */
typedef struct
{
    size_t rows, cols; /**< State/dimension/policy value; see the containing type contract. */
    double *data; /**< Borrowed pointer or owned storage as specified by the containing type. */
} cfwx_matrix;
/**
 * @anchor api_include_datastructures_matrix_h_cfwx_matrix_init
 * @brief Construct and initialize matrix state.
 * @ingroup containers
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @param m Caller-owned destination; must be zero-initialized/empty before construction. Successful construction transfers owned storage into it.
 * @param rows Positive row count of the owning matrix.
 * @param cols Positive column/feature count of the owning matrix.
 * @return CFWX_OK on success; otherwise an invalid-argument, shape, allocation
 * or numerical status as checked by the implementation. Do not consume failed outputs.
 */
cfwx_status cfwx_matrix_init(cfwx_matrix *m, size_t rows, size_t cols);
/**
 * @anchor api_include_datastructures_matrix_h_cfwx_matrix_free
 * @brief Release owned matrix storage and reset its owner.
 * @ingroup containers
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @param m Owner whose allocated storage is released and state reset; NULL is accepted where the implementation documents/checks it.
 * @note This operation returns no value; inspect its documented mutation/lifetime effects.
 */
void cfwx_matrix_free(cfwx_matrix *m);
/**
 * @anchor api_include_datastructures_matrix_h_cfwx_matrix_multiply
 * @brief Multiply row-major matrices into caller-owned output storage.
 * @ingroup containers
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @param a First borrowed numerical operand or internal node/state as declared.
 * @param b Second borrowed numerical operand or internal node/state as declared.
 * @param out Caller-supplied output storage; allocate its required shape before calling. Constructor APIs instead receive an owner destination.
 * @return CFWX_OK on success; otherwise an invalid-argument, shape, allocation
 * or numerical status as checked by the implementation. Do not consume failed outputs.
 */
cfwx_status cfwx_matrix_multiply(const cfwx_matrix *a, const cfwx_matrix *b, cfwx_matrix *out);
/**
 * @anchor api_include_datastructures_matrix_h_cfwx_matrix_transpose
 * @brief Transpose a matrix without allocating an output owner.
 * @ingroup containers
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @param a First borrowed numerical operand or internal node/state as declared.
 * @param out Caller-supplied output storage; allocate its required shape before calling. Constructor APIs instead receive an owner destination.
 * @return CFWX_OK on success; otherwise an invalid-argument, shape, allocation
 * or numerical status as checked by the implementation. Do not consume failed outputs.
 */
cfwx_status cfwx_matrix_transpose(const cfwx_matrix *a, cfwx_matrix *out);
#endif
