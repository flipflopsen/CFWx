/**
 * @file include/datastructures/vector.h
 * @brief vector interfaces and implementation.
 * @ingroup containers
 * @details See the module guide for shape, ownership, failure and execution-policy
 * constraints. Source listings include implementation helpers and ordinary
 * explanatory comments; generated source documentation does not execute code.
 */

/** Public contracts for vector. */

#ifndef CFWX_VECTOR_H
#define CFWX_VECTOR_H
#include "cfwx.h"

/** @brief Record/enumeration used by this module; member contracts describe storage and policy. */
typedef struct
{
    size_t size; /**< State/dimension/policy value; see the containing type contract. */
    double *data; /**< Borrowed pointer or owned storage as specified by the containing type. */
} cfwx_vector;
/**
 * @anchor api_include_datastructures_vector_h_cfwx_vector_init
 * @brief Construct and initialize vector state.
 * @ingroup containers
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @param v Operation-specific owner or scalar as declared; not necessarily an attention operand.
 * @param n Positive scalar element count; each associated vector/array must contain at least n entries.
 * @return CFWX_OK on success; otherwise an invalid-argument, shape, allocation
 * or numerical status as checked by the implementation. Do not consume failed outputs.
 */
cfwx_status cfwx_vector_init(cfwx_vector *v, size_t n);
/**
 * @anchor api_include_datastructures_vector_h_cfwx_vector_free
 * @brief Release owned vector storage and reset its owner.
 * @ingroup containers
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @param v Operation-specific owner or scalar as declared; not necessarily an attention operand.
 * @note This operation returns no value; inspect its documented mutation/lifetime effects.
 */
void cfwx_vector_free(cfwx_vector *v);
/**
 * @anchor api_include_datastructures_vector_h_cfwx_vector_dot
 * @brief Compute a compensated vector dot product.
 * @ingroup containers
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @param a Vector owner/operand with contiguous size-element storage; constructor/free versus input role follows this operation.
 * @param b Vector owner/operand with contiguous size-element storage; constructor/free versus input role follows this operation.
 * @param out Scalar dot-product output pointer; written only after successful finite-result validation.
 * @return CFWX_OK on success; otherwise an invalid-argument, shape, allocation
 * or numerical status as checked by the implementation. Do not consume failed outputs.
 */
cfwx_status cfwx_vector_dot(const cfwx_vector *a, const cfwx_vector *b, double *out);
/**
 * @anchor api_include_datastructures_vector_h_cfwx_vector_add
 * @brief Perform the vector add operation.
 * @ingroup containers
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @param a Vector owner/operand with contiguous size-element storage; constructor/free versus input role follows this operation.
 * @param b Vector owner/operand with contiguous size-element storage; constructor/free versus input role follows this operation.
 * @param out Vector owner/operand with contiguous size-element storage; constructor/free versus input role follows this operation.
 * @return CFWX_OK on success; otherwise an invalid-argument, shape, allocation
 * or numerical status as checked by the implementation. Do not consume failed outputs.
 */
cfwx_status cfwx_vector_add(const cfwx_vector *a, const cfwx_vector *b, cfwx_vector *out);
/**
 * @anchor api_include_datastructures_vector_h_cfwx_vector_scale
 * @brief Perform the vector scale operation.
 * @ingroup containers
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @param v Operation-specific owner or scalar as declared; not necessarily an attention operand.
 * @param scale Finite vector multiplier; numerical overflow can follow ordinary IEEE-754 behavior.
 * @return CFWX_OK on success; otherwise an invalid-argument, shape, allocation
 * or numerical status as checked by the implementation. Do not consume failed outputs.
 */
cfwx_status cfwx_vector_scale(cfwx_vector *v, double scale);
#endif
