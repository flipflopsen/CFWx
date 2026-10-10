/**
 * @file include/datastructures/tensor.h
 * @brief tensor interfaces and implementation.
 * @ingroup containers
 * @details See the module guide for shape, ownership, failure and execution-policy
 * constraints. Source listings include implementation helpers and ordinary
 * explanatory comments; generated source documentation does not execute code.
 */

/** Public contracts for tensor. */

#ifndef CFWX_TENSOR_H
#define CFWX_TENSOR_H
#include "cfwx.h"
/** @def CFWX_MAX_RANK
 * @brief cfwx max rank macro; see expansion and module contract.
 */
#define CFWX_MAX_RANK 8
/** @brief Record/enumeration used by this module; member contracts describe storage and policy. */
typedef struct
{
    size_t rank, size, shape[CFWX_MAX_RANK], stride[CFWX_MAX_RANK]; /**< State/dimension/policy value; see the containing type contract. */
    double *data; /**< Borrowed pointer or owned storage as specified by the containing type. */
} cfwx_tensor;
/**
 * @anchor api_include_datastructures_tensor_h_cfwx_tensor_init
 * @brief Construct and initialize tensor state.
 * @ingroup containers
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @param t Caller-owned destination; must be zero-initialized/empty before construction. Successful construction transfers owned storage into it.
 * @param rank Tensor rank; supported range is 1 through CFWX_MAX_RANK.
 * @param shape Dimension array with rank entries; every dimension must be positive.
 * @return CFWX_OK on success; otherwise an invalid-argument, shape, allocation
 * or numerical status as checked by the implementation. Do not consume failed outputs.
 */
cfwx_status cfwx_tensor_init(cfwx_tensor *t, size_t rank, const size_t *shape);
/**
 * @anchor api_include_datastructures_tensor_h_cfwx_tensor_free
 * @brief Release owned tensor storage and reset its owner.
 * @ingroup containers
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @param t Owner whose allocated storage is released and state reset; NULL is accepted where the implementation documents/checks it.
 * @note This operation returns no value; inspect its documented mutation/lifetime effects.
 */
void cfwx_tensor_free(cfwx_tensor *t);
/**
 * @anchor api_include_datastructures_tensor_h_cfwx_tensor_offset
 * @brief Check tensor indices and compute a contiguous-storage offset.
 * @ingroup containers
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @param t t argument as declared; use the adjacent contract for storage, count and input/output role.
 * @param indices Index array with one entry per tensor dimension.
 * @param offset Position offset or checked flattened tensor offset, as specified by this operation.
 * @return CFWX_OK on success; otherwise an invalid-argument, shape, allocation
 * or numerical status as checked by the implementation. Do not consume failed outputs.
 */
cfwx_status cfwx_tensor_offset(const cfwx_tensor *t, const size_t *indices, size_t *offset);
/**
 * @anchor api_include_datastructures_tensor_h_cfwx_tensor_reshape
 * @brief Replace tensor metadata while preserving element count and allocation.
 * @ingroup containers
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @param t t argument as declared; use the adjacent contract for storage, count and input/output role.
 * @param rank Tensor rank; supported range is 1 through CFWX_MAX_RANK.
 * @param shape Dimension array with rank entries; every dimension must be positive.
 * @return CFWX_OK on success; otherwise an invalid-argument, shape, allocation
 * or numerical status as checked by the implementation. Do not consume failed outputs.
 */
cfwx_status cfwx_tensor_reshape(cfwx_tensor *t, size_t rank, const size_t *shape);
#endif
