/**
 * @file include/datastructures/neuron.h
 * @brief neuron interfaces and implementation.
 * @ingroup containers
 * @details See the module guide for shape, ownership, failure and execution-policy
 * constraints. Source listings include implementation helpers and ordinary
 * explanatory comments; generated source documentation does not execute code.
 */

/** Public contracts for neuron. */

#ifndef CFWX_NEURON_H
#define CFWX_NEURON_H
#include "datastructures/vector.h"

/** @brief Record/enumeration used by this module; member contracts describe storage and policy. */
typedef struct
{
    cfwx_vector weights; /**< State/dimension/policy value; see the containing type contract. */
    double bias; /**< State/dimension/policy value; see the containing type contract. */
} cfwx_neuron;
/**
 * @anchor api_include_datastructures_neuron_h_cfwx_neuron_init
 * @brief Construct and initialize neuron state.
 * @ingroup containers
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @param n Caller-owned destination; must be zero-initialized/empty before construction. Successful construction transfers owned storage into it.
 * @param inputs inputs argument; interpretation follows this operation's implementation and contract.
 * @return CFWX_OK on success; otherwise an invalid-argument, shape, allocation
 * or numerical status as checked by the implementation. Do not consume failed outputs.
 */
cfwx_status cfwx_neuron_init(cfwx_neuron *n, size_t inputs);
/**
 * @anchor api_include_datastructures_neuron_h_cfwx_neuron_free
 * @brief Release owned neuron storage and reset its owner.
 * @ingroup containers
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @param n Owner whose allocated storage is released and state reset; NULL is accepted where the implementation documents/checks it.
 * @note This operation returns no value; inspect its documented mutation/lifetime effects.
 */
void cfwx_neuron_free(cfwx_neuron *n);
/** Affine output; activation selected by caller. */
/**
 * @anchor api_include_datastructures_neuron_h_cfwx_neuron_forward
 * @brief Perform the neuron forward operation.
 * @ingroup containers
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @par Operation contract
 * Affine output; activation selected by caller.
 * @param n Element count or owner/internal node as declared; see containing module contracts.
 * @param x Vector owner/operand with contiguous size-element storage; constructor/free versus input role follows this operation.
 * @param out Caller-supplied output storage; allocate its required shape before calling. Constructor APIs instead receive an owner destination.
 * @return CFWX_OK on success; otherwise an invalid-argument, shape, allocation
 * or numerical status as checked by the implementation. Do not consume failed outputs.
 */
cfwx_status cfwx_neuron_forward(const cfwx_neuron *n, const cfwx_vector *x, double *out);
#endif
