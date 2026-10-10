/**
 * @file include/models/ann.h
 * @brief ann interfaces and implementation.
 * @ingroup dense
 * @details See the module guide for shape, ownership, failure and execution-policy
 * constraints. Source listings include implementation helpers and ordinary
 * explanatory comments; generated source documentation does not execute code.
 */

/** Public contracts for ann. */

#ifndef CFWX_ANN_H
#define CFWX_ANN_H
#include "datastructures/matrix.h"
#include "datastructures/vector.h"
/** W [inputs,outputs], x [batch,inputs], y [batch,outputs]. */
/** @brief Record/enumeration used by this module; member contracts describe storage and policy. */
typedef struct
{
    cfwx_matrix weights; /**< State/dimension/policy value; see the containing type contract. */
    cfwx_vector bias; /**< State/dimension/policy value; see the containing type contract. */
} cfwx_dense;
/**
 * @anchor api_include_models_ann_h_cfwx_dense_init
 * @brief Construct and initialize dense state.
 * @ingroup dense
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @param l Caller-owned destination; must be zero-initialized/empty before construction. Successful construction transfers owned storage into it.
 * @param inputs inputs argument; interpretation follows this operation's implementation and contract.
 * @param outputs outputs argument; interpretation follows this operation's implementation and contract.
 * @param seed Caller-supplied pseudorandom initialization seed or mutable RNG state; noncryptographic.
 * @return CFWX_OK on success; otherwise an invalid-argument, shape, allocation
 * or numerical status as checked by the implementation. Do not consume failed outputs.
 */
cfwx_status cfwx_dense_init(cfwx_dense *l, size_t inputs, size_t outputs, uint64_t seed);
/**
 * @anchor api_include_models_ann_h_cfwx_dense_free
 * @brief Release owned dense storage and reset its owner.
 * @ingroup dense
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @param l Owner whose allocated storage is released and state reset; NULL is accepted where the implementation documents/checks it.
 * @note This operation returns no value; inspect its documented mutation/lifetime effects.
 */
void cfwx_dense_free(cfwx_dense *l);
/**
 * @anchor api_include_models_ann_h_cfwx_dense_forward
 * @brief Perform the dense forward operation.
 * @ingroup dense
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @param l Model/operation state; input versus owner role follows the declaration and operation contract.
 * @param x Borrowed primal/input numeric operand; shape is stated in the operation contract.
 * @param y y argument; interpretation follows this operation's implementation and contract.
 * @return CFWX_OK on success; otherwise an invalid-argument, shape, allocation
 * or numerical status as checked by the implementation. Do not consume failed outputs.
 */
cfwx_status cfwx_dense_forward(const cfwx_dense *l, const cfwx_matrix *x, cfwx_matrix *y);
/** Overwrites dX=dY*W^T, dW=X^T*dY, db=sum_batch(dY).
 * No implicit averaging; reduction belongs to the loss. All outputs disjoint. */
/**
 * @anchor api_include_models_ann_h_cfwx_dense_backward
 * @brief Compute analytic adjoints for dense.
 * @ingroup dense
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @par Operation contract
 * Overwrites dX=dY*W^T, dW=X^T*dY, db=sum_batch(dY). No implicit averaging; reduction belongs to the loss. All outputs disjoint.
 * @param l Model/operation state; input versus owner role follows the declaration and operation contract.
 * @param x Borrowed primal/input numeric operand; shape is stated in the operation contract.
 * @param dy Borrowed upstream adjoint; same output shape as the corresponding forward operation.
 * @param dx Preallocated analytic gradient output; shape matches the corresponding primal.
 * @param dw Preallocated analytic gradient output; shape matches the corresponding primal.
 * @param db Vector owner/operand with contiguous size-element storage; constructor/free versus input role follows this operation.
 * @return CFWX_OK on success; otherwise an invalid-argument, shape, allocation
 * or numerical status as checked by the implementation. Do not consume failed outputs.
 */
cfwx_status cfwx_dense_backward(const cfwx_dense *l, const cfwx_matrix *x, const cfwx_matrix *dy,
                                cfwx_matrix *dx, cfwx_matrix *dw, cfwx_vector *db);
/**
 * @anchor api_include_models_ann_h_cfwx_sgd_step
 * @brief Perform the sgd step operation.
 * @ingroup dense
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @param p Parameter registry, dropout probability, or operand as declared; see the operation-specific formula.
 * @param g Gradient record or internal operand as declared; shape/reduction follows this operation contract.
 * @param n Positive scalar element count; each associated vector/array must contain at least n entries.
 * @param rate Optimizer learning rate; zero is allowed only by APIs documenting gradient-only execution.
 * @return CFWX_OK on success; otherwise an invalid-argument, shape, allocation
 * or numerical status as checked by the implementation. Do not consume failed outputs.
 */
cfwx_status cfwx_sgd_step(double *p, const double *g, size_t n, double rate);
#endif
