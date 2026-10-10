/**
 * @file include/models/attention.h
 * @brief attention interfaces and implementation.
 * @ingroup attention
 * @details See the module guide for shape, ownership, failure and execution-policy
 * constraints. Source listings include implementation helpers and ordinary
 * explanatory comments; generated source documentation does not execute code.
 */

/** Public contracts for attention. */

#ifndef CFWX_ATTENTION_H
#define CFWX_ATTENTION_H
#include <stdbool.h>
#include "datastructures/matrix.h"
/** Q [queries,d], K [keys,d], V [keys,value_dim], output [queries,value_dim].
 * Optional probabilities [queries,keys]. Causal mode requires queries==keys;
 * masks key index > query index. Finite inputs, disjoint outputs. Single head. */
/**
 * @anchor api_include_models_attention_h_cfwx_attention
 * @brief Perform the attention operation.
 * @ingroup attention
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @par Operation contract
 * Q [queries,d], K [keys,d], V [keys,value_dim], output [queries,value_dim]. Optional probabilities [queries,keys]. Causal mode requires queries==keys; masks key index > query index. Finite inputs, disjoint outputs. Single head.
 * @param q Query operand [queries,head_depth] for attention, or corresponding model query state.
 * @param k Key operand [keys,head_depth] for attention.
 * @param v Value operand [keys,value_depth] for attention.
 * @param causal Whether to additionally permit only key positions at or before each query.
 * @param out Caller-supplied output storage; allocate its required shape before calling. Constructor APIs instead receive an owner destination.
 * @param p Parameter registry, dropout probability, or operand as declared; see the operation-specific formula.
 * @return CFWX_OK on success; otherwise an invalid-argument, shape, allocation
 * or numerical status as checked by the implementation. Do not consume failed outputs.
 */
cfwx_status cfwx_attention(const cfwx_matrix *q, const cfwx_matrix *k, const cfwx_matrix *v,
                           bool causal, cfwx_matrix *out, cfwx_matrix *p);
/** Row-wise population variance. gamma/beta each have cols entries.
 * x/out may alias exactly; gamma/beta must be disjoint. epsilon > 0. */
/**
 * @anchor api_include_models_attention_h_cfwx_layer_norm
 * @brief Perform the layer norm operation.
 * @ingroup attention
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @par Operation contract
 * Row-wise population variance. gamma/beta each have cols entries. x/out may alias exactly; gamma/beta must be disjoint. epsilon > 0.
 * @param x Borrowed primal/input numeric operand; shape is stated in the operation contract.
 * @param gamma Feature scale array; one value per feature. Borrowed input storage.
 * @param beta Feature shift array; one value per feature. Borrowed input storage.
 * @param eps Positive finite numerical-stability constant, not a trainable parameter.
 * @param out Caller-supplied output storage; allocate its required shape before calling. Constructor APIs instead receive an owner destination.
 * @return CFWX_OK on success; otherwise an invalid-argument, shape, allocation
 * or numerical status as checked by the implementation. Do not consume failed outputs.
 */
cfwx_status cfwx_layer_norm(const cfwx_matrix *x, const double *gamma, const double *beta,
                            double eps, cfwx_matrix *out);
#endif
