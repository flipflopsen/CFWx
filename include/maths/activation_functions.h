/**
 * @file include/maths/activation_functions.h
 * @brief activation functions interfaces and implementation.
 * @ingroup activations
 * @details See the module guide for shape, ownership, failure and execution-policy
 * constraints. Source listings include implementation helpers and ordinary
 * explanatory comments; generated source documentation does not execute code.
 */

/** Public contracts for activation functions. */

#ifndef CFWX_ACTIVATIONS_H
#define CFWX_ACTIVATIONS_H
#include "cfwx.h"

/**
 * @anchor api_include_maths_activation_functions_h_cfwx_sigmoid
 * @brief Evaluate a sign-stabilized logistic sigmoid.
 * @ingroup activations
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @param x Borrowed primal/input numeric operand; shape is stated in the operation contract.
 * @return Computed value or status described by the adjacent contract.
 */
double cfwx_sigmoid(double x);
/**
 * @anchor api_include_maths_activation_functions_h_cfwx_sigmoid_derivative
 * @brief Perform the sigmoid derivative operation.
 * @ingroup activations
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @param x Borrowed primal/input numeric operand; shape is stated in the operation contract.
 * @return Computed value or status described by the adjacent contract.
 */
double cfwx_sigmoid_derivative(double x);
/**
 * @anchor api_include_maths_activation_functions_h_cfwx_relu
 * @brief Evaluate ReLU, propagating NaN inputs.
 * @ingroup activations
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @param x Borrowed primal/input numeric operand; shape is stated in the operation contract.
 * @return Computed value or status described by the adjacent contract.
 */
double cfwx_relu(double x);
/**
 * @anchor api_include_maths_activation_functions_h_cfwx_relu_derivative
 * @brief Evaluate the ReLU derivative with zero derivative at the origin.
 * @ingroup activations
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @param x Borrowed primal/input numeric operand; shape is stated in the operation contract.
 * @return Computed value or status described by the adjacent contract.
 */
double cfwx_relu_derivative(double x);
/**
 * @anchor api_include_maths_activation_functions_h_cfwx_tanh_derivative
 * @brief Perform the tanh derivative operation.
 * @ingroup activations
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @param x Borrowed primal/input numeric operand; shape is stated in the operation contract.
 * @return Computed value or status described by the adjacent contract.
 */
double cfwx_tanh_derivative(double x);
/**
 * @anchor api_include_maths_activation_functions_h_cfwx_gelu
 * @brief Evaluate exact erf-based GELU rather than the tanh approximation.
 * @ingroup activations
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @param x Borrowed primal/input numeric operand; shape is stated in the operation contract.
 * @return Computed value or status described by the adjacent contract.
 */
double cfwx_gelu(double x);
/**
 * @anchor api_include_maths_activation_functions_h_cfwx_gelu_derivative
 * @brief Evaluate the analytic derivative of exact GELU.
 * @ingroup activations
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @param x Borrowed primal/input numeric operand; shape is stated in the operation contract.
 * @return Computed value or status described by the adjacent contract.
 */
double cfwx_gelu_derivative(double x);
/** Finite logits; exact input/output aliasing allowed. */
/**
 * @anchor api_include_maths_activation_functions_h_cfwx_softmax
 * @brief Evaluate max-shifted softmax for finite logits.
 * @ingroup activations
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @par Operation contract
 * Finite logits; exact input/output aliasing allowed.
 * @param x Borrowed primal/input numeric operand; shape is stated in the operation contract.
 * @param n Positive scalar element count; each associated vector/array must contain at least n entries.
 * @param out Caller-supplied output storage; allocate its required shape before calling. Constructor APIs instead receive an owner destination.
 * @return CFWX_OK on success; otherwise an invalid-argument, shape, allocation
 * or numerical status as checked by the implementation. Do not consume failed outputs.
 */
cfwx_status cfwx_softmax(const double *x, size_t n, double *out);
#endif
