/**
 * @file include/maths/loss_functions.h
 * @brief loss functions interfaces and implementation.
 * @ingroup losses
 * @details See the module guide for shape, ownership, failure and execution-policy
 * constraints. Source listings include implementation helpers and ordinary
 * explanatory comments; generated source documentation does not execute code.
 */

/** Public contracts for loss functions. */

#ifndef CFWX_LOSSES_H
#define CFWX_LOSSES_H
#include "cfwx.h"
/** Optional gradients have n entries and may not overlap inputs.
 * MSE/BCE reduction is mean over scalar elements. CE is one categorical label. */
/**
 * @anchor api_include_maths_loss_functions_h_cfwx_mse
 * @brief Compute mean squared error and optional scalar-element-reduced gradient.
 * @ingroup losses
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @par Operation contract
 * Optional gradients have n entries and may not overlap inputs. MSE/BCE reduction is mean over scalar elements. CE is one categorical label.
 * @param p Parameter registry, dropout probability, or operand as declared; see the operation-specific formula.
 * @param t t argument as declared; use the adjacent contract for storage, count and input/output role.
 * @param n Positive scalar element count; each associated vector/array must contain at least n entries.
 * @param loss Output scalar objective; consult scalar-element versus token-mean reduction.
 * @param g Gradient record or internal operand as declared; shape/reduction follows this operation contract.
 * @return CFWX_OK on success; otherwise an invalid-argument, shape, allocation
 * or numerical status as checked by the implementation. Do not consume failed outputs.
 */
cfwx_status cfwx_mse(const double *p, const double *t, size_t n, double *loss, double *g);
/**
 * @anchor api_include_maths_loss_functions_h_cfwx_cross_entropy_logits
 * @brief Compute stable categorical cross entropy and optional logit gradient.
 * @ingroup losses
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @param x Borrowed primal/input numeric operand; shape is stated in the operation contract.
 * @param n Positive scalar element count; each associated vector/array must contain at least n entries.
 * @param label label argument; interpretation follows this operation's implementation and contract.
 * @param loss Output scalar objective; consult scalar-element versus token-mean reduction.
 * @param g Gradient record or internal operand as declared; shape/reduction follows this operation contract.
 * @return CFWX_OK on success; otherwise an invalid-argument, shape, allocation
 * or numerical status as checked by the implementation. Do not consume failed outputs.
 */
cfwx_status cfwx_cross_entropy_logits(const double *x, size_t n, size_t label, double *loss,
                                      double *g);
/**
 * @anchor api_include_maths_loss_functions_h_cfwx_bce_logits
 * @brief Compute stable mean binary cross entropy from logits.
 * @ingroup losses
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @param x Borrowed primal/input numeric operand; shape is stated in the operation contract.
 * @param t t argument as declared; use the adjacent contract for storage, count and input/output role.
 * @param n Positive scalar element count; each associated vector/array must contain at least n entries.
 * @param loss Output scalar objective; consult scalar-element versus token-mean reduction.
 * @param g Gradient record or internal operand as declared; shape/reduction follows this operation contract.
 * @return CFWX_OK on success; otherwise an invalid-argument, shape, allocation
 * or numerical status as checked by the implementation. Do not consume failed outputs.
 */
cfwx_status cfwx_bce_logits(const double *x, const double *t, size_t n, double *loss, double *g);
#endif
