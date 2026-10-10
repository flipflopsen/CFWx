/**
 * @file include/maths/rewards.h
 * @brief rewards interfaces and implementation.
 * @ingroup rewards
 * @details See the module guide for shape, ownership, failure and execution-policy
 * constraints. Source listings include implementation helpers and ordinary
 * explanatory comments; generated source documentation does not execute code.
 */

/** Public contracts for rewards. */

#ifndef CFWX_REWARDS_H
#define CFWX_REWARDS_H
#include "cfwx.h"
/** G[t]=r[t]+gamma*G[t+1], terminal bootstrap zero; exact in-place supported.
 * Split episode boundaries in the caller. */
/**
 * @anchor api_include_maths_rewards_h_cfwx_discounted_returns
 * @brief Compute reverse discounted returns with zero terminal bootstrap.
 * @ingroup rewards
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @par Operation contract
 * G[t]=r[t]+gamma*G[t+1], terminal bootstrap zero; exact in-place supported. Split episode boundaries in the caller.
 * @param r r argument; interpretation follows this operation's implementation and contract.
 * @param n Positive scalar element count; each associated vector/array must contain at least n entries.
 * @param gamma Feature scale array; one value per feature. Borrowed input storage.
 * @param out Caller-supplied output storage; allocate its required shape before calling. Constructor APIs instead receive an owner destination.
 * @return CFWX_OK on success; otherwise an invalid-argument, shape, allocation
 * or numerical status as checked by the implementation. Do not consume failed outputs.
 */
cfwx_status cfwx_discounted_returns(const double *r, size_t n, double gamma, double *out);
#endif
