/**
 * @file src/maths/rewards.c
 * @brief rewards interfaces and implementation.
 * @ingroup rewards
 * @details See the module guide for shape, ownership, failure and execution-policy
 * constraints. Source listings include implementation helpers and ordinary
 * explanatory comments; generated source documentation does not execute code.
 */

/** Implementation and checks for rewards. */

#include "maths/rewards.h"

/**
 * @anchor api_src_maths_rewards_c_cfwx_discounted_returns
 * @brief Compute reverse discounted returns with zero terminal bootstrap.
 * @ingroup rewards
 * @details Implementation of the public declaration; its canonical contract
 * remains in the corresponding header and is merged by Doxygen.
 * @see cfwx_discounted_returns include/maths/rewards.h
 */
cfwx_status cfwx_discounted_returns(const double *r, size_t n, double gamma, double *out)
{
    /* Reverse traversal supports in-place returns with zero terminal bootstrap. */

    if (!r || !out || !n || !isfinite(gamma) || gamma < 0 || gamma > 1)
        return CFWX_INVALID;
    for (size_t i = 0; i < n; i++)
        if (!isfinite(r[i]))
            return CFWX_NUMERIC;
    double g = 0.0;
    for (size_t i = n; i-- > 0;)
    {
        g = r[i] + gamma * g;
        if (!isfinite(g))
            return CFWX_NUMERIC;
        out[i] = g;
    }
    return CFWX_OK;
}
