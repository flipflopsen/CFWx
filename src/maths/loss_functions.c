/**
 * @file src/maths/loss_functions.c
 * @brief loss functions interfaces and implementation.
 * @ingroup losses
 * @details See the module guide for shape, ownership, failure and execution-policy
 * constraints. Source listings include implementation helpers and ordinary
 * explanatory comments; generated source documentation does not execute code.
 */

/** Implementation and checks for loss functions. */

#include "maths/loss_functions.h"
#include "maths/activation_functions.h"

/**
 * @anchor api_src_maths_loss_functions_c_cfwx_mse
 * @brief Compute mean squared error and optional scalar-element-reduced gradient.
 * @ingroup losses
 * @details Implementation of the public declaration; its canonical contract
 * remains in the corresponding header and is merged by Doxygen.
 * @see cfwx_mse include/maths/loss_functions.h
 */
cfwx_status cfwx_mse(const double *p, const double *t, size_t n, double *loss, double *g)
{
    /* Mean scalar-element reduction also determines the gradient scaling. */

    if (!p || !t || !n || !loss)
        return CFWX_INVALID;
    double s = 0.0;
    for (size_t i = 0; i < n; i++)
    {
        if (!isfinite(p[i]) || !isfinite(t[i]))
            return CFWX_NUMERIC;
        double d = p[i] - t[i];
        s += d * d / (double)n;
        if (g)
            g[i] = 2 * d / (double)n;
    }
    if (!isfinite(s))
        return CFWX_NUMERIC;
    *loss = s;
    return CFWX_OK;
}

/**
 * @anchor api_src_maths_loss_functions_c_cfwx_cross_entropy_logits
 * @brief Compute stable categorical cross entropy and optional logit gradient.
 * @ingroup losses
 * @details Implementation of the public declaration; its canonical contract
 * remains in the corresponding header and is merged by Doxygen.
 * @see cfwx_cross_entropy_logits include/maths/loss_functions.h
 */
cfwx_status cfwx_cross_entropy_logits(const double *x, size_t n, size_t label, double *loss,
                                      double *g)
{
    /* Shifted log-sum-exp is stable; gradient is softmax minus one-hot. */

    if (!x || !n || label >= n || !loss)
        return CFWX_INVALID;
    double max = x[0];
    for (size_t i = 0; i < n; i++)
    {
        if (!isfinite(x[i]))
            return CFWX_NUMERIC;
        if (x[i] > max)
            max = x[i];
    }
    double sum = 0.0;
    for (size_t i = 0; i < n; i++)
        sum += exp(x[i] - max);
    double value = (max - x[label]) + log(sum);
    if (!isfinite(value))
        return CFWX_NUMERIC;
    if (g)
    {
        cfwx_status s = cfwx_softmax(x, n, g);
        if (s != CFWX_OK)
            return s;
        g[label] -= 1;
    }
    *loss = value;
    return CFWX_OK;
}

/**
 * @anchor api_src_maths_loss_functions_c_cfwx_bce_logits
 * @brief Compute stable mean binary cross entropy from logits.
 * @ingroup losses
 * @details Implementation of the public declaration; its canonical contract
 * remains in the corresponding header and is merged by Doxygen.
 * @see cfwx_bce_logits include/maths/loss_functions.h
 */
cfwx_status cfwx_bce_logits(const double *x, const double *t, size_t n, double *loss, double *g)
{
    /* Stable softplus handles large logits; fractional targets are permitted in [0,1]. */

    if (!x || !t || !n || !loss)
        return CFWX_INVALID;
    for (size_t i = 0; i < n; i++)
        if (!isfinite(x[i]) || !isfinite(t[i]) || t[i] < 0 || t[i] > 1)
            return CFWX_INVALID;
    double sum = 0.0;
    for (size_t i = 0; i < n; i++)
    {
        sum +=
            ((x[i] >= 0 ? x[i] * (1 - t[i]) : -x[i] * t[i]) + log1p(exp(-fabs(x[i])))) / (double)n;
        if (g)
            g[i] = (cfwx_sigmoid(x[i]) - t[i]) / (double)n;
    }
    if (!isfinite(sum))
        return CFWX_NUMERIC;
    *loss = sum;
    return CFWX_OK;
}
