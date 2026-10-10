/**
 * @file src/maths/activation_functions.c
 * @brief activation functions interfaces and implementation.
 * @ingroup activations
 * @details See the module guide for shape, ownership, failure and execution-policy
 * constraints. Source listings include implementation helpers and ordinary
 * explanatory comments; generated source documentation does not execute code.
 */

/** Implementation and checks for activation functions. */

#include "maths/activation_functions.h"

/**
 * @anchor api_src_maths_activation_functions_c_cfwx_sigmoid
 * @brief Evaluate a sign-stabilized logistic sigmoid.
 * @ingroup activations
 * @details Implementation of the public declaration; its canonical contract
 * remains in the corresponding header and is merged by Doxygen.
 * @see cfwx_sigmoid include/maths/activation_functions.h
 */
double cfwx_sigmoid(double x)
{
    /* Choose the exponent sign to avoid overflow for extreme inputs. */

    if (x >= 0)
        return 1 / (1 + exp(-x));
    double e = exp(x);
    return e / (1 + e);
}

/**
 * @anchor api_src_maths_activation_functions_c_cfwx_sigmoid_derivative
 * @brief Perform the sigmoid derivative operation.
 * @ingroup activations
 * @details Implementation of the public declaration; its canonical contract
 * remains in the corresponding header and is merged by Doxygen.
 * @see cfwx_sigmoid_derivative include/maths/activation_functions.h
 */
double cfwx_sigmoid_derivative(double x)
{
    double s = cfwx_sigmoid(x);
    return s * (1 - s);
}

/**
 * @anchor api_src_maths_activation_functions_c_cfwx_relu
 * @brief Evaluate ReLU, propagating NaN inputs.
 * @ingroup activations
 * @details Implementation of the public declaration; its canonical contract
 * remains in the corresponding header and is merged by Doxygen.
 * @see cfwx_relu include/maths/activation_functions.h
 */
double cfwx_relu(double x)
{
    return isnan(x) ? x : (x > 0 ? x : 0);
}

/**
 * @anchor api_src_maths_activation_functions_c_cfwx_relu_derivative
 * @brief Evaluate the ReLU derivative with zero derivative at the origin.
 * @ingroup activations
 * @details Implementation of the public declaration; its canonical contract
 * remains in the corresponding header and is merged by Doxygen.
 * @see cfwx_relu_derivative include/maths/activation_functions.h
 */
double cfwx_relu_derivative(double x)
{
    return x > 0 ? 1 : 0;
}

/**
 * @anchor api_src_maths_activation_functions_c_cfwx_tanh_derivative
 * @brief Perform the tanh derivative operation.
 * @ingroup activations
 * @details Implementation of the public declaration; its canonical contract
 * remains in the corresponding header and is merged by Doxygen.
 * @see cfwx_tanh_derivative include/maths/activation_functions.h
 */
double cfwx_tanh_derivative(double x)
{
    double t = tanh(x);
    return 1 - t * t;
}

/**
 * @anchor api_src_maths_activation_functions_c_cfwx_gelu
 * @brief Evaluate exact erf-based GELU rather than the tanh approximation.
 * @ingroup activations
 * @details Implementation of the public declaration; its canonical contract
 * remains in the corresponding header and is merged by Doxygen.
 * @see cfwx_gelu include/maths/activation_functions.h
 */
double cfwx_gelu(double x)
{
    /* Exact erf-based GELU; handle infinite limits before multiplying by zero. */

    if (isinf(x))
        return x > 0 ? x : 0;
    return .5 * x * (1 + erf(x / sqrt(2.0)));
}

/**
 * @anchor api_src_maths_activation_functions_c_cfwx_gelu_derivative
 * @brief Evaluate the analytic derivative of exact GELU.
 * @ingroup activations
 * @details Implementation of the public declaration; its canonical contract
 * remains in the corresponding header and is merged by Doxygen.
 * @see cfwx_gelu_derivative include/maths/activation_functions.h
 */
double cfwx_gelu_derivative(double x)
{
    if (isinf(x))
        return x > 0 ? 1 : 0;
    return .5 * (1 + erf(x / sqrt(2.0))) + x * exp(-.5 * x * x) / sqrt(6.2831853071795864769);
}

/**
 * @anchor api_src_maths_activation_functions_c_cfwx_softmax
 * @brief Evaluate max-shifted softmax for finite logits.
 * @ingroup activations
 * @details Implementation of the public declaration; its canonical contract
 * remains in the corresponding header and is merged by Doxygen.
 * @see cfwx_softmax include/maths/activation_functions.h
 */
cfwx_status cfwx_softmax(const double *x, size_t n, double *out)
{
    /* A common logit shift preserves probabilities and avoids exponential overflow. */

    if (!x || !out || !n)
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
    {
        out[i] = exp(x[i] - max);
        sum += out[i];
    }
    for (size_t i = 0; i < n; i++)
        out[i] /= sum;
    return CFWX_OK;
}
