/**
 * @file src/datastructures/neuron.c
 * @brief neuron interfaces and implementation.
 * @ingroup containers
 * @details See the module guide for shape, ownership, failure and execution-policy
 * constraints. Source listings include implementation helpers and ordinary
 * explanatory comments; generated source documentation does not execute code.
 */

/** Implementation and checks for neuron. */

#include "datastructures/neuron.h"

/**
 * @anchor api_src_datastructures_neuron_c_cfwx_neuron_init
 * @brief Construct and initialize neuron state.
 * @ingroup containers
 * @details Implementation of the public declaration; its canonical contract
 * remains in the corresponding header and is merged by Doxygen.
 * @see cfwx_neuron_init include/datastructures/neuron.h
 */
cfwx_status cfwx_neuron_init(cfwx_neuron *n, size_t inputs)
{
    if (!n)
        return CFWX_INVALID;
    cfwx_status s = cfwx_vector_init(&n->weights, inputs);
    if (s == CFWX_OK)
        n->bias = 0;
    return s;
}

/**
 * @anchor api_src_datastructures_neuron_c_cfwx_neuron_free
 * @brief Release owned neuron storage and reset its owner.
 * @ingroup containers
 * @details Implementation of the public declaration; its canonical contract
 * remains in the corresponding header and is merged by Doxygen.
 * @see cfwx_neuron_free include/datastructures/neuron.h
 */
void cfwx_neuron_free(cfwx_neuron *n)
{
    if (n)
    {
        cfwx_vector_free(&n->weights);
        n->bias = 0;
    }
}

/**
 * @anchor api_src_datastructures_neuron_c_cfwx_neuron_forward
 * @brief Perform the neuron forward operation.
 * @ingroup containers
 * @details Implementation of the public declaration; its canonical contract
 * remains in the corresponding header and is merged by Doxygen.
 * @see cfwx_neuron_forward include/datastructures/neuron.h
 */
cfwx_status cfwx_neuron_forward(const cfwx_neuron *n, const cfwx_vector *x, double *out)
{
    if (!n || !out)
        return CFWX_INVALID;
    double v;
    cfwx_status s = cfwx_vector_dot(&n->weights, x, &v);
    if (s != CFWX_OK)
        return s;
    v += n->bias;
    if (!isfinite(v))
        return CFWX_NUMERIC;
    *out = v;
    return CFWX_OK;
}
