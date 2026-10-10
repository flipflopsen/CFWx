/* Implementation and checks for neuron. */

#include "datastructures/neuron.h"

cfwx_status cfwx_neuron_init(cfwx_neuron *n, size_t inputs)
{
    if (!n)
        return CFWX_INVALID;
    cfwx_status s = cfwx_vector_init(&n->weights, inputs);
    if (s == CFWX_OK)
        n->bias = 0;
    return s;
}

void cfwx_neuron_free(cfwx_neuron *n)
{
    if (n)
    {
        cfwx_vector_free(&n->weights);
        n->bias = 0;
    }
}

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
