/* Implementation and checks for vector. */

#include "datastructures/vector.h"

cfwx_status cfwx_vector_init(cfwx_vector *v, size_t n)
{
    size_t count;
    if (!v || v->data || v->size || !cfwx_count(n, 1, &count))
        return CFWX_INVALID;
    double *p = calloc(count, sizeof(double));
    if (!p)
        return CFWX_ALLOC;
    *v = (cfwx_vector){n, p};
    return CFWX_OK;
}

void cfwx_vector_free(cfwx_vector *v)
{
    if (v)
    {
        free(v->data);
        *v = (cfwx_vector){0};
    }
}

cfwx_status cfwx_vector_dot(const cfwx_vector *a, const cfwx_vector *b, double *out)
{
    /* Kahan correction reduces low-order rounding loss in summation. */

    if (!a || !b || !out || !a->data || !b->data)
        return CFWX_INVALID;
    if (a->size != b->size)
        return CFWX_SHAPE;
    double sum = 0.0, c = 0;
    for (size_t i = 0; i < a->size; i++)
    {
        double y = a->data[i] * b->data[i] - c, t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    if (!isfinite(sum))
        return CFWX_NUMERIC;
    *out = sum;
    return CFWX_OK;
}

cfwx_status cfwx_vector_add(const cfwx_vector *a, const cfwx_vector *b, cfwx_vector *out)
{
    if (!a || !b || !out || !a->data || !b->data || !out->data)
        return CFWX_INVALID;
    if (a->size != b->size || a->size != out->size)
        return CFWX_SHAPE;
    for (size_t i = 0; i < a->size; i++)
        out->data[i] = a->data[i] + b->data[i];
    return CFWX_OK;
}

cfwx_status cfwx_vector_scale(cfwx_vector *v, double scale)
{
    if (!v || !v->data || !isfinite(scale))
        return CFWX_INVALID;
    for (size_t i = 0; i < v->size; i++)
        v->data[i] *= scale;
    return CFWX_OK;
}
