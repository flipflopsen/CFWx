/**
 * @file src/models/attention.c
 * @brief attention interfaces and implementation.
 * @ingroup attention
 * @details See the module guide for shape, ownership, failure and execution-policy
 * constraints. Source listings include implementation helpers and ordinary
 * explanatory comments; generated source documentation does not execute code.
 */

/** Implementation and checks for attention. */

#include "models/attention.h"
#include <float.h>

/**
 * @anchor api_src_models_attention_c_cfwx_attention
 * @brief Perform the attention operation.
 * @ingroup attention
 * @details Implementation of the public declaration; its canonical contract
 * remains in the corresponding header and is merged by Doxygen.
 * @see cfwx_attention include/models/attention.h
 * @note Internal helper state may be temporary. Referenced-by/source links show
 * how composite callers propagate errors and release allocations.
 */
cfwx_status cfwx_attention(const cfwx_matrix *q, const cfwx_matrix *k, const cfwx_matrix *v,
                           bool causal, cfwx_matrix *out, cfwx_matrix *p)
{
    /* Reuse one score row; causal masking allows key indices <= query index. */

    if (!q || !k || !v || !out || !q->data || !k->data || !v->data || !out->data || !q->cols ||
        !k->rows)
        return CFWX_INVALID;
    if (q->cols != k->cols || k->rows != v->rows || out->rows != q->rows || out->cols != v->cols ||
        (causal && q->rows != k->rows) || (p && (p->rows != q->rows || p->cols != k->rows)))
        return CFWX_SHAPE;
    if (p && !p->data)
        return CFWX_INVALID;
    if (out->data == q->data || out->data == k->data || out->data == v->data ||
        (p &&
         (p->data == q->data || p->data == k->data || p->data == v->data || p->data == out->data)))
        return CFWX_INVALID;
    for (size_t i = 0; i < q->rows * q->cols; i++)
        if (!isfinite(q->data[i]))
            return CFWX_NUMERIC;
    for (size_t i = 0; i < k->rows * k->cols; i++)
        if (!isfinite(k->data[i]))
            return CFWX_NUMERIC;
    for (size_t i = 0; i < v->rows * v->cols; i++)
        if (!isfinite(v->data[i]))
            return CFWX_NUMERIC;
    double *row = calloc(k->rows, sizeof(double));
    if (!row)
        return CFWX_ALLOC;
    cfwx_status status = CFWX_OK;
    double scale = 1 / sqrt((double)q->cols);
    for (size_t i = 0; i < q->rows; i++)
    {
        size_t count = causal ? i + 1 : k->rows;
        double max = -DBL_MAX;
        for (size_t j = 0; j < count; j++)
        {
            double s = 0.0;
            for (size_t d = 0; d < q->cols; d++)
                s += q->data[i * q->cols + d] * k->data[j * k->cols + d];
            row[j] = s * scale;
            if (!isfinite(row[j]))
            {
                status = CFWX_NUMERIC;
                goto done;
            }
            if (row[j] > max)
                max = row[j];
        }
        double sum = 0.0;
        for (size_t j = 0; j < count; j++)
        {
            row[j] = exp(row[j] - max);
            sum += row[j];
        }
        for (size_t j = 0; j < k->rows; j++)
        {
            row[j] = j < count ? row[j] / sum : 0;
            if (p)
                p->data[i * p->cols + j] = row[j];
        }
        for (size_t d = 0; d < v->cols; d++)
        {
            double s = 0.0;
            for (size_t j = 0; j < count; j++)
                s += row[j] * v->data[j * v->cols + d];
            if (!isfinite(s))
            {
                status = CFWX_NUMERIC;
                goto done;
            }
            out->data[i * out->cols + d] = s;
        }
    }
done:
    free(row);
    return status;
}

/**
 * @anchor api_src_models_attention_c_cfwx_layer_norm
 * @brief Perform the layer norm operation.
 * @ingroup attention
 * @details Implementation of the public declaration; its canonical contract
 * remains in the corresponding header and is merged by Doxygen.
 * @see cfwx_layer_norm include/models/attention.h
 * @note Internal helper state may be temporary. Referenced-by/source links show
 * how composite callers propagate errors and release allocations.
 */
cfwx_status cfwx_layer_norm(const cfwx_matrix *x, const double *gamma, const double *beta,
                            double eps, cfwx_matrix *out)
{
    /* Welford accumulation computes population variance without subtracting large moments. */

    if (!x || !out || !x->data || !out->data || !gamma || !beta || !x->cols || !isfinite(eps) ||
        eps <= 0 || gamma == out->data || beta == out->data || gamma == x->data || beta == x->data)
        return CFWX_INVALID;
    if (x->rows != out->rows || x->cols != out->cols)
        return CFWX_SHAPE;
    for (size_t j = 0; j < x->cols; j++)
        if (!isfinite(gamma[j]) || !isfinite(beta[j]))
            return CFWX_NUMERIC;
    for (size_t i = 0; i < x->rows; i++)
    {
        double mean = 0.0, m2 = 0;
        for (size_t j = 0; j < x->cols; j++)
        {
            double v = x->data[i * x->cols + j];
            if (!isfinite(v))
                return CFWX_NUMERIC;
            double delta = v - mean;
            mean += delta / (double)(j + 1);
            m2 += delta * (v - mean);
        }
        double var = fmax(0, m2 / (double)x->cols);
        if (!isfinite(mean) || !isfinite(m2) || !isfinite(var + eps))
            return CFWX_NUMERIC;
        double inv = 1 / sqrt(var + eps);
        for (size_t j = 0; j < x->cols; j++)
        {
            double v = (x->data[i * x->cols + j] - mean) * inv * gamma[j] + beta[j];
            if (!isfinite(v))
                return CFWX_NUMERIC;
            out->data[i * out->cols + j] = v;
        }
    }
    return CFWX_OK;
}
