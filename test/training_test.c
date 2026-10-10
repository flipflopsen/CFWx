/**
 * @file test/training_test.c
 * @brief training test interfaces and implementation.
 * @ingroup tests
 * @details See the module guide for shape, ownership, failure and execution-policy
 * constraints. Source listings include implementation helpers and ordinary
 * explanatory comments; generated source documentation does not execute code.
 */

/** Exhaustive parameter/input finite differences on tiny analytic training graphs. */
#include "models/training.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(x)                                                                                   \
    do                                                                                             \
    {                                                                                              \
        if (!(x))                                                                                  \
        {                                                                                          \
            fprintf(stderr, "FAIL %d %s\n", __LINE__, #x);                                         \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)
/** @def OK
 * @brief ok macro; see expansion and module contract.
 */
#define OK(x) CHECK((x) == CFWX_OK)
/** @def CLOSE
 * @brief close macro; see expansion and module contract.
 */
#define CLOSE(a, b) CHECK(isfinite(a) && fabs((a) - (b)) < 2e-5 * (1 + fabs(b)))
/**
 * @anchor api_test_training_test_c_dot
 * @brief Perform the dot operation.
 * @ingroup tests
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @param a First borrowed numerical operand or internal node/state as declared.
 * @param b Second borrowed numerical operand or internal node/state as declared.
 * @return Computed value or status described by the adjacent contract.
 */
static double dot(const cfwx_matrix *a, const cfwx_matrix *b)
{
    double s = 0;
    for (size_t i = 0; i < a->rows * a->cols; i++)
        s += a->data[i] * b->data[i];
    return s;
}

/**
 * @anchor api_test_training_test_c_primitives
 * @brief Perform the primitives operation.
 * @ingroup tests
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @note This operation returns no value; inspect its documented mutation/lifetime effects.
 */
static void primitives(void)
{
    cfwx_matrix q = {0}, k = {0}, v = {0}, dy = {0}, dq = {0}, dk = {0}, dv = {0}, out = {0},
                bias = {0}, db = {0};
    cfwx_matrix *all[] = {&q, &k, &v, &dy, &dq, &dk, &dv, &out, &bias, &db};
    for (size_t i = 0; i < 10; i++)
    {
        OK(cfwx_matrix_init(all[i], 2, 2));
        for (size_t j = 0; j < 4; j++)
            all[i]->data[j] = 0.1 + (double)(i + j) * 0.07;
    }
    bool mask[] = {true, false, true, true};
    OK(cfwx_attention_backward(&q, &k, &v, mask, &bias, false, &dy, &dq, &dk, &dv, &db));
    cfwx_matrix *inputs[] = {&q, &k, &v, &bias}, *grads[] = {&dq, &dk, &dv, &db};
    for (size_t a = 0; a < 4; a++)
        for (size_t j = 0; j < 4; j++)
        {
            double old = inputs[a]->data[j], h = 1e-6;
            inputs[a]->data[j] = old + h;
            OK(cfwx_masked_attention(&q, &k, &v, mask, &bias, false, &out, NULL));
            double plus = dot(&out, &dy);
            inputs[a]->data[j] = old - h;
            OK(cfwx_masked_attention(&q, &k, &v, mask, &bias, false, &out, NULL));
            double minus = dot(&out, &dy);
            inputs[a]->data[j] = old;
            CLOSE(grads[a]->data[j], (plus - minus) / (2 * h));
        }
    double gamma[] = {1.1, 0.9}, beta[] = {0.1, -0.2}, dg[2], dbeta[2];
    for (int rms = 0; rms < 2; rms++)
    {
        if (rms)
            OK(cfwx_rms_norm_backward(&q, gamma, 1e-5, &dy, &dq, dg));
        else
            OK(cfwx_layer_norm_backward(&q, gamma, 1e-5, &dy, &dq, dg, dbeta));
        for (size_t j = 0; j < 8; j++)
        {
            double *p = j < 4 ? &q.data[j] : j < 6 ? &gamma[j - 4] : &beta[j - 6];
            if (rms && j >= 6)
                continue;
            double old = *p, h = 1e-6;
            *p = old + h;
            if (rms)
                OK(cfwx_rms_norm(&q, gamma, 1e-5, &out));
            else
                OK(cfwx_layer_norm(&q, gamma, beta, 1e-5, &out));
            double plus = dot(&out, &dy);
            *p = old - h;
            if (rms)
                OK(cfwx_rms_norm(&q, gamma, 1e-5, &out));
            else
                OK(cfwx_layer_norm(&q, gamma, beta, 1e-5, &out));
            double minus = dot(&out, &dy);
            *p = old;
            double analytic = j < 4 ? dq.data[j] : j < 6 ? dg[j - 4] : dbeta[j - 6];
            CLOSE(analytic, (plus - minus) / (2 * h));
        }
    }
    OK(cfwx_rope_backward(&dy, 3, 10000, &dq));
    for (size_t j = 0; j < 4; j++)
    {
        double old = q.data[j], h = 1e-6;
        q.data[j] = old + h;
        OK(cfwx_rope(&q, 3, 10000, &out));
        double plus = dot(&out, &dy);
        q.data[j] = old - h;
        OK(cfwx_rope(&q, 3, 10000, &out));
        double minus = dot(&out, &dy);
        q.data[j] = old;
        CLOSE(dq.data[j], (plus - minus) / (2 * h));
    }
    for (size_t i = 0; i < 10; i++)
        cfwx_matrix_free(all[i]);
}

/**
 * @anchor api_test_training_test_c_model
 * @brief Perform the model operation.
 * @ingroup tests
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @note This operation returns no value; inspect its documented mutation/lifetime effects.
 */
static void model(void)
{
    cfwx_language_model m = {0}, g = {0};
    OK(cfwx_language_model_init(&m, 3, 4, 2, 5, 1, 42));
    OK(cfwx_language_model_init(&g, 3, 4, 2, 5, 1, 0));
    size_t ids[] = {0, 1, 0}, targets[] = {1, 0, 2};
    double loss;
    OK(cfwx_language_model_loss(&m, ids, targets, 3, &loss, &g));
    cfwx_parameters p = {0}, grad = {0};
    OK(cfwx_parameters_language_model(&p, &m));
    OK(cfwx_parameters_language_model(&grad, &g));
    size_t checked = 0;
    for (size_t a = 0; a < p.count; a++)
        for (size_t j = 0; j < p.size[a]; j++)
        {
            double old = p.data[a][j], h = 1e-6, plus, minus;
            p.data[a][j] = old + h;
            OK(cfwx_language_model_loss(&m, ids, targets, 3, &plus, NULL));
            p.data[a][j] = old - h;
            OK(cfwx_language_model_loss(&m, ids, targets, 3, &minus, NULL));
            p.data[a][j] = old;
            CLOSE(grad.data[a][j], (plus - minus) / (2 * h));
            checked++;
        }
    printf("PASS complete LM finite differences: %zu scalar parameters\n", checked);
    cfwx_language_model_free(&m);
    cfwx_language_model_free(&g);
}

/**
 * @anchor api_test_training_test_c_gated
 * @brief Perform the gated operation.
 * @ingroup tests
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @note This operation returns no value; inspect its documented mutation/lifetime effects.
 */
static void gated(void)
{
    cfwx_feedforward l = {0}, g = {0};
    OK(cfwx_feedforward_init(&l, 2, 3, CFWX_ACT_SILU, true, 2));
    OK(cfwx_feedforward_init(&g, 2, 3, CFWX_ACT_SILU, true, 0));
    cfwx_matrix x = {0}, dy = {0}, dx = {0}, out = {0};
    cfwx_matrix *b[] = {&x, &dy, &dx, &out};
    for (size_t i = 0; i < 4; i++)
        OK(cfwx_matrix_init(b[i], 2, 2));
    for (size_t i = 0; i < 4; i++)
    {
        x.data[i] = 0.1 + (double)i * .2;
        dy.data[i] = 1;
    }
    OK(cfwx_feedforward_backward(&l, &x, &dy, &dx, &g));
    cfwx_parameters p = {0}, grad = {0};
    OK(cfwx_parameters_dense(&p, &l.up));
    OK(cfwx_parameters_dense(&p, &l.down));
    OK(cfwx_parameters_dense(&p, &l.gate));
    OK(cfwx_parameters_dense(&grad, &g.up));
    OK(cfwx_parameters_dense(&grad, &g.down));
    OK(cfwx_parameters_dense(&grad, &g.gate));
    for (size_t a = 0; a <= p.count; a++)
        for (size_t j = 0; j < (a == p.count ? 4 : p.size[a]); j++)
        {
            double *value = a == p.count ? &x.data[j] : &p.data[a][j], old = *value, h = 1e-6;
            *value = old + h;
            OK(cfwx_feedforward_forward(&l, &x, &out));
            double plus = dot(&out, &dy);
            *value = old - h;
            OK(cfwx_feedforward_forward(&l, &x, &out));
            double minus = dot(&out, &dy);
            *value = old;
            CLOSE(a == p.count ? dx.data[j] : grad.data[a][j], (plus - minus) / (2 * h));
        }
    for (size_t i = 0; i < 4; i++)
        cfwx_matrix_free(b[i]);
    cfwx_feedforward_free(&l);
    cfwx_feedforward_free(&g);
}

/**
 * @anchor api_test_training_test_c_main
 * @brief Run this executable driver and propagate its success/failure status.
 * @ingroup tests
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @return Computed value or status described by the adjacent contract.
 */
int main(void)
{
    primitives();
    gated();
    model();
    puts("PASS attention norm RoPE SwiGLU and end-to-end analytic gradients");
    return 0;
}
