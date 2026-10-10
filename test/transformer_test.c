/**
 * @file test/transformer_test.c
 * @brief transformer test interfaces and implementation.
 * @ingroup tests
 * @details See the module guide for shape, ownership, failure and execution-policy
 * constraints. Source listings include implementation helpers and ordinary
 * explanatory comments; generated source documentation does not execute code.
 */

/** Reference, finite-difference, masking, identity and causal invariance tests. */
#include "models/transformer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x)                                                                                   \
    do                                                                                             \
    {                                                                                              \
        if (!(x))                                                                                  \
        {                                                                                          \
            fprintf(stderr, "FAIL %d: %s\n", __LINE__, #x);                                        \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    } while (0)
/** @def OK
 * @brief ok macro; see expansion and module contract.
 */
#define OK(x) CHECK((x) == CFWX_OK)
/** @def NEAR
 * @brief near macro; see expansion and module contract.
 */
#define NEAR(a, b) CHECK(isfinite(a) && fabs((a) - (b)) < 1e-8)

/**
 * @anchor api_test_transformer_test_c_primitives
 * @brief Perform the primitives operation.
 * @ingroup tests
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @note This operation returns no value; inspect its documented mutation/lifetime effects.
 */
static void primitives(void)
{
    cfwx_matrix x = {0}, y = {0}, dy = {0}, dx = {0}, mask = {0};
    cfwx_matrix *parts[] = {&x, &y, &dy, &dx, &mask};
    for (size_t i = 0; i < 5; i++)
        OK(cfwx_matrix_init(parts[i], 2, 4));
    for (size_t i = 0; i < 8; i++)
    {
        x.data[i] = 0.2 + (double)i / 4.0;
        dy.data[i] = 1.0;
    }
    for (int kind = CFWX_ACT_IDENTITY; kind <= CFWX_ACT_SIGMOID; kind++)
    {
        OK(cfwx_activate_backward(&x, &dy, (cfwx_activation)kind, &dx));
        for (size_t i = 0; i < 8; i++)
        {
            double original = x.data[i], h = 1e-6;
            x.data[i] = original + h;
            OK(cfwx_activate(&x, (cfwx_activation)kind, &y));
            double plus = y.data[i];
            x.data[i] = original - h;
            OK(cfwx_activate(&x, (cfwx_activation)kind, &y));
            double minus = y.data[i];
            x.data[i] = original;
            NEAR(dx.data[i], (plus - minus) / (2 * h));
        }
    }
    uint64_t seed = 1, saved = seed;
    OK(cfwx_dropout(&x, 0.5, false, &seed, &y, &mask));
    CHECK(seed == saved);
    for (size_t i = 0; i < 8; i++)
    {
        NEAR(y.data[i], x.data[i]);
        NEAR(mask.data[i], 1.0);
    }
    OK(cfwx_dropout(&x, 0.5, true, &seed, &y, &mask));
    for (size_t i = 0; i < 8; i++)
    {
        CHECK(mask.data[i] == 0.0 || mask.data[i] == 2.0);
        NEAR(y.data[i], x.data[i] * mask.data[i]);
    }
    OK(cfwx_dropout_backward(&dy, &mask, &dx));
    CHECK(cfwx_dropout(&x, 1.0, true, &seed, &y, NULL) == CFWX_INVALID);
    OK(cfwx_sinusoidal_positions(0, &y));
    NEAR(y.data[0], 0.0);
    NEAR(y.data[1], 1.0);
    OK(cfwx_rope(&x, 7, 10000.0, &y));
    for (size_t i = 0; i < 8; i += 2)
        NEAR(y.data[i] * y.data[i] + y.data[i + 1] * y.data[i + 1],
             x.data[i] * x.data[i] + x.data[i + 1] * x.data[i + 1]);
    double gamma[] = {1, 1, 1, 1};
    OK(cfwx_rms_norm(&x, gamma, 1e-5, &y));
    double square = 0;
    for (size_t j = 0; j < 4; j++)
        square += x.data[j] * x.data[j] / 4;
    NEAR(y.data[0], x.data[0] / sqrt(square + 1e-5));
    for (size_t i = 0; i < 5; i++)
        cfwx_matrix_free(parts[i]);
}

/**
 * @anchor api_test_transformer_test_c_embedding_loss
 * @brief Perform the embedding loss operation.
 * @ingroup tests
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @note This operation returns no value; inspect its documented mutation/lifetime effects.
 */
static void embedding_loss(void)
{
    cfwx_matrix table = {0}, out = {0}, grad = {0}, dy = {0};
    OK(cfwx_matrix_init(&table, 3, 2));
    OK(cfwx_matrix_init(&grad, 3, 2));
    OK(cfwx_matrix_init(&out, 2, 2));
    OK(cfwx_matrix_init(&dy, 2, 2));
    for (size_t i = 0; i < 6; i++)
        table.data[i] = (double)i;
    size_t ids[] = {1, 1};
    OK(cfwx_embedding(&table, ids, 2, &out));
    NEAR(out.data[0], 2);
    for (size_t i = 0; i < 4; i++)
        dy.data[i] = 1;
    OK(cfwx_embedding_backward(ids, 2, &dy, &grad));
    NEAR(grad.data[2], 2);
    NEAR(grad.data[0], 0);
    size_t labels[] = {1, SIZE_MAX};
    double loss;
    OK(cfwx_token_cross_entropy(&out, labels, SIZE_MAX, 0.2, &loss, &dy));
    for (size_t i = 0; i < 2; i++)
    {
        double old = out.data[i], h = 1e-6, plus, minus;
        out.data[i] = old + h;
        OK(cfwx_token_cross_entropy(&out, labels, SIZE_MAX, 0.2, &plus, NULL));
        out.data[i] = old - h;
        OK(cfwx_token_cross_entropy(&out, labels, SIZE_MAX, 0.2, &minus, NULL));
        out.data[i] = old;
        NEAR(dy.data[i], (plus - minus) / (2 * h));
    }
    NEAR(dy.data[2], 0);
    labels[0] = SIZE_MAX;
    CHECK(cfwx_token_cross_entropy(&out, labels, SIZE_MAX, 0, &loss, NULL) == CFWX_INVALID);
    cfwx_matrix_free(&table);
    cfwx_matrix_free(&out);
    cfwx_matrix_free(&grad);
    cfwx_matrix_free(&dy);
}

/**
 * @anchor api_test_transformer_test_c_attention_block
 * @brief Perform the attention block operation.
 * @ingroup tests
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @note This operation returns no value; inspect its documented mutation/lifetime effects.
 */
static void attention_block(void)
{
    cfwx_matrix x = {0}, y = {0}, z = {0}, p = {0};
    OK(cfwx_matrix_init(&x, 3, 4));
    OK(cfwx_matrix_init(&y, 3, 4));
    OK(cfwx_matrix_init(&z, 3, 4));
    OK(cfwx_matrix_init(&p, 3, 3));
    for (size_t i = 0; i < 12; i++)
        x.data[i] = sin((double)i);
    bool mask[] = {true, false, false, true, true, false, true, true, true};
    OK(cfwx_masked_attention(&x, &x, &x, mask, NULL, false, &y, &p));
    NEAR(p.data[0], 1);
    NEAR(p.data[1], 0);
    NEAR(y.data[0], x.data[0]);
    mask[0] = false;
    CHECK(cfwx_masked_attention(&x, &x, &x, mask, NULL, false, &y, &p) == CFWX_SHAPE);
    cfwx_multihead mha = {0};
    CHECK(cfwx_multihead_init(&mha, 4, 3, 1) == CFWX_INVALID);
    OK(cfwx_multihead_init(&mha, 4, 2, 1));
    /* Identity projections: independently compare each head with raw attention. */
    cfwx_dense *layers[] = {&mha.query, &mha.key, &mha.value, &mha.output};
    for (size_t a = 0; a < 4; a++)
    {
        memset(layers[a]->weights.data, 0, 16 * sizeof(double));
        for (size_t i = 0; i < 4; i++)
            layers[a]->weights.data[i * 4 + i] = 1;
    }
    OK(cfwx_multihead_forward(&mha, &x, &x, NULL, true, &y));
    cfwx_matrix head = {0}, expected = {0};
    OK(cfwx_matrix_init(&head, 3, 2));
    OK(cfwx_matrix_init(&expected, 3, 2));
    for (size_t h = 0; h < 2; h++)
    {
        for (size_t i = 0; i < 3; i++)
            for (size_t j = 0; j < 2; j++)
                head.data[i * 2 + j] = x.data[i * 4 + h * 2 + j];
        OK(cfwx_attention(&head, &head, &head, true, &expected, NULL));
        for (size_t i = 0; i < 3; i++)
            for (size_t j = 0; j < 2; j++)
                NEAR(y.data[i * 4 + h * 2 + j], expected.data[i * 2 + j]);
    }
    cfwx_matrix_free(&head);
    cfwx_matrix_free(&expected);
    cfwx_multihead_free(&mha);
    cfwx_transformer_block block = {0};
    OK(cfwx_transformer_block_init(&block, 4, 2, 8, 9));
    OK(cfwx_transformer_block_forward(&block, &x, NULL, true, &y));
    for (size_t i = 8; i < 12; i++)
        x.data[i] += 100;
    OK(cfwx_transformer_block_forward(&block, &x, NULL, true, &z));
    for (size_t i = 0; i < 8; i++)
        NEAR(y.data[i], z.data[i]);
    /* Zero residual branches reduce the complete block to identity. */
    memset(block.attention.output.weights.data, 0, 16 * sizeof(double));
    memset(block.feedforward.down.weights.data, 0, 32 * sizeof(double));
    OK(cfwx_transformer_block_forward(&block, &x, NULL, false, &z));
    for (size_t i = 0; i < 12; i++)
        NEAR(x.data[i], z.data[i]);
    cfwx_transformer_block_free(&block);
    cfwx_transformer_block_free(&block);
    cfwx_matrix_free(&x);
    cfwx_matrix_free(&y);
    cfwx_matrix_free(&z);
    cfwx_matrix_free(&p);
}

/**
 * @anchor api_test_transformer_test_c_main
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
    embedding_loss();
    attention_block();
    puts("PASS Transformer primitives, gradients, masking, projected heads, block identity and "
         "causality");
    return EXIT_SUCCESS;
}
