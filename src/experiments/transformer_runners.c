/* Logged examples: tiny causal language model and cross-attention + SwiGLU.
 * Random weights demonstrate wiring, not meaningful language predictions. */
#include "experiments/transformer_runners.h"
#include "models/transformer.h"
#include <stdlib.h>

static bool valid_logger(const experiment_logger *logger)
{
    return logger && logger->experiment_name && logger->experiment_name[0] &&
           logger->minimum_level >= LOG_DEBUG && logger->minimum_level <= LOG_FATAL;
}

int transformer_language_runner(const experiment_logger *logger)
{
    if (!valid_logger(logger))
        return EXIT_FAILURE;
    EXP_LOG_INFO(logger, "Starting tiny causal LM: vocabulary=8, tokens=3, width=4, heads=2");
    EXP_LOG_WARNING(
        logger,
        "Random untrained weights: this is a composition example, not a language model benchmark");
    cfwx_dense embeddings = {0}, lm_head = {0};
    cfwx_transformer_block block = {0};
    cfwx_matrix tokens = {0}, positions = {0}, input = {0}, hidden = {0}, logits = {0},
                gradient = {0};
    cfwx_matrix *buffers[] = {&tokens, &positions, &input, &hidden, &logits, &gradient};
    const size_t ids[] = {1, 2, 3}, targets[] = {2, 3, 4};
    cfwx_status status = cfwx_dense_init(&embeddings, 8, 4, 42);
    if (status == CFWX_OK)
        status = cfwx_dense_init(&lm_head, 4, 8, 43);
    if (status == CFWX_OK)
        status = cfwx_transformer_block_init(&block, 4, 2, 12, 44);
    for (size_t i = 0; i < 6 && status == CFWX_OK; i++)
        status = cfwx_matrix_init(buffers[i], 3, i < 4 ? 4 : 8);
    if (status == CFWX_OK)
        status = cfwx_embedding(&embeddings.weights, ids, 3, &tokens);
    if (status == CFWX_OK)
        status = cfwx_sinusoidal_positions(0, &positions);
    if (status == CFWX_OK)
        status = cfwx_residual_add(&tokens, &positions, &input);
    if (status == CFWX_OK)
        status = cfwx_transformer_block_forward(&block, &input, NULL, true, &hidden);
    if (status == CFWX_OK)
        status = cfwx_dense_forward(&lm_head, &hidden, &logits);
    double loss = 0.0;
    if (status == CFWX_OK)
        status = cfwx_token_cross_entropy(&logits, targets, SIZE_MAX, 0.1, &loss, &gradient);
    if (status == CFWX_OK)
    {
        EXP_LOG_DEBUG(
            logger,
            "Embedding -> positional residual -> pre-LN causal block -> LM head -> smoothed CE");
        EXP_LOG_INFO(logger, "Mean token cross entropy=%.9f; output shape=[3,8]", loss);
    }
    else
        EXP_LOG_ERROR(logger, "Tiny language-model pipeline failed: status=%d", (int)status);
    for (size_t i = 0; i < 6; i++)
        cfwx_matrix_free(buffers[i]);
    cfwx_dense_free(&embeddings);
    cfwx_dense_free(&lm_head);
    cfwx_transformer_block_free(&block);
    return status == CFWX_OK ? EXIT_SUCCESS : EXIT_FAILURE;
}

int transformer_cross_attention_runner(const experiment_logger *logger)
{
    if (!valid_logger(logger))
        return EXIT_FAILURE;
    EXP_LOG_INFO(logger, "Starting cross-attention: queries=2, context=3, width=4, heads=2");
    cfwx_multihead attention = {0};
    cfwx_feedforward swiglu = {0};
    cfwx_matrix query = {0}, context = {0}, attended = {0}, residual = {0}, normalized = {0},
                output = {0};
    cfwx_matrix *buffers[] = {&query, &context, &attended, &residual, &normalized, &output};
    /* true means permitted; context key 2 is padding for both queries. */
    const bool mask[] = {true, true, false, true, true, false};
    const double gamma[] = {1.0, 1.0, 1.0, 1.0};
    cfwx_status status = cfwx_multihead_init(&attention, 4, 2, 100);
    if (status == CFWX_OK)
        status = cfwx_feedforward_init(&swiglu, 4, 10, CFWX_ACT_SILU, true, 110);
    for (size_t i = 0; i < 6 && status == CFWX_OK; i++)
        status = cfwx_matrix_init(buffers[i], i == 1 ? 3 : 2, 4);
    if (status == CFWX_OK)
    {
        for (size_t i = 0; i < 8; i++)
            query.data[i] = (double)i / 8.0;
        for (size_t i = 0; i < 12; i++)
            context.data[i] = (double)i / 12.0;
        status = cfwx_multihead_forward(&attention, &query, &context, mask, false, &attended);
    }
    if (status == CFWX_OK)
        status = cfwx_residual_add(&query, &attended, &residual);
    if (status == CFWX_OK)
        status = cfwx_rms_norm(&residual, gamma, 1e-5, &normalized);
    if (status == CFWX_OK)
        status = cfwx_feedforward_forward(&swiglu, &normalized, &output);
    if (status == CFWX_OK)
    {
        EXP_LOG_DEBUG(
            logger,
            "Projected cross-attention -> residual -> RMSNorm -> SwiGLU; padding mask applied");
        EXP_LOG_INFO(logger, "Completed output shape=[2,4], first value=%.9f", output.data[0]);
    }
    else
        EXP_LOG_ERROR(logger, "Cross-attention pipeline failed: status=%d", (int)status);
    for (size_t i = 0; i < 6; i++)
        cfwx_matrix_free(buffers[i]);
    cfwx_multihead_free(&attention);
    cfwx_feedforward_free(&swiglu);
    return status == CFWX_OK ? EXIT_SUCCESS : EXIT_FAILURE;
}
