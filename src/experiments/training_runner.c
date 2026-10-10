/* Deterministic next-byte training, held-out contiguous split, AdamW and clipping. */
#include "experiments/training_runner.h"
#include "models/training.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int transformer_training_runner(const experiment_logger *logger, const char *dataset, size_t steps)
{
    if (!logger || !logger->experiment_name || !logger->experiment_name[0] ||
        logger->minimum_level < LOG_DEBUG || logger->minimum_level > LOG_FATAL || !steps)
        return EXIT_FAILURE;
    unsigned char *corpus = NULL;
    size_t length = 0;
    FILE *file = NULL;
    if (dataset)
    {
        file = fopen(dataset, "rb");
        if (!file)
        {
            EXP_LOG_ERROR(logger, "Cannot open dataset: %s", dataset);
            return EXIT_FAILURE;
        }
        corpus = malloc(1048577);
        if (corpus)
            length = fread(corpus, 1, 1048577, file);
        bool failed = ferror(file) != 0;
        fclose(file);
        if (!corpus || failed || length > 1048576)
        {
            free(corpus);
            EXP_LOG_ERROR(logger, "Dataset read/allocation failed or exceeds 1 MiB");
            return EXIT_FAILURE;
        }
    }
    else
    {
        const char *text = "abcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcab"
                           "cabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabc";
        length = strlen(text);
        corpus = malloc(length);
        if (corpus)
            memcpy(corpus, text, length);
    }
    if (!corpus || length < 48)
    {
        free(corpus);
        EXP_LOG_ERROR(logger, "Dataset needs at least 48 bytes");
        return EXIT_FAILURE;
    }
    size_t map[256], vocab = 0;
    for (size_t i = 0; i < 256; i++)
        map[i] = SIZE_MAX;
    /* Fixed byte vocabulary built from the training partition only; reserve UNK. */
    size_t split = length * 4 / 5;
    for (size_t i = 0; i < split; i++)
        if (map[corpus[i]] == SIZE_MAX)
            map[corpus[i]] = vocab++;
    size_t unknown = vocab++;
    cfwx_language_model model = {0}, gradient = {0};
    cfwx_parameters parameters = {0}, gradients = {0};
    cfwx_adamw optimizer = {0};
    cfwx_status status = cfwx_language_model_init(&model, vocab, 8, 2, 16, 1, 42);
    if (status == CFWX_OK)
        status = cfwx_language_model_init(&gradient, vocab, 8, 2, 16, 1, 0);
    if (status == CFWX_OK)
        status = cfwx_parameters_language_model(&parameters, &model);
    if (status == CFWX_OK)
        status = cfwx_parameters_language_model(&gradients, &gradient);
    if (status == CFWX_OK)
        status = cfwx_adamw_init(&optimizer, &parameters);
    const size_t context = 8;
    size_t tokens[8], targets[8], validation[8], vtargets[8];
    for (size_t j = 0; j < context; j++)
    {
        size_t a = map[corpus[split + j]], b = map[corpus[split + j + 1]];
        validation[j] = a == SIZE_MAX ? unknown : a;
        vtargets[j] = b == SIZE_MAX ? unknown : b;
    }
    double initial = 0.0, final = 0.0, val = 0.0, norm = 0.0;
    for (size_t j = 0; j < context; j++)
    {
        tokens[j] = map[corpus[j]];
        targets[j] = map[corpus[j + 1]];
    }
    if (status == CFWX_OK)
        status = cfwx_language_model_loss(&model, tokens, targets, context, &initial, NULL);
    EXP_LOG_INFO(logger, "Training bytes=%zu validation bytes=%zu vocabulary=%zu steps=%zu", split,
                 length - split, vocab, steps);
    for (size_t step = 0; step < steps && status == CFWX_OK; step++)
    {
        size_t start = (step * context) % (split - context);
        for (size_t j = 0; j < context; j++)
        {
            tokens[j] = map[corpus[start + j]];
            targets[j] = map[corpus[start + j + 1]];
        }
        double loss = 0.0;
        status = cfwx_language_model_loss(&model, tokens, targets, context, &loss, &gradient);
        if (status == CFWX_OK)
            status = cfwx_clip_gradients(&gradients, 1.0, &norm);
        /* Gradients already contain CE's mean reduction: do not divide twice. */
        if (status == CFWX_OK)
            status =
                cfwx_adamw_step(&optimizer, &parameters, &gradients, 0.01, 0.9, 0.999, 1e-8, 0.0);
        if (step % 50 == 0 || step + 1 == steps)
            EXP_LOG_INFO(logger, "step=%zu pre-update train_loss=%.9f gradient_norm=%.6f", step,
                         loss, norm);
    }
    for (size_t j = 0; j < context; j++)
    {
        tokens[j] = map[corpus[j]];
        targets[j] = map[corpus[j + 1]];
    }
    if (status == CFWX_OK)
        status = cfwx_language_model_loss(&model, tokens, targets, context, &final, NULL);
    if (status == CFWX_OK)
        status = cfwx_language_model_loss(&model, validation, vtargets, context, &val, NULL);
    if (status == CFWX_OK)
        EXP_LOG_INFO(logger,
                     "Fixed-window loss %.9f -> %.9f; held-out-window loss=%.9f perplexity=%.6f",
                     initial, final, val, exp(val));
    else
        EXP_LOG_ERROR(logger, "Training failed with status=%d", (int)status);
    if (!dataset && status == CFWX_OK && !(final < initial * 0.5))
    {
        EXP_LOG_ERROR(logger, "Built-in overfit regression failed");
        status = CFWX_NUMERIC;
    }
    cfwx_adamw_free(&optimizer);
    cfwx_language_model_free(&gradient);
    cfwx_language_model_free(&model);
    free(corpus);
    return status == CFWX_OK ? EXIT_SUCCESS : EXIT_FAILURE;
}
