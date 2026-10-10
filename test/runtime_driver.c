/* Logged dataset runner with real mini-batches, full evaluation, resume and generation. */
#include "models/runtime.h"
#include "util/logger.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

static bool number(const char *s, size_t *n)
{
    char *end;
    errno = 0;
    unsigned long x = strtoul(s, &end, 10);
    if (errno || !s[0] || s[0] == '-' || *end || !x || x > 1000000)
        return false;
    *n = (size_t)x;
    return true;
}

static uint64_t next(uint64_t *s)
{
    *s += UINT64_C(0x9e3779b97f4a7c15);
    uint64_t z = *s;
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}

static int generate(const experiment_logger *logger, const char *path, const char *prompt,
                    size_t count)
{
    cfwx_language_model m = {0};
    cfwx_adamw o = {0};
    cfwx_training_state state = {0};
    cfwx_kv_cache cache = {0};
    cfwx_matrix logits = {0};
    cfwx_status s = cfwx_checkpoint_load(path, &m, &o, &state);
    size_t length = strlen(prompt);
    if (s == CFWX_OK && (!length || length > SIZE_MAX - count))
        s = CFWX_INVALID;
    if (s == CFWX_OK)
        s = cfwx_kv_cache_init(&cache, &m, length + count);
    if (s == CFWX_OK)
        s = cfwx_matrix_init(&logits, 1, m.vocabulary);
    for (size_t i = 0; i < length && s == CFWX_OK; i++)
    {
        size_t id = state.byte_map[(unsigned char)prompt[i]];
        if (id == SIZE_MAX)
            id = state.unknown;
        s = cfwx_language_model_decode(&cache, id, &logits);
    }
    if (s == CFWX_OK)
        fputs(prompt, stdout);
    for (size_t i = 0; i < count && s == CFWX_OK; i++)
    {
        size_t id;
        s = cfwx_sample_logits(logits.data, m.vocabulary, 0.8, 0, &state.rng, &id);
        if (s != CFWX_OK)
            break;
        int byte = -1;
        for (size_t b = 0; b < 256; b++)
            if (state.byte_map[b] == id)
            {
                byte = (int)b;
                break;
            }
        if (byte < 0)
            fputs("<UNK>", stdout);
        else
            fputc(byte, stdout);
        s = cfwx_language_model_decode(&cache, id, &logits);
    }
    if (s == CFWX_OK)
        fputc('\n', stdout);
    else
        EXP_LOG_ERROR(logger, "Generation failed: status=%d", (int)s);
    cfwx_matrix_free(&logits);
    cfwx_kv_cache_free(&cache);
    cfwx_adamw_free(&o);
    cfwx_language_model_free(&m);
    return s == CFWX_OK ? 0 : 1;
}

static int train(const experiment_logger *logger, const char *dataset, const char *checkpoint,
                 size_t steps, bool resume)
{
    FILE *file = fopen(dataset, "rb");
    if (!file)
    {
        EXP_LOG_ERROR(logger, "Cannot open dataset");
        return 1;
    }
    unsigned char *bytes = malloc(1048577);
    size_t size = bytes ? fread(bytes, 1, 1048577, file) : 0;
    bool error = ferror(file) != 0;
    fclose(file);
    if (!bytes || error || size < 48 || size > 1048576)
    {
        free(bytes);
        EXP_LOG_ERROR(logger, "Dataset must contain 48 bytes..1 MiB");
        return 1;
    }
    size_t split = size * 4 / 5;
    uint64_t fingerprint = cfwx_dataset_hash(bytes, size);
    cfwx_training_state state = {0};
    cfwx_language_model m = {0}, g = {0};
    cfwx_adamw optimizer = {0};
    cfwx_parameters p = {0};
    cfwx_status status = CFWX_OK;
    if (resume)
    {
        status = cfwx_checkpoint_load(checkpoint, &m, &optimizer, &state);
        if (status == CFWX_OK && state.dataset_hash != fingerprint)
        {
            EXP_LOG_ERROR(logger, "Dataset changed since checkpoint");
            status = CFWX_INVALID;
        }
    }
    else
    {
        size_t vocabulary = 0;
        for (size_t i = 0; i < 256; i++)
            state.byte_map[i] = SIZE_MAX;
        for (size_t i = 0; i < split; i++)
            if (state.byte_map[bytes[i]] == SIZE_MAX)
                state.byte_map[bytes[i]] = vocabulary++;
        state.unknown = vocabulary++;
        state.rng = 42;
        state.dataset_hash = fingerprint;
        state.context = 8;
        state.batch = 4;
        state.dropout = 0.1;
        state.rate = 0.01;
        state.beta1 = 0.9;
        state.beta2 = 0.999;
        state.epsilon = 1e-8;
        state.decay = 0;
        state.clip = 1;
        status = cfwx_language_model_init(&m, vocabulary, 8, 2, 16, 1, 42);
    }
    if (status == CFWX_OK)
        status = cfwx_parameters_language_model(&p, &m);
    if (status == CFWX_OK && !resume)
        status = cfwx_adamw_init(&optimizer, &p);
    if (status == CFWX_OK)
        status = cfwx_language_model_init(&g, m.vocabulary, m.width, m.blocks[0].attention.heads,
                                          m.blocks[0].feedforward.up.weights.cols, m.depth, 0);
    size_t *ids = malloc(size * sizeof(size_t));
    const size_t **inputs = NULL, **targets = NULL;
    if (status == CFWX_OK && (!ids || split <= state.context))
        status = CFWX_INVALID;
    if (status == CFWX_OK)
    {
        inputs = calloc(state.batch, sizeof(*inputs));
        targets = calloc(state.batch, sizeof(*targets));
        if (!inputs || !targets)
            status = CFWX_ALLOC;
        for (size_t i = 0; i < size; i++)
        {
            size_t id = state.byte_map[bytes[i]];
            ids[i] = id == SIZE_MAX ? state.unknown : id;
        }
    }
    cfwx_parameters grads = {0};
    if (status == CFWX_OK)
        status = cfwx_parameters_language_model(&grads, &g);
    double initial = 0.0, final = 0.0, validation = 0.0;
    size_t scored = 0;
    if (status == CFWX_OK)
        status = cfwx_language_model_evaluate(&m, ids, split, state.context, &initial, &scored);
    if (status == CFWX_OK)
        EXP_LOG_INFO(
            logger,
            "starting optimizer_step=%zu context=%zu batch=%zu dropout=%.3f full_train_loss=%.9f",
            optimizer.step, state.context, state.batch, state.dropout, initial);
    for (size_t step = 0; step < steps && status == CFWX_OK; step++)
    {
        for (size_t b = 0; b < state.batch; b++)
        {
            size_t start = (size_t)(next(&state.rng) % (split - state.context));
            inputs[b] = ids + start;
            targets[b] = ids + start + 1;
        }
        double loss = 0.0, norm = 0.0;
        status = cfwx_language_model_batch(&m, inputs, targets, NULL, state.batch, state.context,
                                           state.dropout, &state.rng, &loss, &g);
        if (status == CFWX_OK)
            status = cfwx_clip_gradients(&grads, state.clip, &norm);
        if (status == CFWX_OK)
            status = cfwx_adamw_step(&optimizer, &p, &grads, state.rate, state.beta1, state.beta2,
                                     state.epsilon, state.decay);
        if (step % 50 == 0 || step + 1 == steps)
            EXP_LOG_INFO(logger, "optimizer_step=%zu train_batch_loss=%.9f preclip_norm=%.6f",
                         optimizer.step, loss, norm);
    }
    if (status == CFWX_OK)
        status = cfwx_language_model_evaluate(&m, ids, split, state.context, &final, &scored);
    if (status == CFWX_OK)
        status = cfwx_language_model_evaluate(&m, ids + split, size - split, state.context,
                                              &validation, &scored);
    if (status == CFWX_OK)
        EXP_LOG_INFO(
            logger,
            "full_train_loss %.9f -> %.9f full_validation_loss=%.9f perplexity=%.6f scored=%zu",
            initial, final, validation, exp(validation), scored);
    if (status == CFWX_OK)
        status = cfwx_checkpoint_save(checkpoint, &m, &optimizer, &state);
    if (status == CFWX_OK)
        EXP_LOG_INFO(logger, "Saved complete checkpoint: %s", checkpoint);
    else
        EXP_LOG_ERROR(logger, "Training/resume failed status=%d; no checkpoint replacement",
                      (int)status);
    free(ids);
    free(inputs);
    free(targets);
    free(bytes);
    cfwx_adamw_free(&optimizer);
    cfwx_language_model_free(&m);
    cfwx_language_model_free(&g);
    return status == CFWX_OK ? 0 : 1;
}

int main(int argc, char **argv)
{
    experiment_logger logger;
    if (experiment_logger_init(&logger, "transformer.runtime", LOG_INFO, NULL, NULL) != 0)
        return 1;
    size_t count;
    if (argc == 6 && (!strcmp(argv[1], "train") || !strcmp(argv[1], "resume")) &&
        number(argv[4], &count) && !strcmp(argv[5], "cpu"))
        return train(&logger, argv[2], argv[3], count, !strcmp(argv[1], "resume"));
    if (argc == 5 && !strcmp(argv[1], "generate") && number(argv[4], &count))
        return generate(&logger, argv[2], argv[3], count);
    fprintf(stderr,
            "Usage:\n %s train DATA CHECKPOINT STEPS cpu\n %s resume DATA CHECKPOINT MORE_STEPS "
            "cpu\n %s generate CHECKPOINT PROMPT TOKENS\n",
            argv[0], argv[0], argv[0]);
    fprintf(stderr,
            "Only the tested CPU backend is available; GPU requests are not silently emulated.\n");
    return 1;
}
